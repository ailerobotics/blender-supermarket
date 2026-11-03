#!/usr/bin/env python3
"""Rebuild the full Isaac store dataset (USD + manifest) headlessly, from the
CURRENT branch's scene code.

This is the one command that reproduces what the mission actually loads at GPU
time: the runtime points its ``store_usd`` config key at an EXTERNAL dataset
(``~/datasets/sim-store-realism-v2/supermarket.usdc`` historically), not at the
repo's ``export/``. When the scene code changes (portal width, fixtures, station
geometry) the dataset must be regenerated or the GPU runs keep driving through
the stale geometry. See docs/dataset_regen.md.

Pipeline (RUNBOOK §10), all CPU / headless — never the live Blender GUI session:

  1. gen_labels.py                       -> assets/labels, assets/signs (offline PIL)
  2. build_supermarket.py --all --save   -> <out>/store.blend  (+ environment.json manifest)
  3. check_environment.py                -> <out>/geometry-check.json (static clearance)
  4. export_usd.py --out                 -> <out>/supermarket.usdc (+ textures/)
  5. validate_usd.py --store             -> strict Isaac-loadable check (CPU usd-core)
  6. a portal-geometry probe of the exported stage + a manifest.json recording
     git HEAD, PORTAL_WIDTH, measured portal/backroom geometry, timestamps.

It never writes into the existing sim-store-realism-v2 directory and never opens
or saves scenes/*.blend (the protected live GUI session). Output goes to a fresh
``--out`` directory outside the repo.

Usage:
  scripts/regen_store_dataset.py --out ~/datasets/sim-store-realism-v3-portal26
  scripts/regen_store_dataset.py --out <dir> --blender /path/to/blender \
      --usd-python <repo>/.venv-usd/bin/python --threads 4
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SUPERMARKET_ROOT") or Path(__file__).resolve().parent.parent)
# The one dataset directory we must never overwrite (the old 2.5 m-portal dataset
# the running mission still loads until integration repoints its config).
PROTECTED_DATASET = Path("~/datasets/sim-store-realism-v2").expanduser().resolve()


def _log(msg: str) -> None:
    print(f"[regen] {msg}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> dict:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def _portal_width_constant() -> float:
    """Read PORTAL_WIDTH from store_environment.py without importing bpy."""
    import ast
    src = (ROOT / "scripts" / "store_environment.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "PORTAL_WIDTH":
                    return float(ast.literal_eval(node.value))
    raise RuntimeError("PORTAL_WIDTH not found in store_environment.py")


def _run(cmd: list[str], *, env: dict | None = None, logpath: Path | None = None) -> None:
    _log("$ " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    if logpath is not None:
        with open(logpath, "w") as fh:
            proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    else:
        proc = subprocess.run(cmd, env=env, cwd=ROOT)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = ""
        if logpath is not None and logpath.exists():
            tail = "\n--- tail of %s ---\n%s" % (logpath, "\n".join(logpath.read_text().splitlines()[-25:]))
        raise SystemExit(f"[regen] command failed (exit {proc.returncode}, {dt:.1f}s): {' '.join(map(str, cmd))}{tail}")
    _log(f"  ok ({dt:.1f}s)")


def _ensure_usd_python(usd_python: Path) -> Path:
    if usd_python.exists():
        return usd_python
    venv = ROOT / ".venv-usd"
    _log(f"creating CPU usd-core venv at {venv}")
    _run([sys.executable, "-m", "venv", str(venv)])
    _run([str(venv / "bin" / "pip"), "install", "-q", "usd-core", "numpy"])
    return venv / "bin" / "python"


def clear_width_from_rows(rows: dict) -> dict:
    """Pure: derive the portal clear opening from raw prim X-extents.

    ``rows`` maps a prim-name substring key -> list of ``(name, [x_min, x_max])``.
    The clear opening is the gap between the east face of the west rear wall and
    the west face of the east rear wall. Kept pure (no pxr) so it is unit-testable.
    """
    west = rows.get("Wall_back_west", [])
    east = rows.get("Wall_back_east", [])
    west_east_edge = max((x[1][1] for x in west), default=None)   # east face of west wall
    east_west_edge = min((x[1][0] for x in east), default=None)   # west face of east wall
    clear = (east_west_edge - west_east_edge) if (west and east) else None
    return {
        "wall_back_west": west,
        "wall_back_east": east,
        "portal_jambs": rows.get("Portal_jamb", []),
        "portal_lintel": rows.get("Portal_lintel", []),
        "west_wall_east_edge_x": west_east_edge,
        "east_wall_west_edge_x": east_west_edge,
        "clear_width_m": clear,
    }


def _probe_portal(usd_python: Path, usd_path: Path, out_json: Path) -> dict:
    """Measure the portal clear opening from the exported stage (CPU, no GPU)."""
    probe = out_json.parent / "_portal_probe_tmp.py"  # scratch, deleted after
    rows_json = out_json.parent / "_portal_rows_tmp.json"
    script = r'''
import json, sys
from pxr import Usd, UsdGeom
stage = Usd.Stage.Open(sys.argv[1])
def xrange_of(substr):
    hits = []
    for p in stage.Traverse():
        if substr in p.GetName() and p.IsA(UsdGeom.Boundable):
            b = UsdGeom.Imageable(p).ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
            r = b.ComputeAlignedRange()
            hits.append((p.GetName(), [r.GetMin()[0], r.GetMax()[0]]))
    return hits
rows = {k: xrange_of(k) for k in ("Wall_back_west", "Wall_back_east", "Portal_jamb", "Portal_lintel")}
json.dump(rows, open(sys.argv[2], "w"))
'''
    probe.write_text(script)
    try:
        _run([str(usd_python), str(probe), str(usd_path), str(rows_json)])
        rows = json.loads(rows_json.read_text())
    finally:
        probe.unlink(missing_ok=True)
        rows_json.unlink(missing_ok=True)
    result = clear_width_from_rows(rows)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _validate(usd_python: Path, usd_path: Path, out_dir: Path) -> dict:
    """Strict validate; repair cosmetic textures only if needed. Record both."""
    validator = ROOT / "scripts" / "validate_usd.py"
    strict_json = out_dir / "usd-validation.json"
    strict_log = out_dir / "usd-validation.log"
    # First pass: strict, no repair — proves whether the source export is clean.
    with open(strict_log, "w") as fh:
        proc = subprocess.run(
            [str(usd_python), str(validator), "--store", str(usd_path), "--json"],
            stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
        )
    strict_rc = proc.returncode
    repaired = False
    repair_log = out_dir / "usd-validation-repair.log"
    if strict_rc != 0:
        _log("strict validation non-zero; applying --repair-textures (cosmetic reconcile)")
        with open(repair_log, "w") as fh:
            rproc = subprocess.run(
                [str(usd_python), str(validator), "--store", str(usd_path), "--repair-textures"],
                stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
            )
        repaired = True
        if rproc.returncode != 0:
            raise SystemExit(
                "[regen] USD validation still fails after --repair-textures (BLOCKING, not a cosmetic "
                f"texture issue). See {repair_log}"
            )
    verdict = "PASS" if strict_rc == 0 else "PASS_AFTER_TEXTURE_REPAIR"
    _log(f"USD validation: {verdict}")
    return {
        "verdict": verdict,
        "strict_exit_code": strict_rc,
        "texture_repair_applied": repaired,
        "strict_log": str(strict_log.relative_to(out_dir)),
        "repair_log": str(repair_log.relative_to(out_dir)) if repaired else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output dataset directory (created; outside the repo).")
    ap.add_argument("--blender", default=None, help="Blender binary (default: $BLENDER, `blender` on PATH, or ~/.local/bin/blender).")
    ap.add_argument("--usd-python", default=None, help="Python with usd-core (default: <repo>/.venv-usd/bin/python, auto-created).")
    ap.add_argument("--threads", type=int, default=4, help="Blender -t threads (default 4).")
    ap.add_argument("--base-scene", default=None, help="Base .blend to build from (default: scenes/supermarket_base.blend).")
    ap.add_argument("--skip-labels", action="store_true", help="Reuse existing assets/labels (default: regenerate).")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).expanduser().resolve()
    if out_dir == PROTECTED_DATASET or PROTECTED_DATASET in out_dir.parents:
        raise SystemExit(f"[regen] refusing to write into the protected dataset {PROTECTED_DATASET}")
    if ROOT in out_dir.parents or out_dir == ROOT:
        raise SystemExit(f"[regen] refusing to write inside the repo {ROOT}; pick a scratch dataset dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    blender = args.blender or os.environ.get("BLENDER") or shutil.which("blender") or str(Path("~/.local/bin/blender").expanduser())
    if not Path(blender).exists() and shutil.which(blender) is None:
        raise SystemExit(f"[regen] blender binary not found: {blender}")
    usd_python = Path(args.usd_python).expanduser() if args.usd_python else ROOT / ".venv-usd" / "bin" / "python"
    usd_python = _ensure_usd_python(usd_python)

    base_scene = Path(args.base_scene) if args.base_scene else ROOT / "scenes" / "supermarket_base.blend"
    base_scene = base_scene.resolve()
    # Hard refuse to open/save the live GUI scene file.
    if base_scene.name == "supermarket.blend":
        raise SystemExit("[regen] refusing to build FROM the live scenes/supermarket.blend; use supermarket_base.blend")
    if not base_scene.exists():
        raise SystemExit(f"[regen] base scene not found: {base_scene}")

    store_blend = out_dir / "store.blend"
    env_json = out_dir / "environment.json"
    geom_json = out_dir / "geometry-check.json"
    usd_path = out_dir / "supermarket.usdc"

    env = dict(os.environ)
    env["SUPERMARKET_ROOT"] = str(ROOT)

    portal_width = _portal_width_constant()
    _log(f"PORTAL_WIDTH (source constant) = {portal_width} m")
    _log(f"blender   = {blender}")
    _log(f"usd-python= {usd_python}")
    _log(f"out       = {out_dir}")

    t_start = time.time()

    # 1. labels / signs / price tags (deterministic offline PIL output into assets/)
    if not args.skip_labels:
        _run([sys.executable, str(ROOT / "scripts" / "gen_labels.py")], env=env, logpath=out_dir / "labels.log")

    # 2. headless build of the full store + manifest
    _run([blender, "-b", str(base_scene), "-t", str(args.threads), "--python-exit-code", "1",
          "--python", str(ROOT / "scripts" / "build_supermarket.py"), "--",
          "--all", "--save", str(store_blend), "--manifest", str(env_json)],
         env=env, logpath=out_dir / "build.log")

    # 3. static clearance / geometry sanity (not dynamics)
    _run([blender, "-b", str(store_blend), "-t", str(args.threads), "--python-exit-code", "1",
          "--python", str(ROOT / "scripts" / "check_environment.py"), "--",
          "--manifest", str(env_json), "--out", str(geom_json)],
         env=env, logpath=out_dir / "geometry-check.log")

    # 4. USD export (authors resolvable texture paths; stages textures/ beside the usdc)
    _run([blender, "-b", str(store_blend), "-t", str(args.threads), "--python-exit-code", "1",
          "--python", str(ROOT / "scripts" / "export_usd.py"), "--", "--out", str(usd_path)],
         env=env, logpath=out_dir / "export.log")

    # 5. strict offline USD validation (Isaac-loadable); cosmetic texture repair only if needed
    validation = _validate(usd_python, usd_path, out_dir)

    # 6. portal-geometry probe of the exported stage + manifest
    portal_probe = _probe_portal(usd_python, usd_path, out_dir / "portal-probe.json")
    env_data = json.loads(env_json.read_text())

    manifest = {
        "dataset": "sim-store-realism-v3-portal26 family (regenerated)",
        "generator": "scripts/regen_store_dataset.py",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t_start, 1),
        "git": _git_head(),
        "blender": subprocess.run([blender, "--version"], capture_output=True, text=True).stdout.splitlines()[0].strip(),
        "base_scene": str(base_scene),
        "source_scripts": {
            name: _sha256(ROOT / "scripts" / name)
            for name in ("store_environment.py", "build_supermarket.py", "export_usd.py",
                         "gen_labels.py", "validate_usd.py")
        },
        "geometry": {
            "portal_width_source_constant_m": portal_width,
            "portal_centre_xy": env_data["portal"]["centre"][:2],
            "portal_clear_width_manifest_m": env_data["portal"]["clear_width"],
            "portal_clear_height_m": env_data["portal"]["clear_height"],
            "portal_clear_width_measured_usd_m": portal_probe["clear_width_m"],
            "portal_west_wall_east_edge_x": portal_probe["west_wall_east_edge_x"],
            "portal_east_wall_west_edge_x": portal_probe["east_wall_west_edge_x"],
            "portal_jambs_usd": portal_probe["portal_jambs"],
            "backroom_bounds": env_data["backroom_bounds"],
            "map_bounds": env_data["map_bounds"],
            "schema_version": env_data["schema_version"],
            "stocking_stations": len(env_data.get("stocking_stations", [])),
            "static_obstacles": len(env_data.get("static_obstacles", [])),
        },
        "usd_validation": validation,
        "artifacts": {
            "store_usd": "supermarket.usdc",
            "textures_dir": "textures",
            "environment_manifest": "environment.json",
            "geometry_check": "geometry-check.json",
            "store_blend": "store.blend",
        },
        "files": {
            name: {"sha256": _sha256(out_dir / name), "bytes": (out_dir / name).stat().st_size}
            for name in ("supermarket.usdc", "environment.json")
            if (out_dir / name).exists()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Consistency gate: source constant, manifest and measured USD must agree.
    g = manifest["geometry"]
    measured = g["portal_clear_width_measured_usd_m"]
    for label, val in (("manifest", g["portal_clear_width_manifest_m"]), ("measured-USD", measured)):
        if abs(val - portal_width) > 1e-3:
            raise SystemExit(f"[regen] portal clear width mismatch: source={portal_width} {label}={val}")

    _log("=" * 70)
    _log(f"dataset ready: {out_dir}")
    _log(f"portal clear width: source={portal_width} m, manifest={g['portal_clear_width_manifest_m']} m, "
         f"measured-USD={measured:.4f} m")
    _log(f"USD validation: {validation['verdict']}")
    _log(f"manifest: {out_dir / 'manifest.json'}")
    _log("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
