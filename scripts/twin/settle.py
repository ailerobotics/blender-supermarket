"""Analyse the robot's pose time-series from a real Isaac Sim run — pure, no pxr.

The real run (``compose_robot_in_store.py`` with no mode flag) steps physics and,
each few steps, records the composed robot's world-space axis-aligned bounding box
(min/max Z, footprint centre XY, Z-extent). This module turns that series into a
verdict: does the robot **rest on the store floor** — not fall through, not drift,
not explode — which is the task's success criterion.

It is deliberately free of ``pxr``/``isaacsim`` so it can be unit-tested offline
(``tests/test_settle.py``); the simulator-side sampling that produces the series is
the only part that needs a GPU boot.

A "sample" is a mapping with (at least) these numeric keys::

    step   int    physics step index
    t      float  simulated seconds (step * dt)
    min_z  float  world-space minimum Z of the whole robot prim (pedestal foot)
    max_z  float  world-space maximum Z (arm top)
    cx,cy  float  world-space footprint centre (X, Y)
    z_ext  float  max_z - min_z (a cheap "did the solver explode?" proxy)
"""
from __future__ import annotations

import math
from typing import List, Mapping, Sequence


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def analyze_settling(
    samples: Sequence[Mapping],
    *,
    floor_z: float = 0.0,
    on_floor_tol: float = 0.05,
    fall_tol: float = 0.05,
    drift_tol: float = 0.05,
    explode_ratio: float = 3.0,
    settle_fraction: float = 0.5,
) -> dict:
    """Decide whether the robot settled resting on the floor.

    Returns a dict with a boolean ``ok`` and the measured numbers behind it, plus a
    human-readable ``reasons`` list explaining any failure. Semantics:

    * ``rests_on_floor`` — over the final ``settle_fraction`` of the run, the robot's
      minimum Z stays within ``on_floor_tol`` of ``floor_z`` (the pedestal foot sits
      on the floor, neither hovering above nor sunk into it).
    * ``fell_through``   — at any sample the minimum Z dropped more than ``fall_tol``
      below ``floor_z`` (the robot punched through the floor collider).
    * ``drifted``        — the footprint centre moved more than ``drift_tol`` (m) in XY
      from the first sample (a pedestal-mounted arm should stay put).
    * ``exploded``       — the Z-extent grew by more than ``explode_ratio``× its initial
      value, or any sampled value was non-finite (solver instability / NaN).

    ``ok`` is True iff it rests on the floor and did not fall through, drift, or explode.
    """
    reasons: List[str] = []
    result = {
        "n_samples": len(samples),
        "ok": False,
        "rests_on_floor": False,
        "fell_through": False,
        "drifted": False,
        "exploded": False,
        "any_nonfinite": False,
        "floor_z": floor_z,
        "reasons": reasons,
    }
    if not samples:
        reasons.append("no samples recorded")
        return result

    # NaN / inf guard first — a blown-up solver often yields non-finite transforms.
    keys = ("min_z", "max_z", "cx", "cy", "z_ext")
    any_nonfinite = any(not _finite(s.get(k)) for s in samples for k in keys if k in s)
    result["any_nonfinite"] = any_nonfinite
    if any_nonfinite:
        result["exploded"] = True
        reasons.append("non-finite pose value(s) sampled (solver instability / NaN)")

    min_zs = [float(s["min_z"]) for s in samples if _finite(s.get("min_z"))]
    z_exts = [float(s["z_ext"]) for s in samples if _finite(s.get("z_ext"))]
    cxs = [float(s["cx"]) for s in samples if _finite(s.get("cx"))]
    cys = [float(s["cy"]) for s in samples if _finite(s.get("cy"))]

    # fall-through: any min_z well below the floor.
    if min_zs:
        min_min_z = min(min_zs)
        result["min_min_z"] = round(min_min_z, 5)
        result["max_min_z"] = round(max(min_zs), 5)
        result["first_min_z"] = round(min_zs[0], 5)
        if min_min_z < floor_z - fall_tol:
            result["fell_through"] = True
            reasons.append(
                f"min Z {min_min_z:.4f} fell {floor_z - min_min_z:.4f} m below floor "
                f"{floor_z} (> {fall_tol} tol)")

    # rests-on-floor: the settled tail stays on the floor plane.
    if min_zs:
        n_tail = max(1, int(math.ceil(len(min_zs) * settle_fraction)))
        tail = min_zs[-n_tail:]
        settled_min_z = sum(tail) / len(tail)
        result["settled_min_z"] = round(settled_min_z, 5)
        result["settle_samples"] = n_tail
        on_floor = abs(settled_min_z - floor_z) <= on_floor_tol
        result["rests_on_floor"] = bool(on_floor and not result["fell_through"])
        if not on_floor:
            reasons.append(
                f"settled min Z {settled_min_z:.4f} is {abs(settled_min_z - floor_z):.4f} m "
                f"off the floor {floor_z} (> {on_floor_tol} tol)")

    # drift: footprint centre wandering in XY.
    if cxs and cys:
        cx0, cy0 = cxs[0], cys[0]
        drift = max(math.hypot(x - cx0, y - cy0) for x, y in zip(cxs, cys))
        result["drift_xy"] = round(drift, 5)
        if drift > drift_tol:
            result["drifted"] = True
            reasons.append(f"footprint drifted {drift:.4f} m in XY (> {drift_tol} tol)")

    # explode: Z-extent ballooning.
    if z_exts:
        z0 = z_exts[0]
        z_max = max(z_exts)
        result["z_ext_first"] = round(z0, 5)
        result["z_ext_max"] = round(z_max, 5)
        if z0 > 1e-6 and z_max > z0 * explode_ratio:
            result["exploded"] = True
            reasons.append(
                f"Z-extent grew {z_max / z0:.1f}× (from {z0:.3f} to {z_max:.3f} m) "
                f"— solver instability")

    result["ok"] = bool(
        result["rests_on_floor"]
        and not result["fell_through"]
        and not result["drifted"]
        and not result["exploded"])
    if result["ok"]:
        reasons.append("robot rests on the floor: on-plane, no fall-through, no drift, stable")
    return result


def format_verdict(v: Mapping) -> str:
    """One-block human summary of an analyze_settling() result (for stderr / report)."""
    lines = ["--- settle analysis (robot base over time) ---",
             f"samples            : {v.get('n_samples')}"]
    if "first_min_z" in v:
        lines.append(f"min Z first/settled: {v.get('first_min_z')} -> {v.get('settled_min_z')} "
                     f"(floor {v.get('floor_z')})")
    if "min_min_z" in v:
        lines.append(f"min Z range        : [{v.get('min_min_z')}, {v.get('max_min_z')}]")
    if "drift_xy" in v:
        lines.append(f"footprint drift XY : {v.get('drift_xy')} m")
    if "z_ext_first" in v:
        lines.append(f"Z-extent first/max : {v.get('z_ext_first')} -> {v.get('z_ext_max')} m")
    flags = []
    for k in ("rests_on_floor", "fell_through", "drifted", "exploded", "any_nonfinite"):
        flags.append(f"{k}={v.get(k)}")
    lines.append("flags              : " + ", ".join(flags))
    lines.append(f"VERDICT            : {'PASS — robot stands on the floor' if v.get('ok') else 'FAIL'}")
    for r in v.get("reasons", []):
        lines.append(f"  - {r}")
    return "\n".join(lines)
