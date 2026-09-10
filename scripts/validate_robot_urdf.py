#!/usr/bin/env python3
"""Validate the shelf robot's URDF **offline** — no Isaac Sim, no GPU.

This is the robot-side pre-flight for standing the shelfbot UR10 in the supermarket
twin, the sibling of ``scripts/validate_usd.py`` (which checks the store USD). It reads
the URDF with pure ``xml`` + ``numpy`` (via ``twin/urdf_probe.py``) and proves the things
that otherwise only fail — often *silently* — once Isaac tries to import the robot on the
GPU. Isaac's URDF importer **imports even when a mesh cannot be found** (it logs an error
and keeps going — see its CHANGELOG), so a mesh that fails to resolve produces a robot with
missing geometry rather than a loud crash. Offline detection is the only thing between the
captain and a bad first boot.

Checks (hard = affects the exit code):
  * the URDF parses and is one connected kinematic tree (unique names, every joint's
    parent/child link exists, exactly one root, every link reachable);
  * **every mesh reference resolves** to a file that exists under the configured ROS
    package path — no dangling ``package://`` meshes, no absolute build-machine paths
    (the same defect class found in the store texture export). Same-name package
    ambiguity is surfaced;
  * every DYNAMIC link (child of a movable joint) has a positive mass and a physically
    valid inertia tensor (present, positive-definite, triangle inequality); no geometry
    link is left without inertia;
  * every movable joint has finite limits with lower < upper and positive effort/velocity
    — Isaac/PhysX needs these or the solver misbehaves;
  * **store consistency**: the URDF's pedestal-mount XY offset matches the composer's
    hard-coded ``geometry.PEDESTAL_OFFSET_FROM_ROOT`` (guards against URDF drift silently
    mis-placing the robot), and the arm's HOME-pose collision envelope fits the store
    aisle it must stand in (real numbers, from forward kinematics).

Soft signals (WARN/INFO, never fail): implausible placeholder inertias, mesh colliders
Isaac will convex-decompose, the import-config reminders.

Exit code: ``0`` all hard checks pass, ``1`` a hard check failed, ``2`` bad usage,
``3`` numpy unavailable.

Run it under the same CPU-only usd-core venv as validate_usd.py (it only needs numpy):

    .venv-usd/bin/python scripts/validate_robot_urdf.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

HOME = os.path.expanduser("~")
DEFAULT_URDF = os.path.join(HOME, "workspaces/shelfbot_newgripper/description/shelfbot_gripper_isaac.urdf")
DEFAULT_STORE_FACTS = os.path.join(_ROOT, "docs", "twin_usd_facts.json")

# The ROS package-search roots the compose wrapper configures (see
# scripts/compose_robot_in_store.sh). Package `shelfbot` is the repo that DECLARES
# name `shelfbot` in its package.xml AND carries the gripper meshes — that is
# ~/workspaces/shelfbot_newgripper, NOT the older ~/workspaces/shelfbot. Keep this list
# in lock-step with the wrapper; the wrapper-agreement check below asserts it.
DEFAULT_PACKAGE_ROOTS = [
    "/opt/ros/jazzy/share",
    os.path.join(HOME, "workspaces/shelfbot_newgripper"),
]
WRAPPER = os.path.join(_HERE, "compose_robot_in_store.sh")
PEDESTAL_LINK = "pedestal_link"        # the link whose ground box is the footprint
PEDESTAL_MOUNT_TOL = 1e-6              # m — URDF offset vs composer constant


def ensure_numpy():
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "ERROR: numpy is not importable.\n"
            "  Offline (recommended): use the same usd-core venv as validate_usd.py —\n"
            "    python3 -m venv .venv-usd && .venv-usd/bin/pip install usd-core\n"
            "    .venv-usd/bin/python scripts/validate_robot_urdf.py\n")
        raise SystemExit(3)


# --------------------------------------------------------------------------- #
# Report plumbing — same shape/spirit as validate_usd.Report.                  #
# --------------------------------------------------------------------------- #
class Report:
    PASS, FAIL, INFO, WARN = "PASS", "FAIL", "INFO", "WARN"

    def __init__(self):
        self.checks = []
        self.facts = {}

    def add(self, s, n, d=""):
        self.checks.append((s, n, d))

    def ok(self, n, d=""):
        self.add(self.PASS, n, d)

    def fail(self, n, d=""):
        self.add(self.FAIL, n, d)

    def info(self, n, d=""):
        self.add(self.INFO, n, d)

    def warn(self, n, d=""):
        self.add(self.WARN, n, d)

    def check(self, cond, n, d=""):
        self.add(self.PASS if cond else self.FAIL, n, d)

    @property
    def failed(self):
        return any(s == self.FAIL for s, _, _ in self.checks)

    def render(self):
        icon = {self.PASS: "PASS", self.FAIL: "FAIL", self.INFO: "····", self.WARN: "warn"}
        return "\n".join(f"[{icon[s]}] {n}" + (f": {d}" if d else "") for s, n, d in self.checks)


# --------------------------------------------------------------------------- #
# Individual check groups                                                      #
# --------------------------------------------------------------------------- #
def check_tree(robot, rep):
    """One connected kinematic tree: unique names, resolvable joints, one root."""
    import twin.urdf_probe as P

    names = list(robot.links.keys())
    dup_links = {n for n in names if names.count(n) > 1}
    jnames = [j.name for j in robot.joints]
    dup_joints = {n for n in jnames if jnames.count(n) > 1}
    rep.check(not dup_links and not dup_joints, "unique link + joint names",
              f"{len(robot.links)} links, {len(robot.joints)} joints" +
              (f"; DUP links={dup_links} joints={dup_joints}" if (dup_links or dup_joints) else ""))

    dangling = [(j.name, j.parent, j.child) for j in robot.joints
                if j.parent not in robot.links or j.child not in robot.links]
    rep.check(not dangling, "every joint references existing links",
              "all joint parent/child links exist" if not dangling
              else f"{len(dangling)} joint(s) reference a missing link, e.g. {dangling[0]}")

    roots = P.find_roots(robot)
    rep.check(len(roots) == 1, "exactly one kinematic root",
              f"root='{roots[0]}'" if len(roots) == 1 else f"roots={roots}")

    # reachability + acyclicity from the (first) root
    if roots and not dangling:
        world = {}
        try:
            world = P.link_world_transforms(robot, robot.initial_positions, root=roots[0])
        except RecursionError:
            rep.fail("kinematic tree is acyclic + fully connected", "cycle detected")
            return roots[0] if roots else None
        unreached = sorted(set(robot.links) - set(world))
        rep.check(not unreached, "kinematic tree is acyclic + fully connected",
                  "all links reachable from root"
                  if not unreached else f"{len(unreached)} unreachable: {unreached[:5]}")
    return roots[0] if roots else None


def check_meshes(robot, roots, rep):
    """Every mesh reference resolves; no absolute paths; surface package ambiguity."""
    import twin.urdf_probe as P

    refs = P.mesh_references(robot)
    distinct = sorted({fn for _, _, fn in refs})
    rep.info("mesh references", f"{len(refs)} uses across {len(distinct)} distinct files")
    rep.info("ROS package-search roots", os.pathsep.join(roots))

    missing, absolute, resolved = [], [], []
    pkgs = set()
    for fn in distinct:
        if fn.startswith("package://"):
            pkgs.add(fn[len("package://"):].split("/", 1)[0])
        if os.path.isabs(fn) or fn.startswith("file:///"):
            absolute.append(fn)
        rp, cand = P.resolve_package_uri(fn, roots)
        if rp:
            resolved.append((fn, rp))
        else:
            missing.append((fn, cand))

    rep.facts["mesh_refs"] = len(distinct)
    rep.facts["mesh_missing"] = [m[0] for m in missing]

    rep.check(not absolute, "no absolute build-machine mesh paths",
              "all mesh paths are package://-relative" if not absolute
              else f"{len(absolute)} absolute path(s), e.g. {absolute[0]}")

    if missing:
        # Diagnose each miss: is the mesh actually present in a differently-configured
        # package on this box? (the 'pointed at the wrong repo' case).
        lines = []
        for fn, _cand in missing:
            hint = _diagnose_missing(fn, roots)
            lines.append(f"{fn}" + (f"  [{hint}]" if hint else ""))
        rep.fail("every mesh reference resolves",
                 f"{len(missing)} unresolved:\n      " + "\n      ".join(lines))
    else:
        rep.ok("every mesh reference resolves",
               f"all {len(distinct)} mesh files exist under the package path")

    # Same-name package ambiguity — a real hazard here (two repos declare `shelfbot`).
    for pkg in sorted(pkgs):
        found = P.discover_packages(pkg)
        if len(found) > 1:
            rep.warn(f"package '{pkg}' name is ambiguous",
                     f"{len(found)} dirs declare/are named '{pkg}': {found} — the ROS "
                     f"package path must point at the one carrying the meshes")


def _diagnose_missing(filename, roots):
    """Return a one-line hint when a package:// mesh is missing under ``roots`` but
    present in some discovered package of the same name elsewhere on the box."""
    import twin.urdf_probe as P
    if not filename.startswith("package://"):
        return "file not found"
    pkg, _, rel = filename[len("package://"):].partition("/")
    for d in P.discover_packages(pkg):
        if os.path.isfile(os.path.join(d, rel)):
            return f"present in {d} — put it on ROS_PACKAGE_PATH instead"
    return f"package '{pkg}' not found with this file on the box"


def check_dynamics(robot, rep):
    """Every dynamic link has a positive mass + valid inertia; flag placeholders."""
    import numpy as np
    import twin.urdf_probe as P

    dyn = P.child_of_movable_joint(robot)
    bad_mass, bad_inertia, placeholders = [], [], []
    for ln in dyn:
        L = robot.links[ln]
        I = L.inertial
        if I is None or I.mass is None or I.mass <= 0:
            bad_mass.append(ln)
            continue
        if I.ixx is None:
            bad_inertia.append((ln, "no <inertia> tensor"))
            continue
        M = np.array([[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]])
        eig = np.linalg.eigvalsh(M)
        pd = bool((eig > 0).all())
        tri = (I.ixx + I.iyy >= I.izz - 1e-9 and I.iyy + I.izz >= I.ixx - 1e-9
               and I.ixx + I.izz >= I.iyy - 1e-9)
        if not (pd and tri):
            bad_inertia.append((ln, f"eig>0={pd} triangle={tri}"))
            continue
        kg = math.sqrt(max(float(eig[-1]), 0.0) / I.mass)
        # Radius of gyration vs the link's own geometry extent — a gross mismatch is a
        # placeholder inertia (physically implausible), a data-quality WARN not a fail.
        ext = _link_extent(L)
        if ext and kg > 3.0 * ext:
            placeholders.append((ln, round(kg, 3), round(ext, 3), I.mass))

    rep.check(not bad_mass, "every dynamic link has positive mass",
              f"{len(dyn)} dynamic links, all with mass > 0" if not bad_mass
              else f"missing/zero mass: {bad_mass}")
    rep.check(not bad_inertia, "every dynamic link has a valid inertia tensor",
              "positive-definite + triangle inequality hold" if not bad_inertia
              else f"invalid: {bad_inertia}")

    # No geometry link left massless (fixed frames are fine; geometry without inertia is not)
    geom_no_inertia = [n for n, L in robot.links.items()
                       if (L.visuals or L.collisions) and (L.inertial is None or L.inertial.mass is None)]
    rep.check(not geom_no_inertia, "no geometry link is left without inertia",
              "every link carrying geometry has mass+inertia" if not geom_no_inertia
              else f"geometry but no inertia: {geom_no_inertia}")

    if placeholders:
        rep.warn("implausible placeholder inertia(s)",
                 "; ".join(f"{n}: gyration r={r} m for mass {m} kg on a ~{e} m link"
                           for n, r, e, m in placeholders) +
                 " — inflates the composite EE inertia; report upstream (URDF, not this repo)")


def _link_extent(link):
    """Rough half-extent (m) of a link from its collision/visual box, else None."""
    for g in list(link.collisions) + list(link.visuals):
        if g.kind == "box" and g.box_size:
            return max(g.box_size) / 2.0
    return None


def check_joint_limits(robot, rep):
    import twin.urdf_probe as P
    bad = []
    n_mov = 0
    for j in robot.joints:
        if j.jtype not in P.MOVABLE:
            continue
        n_mov += 1
        if j.jtype == "continuous":
            continue  # continuous joints legitimately have no lower/upper
        problems = []
        if j.lower is None or j.upper is None:
            problems.append("no lower/upper")
        elif j.lower >= j.upper:
            problems.append(f"lower {j.lower} >= upper {j.upper}")
        if not j.effort or j.effort <= 0:
            problems.append("effort <= 0")
        if not j.velocity or j.velocity <= 0:
            problems.append("velocity <= 0")
        if problems:
            bad.append((j.name, problems))
    rep.check(not bad, "every movable joint has valid limits",
              f"{n_mov} movable joints, all with lower<upper + effort/velocity>0" if not bad
              else f"{len(bad)} bad: {bad}")


def check_colliders(robot, rep):
    """Collision presence + the 'visual mesh used as collider' defect."""
    reuse = []
    for n, L in robot.links.items():
        vis = {g.mesh_filename for g in L.visuals if g.kind == "mesh"}
        col = {g.mesh_filename for g in L.collisions if g.kind == "mesh"}
        if vis & col:
            reuse.append((n, sorted(vis & col)))
    rep.check(not reuse, "no collider reuses the full visual mesh",
              "collisions are dedicated meshes/primitives (not the visual mesh)" if not reuse
              else f"{len(reuse)} link(s) collide with their visual mesh: {reuse}")

    vis_no_col = [n for n, L in robot.links.items() if L.visuals and not L.collisions]
    if vis_no_col:
        rep.warn("links with visual but no collision", ", ".join(vis_no_col) +
                 " — invisible to PhysX (no contact); OK only for pure decoration")
    mesh_col = sorted(n for n, L in robot.links.items()
                      if any(g.kind == "mesh" for g in L.collisions))
    if mesh_col:
        rep.info("mesh colliders (Isaac will convex-hull/decompose on import)",
                 f"{len(mesh_col)} link(s): {mesh_col} — dedicated collision meshes, expected")


def check_store_consistency(robot, root, store_facts, rep):
    """Two numbers the composer/store already assumed: the pedestal-mount XY offset,
    and whether the arm's HOME-pose collision envelope fits the aisle it stands in."""
    import numpy as np
    import twin.urdf_probe as P
    try:
        from twin import geometry
    except ImportError:
        import geometry  # type: ignore

    world = P.link_world_transforms(robot, robot.initial_positions, root=root)

    # 1) pedestal-mount XY offset (URDF world->pedestal_link) vs composer constant.
    if PEDESTAL_LINK not in world:
        rep.warn("pedestal-mount offset matches composer",
                 f"link '{PEDESTAL_LINK}' not found — cannot verify composer's "
                 f"PEDESTAL_OFFSET_FROM_ROOT={geometry.PEDESTAL_OFFSET_FROM_ROOT}")
        ped_xy = None
    else:
        p = world[PEDESTAL_LINK][:3, 3]
        ped_xy = (float(p[0]), float(p[1]))
        ox, oy = geometry.PEDESTAL_OFFSET_FROM_ROOT[0], geometry.PEDESTAL_OFFSET_FROM_ROOT[1]
        match = abs(ped_xy[0] - ox) <= PEDESTAL_MOUNT_TOL and abs(ped_xy[1] - oy) <= PEDESTAL_MOUNT_TOL
        rep.facts["pedestal_offset_urdf"] = [round(v, 6) for v in ped_xy]
        rep.check(match, "pedestal-mount offset matches composer",
                  f"URDF world→{PEDESTAL_LINK} XY={tuple(round(v,4) for v in ped_xy)} vs "
                  f"geometry.PEDESTAL_OFFSET_FROM_ROOT=({ox},{oy}) "
                  + ("(match)" if match else "MISMATCH — composer would mis-place the robot"))

    # 2) HOME-pose collision envelope (yaw 0). Build mesh resolver from the package path
    #    so the UR collision STLs are transformed for real (not just link frames).
    roots = rep.facts.get("_package_roots", DEFAULT_PACKAGE_ROOTS)

    def resolver(fn):
        rp, _ = P.resolve_package_uri(fn, roots)
        return rp

    pts, used_frame = [], []
    for n, L in robot.links.items():
        if n not in world:
            continue
        for g in L.collisions:
            wp = P.geometry_world_points(g, world[n], mesh_resolver=resolver)
            if wp is None:
                pts.append(world[n][:3, 3][None, :]); used_frame.append(n)
            else:
                pts.append(wp)
    if not pts:
        rep.warn("home-pose envelope", "no collision geometry to measure")
        return
    Pw = np.vstack(pts)
    mn, mx = Pw.min(0), Pw.max(0)
    rep.facts["home_envelope_local"] = {"x": [round(mn[0], 3), round(mx[0], 3)],
                                        "y": [round(mn[1], 3), round(mx[1], 3)],
                                        "z": [round(mn[2], 3), round(mx[2], 3)]}
    rep.info("robot home pose (ros2_control initial_value)",
             ", ".join(f"{k.replace('ur_','').replace('_joint','')}={v}"
                       for k, v in robot.initial_positions.items()))
    rep.info("home-pose collision envelope (robot-local, root at origin)",
             f"X[{mn[0]:.3f},{mx[0]:.3f}] Y[{mn[1]:.3f},{mx[1]:.3f}] Z[{mn[2]:.3f},{mx[2]:.3f}] m"
             + (f"  (frame-only for: {sorted(set(used_frame))})" if used_frame else ""))

    if not store_facts or ped_xy is None:
        rep.info("aisle-fit check skipped", "no store facts / pedestal link")
        return

    spawn = store_facts.get("spawn", {})
    ranges = store_facts.get("geometry", {}).get("gondola_x_ranges", [])
    spawn_x = spawn.get("x")
    clearance = spawn.get("clearance_m")
    if spawn_x is None or not ranges:
        rep.info("aisle-fit check skipped", "store facts lack spawn.x / gondola_x_ranges")
        return

    # World X-envelope when placed at the spawn (yaw 0; pedestal offset_x is 0 so
    # robot-local X maps directly to world X - spawn_x).
    wxmin, wxmax = mn[0] + spawn_x, mx[0] + spawn_x
    left_faces = [x1 for (x0, x1) in ranges if x1 <= spawn_x]
    right_faces = [x0 for (x0, x1) in ranges if x0 >= spawn_x]
    left_face = max(left_faces) if left_faces else None
    right_face = min(right_faces) if right_faces else None
    if left_face is None or right_face is None:
        rep.info("aisle-fit check skipped", "spawn X is not between two gondola runs")
        return
    aisle_w = round(right_face - left_face, 4)
    clr_left = round(wxmin - left_face, 4)     # gap to the shelf on -X
    clr_right = round(right_face - wxmax, 4)    # gap to the shelf on +X
    rep.facts["aisle_fit"] = {"aisle_width_m": aisle_w, "spawn_x": spawn_x,
                              "robot_world_x": [round(wxmin, 3), round(wxmax, 3)],
                              "shelf_faces_x": [left_face, right_face],
                              "clearance_to_shelves_m": [clr_left, clr_right]}
    fits = clr_left > 0 and clr_right > 0
    rep.check(fits, "arm HOME-pose fits the aisle (does not penetrate a shelf)",
              f"aisle {aisle_w} m at X={spawn_x}; robot world X∈[{wxmin:.3f},{wxmax:.3f}]; "
              f"clearance to shelves L={clr_left} m R={clr_right} m")

    # Cross-check: planar radius from the pedestal footprint centre vs store clearance.
    # Both in the robot's own frame (invariant to placement/yaw), so compare directly.
    ped_local = world[PEDESTAL_LINK][:3, 3]
    radial = float(np.sqrt((Pw[:, 0] - ped_local[0]) ** 2
                           + (Pw[:, 1] - ped_local[1]) ** 2).max())
    if clearance is not None:
        rep.info("home-pose planar radius vs store clearance",
                 f"arm reaches {radial:.3f} m from the pedestal centre; store clearance "
                 f"here is {clearance} m " + ("(within)" if radial <= clearance else "(EXCEEDS)"))


def check_wrapper_agreement(rep):
    """The offline package roots must match what the real-run wrapper puts on the ROS
    package path, or the pre-flight would validate a different setup than Isaac sees."""
    if not os.path.isfile(WRAPPER):
        rep.info("wrapper present", f"{WRAPPER} not found (skipping agreement check)")
        return
    try:
        text = open(WRAPPER, encoding="utf-8").read()
    except OSError:
        return
    # Inspect only executable lines — the warning comment legitimately names the old repo.
    import re
    code = "\n".join(re.sub(r"#.*$", "", ln) for ln in text.splitlines()
                     if not ln.lstrip().startswith("#"))
    ng = os.path.join(HOME, "workspaces/shelfbot_newgripper")
    old = os.path.join(HOME, "workspaces/shelfbot")
    uses_ng = "workspaces/shelfbot_newgripper" in code
    # The stale/wrong root is ~/workspaces/shelfbot used as a package root (not _newgripper).
    uses_old = bool(re.search(r"workspaces/shelfbot(?![_a-zA-Z0-9])", code))
    rep.check(uses_ng and not uses_old, "compose wrapper points package 'shelfbot' at the right repo",
              "wrapper ROS package path includes shelfbot_newgripper" if (uses_ng and not uses_old)
              else f"wrapper package path {'still references ' + old if uses_old else 'does not reference ' + ng} "
                   f"— the gripper meshes live only in {ng}")


# --------------------------------------------------------------------------- #
# Store facts                                                                  #
# --------------------------------------------------------------------------- #
def load_store_facts(path):
    import json
    if not path or not os.path.isfile(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def validate(urdf_path, package_roots=None, store_facts_path=DEFAULT_STORE_FACTS,
             check_wrapper=True):
    import twin.urdf_probe as P
    rep = Report()
    if not os.path.isfile(urdf_path):
        rep.fail("robot URDF exists", urdf_path)
        return rep
    try:
        robot = P.parse_urdf(urdf_path)
    except Exception as exc:
        rep.fail("URDF parses", f"{type(exc).__name__}: {exc}")
        return rep
    rep.ok("URDF parses", f"{urdf_path} (robot name '{robot.name}')")

    roots = package_roots or DEFAULT_PACKAGE_ROOTS
    rep.facts["_package_roots"] = roots

    root = check_tree(robot, rep)
    check_meshes(robot, roots, rep)
    check_dynamics(robot, rep)
    check_joint_limits(robot, rep)
    check_colliders(robot, rep)
    if root:
        check_store_consistency(robot, root, load_store_facts(store_facts_path), rep)
    if check_wrapper:
        check_wrapper_agreement(rep)

    rep.facts.pop("_package_roots", None)
    return rep


# --------------------------------------------------------------------------- #
# Self-test — synthetic good/bad URDFs, no real robot needed.                  #
# --------------------------------------------------------------------------- #
def build_synthetic_urdf(tmpdir, good=True):
    """A tiny 2-link arm URDF + a mesh file, written under tmpdir. ``good`` resolves and
    passes; ``not good`` seeds a dangling mesh + a massless dynamic link."""
    os.makedirs(os.path.join(tmpdir, "meshes"), exist_ok=True)
    # minimal ASCII STL the good arm points at
    with open(os.path.join(tmpdir, "meshes", "link.stl"), "w") as f:
        f.write("solid s\nfacet normal 0 0 1\nouter loop\n"
                "vertex 0 0 0\nvertex 0.1 0 0\nvertex 0 0.1 0\n"
                "endloop\nendfacet\nendsolid s\n")
    mesh = "meshes/link.stl" if good else "meshes/MISSING.stl"
    mass = "1.0" if good else "0.0"
    path = os.path.join(tmpdir, "robot.urdf")
    with open(path, "w") as f:
        f.write(f"""<?xml version="1.0"?>
<robot name="synthetic">
  <link name="base"/>
  <link name="arm">
    <inertial><mass value="{mass}"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual><geometry><mesh filename="{mesh}"/></geometry></visual>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="j" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit effort="10" velocity="1" lower="-1" upper="1"/>
  </joint>
</robot>
""")
    return path


def run_selftest():
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        gd = os.path.join(td, "good"); os.makedirs(gd)
        good = build_synthetic_urdf(gd, good=True)
        rep = validate(good, package_roots=[gd], store_facts_path=None, check_wrapper=False)
        if rep.failed:
            print("SELFTEST good-urdf unexpectedly FAILED:\n" + rep.render()); ok = False
        else:
            print("SELFTEST good-urdf: PASS (clean URDF validates)")
        bd = os.path.join(td, "bad"); os.makedirs(bd)
        bad = build_synthetic_urdf(bd, good=False)
        rep2 = validate(bad, package_roots=[bd], store_facts_path=None, check_wrapper=False)
        if not rep2.failed:
            print("SELFTEST bad-urdf was NOT flagged (should fail on the dangling mesh + zero mass)")
            ok = False
        else:
            print("SELFTEST bad-urdf: correctly FAILED")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validate the shelf robot URDF offline for Isaac readiness (no GPU).")
    ap.add_argument("--urdf", default=DEFAULT_URDF, help="Robot URDF (default: %(default)s)")
    ap.add_argument("--package-path", default=None,
                    help="Override the ROS package-search roots (':'-separated). Default = the "
                         "roots the compose wrapper configures.")
    ap.add_argument("--store-facts", default=DEFAULT_STORE_FACTS,
                    help="docs/twin_usd_facts.json for the aisle-fit check (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="Also emit measured facts as JSON.")
    ap.add_argument("--selftest", action="store_true", help="Validate synthetic URDFs and exit.")
    args = ap.parse_args(argv)

    ensure_numpy()
    if args.selftest:
        return run_selftest()

    roots = args.package_path.split(os.pathsep) if args.package_path else None
    rep = validate(args.urdf, package_roots=roots, store_facts_path=args.store_facts)
    print("=" * 78)
    print(f"Robot URDF validation — {args.urdf}")
    print("=" * 78)
    print(rep.render())
    print("-" * 78)
    verdict = "FAIL" if rep.failed else "PASS"
    n_fail = sum(1 for s, _, _ in rep.checks if s == Report.FAIL)
    print(f"VERDICT: {verdict}" + (f"  ({n_fail} hard check(s) failed)" if rep.failed else ""))
    if args.json:
        import json
        print("-" * 78)
        print(json.dumps(rep.facts, indent=2, default=str))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
