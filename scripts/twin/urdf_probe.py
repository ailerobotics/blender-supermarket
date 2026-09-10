"""Read a robot URDF WITHOUT booting Isaac Sim.

Pure ``xml`` + ``numpy`` — no ``isaacsim``, no ``omni.*``, no ``pxr``, no GPU — so it
is safe to run beside the captain's Blender GUI and unit-testable with nothing but
numpy. It is the robot-side twin of ``twin/usd_probe.py`` (which reads the store USD):
it parses the URDF into a structured model, resolves every ``package://`` /relative
mesh reference the way Isaac's importer will have to, and runs forward kinematics for
a given joint configuration so the robot's real footprint in the store can be measured.

Why this exists
---------------
The store USD and the composer are both proven offline (PRs #2, #3). The robot asset
(``shelfbot_gripper_isaac.urdf``) was the one link never checked without a GPU. Isaac's
URDF importer *imports even when a mesh is not found* (it logs an error and continues —
see its CHANGELOG), so a mesh that fails to resolve produces a silently wrong robot
rather than a loud crash. That makes an offline resolution + physical-sanity check the
only thing standing between the captain and a bad first boot.

Nothing here edits the URDF; the robot description lives in another repository and is
strictly read-only.
"""
from __future__ import annotations

import math
import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Data model                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Inertial:
    mass: Optional[float]
    # inertia tensor components; None if <inertia> absent
    ixx: Optional[float] = None
    iyy: Optional[float] = None
    izz: Optional[float] = None
    ixy: Optional[float] = None
    ixz: Optional[float] = None
    iyz: Optional[float] = None
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Geometry:
    kind: str                              # "box" | "cylinder" | "sphere" | "mesh"
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # box
    box_size: Optional[Tuple[float, float, float]] = None
    # cylinder / sphere
    radius: Optional[float] = None
    length: Optional[float] = None
    # mesh
    mesh_filename: Optional[str] = None    # raw authored string, e.g. package://pkg/rel
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Link:
    name: str
    inertial: Optional[Inertial] = None
    visuals: List[Geometry] = field(default_factory=list)
    collisions: List[Geometry] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    jtype: str                             # revolute|prismatic|fixed|continuous|...
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    lower: Optional[float] = None
    upper: Optional[float] = None
    effort: Optional[float] = None
    velocity: Optional[float] = None


@dataclass
class Robot:
    name: str
    links: Dict[str, Link]
    joints: List[Joint]
    # command_interface initial_value per joint (from ros2_control), the "home" pose
    initial_positions: Dict[str, float] = field(default_factory=dict)
    source_path: str = ""

    @property
    def joints_by_name(self) -> Dict[str, Joint]:
        return {j.name: j for j in self.joints}

    @property
    def joints_by_child(self) -> Dict[str, Joint]:
        return {j.child: j for j in self.joints}


MOVABLE = {"revolute", "prismatic", "continuous"}


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #
def _f3(s: Optional[str], default=(0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    if not s:
        return tuple(float(v) for v in default)
    parts = [float(v) for v in s.replace(",", " ").split()]
    return (parts[0], parts[1], parts[2])


def _geom(node: ET.Element) -> Optional[Geometry]:
    """Parse a <visual> or <collision> element into a Geometry (first shape found)."""
    origin = node.find("origin")
    oxyz = _f3(origin.get("xyz")) if origin is not None else (0.0, 0.0, 0.0)
    orpy = _f3(origin.get("rpy")) if origin is not None else (0.0, 0.0, 0.0)
    g = node.find("geometry")
    if g is None:
        return None
    box = g.find("box")
    if box is not None:
        return Geometry("box", oxyz, orpy, box_size=_f3(box.get("size")))
    cyl = g.find("cylinder")
    if cyl is not None:
        return Geometry("cylinder", oxyz, orpy,
                        radius=float(cyl.get("radius", 0.0)),
                        length=float(cyl.get("length", 0.0)))
    sph = g.find("sphere")
    if sph is not None:
        return Geometry("sphere", oxyz, orpy, radius=float(sph.get("radius", 0.0)))
    mesh = g.find("mesh")
    if mesh is not None:
        scale = _f3(mesh.get("scale"), default=(1.0, 1.0, 1.0)) if mesh.get("scale") else (1.0, 1.0, 1.0)
        return Geometry("mesh", oxyz, orpy, mesh_filename=mesh.get("filename"), mesh_scale=scale)
    return None


def parse_urdf(path: str) -> Robot:
    """Parse a URDF file into a :class:`Robot`. Raises on malformed XML."""
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"root element is <{root.tag}>, expected <robot>")

    links: Dict[str, Link] = {}
    for lnode in root.findall("link"):
        name = lnode.get("name")
        link = Link(name=name)
        inode = lnode.find("inertial")
        if inode is not None:
            mnode = inode.find("mass")
            iner = inode.find("inertia")
            onode = inode.find("origin")
            link.inertial = Inertial(
                mass=float(mnode.get("value")) if mnode is not None and mnode.get("value") is not None else None,
                ixx=float(iner.get("ixx")) if iner is not None else None,
                iyy=float(iner.get("iyy")) if iner is not None else None,
                izz=float(iner.get("izz")) if iner is not None else None,
                ixy=float(iner.get("ixy", 0.0)) if iner is not None else None,
                ixz=float(iner.get("ixz", 0.0)) if iner is not None else None,
                iyz=float(iner.get("iyz", 0.0)) if iner is not None else None,
                origin_xyz=_f3(onode.get("xyz")) if onode is not None else (0.0, 0.0, 0.0),
            )
        for vnode in lnode.findall("visual"):
            gm = _geom(vnode)
            if gm:
                link.visuals.append(gm)
        for cnode in lnode.findall("collision"):
            gm = _geom(cnode)
            if gm:
                link.collisions.append(gm)
        links[name] = link

    joints: List[Joint] = []
    for jnode in root.findall("joint"):
        origin = jnode.find("origin")
        axis = jnode.find("axis")
        limit = jnode.find("limit")
        joints.append(Joint(
            name=jnode.get("name"),
            jtype=jnode.get("type"),
            parent=jnode.find("parent").get("link"),
            child=jnode.find("child").get("link"),
            origin_xyz=_f3(origin.get("xyz")) if origin is not None else (0.0, 0.0, 0.0),
            origin_rpy=_f3(origin.get("rpy")) if origin is not None else (0.0, 0.0, 0.0),
            axis=_f3(axis.get("xyz"), default=(1.0, 0.0, 0.0)) if axis is not None else (1.0, 0.0, 0.0),
            lower=float(limit.get("lower")) if limit is not None and limit.get("lower") is not None else None,
            upper=float(limit.get("upper")) if limit is not None and limit.get("upper") is not None else None,
            effort=float(limit.get("effort")) if limit is not None and limit.get("effort") is not None else None,
            velocity=float(limit.get("velocity")) if limit is not None and limit.get("velocity") is not None else None,
        ))

    initial: Dict[str, float] = {}
    for rc in root.findall("ros2_control"):
        for j in rc.findall("joint"):
            jn = j.get("name")
            ci = j.find("command_interface")
            if ci is None:
                continue
            for p in ci.findall("param"):
                if p.get("name") == "initial_value":
                    try:
                        initial[jn] = float(p.text)
                    except (TypeError, ValueError):
                        pass
    return Robot(name=root.get("name", ""), links=links, joints=joints,
                 initial_positions=initial, source_path=os.path.abspath(path))


# --------------------------------------------------------------------------- #
# Kinematic-tree helpers                                                       #
# --------------------------------------------------------------------------- #
def find_roots(robot: Robot) -> List[str]:
    """Link names that are never a joint child — the tree root(s)."""
    children = {j.child for j in robot.joints}
    return [n for n in robot.links if n not in children]


def child_of_movable_joint(robot: Robot) -> Dict[str, Joint]:
    """Map link name -> the movable joint that has it as child (dynamic links)."""
    return {j.child: j for j in robot.joints if j.jtype in MOVABLE}


# --------------------------------------------------------------------------- #
# Forward kinematics (pure numpy, no simulator)                                #
# --------------------------------------------------------------------------- #
def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw -> 3x3 rotation R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_rotation(axis: Sequence[float], q: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(3)
    a = a / n
    x, y, z = a
    c, s, C = math.cos(q), math.sin(q), 1.0 - math.cos(q)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _homog(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def joint_transform(joint: Joint, q: float) -> np.ndarray:
    """4x4 transform from a joint's parent frame to its child frame at value ``q``."""
    T = _homog(rpy_to_matrix(joint.origin_rpy), joint.origin_xyz)
    if joint.jtype in ("revolute", "continuous"):
        T = T @ _homog(_axis_rotation(joint.axis, q), (0, 0, 0))
    elif joint.jtype == "prismatic":
        a = np.asarray(joint.axis, dtype=float)
        n = np.linalg.norm(a)
        a = a / n if n else a
        T = T @ _homog(np.eye(3), a * q)
    return T


def link_world_transforms(robot: Robot, config: Optional[Dict[str, float]] = None,
                          root: Optional[str] = None) -> Dict[str, np.ndarray]:
    """World 4x4 transform of every link frame at joint configuration ``config``
    (missing joints default to 0). Root link frame is identity (the URDF ``world``)."""
    config = config or {}
    roots = find_roots(robot)
    if root is None:
        root = roots[0] if roots else next(iter(robot.links))
    by_parent: Dict[str, List[Joint]] = {}
    for j in robot.joints:
        by_parent.setdefault(j.parent, []).append(j)

    world: Dict[str, np.ndarray] = {root: np.eye(4)}
    stack = [root]
    while stack:
        parent = stack.pop()
        for j in by_parent.get(parent, []):
            q = float(config.get(j.name, 0.0))
            world[j.child] = world[parent] @ joint_transform(j, q)
            stack.append(j.child)
    return world


# --------------------------------------------------------------------------- #
# Geometry -> world points (for the footprint / reach envelope)                #
# --------------------------------------------------------------------------- #
def _box_corners(size: Sequence[float]) -> np.ndarray:
    sx, sy, sz = (s / 2.0 for s in size)
    return np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])


def _read_stl_vertices(path: str) -> Optional[np.ndarray]:
    """Return the Nx3 vertex array of a binary or ASCII STL, or None on failure."""
    try:
        with open(path, "rb") as f:
            head = f.read(84)
            if len(head) < 84:
                # try ASCII
                f.seek(0)
                text = f.read().decode("ascii", "ignore")
                verts = [list(map(float, ln.split()[1:4]))
                         for ln in text.splitlines() if ln.strip().startswith("vertex")]
                return np.array(verts) if verts else None
            n_tri = struct.unpack("<I", head[80:84])[0]
            # Heuristic: is this really binary? Expected size vs ASCII fallback.
            expected = 84 + n_tri * 50
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == expected and n_tri > 0:
                f.seek(84)
                data = f.read(n_tri * 50)
                verts = []
                for i in range(n_tri):
                    off = i * 50 + 12  # skip normal
                    tri = struct.unpack("<9f", data[off:off + 36])
                    verts.extend([tri[0:3], tri[3:6], tri[6:9]])
                return np.array(verts)
            # ASCII
            f.seek(0)
            text = f.read().decode("ascii", "ignore")
            verts = [list(map(float, ln.split()[1:4]))
                     for ln in text.splitlines() if ln.strip().startswith("vertex")]
            return np.array(verts) if verts else None
    except Exception:
        return None


def _read_obj_vertices(path: str) -> Optional[np.ndarray]:
    try:
        verts = []
        with open(path, "r", errors="ignore") as f:
            for ln in f:
                if ln.startswith("v "):
                    parts = ln.split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(verts) if verts else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# package:// mesh resolution (the way Isaac's importer must resolve them)       #
# --------------------------------------------------------------------------- #
import glob as _glob


def parse_package_path(value: Optional[str]) -> List[str]:
    """Split a ``ROS_PACKAGE_PATH``-style ``:``-separated string into roots."""
    if not value:
        return []
    return [p for p in value.split(os.pathsep) if p]


def package_xml_name(pkg_dir: str) -> Optional[str]:
    """The ``<name>`` in ``pkg_dir/package.xml``, or None if absent/unparseable."""
    px = os.path.join(pkg_dir, "package.xml")
    if not os.path.isfile(px):
        return None
    try:
        n = ET.parse(px).getroot().find("name")
        return n.text.strip() if n is not None and n.text else None
    except Exception:
        return None


# Where a package named ``pkg`` may live on this box. Kept small and explicit so a
# reviewer can see exactly what is scanned (no unbounded filesystem walk).
DEFAULT_DISCOVERY_GLOBS = (
    os.path.expanduser("~/workspaces/*"),
    "/opt/ros/*/share/*",
)


def discover_packages(pkg: str, globs: Optional[Sequence[str]] = None) -> List[str]:
    """Every directory on this box that declares ROS package ``pkg`` (package.xml
    ``<name>`` matches, or the directory is literally named ``pkg``). Newest first is
    not guaranteed — order is glob order. Used to diagnose 'pointed at the wrong repo'
    and same-name ambiguity. ``globs`` defaults to the module-level
    :data:`DEFAULT_DISCOVERY_GLOBS` (read at call time so it stays overridable)."""
    if globs is None:
        globs = DEFAULT_DISCOVERY_GLOBS
    hits = []
    for pat in globs:
        for d in sorted(_glob.glob(pat)):
            if not os.path.isdir(d):
                continue
            if package_xml_name(d) == pkg or os.path.basename(os.path.normpath(d)) == pkg:
                hits.append(os.path.realpath(d))
    # de-dup, preserve order
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h); out.append(h)
    return out


def resolve_package_uri(filename: str, roots: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    """Resolve a mesh ``filename`` (``package://pkg/rel``, ``file://…`` or a bare path)
    against ordered package-search ``roots`` the way a ROS_PACKAGE_PATH resolver does.

    Returns ``(resolved_abspath_or_None, candidates_tried)``. For ``package://pkg/rel``
    each root R is tried as ``R/pkg/rel`` (R is a parent-of-packages dir, e.g.
    ``/opt/ros/jazzy/share``), as ``R/rel`` (R is itself the package root, i.e.
    ``basename(R)==pkg`` or R declares ``pkg``), and as ``R/share/pkg/rel`` (R is an
    ament prefix). The first existing file wins."""
    cand: List[str] = []
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        pkg, _, rel = rest.partition("/")
        for R in roots:
            cand.append(os.path.join(R, pkg, rel))
            if os.path.basename(os.path.normpath(R)) == pkg or package_xml_name(R) == pkg:
                cand.append(os.path.join(R, rel))
            cand.append(os.path.join(R, "share", pkg, rel))
    elif filename.startswith("file://"):
        cand.append(filename[len("file://"):])
    elif os.path.isabs(filename):
        cand.append(filename)
    else:
        for R in roots:
            cand.append(os.path.join(R, filename))
    for c in cand:
        if os.path.isfile(c):
            return os.path.realpath(c), cand
    return None, cand


def mesh_references(robot: Robot) -> List[Tuple[str, str, str]]:
    """Distinct (link, role, filename) for every mesh geometry in the robot."""
    out = []
    for n, L in robot.links.items():
        for g in L.visuals:
            if g.kind == "mesh" and g.mesh_filename:
                out.append((n, "visual", g.mesh_filename))
        for g in L.collisions:
            if g.kind == "mesh" and g.mesh_filename:
                out.append((n, "collision", g.mesh_filename))
    return out


def geometry_world_points(geom: Geometry, link_world: np.ndarray,
                          mesh_resolver=None) -> Optional[np.ndarray]:
    """World-space points bounding a geometry: box/cylinder/sphere corners, or the
    transformed mesh vertices (via ``mesh_resolver(filename) -> path`` for meshes).
    Returns None when a mesh cannot be located/read (caller falls back to the frame)."""
    local = _homog(rpy_to_matrix(geom.origin_rpy), geom.origin_xyz)
    T = link_world @ local
    if geom.kind == "box" and geom.box_size:
        pts = _box_corners(geom.box_size)
    elif geom.kind == "sphere" and geom.radius:
        r = geom.radius
        pts = _box_corners((2 * r, 2 * r, 2 * r))
    elif geom.kind == "cylinder" and geom.radius and geom.length:
        r, h = geom.radius, geom.length
        pts = _box_corners((2 * r, 2 * r, h))
    elif geom.kind == "mesh" and geom.mesh_filename is not None:
        if mesh_resolver is None:
            return None
        p = mesh_resolver(geom.mesh_filename)
        if not p or not os.path.exists(p):
            return None
        ext = os.path.splitext(p)[1].lower()
        verts = _read_stl_vertices(p) if ext == ".stl" else (
            _read_obj_vertices(p) if ext == ".obj" else None)
        if verts is None or len(verts) == 0:
            return None
        pts = verts * np.asarray(geom.mesh_scale, dtype=float)
    else:
        return None
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ hom.T).T[:, :3]
