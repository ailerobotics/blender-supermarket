"""Static world obstacles for cuMotion + a CPU sphere-clearance check (no GPU, no ROS).

cuMotion plans the arm and the attached carton against only (a) the nvblox ESDF voxel world and
(b) the MoveIt planning-scene world CollisionObjects.  The ESDF is limited to what the forward-looking
eye-in-hand camera observed, so the backstock rack above/around the pick slot and the cart cage behind
the robot are never imaged -- the carton swings into them during the tuck/pan fold (run-249: wrist_1
pinned ~1.3 rad short, a rigid environment reaction, no robot self-collision and no joint limit).  The
measured ESDF grid of that run was shape [409, 217, 257] voxels at origin [-2.00, -0.08, -0.48] m,
voxel 0.01 m: it spans the rack region (base y up to +2.09 m) yet the rack voxels the carton grazes are
unobserved (==free), and the cart at base y ~= -0.8 m is outside the grid entirely (y_min -0.08).  No
enlargement of the query can synthesise occupancy for a fixture the camera never saw, so the two
fixtures must be published as explicit static box CollisionObjects.

curobo's collision model is rooted at the arm base frame (``set_base_frame: ur_base_link``), so ONLY the
arm-chain links (ur_base_link_inertia + shoulder..wrist + gripper plates/cube/suction_cups) and the
attached carton are checked against world obstacles -- the fixed mobile base below the root (chassis,
wheels, hitch) is not in the model (exactly why the carton otherwise passes through everything there).
curobo also ignores the MoveIt allowed-collision matrix for WORLD obstacles, so a box overlapping the
start/stow configuration rejects the plan with INVALID_INITIAL_CSPACE (the base-box lesson, firstmate
295).  This module therefore (1) describes each fixture as one or more boxes in the ur_base_link frame
and (2) proves on CPU, from the same URDF FK + XRDF collision spheres the asset model ships, that those
boxes clear every arm (and optionally carton) collision sphere at the stow and start configurations
before any GPU run.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from twin.urdf_probe import parse_urdf, link_world_transforms

BASE_FRAME = 'ur_base_link'


@dataclass(frozen=True)
class Box:
    """One axis-aligned-in-its-own-frame box obstacle, posed in the ur_base_link frame.

    ``size`` is full extents (x, y, z) in metres; ``center`` the box centre in ur_base_link;
    ``rpy`` an optional intrinsic roll/pitch/yaw of the box frame (default axis-aligned)."""
    id: str
    size: Tuple[float, float, float]
    center: Tuple[float, float, float]
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def rotation(self) -> np.ndarray:
        r, p, y = self.rpy
        cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                                  math.sin(p), math.cos(y), math.sin(y))
        rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
        ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
        rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
        return rz @ ry @ rx

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of each world point to the box surface: negative inside, positive
        outside (exact Euclidean distance to the box, the standard AABB SDF in box frame)."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        local = (pts - np.asarray(self.center)) @ self.rotation()  # world->box (R is orthonormal)
        half = np.asarray(self.size, dtype=float) / 2.0
        q = np.abs(local) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside


def load_collision_spheres(xrdf_path: str) -> Dict[str, List[dict]]:
    """The ``collision`` geometry sphere model keyed by link, as cuMotion loads it for world checks."""
    import yaml
    xrdf = yaml.safe_load(Path(xrdf_path).read_text())
    name = xrdf.get('collision', {}).get('geometry')
    spheres = xrdf.get('geometry', {}).get(name, {}).get('spheres', {}) if isinstance(name, str) else {}
    if not spheres:
        raise ValueError('xrdf has no collision sphere geometry')
    return spheres


def model_link_transforms(urdf_path: str, config: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Link transforms in the ur_base_link frame for exactly the links in curobo's kinematic
    model (the arm chain rooted at ur_base_link); the fixed base below the root is excluded."""
    robot = parse_urdf(str(urdf_path))
    return link_world_transforms(robot, config or {}, root=BASE_FRAME)


def arm_spheres_world(urdf_path: str, xrdf_path: str,
                      config: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """World (ur_base_link-frame) centres + radii of every collision sphere on a curobo-model link
    at ``config``.  Returns (centres Nx3, radii N, link-name-per-sphere)."""
    transforms = model_link_transforms(urdf_path, config)
    spheres = load_collision_spheres(xrdf_path)
    centres, radii, owners = [], [], []
    for link, entries in spheres.items():
        tf = transforms.get(link)
        if tf is None:                       # a fixed-base link outside curobo's model: not checked
            continue
        local = np.array([s['center'] for s in entries], dtype=float)
        world = local @ tf[:3, :3].T + tf[:3, 3]
        centres.append(world)
        radii.append(np.array([float(s['radius']) for s in entries]))
        owners.extend([link] * len(entries))
    if not centres:
        raise ValueError('no curobo-model collision spheres found (check XRDF/URDF frames)')
    return np.vstack(centres), np.concatenate(radii), owners


def carton_cover_local(dims: Sequence[float], *, offset_tcp: Sequence[float] = (0.0, 0.0, 0.20),
                       pitch: float = 0.08) -> Tuple[np.ndarray, float]:
    """Conservative sphere cover of the carried carton box, in the TCP/attached-object LOCAL frame.

    ``dims`` is the carton box (x, y, z) in metres; the carton centre sits ``offset_tcp`` (default
    0.20 m along tool-z, the measured side-grip of the 0.6 m carton) from the TCP.  Fills the box with
    a grid at ``pitch`` and gives each cell a circumsphere, so the box is conservatively enclosed.
    Returns (centres Nx3 in the local frame, single radius).  This is the ONE carton-cover definition
    shared by the CPU clearance check (``carton_spheres_world``) and the XRDF ``attached_object``
    placeholder the generator emits, so both reason about the same carton model."""
    dims = np.asarray(dims, dtype=float)
    centre_local = np.asarray(offset_tcp, dtype=float)
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        raise ValueError('carton dims must be three finite positive extents')
    axes = [np.linspace(-d / 2.0, d / 2.0, max(2, int(math.ceil(d / pitch)) + 1)) for d in dims]
    grid = np.array([[x, y, z] for x in axes[0] for y in axes[1] for z in axes[2]]) + centre_local
    step = np.array([(a[1] - a[0]) if len(a) > 1 else 0.0 for a in axes])
    radius = float(np.linalg.norm(step) / 2.0)
    return grid, radius


def carton_spheres_world(urdf_path: str, config: Dict[str, float], *, dims: Sequence[float],
                         offset_tcp: Sequence[float] = (0.0, 0.0, 0.20),
                         tcp_link: str = 'gripper_pad_tcp', pitch: float = 0.08
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate the attached carton as a sphere grid at the gripper, in the ur_base_link frame.

    Places the shared ``carton_cover_local`` grid at the TCP link's pose -- the attached-object
    analogue of the XRDF cover, in world (ur_base_link) coordinates for the clearance check."""
    transforms = model_link_transforms(urdf_path, config)
    if tcp_link not in transforms:
        raise ValueError(f'tcp link {tcp_link} not in curobo model')
    tf = transforms[tcp_link]
    grid, radius = carton_cover_local(dims, offset_tcp=offset_tcp, pitch=pitch)
    world = grid @ tf[:3, :3].T + tf[:3, 3]
    return world, np.full(len(world), radius)


def clearance(box: Box, centres: np.ndarray, radii: np.ndarray) -> float:
    """Min signed gap (metres) between the box and the sphere set: surface distance minus radius.
    Negative == penetration (would make the box block a plan start / grasp goal at ``config``)."""
    return float(np.min(box.signed_distance(centres) - radii))


# Links the carton legitimately rests against at grasp (the gripper seat + its own tool-chain
# neighbours); the planner must IGNORE these but still check the carton against the rest of the arm.
# This set is the single source of the attached_object self_collision.ignore row (generate_assets
# emits exactly these, intersected with the links that carry spheres in the CAD model).
CARTON_SEAT_LINKS = ('suction_cups', 'cube_without_surface_plates', 'ur_wrist_3_link',
                     'ur_wrist_2_link') + tuple('surface_plate_%d' % i for i in range(1, 7))


def carton_arm_clearance(urdf_path: str, xrdf_path: str, config: Dict[str, float], *,
                         carton: Optional[dict] = None,
                         seat_links: Sequence[str] = CARTON_SEAT_LINKS) -> dict:
    """Min signed gap (metres) between the attached carton and each arm link curobo should check it
    against (every curobo-model link except the ``seat_links`` the carton legitimately touches).

    This is the planner-side question made measurable on CPU: cuMotion plans the carton against the
    world (nvblox ESDF + world obstacles) but the asset design also intends it self-checked against the
    arm. A negative result here at a planned configuration means the arm planner is NOT catching a
    carton-vs-arm contact it should -- exactly the gap the XRDF attached_object self-collision
    registration (add_frame + ignore, spheres supplied at runtime) is meant to close. Used to
    reproduce the run-250 finding and to prove the carry fold contact the self-collision must catch."""
    ac, ar, owners = arm_spheres_world(urdf_path, xrdf_path, config)
    carton = carton or {'dims': [0.6, 0.4, 0.16]}
    cc, cr = carton_spheres_world(urdf_path, config, **carton)
    seat = set(seat_links)
    per_link, worst = {}, None
    for j, link in enumerate(owners):
        if link in seat:
            continue
        gaps = np.linalg.norm(cc - ac[j], axis=1) - cr - ar[j]
        g = float(np.min(gaps))
        if link not in per_link or g < per_link[link]:
            per_link[link] = g
        if worst is None or g < worst[1]:
            worst = (link, g)
    return {'config': dict(config), 'worst_link': None if worst is None else worst[0],
            'clearance_m': None if worst is None else worst[1],
            'clear': worst is None or worst[1] > 0.0, 'per_link': per_link}


def simulate_object_attachment_reload(xrdf: dict, n_spheres: int = 34, radius: float = 0.05) -> dict:
    """Model cuMotion's ObjectAttachmentNode.UpdateRobotDescriptionWithSpheres reload on CPU.

    On attach the node reads the geometry name from the XRDF ``collision`` section, generates the
    carton collision spheres, transforms them into the grasp frame, and writes them under
    geometry[geom].spheres['attached_object'] -- OVERWRITING whatever was there -- then reloads the
    robot description via set_robot_description (run-259 host-stack.log: "Updating robot description
    with 34 collision spheres for link 'attached_object'" -> "Loading robot description from memory").
    Returns a NEW XRDF dict with that mutation applied, so a test can check the post-attach robot
    description the planner actually runs the carry plan against. The sphere values are synthetic
    (the real ones come from the carton mesh at runtime); only their PRESENCE under that frame key
    matters to the self-collision mask resolution modelled by resolve_self_collision_masks()."""
    import copy
    out = copy.deepcopy(xrdf)
    geom = out['collision']['geometry']
    out['geometry'][geom]['spheres']['attached_object'] = [
        {'center': [0.0, 0.0, 0.20], 'radius': float(radius)} for _ in range(int(n_spheres))]
    return out


def kinematic_frames(xrdf: dict, urdf_links: Iterable[str]) -> set:
    """The frames curobo recognises as collision-sphere carriers: every URDF link plus every frame
    inserted by an ``add_frame`` modifier. Geometry spheres keyed by a name that is NOT such a frame
    are orphaned and dropped -- the root cause of run-256/259: ``attached_object`` carried spheres in
    the geometry but, lacking an add_frame modifier, curobo logged 'no self-collision spheres
    specified' and never added its masks."""
    frames = set(urdf_links)
    for mod in xrdf.get('modifiers', []) or []:
        if isinstance(mod, dict) and 'add_frame' in mod:
            frames.add(mod['add_frame']['frame_name'])
    return frames


def resolve_self_collision_masks(xrdf: dict, urdf_links: Iterable[str]) -> dict:
    """Model how cuMotion turns ``self_collision.ignore`` pairs into actual self-collision masks.

    A pair (A, B) becomes a mask ONLY if BOTH A and B are kinematic frames (see kinematic_frames)
    that carry >=1 sphere in the self_collision geometry; otherwise curobo logs "Found self-collision
    mask between frame '<A>' and frame '<B>', but frame '<X>' has no self-collision spheres specified.
    Self-collision mask will not be added" and SKIPS the pair. This is the exact predicate behind the
    run-256/259 warnings. Returns {'resolved': [...], 'skipped': [...]} of (A, B) pairs."""
    frames = kinematic_frames(xrdf, urdf_links)
    geom = xrdf['geometry'][xrdf['self_collision']['geometry']]['spheres']

    def carries(frame):
        return frame in frames and bool(geom.get(frame))

    resolved, skipped = [], []
    for a, others in (xrdf.get('self_collision', {}).get('ignore', {}) or {}).items():
        for b in others:
            (resolved if carries(a) and carries(b) else skipped).append((a, b))
    return {'resolved': resolved, 'skipped': skipped}


def check_configuration(urdf_path: str, xrdf_path: str, boxes: Sequence[Box],
                        config: Dict[str, float], *, carton: Optional[dict] = None) -> dict:
    """Per-box min clearance (and worst sphere/link) for the arm -- and, if ``carton`` given, the
    attached carton -- at a single configuration.  ``carton`` is a dict of carton_spheres_world kwargs."""
    ac, ar, owners = arm_spheres_world(urdf_path, xrdf_path, config)
    sets = [('arm', ac, ar, owners)]
    if carton is not None:
        cc, cr = carton_spheres_world(urdf_path, config, **carton)
        sets.append(('carton', cc, cr, ['carton'] * len(cc)))
    results = []
    for box in boxes:
        worst = None
        for label, centres, radii, names in sets:
            gaps = box.signed_distance(centres) - radii
            i = int(np.argmin(gaps))
            cand = (float(gaps[i]), label, names[i])
            if worst is None or cand[0] < worst[0]:
                worst = cand
        results.append({'box': box.id, 'clearance_m': worst[0],
                        'worst_set': worst[1], 'worst_link': worst[2],
                        'clear': worst[0] > 0.0})
    return {'config': dict(config), 'boxes': results, 'clear': all(r['clear'] for r in results)}


def boxes_from_specs(specs: Sequence[dict]) -> List[Box]:
    """Build Box list from JSON-friendly specs: {id, size[3], center[3], rpy?[3]} in ur_base_link."""
    out = []
    for s in specs:
        if set(('id', 'size', 'center')) - set(s):
            raise ValueError(f'world obstacle spec missing id/size/center: {s}')
        size = tuple(float(v) for v in s['size'])
        center = tuple(float(v) for v in s['center'])
        rpy = tuple(float(v) for v in s.get('rpy', (0.0, 0.0, 0.0)))
        if len(size) != 3 or len(center) != 3 or len(rpy) != 3 or any(v <= 0 for v in size):
            raise ValueError(f'world obstacle spec invalid size/center/rpy: {s}')
        out.append(Box(id=str(s['id']), size=size, center=center, rpy=rpy))
    return out


def transport_config(manifest_path: str) -> Dict[str, float]:
    """The stow (transport) joint configuration from the mobile manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    return {k: float(v) for k, v in manifest.get('transport_positions', {}).items()}


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='CPU sphere-clearance of static world obstacles vs the arm.')
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--xrdf', required=True)
    ap.add_argument('--manifest', required=True, help='mobile manifest (for the stow/transport config)')
    ap.add_argument('--obstacles', required=True, help='JSON file: {"world_obstacles": [ {id,size,center,rpy?} ]}')
    ap.add_argument('--config', action='append', default=[],
                    help='extra named config as name=j0,j1,... (6 arm joints in ur_ order); repeatable')
    ap.add_argument('--carton', action='store_true', help='also model the attached carton at the gripper')
    ap.add_argument('--carton-dims', default='0.6,0.4,0.16')
    args = ap.parse_args(argv)
    specs = json.loads(Path(args.obstacles).read_text()).get('world_obstacles', [])
    boxes = boxes_from_specs(specs)
    from twin.restock_host.config import JOINTS
    configs = {'stow': transport_config(args.manifest)}
    for item in args.config:
        name, _, vals = item.partition('=')
        qs = [float(v) for v in vals.split(',')]
        configs[name] = dict(zip(JOINTS, qs))
    carton = None
    if args.carton:
        carton = {'dims': [float(v) for v in args.carton_dims.split(',')]}
    ok = True
    for name, cfg in configs.items():
        res = check_configuration(args.urdf, args.xrdf, boxes, cfg, carton=carton)
        ok = ok and res['clear']
        print(f'[{name}] clear={res["clear"]}')
        for r in res['boxes']:
            print(f'    {r["box"]:16s} clearance={r["clearance_m"]:+.4f} m  '
                  f'(worst {r["worst_set"]}:{r["worst_link"]})')
    print('OVERALL', 'CLEAR' if ok else 'BLOCKED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(_cli())
