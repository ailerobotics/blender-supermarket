#!/usr/bin/env python3
"""CPU geometry proof: can the carton get into / out of the rollwagen upper-shelf cage on a clear path?

No Isaac, ROS or GPU -- usd-core (the authored cage via ``twin.mobile_assets.author_cart``) + numpy + the
mobile URDF. It quantifies the two motions sim-video-preflight-review-s1 flagged (L1 collect place into the
slot, L2 transfer lift out of it), and is the evidence behind shipping the sim-only contact filter
(``cage_collision: filtered``, ``twin.restock.cage_collision``) instead of a precision path:

* STATIC: the 0.6 x 0.4 x 0.16 carton in its slot (cart body ``load_center``, upper shelf, cart-relative yaw
  -90 deg) -- clearance per side to the front corner posts and the side-wall bars, the travel needed to
  leave the cage through the open +x mouth, and the rise needed to leave it over the top.
* PATHS, per UR10 IK branch of the side-grip TCP on the carton's mouth face (``WallCartonPolicy`` geometry):
  COLLECT = the place at ``mission_poses.place_approach``; TRANSFER = the pick from the coupled park pose
  (cart straight behind the base on the hitch). For each: how far a fixed-orientation straight line out
  along the slot axis stays IK-feasible, how far the controller's existing ``straight_line=True`` move
  bows off that line -- it is ONE 2-point joint-space smoothstep (``ros_adapter._straight_line_goal``), not
  a Cartesian line -- versus K Cartesian segments, and the carton-to-cage clearance along each.

    PYTHONPATH=scripts python3 scripts/check_cage_clearance.py [--urdf MOBILE_UR10_URDF] [--json OUT]

Exit 0 always (a measurement, not a gate). A kinematic sweep of the carton box, not a physics run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
ARM = ('ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
       'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint')
CARTON = (0.6, 0.4, 0.16)                 # handling_unit.dimensions_m
LOAD_CENTER = (0.18, 0.0)                 # initial_staging.cart_load_center_body_xy
SUPPORT_Z = 0.9325                        # initial_staging.cart_support_z_world (upper shelf top)
CARTON_YAW_IN_CART = -math.pi / 2         # unit_yaws_world.cart - cart yaw
TCP_SURFACE_OFFSET = 0.00275              # host profile tcp_surface_offset
GRASP_QUAT = (0.5, 0.5, 0.5, 0.5)         # host profile tcp_quaternion (side grip, shelfbot wall_grip)
PLACE_APPROACH = (-4.5, 15.4, math.pi)    # mission_poses.place_approach (collect place base pose)
CART_AT_COLLECT = (-5.73, 15.25, 0.0)     # initial_staging.cart_body_pose (static pre-couple)
LIFT = 0.03                               # WallCartonPolicy lift (preplace/lift rise)


# ---------------------------------------------------------------- cage + carton geometry (cart frame)
def cage_boxes(spec=None):
    """{name: (lo, hi)} world-z AABBs of every rollwagen CAGE collider, in the cart body frame (x fwd to the
    open mouth, origin at the cart's ground point), read from the authored USD -- not re-derived."""
    from pxr import Gf, Usd, UsdGeom
    from twin.mobile import MobileSpec
    from twin.mobile_assets import author_cart
    from twin.restock.cage_collision import CAGE_PREFIXES
    stage = Usd.Stage.CreateInMemory()
    cart = author_cart(stage, spec=spec or MobileSpec())
    root = UsdGeom.Xformable(stage.GetPrimAtPath(cart['root']))
    boxes = {}
    for child in stage.GetPrimAtPath(cart['body']).GetChildren():
        if not child.GetName().startswith(CAGE_PREFIXES):
            continue
        m = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        corners = np.array([list(m.Transform(Gf.Vec3d(x, y, z)))
                            for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        boxes[child.GetName()] = (corners.min(0), corners.max(0))
    del root
    return boxes


def carton_rest_pose():
    """4x4 carton pose in the cart frame at rest in the slot."""
    T = np.eye(4)
    T[:3, :3] = _rotz(CARTON_YAW_IN_CART)
    T[:3, 3] = (*LOAD_CENTER, SUPPORT_Z + CARTON[2] / 2)
    return T


def static_clearance(boxes):
    """The slot's clearances (m), from the authored boxes and the carton at rest (axis-aligned in the cart)."""
    half = np.abs(_rotz(CARTON_YAW_IN_CART) @ (np.array(CARTON) / 2))   # carton half extents in cart axes
    cx, cy = LOAD_CENTER
    posts = [b for n, b in boxes.items() if n.startswith('post_')]
    front = [b for b in posts if b[0][0] > 0]
    sides = [b for n, b in boxes.items() if n.startswith('side_')]
    post_inner = min(min(abs(lo[1]), abs(hi[1])) for lo, hi in front)
    side_inner = min(min(abs(lo[1]), abs(hi[1])) for lo, hi in sides)
    mouth_x = max(hi[0] for lo, hi in front)                  # outer face of the front posts
    top = max(hi[2] for lo, hi in boxes.values())
    return dict(carton_half_extents_cart=tuple(round(float(v), 4) for v in half),
                front_post_gap_m=round(2 * post_inner, 4),
                front_post_clearance_per_side_m=round(post_inner - (abs(cy) + half[1]), 4),
                side_bar_clearance_per_side_m=round(side_inner - (abs(cy) + half[1]), 4),
                carton_front_face_x=round(cx + half[0], 4), front_post_x_range=(
                    round(min(lo[0] for lo, hi in front), 4), round(mouth_x, 4)),
                mouth_exit_travel_m=round(mouth_x - (cx - half[0]), 4),
                cage_top_z=round(float(top), 4),
                vertical_exit_rise_m=round(float(top) - SUPPORT_Z, 4))


def carton_clearance(T_carton_cart, boxes, step=0.01):
    """Min signed distance (m, negative = penetration) between the carton box at ``T_carton_cart`` and every
    cage AABB, by sampling both: carton edges/faces against each AABB and each AABB's edges against the
    carton (exact point-to-box distances; the error is bounded by the sample spacing along the edge)."""
    h = np.array(CARTON) / 2
    grid = [np.arange(-v, v + 1e-9, step) for v in h]
    pts = []
    for ax in range(3):                       # the 6 faces as grids
        a, b = [i for i in range(3) if i != ax]
        A, B = np.meshgrid(grid[a], grid[b], indexing='ij')
        for sgn in (-1, 1):
            P = np.zeros((A.size, 3))
            P[:, a], P[:, b], P[:, ax] = A.ravel(), B.ravel(), sgn * h[ax]
            pts.append(P)
    P = np.vstack(pts)
    Pw = P @ T_carton_cart[:3, :3].T + T_carton_cart[:3, 3]
    best = math.inf
    inv = np.linalg.inv(T_carton_cart)
    for lo, hi in boxes.values():
        best = min(best, float(_box_sdf(Pw, lo, hi).min()))
        # the AABB's 12 edges sampled, measured against the carton box in its own frame
        E = _box_edge_points(lo, hi, step)
        El = E @ inv[:3, :3].T + inv[:3, 3]
        best = min(best, float(_box_sdf(El, -h, h).min()))
    return best


def _box_sdf(P, lo, hi):
    c, e = (np.asarray(lo) + np.asarray(hi)) / 2, (np.asarray(hi) - np.asarray(lo)) / 2
    q = np.abs(P - c) - e
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)


def _box_edge_points(lo, hi, step):
    lo, hi = np.asarray(lo), np.asarray(hi)
    out = []
    for ax in range(3):
        a, b = [i for i in range(3) if i != ax]
        t = np.arange(lo[ax], hi[ax] + 1e-9, step)
        for va in (lo[a], hi[a]):
            for vb in (lo[b], hi[b]):
                P = np.zeros((t.size, 3))
                P[:, ax], P[:, a], P[:, b] = t, va, vb
                out.append(P)
    return np.vstack(out)


# ---------------------------------------------------------------- arm kinematics (ur_base_link -> pad TCP)
class Chain:
    """Fast FK of ur_base_link -> gripper_pad_tcp over the six UR10 joints of the mobile URDF."""

    def __init__(self, urdf):
        from twin.urdf_probe import joint_transform, link_world_transforms, parse_urdf
        robot = parse_urdf(str(urdf))
        by_child = robot.joints_by_child
        link, chain = 'gripper_pad_tcp', []
        while link != 'ur_base_link':
            joint = by_child[link]
            chain.append(joint)
            link = joint.parent
        joints = list(reversed(chain))
        if [j.name for j in joints if j.name in ARM] != list(ARM):
            raise ValueError('ur_base_link -> gripper_pad_tcp chain does not carry the six UR10 joints in order')
        self.limits = [(j.lower, j.upper) for j in joints if j.name in ARM]
        # Precompute: fixed transforms folded into the following joint's origin; each arm joint is
        # origin @ rotation(axis, q) (joint_transform at q=0 gives the origin).
        self._steps, pending = [], np.eye(4)
        for j in joints:
            origin = pending @ joint_transform(j, 0.0)
            if j.name in ARM:
                self._steps.append((origin, np.asarray(j.axis, float) / np.linalg.norm(j.axis)))
                pending = np.eye(4)
            else:
                pending = origin
        self._tail = pending
        world = link_world_transforms(robot)
        self.base_to_arm = np.linalg.inv(world['base_link']) @ world['ur_base_link']   # base_link -> arm

    def fk(self, q):
        T = np.eye(4)
        for (origin, axis), angle in zip(self._steps, q):
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(4)
            R[:3, :3] = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
            T = T @ origin @ R
        return T @ self._tail

    def ik(self, goal, seed, iters=300):
        q = np.array(seed, float)
        for _ in range(iters):
            e = _pose_err(self.fk(q), goal)
            if np.linalg.norm(e) < 1e-10:
                break
            J = np.empty((6, 6))
            for i in range(6):
                dq = q.copy()
                dq[i] += 1e-6
                J[:, i] = (_pose_err(self.fk(dq), goal) - e) / 1e-6
            step = np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), J.T @ e)
            n = np.linalg.norm(step)
            q = q - (step * (0.1 / n) if n > 0.1 else step)
        seed = np.asarray(seed, float)
        q = seed + (q - seed + np.pi) % (2 * np.pi) - np.pi          # the branch nearest the seed
        ok = np.linalg.norm(_pose_err(self.fk(q), goal)) < 1e-6
        ok = ok and all(lo is None or lo - 1e-9 <= v <= hi + 1e-9 for v, (lo, hi) in zip(q, self.limits))
        return q, ok


def _pose_err(M, G):
    """(position, rotation-vector) error of ``M`` to ``G``. The rotation part is the full log map: the skew
    part of dR alone is sin(angle) * axis, which VANISHES at a 180 deg error and would accept a flipped tool
    (a wrist_3 half-turn) as an exact IK solution."""
    dR = G[:3, :3] @ M[:3, :3].T
    v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2
    s, c = float(np.linalg.norm(v)), (np.trace(dR) - 1) / 2
    angle = math.atan2(s, c)
    if s > 1e-6:
        v = v * (angle / s)
    elif c < 0:                                   # ~180 deg: the axis from the symmetric part, dR + I = 2aa^T
        B = (dR + np.eye(3)) / 2
        i = int(np.argmax(np.diag(B)))
        v = angle * B[:, i] / math.sqrt(B[i, i])
    return np.concatenate([G[:3, 3] - M[:3, 3], v])


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def _quat_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def mouth_face_tcp(T_cart_arm):
    """The side-grip TCP (4x4, arm frame) on the carton's mouth face, as WallCartonPolicy builds it
    (shelfbot wall_grip: contact = centre - d*(cos a, sin a), quat = yaw(a) * grasp_quat), with ``a`` the
    wall angle whose outward face normal -(cos a, sin a) is the cart's +x mouth axis. Also returns the
    outward slot axis (arm frame) and the carton pose (arm frame) at rest."""
    T_carton = T_cart_arm @ carton_rest_pose()
    m = T_cart_arm[:3, :3] @ np.array([1., 0., 0.])                 # mouth axis, outward
    a = math.atan2(-m[1], -m[0])
    d = CARTON[1] / 2 + TCP_SURFACE_OFFSET                          # the 0.6-wide face: half the 0.4 depth
    T = np.eye(4)
    T[:3, :3] = _quat_R(*_qmul((0., 0., math.sin(a / 2), math.cos(a / 2)), GRASP_QUAT))
    T[:3, 3] = T_carton[:3, 3] + d * np.array([m[0], m[1], 0.])
    return T, np.array([m[0], m[1], 0.]) / math.hypot(m[0], m[1]), T_carton


def cart_in_arm(base_pose, cart_pose, chain, base_z):
    """4x4 cart frame (ground point) in the arm (ur_base_link) frame for map poses (x, y, yaw)."""
    def T2(p, z=0.):
        T = np.eye(4)
        T[:3, :3] = _rotz(p[2])
        T[:3, 3] = (p[0], p[1], z)
        return T
    return np.linalg.inv(T2(base_pose, base_z) @ chain.base_to_arm) @ T2(cart_pose)


def branches(chain, goal, pan):
    """Distinct in-limit IK solutions of ``goal`` from a seed grid around the arm pointing at ``pan``."""
    out = []
    for dp in (0., 0.4, -0.4):
        for sh, el in ((-1.2, 1.6), (-1.0, 1.2), (-0.8, 1.0), (-1.6, 2.0), (-2.0, -1.6), (0.4, -1.5), (-0.5, -1.5)):
            for w2 in (1.57, -1.57):
                for w3 in (0., math.pi):
                    q, ok = chain.ik(goal, (pan + dp, sh, el, -0.3, w2, w3))
                    if ok and not any(np.allclose((q - p + np.pi) % (2 * np.pi) - np.pi, 0, atol=1e-3) for p in out):
                        out.append(q)
    return out


def straight_path(chain, q0, start, axis, dist, lift, n=45):
    """IK-follow the fixed-orientation line start -> start + dist*axis + lift*z in n steps (each seeded from
    the previous). Returns (joint waypoints, distance reached along the line)."""
    qs = [np.asarray(q0, float)]
    for k in range(1, n + 1):
        G = start.copy()
        G[:3, 3] = start[:3, 3] + k / n * (dist * axis + np.array([0., 0., lift]))
        q, ok = chain.ik(G, qs[-1])
        if not ok or np.abs(q - qs[-1]).max() > 0.5:         # an IK failure or a branch jump ends the line
            return qs, (k - 1) / n * dist
        qs.append(q)
    return qs, dist


def path_metrics(chain, qs, start, segments, T_cart_arm, boxes):
    """Bow off the line (m) and the min carton-to-cage clearance (m) when the line is executed as
    ``segments`` 2-point joint moves (1 == the controller's single straight_line move)."""
    p0, p1 = start[:3, 3], chain.fk(qs[-1])[:3, 3]
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    T_tcp_carton = np.linalg.inv(start) @ (T_cart_arm @ carton_rest_pose())
    to_cart = np.linalg.inv(T_cart_arm)
    idx = [round(i * (len(qs) - 1) / segments) for i in range(segments + 1)]
    bow, clear = 0., math.inf
    for a, b in zip(idx[:-1], idx[1:]):
        for s in np.linspace(0, 1, 25):
            T = chain.fk(qs[a] + s * (qs[b] - qs[a]))
            v = T[:3, 3] - p0
            bow = max(bow, float(np.linalg.norm(v - (v @ u) * u)))
            clear = min(clear, carton_clearance(to_cart @ T @ T_tcp_carton, boxes, step=0.01))
    return bow, clear


def analyse(urdf, *, distance=0.45, segments=(1, 3, 5, 9)):
    from twin.mobile import MobileSpec
    spec = MobileSpec()
    boxes = cage_boxes(spec)
    chain = Chain(urdf)
    result = {'static': static_clearance(boxes)}
    coupled_cart = (spec.hitch_x - spec.cart_hitch_x, 0., 0.)   # cart straight behind the base on the hitch
    motions = {
        'collect_place_insert': cart_in_arm(PLACE_APPROACH, CART_AT_COLLECT, chain, spec.base_z),
        'transfer_withdraw': cart_in_arm((0., 0., 0.), coupled_cart, chain, spec.base_z),
    }
    for name, T_cart_arm in motions.items():
        tcp, axis, carton = mouth_face_tcp(T_cart_arm)
        pan = math.atan2(tcp[1, 3], tcp[0, 3])
        rows = []
        for q0 in branches(chain, tcp, pan):
            qs, reached = straight_path(chain, q0, tcp, axis, distance, LIFT)
            row = {'q_start': [round(float(v), 3) for v in q0], 'line_feasible_m': round(reached, 3)}
            if reached >= distance - 1e-9:
                for k in segments:
                    bow, clear = path_metrics(chain, qs, tcp, k, T_cart_arm, boxes)
                    row[f'{k}seg'] = {'bow_mm': round(bow * 1000, 1), 'min_cage_clearance_mm': round(clear * 1000, 1)}
            rows.append(row)
        result[name] = {'tcp_arm': [round(float(v), 4) for v in tcp[:3, 3]],
                        'slot_axis_out_arm': [round(float(v), 4) for v in axis],
                        'tcp_reach_m': round(float(np.linalg.norm(tcp[:3, 3])), 3),
                        'rest_clearance_mm': round(carton_clearance(np.linalg.inv(T_cart_arm) @ carton, boxes) * 1000, 1),
                        'line_distance_m': distance, 'lift_m': LIFT, 'branches': rows}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF, help='mobile_ur10.urdf (mobile manifest)')
    parser.add_argument('--distance', type=float, default=0.45, help='straight line length out of the slot (m)')
    parser.add_argument('--json', type=Path, help='also write the result here')
    args = parser.parse_args(argv)
    result = analyse(args.urdf, distance=args.distance)
    text = json.dumps(result, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
