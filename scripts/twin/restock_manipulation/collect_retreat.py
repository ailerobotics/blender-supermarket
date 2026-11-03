"""Vertical lift-off for the collect deck-place EMPTY retreat (pure / CPU-testable; no Isaac, ROS, GPU).

sim-collect-retreat-abort-v2. Diagnosis of the intermittent collect empty-retreat abort
(physcon1-fail1 / rtfcollect2-fail1): after the carton is set down on the rollwagen shelf and
RELEASED, the single collision-OFF straight-line retreat drives the arm from the folded place config
straight to the clear ``collect_place_preplace_shape``. That first increment is almost entirely a
HORIZONTAL retract (gripper TCP x ~1.07 -> ~0.71 m, z only +0.07 m), which drags the still-ENABLED
wrist links (ur_wrist_2/ur_wrist_3 -- the carton's seat links, NOT on the deferred tool-side collider
set) sideways against the just-RE-ENABLED carton (SIM_ATTACH_FALLBACK re-enables the carton collider at
release) and the rollwagen cage, whose arm-vs-cage contacts are NOT collision-filtered. The host trace
is unambiguous and rules out the alternatives:

* NOT static drive authority: at the stall config the gravity holding torque is ~103 N*m (shoulder_lift)
  / ~75 N*m (elbow), far under the release-phase effort cap (400 N*m), and the retreat trajectory is
  already slow (rest-to-rest smoothstep, peak accel ~0.03 rad/s^2), so the inertial demand is ~1 N*m.
  The drive nonetheless SATURATES (eff=400) with shoulder_lift pinned ~0.6 rad and elbow ~0.5 rad short
  for the whole 30 s settle window -- only a ~300 N*m EXTERNAL reaction (a contact) can hold it there.
* NOT a wrist self-collision: the stall wrists sit inside the clear carry region (wrist_1<=0.65,
  wrist_2>=0), so PR 289's ur_wrist_2-vs-forearm stop is not the cause.
* The distal wrists run at velocity SATURATION (~pi rad/s) about their already-reached targets while the
  proximal joints are pinned -- the contact-pin fingerprint (``trajectory_host.classify_contact_pin``).

Fix: raise the gripper straight UP first (separating the enabled wrist links from the carton/shelf/cage),
THEN fold to the clear preplace shape. This module solves that lift-off config: a damped-least-squares IK
that moves the TCP by ``+dz`` along world +Z while PRESERVING the seed's tool orientation, seeded from the
measured release config so it stays on the current (self-clear) IK branch. It leans on the elbow +
wrist_1 (the shoulder barely moves), so it executes even while the shoulder is pinned against a downward
contact -- lifting releases that contact. Returns a joint map in ARM order; raises ValueError if no
in-reach lift-off converges (the caller then fails honestly rather than retreating blind).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

from twin.restock_manipulation.cad_tool import ARM_JOINTS

WORLD_UP = np.array([0.0, 0.0, 1.0])
DEFAULT_LIFTOFF_DZ = 0.15          # m: straight-up TCP rise before the fold (clears the 0.18 m carton top
                                   # + margin from the mid-height side grip; validated by check_collect_retreat)


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Axis*angle (rotation vector) of a rotation matrix, numerically safe near 0 and pi."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if math.pi - ang < 1e-6:                       # near pi: use the symmetric part
        w, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = V[:, int(np.argmax(w))]
        return ang * axis / (np.linalg.norm(axis) or 1.0)
    return ang / (2.0 * math.sin(ang)) * np.array([R[2, 1] - R[1, 2],
                                                   R[0, 2] - R[2, 0],
                                                   R[1, 0] - R[0, 1]])


def tcp_pose(robot, config: Dict[str, float], tcp_link: str, root: str = 'ur_base_link'):
    """(position 3, rotation 3x3) of ``tcp_link`` in ``root`` at ``config`` (FK via urdf_probe)."""
    from twin.urdf_probe import link_world_transforms
    T = link_world_transforms(robot, {j: float(v) for j, v in config.items()}, root=root)
    if tcp_link not in T:
        raise ValueError(f'tcp link {tcp_link} not in FK model')
    M = T[tcp_link]
    return M[:3, 3].copy(), M[:3, :3].copy()


def liftoff_goal(robot, seed_config, *, dz: float = DEFAULT_LIFTOFF_DZ,
                 tcp_link: str = 'gripper_pad_tcp', root: str = 'ur_base_link',
                 limits: Optional[Dict[str, Sequence[float]]] = None,
                 w_ori: float = 0.3, max_iter: int = 200, pos_tol: float = 2e-3,
                 ori_tol: float = 0.02) -> Dict[str, float]:
    """A joint config that raises the TCP straight up by ``dz`` (world +Z) from ``seed_config`` while
    preserving the seed's tool orientation. Full-DOF damped-least-squares, seeded from the measured
    release config so the solution stays on the current self-clear IK branch. ``limits`` optionally
    clamps joints (e.g. the clear carry wrist region). Raises ValueError if no in-tolerance lift-off is
    found (unreachable: the caller fails honestly)."""
    if dz <= 0:
        raise ValueError('lift-off dz must be positive')
    seed = {j: float(v) for j, v in (seed_config.items() if hasattr(seed_config, 'items')
                                     else zip(ARM_JOINTS, seed_config))}
    if any(j not in seed for j in ARM_JOINTS):
        raise ValueError('seed config must carry all six arm joints')
    p0, R0 = tcp_pose(robot, seed, tcp_link, root)
    target = p0 + dz * WORLD_UP

    def residual(m):
        p, R = tcp_pose(robot, m, tcp_link, root)
        return np.concatenate([p - target, w_ori * _rotvec(R @ R0.T)])

    def clamp(m):
        if limits:
            for j, (lo, hi) in limits.items():
                if j in m:
                    m[j] = min(max(m[j], lo), hi)
        return m

    m = clamp(dict(seed))
    fd = 1e-5
    for _ in range(int(max_iter)):
        r0 = residual(m)
        p, R = tcp_pose(robot, m, tcp_link, root)
        if np.linalg.norm(p - target) <= pos_tol and np.linalg.norm(_rotvec(R @ R0.T)) <= ori_tol:
            break
        J = np.zeros((6, 6))
        for k, j in enumerate(ARM_JOINTS):
            m2 = dict(m)
            m2[j] += fd
            J[:, k] = (residual(m2) - r0) / fd
        dq = -np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ r0)
        mx = float(np.max(np.abs(dq))) if dq.size else 0.0
        if mx > 0.2:                                  # cap per-iteration step for stability
            dq *= 0.2 / mx
        for k, j in enumerate(ARM_JOINTS):
            m[j] += float(dq[k])
        clamp(m)
    p, R = tcp_pose(robot, m, tcp_link, root)
    if np.linalg.norm(p - target) > pos_tol or np.linalg.norm(_rotvec(R @ R0.T)) > ori_tol:
        raise ValueError(
            'collect retreat lift-off IK did not converge: rose %.3f m (want %.3f), ori err %.3f rad'
            % (float(p[2] - p0[2]), dz, float(np.linalg.norm(_rotvec(R @ R0.T)))))
    return {j: float(m[j]) for j in ARM_JOINTS}
