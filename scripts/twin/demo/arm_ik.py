"""Tiny position-only damped-least-squares IK for the UR arm, so the demo gripper visibly
tracks the (scripted) carton -- a cosmetic arm 20 cm from the carton reads as broken.

Pure numpy + the repo's `twin.urdf_probe` FK (no Isaac). CPU-testable against the imported
mobile URDF. The arm base and the carton are both expressed in the arm-base (ur_base_link)
frame by the caller; this solves the 6 arm joints to place gripper_pad_tcp at the target.
"""
from __future__ import annotations

import numpy as np

from twin import urdf_probe as U


class ArmIK:
    def __init__(self, urdf_path, arm_joint_names, *, base_link='ur_base_link',
                 tip_link='gripper_pad_tcp', limits=None):
        self.robot = U.parse_urdf(str(urdf_path))
        self.arm = list(arm_joint_names)
        self.arm_set = set(self.arm)
        child_joint = {j.child: j for j in self.robot.joints}
        # Ordered joint chain base_link -> ... -> tip_link.
        chain = []
        link = tip_link
        for _ in range(64):
            j = child_joint.get(link)
            if j is None:
                raise ValueError(f'broken URDF chain above {link!r}')
            chain.append(j)
            link = j.parent
            if link == base_link:
                break
        else:
            raise ValueError('did not reach the arm base link')
        chain.reverse()
        self.chain = chain
        # Per-arm-joint limits (lo, hi) for clamping; default wide.
        self.limits = {}
        if limits:
            for name, (lo, hi) in zip(self.arm, limits):
                self.limits[name] = (float(lo), float(hi))

    def fk(self, q6):
        """gripper_pad_tcp 4x4 pose in the ur_base_link frame for arm config q6 (in arm order)."""
        qmap = dict(zip(self.arm, q6))
        T = np.eye(4)
        for j in self.chain:
            angle = float(qmap.get(j.name, 0.0)) if j.name in self.arm_set else 0.0
            T = T @ U.joint_transform(j, angle)
        return T

    def _clamp(self, q6):
        out = np.array(q6, float)
        for i, name in enumerate(self.arm):
            if name in self.limits:
                lo, hi = self.limits[name]
                out[i] = min(max(out[i], lo), hi)
        return out

    def solve(self, target_pos_base, q0, *, iters=60, lam=0.08, eps=1e-4, tol=2e-3):
        """DLS position IK: return arm config placing the tip at ``target_pos_base`` (3-vec in
        the arm-base frame), warm-started from q0. Position only (orientation is free)."""
        q = self._clamp(np.array(q0, float))
        target = np.asarray(target_pos_base, float)
        for _ in range(iters):
            p = self.fk(q)[:3, 3]
            err = target - p
            if np.linalg.norm(err) < tol:
                break
            J = np.zeros((3, len(self.arm)))
            for i in range(len(self.arm)):
                dq = q.copy()
                dq[i] += eps
                J[:, i] = (self.fk(dq)[:3, 3] - p) / eps
            # dq = J^T (J J^T + lam^2 I)^-1 err
            JJt = J @ J.T + (lam ** 2) * np.eye(3)
            dqv = J.T @ np.linalg.solve(JJt, err)
            step = np.clip(dqv, -0.3, 0.3)  # bound per-iteration joint step
            q = self._clamp(q + step)
        return q
