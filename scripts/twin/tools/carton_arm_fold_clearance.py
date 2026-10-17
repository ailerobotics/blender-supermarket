"""CPU carton-vs-arm clearance along the carry tuck fold (no GPU, no Isaac, no ROS).

The mission carry-fold stall is a GEOMETRY problem, not a drive-gain one (run-250: wrist_1 pins at
0.6965 while shoulder/elbow/wrist_3 track; gain-, drive-type- and carton-weld-independent): the
attached carton, swung by wrist_1's rotation toward the tuck goal, collides with the robot's OWN
``ur_upper_arm_link`` -- and cuMotion does not self-collision-check the attached carton against the
arm, so it plans that colliding tuck "collision-free" and PhysX/the drive then cannot complete it.

This tool measures that directly and offline: it interpolates the arm from the LIFT pose to the TUCK
goal and, at each step, computes the minimum clearance between the attached carton's conservative
sphere model (``world_obstacles.carton_spheres_world``, carton centre 0.20 m along tool-z of the
0.6 m side-grip) and each arm link's collision spheres (``world_obstacles.arm_spheres_world``). It
reports the clearance profile for the OLD fold (wrist_1 -> 2.1066) and a RESHAPED fold (wrist_1 -> a
smaller value, e.g. 0.9) so the reshape's margin is quantified. Reuses the integration lane's
validated sphere geometry; this module only adds the carton-vs-arm (self-collision) pairing + the
fold sweep, which that lane's ``check_configuration`` (carton-vs-world-box only) does not cover.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_host.world_obstacles import arm_spheres_world, carton_spheres_world

# Arm links the attached carton can strike as the fold completes (run-250 names the upper arm).
SELF_COLLISION_LINKS = ('ur_upper_arm_link', 'ur_forearm_link', 'ur_shoulder_link',
                        'ur_wrist_1_link', 'ur_wrist_2_link', 'ur_wrist_3_link', 'arm_mount')


def min_sphere_clearance(a_centres, a_radii, b_centres, b_radii):
    """Minimum signed clearance (m) between two sphere sets: min over pairs of
    ||ci-cj|| - ri - rj. Negative = interpenetration. Pure numpy."""
    if len(a_centres) == 0 or len(b_centres) == 0:
        return math.inf
    d = np.linalg.norm(a_centres[:, None, :] - b_centres[None, :, :], axis=2)
    gaps = d - a_radii[:, None] - b_radii[None, :]
    return float(gaps.min())


def fold_configs(lift_joints, tuck_shape, *, steps=25):
    """Linear joint interpolation from the LIFT pose to the TUCK goal (joint0 held at the lift pan,
    joints 1-5 -> tuck_shape), as a list of (fraction, wrist_1, config-dict)."""
    lift = np.asarray(lift_joints, float)
    goal = np.array([lift[0], *[float(v) for v in tuck_shape]], float)
    out = []
    for i in range(steps + 1):
        f = i / steps
        q = lift + f * (goal - lift)
        out.append((f, float(q[ARM_JOINTS.index('ur_wrist_1_joint')]),
                    dict(zip(ARM_JOINTS, q.tolist()))))
    return out


@dataclass
class FoldClearance:
    label: str
    wrist1_goal: float
    min_clearance_m: float = math.inf        # worst over the whole fold
    min_at_wrist1: float = float('nan')
    min_link: str = ''
    profile: list = field(default_factory=list)  # [(fraction, wrist1, worst_link, gap_m)]

    @property
    def collides(self):
        return self.min_clearance_m < 0.0


def sweep_fold(urdf_path, xrdf_path, lift_joints, tuck_shape, *, label, dims=(0.6, 0.4, 0.16),
               offset_tcp=(0.0, 0.0, 0.20), links=SELF_COLLISION_LINKS, steps=25):
    """Carton-vs-arm-link clearance along one fold. Returns a FoldClearance."""
    wrist1_goal = float(tuck_shape[ARM_JOINTS.index('ur_wrist_1_joint') - 1])  # shape is joints 1-5
    fc = FoldClearance(label=label, wrist1_goal=wrist1_goal)
    for frac, w1, cfg in fold_configs(lift_joints, tuck_shape, steps=steps):
        ac, ar, owners = arm_spheres_world(urdf_path, xrdf_path, cfg)
        cc, cr = carton_spheres_world(urdf_path, cfg, dims=list(dims), offset_tcp=list(offset_tcp))
        owners = np.asarray(owners)
        worst_gap, worst_link = math.inf, ''
        for link in links:
            mask = owners == link
            if not mask.any():
                continue
            g = min_sphere_clearance(cc, cr, ac[mask], ar[mask])
            if g < worst_gap:
                worst_gap, worst_link = g, link
        fc.profile.append((round(frac, 3), round(w1, 4), worst_link, round(worst_gap, 4)))
        if worst_gap < fc.min_clearance_m:
            fc.min_clearance_m, fc.min_at_wrist1, fc.min_link = worst_gap, w1, worst_link
    return fc


def report(folds):
    lines = ['carton-vs-arm clearance along the tuck fold (CPU sphere model; negative = collision)', '']
    for fc in folds:
        verdict = 'COLLIDES' if fc.collides else 'clears'
        lines.append('%-18s wrist1_goal=%+.4f  min_clearance=%+.1f mm %s (worst link %s at wrist1=%+.3f)'
                     % (fc.label, fc.wrist1_goal, fc.min_clearance_m * 1000, verdict, fc.min_link,
                        fc.min_at_wrist1))
    lines.append('')
    lines.append('fraction  wrist1     ' + '  '.join('%-14s' % fc.label for fc in folds))
    n = max(len(fc.profile) for fc in folds)
    for i in range(n):
        frac = folds[0].profile[i][0]
        w1 = folds[0].profile[i][1]
        cells = []
        for fc in folds:
            _, _, link, gap = fc.profile[i]
            cells.append('%+7.1fmm %-6s' % (gap * 1000, link.replace('ur_', '').replace('_link', '')))
        lines.append('%-9.2f %+8.3f  %s' % (frac, w1, '  '.join(cells)))
    return '\n'.join(lines)
