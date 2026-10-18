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


def sweep_arm_self_collision(urdf_path, xrdf_path, lift_joints, tuck_shape, *,
                             distal=('ur_wrist_2_link', 'ur_wrist_3_link', 'gripper_pad_tcp',
                                     'gripper_tcp', 'RestockPanelGripper', 'ur_tool0'),
                             proximal=('ur_forearm_link', 'ur_upper_arm_link'),
                             w1_lo=0.65, w1_hi=0.95, steps=30):
    """inbox 020 (2): arm-vs-arm self-collision sweep. Hold the arm at the TUCK configuration (joint0 at
    the lift pan, joints 1-5 at ``tuck_shape``) and sweep ONLY wrist_1 over [w1_lo, w1_hi]; at each
    angle compute, from the PLANNER's own collision-sphere model (``arm_spheres_world`` = the XRDF
    spheres), the min clearance between each DISTAL link (wrist_2/3 + tool) and each PROXIMAL link
    (forearm/upper_arm). Returns (first_contact dict | None, profile list). first_contact names the
    pair and the wrist_1 angle where a distal-vs-proximal pair first interpenetrates (<=0).

    Because the sphere set IS the planner's model, a negative gap here means the geometry IS
    representable to the planner -- so the planner would catch the pair UNLESS it is on the XRDF
    self-collision ignore list (the known cuMotion attached-object / ignore-list gap)."""
    base = np.array([float(lift_joints[0]), *[float(v) for v in tuck_shape]], float)
    w1_idx = ARM_JOINTS.index('ur_wrist_1_joint')
    profile, first = [], None
    for i in range(steps + 1):
        w1 = w1_lo + (w1_hi - w1_lo) * i / steps
        q = base.copy(); q[w1_idx] = w1
        cfg = dict(zip(ARM_JOINTS, q.tolist()))
        ac, ar, owners = arm_spheres_world(urdf_path, xrdf_path, cfg)
        owners = np.asarray(owners)
        worst_gap, worst_pair = math.inf, ''
        for dl in distal:
            dmask = owners == dl
            if not dmask.any():
                continue
            for pl in proximal:
                pmask = owners == pl
                if not pmask.any():
                    continue
                g = min_sphere_clearance(ac[dmask], ar[dmask], ac[pmask], ar[pmask])
                if g < worst_gap:
                    worst_gap, worst_pair = g, '%s<->%s' % (dl, pl)
        profile.append((round(w1, 4), worst_pair, round(worst_gap, 4)))
        if first is None and worst_gap <= 0.0:
            first = dict(wrist1=round(w1, 4), pair=worst_pair, gap_mm=round(worst_gap * 1000, 1))
    present = sorted({o for o in {*distal, *proximal} if (np.asarray(owners) == o).any()})
    return first, profile, present


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='CPU carton-vs-arm clearance along the carry tuck fold.')
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--xrdf', required=True)
    ap.add_argument('--lift', required=True, help='lift pose: 6 arm joints (ur_ order), comma-separated')
    ap.add_argument('--tuck-shape', action='append', default=[], metavar='LABEL=j1,..,j5',
                    help='a tuck shape (joints 1-5) to sweep, e.g. old=-2.3,1.0,2.1066,0.8,-1.2545; repeatable')
    ap.add_argument('--carton-dims', default='0.6,0.4,0.16')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--self-collision', action='store_true',
                    help='inbox 020 (2): arm-vs-arm (wrist_2/3+tool vs forearm/upper_arm) sweep over '
                         'wrist_1 in [0.65,0.95] at the (first) tuck shape, naming the pair + first-contact angle')
    args = ap.parse_args(argv)
    lift = [float(v) for v in args.lift.split(',')]
    dims = [float(v) for v in args.carton_dims.split(',')]
    if args.self_collision:
        label, _, vals = (args.tuck_shape[0] if args.tuck_shape else 'tuck=-2.3,1.0,0.9,0.8,-1.2545').partition('=')
        shape = [float(v) for v in vals.split(',')]
        first, profile, present = sweep_arm_self_collision(args.urdf, args.xrdf, lift, shape)
        print('arm-vs-arm self-collision sweep (planner XRDF sphere model), wrist_1 in [0.65,0.95]')
        print('distal links present:', [p for p in present if 'forearm' not in p and 'upper_arm' not in p])
        print('proximal links present:', [p for p in present if 'forearm' in p or 'upper_arm' in p])
        print('FIRST CONTACT:', first if first else 'NONE (no distal-vs-proximal interpenetration in range)')
        print('\nwrist_1    worst_pair                                   gap_mm')
        for w1, pair, gap in profile:
            print('%+8.4f  %-44s %+8.1f' % (w1, pair or '-', gap * 1000))
        return 0 if first is None else 1
    folds = []
    for item in args.tuck_shape:
        label, _, vals = item.partition('=')
        shape = [float(v) for v in vals.split(',')]
        folds.append(sweep_fold(args.urdf, args.xrdf, lift, shape, label=label or 'fold',
                                dims=dims, steps=args.steps))
    print(report(folds))
    return 0 if all(not f.collides for f in folds) else 1


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


if __name__ == '__main__':
    raise SystemExit(_cli())
