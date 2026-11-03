#!/usr/bin/env python3
"""CPU proof (no Isaac/ROS/GPU) for sim-collect-retreat-abort-v2: the collect deck-place EMPTY retreat
lift-off-then-fold. Reproduces the recorded abort condition and shows the fix clears it.

The recorded abort (physcon1-fail1 / rtfcollect2-fail1): after the carton is set down on the rollwagen
and RELEASED, the single collision-OFF straight-line retreat (place fold -> collect_place_preplace_shape)
pins -- shoulder_lift ~0.6 rad / elbow ~0.5 rad short for the whole settle window while the distal wrists
run at velocity saturation -- and the settle gate aborts (moveit_err=-4). The retreat's first increment is
an almost pure HORIZONTAL retract that drags the still-ENABLED wrist/forearm links along the re-enabled
carton / shelf / cage at the folded place height.

This bench establishes, offline:
  * DETECTION -- the recorded terminal (per_joint_err, per_joint_vel) trips the settle gate and
    classify_contact_pin NAMES the proximal-pin + wrist-saturation contact (not the generic oscillation);
  * NOT DRIVE AUTHORITY -- the stall-pose gravity holding torques are far under the release-phase effort
    cap and the retreat is already slow, so the saturated-yet-pinned drive implies a ~300 N*m contact;
  * THE FIX -- the lift-off raises the TCP straight up (+dz, orientation preserved), stays in the clear
    carry wrist region, and raises the whole wrist/forearm cluster VERTICALLY by ~dz BEFORE any horizontal
    retract (the separation the direct retreat never provides). Writes a before/after plot + JSON.

    PYTHONPATH=scripts python3 scripts/check_collect_retreat.py [--urdf U] [--xrdf X] [--out DIR]

Exit 0 always (a measurement, not a gate). Kinematic + static-torque accounting, not a physics run.
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

from twin import urdf_probe as U
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock_manipulation import collect_retreat as crt
from twin.restock.trajectory_host import classify_contact_pin, settle_decision

FIX = ROOT / 'twin' / 'tests' / 'fixtures' / 'collect_stow_arm.urdf'
DS = Path.home() / 'datasets' / 'twin-robot-in-store-boot-v1' / 'host-cad'
DEFAULT_URDF = DS / 'mobile.urdf'
DEFAULT_XRDF = DS / 'mobile.xrdf'

PREPLACE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)
STALL = (-0.0127, -1.3112, 1.5111, -0.1039, 1.6648, -0.0001)
WRIST_FOREARM = ('ur_forearm_link', 'ur_wrist_1_link', 'ur_wrist_2_link', 'ur_wrist_3_link')
# physcon1-fail1 recorded terminal settle sample (TRAJ_HOST_SETTLE settle=29.11).
REC_ERR = np.array([-0.1058, 0.6082, -0.5171, -0.0028, 0.0006, 0.0014])
REC_VEL = np.array([-0.3736, 0.5015, -1.2406, 3.15, -1.4624, -3.1408])


def _cfg(q):
    return dict(zip(ARM_JOINTS, [float(v) for v in q]))


def _cluster_z(robot, q):
    T = U.link_world_transforms(robot, _cfg(q), root='ur_base_link')
    return float(np.mean([T[l][2, 3] for l in WRIST_FOREARM]))


def _gravity_torques(robot, q):
    g = np.array([0.0, 0.0, -9.81])
    T = U.link_world_transforms(robot, _cfg(q), root='ur_base_link')
    com, mass = {}, {}
    for name, L in robot.links.items():
        if L.inertial and L.inertial.mass and name in T:
            com[name] = T[name][:3, :3] @ np.array(L.inertial.origin_xyz, float) + T[name][:3, 3]
            mass[name] = L.inertial.mass
    children = {}
    for j in robot.joints:
        children.setdefault(j.parent, []).append(j)

    def subtree(link):
        out, stack = set(), [link]
        while stack:
            x = stack.pop(); out.add(x)
            for j in children.get(x, []):
                stack.append(j.child)
        return out
    jb = {j.name: j for j in robot.joints}
    tau = {}
    for jn in ARM_JOINTS:
        j = jb[jn]
        Tc = T.get(j.child)
        if Tc is None:
            continue
        pj, axis = Tc[:3, 3], Tc[:3, :3] @ np.array(j.axis, float)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        tau[jn] = sum(float(np.dot(axis, np.cross(com[n] - pj, mass[n] * g)))
                      for n in subtree(j.child) if n in com)
    return tau


def run(urdf, xrdf, out_dir, dz=0.15):
    robot = U.parse_urdf(str(urdf))
    res = {'urdf': str(urdf), 'dz': dz}

    # 1. detection -----------------------------------------------------------------------------------
    samples = [(float(i), float(np.linalg.norm(REC_ERR)), float(np.linalg.norm(REC_VEL))) for i in range(8)]
    verdict, _ = settle_decision(samples, pos_tol=0.01, vel_tol=0.05, final=True)
    pin = classify_contact_pin(REC_ERR, REC_VEL, 0.01)
    res['detection'] = {'settle_verdict': verdict, 'settles': verdict == 'settled',
                        'contact_pin_named': pin is not None, 'reason': pin}

    # 2. torque accounting ---------------------------------------------------------------------------
    tau = _gravity_torques(robot, STALL)
    res['torque'] = {'release_effort_cap_Nm': 400.0,
                     'gravity_shoulder_lift_Nm': round(tau['ur_shoulder_lift_joint'], 1),
                     'gravity_elbow_Nm': round(tau['ur_elbow_joint'], 1),
                     'drive_authority_is_the_limit': abs(tau['ur_shoulder_lift_joint']) >= 400.0}

    # 3. lift-off kinematics + the before/after cluster-z profile ------------------------------------
    goal = crt.liftoff_goal(robot, _cfg(STALL), dz=dz,
                            limits={'ur_wrist_1_joint': (-2 * math.pi, 0.65),
                                    'ur_wrist_2_joint': (0.0, 2 * math.pi)})
    liftoff_q = tuple(goal[j] for j in ARM_JOINTS)
    z_seed, z_lift, z_pre = (_cluster_z(robot, STALL), _cluster_z(robot, liftoff_q),
                             _cluster_z(robot, PREPLACE))
    # direct retreat cluster-z along seed->preplace; lift-off path seed->liftoff->preplace
    fr = np.linspace(0, 1, 41)
    a, b = np.array(STALL), np.array(PREPLACE)
    direct_z = [_cluster_z(robot, a + (b - a) * f) for f in fr]
    seg0 = [_cluster_z(robot, a + (np.array(liftoff_q) - a) * f) for f in fr]
    seg1 = [_cluster_z(robot, np.array(liftoff_q) + (b - np.array(liftoff_q)) * f) for f in fr]
    res['liftoff'] = {
        'cluster_z_seed': round(z_seed, 4), 'cluster_z_liftoff': round(z_lift, 4),
        'cluster_z_preplace': round(z_pre, 4),
        'liftoff_vertical_rise_m': round(z_lift - z_seed, 4),
        'direct_total_rise_m': round(z_pre - z_seed, 4),
        'liftoff_rise_vs_direct_total': round((z_lift - z_seed) / (abs(z_pre - z_seed) or 1e-9), 1),
        'wrist1': round(liftoff_q[3], 4), 'wrist2': round(liftoff_q[4], 4),
        'wrist_in_clear_region': liftoff_q[3] <= 0.65 + 1e-6 and liftoff_q[4] >= -1e-6}

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(fr, direct_z, 'r-', lw=2, label='direct retreat (seed->preplace): barely rises')
        ax.plot(fr * 0.5, seg0, 'g-', lw=2, label='lift-off then fold: lift-off leg (+dz up)')
        ax.plot(0.5 + fr * 0.5, seg1, 'g--', lw=2, label='lift-off then fold: fold leg (free space)')
        ax.axhline(z_seed, color='0.6', ls=':', lw=1)
        ax.set_xlabel('path fraction'); ax.set_ylabel('wrist/forearm cluster mean z (ur_base_link, m)')
        ax.set_title('Collect empty-retreat: direct (horizontal-first) vs lift-off-first separation')
        ax.legend(loc='best', fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out_dir / 'retreat_cluster_z.png', dpi=110)
        plt.close(fig)
        res['plot'] = str(out_dir / 'retreat_cluster_z.png')
    except Exception as exc:                     # noqa: BLE001 -- plot is advisory
        res['plot_error'] = str(exc)

    (out_dir / 'check_collect_retreat.json').write_text(json.dumps(res, indent=2))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--urdf', default=str(DEFAULT_URDF if DEFAULT_URDF.is_file() else FIX))
    ap.add_argument('--xrdf', default=str(DEFAULT_XRDF))
    ap.add_argument('--out', default=str(Path.home() / 'firstmate' / 'data' / 'sim-collect-retreat-abort-v2'))
    ap.add_argument('--dz', type=float, default=0.15)
    args = ap.parse_args(argv)
    res = run(Path(args.urdf), Path(args.xrdf), Path(args.out), dz=args.dz)
    d, t, l = res['detection'], res['torque'], res['liftoff']
    print('COLLECT RETREAT LIFT-OFF -- CPU proof  (urdf=%s)' % Path(res['urdf']).name)
    print('  [reproduce] recorded terminal: settle verdict=%s (settles=%s); contact pin named=%s'
          % (d['settle_verdict'], d['settles'], d['contact_pin_named']))
    if d['reason']:
        print('      %s' % d['reason'])
    print('  [not drive authority] gravity shoulder_lift=%.1f Nm elbow=%.1f Nm  vs release cap 400 Nm '
          '(authority is the limit=%s)'
          % (t['gravity_shoulder_lift_Nm'], t['gravity_elbow_Nm'], t['drive_authority_is_the_limit']))
    print('  [fix] lift-off raises wrist/forearm cluster +%.3f m vs the direct retreat TOTAL +%.3f m '
          '(%.1fx); wrist in clear region=%s'
          % (l['liftoff_vertical_rise_m'], l['direct_total_rise_m'], l['liftoff_rise_vs_direct_total'],
             l['wrist_in_clear_region']))
    print('  wrote %s' % res.get('plot', res.get('plot_error', '(no plot)')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
