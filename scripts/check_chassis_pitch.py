#!/usr/bin/env python3
"""CPU proof + GPU-trace grader for the chassis pitch see-saw and its compliant-suspension fix.

sim-chassis-pitch-stability-v1. Two jobs, no Isaac/GPU:

  1. DESIGN PROOF (default): from a mobile manifest's ``spec`` (caster_lift_m, length, suspension
     fields) and the robot URDF, compute
       * the UNSPRUNG free pitch travel +-atan(caster_lift_m / caster_spacing) -- the see-saw amplitude
         that produced the coldwin33-37 CARTON_TILT +2.06 (the two end stops at -4.1 / +4.9 deg), and
       * the worst-case ARM chassis moment about the drive axle (the UR10's own mass out at reach, from
         FK over the arm workspace + the per-body masses), and
       * for a suspension spec, the SPRUNG chassis pitch at that worst-case moment (criterion (a):
         <= 0.5 deg) and the drive-wheel traction share + budget margin vs the stuck / caster_lift 0.03
         configs (criterion (b): strictly dominate the stuck rig).

  2. GPU-TRACE GRADE (``--base-att-log run.console.log``): parse the run's BASE_ATT pitch series,
     report the per-arm-phase excursion, and GRADE the collect stage: PASS iff the pitch excursion is
     <= ``--bound`` (default 0.5 deg) across the whole collect AND there is no CARTON_TILT trip.

Pure (twin.traction + twin.urdf_probe + the physics_setup body-mass table); exits 1 on a FAIL grade."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from twin import urdf_probe as up
from twin.traction import (free_pitch_travel_deg, caster_spacing_m, CasterSuspension,
                           sprung_chassis_pitch_deg, suspension_drive_wheel_share,
                           static_traction_budget)

ARM_JOINTS = ['ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
              'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']
G = 9.81


def worst_case_arm_chassis_moment(urdf_path, body_masses, payload_kg=0.45, axle_x=0.0):
    """Max |pitch moment| (N.m) the arm + payload apply to the chassis about the drive axle (base x=0),
    swept over the shoulder_lift/elbow workspace. Pure FK; the UR10's own mass dominates, not the carton."""
    model = up.parse_urdf(str(urdf_path))
    above = {k: v for k, v in body_masses.items()
             if k.startswith('ur_') or k.startswith('surface_plate') or k in
             ('arm_mount', 'big_cube', 'cube_without_surface_plates', 'suction_cups',
              'gripper_tcp', 'gripper_pad_tcp')}
    coms = {}
    for ln, lk in model.links.items():
        m = above.get(ln)
        if m:
            c = lk.inertial.origin_xyz if (lk and lk.inertial) else (0.0, 0.0, 0.0)
            coms[ln] = (float(m), np.array([c[0], c[1], c[2], 1.0]))
    payload_link = 'gripper_pad_tcp'

    def moment(q6):
        world = up.link_world_transforms(model, dict(zip(ARM_JOINTS, [float(x) for x in q6])))
        total = 0.0
        for ln, (mkg, comh) in coms.items():
            if ln in world:
                total += mkg * G * ((world[ln] @ comh)[0] - axle_x)
        if payload_kg > 0 and payload_link in world:
            total += payload_kg * G * (world[payload_link][0, 3] - axle_x)
        return total

    best = 0.0
    for sl in np.linspace(-3.14, 0.6, 72):
        for el in np.linspace(-2.6, 2.6, 72):
            best = max(best, abs(moment([0.0, sl, el, 0.0, 0.0, 0.0])))
    return best


def _spec_from_manifest(path):
    return (json.loads(Path(path).read_text()).get('spec') or {})


def design_proof(manifest_path, urdf_path, bound_deg=0.5):
    spec = _spec_from_manifest(manifest_path)
    length = float(spec.get('length', 1.0))
    lift = float(spec.get('caster_lift_m', 0.0))
    d = caster_spacing_m(length)
    print('=== chassis pitch DESIGN PROOF ===')
    print(f'caster_spacing d = {d:.3f} m (length/2 - 0.12)')
    print(f'UNSPRUNG free pitch travel @caster_lift_m={lift}: +-{free_pitch_travel_deg(lift, d):.3f} deg '
          f'(the see-saw amplitude; coldwin37 end stops measured -4.10 / +4.89)')
    from twin.restock.physics_setup import _ISAAC_BODY_MASSES
    m_max = worst_case_arm_chassis_moment(urdf_path, _ISAAC_BODY_MASSES)
    print(f'worst-case ARM chassis moment about the drive axle = {m_max:.1f} N.m '
          f'(UR10 own mass + 0.45 kg carton, swept over the arm workspace)')
    if spec.get('caster_suspension'):
        susp = CasterSuspension(spring_n_per_m=float(spec['caster_spring_n_per_m']),
                                preload_n=float(spec.get('caster_preload_n', 0.0)),
                                damping_n_s_per_m=float(spec.get('caster_damping_n_s_per_m', 0.0)),
                                travel_m=float(spec['caster_travel_m']))
        pitch = abs(sprung_chassis_pitch_deg(m_max, susp, d))
        share = suspension_drive_wheel_share(susp.preload_n)
        stuck = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.005, drive_max_force_nm=80.0)
        lift03 = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.03, drive_max_force_nm=200.0)
        rec = static_traction_budget(wheel_friction=1.0, caster_lift_m=0.0, drive_max_force_nm=200.0,
                                     suspension_preload_n=susp.preload_n)
        print(f'\nCOMPLIANT SUSPENSION spring={susp.spring_n_per_m:.0f} N/m preload={susp.preload_n:.0f} N '
              f'damping={susp.damping_n_s_per_m:.0f} travel={susp.travel_m} m')
        print(f'  (a) sprung chassis pitch @worst-case moment = {pitch:.3f} deg '
              f'({"PASS" if pitch <= bound_deg else "FAIL"} <= {bound_deg} deg)')
        print(f'  (b) drive-wheel share = {share:.3f}  margin={rec["margin"]}  bind={rec["binding_cap"]}')
        print(f'      stuck (pre-PR207) share={stuck["drive_wheel_share"]} margin={stuck["margin"]}')
        print(f'      caster_lift 0.03     share={lift03["drive_wheel_share"]} margin={lift03["margin"]}')
        dominates = rec['margin'] > stuck['margin'] and share > stuck['drive_wheel_share']
        print(f'      suspension strictly DOMINATES stuck: {dominates}; '
              f'margin vs caster_lift 0.03 = {rec["margin"] - lift03["margin"]:+.2f}')
        ok = pitch <= bound_deg and dominates
        print(f'\nDESIGN VERDICT: {"PASS" if ok else "FAIL"}')
        return 0 if ok else 1
    print('\n(no compliant suspension in this spec -- unsprung rig; see the free pitch travel above)')
    return 0


_BATT = re.compile(r'BASE_ATT .*pitch_deg=(?P<p>[-0-9.]+).*action=(?P<a>[a-z_]+) now=(?P<t>[0-9.]+)')
_SEG = re.compile(r'BASE_ATT_SEGMENT action=(?P<a>\S+) phase=(?P<ph>\S+) .*excursion=(?P<ex>[-0-9.]+)')


def grade_log(log_path, bound_deg=0.5, action='collect'):
    text = Path(log_path).read_text(errors='replace')
    pitches = [(float(m['t']), float(m['p'])) for m in (_BATT.search(l) for l in text.splitlines())
               if m and m['a'] == action]
    trip = 'CARTON_TILT' in text and re.search(r"MANIP_HOST_FAILED op=%s" % action, text)
    print(f'=== BASE_ATT GPU-trace GRADE ({action}) ===')
    if not pitches:
        print('no BASE_ATT pitch samples for this action'); return 1
    ps = [p for _, p in pitches]
    excursion = max(ps) - min(ps)
    print(f'samples={len(ps)} pitch[min={min(ps):.3f} max={max(ps):.3f} excursion={excursion:.3f} deg]')
    for m in (_SEG.search(l) for l in text.splitlines()):
        if m and m['a'] == action:
            print(f"  segment phase={m['ph']} excursion={m['ex']} deg")
    passed = excursion <= bound_deg and not trip
    print(f'CARTON_TILT trip present: {bool(trip)}')
    print(f'GRADE: {"PASS" if passed else "FAIL"} (excursion {excursion:.3f} <= {bound_deg} '
          f'and no CARTON_TILT trip)')
    return 0 if passed else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', help='mobile-manifest.json (reads .spec)')
    ap.add_argument('--urdf', help='resolved mobile URDF (mobile-import.urdf) for the FK moment')
    ap.add_argument('--base-att-log', help='a run console log to GRADE its BASE_ATT series')
    ap.add_argument('--bound', type=float, default=0.5, help='pitch excursion bound (deg)')
    ap.add_argument('--action', default='collect')
    args = ap.parse_args(argv)
    rc = 0
    if args.manifest and args.urdf:
        rc |= design_proof(args.manifest, args.urdf, args.bound)
    if args.base_att_log:
        rc |= grade_log(args.base_att_log, args.bound, args.action)
    if not (args.manifest or args.base_att_log):
        ap.error('pass --manifest/--urdf for the design proof and/or --base-att-log to grade a run')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
