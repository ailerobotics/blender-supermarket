"""sim-preplace-sweep-tilt-v1: CPU characterisation + replay of the collect preplace carried-carton
swing (coldwin50/51/52), and the drive-profile fix selection.

The replay runs on the repo UR10 fixture URDF + the committed commanded-trajectory fixture (the runs'
own TRAJ_HOST_PLAN), so it needs no dataset. It asserts the four ruled-out candidates and the
drive-profile lever the verdict rests on. The exact executed 3-8 deg is NOT reproducible on CPU (the
per-tick executed joint / carton attitude is not logged -- the honest gap coldwin53 + the new
CARTON_TICK log close); the test asserts what IS: the commanded path is level, the droop is flat and
small, the swing is anti-correlated with speed, the reduced per-joint drive tracks the slow sweep for
BOTH profiles, and the stiff profile is the harder/higher-authority one that the fix routes to SOFT.
"""
import os

import numpy as np
import pytest

from twin.urdf_probe import parse_urdf
from twin.restock_manipulation import carton_level as cl
import twin.tools.preplace_drive_bench as b

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'preplace_sweep_commanded.json')
URDF = os.path.join(os.path.dirname(__file__), 'fixtures', 'collect_stow_arm.urdf')


@pytest.fixture(scope='module')
def report():
    return b.run(URDF, FIX)


def _row(report, run):
    return next(r for r in report['rows'] if r['run'] == run)


def test_commanded_path_is_level_not_the_cause(report):
    # Every run's COMMANDED preplace path is ~1.7 deg (the integration worker's static-FK figure),
    # far under the executed 3-8 -- so the commanded path is not the cause.
    for r in report['rows']:
        assert 1.3 <= r['commanded_tilt_deg'] <= 2.1, r


def test_static_sag_ruled_out_droop_flat_and_small(report):
    # The quasi-static droop (K=2850) per waypoint is small (<2 deg) and nowhere near the executed
    # peak; the model error (executed - droop) is a large flat residual -> NOT a static wrist-config sag.
    for r in report['rows']:
        assert r['droop_max_deg'] < 2.0, r
        assert r['model_error_deg'] >= 2.0, r         # executed is >=2 deg OVER the static droop
        # droop does not track wrist_2: its spread across the sweep is small vs the executed gap
        droop_vals = [d[1] for d in r['droop_vs_w2']]
        assert (max(droop_vals) - min(droop_vals)) < r['model_error_deg'], r


def test_trajectory_speed_ruled_out_anticorrelation(report):
    cw50, cw52 = _row(report, 'coldwin50'), _row(report, 'coldwin52')
    # cw52's commanded peak joint speed is <= cw50's, yet cw52 tilted MORE -> anti-correlated with
    # speed, so the swing is not driven by the planned trajectory speed.
    assert cw52['peak_cmd_v_rad_s'] <= cw50['peak_cmd_v_rad_s'] + 1e-6
    assert cw52['executed_peak_deg'] > cw50['executed_peak_deg']


def test_reduced_per_joint_model_tracks_both_profiles(report):
    # A stable reduced per-joint PD+saturation model tracks the slow commanded sweep to <0.1 deg for
    # BOTH the soft and the stiff profile -> the swing is NOT a linear position-control lag; it is the
    # multi-body / finite-solver ringing the reduced model cannot see (the honest GPU gap).
    for r in report['rows']:
        assert r['reduced_track_soft_deg'] < 0.1, r
        assert r['reduced_track_stiff_deg'] < 0.1, r


def test_drive_contrast_stiff_is_less_damped_and_higher_authority(report):
    c = report['contrast']
    # Stiff is LESS damped (zeta ratio < 1), HIGHER natural frequency, and 5x the effort ceiling, and
    # sits deeper in the numerically-stiff regime at the Isaac step (dt*wn larger) -- the regime a
    # finite-iteration solver rings in. The SOFT profile is a step milder on every axis.
    assert c['zeta_ratio'] < 1.0
    assert c['wn_ratio'] > 1.0
    assert c['effort_ratio'] == pytest.approx(5.0)
    assert c['stiff']['dt_wn'] > c['soft']['dt_wn']


def test_fix_routes_preplace_to_soft_keeps_place_stiff(report):
    f = report['fix']
    assert f['baseline_preplace']['kp'] == 1e6          # before: stiff
    assert f['fixed_preplace']['kp'] == 3e5             # after: soft
    assert f['fixed_place_stays_stiff']['kp'] == 1e6    # place descent stays stiff
    assert f['stiff_phases_baseline'] == ['pan', 'place', 'preplace']
    assert f['stiff_phases_fixed'] == ['pan', 'place']


def test_secondary_wrist_sweep_is_geometrically_forced(report):
    # The brief's secondary (shrink the 1.64 rad wrist_2 sweep by reseeding the standoff) is
    # geometrically infeasible: the above-support standoff orientation FORCES wrist_2 ~1.63 for any
    # level+reach branch, so the sweep from the carry pose (wrist_2 ~0) cannot be shrunk by reseeding.
    # This pins that finding so the report's "primary is the fix" conclusion stays honest.
    robot = parse_urdf(URDF)
    import json
    fix = json.loads(open(FIX).read())
    carry = fix['runs']['coldwin52']['waypoints'][0]
    standoff = fix['runs']['coldwin52']['waypoints'][-1]
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, list(carry)))
    tpos = cl.tool_position(robot, list(standoff))
    # solve the level standoff seeded from the CARRY pose (minimum-sweep seed)
    sol = cl.level_place_ik(robot, tpos, up, list(carry))
    w2 = float(sol['ur_wrist_2_joint'])
    # the solved wrist_2 (mod 2pi to the principal branch) is ~1.6, never near the carry ~0
    principal = (w2 + np.pi) % (2 * np.pi) - np.pi
    assert abs(principal) > 1.2, w2    # a >1.2 rad sweep is unavoidable


def test_fixture_matches_known_run_peaks():
    import json
    fix = json.loads(open(FIX).read())
    assert fix['runs']['coldwin50']['trip']['peak'] == 7.64
    assert fix['runs']['coldwin51']['trip']['peak'] == 3.28
    assert fix['runs']['coldwin52']['trip']['peak'] == 8.44
