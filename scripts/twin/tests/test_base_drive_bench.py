"""CPU tests for the base-drive-fidelity bench core (no Isaac/ROS/GPU) -- fm434."""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.tools.base_drive_bench import step_profile, tracking_report, DriveCommand


def test_step_profile_straight_then_min_radius_arc():
    cmds = step_profile(dt=0.1, straight_s=1.0, arc_s=1.0, v=0.25, min_radius=1.5)
    assert len(cmds) == 20
    straight = [c for c in cmds if c.phase == 'straight']
    arc = [c for c in cmds if c.phase == 'arc']
    assert len(straight) == 10 and len(arc) == 10
    assert all(c.v == 0.25 and c.omega == 0.0 for c in straight)
    assert all(c.v == 0.25 and abs(c.omega - 0.25 / 1.5) < 1e-9 for c in arc)   # omega = v / min_radius


def test_step_profile_rejects_bad_params():
    for bad in (dict(v=0.0), dict(min_radius=0.0), dict(dt=0.0)):
        try:
            step_profile(**bad)
            assert False, 'expected ValueError'
        except ValueError:
            pass


def test_tracking_report_perfect_is_within_tol():
    cmds = step_profile(dt=0.1, straight_s=1.0, arc_s=1.0)
    achieved = [(c.v, c.omega) for c in cmds]              # perfect tracking
    rep = tracking_report(cmds, achieved)
    assert rep['within_tol'] is True
    assert abs(rep['straight_v_ratio'] - 1.0) < 1e-6
    assert abs(rep['arc_v_ratio'] - 1.0) < 1e-6
    assert abs(rep['arc_omega_ratio'] - 1.0) < 1e-6


def test_tracking_report_undertracking_flags_not_within_tol():
    # the run-325/326 signature: the base achieves ~1/7 of commanded forward speed -> well outside 15%
    cmds = step_profile(dt=0.1, straight_s=1.0, arc_s=1.0)
    achieved = [(c.v * 0.14, c.omega) for c in cmds]       # under-translates badly
    rep = tracking_report(cmds, achieved)
    assert rep['within_tol'] is False
    assert rep['straight_v_ratio'] < 0.2


def test_tracking_report_within_15pct_passes():
    cmds = step_profile(dt=0.1, straight_s=1.0, arc_s=1.0)
    achieved = [(c.v * 0.9, c.omega * 1.1) for c in cmds]  # within 15% both axes
    rep = tracking_report(cmds, achieved)
    assert rep['within_tol'] is True


def test_tracking_report_drops_spinup_settle_window():
    # the first settle_frac of a phase is dropped: a bad first tick does not fail an otherwise-good phase
    cmds = step_profile(dt=0.1, straight_s=1.0, arc_s=0.0)
    achieved = [(0.0, 0.0)] + [(c.v, 0.0) for c in cmds[1:]]
    rep = tracking_report(cmds, achieved, settle_frac=0.25)
    assert rep['within_tol'] is True
