"""sim-gripper-status-relay-v1 -- CPU proof (no Isaac/ROS/GPU) for the coldwin65 pick the new named guard
(CONTACT_SUCTION_NOT_ENABLED) e-stopped although the suction WAS enabled and the pad gap was closing.

ROOT CAUSE (from coldwin65.console.log ~6815-6855 + coldwin65/host-stack.log:3203 + the code): PR 356 added
the gripper contact-feedback fields (gripper_enabled / seal_gap_mm / seal_pad_angle_deg / seal_hit_target)
to the ``Sensors`` packet and populated them in ``isaac_sensors.Sensors.capture`` via
``_contact_feedback_kwargs`` -- but the LIVE mission's host-facing /sensors come from
``isaac_factory.CollectTargetBridge`` (``capture`` / ``_capture_ground_truth_collect`` /
``_capture_transfer`` / ``_survival_packet``), which NEVER populated them. So every /sensors packet that
reached the host carried the all-default feedback (enabled=False, gap=None, hit=False) regardless of the
true Isaac state. coldwin65's Isaac side was ``Closing``/``Closed`` with the pad gap 9.25->4.87 mm (two
SEAL_DIAG instances: 96d0 = the IsaacPort/command instance, enabled=True; 5c90 = the build_sensing read
instance, commanded flag False but the SHARED plugin status Closing/Closed -- neither is a host mirror; the
host saw NO gripper status at all), yet the host's ``contact_require_suction_enabled`` guard read the
default False and e-stopped (host-stack.log:3203). The last-value /manipulation/status relay PR 356 blamed
is NOT implicated: the enable landed (reissue-each-tick; SEAL_DIAG command:close enabled=True throughout),
and /sensors has no heartbeat / last-value drop.

THE FIX (this lane): (1) the host-facing CollectTargetBridge capture paths carry the gripper feedback +
a gripper_status_time stamp when the opt-in ``RESTOCK_CONTACT_STATUS_FEEDBACK`` env is on (off =>
byte-identical); (2) ``contact_require_suction_enabled`` reads an AUTHORITATIVE status -- it aborts only on
a FRESH not-enabled, and on an ABSENT status logs CONTACT_STATUS_STALE and continues (never aborts on
staleness alone, the coldwin65 false abort). coldwin63's genuine lost-enable (a FRESH Open report) still
fails fast, so test_pick_contact_seal_fallback.py stays green.
"""
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock_manipulation.isaac_sensors import _contact_feedback_kwargs  # noqa: E402
from twin.restock_manipulation.isaac_factory import CollectTargetBridge  # noqa: E402
from twin.restock_manipulation import contact_status_feedback as csf  # noqa: E402
from twin.restock_manipulation import config_preflight as cp  # noqa: E402
from twin.restock_manipulation.types import Sensors  # noqa: E402
from twin.restock_manipulation.ros_adapter import sensors_from_json  # noqa: E402
from scripts.twin.tests.test_restock_manipulation import make_controller, packet, run_to  # noqa: E402


# --------------------------------------------------------------------------------------------------
# The opt-in env switch
# --------------------------------------------------------------------------------------------------
def test_contact_status_feedback_env_flag_is_explicit_truthy_only():
    assert csf.enabled({}) is False
    assert csf.enabled({csf.ENV: ''}) is False
    assert csf.enabled({csf.ENV: '0'}) is False
    assert csf.enabled({csf.ENV: 'nonsense'}) is False
    for on in ('1', 'true', 'TRUE', 'Yes', 'on'):
        assert csf.enabled({csf.ENV: on}) is True


# --------------------------------------------------------------------------------------------------
# _contact_feedback_kwargs now carries a sim-clock sample time (presence signal for the host guard)
# --------------------------------------------------------------------------------------------------
class _Contact:
    def __init__(self, fb):
        self._fb = fb

    def contact_feedback(self):
        return self._fb


def test_feedback_kwargs_without_now_is_byte_identical_to_pr356():
    # Legacy callers (isaac_sensors) pass no `now` -> no gripper_status_time key (PR 356 shape preserved).
    assert _contact_feedback_kwargs(_Contact((True, 0.3, 0.4, True))) == dict(
        gripper_enabled=True, seal_gap_mm=0.3, seal_pad_angle_deg=0.4, seal_hit_target=True)


def test_feedback_kwargs_with_now_stamps_gripper_status_time():
    assert _contact_feedback_kwargs(_Contact((True, 0.3, 0.4, True)), now=12.5) == dict(
        gripper_enabled=True, seal_gap_mm=0.3, seal_pad_angle_deg=0.4, seal_hit_target=True,
        gripper_status_time=12.5)


def test_feedback_kwargs_empty_on_missing_or_raising_probe_even_with_now():
    assert _contact_feedback_kwargs(object(), now=9.0) == {}

    class _Boom:
        def contact_feedback(self):
            raise RuntimeError('probe failed')
    assert _contact_feedback_kwargs(_Boom(), now=9.0) == {}


# --------------------------------------------------------------------------------------------------
# CollectTargetBridge._contact_status_kwargs: off => byte-identical, on => populated with the time
# --------------------------------------------------------------------------------------------------
def _bridge(feedback_on, fb=(True, 0.3, 0.4, True)):
    b = CollectTargetBridge.__new__(CollectTargetBridge)   # bypass the pxr/Isaac __init__
    b.contact_status_feedback = bool(feedback_on)
    b.contact = _Contact(fb)
    return b


def test_bridge_kwargs_empty_when_feedback_off_byte_identical():
    assert _bridge(False)._contact_status_kwargs(20.0) == {}


def test_bridge_kwargs_populated_when_feedback_on():
    assert _bridge(True)._contact_status_kwargs(20.0) == dict(
        gripper_enabled=True, seal_gap_mm=0.3, seal_pad_angle_deg=0.4, seal_hit_target=True,
        gripper_status_time=20.0)


def test_bridge_kwargs_on_but_probeless_contact_is_empty():
    # Feedback on but the contact has no probe -> empty (Sensors defaults), never a fabricated status.
    b = CollectTargetBridge.__new__(CollectTargetBridge)
    b.contact_status_feedback = True
    b.contact = object()
    assert b._contact_status_kwargs(20.0) == {}


# --------------------------------------------------------------------------------------------------
# Sensors round-trips gripper_status_time across the asdict -> JSON -> Sensors(**data) ROS boundary
# --------------------------------------------------------------------------------------------------
def test_sensors_round_trips_the_gripper_status_time():
    from dataclasses import asdict
    s = Sensors(simulation_time=5.0, observations=(), target=None, parked=True, contact=False,
                feedback_time=5.0, collision_time=5.0, joint_positions=(0.,) * 6,
                scanned_locations=(), gripper_enabled=True, seal_gap_mm=0.3, seal_hit_target=True,
                gripper_status_time=5.0)
    back = sensors_from_json(json.dumps(asdict(s), allow_nan=False))
    assert back.gripper_enabled is True and back.seal_gap_mm == 0.3 and back.gripper_status_time == 5.0
    # A legacy packet (no gripper_status_time key) deserializes to the None default (never a fake status).
    data = asdict(s)
    del data['gripper_status_time']
    assert sensors_from_json(json.dumps(data, allow_nan=False)).gripper_status_time is None


# --------------------------------------------------------------------------------------------------
# Controller guard: authoritative-status evaluation (coldwin65)
# --------------------------------------------------------------------------------------------------
def _contact_controller(**kwargs):
    c, m = make_controller('collect', contact_straight_line=True, contact_overpress_step_m=0.003,
                           contact_overpress_cap_m=0.02, contact_require_suction_enabled=True, **kwargs)
    return c, m


def _press(c, t, n, **fb):
    last = None
    for i in range(n):
        tick = t + i * .1
        last = c.step(.1, packet(c, tick, contact=False, **fb))
        if last.status == 'failed' or c.phase != 'contact':
            break
    return last


def test_fix_on_fresh_enabled_true_does_not_abort_the_sealing_pick_coldwin65():
    # coldwin65 WITH the feedback relay on: the host reads the real enabled=True (plugin Closing) + a fresh
    # gripper_status_time, so the guard lets the bounded over-press proceed (no CONTACT_SUCTION_NOT_ENABLED).
    c, _ = _contact_controller()
    t = run_to(c, 'contact')
    r = None
    for i in range(30):
        tick = t + i * .1
        r = c.step(.1, packet(c, tick, contact=False, gripper_enabled=True, seal_gap_mm=4.8,
                              seal_hit_target=True, gripper_status_time=tick))
        if r.status == 'failed' or c.phase != 'contact':
            break
    assert not (r.status == 'failed' and 'CONTACT_SUCTION_NOT_ENABLED' in r.reason), r.reason
    assert c._contact_step > 0.0, 'with a fresh enabled status the over-press advances toward the seal'


def test_fresh_not_enabled_aborts_with_authoritative_status():
    c, _ = _contact_controller()
    t = run_to(c, 'contact')
    r = None
    for i in range(40):
        tick = t + i * .1
        r = c.step(.1, packet(c, tick, contact=False, gripper_enabled=False, gripper_status_time=tick))
        if r.status == 'failed':
            break
    assert r.status == 'failed' and 'CONTACT_SUCTION_NOT_ENABLED' in r.reason, r.reason
    assert 'FRESH Isaac status' in r.reason
    assert c._contact_step == 0.0, 'a fresh not-enabled fails at the first over-press step'


def test_absent_status_never_aborts_on_staleness_alone_coldwin65_false_abort_removed(capsys):
    # The coldwin65 pre-fix case: no gripper status reached the host (gripper_status_time absent). The guard
    # must NOT raise CONTACT_SUCTION_NOT_ENABLED; it logs CONTACT_STATUS_STALE and continues the over-press,
    # failing (if the seal never forms) by the HONEST cap reason -- never falsely blaming the enable command.
    c, _ = _contact_controller()
    t = run_to(c, 'contact')
    r = _press(c, t, 120, gripper_enabled=False)  # no gripper_status_time => absent/starved
    assert r.status == 'failed'
    assert 'CONTACT_SUCTION_NOT_ENABLED' not in r.reason, r.reason
    assert 'contact not reached within capped over-press travel' in r.reason
    assert 'CONTACT_STATUS_STALE' in capsys.readouterr().out
    assert c._contact_step > 0.015, 'staleness => the bounded over-press proceeds (no false fast-fail)'


def test_status_max_age_bound_treats_an_old_present_status_as_stale(capsys):
    # With an explicit bound, a present-but-too-old gripper_status_time is treated as 'no fresh status'
    # (stale) rather than a judgement -> no CONTACT_SUCTION_NOT_ENABLED abort.
    c, _ = _contact_controller(contact_status_max_age_s=0.2)
    t = run_to(c, 'contact')
    r = None
    for i in range(120):
        tick = t + i * .1
        # stamp the status 1.0 s in the past (older than the 0.2 s bound) with enabled=False
        r = c.step(.1, packet(c, tick, contact=False, gripper_enabled=False, gripper_status_time=tick - 1.0))
        if r.status == 'failed':
            break
    assert r.status == 'failed' and 'CONTACT_SUCTION_NOT_ENABLED' not in r.reason, r.reason
    assert 'CONTACT_STATUS_STALE' in capsys.readouterr().out


def test_guard_off_is_byte_identical_regardless_of_status_fields():
    # With contact_require_suction_enabled off the status fields are never consulted (byte-identical).
    c, m = make_controller('collect', contact_straight_line=True, contact_overpress_step_m=0.003,
                           contact_overpress_cap_m=0.02)
    assert c.contact_require_suction_enabled is False
    t = run_to(c, 'contact')
    r = _press(c, t, 120, gripper_enabled=False, gripper_status_time=t)
    assert r.status == 'failed' and 'contact not reached within capped over-press travel' in r.reason


# --------------------------------------------------------------------------------------------------
# Replay regression: the coldwin65 tick sequence (Isaac enabled=True + closing gap while the host snapshot
# is starved) -- NO false CONTACT_SUCTION_NOT_ENABLED with the fix on; a genuine lost-enable still aborts.
# --------------------------------------------------------------------------------------------------
# coldwin65 SEAL_DIAG: status Closed/Closing (enabled=True), the pad gap closing 9.25 -> 4.87 mm, the only
# hit body is the carton. The host, pre-fix, saw gap=None hit=False enabled=False for every tick.
_CW65_GAPS_MM = [9.25, 9.25, 9.25, 9.24, 9.27, 9.25, 9.25, 9.23, 4.87]


def _replay_cw65(feedback_on):
    """Drive the contact press with the coldwin65 Isaac truth. feedback_on=True => the relay carries the
    real enabled/gap/time to the host; False => the starved pre-fix view (no status)."""
    c, _ = _contact_controller()
    t = run_to(c, 'contact')
    r = None
    for i in range(60):
        tick = t + i * .1
        gap = _CW65_GAPS_MM[min(i, len(_CW65_GAPS_MM) - 1)]
        if feedback_on:
            fb = dict(gripper_enabled=True, seal_gap_mm=gap, seal_hit_target=True, gripper_status_time=tick)
        else:
            fb = {}  # the host saw nothing populated
        r = c.step(.1, packet(c, tick, contact=False, **fb))
        if r.status == 'failed' or c.phase != 'contact':
            break
    return c, r


def test_replay_cw65_fix_on_no_false_suction_abort():
    c, r = _replay_cw65(feedback_on=True)
    assert not (r.status == 'failed' and 'CONTACT_SUCTION_NOT_ENABLED' in r.reason), r.reason
    assert c._contact_step > 0.0


def test_replay_cw65_fix_off_removes_the_false_abort_but_still_fails_honestly():
    # The coldwin65 bug was the FALSE CONTACT_SUCTION_NOT_ENABLED on a starved view. With the stale-aware
    # guard that false abort is gone even when the relay is off; the honest cap fault remains the outcome.
    c, r = _replay_cw65(feedback_on=False)
    assert r.status == 'failed'
    assert 'CONTACT_SUCTION_NOT_ENABLED' not in r.reason, r.reason
    assert 'contact not reached within capped over-press travel' in r.reason


# --------------------------------------------------------------------------------------------------
# Merge/no-erase: a heartbeat / empty-controls status does not erase the Isaac-side gripper command, and a
# survival ('empty observations') packet still carries the gripper status when the relay is on.
# --------------------------------------------------------------------------------------------------
def test_heartbeat_empty_controls_does_not_command_the_gripper_closed():
    # /manipulation/status (host->Isaac) heartbeats carry NO gripper control; IsaacRosGraph.poll must not
    # toggle the plugin on them, so a landed enable is never erased by a later empty-controls status.
    from scripts.twin.tests.test_pick_contact_seal_fallback import (
        _RecordingContact, _LastValueRelay, _poll_graph, _status)
    contact, relay = _RecordingContact(), _LastValueRelay()
    g, T = _poll_graph(contact, relay)
    relay.host_publish(T['status'], _status(1, [{'kind': 'gripper', 'enabled': True}], 10.0))
    g.poll(10.05)
    assert contact.commands == [True]
    # a heartbeat (empty controls, fresh sequence) must NOT re-command / erase the gripper
    relay.host_publish(T['status'], _status(2, [], 10.1))
    g.poll(10.15)
    assert contact.commands == [True], 'an empty-controls heartbeat must not erase the landed enable'


def test_survival_packet_path_still_carries_gripper_status_when_on():
    # _survival_packet (empty observations) uses the same _contact_status_kwargs, so an occlusion tick does
    # not erase the gripper status from the host's view (merge-per-tick, never a blank snapshot).
    assert _bridge(True)._contact_status_kwargs(31.0)['gripper_status_time'] == 31.0


# --------------------------------------------------------------------------------------------------
# Preflight registration (the new bound is a HOST key; a misfile is refused, never silently inert)
# --------------------------------------------------------------------------------------------------
def test_contact_status_max_age_s_registered_host():
    assert cp.KEY_CONSUMER.get('contact_status_max_age_s') == cp.HOST


def test_contact_status_max_age_s_misplaced_in_runtime_is_refused():
    problems = cp.check_activation({}, {'contact_status_max_age_s': 0.5}, ['contact_status_max_age_s'])
    assert len(problems) == 1 and 'MIS-PLACED' in problems[0]
    assert cp.check_activation({'contact_status_max_age_s': 0.5}, {}, ['contact_status_max_age_s']) == []


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
