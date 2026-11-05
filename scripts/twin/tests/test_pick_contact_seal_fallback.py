"""sim-pick-contact-seal-fallback-v1 -- CPU proof (no Isaac/ROS/GPU) for the coldwin63 pick that never
sealed, and for the three opt-in guards that make the contact phase either seal reliably or fail by name.

ROOT CAUSE (reconstructed from the two host/console logs + the code; full write-up in
docs/examples/pick-contact-seal-fallback-fragment.json):

  coldwin63 host-stack.log:2540 shows the controller DID issue ``CONTACT_SUCTION enable`` and pressed six
  CONTACT_STEP advances (3..18 mm) before HOST_FAULT_ESTOP 'contact not reached within capped over-press
  travel'. But the mission CONSOLE never logged ``SEAL_DIAG command:close`` (0 of 959 reads; plugin stayed
  status=Open enabled=False) -- the one-shot gripper-enable control was LOST. The controller runs in the
  host subprocess; its enable reaches the Isaac plugin ONLY via the /manipulation/status ``controls`` field,
  relayed by control_ipc.HostControlPublisher which keeps ONLY the latest payload per topic (a last-value
  relay). A later heartbeat / empty-controls status overwrote the single control-bearing status before
  ``isaac_ros_graph.IsaacRosGraph.poll`` read it; poll then advances ``last_sequence`` past it and
  ``cad_tool.NativePanelContact.command(close)`` is never called -> the plugin never enables -> the gated
  SIM_ATTACH_FALLBACK stays skip=suction_off -> contact never forms -> the over-press cap fault. coldwin62
  (and cw61) won the race (console:4881 command:close BEFORE contact); cw63 lost it (its 22 vs 9 pre-grasp
  'suction_off' ticks are that larger overwrite window).

The guards (all opt-in HOST keys, default off => byte-identical):
  (item 2) contact_reissue_suction_each_tick -- re-publish the enable every press tick so the relay always
           carries a fresh control (the RealRelayDrop test proves the single-shot drop and the re-emit fix
           through the REAL IsaacRosGraph.poll + a faithful last-value relay).
  (3a) contact_require_suction_enabled [+ contact_reissue_close_once] -- fail fast CONTACT_SUCTION_NOT_ENABLED
       (or re-send close once, then fail) at the first over-press step if the plugin never reported enabled.
  (3b) contact_accept_seal_diag [+ dwell/gap/angle] -- treat a dwell-stable flush geometric SEAL_DIAG on a
       body under the pad (with suction enabled) as contact.
"""
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock_manipulation.cad_tool import seal_feedback_from_probe  # noqa: E402
from twin.restock_manipulation.isaac_sensors import _contact_feedback_kwargs  # noqa: E402
from twin.restock_manipulation import config_preflight as cp  # noqa: E402
# reuse the controller-tier harness (FixtureMotion / make_controller / packet / run_to)
from scripts.twin.tests.test_restock_manipulation import make_controller, packet, run_to  # noqa: E402


# --------------------------------------------------------------------------------------------------
# Pure feedback extraction: seal_feedback_from_probe (SEAL_DIAG probe -> controller-facing fields)
# --------------------------------------------------------------------------------------------------
def test_seal_feedback_from_probe_enabled_keys_on_plugin_status_not_commanded_flag():
    # cw63: commanded enabled stayed False AND status Open -> not enabled.
    assert seal_feedback_from_probe(
        {'status': 'GripperStatus.Open', 'enabled': False, 'min_gap_mm': 0.0,
         'pad_vs_surface_deg': 0.0, 'raycast': [['/RestockScenario/HandlingUnit', 0.0]]}
    ) == (False, 0.0, 0.0, True)
    # cw62: the close landed -> status Closing -> enabled True even though 'enabled' flag unseen here.
    enabled, gap, ang, hit = seal_feedback_from_probe(
        {'status': 'GripperStatus.Closing', 'min_gap_mm': 0.3, 'pad_vs_surface_deg': 0.4,
         'raycast': [['/RestockScenario/HandlingUnit', 0.3]]})
    assert enabled is True and gap == 0.3 and ang == 0.4 and hit is True
    assert seal_feedback_from_probe({'status': 'GripperStatus.Closed'})[0] is True


def test_seal_feedback_from_probe_is_safe_on_garbage_and_missing():
    assert seal_feedback_from_probe(None) == (False, None, None, False)
    assert seal_feedback_from_probe({}) == (False, None, None, False)
    assert seal_feedback_from_probe({'status': 'ERR:boom'}) == (False, None, None, False)
    # non-finite gap/angle degrade to None (no false seal evidence); no raycast => no hit
    assert seal_feedback_from_probe(
        {'status': 'GripperStatus.Closing', 'min_gap_mm': float('nan'),
         'pad_vs_surface_deg': float('inf'), 'raycast': [[None, None]]}) == (True, None, None, False)


def test_contact_feedback_kwargs_is_empty_without_the_method_and_on_raise():
    assert _contact_feedback_kwargs(object()) == {}            # no contact_feedback attr -> Sensors defaults

    class _Boom:
        def contact_feedback(self):
            raise RuntimeError('probe failed')
    assert _contact_feedback_kwargs(_Boom()) == {}             # best-effort; never propagates

    class _Ok:
        def contact_feedback(self):
            return (True, 0.2, 0.5, True)
    assert _contact_feedback_kwargs(_Ok()) == dict(
        gripper_enabled=True, seal_gap_mm=0.2, seal_pad_angle_deg=0.5, seal_hit_target=True)


# --------------------------------------------------------------------------------------------------
# The DELIVERY race through the REAL IsaacRosGraph.poll: single-shot drop vs the re-emit fix
# --------------------------------------------------------------------------------------------------
class _LastValueRelay:
    """Faithful model of control_ipc.HostControlPublisher: keeps ONLY the latest payload per topic."""
    def __init__(self):
        self._latest = {}

    def host_publish(self, topic, data):   # the host side writes every status here
        self._latest[topic] = data

    def read(self, topic):                 # the mission poll reads the latest
        return self._latest.get(topic)


class _RecordingContact:
    def __init__(self):
        self.commands = []

    def command(self, enabled):
        self.commands.append(bool(enabled))


def _poll_graph(contact, relay):
    """A minimal IsaacRosGraph wired to the fake relay + contact, bypassing the OmniGraph __init__.
    poll() is the REAL mission-side delivery path (sequence gate + gripper acquisition freshness +
    contact.command)."""
    from twin.restock_manipulation.isaac_ros_graph import IsaacRosGraph, TOPICS

    class _G(IsaacRosGraph):
        def __init__(self):
            pass
        def check_health(self):
            return None
        def hold(self):
            self._held = True

    g = _G()
    g.og = None                                   # unused when host_control is wired (line `og=self.og`)
    g.host_control, g.contact = relay, contact
    g.active, g.job_id = True, 'job-1'
    g.last_status, g.last_sequence, g.last_receive, g.last_receive_sim = '', 0, 0.0, None
    g.faulted, g.timeout = False, 5.0
    g.TOPICS = TOPICS
    return g, TOPICS


def _status(seq, controls, sim_now):
    import json
    return json.dumps({'status': 'running', 'phase': 'contact', 'reason': '', 'controls': controls,
                       'evidence': [], 'sequence': seq, 'job_id': 'job-1',
                       'valid_until_steady': sim_now + 100.0})


def test_single_shot_enable_is_dropped_by_the_last_value_relay_overwrite_cw63():
    # cw63: the host emits the control-bearing status (seq 1) then a heartbeat (seq 2, no controls) before
    # the mission polls -> the relay holds only seq 2 -> command:close is NEVER applied (the observed bug).
    contact, relay = _RecordingContact(), _LastValueRelay()
    g, T = _poll_graph(contact, relay)
    relay.host_publish(T['status'], _status(1, [{'kind': 'gripper', 'enabled': True}], 10.0))
    relay.host_publish(T['status'], _status(2, [], 10.1))      # overwrites seq 1 before the poll
    g.poll(10.2)
    assert contact.commands == [], 'the one-shot gripper enable must be shown dropped (reproduces cw63)'


def test_reissue_each_tick_defeats_the_relay_overwrite():
    # item 2: when the enable rides EVERY status, the relay's latest payload always carries it, so the poll
    # applies command:close regardless of how many heartbeats intervened.
    contact, relay = _RecordingContact(), _LastValueRelay()
    g, T = _poll_graph(contact, relay)
    for seq, now in ((1, 10.0), (2, 10.1), (3, 10.2)):         # every tick re-publishes the enable
        relay.host_publish(T['status'], _status(seq, [{'kind': 'gripper', 'enabled': True}], now))
        g.poll(now + 0.05)
    assert contact.commands and contact.commands[0] is True, 'command:close must be applied under re-emit'


# --------------------------------------------------------------------------------------------------
# Controller tier: the contact-phase guards (fixture motion; packet() carries the new feedback via kwargs)
# --------------------------------------------------------------------------------------------------
def _contact_controller(**kwargs):
    c, m = make_controller('collect', contact_straight_line=True, contact_overpress_step_m=0.003,
                           contact_overpress_cap_m=0.02, **kwargs)
    return c, m


def _press(c, t, n, **fb):
    """Step the contact press n ticks with contact never latching; fb overrides the feedback fields.

    sim-gripper-status-relay-v1: stamp a FRESH gripper_status_time on every packet (unless the caller
    overrides it) so the require_suction_enabled guard sees an AUTHORITATIVE Isaac status -- the cw63 case
    is a plugin that freshly reports not-enabled, not an absent status (which the guard must never abort on).
    The coldwin65 starve (no status) is exercised explicitly in test_gripper_status_relay.py."""
    last = None
    for i in range(n):
        tick = t + i * .1
        last = c.step(.1, packet(c, tick, contact=False, **dict(gripper_status_time=tick, **fb)))
        if last.status == 'failed' or c.phase != 'contact':
            break
    return last


# --- item 2: controller re-emits the enable every press tick (opt-in) ---
def test_controller_reissues_enable_each_press_tick_when_on():
    c, _ = _contact_controller(contact_reissue_suction_each_tick=True)
    t = run_to(c, 'contact')
    c.step(.1, packet(c, t, contact=False))                    # issue the contact move
    r = c.step(.1, packet(c, t + .1, contact=False))           # CONTACT_SUCTION enable (first emit)
    assert any(ctl.get('kind') == 'gripper' and ctl['enabled'] for ctl in r.controls)
    # subsequent press ticks RE-EMIT the enable (defeats the relay overwrite)
    seen = 0
    for i in range(12):
        r = c.step(.1, packet(c, t + .2 + i * .1, contact=False))
        if any(ctl.get('kind') == 'gripper' and ctl['enabled'] for ctl in r.controls):
            seen += 1
        if r.status == 'failed':
            break
    assert seen >= 3, 'the enable must be re-published on multiple press ticks'


def test_controller_does_not_reissue_when_off_byte_identical():
    c, _ = _contact_controller()                               # default: reissue off
    assert c.contact_reissue_suction_each_tick is False
    t = run_to(c, 'contact')
    c.step(.1, packet(c, t, contact=False))
    c.step(.1, packet(c, t + .1, contact=False))               # the single CONTACT_SUCTION enable
    emits = 0
    for i in range(12):
        r = c.step(.1, packet(c, t + .2 + i * .1, contact=False))
        emits += sum(1 for ctl in r.controls if ctl.get('kind') == 'gripper')
        if r.status == 'failed':
            break
    assert emits == 0, 'default (off) must not re-emit the enable after the one-shot (byte-identical)'


# --- guard 3a: require_suction_enabled ---
def test_guard_3a_fails_fast_when_plugin_never_enables():
    c, _ = _contact_controller(contact_require_suction_enabled=True)
    t = run_to(c, 'contact')
    r = _press(c, t, 30, gripper_enabled=False)                # the cw63 case: plugin never enables
    assert r.status == 'failed' and 'CONTACT_SUCTION_NOT_ENABLED' in r.reason, r.reason
    assert c._contact_step == 0.0, 'must fail at the FIRST over-press step, not press the whole cap'


def test_guard_3a_reissue_close_once_then_recovers_when_the_enable_lands():
    c, _ = _contact_controller(contact_require_suction_enabled=True, contact_reissue_close_once=True)
    t = run_to(c, 'contact')
    # step until the first boundary miss re-sends the close (exactly once), capturing that control
    reissued = False
    for i in range(15):
        r = c.step(.1, packet(c, t + i * .1, contact=False, gripper_enabled=False,
                              gripper_status_time=t + i * .1))
        if any(ctl.get('kind') == 'gripper' and ctl['enabled'] for ctl in r.controls) and c._contact_suction:
            if c._contact_suction_miss == 1:                   # the guard-3a re-send (not the initial enable)
                reissued = True
                break
        assert r.status == 'running', r.reason
    assert reissued and c._contact_suction_miss == 1
    # the re-sent close now lands (plugin reports enabled) -> the press advances normally (no fail)
    r = _press(c, c.last_time + .1, 15, gripper_enabled=True)
    assert r.status != 'failed', r.reason
    assert c._contact_step > 0.0, 'with the enable confirmed the over-press proceeds'


def test_guard_3a_reissue_close_once_fails_on_a_second_miss():
    c, _ = _contact_controller(contact_require_suction_enabled=True, contact_reissue_close_once=True)
    t = run_to(c, 'contact')
    r = _press(c, t, 40, gripper_enabled=False)                # never enables: reissue once, then fail
    assert r.status == 'failed' and 'CONTACT_SUCTION_NOT_ENABLED' in r.reason, r.reason
    assert c._contact_suction_miss == 2


def test_guard_3a_default_off_presses_to_cap_byte_identical():
    c, _ = _contact_controller()                               # guard off
    assert c.contact_require_suction_enabled is False
    t = run_to(c, 'contact')
    r = _press(c, t, 120, gripper_enabled=False)
    assert r.status == 'failed' and 'contact not reached within capped over-press travel' in r.reason
    assert c._contact_step > 0.015, 'off => the historical over-press-to-cap path (byte-identical)'


# --- guard 3b: accept_seal_diag ---
_GOOD_SEAL = dict(gripper_enabled=True, seal_hit_target=True, seal_gap_mm=0.3, seal_pad_angle_deg=0.5)


def test_guard_3b_accepts_a_dwell_stable_flush_seal_as_contact():
    c, _ = _contact_controller(contact_accept_seal_diag=True, contact_seal_diag_dwell_ticks=5)
    t = run_to(c, 'contact')
    c.step(.1, packet(c, t, contact=False, **_GOOD_SEAL))      # issue
    c.step(.1, packet(c, t + .1, contact=False, **_GOOD_SEAL))  # suction enable
    reached = False
    for i in range(20):
        c.step(.1, packet(c, t + .2 + i * .1, contact=False, **_GOOD_SEAL))
        if c.phase in ('grasp', 'lift'):
            reached = True
            break
    assert reached, 'a dwell-stable flush seal with suction enabled must advance out of the press'
    assert c._contact_seal_disagree_logged is True


def test_guard_3b_dwell_boundary_needs_consecutive_ticks():
    c, _ = _contact_controller(contact_accept_seal_diag=True, contact_seal_diag_dwell_ticks=5)
    t = run_to(c, 'contact')
    c.step(.1, packet(c, t, contact=False, **_GOOD_SEAL))
    c.step(.1, packet(c, t + .1, contact=False, **_GOOD_SEAL))
    # four good ticks (dwell 4 < 5) must NOT yet accept
    for i in range(4):
        c.step(.1, packet(c, t + .2 + i * .1, contact=False, **_GOOD_SEAL))
    assert c.phase == 'contact' and c._contact_seal_dwell == 4
    c.step(.1, packet(c, t + .7, contact=False, **_GOOD_SEAL))  # 5th consecutive -> accept
    assert c.phase in ('grasp', 'lift')


def test_guard_3b_resets_dwell_on_a_bad_tick():
    c, _ = _contact_controller(contact_accept_seal_diag=True, contact_seal_diag_dwell_ticks=5)
    t = run_to(c, 'contact')
    c.step(.1, packet(c, t, contact=False, **_GOOD_SEAL))
    c.step(.1, packet(c, t + .1, contact=False, **_GOOD_SEAL))
    for i in range(3):
        c.step(.1, packet(c, t + .2 + i * .1, contact=False, **_GOOD_SEAL))
    assert c._contact_seal_dwell == 3
    c.step(.1, packet(c, t + .5, contact=False, **dict(_GOOD_SEAL, seal_gap_mm=9.0)))  # gap out of bound
    assert c._contact_seal_dwell == 0 and c.phase == 'contact'


def test_guard_3b_rejects_wrong_body_angle_and_disabled_suction():
    for bad in (dict(seal_hit_target=False),                   # no body under the pad
                dict(seal_pad_angle_deg=5.0),                  # pad not flush
                dict(seal_gap_mm=3.0),                         # not in contact
                dict(gripper_enabled=False)):                  # suction not enabled
        c, _ = _contact_controller(contact_accept_seal_diag=True, contact_seal_diag_dwell_ticks=3)
        t = run_to(c, 'contact')
        c.step(.1, packet(c, t, contact=False, **dict(_GOOD_SEAL, **bad)))
        c.step(.1, packet(c, t + .1, contact=False, **dict(_GOOD_SEAL, **bad)))
        for i in range(10):
            c.step(.1, packet(c, t + .2 + i * .1, contact=False, **dict(_GOOD_SEAL, **bad)))
        assert c.phase == 'contact', f'{bad} must NOT be accepted as contact'


def test_guard_3b_default_off_ignores_seal_feedback_byte_identical():
    c, _ = _contact_controller()                               # accept off
    assert c.contact_accept_seal_diag is False
    t = run_to(c, 'contact')
    r = _press(c, t, 120, **_GOOD_SEAL)                        # perfect seal every tick, but key off
    assert r.status == 'failed' and 'contact not reached within capped over-press travel' in r.reason
    assert c._contact_seal_dwell == 0, 'off => the seal diagnostic is never consulted (byte-identical)'


# --------------------------------------------------------------------------------------------------
# Preflight registration (a key filed in the wrong config is refused, never silently inert)
# --------------------------------------------------------------------------------------------------
def test_new_keys_registered_as_host():
    for key in ('contact_reissue_suction_each_tick', 'contact_require_suction_enabled',
                'contact_reissue_close_once', 'contact_accept_seal_diag',
                'contact_seal_diag_dwell_ticks', 'contact_seal_diag_max_gap_mm',
                'contact_seal_diag_max_angle_deg'):
        assert cp.KEY_CONSUMER.get(key) == cp.HOST, key


def test_misplaced_guard_key_in_runtime_config_is_refused():
    problems = cp.check_activation({}, {'contact_accept_seal_diag': True}, ['contact_accept_seal_diag'])
    assert len(problems) == 1 and 'MIS-PLACED' in problems[0]
    assert cp.check_activation({'contact_accept_seal_diag': True}, {}, ['contact_accept_seal_diag']) == []


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
