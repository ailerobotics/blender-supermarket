"""CPU tests for the release-open RE-ISSUE fix (sim-release-open-reissue-v1).

coldwin84 was a byte-identical re-run of the cleanly-released coldwin83 that HUNG at the cart set-down: the
contact-gated weld release fired, the release phase emitted the gripper-open control ONCE, but the Isaac
side NEVER ran ``cad_tool.command(open)`` -- ``coldwin84.console.log`` has ZERO ``SEAL_DIAG command:open``
lines (the last ``SEAL_DIAG`` is still ``command:close ... enabled=True``) while coldwin83 has exactly ONE.
The single open-carrying ``/manipulation/status`` was overwritten on the LAST-VALUE control relay by a
newer host status before ``IsaacRosGraph.poll`` read it (the same one-shot drop PR 356 fixed for the
contact ENABLE), so the ``SIM_ATTACH_FALLBACK`` weld never released and the run e-stopped: ``release did
not detach the held carton within 8.0s sim``.

The fix (opt-in ``carton_level.release_reissue``) re-emits the gripper-open control -- which on the Isaac
side is BOTH the open AND the weld detach -- EVERY release tick until the PR 358 per-tick feedback confirms
gripper open + carton not held (seal released), then proceeds to the cuMotion detach; it FAILS BY NAME
(``RELEASE_NOT_CONFIRMED``) if unconfirmed within the bounded sim window, never restarting the stage with
the carton welded. Default off => byte-identical.

These tests cover:
  * the pure ``ReleaseReissueGate``: confirms on the first tick, confirms after dropped commands, never
    confirms -> timeout; and the no-feedback / still-enabled gating.
  * the controller release phase: OFF is byte-identical (single-shot open + legacy detach); ON re-emits
    the open every tick, logs ``RELEASE_REISSUE`` / ``RELEASE_CONFIRMED``, detaches on confirm, and FAILS
    ``RELEASE_NOT_CONFIRMED`` with NO detach when it never confirms.
  * replay regressions: the coldwin84 hang (OFF reproduces the never-detach e-stop; ON confirms once the
    re-emitted open lands) and the coldwin83 single-shot success (ON still confirms on the first tick).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_release_open_reissue_v1.py
"""
import os
import time

import pytest

from twin.restock_manipulation import Observation, ProductProfile, Sensors, Target, TransferController
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import release_reissue as rr

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='dataset mobile URDF not present')

PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))
_SL_STOW = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
_SL_TUCK = (-2.3, 1.0, 2.1066, 0.8, -1.2545)
_SL_PREPLACE_SHAPE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)
_GRIPPER_OPEN = ({'kind': 'gripper', 'enabled': False},)


# ================================================================= pure ReleaseReissueGate

def test_gate_confirms_on_first_tick():
    """The clean coldwin83 class: the open lands immediately, so the first release tick already reads the
    carton not held with the gripper open -> confirm on tick 1, no re-issue."""
    g = rr.ReleaseReissueGate(confirm_timeout_s=8.0)
    g.reset(10.0)
    d = g.observe(now=10.0, held=False, gripper_enabled=False, gripper_status_time=10.0)
    assert d.action == 'confirmed' and d.reissue_control is False and d.ticks == 1
    assert g.confirmed_ticks == 1


def test_gate_confirms_after_dropped_commands():
    """The coldwin84 class: the first N opens are dropped on the relay (carton stays held), so the gate
    re-issues each tick; once an open lands (held clears, gripper reads open) it confirms."""
    g = rr.ReleaseReissueGate(confirm_timeout_s=8.0, log_every=2)
    g.reset(0.0)
    logs = []
    for i, t in enumerate((0.0, 0.1, 0.2)):          # three dropped opens: still held, gripper still closed
        d = g.observe(now=t, held=True, gripper_enabled=True, gripper_status_time=t)
        assert d.action == 'reissue' and d.reissue_control is True
        if d.log is not None:
            logs.append(d.log)
    # the open finally lands: weld gone (not held) + plugin Open
    d = g.observe(now=0.3, held=False, gripper_enabled=False, gripper_status_time=0.3)
    assert d.action == 'confirmed' and d.ticks == 4
    assert logs and logs[0].startswith('RELEASE_REISSUE n=1')      # one line per log_every ticks


def test_gate_never_confirms_times_out():
    """A weld that never releases within the window times out (the host then fails by name)."""
    g = rr.ReleaseReissueGate(confirm_timeout_s=1.0)
    g.reset(0.0)
    d = None
    for t in (0.0, 0.5, 1.5):
        d = g.observe(now=t, held=True, gripper_enabled=True, gripper_status_time=t)
    assert d.action == 'timeout' and d.held is True and d.gripper_open is False


def test_gate_not_held_with_no_feedback_confirms():
    """No fresh gripper status (feedback relay off): the physical seal-released signal (not held) carries
    the confirmation -- the feedback cannot contradict."""
    g = rr.ReleaseReissueGate(confirm_timeout_s=5.0)
    g.reset(0.0)
    d = g.observe(now=0.0, held=False, gripper_enabled=True, gripper_status_time=None)
    assert d.action == 'confirmed' and d.gripper_status_fresh is False


def test_gate_not_held_but_gripper_still_enabled_keeps_reissuing():
    """Carton free but a FRESH status still reads the gripper closed -> not confirmed yet (keep re-issuing
    until the plugin reports Open); this never confirms a contradictory state."""
    g = rr.ReleaseReissueGate(confirm_timeout_s=8.0)
    g.reset(0.0)
    d = g.observe(now=0.0, held=False, gripper_enabled=True, gripper_status_time=0.0)
    assert d.action == 'reissue'


def test_gate_rejects_bad_timeout():
    for bad in (0.0, -1.0, float('nan')):
        with pytest.raises(ValueError):
            rr.ReleaseReissueGate(confirm_timeout_s=bad)


# ================================================================= config resolution

def test_config_default_off_byte_identical():
    lc = cl.resolve_level_config({'place_contact_release': True, 'place_contact_gap_m': 0.012})
    assert lc.release_reissue is False
    assert lc.release_confirm_timeout_s == 8.0 and lc.release_reissue_log_every == 30


def test_config_on_parses_fields():
    lc = cl.resolve_level_config({'release_reissue': True, 'release_confirm_timeout_s': 6.5,
                                  'release_reissue_log_every': 12})
    assert lc.release_reissue is True
    assert lc.release_confirm_timeout_s == 6.5 and lc.release_reissue_log_every == 12


@pytest.mark.parametrize('spec', (
    {'release_confirm_timeout_s': 0.0},
    {'release_confirm_timeout_s': -1.0},
    {'release_reissue_log_every': 0},
    {'release_reissue_log_every': -3},
))
def test_config_rejects_bad_fields(spec):
    with pytest.raises(ValueError):
        cl.resolve_level_config(spec)


# ================================================================= controller fixture

class ReleaseMotion:
    """A minimal terminal motion port for the release gripper-open stage: ``update_payload`` records the
    detach, ``stop`` is the fault path. The state is terminal ('succeeded') so the detach is always legal."""

    def __init__(self):
        self.state = 'succeeded'
        self.payloads = []
        self.stops = 0

    def begin(self, *a, **k):
        self.state = 'execute_result'

    def begin_joint_goal(self, *a, **k):
        self.state = 'execute_result'

    def begin_joint_path(self, *a, **k):
        self.state = 'execute_result'

    def poll(self):
        return 'running'

    def update_payload(self, held, profile):
        self.payloads.append((held, self.state))
        return True

    def stop(self):
        self.stops += 1

    def reset_crash_budget(self):
        pass

    def crash_recovery_active(self):
        return False

    def last_fault_was_sim_deadline(self):
        return False

    def take_exec_abort_reason(self):
        return None


def _packet(c, t, *, contact, q=(0.,) * 6, gripper_enabled=None, gripper_status_time=None,
            feedback=True):
    observation = Observation(f'src-{t}', 'fixture', t, 'sku-box', c.source,
                              (.4, 0., .2, 0., 0., 0., 1.), .99)
    target = Target(f'tgt-{t}', 'fixture', t, c.destination, (.5, .2, .1, 0., 0., 0., 1.), True, (.3, .3, .4))
    # By default the PR 358 feedback mirrors the physical latch: closed while held, open once free.
    if feedback:
        ge = (contact if gripper_enabled is None else gripper_enabled)
        gst = (t if gripper_status_time is None else gripper_status_time)
    else:
        ge, gst = False, None
    return Sensors(simulation_time=t, observations=(observation,), target=target, parked=True,
                   contact=contact, feedback_time=t, collision_time=t, joint_positions=tuple(q),
                   scanned_locations=(c.source, c.destination), scan_time=t,
                   valid_until_steady=time.monotonic() + 100., placed_pose=None,
                   gripper_enabled=bool(ge), gripper_status_time=gst)


def _release_controller(*, release_reissue, confirm_timeout_s=8.0, log_every=30):
    JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')
    spec = {'place_contact_release': True, 'place_contact_gap_m': 0.012, 'trip': False,
            'release_reissue': release_reissue, 'release_confirm_timeout_s': confirm_timeout_s,
            'release_reissue_log_every': log_every}
    lc = cl.resolve_level_config(spec)
    motion = ReleaseMotion()
    c = TransferController(
        motion, {'sku-box': PROFILE}, fixture_mode=True, transport_joints=_SL_STOW,
        staged_cart_swing=True, cart_pan_joint0=-0.8605, cart_tuck_shape=_SL_TUCK,
        reposition_before_place=True, collect_place_preplace_shape=_SL_PREPLACE_SHAPE,
        collect_place_wrist_guard=True, collect_place_straight_line=True, carton_level=lc,
        fk_urdf=URDF, carry_wrist1_cap=0.65, carry_wrist2_min=0.0, release_detach_timeout_s=8.0)
    c.start('collect', JOB)
    # Jump straight to the release gripper-open stage (post place-descent / post cancel-handshake): the
    # carton is still welded (contact True), the first open has NOT been emitted yet.
    c.phase, c.held, c.issued = 'release', True, False
    c._release_cancel_pending = False
    c.poses = {'place': (0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0)}
    return c, motion


# ================================================================= controller: OFF byte-identical

@need_urdf
def test_controller_off_single_shot_then_legacy_detach():
    """OFF: the first release tick emits ONE gripper-open; when contact clears the legacy path detaches and
    transitions to retreat -- unchanged behaviour (no re-issue gate constructed)."""
    c, m = _release_controller(release_reissue=False)
    assert c._release_reissue is None
    r = c.step(.1, _packet(c, 1.0, contact=True))
    assert r.controls == _GRIPPER_OPEN and c.issued is True and m.payloads == []
    # contact still held: NO re-issue (byte-identical), just a running result
    r = c.step(.1, _packet(c, 1.1, contact=True))
    assert r.controls == () and m.payloads == []
    # contact clears -> legacy detach + retreat
    r = c.step(.1, _packet(c, 1.2, contact=False))
    assert m.payloads == [(False, 'succeeded')] and c.phase == 'retreat'


@need_urdf
def test_controller_off_reproduces_cw84_hang():
    """OFF replay of the coldwin84 drop: the open never lands (contact stays True forever), so the legacy
    8 s ``release_detach_timeout_s`` fires the named e-stop and the carton is NEVER detached."""
    c, m = _release_controller(release_reissue=False)
    c.step(.1, _packet(c, 1.0, contact=True))        # the one-shot open (lost on the relay in the real run)
    r = None
    for i in range(200):
        r = c.step(.1, _packet(c, 2.0 + i * 0.1, contact=True))   # weld never releases
        if r.status == 'failed':
            break
    assert r.status == 'failed' and 'release did not detach' in r.reason
    assert m.payloads == []                           # carton still welded (never detached)


# ================================================================= controller: ON

@need_urdf
def test_controller_on_reissues_then_confirms(capsys):
    """ON: the open is re-emitted EVERY tick while the carton stays held; once the feedback confirms (not
    held + gripper open) the release detaches and transitions. RELEASE_REISSUE / RELEASE_CONFIRMED logged."""
    c, m = _release_controller(release_reissue=True, log_every=2)
    assert c._release_reissue is not None
    r = c.step(.1, _packet(c, 1.0, contact=True))     # first open, arms the gate
    assert r.controls == _GRIPPER_OPEN and m.payloads == []
    # three dropped opens: carton still held, every tick RE-EMITS the open (the relay-resilient re-issue)
    for t in (1.1, 1.2, 1.3):
        r = c.step(.1, _packet(c, t, contact=True))
        assert r.controls == _GRIPPER_OPEN and m.payloads == []
    # the open lands: contact clears, gripper reads open -> confirm, detach, retreat
    r = c.step(.1, _packet(c, 1.4, contact=False))
    assert m.payloads == [(False, 'succeeded')] and c.phase == 'retreat' and r.status != 'failed'
    out = capsys.readouterr().out
    assert 'RELEASE_REISSUE n=1' in out and 'RELEASE_CONFIRMED' in out


@need_urdf
def test_controller_on_confirms_first_tick_cw83():
    """ON replay of the clean coldwin83 single-shot success: the open lands immediately, so the tick after
    the first open already confirms (held clears) -- the fix does not slow the success path."""
    c, m = _release_controller(release_reissue=True)
    c.step(.1, _packet(c, 1.0, contact=True))         # first open
    r = c.step(.1, _packet(c, 1.1, contact=False))    # open landed: free + gripper open -> confirm
    assert m.payloads == [(False, 'succeeded')] and c.phase == 'retreat' and r.status != 'failed'


@need_urdf
def test_controller_on_never_confirms_fails_by_name(capsys):
    """ON: a weld that never releases within the bounded window FAILS BY NAME (RELEASE_NOT_CONFIRMED) with
    the feedback fields and NEVER detaches -- the stage is never restarted with the carton welded."""
    c, m = _release_controller(release_reissue=True, confirm_timeout_s=1.0)
    c.step(.1, _packet(c, 1.0, contact=True))         # first open, arms the gate
    r = None
    for i in range(60):
        r = c.step(.1, _packet(c, 1.1 + i * 0.1, contact=True))   # weld never releases
        if r.status == 'failed':
            break
    assert r.status == 'failed' and 'RELEASE_NOT_CONFIRMED' in r.reason
    assert 'carton welded' in r.reason and m.payloads == []        # never detached
    assert m.stops == 1                                            # the fault stopped the motion


@need_urdf
def test_controller_on_cw84_confirms_after_drop():
    """The headline regression: the coldwin84 timeline (the first ~several opens dropped) now CONFIRMS once
    the re-emitted open lands, instead of the 8 s never-detach e-stop."""
    c, m = _release_controller(release_reissue=True)
    c.step(.1, _packet(c, 1.0, contact=True))
    for i in range(6):                                # the dropped-open window
        r = c.step(.1, _packet(c, 1.1 + i * 0.1, contact=True))
        assert r.status != 'failed' and r.controls == _GRIPPER_OPEN
    r = c.step(.1, _packet(c, 2.0, contact=False))    # open finally lands
    assert c.phase == 'retreat' and m.payloads == [(False, 'succeeded')]
