"""CPU tests for sim-release-threshold-carton-frame-v1.

Two fixes, from coldwin83/85/86/87 (all the same deck/shelf collect set-down config):

  1. THRESHOLD FRAME. ``Sensors.placed_pose[2]`` / ``carton_z`` is the carton CENTRE (types.py). coldwin87's
     ``PLACE_CONTACT_RELEASE_ARM start_carton_z=0.6422 descent_drop=0.0907 gap=0.0120 -> release at
     carton_z<=0.5635`` expressed the carton BOTTOM frame, so with PR 373's startup-stall suppression on the
     welded carton rested on the shelf at carton_z 0.6422 (half - gap = 0.078 m ABOVE 0.5635) and the arm
     pressed it in to PATH_TOLERANCE_VIOLATED -> HOST_FAULT_ESTOP. ``place_contact_release_z_centre`` adds the
     carton half-height so the trigger fires at the true contact (carton BOTTOM at support_top + gap).

  2. FREE-AND-RESTING + CLEARANCE LIFT. coldwin85/86 released at the set-down and the retreat's first motion
     swept the barely-rested carton off the shelf (verify: carton not resting on deck, cw85 bottom_gap=-0.954,
     cw86 -0.815); coldwin83 on the same config happened to rest. ``release_free.ReleaseFreeGate`` holds until
     the carton is proven free+resting, clears the gripper straight up, and refuses RELEASE_NOT_FREE if the
     carton follows the gripper up.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
        scripts/twin/tests/test_release_threshold_carton_frame_v1.py
"""
from types import SimpleNamespace as NS

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.release_free import ReleaseFreeGate, ReleaseFreeDecision

CARTON_H = 0.18              # handling_unit.dimensions_m height (single source of truth)
HALF = CARTON_H / 2.0        # 0.09


# ===================================================================== 1. THRESHOLD FRAME

def test_centre_frame_adds_the_carton_half_height():
    """The centre-frame threshold is the bottom-frame threshold plus the carton half-height."""
    bottom = cl.place_contact_release_z(0.6422, 0.0907, 0.0120)
    centre = cl.place_contact_release_z_centre(0.6422, 0.0907, 0.0120, CARTON_H)
    assert centre == pytest.approx(bottom + HALF, abs=1e-9)


def test_coldwin87_threshold_now_fires_at_the_physical_rest():
    """cw87: the welded carton plateaued at carton_z (centre) 0.6422 while the arm pressed. The legacy
    bottom-frame threshold 0.5635 sat half - gap = 0.078 m BELOW that rest, so it never fired; the
    centre-frame threshold fires AT/above the measured rest."""
    start_z, drop, gap, rest = 0.6422, 0.0907, 0.0120, 0.6422
    legacy = cl.place_contact_release_z(start_z, drop, gap)
    fixed = cl.place_contact_release_z_centre(start_z, drop, gap, CARTON_H)
    assert legacy == pytest.approx(0.5635, abs=5e-4)
    assert fixed == pytest.approx(0.6535, abs=5e-4)
    # the gap between the legacy threshold and the measured rest is half - gap (the centre-vs-bottom mistake)
    assert (rest - legacy) == pytest.approx(HALF - gap, abs=2e-3)   # ~0.078 m
    # legacy never fires (rest stays above it); fixed fires (rest is at/below it)
    assert rest > legacy                      # carton_z <= legacy is never reached -> press forever
    assert rest <= fixed                      # carton_z <= fixed is satisfied at the measured rest


@pytest.mark.parametrize('start_z,drop', [
    (0.6422, 0.0907),   # coldwin87
    (0.6257, 0.0742),   # coldwin83
    (0.6251, 0.0743),   # coldwin85
    (0.6251, 0.0741),   # coldwin86
])
def test_replay_release_fires_at_measured_rest(start_z, drop):
    """Replay the PLACE_CONTACT_RELEASE_ARM numbers of each cold run: the centre-frame threshold is >= the
    measured carton plateau (carton_z ~= start_z, the carton rested at the standoff height), so the geometric
    z trigger fires at the rest, while the legacy bottom-frame threshold stays below it (never fires)."""
    gap = 0.0120
    legacy = cl.place_contact_release_z(start_z, drop, gap)
    fixed = cl.place_contact_release_z_centre(start_z, drop, gap, CARTON_H)
    rest = start_z                                   # the PLACE_RELEASE_TICK plateau ~= the standoff centre
    assert rest > legacy                             # legacy never fires -> arm presses
    assert rest <= fixed                             # fixed fires at the measured rest
    assert fixed == pytest.approx(legacy + HALF, abs=1e-9)


def test_centre_frame_rejects_bad_carton_height():
    for bad in (-0.1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            cl.place_contact_release_z_centre(0.64, 0.09, 0.012, bad)


# ===================================================================== 2. FREE-AND-RESTING GATE (pure)

def _free_gate(**kw):
    kw.setdefault('settle_ticks', 3)
    kw.setdefault('stable_eps_m', 0.005)
    kw.setdefault('rest_gap_m', 0.03)
    kw.setdefault('settle_window_ticks', 20)
    kw.setdefault('lift_dz_m', 0.06)
    kw.setdefault('follow_frac', 0.5)
    return ReleaseFreeGate(**kw)


def test_settle_lifts_once_stable_and_resting():
    g = _free_gate(settle_ticks=3)
    # first tick has no prev z -> not stable; then 3 consecutive stable+resting ticks -> lift
    outs = [g.settle(carton_z=0.500, bottom_gap=0.004) for _ in range(5)]
    actions = [o.action for o in outs]
    assert actions[0] == 'settle'                       # no prev z yet
    assert 'lift' in actions
    lift = next(o for o in outs if o.action == 'lift')
    assert g.lift_baseline == pytest.approx(0.500, abs=1e-9)
    assert lift.settle_ticks >= 3


def test_settle_not_resting_refuses_release_not_free():
    """A carton whose bottom never comes to rest on the support (big |bottom_gap|) is NOT free+resting:
    after the bounded window the gate refuses (the retreat must not sweep a cradled/fallen carton)."""
    g = _free_gate(settle_ticks=3, settle_window_ticks=8)
    last = None
    for _ in range(8):
        last = g.settle(carton_z=0.500, bottom_gap=-0.20)   # 0.20 m below the support: not resting
    assert last.action == 'refuse' and 'never came to rest' in last.reason


def test_settle_moving_carton_resets_the_streak():
    g = _free_gate(settle_ticks=3, settle_window_ticks=50)
    # resting but jittering more than stable_eps each tick -> streak never builds -> no lift within a few ticks
    zs = [0.50, 0.52, 0.50, 0.52, 0.50, 0.52]
    actions = [g.settle(carton_z=z, bottom_gap=0.0).action for z in zs]
    assert 'lift' not in actions


def test_follow_proceeds_when_carton_stays():
    g = _free_gate()
    for _ in range(4):
        d = g.settle(carton_z=0.500, bottom_gap=0.0)
    assert d.action == 'lift' and g.lift_baseline == pytest.approx(0.500)
    # the gripper lifted +0.06 but the carton stayed at ~0.500 -> free
    out = g.follow(carton_z=0.5005)
    assert out.action == 'proceed' and out.rise == pytest.approx(0.0005, abs=1e-6)


def test_follow_refuses_when_carton_follows_the_gripper():
    g = _free_gate(lift_dz_m=0.06, follow_frac=0.5)
    for _ in range(4):
        g.settle(carton_z=0.500, bottom_gap=0.0)
    # the carton rose ~the full lift with the gripper -> still held -> RELEASE_NOT_FREE
    out = g.follow(carton_z=0.560)
    assert out.action == 'refuse' and 'with the gripper' in out.reason


def test_follow_proceeds_without_a_baseline():
    g = _free_gate()
    out = g.follow(carton_z=0.5)          # no settle/lift ran -> no baseline -> cannot prove following
    assert out.action == 'proceed'


def test_gate_validates_params():
    for bad in ({'stable_eps_m': 0}, {'rest_gap_m': -1}, {'lift_dz_m': 0}, {'follow_frac': 0},
                {'follow_frac': 1.5}):
        with pytest.raises(ValueError):
            _free_gate(**bad)


# ===================================================================== 3. CONFIG resolve / byte-identical off

def test_release_free_check_off_by_default():
    c = cl.resolve_level_config(True)
    assert c.release_free_check is False
    # an explicit block that does not mention it stays off
    c2 = cl.resolve_level_config({'place_release_on_contact': True})
    assert c2.release_free_check is False


def test_release_free_params_resolve_and_validate():
    c = cl.resolve_level_config({
        'release_free_check': True, 'release_free_settle_ticks': 4, 'release_free_stable_eps_m': 0.004,
        'release_free_rest_gap_m': 0.02, 'release_free_settle_window_ticks': 40,
        'release_free_lift_dz_m': 0.06, 'release_free_follow_frac': 0.6})
    assert c.release_free_check is True
    assert c.release_free_settle_ticks == 4 and c.release_free_lift_dz_m == pytest.approx(0.06)
    assert c.release_free_follow_frac == pytest.approx(0.6)


def test_release_free_window_must_cover_settle():
    with pytest.raises(ValueError):
        cl.resolve_level_config({'release_free_check': True, 'release_free_settle_ticks': 10,
                                 'release_free_settle_window_ticks': 3})


# ===================================================================== 4. CONTROLLER sub-state wiring

def _free_controller(stage='settle', gate=None, poll='running', liftoff=(0, 0, 0, 0, 0, 0)):
    from twin.restock_manipulation.controller import TransferController
    c = NS()
    c._release_free = gate
    c._rf_stage = stage
    c._released_detached = True
    c.profile = NS(dimensions=(0.45, 0.30, CARTON_H))
    c.target_surface = (1.11, 0.08, 0.448)
    c._transitions = []
    c._transition = lambda ph: (c._transitions.append(ph), setattr(c, 'phase', ph))[1]
    c._result = lambda controls=(): ('RESULT', tuple(controls))
    c._fail = lambda r: ('FAIL', r)
    c.evidence = []
    c._begun = []
    c.motion = NS(begin_joint_goal=lambda *a, **k: c._begun.append((a, k)),
                  poll=lambda: poll)
    c._retreat_lifted = False
    c._collect_retreat_liftoff_goal = lambda measured: liftoff
    c._begin = TransferController._begin_retreat_after_detach.__get__(c, type(c))
    c._step = TransferController._release_free_step.__get__(c, type(c))
    return c


def _sensors(carton_z=0.538, joints=(0,) * 6):
    # carton bottom = carton_z - half; support top 0.448 -> bottom_gap = carton_z - 0.09 - 0.448
    return NS(placed_pose=(1.11, 0.08, carton_z, 0, 0, 0, 1), joint_positions=joints)


def test_begin_retreat_after_detach_off_transitions_straight_to_retreat():
    c = _free_controller(gate=None)
    c._begin(1.0)
    assert c._transitions == ['retreat']       # byte-identical with the old self._transition('retreat')


def test_begin_retreat_after_detach_on_holds_for_the_free_check():
    c = _free_controller(gate=_free_gate())
    c._begin(1.0)
    assert c._transitions == []                 # does NOT transition yet
    assert c._released_detached is True and c._rf_stage == 'settle'


def test_step_skips_without_measured_pose():
    c = _free_controller(gate=_free_gate())
    res = c._step(1.0, NS(placed_pose=None, joint_positions=(0,) * 6))
    assert c._transitions == ['retreat'] and c._released_detached is False


def test_step_settle_holds_then_lifts_then_retreats():
    gate = _free_gate(settle_ticks=3)
    c = _free_controller(stage='settle', gate=gate)
    # carton resting (bottom_gap ~ 0) at a stable z: settle, settle, ... then lift
    s = _sensors(carton_z=0.538)                # bottom_gap = 0.538 - 0.09 - 0.448 = 0.0
    actions = []
    for _ in range(5):
        res = c._step(2.0, s)
        actions.append((c._rf_stage, res))
        if c._rf_stage == 'lift':
            break
    assert c._rf_stage == 'lift' and c._begun            # a vertical clearance lift was commanded
    assert c._retreat_lifted is True                     # the retreat will skip its own lift-off
    # now the lift completes with the carton having stayed -> retreat
    c.motion.poll = lambda: 'succeeded'
    res = c._step(2.5, _sensors(carton_z=0.5385))
    assert c._transitions == ['retreat'] and c._released_detached is False


def test_step_lift_refuses_when_carton_follows():
    gate = _free_gate(settle_ticks=1, lift_dz_m=0.06, follow_frac=0.5)
    c = _free_controller(stage='settle', gate=gate)
    # settle to lift
    for _ in range(3):
        c._step(2.0, _sensors(carton_z=0.538))
        if c._rf_stage == 'lift':
            break
    assert c._rf_stage == 'lift'
    c.motion.poll = lambda: 'succeeded'
    # the carton rose with the gripper -> RELEASE_NOT_FREE
    res = c._step(2.6, _sensors(carton_z=0.538 + 0.06))
    assert res[0] == 'FAIL' and 'RELEASE_NOT_FREE' in res[1]


def test_step_settle_refuses_when_never_resting():
    gate = _free_gate(settle_ticks=3, settle_window_ticks=5)
    c = _free_controller(stage='settle', gate=gate)
    s = _sensors(carton_z=0.80)                  # bottom_gap = 0.80 - 0.09 - 0.448 = 0.262 (not resting)
    res = None
    for _ in range(6):
        res = c._step(2.0, s)
    assert res[0] == 'FAIL' and 'RELEASE_NOT_FREE' in res[1]


def test_step_lift_unsolved_falls_through_to_retreat():
    gate = _free_gate(settle_ticks=1)
    c = _free_controller(stage='settle', gate=gate, liftoff=None)  # no straight-up IK solution
    for _ in range(3):
        res = c._step(2.0, _sensors(carton_z=0.538))
        if c._transitions:
            break
    assert c._transitions == ['retreat'] and c._released_detached is False


# ===================================================================== rest-at-seat free proof
# sim-place-release-rest-at-seat-v1: a carton the monotonic-z descent places EXACTLY at the seat (no drop,
# no carton-vs-support contact) is proved free via |bottom_gap| <= rest_at_seat_m, ONLY when the release
# coupling diag is CLEAN (release_diag_clean True). A still-bound carton presents a dirty diag.
def _drive_free_proof(gate, *, carton_z, bottom_gap, support_contact, release_diag_clean, ticks):
    last = None
    for _ in range(ticks):
        last = gate.free_proof(carton_z=carton_z, bottom_gap=bottom_gap,
                               support_contact=support_contact, release_diag_clean=release_diag_clean)
    return last


def test_rest_at_seat_accepts_a_diag_clean_carton_at_the_seat():
    g = _free_gate(settle_ticks=3, positive_free=True, rest_at_seat_m=0.002)
    # exactly at the seat (bottom_gap -0.3 mm, within 2 mm), no contact, no drop, but the diag is CLEAN
    dec = _drive_free_proof(g, carton_z=0.5306, bottom_gap=-0.0003, support_contact=False,
                            release_diag_clean=True, ticks=6)
    assert dec.action == 'lift'
    assert 'rest-at-seat' in dec.reason


def test_rest_at_seat_refuses_a_dirty_diag_carton_at_the_seat():
    g = _free_gate(settle_ticks=3, settle_window_ticks=8, positive_free=True, rest_at_seat_m=0.002)
    # same rest geometry, but a DIRTY diag (a bound carton held rigid at the seat) -> never lifts, refuses
    dec = _drive_free_proof(g, carton_z=0.5306, bottom_gap=-0.0003, support_contact=False,
                            release_diag_clean=False, ticks=10)
    assert dec.action == 'refuse'
    assert 'no-support-contact' in dec.reason


def test_rest_at_seat_off_is_byte_identical_refuse():
    # rest_at_seat_m None (off): the clean diag alone does NOT accept -> refuses like the pre-change gate
    g = _free_gate(settle_ticks=3, settle_window_ticks=8, positive_free=True)
    dec = _drive_free_proof(g, carton_z=0.5306, bottom_gap=-0.0003, support_contact=False,
                            release_diag_clean=True, ticks=10)
    assert dec.action == 'refuse'
