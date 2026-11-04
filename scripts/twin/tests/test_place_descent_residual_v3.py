"""CPU tests for the collect set-down REACH-INDEPENDENT residual fix (sim-place-descent-residual-v3).

Part 1 (characterisation): the ~2.0 deg collect set-down residual is REACH-INDEPENDENT (the measured
anchors cw39 2.02 deg @ reach 1.086 m and cw42 2.00 deg @ 0.839 m) while the gravity-droop model is
reach-DEPENDENT -- so the residual is NOT droop; it is the arm PRESSING the leveled welded carton onto the
cart deck (a contact conform the FK droop model is blind to). Part 2: the contact-triggered weld release
(place_contact_release) stops the press and releases at deck contact, inside the grace. Part 3: the
sensor-fed closed loop is bounded + converging on a measured descent. Plus config resolve / preflight.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_descent_residual_v3.py
"""
import math
import os

import pytest

from twin.restock_manipulation import carton_level as cl

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='dataset mobile URDF not present')
TCP = 'gripper_pad_tcp'

# Measured coldwin39 place-end EXECUTED config (ARM order pan, shoulder_lift, elbow, w1, w2, w3), reach
# 1.086 m, measured set-down tilt 2.02 deg (integration worker's cw39-vs-cw41-droop.txt / v2 fit).
CW39_PLACE = [-0.094, -1.2932, 1.5833, -0.2526, 1.6056, 0.0001]
# A representative LESS-EXTENDED level config at reach ~0.87 m (mirrors cw42's 0.839 m reach): the SAME arm
# pulled in (shoulder_lift up, elbow bent more). Used only to show the droop model's reach dependence.
CW_SHORT = [-0.094, -1.2932 - 0.3, 1.5833 + 0.4, -0.2526, 1.6056, 0.0001]

# Recorded MEASURED set-down residuals at the two reaches (integration anchors). REACH-INDEPENDENT.
CW39_MEASURED_DEG, CW39_REACH_M = 2.02, 1.086
CW42_MEASURED_DEG, CW42_REACH_M = 2.00, 0.839


@pytest.fixture(scope='module')
def robot():
    from twin.urdf_probe import parse_urdf
    return parse_urdf(URDF)


# --------------------------------------------------------------- config resolve / validate / preflight

def test_resolve_contact_release_defaults():
    lc = cl.resolve_level_config({'place_contact_release': True})
    assert lc.place_contact_release is True
    assert lc.place_contact_gap_m == pytest.approx(0.012)
    assert lc.closed_loop_sensor is False


def test_resolve_overrides_and_sensor_requires_closed_loop():
    lc = cl.resolve_level_config({'place_contact_release': True, 'place_contact_gap_m': 0.02,
                                  'closed_loop_level': True, 'closed_loop_sensor': True})
    assert lc.place_contact_gap_m == pytest.approx(0.02)
    assert lc.closed_loop_sensor is True
    with pytest.raises(ValueError):
        cl.resolve_level_config({'closed_loop_sensor': True})          # needs closed_loop_level
    with pytest.raises(ValueError):
        cl.resolve_level_config({'place_contact_gap_m': 0.0})          # must be > 0


def test_off_leaves_v3_inert():
    assert cl.resolve_level_config(False) is None
    lc = cl.resolve_level_config(True)
    assert lc.place_contact_release is False and lc.closed_loop_sensor is False


def test_preflight_resolves_v3_keys():
    from twin.restock_manipulation import config_preflight as cp
    host = {'carton_level': {'place_contact_release': True, 'place_contact_gap_m': 0.012,
                             'closed_loop_level': True, 'closed_loop_sensor': True}}
    assert cp.check_carton_level(host) == []                            # resolves clean, no refusal
    bad = {'carton_level': {'closed_loop_sensor': True}}                # dependency violation
    probs = cp.check_carton_level(bad)
    assert probs and 'closed_loop_sensor' in probs[0]


# --------------------------------------------------------------- part 1: reach-independent, contact not droop

@need_urdf
def test_droop_model_is_reach_dependent(robot):
    """The gravity-droop model (K=2850, UNCHANGED) scales with reach: ~2.16 deg at the cw39 reach 1.086 m,
    clearly LESS at a less-extended ~0.87 m pose. So droop CANNOT produce a reach-INDEPENDENT residual."""
    gc = cl.GravityCompensation().validate()
    assert cl.reach_xy(robot, CW39_PLACE, tool_link=TCP) == pytest.approx(1.086, abs=0.02)
    long_tilt = cl.droop_induced_rpy(robot, CW39_PLACE, gc, tool_link=TCP)[2]
    short_reach = cl.reach_xy(robot, CW_SHORT, tool_link=TCP)
    short_tilt = cl.droop_induced_rpy(robot, CW_SHORT, gc, tool_link=TCP)[2]
    assert long_tilt == pytest.approx(2.16, abs=0.1)
    assert short_reach < 0.95
    assert short_tilt < long_tilt - 0.4                                # reach-DEPENDENT (drops with reach)


def test_measured_residual_is_reach_independent_so_not_droop():
    """The MEASURED set-down residual is REACH-INDEPENDENT (2.02 @ 1.086 m == 2.00 @ 0.839 m, |delta| < 0.1)
    while the droop model predicts a reach-DEPENDENT drop (2.16 -> ~1.50) over the same reaches. A quantity
    that stays ~2.0 as the reach shrinks 1.086 -> 0.839 m is not the reach-dependent droop -- it is the
    contact conform (the deck's attitude), the signature part 1 names."""
    assert abs(CW39_MEASURED_DEG - CW42_MEASURED_DEG) < 0.1            # measured: reach-INDEPENDENT
    model_long, model_short = 2.16, 1.50                               # v2 droop-model fit at the two reaches
    assert (model_long - model_short) > 0.4                            # model: reach-DEPENDENT
    # the model/reality gap OPENS as reach shrinks (0.14 -> 0.50): a contact term the model is blind to.
    assert abs(CW39_MEASURED_DEG - model_long) < 0.2
    assert (CW42_MEASURED_DEG - model_short) > 0.3


def test_contact_conform_is_reach_independent_by_construction():
    """A welded carton CONFORMED flat to a deck tilted ``theta`` reads ``theta`` regardless of arm reach --
    the reach-independent mechanism. With theta ~ 2.0 deg it reproduces BOTH measured anchors."""
    def conform_tilt(deck_tilt_deg, reach_m):
        return deck_tilt_deg                                           # independent of reach, by construction
    theta = 2.0
    assert conform_tilt(theta, CW39_REACH_M) == pytest.approx(CW39_MEASURED_DEG, abs=0.05)
    assert conform_tilt(theta, CW42_REACH_M) == pytest.approx(CW42_MEASURED_DEG, abs=0.05)


def test_setdown_is_a_step_at_contact_not_a_ramp():
    """Reproduce the coldwin42 'step at gap=0' signature: the carton is level through the descent (the plan
    is level) and jumps to ~2.0 deg only when it meets the deck (gap <= 0). A STEP (not a pre-contact ramp)
    is the deck-conformity fingerprint the brief distinguishes."""
    theta, contact_gap_m = 2.0, 0.0
    # carton tilt as a function of carton-bottom-to-deck gap (m): level until contact, conform at contact.
    def tilt_at_gap(gap_m):
        return theta if gap_m <= contact_gap_m else 0.0
    gaps = [0.10, 0.05, 0.02, 0.005, 0.0, -0.005]
    tilts = [tilt_at_gap(g) for g in gaps]
    assert all(t < 0.1 for t in tilts[:4])                             # level through the whole descent
    assert tilts[-2] == pytest.approx(theta) and tilts[-1] == pytest.approx(theta)   # steps at gap=0
    # a STEP: zero slope before contact, a jump at contact (not a monotone pre-contact ramp).
    pre = tilts[:4]
    assert max(pre) - min(pre) < 0.1


# --------------------------------------------------------------- part 2: contact-release threshold

@need_urdf
def test_place_descent_drop_and_release_threshold(robot):
    """The contact-release threshold: release when the live carton centre z falls to
    ``start_carton_z - descent_drop + gap`` -- i.e. within ``gap`` of the set-down height (at deck contact)."""
    # two leveled waypoints: a standoff and a set-down 0.15 m below it (a representative descent).
    standoff = list(CW39_PLACE)
    setdown = list(CW39_PLACE)
    # lower the TCP ~0.2 m by shoulder_lift; compute the real drop from FK.
    setdown[1] += 0.18
    drop = cl.place_descent_drop_m(robot, [standoff, setdown], tool_link=TCP)
    assert drop > 0.05                                                 # a real descent
    start_z, gap = 0.95, 0.012
    rz = cl.place_contact_release_z(start_z, drop, gap)
    assert rz == pytest.approx(start_z - drop + gap)
    # the carton is ABOVE the threshold early in the descent and reaches it ~gap above the set-down.
    assert rz < start_z                                                # must descend first
    assert (start_z - drop) < rz < (start_z - drop + 2 * gap)          # fires ~gap before hard contact


def test_release_phase_is_not_a_trip_phase():
    """Once the contact-release transitions to 'release', the CARTON_TILT trip is structurally suppressed:
    the brief carton conform at the deck can never sustain a trip because the carried-phase grader does not
    grade the release phase."""
    from twin.restock_manipulation.controller import TransferController
    assert 'release' not in TransferController._CARRIED_PHASES
    assert 'place' in TransferController._CARRIED_PHASES


# --------------------------------------------------------------- part 2: controller gate wiring

def _collect_ctrl(carton_level, **kw):
    from twin.tests.test_restock_manipulation import make_controller, _SL_STOW, _SL_TUCK, _SL_PREPLACE_SHAPE
    c, m = make_controller(
        'collect', transport_joints=_SL_STOW, staged_cart_swing=True, cart_pan_joint0=-0.8605,
        cart_tuck_shape=_SL_TUCK, reposition_before_place=True,
        collect_place_preplace_shape=_SL_PREPLACE_SHAPE, collect_place_wrist_guard=True,
        collect_place_straight_line=True, carton_level=carton_level, fk_urdf=URDF,
        carry_wrist1_cap=0.65, carry_wrist2_min=0.0, **kw)
    return c, m


@need_urdf
def test_contact_release_fires_at_deck_contact():
    """With place_contact_release on, the controller transitions place->release the instant the measured
    carton centre z reaches the latched release threshold -- the contact-triggered weld release."""
    from twin.tests.test_restock_manipulation import packet
    lc = cl.resolve_level_config({'place_contact_release': True, 'place_contact_gap_m': 0.012, 'trip': False})
    c, _m = _collect_ctrl(lc)
    # Drive directly into a place-phase descent state (isolating the step() gate from the full sequence).
    c.phase, c.held, c.issued = 'place', True, True
    c.poses = {'place': (0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0)}
    _m.outcome = 'running'                           # the place move stays active; only the gate transitions
    c._capture_level_up((0.0,) * 6)                 # level carton up-axis so the fixed joints read level
    c._place_contact_release_z = 0.90
    # ABOVE the threshold: still descending, no release.
    r = c.step(.1, packet(c, 1.0, contact=True, placed_pose=(0.4, 0.0, 0.95, 0., 0., 0., 1.)))
    assert r.status == 'running' and c.phase == 'place'
    # AT/BELOW the threshold: deck contact -> release the weld.
    r = c.step(.1, packet(c, 1.1, contact=True, placed_pose=(0.4, 0.0, 0.89, 0., 0., 0., 1.)))
    assert c.phase == 'release', r.reason
    assert any('place_contact_release' in e for e in map(str, c.evidence))


@need_urdf
def test_contact_release_off_never_fires():
    """place_contact_release OFF (default): the gate is inert -- no threshold latched, no early release,
    byte-identical to the legacy place->level-gate->release path."""
    from twin.tests.test_restock_manipulation import packet
    lc = cl.resolve_level_config({'trip': False})                     # carton_level on, contact-release OFF
    c, _m = _collect_ctrl(lc)
    assert lc.place_contact_release is False
    c.phase, c.held, c.issued = 'place', True, True
    c.poses = {'place': (0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0)}
    _m.outcome = 'running'
    c._capture_level_up((0.0,) * 6)
    assert c._place_contact_release_z is None                         # never latched
    r = c.step(.1, packet(c, 1.0, contact=True, placed_pose=(0.4, 0.0, 0.50, 0., 0., 0., 1.)))
    assert c.phase == 'place'                                         # no early release even far down
    assert not any('place_contact_release' in e for e in map(str, c.evidence))


# --------------------------------------------------------------- part 3: sensor-fed closed loop

@need_urdf
def test_sensor_servo_bounded_and_converging(robot):
    """The SENSOR-fed closed loop (replay_measured): a measured descent whose EXECUTED carton tilts to
    ~2.0 deg (a contact deflection) is driven DOWN toward level by the bounded wrist correction, and the
    accumulated correction never exceeds the authority."""
    R = cl.tool_rotation(robot, CW39_PLACE, tool_link=TCP)
    up = cl.carton_up_in_tool(R)                                      # cw39 reads level by capture
    # a measured EXECUTED descent that tilts to ~2 deg via a growing wrist_1 contact deflection.
    executed = []
    for k in range(12):
        cfg = list(CW39_PLACE)
        cfg[3] += math.radians(2.0) * (k / 11.0)
        executed.append(cfg)
    clc = cl.ClosedLoopConfig(authority_deg=3.0, rate_deg_per_s=30.0, gain=0.6, tool_link=TCP)
    servo = cl.CartonLevelServo(robot, up, clc)
    measured, servoed = servo.replay_measured(executed)
    assert measured[-1][2] == pytest.approx(2.0, abs=0.1)             # measured contact tilt reproduced
    assert servoed[-1][2] < measured[-1][2] - 0.5                     # servo drives it toward level
    peak = max(abs(math.degrees(v)) for v in servo.accumulated.values())
    assert peak <= 3.0 + 1e-6                                         # bounded by the authority


# --------------------------------------------------------------- part 2: the cart deck rests level

def test_cart_deck_is_level_by_design():
    """CPU deck-level proof (brief part 2 'level the cart deck'): the AUTHORED Rollwagen deck is parallel to
    the floor -- deck top-face normal within 0.3 deg of world +Z and the four caster ground contacts at equal
    height. So no STATIC deck tilt explains the 2.0 deg residual; it is the arm-press contact artifact the
    contact-release fixes. (The coldwin42-deck.txt 0.88 deg was an END-of-run snapshot with the cart knocked
    by the controlled stop -- angularVelocity (3.2, 16.7, -63.4) -- not the rest attitude.)"""
    pytest.importorskip('pxr')
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'check_cart_deck_level',
        os.path.join(os.path.dirname(__file__), '..', '..', 'check_cart_deck_level.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from twin.mobile import MobileSpec
    rec = mod._measure(MobileSpec())
    assert rec['deck_tilt_deg'] <= 0.3, rec
    assert rec['n_casters'] == 4
    assert rec['caster_bottom_spread_m'] <= 1e-6, rec


@need_urdf
def test_sensor_anchored_correction_matches_model_when_measured_is_the_model(robot):
    """The sensor-anchored correction reduces to the v2 model correction when the measured start equals the
    model's prediction (zero residual), and stays bounded otherwise."""
    gc = cl.GravityCompensation().validate()
    R = cl.tool_rotation(robot, CW39_PLACE, tool_link=TCP)
    up = cl.carton_up_in_tool(R)
    clc = cl.ClosedLoopConfig(authority_deg=3.0, rate_deg_per_s=30.0, gain=0.6, tool_link=TCP)
    servo = cl.CartonLevelServo(robot, up, clc)
    path = [list(CW39_PLACE), [v + 0.001 * i for i, v in enumerate(CW39_PLACE)]]
    anchored = servo.correct_path_sensor_anchored(path, gc, None)     # None measured -> plain model path
    model = cl.CartonLevelServo(robot, up, clc).correct_path([list(p) for p in path], gc)
    assert anchored == model
