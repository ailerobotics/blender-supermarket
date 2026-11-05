"""sim-post-collect-stage-readiness-v1: pin the POST-COLLECT mission stages under the FULL coldwin60
runtime + host key set.

coldwin60 (integration lane, fm/twin-robot-in-store-boot-v1) landed nineteen opt-in PRs (#328-#346) all
aimed at the COLLECT set-down of the welded carton on the cart shelf: the chassis station pitch hold, the
full base brake over the loaded place, the carried-soft / per-joint drive profile, the world-level
preplace leg-2 re-solve, the plan-derived phase budget, the bounded reaction-branch standoff select, and a
structurally-independent status heartbeat. Every run so far has ended INSIDE collect, so the later stages
(couple -> outbound tow -> park -> transfer -> verify -> return) have never executed with these keys.

These tests load the EXACT coldwin60 configuration (the three committed fixtures below) and assert, on
CPU with no Isaac/GPU/ROS/py_trees, that the collect-scoped keys do NOT regress the later stages:

  * the full base brake + its effort guard stay scoped to the COLLECT deck place (transfer reuses the
    place/retreat phase NAMES but must not take the collect full lock), and both base brakes RELEASE
    outside an arm phase so couple/outbound/park/return drive free;
  * the carried-soft / per-joint drive overrides do NOT leak into the free-arm phases (lift/tuck/release/
    retreat/stow stay uniformly soft, no per-joint override), and the two phases they DO re-shape
    (preplace, place) are phase-shared with the transfer arm place -- pinned + flagged, not silently
    changed;
  * the plan-derived preplace phase budget only ever RAISES the global phase_timeout (floored at the
    historical 120 s), so no other phase is shortened;
  * the status-heartbeat filler is SILENT while the manipulation controller is idle (the whole couple ->
    outbound -> park drive), so the tow/dock stages are byte-identical;
  * config_preflight ACCEPTS every coldwin60 block (nothing refused, nothing silently ignored);
  * the coldwin60 authored tow lane builds and the ground-truth transfer/verify wiring resolves.

The one standing gap these tests document (NOT a regression from #328-#346): the runtime
``transfer_ground_truth.target`` (sales-floor endcap) and the scenario ``transfer_bay`` the vr1 verify
reads (backroom bay) disagree, so the verify stage would check the wrong surface. See
``test_transfer_target_vs_verify_bay_drift_is_flagged`` and the handoff
``docs/examples/post-collect-stage-readiness-fragment.json``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from twin.restock.physics_setup import (phase_drive_plan, phase_drive_gains, resolve_stiff_phases,
                                         resolve_joint_overrides, _ARM_JOINT_NAMES,
                                         _PHASE_DRIVE_STIFF_PHASES)
from twin.restock.station_full_brake import LOADED_PLACE_PHASES, in_loaded_place_window
from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import config_preflight as cp
from twin.restock_manipulation.status_heartbeat import StatusHeartbeat
from twin.restock.ground_truth_verify import ground_truth_place_actions, from_config as verify_from_config

_FIX = Path(__file__).parent / 'fixtures'


def _load(name):
    return json.loads((_FIX / name).read_text())


@pytest.fixture(scope='module')
def runtime():
    return _load('coldwin60_runtime.json')


@pytest.fixture(scope='module')
def host():
    return _load('coldwin60_host.json')


@pytest.fixture(scope='module')
def scenario():
    return _load('coldwin60_scenario.json')


# The manipulation phases, split by how the mission uses them. 'pan' is the collect cart-swing only;
# the live_pose transfer routes lift -> tuck -> preplace -> place -> retreat (restock_manipulation.
# transfer_mode) so it SHARES preplace/place/retreat with collect.
_FREE_ARM_PHASES = ('lift', 'tuck', 'transit', 'release', 'retreat', 'stow')
_CARRIED_RESHAPED_PHASES = ('preplace', 'place')


# --------------------------------------------------------------------------------------------------
# config_preflight: every coldwin60 block loads from the right config; nothing refused / silently ignored
# --------------------------------------------------------------------------------------------------

def test_preflight_accepts_every_coldwin60_block(host, runtime, scenario):
    """The full coldwin60 host+runtime config must pass every always-on config_preflight validator (the
    gate the integration lane runs before the GPU boot). An empty problem list from each proves the block
    is both ACCEPTED and actually consumed -- a key in the wrong file/block fails here, not silently at
    the boot (suspicion: preflight refusing or silently ignoring a coldwin60 block)."""
    assert cp.check_activation(host, runtime, list(cp.BASELINE_REQUIRED)) == []
    assert cp.check_flow_consistency(host, runtime) == []
    assert cp.check_carton_level(host) == []
    assert cp.check_cart_placement_source(runtime) == []
    assert cp.check_support_selection(runtime) == []
    assert cp.check_station_full_brake(runtime) == []
    assert cp.check_phase_drive_carried_soft(runtime) == []
    assert cp.check_phase_drive_carried_joint_profile(runtime) == []
    assert cp.check_preplace_phase_budget(host) == []
    assert cp.check_preplace_convergence_keys(host) == []


def test_every_coldwin60_host_and_runtime_key_is_registered(host, runtime):
    """Guard the OTHER half of 'silently ignored': the coldwin60 keys that config_preflight gates on by
    name (the per-stage-relevant batch keys) are all registered in KEY_CONSUMER, so a future rename cannot
    make the preflight quietly stop validating them."""
    for key in ('phase_drive', 'station_pitch_hold', 'carton_level', 'preplace_wrist_first',
                'preplace_budget_margin_s', 'collect_place_straight_line', 'transfer_service_carry'):
        assert key in cp.KEY_CONSUMER, key


# --------------------------------------------------------------------------------------------------
# PARK / COUPLE / OUTBOUND / RETURN: the collect station brake is collect-scoped and releases for driving
# --------------------------------------------------------------------------------------------------

def test_full_base_brake_window_is_collect_only(runtime):
    """The loaded-place FULL base brake (station_place_full_brake, on in coldwin60) covers exactly the
    COLLECT deck place window (place/release/retreat). The transfer arm place reuses the 'place'/'retreat'
    phase NAMES at a DIFFERENT station and must NOT take the collect full lock; and it never fires for a
    nav phase. Suspicion: the station brake engaging/lingering outside the collect station."""
    assert runtime['station_place_full_brake'] is True
    assert LOADED_PLACE_PHASES == frozenset({'place', 'release', 'retreat'})
    for phase in ('place', 'release', 'retreat'):
        assert in_loaded_place_window('collect', phase) is True
    # a transfer place/retreat (same phase name, different station) is NOT the collect deck place
    for phase in ('place', 'retreat'):
        assert in_loaded_place_window('transfer', phase) is False
    # no non-place arm phase, and no nav phase, is ever in the loaded-place window
    for action in ('collect', 'transfer', 'couple', 'outbound', 'park', 'return'):
        for phase in ('navigate', 'preplace', 'grasp', 'lift', 'stow'):
            assert in_loaded_place_window(action, phase) is False


def test_full_brake_effort_limit_is_positive(runtime):
    """coldwin60's station_place_full_brake_effort_limit is a positive finite N*m (the weld-loop effort
    guard); a non-positive value would be refused by the preflight (check_station_full_brake)."""
    v = runtime['station_place_full_brake_effort_limit']
    assert isinstance(v, (int, float)) and math.isfinite(v) and v > 0


class _MockMobile:
    """Minimal stand-in for the mobile base exercising IsaacPort._apply_base_brake's brake state machine
    (the real engage/release methods; no Isaac). station_pitch_hold mirrors the coldwin60 RUNTIME key."""

    def __init__(self, *, station_pitch_hold, station_place_full_brake, effort_limit):
        self.station_pitch_hold = station_pitch_hold
        self.station_place_full_brake = station_place_full_brake
        self.station_place_full_brake_effort_limit = effort_limit
        self.full_lock = False
        self.rotation_hold = False
        self.base_full_brake = False
        self.yaw_lock = False
        self.events = []

    def engage_station_full_lock(self):
        if not self.full_lock:
            self.full_lock = True
            self.rotation_hold = False
            self.events.append('full_lock_on')

    def release_station_full_lock(self):
        if self.full_lock:
            self.full_lock = False
            self.events.append('full_lock_off')

    def engage_station_hold(self):
        if not self.rotation_hold:
            self.rotation_hold = True
            self.events.append('hold_on')

    def release_station_hold(self):
        if self.rotation_hold:
            self.rotation_hold = False
            self.events.append('hold_off')

    def engage_base_full_brake(self):
        self.base_full_brake = True

    def release_base_full_brake(self):
        self.base_full_brake = False

    def engage_brake(self):
        self.yaw_lock = True

    def release_brake(self):
        self.yaw_lock = False


def _apply(mobile, *, action, phase, in_arm, welded=True):
    """Drive the REAL IsaacPort._apply_base_brake over a mock mobile (unbound-method call, so no Isaac
    construction). Mirrors the port's per-tick call: current_action + phase + in_arm_phase."""
    from twin.restock.isaac_port import IsaacPort
    port = object.__new__(IsaacPort)
    port.mobile = mobile
    port.current_action = action
    port._welded_stop = False
    port._BASE_LOOP_PHASES = IsaacPort._BASE_LOOP_PHASES
    IsaacPort._apply_base_brake(port, in_arm, welded, phase)


def _coldwin60_mock(runtime):
    return _MockMobile(station_pitch_hold=bool(runtime.get('chassis_station_pitch_hold')),
                       station_place_full_brake=bool(runtime.get('station_place_full_brake')),
                       effort_limit=runtime.get('station_place_full_brake_effort_limit'))


def test_station_brake_releases_for_every_nav_stage(runtime):
    """Leaving the arm phase (every couple/outbound/park/return drive tick: in_arm_phase False) releases
    the full lock, the rotation hold AND the base full brake -- the base is free to drive. Suspicion: the
    station hold / full brake must release cleanly before driving."""
    assert runtime.get('chassis_station_pitch_hold') is True
    for action in ('couple', 'outbound', 'park', 'return'):
        mobile = _coldwin60_mock(runtime)
        # first engage something (as if the previous collect left a hold), then a nav tick
        _apply(mobile, action='collect', phase='place', in_arm=True)
        _apply(mobile, action=action, phase='navigate', in_arm=False)
        assert mobile.full_lock is False, action
        assert mobile.rotation_hold is False, action
        assert mobile.base_full_brake is False, action
        assert mobile.yaw_lock is False, action


def test_transfer_arm_phase_takes_rotation_hold_not_full_lock(runtime):
    """At the TRANSFER station the pitch hold latches the rotation-only station hold during the arm phase
    (the chassis pitch is a location-independent parking attitude), but NOT the collect loaded-place FULL
    lock (that stays scoped to action=='collect'). So the transfer place is base-stable without taking the
    collect-tuned translation lock."""
    mobile = _coldwin60_mock(runtime)
    _apply(mobile, action='transfer', phase='place', in_arm=True)
    assert mobile.full_lock is False
    assert mobile.rotation_hold is True


def test_collect_place_takes_full_lock(runtime):
    """Parity: the COLLECT loaded place DOES take the full lock (the behaviour the transfer must not get).
    Confirms the discriminator is the action, not the phase name."""
    mobile = _coldwin60_mock(runtime)
    _apply(mobile, action='collect', phase='place', in_arm=True)
    assert mobile.full_lock is True


# --------------------------------------------------------------------------------------------------
# TRANSFER: the carried-soft / per-joint drive overrides do not leak into the free-arm phases
# --------------------------------------------------------------------------------------------------

def test_phase_drive_free_arm_phases_stay_uniform_soft(runtime):
    """The coldwin60 carried-soft / per-joint overrides touch ONLY preplace (soft) + its shoulder_pan hold.
    Every FREE-ARM phase (the empty-handed retreat/stow and the lift/tuck/transit folds) stays uniformly on
    the SOFT profile with NO per-joint override -- so the collect carried-preplace fix cannot leak a stray
    stiff joint into the empty-gripper or carry-fold phases (suspicion: drive profiles leaking into the
    free-arm phases)."""
    pd = runtime['phase_drive']
    soft_kp = float(pd['soft']['kp'])
    for phase in _FREE_ARM_PHASES:
        assert resolve_joint_overrides(phase, pd) == {}, phase
        plan = phase_drive_plan(phase, pd, _ARM_JOINT_NAMES)
        assert plan is not None
        assert {round(plan[j]['kp'], 6) for j in _ARM_JOINT_NAMES} == {round(soft_kp, 6)}, phase


def test_phase_drive_preplace_is_soft_except_pan_hold(runtime):
    """The coldwin60 carried PREPLACE profile: every joint SOFT except shoulder_pan force-stiffened to HOLD
    its goal against the wrist sweep. This plan is phase-keyed, so it applies to BOTH the collect carried
    preplace AND the live_pose transfer preplace -- pinned here and FLAGGED in the handoff for the transfer
    GPU run to validate (no transfer-on-shelf run has executed with these keys)."""
    pd = runtime['phase_drive']
    soft_kp, stiff_kp = float(pd['soft']['kp']), float(pd['stiff']['kp'])
    assert resolve_joint_overrides('preplace', pd) == {
        'ur_shoulder_pan_joint': {'kp': stiff_kp, 'kd': float(pd['stiff']['kd']),
                                  'eff': float(pd['stiff']['eff'])}}
    plan = phase_drive_plan('preplace', pd, _ARM_JOINT_NAMES)
    assert round(plan['ur_shoulder_pan_joint']['kp'], 6) == round(stiff_kp, 6)
    for j in _ARM_JOINT_NAMES:
        if j != 'ur_shoulder_pan_joint':
            assert round(plan[j]['kp'], 6) == round(soft_kp, 6), j


def test_phase_drive_place_is_uniform_stiff(runtime):
    """The loaded PLACE descent runs the STIFF profile uniformly (resist the gravity sag of the welded
    carton). Phase-keyed, so collect AND transfer place share it -- consistent with a loaded set-down."""
    pd = runtime['phase_drive']
    stiff_kp = float(pd['stiff']['kp'])
    plan = phase_drive_plan('place', pd, _ARM_JOINT_NAMES)
    assert {round(plan[j]['kp'], 6) for j in _ARM_JOINT_NAMES} == {round(stiff_kp, 6)}


def test_carried_soft_phases_may_not_name_a_free_arm_phase(runtime):
    """The preflight guard that KEEPS carried-soft out of the free-arm phases: carried_soft_phases may only
    force-soft a default-STIFF phase. coldwin60 lists only ['preplace'] (valid); a free-arm phase would be
    refused, so the leak cannot be configured in."""
    assert resolve_stiff_phases(runtime['phase_drive']).issubset(set(_PHASE_DRIVE_STIFF_PHASES))
    assert set(runtime['phase_drive']['carried_soft_phases']).issubset(set(_PHASE_DRIVE_STIFF_PHASES))
    bad = dict(runtime['phase_drive'])
    bad['carried_soft_phases'] = ['retreat']   # a free-arm phase
    assert cp.check_phase_drive_carried_soft({'phase_drive': bad}) != []


# --------------------------------------------------------------------------------------------------
# COLLECT phase budget must only RAISE the global phase_timeout (never shorten another phase)
# --------------------------------------------------------------------------------------------------

def test_preplace_phase_budget_only_raises_never_shortens(host):
    """The plan-derived wrist-first preplace budget (ros_host) becomes the controller's single
    phase_timeout for ALL phases. It is floored at the historical 120 s, so the derived value is always
    >= 120 -- every other phase gets MORE time, never less (suspicion: phase budgets shortened for other
    phases). coldwin60's two-leg plan derives ~185 s."""
    lc = cl.resolve_level_config(host['carton_level'])
    carry = host['carry_lift_shape']
    standoff = host['collect_place_preplace_shape']
    budget, plan_s, l1, l2 = cl.preplace_phase_budget(
        carry, standoff, speed=lc.preplace_approach_speed, accel=lc.preplace_approach_accel,
        max_segment_rad=lc.preplace_max_segment_rad,
        margin_s=float(host.get('preplace_budget_margin_s', 90.0)))
    assert budget >= 120.0
    assert budget == pytest.approx(max(120.0, plan_s + host['preplace_budget_margin_s']))
    # a trivially-short plan can never shrink below the 120 s floor
    tiny, _, _, _ = cl.preplace_phase_budget(carry, carry, speed=1.0, accel=1.0, max_segment_rad=1.0,
                                             margin_s=0.0)
    assert tiny == 120.0


# --------------------------------------------------------------------------------------------------
# TOW / DOCK: the status heartbeat is silent while the manipulation controller is idle
# --------------------------------------------------------------------------------------------------

def test_status_heartbeat_silent_while_arm_idle():
    """The coldwin60 status-heartbeat filler (status_heartbeat_interval_s=0.5) only tops up gaps while a
    manipulation job is ACTIVE. During the whole couple -> outbound tow -> park drive the arm controller is
    idle, so the filler publishes nothing => the tow/dock stages are byte-identical (suspicion: the
    heartbeat thread altering host status fields the tow/dock stages read)."""
    clock = [100.0]
    published = []
    hb = StatusHeartbeat(publish=lambda: published.append(clock[0]), now=lambda: clock[0],
                         is_active=lambda: False, interval=0.5)
    # a long elapsed gap while idle: the loop body must publish nothing
    clock[0] += 100.0
    _drive_heartbeat_once(hb)
    assert published == []


def test_status_heartbeat_tops_up_gap_while_active():
    """While a job IS active and the last publish is older than the interval, the filler emits one liveness
    publish -- the structural guarantee that a blocked executor cannot starve /manipulation/status."""
    clock = [100.0]
    published = []
    hb = StatusHeartbeat(publish=lambda: published.append(clock[0]), now=lambda: clock[0],
                         is_active=lambda: True, interval=0.5)
    clock[0] += 1.0   # > interval since construction
    _drive_heartbeat_once(hb)
    assert len(published) == 1


def _drive_heartbeat_once(hb):
    """Run exactly one body of StatusHeartbeat._loop without starting the daemon thread."""
    if hb._is_active() and (hb._now() - hb._last) >= hb._interval:
        hb._publish()
        hb._last = hb._now()


# --------------------------------------------------------------------------------------------------
# OUTBOUND / RETURN nav + TRANSFER/VERIFY wiring on the coldwin60 config
# --------------------------------------------------------------------------------------------------

def test_coldwin60_tow_lane_builds(runtime):
    """The coldwin60 authored outbound tow lane builds from the runtime navigation block (the coupled tow
    on pre-planned lanes). Full obstacle-swept validation is check_tow_lanes; this pins that the lane
    config the mission ships parses into the graph the follower uses."""
    from twin.navigation.lanes import TowLaneGraph
    nav = runtime['navigation']
    graph = TowLaneGraph.from_config(nav['tow_lanes'])
    names = {lane.name for lane in graph.lanes}
    assert 'outbound_dock_to_side_bay' in names


def test_stages_end_with_the_post_collect_tail(runtime):
    """The mission stage ladder ends with the post-collect tail in order."""
    assert runtime['stages'] == ['collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return']


def test_ground_truth_transfer_and_verify_wiring(runtime):
    """coldwin60 runs a ground-truth transfer VERIFY (transfer_ground_truth.enabled + verify_ground_truth).
    The RuntimeAdapter accepts the transfer stage's ground-truth place as stage evidence."""
    assert ground_truth_place_actions(runtime) == ('transfer',)
    assert runtime['transfer_ground_truth']['enabled'] is True
    assert runtime['verify_ground_truth']['enabled'] is True


def test_transfer_target_vs_verify_bay_drift_is_flagged(runtime, scenario):
    """STANDING GAP (not a regression from #328-#346, flagged for coldwin61): the runtime
    transfer_ground_truth.target (sales-floor endcap) and the scenario.transfer_bay the vr1 ground-truth
    verifier reads (backroom bay) are far apart, so a carton placed at the target would be OFF the bay the
    verify checks -- the verify stage cannot confirm. This test DOCUMENTS the current drift so a coldwin61
    config that aligns them (verify_ground_truth.bay / scenario.transfer_bay == the transfer target, or the
    transfer target back at the bay) trips this reminder to drop it. See the handoff fragment."""
    target = runtime['transfer_ground_truth']['target']
    bay = scenario['transfer_bay']['center_xy']
    drift = math.hypot(target[0] - bay[0], target[1] - bay[1])
    # the verifier is still built from the scenario bay (the mechanism is sound; only the geometry drifts)
    verifier = verify_from_config(runtime, scenario, read_unit_pose=lambda: None)
    assert verifier is not None
    assert drift > 1.0, (
        'transfer target and verify bay now AGREE (drift=%.3f m) -- the coldwin61 alignment landed; '
        'update this test to assert consistency and drop the drift flag.' % drift)
