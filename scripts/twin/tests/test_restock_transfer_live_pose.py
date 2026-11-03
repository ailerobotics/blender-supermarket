"""RESTOCK_TRANSFER_MODE=live_pose -- CPU proof against the run-356 transfer park pose (no Isaac/GPU).

run-354/356 failed the transfer observe gate 'approach outside reach envelope' (controller REACH_ENVELOPE_FAIL
approach 1.376 / contact 1.336 / lift 1.328 > max_reach 1.3). The transfer SOURCE was the deck-top AABB
centre (isaac_factory._cart_pose_station, RESTOCK_PLACE_ON_DECK) -- 1.27 m behind the arm base at deck level
-- which the side-grip policy reads as the carton TOP (types.Observation.pose), so the grip landed 0.16 m
low; the warm-start carton actually rests on the UPPER shelf (z 0.93) at cart body +0.18 m. The place side
was not gated (host reposition_before_place) but was out too (place 1.41 / preplace 1.59): the wall-3 grip
lands on the bay's FAR side.

Measured run-356 inputs (dataset run-356/, read-only):
  * arm base ur_base_link: nav-trace transfer pose (-2.079, 15.254, yaw -0.5 deg); z 0.50 (BASE_ATT 0.24 +
    arm_mount 0.26, mobile-import.urdf)
  * legacy source (REACH_ENVELOPE_FAIL source=) (-1.27, -0.0, -0.275) in ur_base_link, cart yaw ~0 world
  * bay target (transfer_ground_truth.target) (-2.0, 16.424, 0.73) world -> TRANSFER_GT_REPROJECT base
    (0.069, 1.170, 0.230)
  * warm-start carton (START_PHASE_SHORTCUT): cart body (+0.18, 0), centre z 1.0125, yaw -90 deg world
  * host profile + walls: the repo ros_host_config.json (approach 0.18, wall 3 everywhere)
"""
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
from twin.restock_manipulation import transfer_carry as tc  # noqa: E402
from twin.restock_manipulation import transfer_mode as tm  # noqa: E402
from twin.restock_manipulation.controller import TransferController  # noqa: E402
from twin.restock_manipulation.types import Observation, ProductProfile, Sensors, Target  # noqa: E402
from twin.restock_manipulation.wall_policy import WallCartonPolicy  # noqa: E402

HOST = json.loads((SCRIPTS / 'twin/restock_manipulation/ros_host_config.json').read_text())
PROFILE = ProductProfile(**HOST['profiles'][0])
SHELFBOT = Path.home() / 'workspaces' / 'shelfbot_newgripper'

BASE_356 = (-2.079, 15.254, math.radians(-0.5))
UB_Z = 0.50
LEGACY_SOURCE_356 = (-1.27, -0.0, -0.275)
BAY = (-2.0, 16.424, 0.73)
JOB = dict(job_id='restock-one', sku='juice_orange', quantity=1, source_bin='backstock_00', shelf_id='G3_back_s2')


def yaw_q(yaw):
    return (0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.))


def to_base(world, base=BASE_356):
    dx, dy = world[0] - base[0], world[1] - base[1]
    c, s = math.cos(-base[2]), math.sin(-base[2])
    return (c * dx - s * dy, s * dx + c * dy, world[2] - UB_Z)


def reach(poses):
    return {k: math.dist(v[:3], (0., 0., 0.)) for k, v in poses.items()}


class _FaithfulGeometry:
    """Verbatim shelfbot.motion_planning.geometry.wall_grip (+ primitives.approach_contact with a zero pad
    offset), so the proof also runs where the shelfbot source is absent; cross-checked below when present."""
    @staticmethod
    def wall_grip(tray, placement, wall=0):
        angle = placement.yaw + wall * (math.pi / 2.0)
        half = (tray.tray_size[0] if wall % 2 == 0 else tray.tray_size[1]) / 2.0
        d = half + tray.contact_margin
        c, s = math.cos(angle), math.sin(angle)
        ox, oy = -d * c, -d * s
        rx, ry = -tray.approach_offset_x_m * c, -tray.approach_offset_x_m * s
        qa = (0., 0., math.sin(angle / 2.), math.cos(angle / 2.))
        bx, by, bz, bw = tray.grasp_quat
        ax, ay, az, aw = qa
        quat = [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]
        return ox, oy, rx, ry, quat

    @staticmethod
    def approach_contact(cx, cy, grip_z, g_rx, g_ry, g_quat, pad_offset_tool):
        return [cx + g_rx, cy + g_ry, grip_z, *g_quat], [cx, cy, grip_z, *g_quat]


def _tool():
    return NS(validate_profile=lambda profile: None, panel_size=(0.2604, 0.1082))


def faithful_policy():
    policy = WallCartonPolicy.__new__(WallCartonPolicy)
    policy.tool, policy.walls, policy.lift = _tool(), dict(HOST['walls']), .03
    policy.geometry = policy.primitives = _FaithfulGeometry()
    return policy


@pytest.fixture
def policy():
    if SHELFBOT.is_dir():
        return WallCartonPolicy(str(SHELFBOT), _tool(), walls=HOST['walls'])
    return faithful_policy()


def bay_target(t=1.0):
    return Target(f'shelf:{t}', 'ground_truth', t, 'G3_back_s2', (*to_base(BAY), *yaw_q(-BASE_356[2])),
                  True, (0.7, 0.5, 0.5))


def legacy_source(t=1.0):
    # the deck-top AABB centre, cart yaw ~0 world => +0.5 deg relative to the -0.5 deg base
    return Observation(f'cart:{t}', 'sensor', t, 'juice_orange', 'rollwagen',
                       (*LEGACY_SOURCE_356, *yaw_q(-BASE_356[2])), 1.0)


def live_carton_centre_world():
    # deck centre world (the logged base-frame source mapped back) + the shortcut's cart-body +0.18 m
    bx, by, byaw = BASE_356
    dx = bx + math.cos(byaw) * LEGACY_SOURCE_356[0] + 0.18
    dy = by + math.sin(byaw) * LEGACY_SOURCE_356[0]
    return (dx, dy, 0.9325 + PROFILE.dimensions[2] / 2.0)


def live_source(t=1.0):
    cw = live_carton_centre_world()
    top = to_base((cw[0], cw[1], cw[2] + PROFILE.dimensions[2] / 2.0))
    return Observation(f'cart:{t}', 'sensor', t, 'juice_orange', 'rollwagen',
                       (*top, *yaw_q(-math.pi / 2 - BASE_356[2])), 1.0)


# -- the switch -----------------------------------------------------------------------------------------

def test_transfer_mode_parsing_default_off_and_loud_on_typos():
    assert tm.transfer_mode({}) is None
    for off in ('', '0', 'off', 'none', 'false', ' OFF '):
        assert tm.transfer_mode({tm.ENV: off}) is None
    assert tm.transfer_mode({tm.ENV: 'live_pose'}) == 'live_pose'
    assert tm.transfer_mode({tm.ENV: ' Live_Pose '}) == 'live_pose'
    with pytest.raises(ValueError, match='unknown transfer mode'):
        tm.transfer_mode({tm.ENV: 'livepose'})


def test_stamp_summary_records_the_mode_and_is_a_noop_when_off():
    off = {'state': 'complete', 'note': 'n'}
    assert tm.stamp_summary(dict(off), {}) == off                       # byte-identical when unset
    s = tm.stamp_summary({'state': 'failed', 'note': 'n.'}, {tm.ENV: 'live_pose'})
    assert s['transfer_mode'] == 'live_pose'
    assert s['workarounds'] == ['transfer_mode=live_pose']
    assert 'TRANSFER_MODE=live_pose' in s['note']
    tm.stamp_summary(s, {tm.ENV: 'live_pose'})                           # idempotent
    assert s['workarounds'] == ['transfer_mode=live_pose'] and s['note'].count('TRANSFER_MODE') == 1
    bad = tm.stamp_summary({}, {tm.ENV: 'bogus'})                       # recorded, never hidden/raised
    assert bad['transfer_mode'] == 'invalid' and 'bogus' in bad['transfer_mode_error']


def test_flip_yaw_is_a_180_deg_turn_about_the_box_axis():
    for yaw in (-math.pi / 2, 0.0, 0.3, math.pi):
        x, y, z, w = tm.flip_yaw(yaw_q(yaw))
        assert x == pytest.approx(0.) and y == pytest.approx(0.)
        got = 2 * math.atan2(z, w)
        assert math.cos(got - (yaw + math.pi)) == pytest.approx(1.)
        assert x * x + y * y + z * z + w * w == pytest.approx(1.)


# -- run-356 geometry -----------------------------------------------------------------------------------

def test_faithful_geometry_matches_shelfbot_when_available():
    if not SHELFBOT.is_dir():
        pytest.skip('shelfbot source not on this box')
    real = WallCartonPolicy(str(SHELFBOT), _tool(), walls=HOST['walls'])
    for src in (legacy_source(), live_source()):
        a = real.build(src, bay_target(), PROFILE)
        b = faithful_policy().build(src, bay_target(), PROFILE)
        for k in a:
            assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_run356_legacy_deck_source_reproduces_the_logged_reach_failure(policy):
    r = reach(policy.build(legacy_source(), bay_target(), PROFILE))
    # REACH_ENVELOPE_FAIL max_reach=1.3 bad={'approach': 1.376, 'contact': 1.336, 'lift': 1.328}
    assert r['approach'] == pytest.approx(1.376, abs=2e-3)
    assert r['contact'] == pytest.approx(1.336, abs=2e-3)
    assert r['lift'] == pytest.approx(1.328, abs=2e-3)
    # the ungated place side is out as well: the wall-3 grip lands on the bay's FAR side
    assert r['place'] == pytest.approx(1.41, abs=5e-3)
    assert r['preplace'] > 1.55 and r['retreat'] > 1.55


def test_run356_live_pose_puts_every_station_in_reach_from_the_park_pose(policy):
    poses, choice = tm.near_face_poses(policy, live_source(), bay_target(), PROFILE)
    r = reach(poses)
    assert max(r.values()) <= 1.05 < PROFILE.max_reach                   # >= 0.25 m margin everywhere
    assert r['approach'] == pytest.approx(0.873, abs=5e-3)
    assert r['contact'] == pytest.approx(1.025, abs=5e-3)
    assert r['place'] == pytest.approx(1.019, abs=5e-3)
    # bay place yaw pi: the grip stays on the arm side of the bay; the carton keeps its pick face
    assert choice == dict(source_flipped=False, target_flipped=True, worst_reach=pytest.approx(max(r.values())))
    # the near-face pick approaches from the arm side (+x, through the cart's open mouth)
    assert poses['approach'][0] > poses['contact'][0]


def test_live_pose_needs_both_halves(policy):
    # host near-face alone cannot rescue the deck-centre source (its grip face is never toward the arm) ...
    poses, _ = tm.near_face_poses(policy, legacy_source(), bay_target(), PROFILE)
    assert max(reach(poses)[k] for k in ('approach', 'contact', 'lift')) > PROFILE.max_reach
    # ... and the live source without the near-face place still puts the place side out of reach
    plain = reach(policy.build(live_source(), bay_target(), PROFILE))
    assert max(plain[k] for k in ('preplace', 'place', 'retreat')) > PROFILE.max_reach


def test_near_face_keeps_unflipped_poses_on_a_tie_and_fails_closed():
    class Policy:
        def __init__(self, refuse=()):
            self.refuse = refuse
        def build(self, source, target, profile):
            if (source.pose[5] != 0.0, target.pose[5] != 0.0) in self.refuse:
                raise ValueError('refused')
            return {'place': (0.5, 0.0, 0.0, 0., 0., 0., 1.)}
    src = Observation('s', 'sensor', 1., 'sku', 'loc', (0., 0., 0., 0., 0., 0., 1.), 1.0)
    tgt = Target('t', 'sensor', 1., 'dst', (0., 0., 0., 0., 0., 0., 1.), True, (1., 1., 1.))
    _, choice = tm.near_face_poses(Policy(), src, tgt, PROFILE)
    assert choice['source_flipped'] is False and choice['target_flipped'] is False
    every = {(a, b) for a in (False, True) for b in (False, True)}
    with pytest.raises(ValueError, match='refused'):
        tm.near_face_poses(Policy(refuse=every), src, tgt, PROFILE)


# -- stubbed transfer stage (host controller, host-config staging) --------------------------------------

class Motion:
    def __init__(self):
        self.moves, self.joint_goals, self.straights = [], [], []

    def begin(self, pose, joints, **kw):
        self.moves.append(pose)
        self.straights.append(bool(kw.get('straight_line', False)))

    def begin_joint_goal(self, goal, measured, slow=False, goal_tol=None):
        self.joint_goals.append(tuple(goal))

    def poll(self):
        return 'succeeded'

    def last_fault_was_sim_deadline(self):
        return False

    def stop(self):
        pass

    def update_payload(self, held, profile):
        return True


def host_controller(policy):
    # the host's real staging from the repo ros_host_config.json (ros_host.py wiring)
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    m = Motion()
    c = TransferController(m, {'juice_orange': PROFILE}, pose_policy=policy, transport_joints=stow,
                           contact_straight_line=True, staged_cart_swing=True,
                           cart_pan_joint0=HOST['cart_pan_joint0'], cart_tuck_shape=HOST['cart_tuck_shape'],
                           carry_lift_shape=HOST['carry_lift_shape'], carry_wrist1_cap=HOST['carry_wrist1_cap'],
                           carry_wrist2_min=HOST['carry_wrist2_min'], grasp_track_m=HOST['grasp_track_m'],
                           reposition_before_place=HOST['reposition_before_place'],
                           transfer_ground_truth=True, phase_timeout=120.)
    c.start('transfer', JOB)
    return c, m, stow


def run_stage(c, source, stow, ticks=200):
    """Drive the transfer with run-356-shaped packets; the ground-truth carton centre follows the grip."""
    target = bay_target()
    cart_centre = (*source.pose[:2], source.pose[2] - PROFILE.dimensions[2] / 2.0)
    bay_centre = (*target.pose[:2], target.pose[2] + PROFILE.dimensions[2] / 2.0)
    phases, released, r = [], False, None
    for i in range(ticks):
        t = 1.0 + 0.1 * i
        if c.phase in ('succeeded', 'failed'):
            break
        phases.append(c.phase)
        contact = c.held or (c.phase == 'grasp' and c.issued)
        if c.phase == 'release' and c.issued:
            contact, released = False, True
        lifted = c.held and c.phase not in ('grasp',)
        placed = bay_centre if released else ((*cart_centre[:2], cart_centre[2] + 0.05) if lifted else cart_centre)
        src = Observation(f'cart:{t}', 'sensor', t, 'juice_orange', 'rollwagen', source.pose, 1.0)
        tgt = Target(f'shelf:{t}', 'ground_truth', t, 'G3_back_s2', target.pose, True, target.free_dimensions)
        r = c.step(0.1, Sensors(t, (src,), tgt, True, contact, t, t,
                                stow if c.phase == 'stow' else (0.,) * 6, (), t,
                                time.monotonic() + 100., (*placed, 0., 0., 0., 1.)))
    return phases, r


def test_live_pose_transfer_completes_from_the_run356_park_pose(policy, monkeypatch, capsys):
    # The honest live transfer requires BOTH switches: live_pose (pick reach) + wrist_guard (place branch).
    monkeypatch.setenv(tm.ENV, 'live_pose')
    monkeypatch.setenv(tc.ENV, 'wrist_guard')
    c, m, stow = host_controller(policy)
    phases, r = run_stage(c, live_source(), stow)
    assert r.status == 'succeeded', (c.phase, r.reason)
    # every station (place side included) was reach-gated at observe and passed
    assert 'REACH_ENVELOPE_FAIL' not in capsys.readouterr().out
    # lift -> tuck -> preplace: no TRANSIT hold (no transfer reposition) and no collect-cart pan
    assert 'transit' not in phases and 'pan' not in phases
    seq = [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p]
    assert seq[seq.index('tuck') + 1] == 'preplace'
    assert {'approach', 'contact', 'grasp', 'lift', 'preplace', 'place', 'release', 'retreat', 'verify'} <= set(phases)
    labels = [e for e in r.evidence if e.get('source') == 'transfer_mode']
    assert labels and labels[0]['transfer_mode'] == 'live_pose' and labels[0]['target_flipped'] is True
    assert r.evidence[-1]['verify_mode'] == 'ground_truth'
    # the place the arm was sent to is the bay with the grip on the arm side, within reach
    assert max(math.dist(p[:3], (0, 0, 0)) for p in m.moves) <= 1.05


def test_default_off_reproduces_run356_and_keeps_the_legacy_staging(policy, monkeypatch, capsys):
    monkeypatch.delenv(tm.ENV, raising=False)
    c, _, stow = host_controller(policy)
    assert c.transfer_mode is None
    _, r = run_stage(c, legacy_source(), stow)
    assert r.status == 'failed' and r.reason == 'approach outside reach envelope'
    assert "'approach': 1.376" in capsys.readouterr().out
    # off: a transfer that passes the pick still folds into TRANSIT (the latent hold live_pose removes)
    c2, _, stow = host_controller(policy)
    phases, _ = run_stage(c2, live_source(), stow, ticks=40)
    assert 'transit' in phases and 'preplace' not in phases


def test_collect_is_untouched_by_the_switch(policy, monkeypatch):
    monkeypatch.setenv(tm.ENV, 'live_pose')
    c = TransferController(Motion(), {'juice_orange': PROFILE}, pose_policy=policy,
                           reposition_before_place=True)
    c.start('collect', JOB)
    assert c.transfer_mode == 'live_pose' and not c._live_pose_transfer()
    assert c._repositions_before_place() is True


# -- live_pose requires the wrist_guard: fail fast, never silently run the jam-prone place -------------

def test_live_pose_transfer_without_wrist_guard_fails_fast_with_a_named_reason(policy, monkeypatch, capsys):
    # live_pose fixes the PICK reach but its preplace/place/retreat are seeded-KDL joint-space moves that
    # jam the ur_wrist_2-vs-forearm branch without the guard (audit transfer-preplace-wrist-branch-jam).
    # The controller must refuse FAST with a named reason, not silently drive into the self-collision stop.
    monkeypatch.setenv(tm.ENV, 'live_pose')
    monkeypatch.delenv(tc.ENV, raising=False)
    c, m, stow = host_controller(policy)
    assert c.transfer_mode == 'live_pose' and not c.transfer_carry
    phases, r = run_stage(c, live_source(), stow)
    assert r.status == 'failed'
    assert 'RESTOCK_TRANSFER_CARRY=wrist_guard' in r.reason and 'wrist_2' in r.reason
    # fail fast: it never issued any arm move (no jam-prone preplace reached the planner)
    assert m.moves == [] and m.joint_goals == []
    # it failed at the first transfer step, in observe, before any reach work
    assert set(phases) <= {'observe'}


def test_live_pose_with_wrist_guard_does_not_trip_the_requirement(policy, monkeypatch):
    # The guard present => the requirement is satisfied and the transfer proceeds past observe.
    monkeypatch.setenv(tm.ENV, 'live_pose')
    monkeypatch.setenv(tc.ENV, 'wrist_guard')
    c, _, stow = host_controller(policy)
    assert tc.WRIST_GUARD in c.transfer_carry
    phases, r = run_stage(c, live_source(), stow)
    assert r.status == 'succeeded', (c.phase, r.reason)
    assert 'approach' in phases   # it advanced past observe into the reach


def test_wrist_guard_requirement_is_live_pose_only_backward_compatible(policy, monkeypatch, capsys):
    # The requirement is scoped to live_pose: a DEFAULT-OFF transfer (no mode) is byte-identical and must
    # NOT demand the guard -- it still reaches its legacy reach-envelope failure, not the new named reason.
    monkeypatch.delenv(tm.ENV, raising=False)
    monkeypatch.delenv(tc.ENV, raising=False)
    c, _, stow = host_controller(policy)
    assert c.transfer_mode is None and not c.transfer_carry
    _, r = run_stage(c, legacy_source(), stow)
    assert r.status == 'failed' and r.reason == 'approach outside reach envelope'
    assert 'RESTOCK_TRANSFER_CARRY=wrist_guard' not in r.reason
    # collect with live_pose set is also never subject to the transfer-only requirement
    monkeypatch.setenv(tm.ENV, 'live_pose')
    cc = TransferController(Motion(), {'juice_orange': PROFILE}, pose_policy=policy)
    cc.start('collect', JOB)
    assert not cc._live_pose_transfer()   # collect is exempt; the observe check never fires for it


# -- live_pose place-onto-surface: collision-off straight-line descent (armcam2, 2026-10-22) ----------

def _pose_key(pose):
    return tuple(round(float(x), 6) for x in pose)


def test_live_pose_place_and_retreat_route_collision_off_straight_line(policy, monkeypatch):
    # armcam2: the live_pose transfer PLACE lowers the carton ONTO the bay surface, so the place config
    # brings the gripper tines to within ~13 mm of the TransferBay top -- inside cuMotion's ESDF collision
    # margin -- and a collision-AWARE joint-space plan to it is rejected ('world collision detected').
    # The PREPLACE standoff (+0.14 m) plans cleanly. So the final place descent and the empty-handed
    # retreat must route as the collision-off straight-line seeded insert (begin(straight_line=True)),
    # exactly like the cart-cage place; the preplace stays collision-aware.
    monkeypatch.setenv(tm.ENV, 'live_pose')
    monkeypatch.setenv(tc.ENV, 'wrist_guard')
    c, m, stow = host_controller(policy)
    phases, r = run_stage(c, live_source(), stow)
    assert r.status == 'succeeded', (c.phase, r.reason)
    by_pose = {}
    for pose, straight in zip(m.moves, m.straights):
        by_pose[_pose_key(pose)] = straight            # last begin() of each pose (retries keep the flag)
    flag = lambda phase: by_pose[_pose_key(c.poses[phase])]
    assert flag('preplace') is False                   # collision-aware standoff above the bay
    assert flag('place') is True                       # collision-off descent onto the bay surface
    assert flag('retreat') is True                     # collision-off lift back to the standoff
    assert flag('approach') is False                   # the free-space pick approach stays collision-aware


# A minimal default (transfer_mode OFF) transfer with a reachable in-front deck target, to prove the new
# straight-line routing is live_pose-ONLY and the baseline place/retreat stay collision-aware (byte-identical).
_OFF_PROFILE = ProductProfile('sku-box', (.6, .4, .16), (.5, .5, .5, .5), max_reach=1.3, grasp_mode='top')
_OFF_JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')


def _off_packet(c, t):
    obs = Observation(f'source-{t}', 'test_fixture', t, 'sku-box', c.source, (.4, 0., .2, 0, 0, 0, 1), .99)
    target = Target(f'target-{t}', 'test_fixture', t, c.destination, (-.5, 0., .1, 0, 0, 0, 1), True, (.7, .7, .5))
    contact = c.held or (c.phase == 'grasp' and c.issued)
    if c.phase == 'release' and c.issued:
        contact = False
    return Sensors(t, (obs,), target, True, contact, t, t, (0.,) * 6, (c.source, c.destination), t,
                   time.monotonic() + 100.)


def test_default_off_place_and_retreat_stay_collision_aware_byte_identical(monkeypatch):
    # transfer_mode OFF + no cart-cage: the whole place chain is the baseline collision-aware joint-space
    # move; the new live_pose term never fires, so NO begin() is a straight-line insert.
    monkeypatch.delenv(tm.ENV, raising=False)
    monkeypatch.delenv(tc.ENV, raising=False)
    c = TransferController(Motion(), {'sku-box': _OFF_PROFILE}, fixture_mode=True)
    c.start('transfer', _OFF_JOB)
    assert c.transfer_mode is None and not c._live_pose_transfer()
    m = c.motion
    for i in range(80):
        t = 1.0 + i * .1
        r = c.step(.1, _off_packet(c, t))
        if c.phase in ('succeeded', 'failed') or r.status == 'failed':
            break
    assert c.phase != 'failed' and r.status != 'failed', getattr(c, 'reason', '')
    # the place chain ran (reaching verify means place+retreat already executed) ...
    assert {_pose_key(c.poses['preplace']), _pose_key(c.poses['place']), _pose_key(c.poses['retreat'])} \
        <= {_pose_key(p) for p in m.moves}
    assert not any(m.straights)                                                   # ... none routed straight-line


def test_collect_place_is_not_routed_straight_line_by_the_live_pose_term(policy, monkeypatch):
    # The new term is scoped to a live_pose TRANSFER: a collect (even with live_pose set) is never a
    # live_pose transfer, so its place is unaffected.
    monkeypatch.setenv(tm.ENV, 'live_pose')
    c = TransferController(Motion(), {'juice_orange': PROFILE}, pose_policy=policy, reposition_before_place=True)
    c.start('collect', JOB)
    assert c.transfer_mode == 'live_pose' and not c._live_pose_transfer()


# -- recorded world-model clearance (the diagnosis, as a committed regression fixture) ----------------

# armcam run (dataset twin-robot-in-store-boot-v1/mission/armcam, 2026-10-22): reconstructed on CPU from
# mission.usda + mobile-import.urdf + the shelfbot_gripper cuRobo collision spheres. The TransferBay top
# (the place surface) is at world z 0.73; cuMotion rasterises obstacles into the nvblox ESDF at a 0.02 m
# voxel pitch, so a target config whose nearest arm sphere sits inside roughly one voxel (~0.02 m) of the
# bay box is flagged 'world collision detected'. These are the worst (nearest) arm collision spheres for
# the two place-side configs the host IK produced (both wrist-guarded, w1<=0.65 / w2>=0): the PREPLACE
# standoff clears by 0.143 m (cuMotion planned it); the PLACE config clears by only 0.0126 m (rejected).
_BAY_CENTER = (-2.0, 16.42, 0.365)
_BAY_HALF = (0.35, 0.25, 0.365)
_PREPLACE_WORST = dict(center=(-2.1187, 16.0302, 0.7982), radius=0.0128)   # suction_cups sphere
_PLACE_WORST = dict(center=(-2.1360, 16.1924, 0.7476), radius=0.0050)      # surface_plate_6 tine sphere
_ESDF_VOXEL_M = 0.02


def _signed_dist_to_bay(center, radius):
    lo = [_BAY_CENTER[i] - _BAY_HALF[i] for i in range(3)]
    hi = [_BAY_CENTER[i] + _BAY_HALF[i] for i in range(3)]
    out = [max(lo[i] - center[i], center[i] - hi[i], 0.0) for i in range(3)]
    outside = math.sqrt(sum(d * d for d in out))
    if outside > 0:
        return outside - radius
    inside = -min(min(center[i] - lo[i], hi[i] - center[i]) for i in range(3))
    return inside - radius


def test_recorded_world_model_place_is_in_the_esdf_margin_preplace_is_clear():
    preplace = _signed_dist_to_bay(**_PREPLACE_WORST)
    place = _signed_dist_to_bay(**_PLACE_WORST)
    assert preplace == pytest.approx(0.1427, abs=1e-3)      # collision-aware preplace plans cleanly
    assert place == pytest.approx(0.0126, abs=1e-3)         # place config: only ~13 mm to the bay top
    # the place config is within one ESDF voxel of the bay -> cuMotion rejects a collision-aware plan to
    # it, while the preplace standoff is many voxels clear: this is why place must be collision-off.
    assert place < _ESDF_VOXEL_M < preplace


# -- Isaac capture: the live carton TOP from the handling-unit prim (usd-core, no Isaac) ---------------

def test_isaac_capture_reads_the_live_carton_top(monkeypatch):
    pxr = pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom
    from twin.restock_manipulation.isaac_factory import CollectTargetBridge
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    base = UsdGeom.Xform.Define(stage, '/World/ur_base_link')
    bx = UsdGeom.Xformable(base.GetPrim())
    bx.AddTranslateOp().Set(Gf.Vec3d(BASE_356[0], BASE_356[1], UB_Z))
    bx.AddRotateZOp().Set(math.degrees(BASE_356[2]))
    UsdGeom.Xform.Define(stage, '/RestockScenario')
    unit = UsdGeom.Mesh.Define(stage, '/RestockScenario/HandlingUnit')
    unit.GetPointsAttr().Set([Gf.Vec3f(x, y, z) for z in (-.08, .08) for x, y in
                              ((-.3, -.2), (.3, -.2), (.3, .2), (-.3, .2))])
    unit.GetExtentAttr().Set([Gf.Vec3f(-.3, -.2, -.08), Gf.Vec3f(.3, .2, .08)])
    ux = UsdGeom.Xformable(unit.GetPrim())
    cw = live_carton_centre_world()
    ux.AddTranslateOp().Set(Gf.Vec3d(*cw))
    ux.AddRotateZOp().Set(-90.0)
    monkeypatch.setenv(tm.ENV, 'live_pose')
    bridge = CollectTargetBridge(
        stage=stage, arm_camera=NS(), arm_camera_path='/arm', base_path='/World/ur_base_link',
        profiles={'juice_orange': PROFILE}, motion=NS(parked=lambda: True, collision_time=9.0),
        contact=NS(read=lambda: False), joints=lambda: (0.,) * 6, source_location='backstock_00',
        cart_placement={'location': 'rollwagen', 'load_center_body_xy': (0.18, 0.0), 'support_z_world': 0.225,
                        'free_dimensions': (0.65, 0.45, 0.3)}, cart_root='/Rollwagen')
    assert bridge.transfer_mode == 'live_pose'
    (x, y, z), yaw = bridge._carton_top_station()
    want = live_source().pose
    assert (x, y, z) == pytest.approx(want[:3], abs=1e-4)
    assert math.cos(yaw - 2 * math.atan2(want[5], want[6])) == pytest.approx(1.0)

    # _capture_transfer swaps ONLY the source pose; gates (parked + cart pose + fresh slot) are unchanged
    from twin.restock_camera.shelf_target import build_ground_truth_shelf_target_provider
    bridge.set_shelf_target_provider(build_ground_truth_shelf_target_provider('G3_back_s2', BAY))
    bridge._calls = 1
    monkeypatch.setattr(bridge, '_cart_pose_station', lambda: ((-1.27, 0.0, -0.275), 0.0))
    s = bridge._capture_transfer(5.0)
    obs = s.observations[0]
    assert obs.location == 'rollwagen' and obs.pose[:3] == pytest.approx(want[:3], abs=1e-4)
    assert s.target.pose[:3] == pytest.approx(to_base(BAY), abs=1e-3)
    monkeypatch.setattr(bridge, '_carton_top_station', lambda: None)    # no live carton => no packet
    assert bridge._capture_transfer(5.1) is None
    monkeypatch.delenv(tm.ENV)                                           # off => the deck-top source
    legacy = CollectTargetBridge(
        stage=stage, arm_camera=NS(), arm_camera_path='/arm', base_path='/World/ur_base_link',
        profiles={'juice_orange': PROFILE}, motion=NS(parked=lambda: True, collision_time=9.0),
        contact=NS(read=lambda: False), joints=lambda: (0.,) * 6, source_location='backstock_00',
        cart_placement=bridge.cart_placement, cart_root='/Rollwagen')
    legacy.set_shelf_target_provider(bridge.shelf_provider)
    legacy._calls = 1
    monkeypatch.setattr(legacy, '_cart_pose_station', lambda: ((-1.27, 0.0, -0.275), 0.0))
    assert legacy.transfer_mode is None
    assert legacy._capture_transfer(5.2).observations[0].pose[:3] == (-1.27, 0.0, -0.275)
