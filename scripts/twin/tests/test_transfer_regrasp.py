"""CPU tests for the cage-aware re-grasp + reposition place-park (sim-transfer-regrasp-reposition-v1).

coldwin93 died at the TRANSFER PICK: cuMotion could not inverse-kinematics-solve the near-face SIDE reach
into the rollwagen +x cage mouth (INVERSE_KINEMATICS_FAILURE x10 -> PLAN_TOTAL_ATTEMPTS_CAP ->
HOST_FAULT_ESTOP phase=approach), and the observed store-shelf place target lay > 2 m from the parked base.
These tests pin the fix:

  * the re-grasp PLANNER INPUTS -- the parked cart-body -> ur_base_link transform (from the parked prim, not
    the coupled hitch), the PR 350 world_obstacles built at that pose, and the PR 376 pad-down geometry
    REVERSED into a through-the-mouth re-grasp whose approach/descent/lift/withdraw CLEAR the cage (sweep
    tool), with the carried withdraw inside the 2.0 deg level bound;
  * the REPOSITION goal -- the place-park pose computed from the endcap target + the arm reach (target within
    ~0.9 m of the base, facing the shelf) and the place + level descent onto the shelf top (0.73) in reach;
  * a coldwin93 REPLAY: the pad-down approach is now plannable (clears the cage) and the place target is in
    reach from the new park;
  * the BOOT-PATH wiring: the HOST key flows through the ros_host controller-build path and makes the new
    stages run (pad-down pick + reposition), and OFF => byte-identical;
  * config_preflight registers the HOST key.

No Isaac / ROS / GPU. The cage-sweep tests are guarded on usd-core (the authored rollwagen) + the dataset
URDF/XRDF; the pure-geometry + controller-wiring tests need neither.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_transfer_regrasp.py
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock_manipulation import transfer_regrasp as tr  # noqa: E402
from twin.restock_manipulation import transfer_mode as tm  # noqa: E402
from twin.restock_manipulation.controller import TransferController  # noqa: E402
from twin.restock_manipulation.types import Observation, ProductProfile, Sensors, Target  # noqa: E402
from twin.restock_manipulation.wall_policy import WallCartonPolicy  # noqa: E402

HOST = json.loads((SCRIPTS / 'twin/restock_manipulation/ros_host_config.json').read_text())
PROFILE = ProductProfile(**HOST['profiles'][0])
SHELFBOT = Path.home() / 'workspaces' / 'shelfbot_newgripper'
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'

# coldwin93 geometry (nav-trace.jsonl terminal transfer poses + the observe line).
CW93_ROBOT_PARK = (-4.294, 6.987, math.radians(-89.8))      # ur_base_link world (x, y, yaw)
CW93_CART_PARK = (-4.2986, 8.2593, math.radians(-89.81))    # parked cart prim world (x, y, yaw)
ENDCAP = (-3.05, 4.826, 0.73)                               # scenario.initial_staging.target_surface_pose_world
CARTON_H = 0.18

# A pre-baked cart-frame collider subset (so the no-usd tests exercise the world-obstacle shape without pxr):
# the two +x mouth posts (CAGE) + the deck (SUPPORT) + the drawbar (other).
FAKE_BOXES = {
    'post_2': [[0.367, -0.333, 0.200], [0.393, -0.307, 1.650]],
    'post_3': [[0.367, 0.307, 0.200], [0.393, 0.333, 1.650]],
    'deck': [[-0.400, -0.340, 0.175], [0.400, 0.340, 0.225]],
    'drawbar': [[0.400, -0.020, 0.260], [0.650, 0.020, 0.300]],
}

BASE_356 = (-2.079, 15.254, math.radians(-0.5))
UB_Z = 0.50
LEGACY_SOURCE_356 = (-1.27, -0.0, -0.275)
BAY = (-2.0, 16.424, 0.73)
JOB = dict(job_id='restock-one', sku='juice_orange', quantity=1, source_bin='backstock_00', shelf_id='G3_back_s2')


def _usd_available():
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------- 1) cart pose in the base frame
def test_cart_pose_in_base_from_the_parked_prim():
    cp = tr.cart_pose_in_base(CW93_ROBOT_PARK, CW93_CART_PARK)
    # the parked cart sits ~1.27 m BEHIND the base (x negative), its floor 0.5 m below the arm base, aligned
    assert cp[0] == pytest.approx(-1.272, abs=5e-3)
    assert cp[1] == pytest.approx(0.0, abs=5e-3)
    assert cp[2] == pytest.approx(-0.5, abs=5e-3)
    assert abs(cp[3]) < math.radians(1.0)          # the tow aligned the cart yaw with the base


def test_cart_pose_in_base_round_trips_a_simple_case():
    # robot at origin facing +x, cart 2 m ahead facing +x => cart is at base (+2, 0, 0, 0)
    cp = tr.cart_pose_in_base((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), arm_base_z=0.0, cart_base_z=0.0)
    assert cp == pytest.approx((2.0, 0.0, 0.0, 0.0), abs=1e-9)


# --------------------------------------------------------------------- 2) parked-cart world obstacles (PR 350)
def test_parked_cart_world_obstacles_shape_from_prebaked_boxes():
    cp = tr.cart_pose_in_base(CW93_ROBOT_PARK, CW93_CART_PARK)
    specs = tr.parked_cart_world_obstacles(cp[:4], collider_boxes=FAKE_BOXES)
    ids = [s['id'] for s in specs]
    assert 'rollwagen_cage' in ids                 # the two posts group into one cage obstacle
    assert 'rollwagen_deck' in ids                 # the support stays its own box
    assert 'rollwagen_drawbar' not in ids          # the tow hitch is dropped by default (overlaps the arm)
    # exclude=() keeps everything
    assert 'rollwagen_drawbar' in [s['id'] for s in
                                   tr.parked_cart_world_obstacles(cp[:4], collider_boxes=FAKE_BOXES, exclude=())]
    # every spec is a {id, boxes:[{size,center,rpy}]} ready for config['world_obstacles']
    for s in specs:
        assert set(('id', 'boxes')) <= set(s)
        for b in s['boxes']:
            assert set(('size', 'center', 'rpy')) <= set(b) and len(b['size']) == 3


_JOINTS = ('ur_shoulder_pan_joint', 'ur_shoulder_lift_joint', 'ur_elbow_joint',
           'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint')


@pytest.mark.skipif(not (_usd_available() and _URDF.is_file() and _XRDF.is_file()),
                    reason='usd-core or dataset URDF/XRDF absent')
def test_parked_cart_world_obstacles_clear_the_arm_at_rest():
    """A world obstacle overlapping the plan-START config rejects EVERY plan (INVALID_INITIAL_CSPACE,
    firstmate 295). Prove the published parked-cart cage clears the arm at its zero + tucked stow starts
    (the drawbar tow hitch -- which overlaps the arm base -- is excluded by default)."""
    cp = tr.cart_pose_in_base(CW93_ROBOT_PARK, CW93_CART_PARK)
    specs = tr.parked_cart_world_obstacles(cp[:4])
    assert 'rollwagen_drawbar' not in [s['id'] for s in specs]        # the tow hitch is dropped by default
    stow = dict(zip(_JOINTS, (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)))
    out = tr.check_regrasp_obstacles_clear_arm(specs, str(_URDF), str(_XRDF), configs=[{}, stow])
    assert out and min(out.values()) > 0.0         # raises if any obstacle intersects the arm


@pytest.mark.skipif(not (_usd_available() and _URDF.is_file() and _XRDF.is_file()),
                    reason='usd-core or dataset URDF/XRDF absent')
def test_including_the_drawbar_is_refused_because_it_overlaps_the_arm():
    """Publishing the drawbar (the tow hitch at the arm base) as a world obstacle would brick planning;
    check_regrasp_obstacles_clear_arm catches it loudly (never a silent INVALID_INITIAL_CSPACE on GPU)."""
    cp = tr.cart_pose_in_base(CW93_ROBOT_PARK, CW93_CART_PARK)
    specs = tr.parked_cart_world_obstacles(cp[:4], exclude=())        # keep the drawbar
    with pytest.raises(ValueError, match='intersects the arm'):
        tr.check_regrasp_obstacles_clear_arm(specs, str(_URDF), str(_XRDF), configs=[{}])


# --------------------------------------------------------------------- 3) pad-down re-grasp path + cage sweep
@pytest.mark.skipif(not _usd_available(), reason='usd-core absent (authored rollwagen cage)')
@pytest.mark.parametrize('support_z,support', [(0.225, 'deck'), (0.933, 'shelf')])
def test_pad_down_regrasp_path_clears_the_cage(support_z, support):
    from twin.restock_manipulation.place_clearance import rollwagen_collider_boxes
    bodies = rollwagen_collider_boxes()
    poses, carried = tr.mouth_pad_down_path((0.08, 0.0), support_z, 0.0, carton_h=CARTON_H)
    # four legs: approach(empty)+descend(empty)+lift(carried)+withdraw(carried)
    assert len(poses) == len(carried) and sum(carried) > 0 and sum(c is False for c in carried) > 0
    per_body = tr.regrasp_cage_clearance(poses, carried, bodies, support=support)
    reject = tr.regrasp_sweep_reject(per_body, margin=tr.DEFAULT_REGRASP_CAGE_MARGIN_M)
    assert reject is None, (support, tr.regrasp_clearance_summary(per_body))
    # every non-support cage body stays clear of the pad-down path by >= the 30 mm margin
    worst = min(v for k, v in per_body.items() if k not in ('deck', 'shelf'))
    assert worst >= tr.DEFAULT_REGRASP_CAGE_MARGIN_M


@pytest.mark.skipif(not _usd_available(), reason='usd-core absent')
def test_regrasp_sweep_names_a_strike_when_the_path_hits_a_bar():
    """An honest refuse: a path that rams the carton +x INTO a front post (no mouth clearance) is named."""
    from twin.restock_manipulation.place_clearance import rollwagen_collider_boxes
    bodies = rollwagen_collider_boxes()
    # force the standoff down INTO a side bar band and the exit SHORT of the mouth so the housing straddles a post
    poses, carried = tr.mouth_pad_down_path((0.38, 0.0), 0.225, 0.0, carton_h=CARTON_H,
                                            standoff_above=0.0, mouth_exit_x=0.38, n=2)
    per_body = tr.regrasp_cage_clearance(poses, carried, bodies, support='deck')
    reject = tr.regrasp_sweep_reject(per_body, margin=tr.DEFAULT_REGRASP_CAGE_MARGIN_M)
    assert reject is not None and reject.startswith(tr.REGRASP_CAGE_CLEARANCE_REJECTED)


def test_carried_withdraw_stays_level_within_the_2deg_bound():
    poses, carried = tr.mouth_pad_down_path((0.08, 0.0), 0.225, 0.3, carton_h=CARTON_H)
    tilt, ok = tr.carried_withdraw_level_ok(poses, carried, bound_deg=2.0)
    assert ok and tilt == pytest.approx(0.0, abs=1e-9)   # level by construction (position + yaw only)


# --------------------------------------------------------------------- 4) reposition place-park goal
def test_place_park_puts_the_endcap_within_the_reach_goal_facing_the_shelf():
    pp = tr.place_park_pose(ENDCAP, reach_m=0.90, carton_h=CARTON_H)
    # the carton-rest target sits within the ~0.9 m goal reach, the base faces the shelf, and the base is
    # CLOSER than the authored 1.217 m shelf_approach reach
    assert pp.within_goal and pp.faces_shelf and pp.within_envelope
    assert pp.reach_center_m == pytest.approx(0.90, abs=1e-6)
    assert pp.standoff_xy_m == pytest.approx(0.841, abs=2e-3)
    # the park is on the +y aisle north of the endcap, yaw pointing south at it
    assert pp.pose[0] == pytest.approx(ENDCAP[0], abs=1e-6)
    assert pp.pose[1] == pytest.approx(ENDCAP[1] + 0.841, abs=2e-3)
    assert pp.pose[2] == pytest.approx(-math.pi / 2, abs=1e-6)


def test_place_reach_ok_covers_the_level_descent_onto_the_shelf_top():
    pp = tr.place_park_pose(ENDCAP, reach_m=0.90, carton_h=CARTON_H)
    res = tr.place_reach_ok(pp.pose, ENDCAP, carton_h=CARTON_H)
    assert res['within_envelope'] and res['level_descent_ok']
    # place (carton bottom on the shelf top 0.73) + preplace standoff both inside the UR10 envelope
    assert res['reach_place_m'] < tr.DEFAULT_MAX_REACH_M and res['reach_preplace_m'] < tr.DEFAULT_MAX_REACH_M
    assert res['reach_place_m'] == pytest.approx(0.936, abs=3e-3)


def test_place_park_reach_shrinks_with_a_tighter_goal():
    far = tr.place_park_pose(ENDCAP, reach_m=1.20, carton_h=CARTON_H)
    near = tr.place_park_pose(ENDCAP, reach_m=0.90, carton_h=CARTON_H)
    assert near.standoff_xy_m < far.standoff_xy_m         # a tighter reach parks the base closer in
    assert near.reach_center_m < far.reach_center_m


# --------------------------------------------------------------------- 5) coldwin93 replay
@pytest.mark.skipif(not _usd_available(), reason='usd-core absent')
def test_coldwin93_replay_pad_down_plannable_and_place_in_reach():
    """The coldwin93 failure, replayed on CPU: (a) the pad-down re-grasp path clears the cage (the geometric
    precondition the near-face side reach lacked -- INVERSE_KINEMATICS_FAILURE), and (b) the endcap place
    target is in reach from the computed park (the observed place_surface was 2.39 m away)."""
    from twin.restock_manipulation.place_clearance import rollwagen_collider_boxes
    bodies = rollwagen_collider_boxes()
    # (a) the pad-down approach/descent/lift/withdraw off the parked cart clears the cage
    poses, carried = tr.mouth_pad_down_path((0.08, 0.0), 0.225, 0.0, carton_h=CARTON_H)
    per_body = tr.regrasp_cage_clearance(poses, carried, bodies, support='deck')
    assert tr.regrasp_sweep_reject(per_body, margin=tr.DEFAULT_REGRASP_CAGE_MARGIN_M) is None
    # (b) the place target the observe line reported 2.39 m away is in reach from the computed park
    pp = tr.place_park_pose(ENDCAP, reach_m=0.90, carton_h=CARTON_H)
    assert pp.reach_center_m <= 0.90 + 1e-6
    assert tr.place_reach_ok(pp.pose, ENDCAP, carton_h=CARTON_H)['within_envelope']
    # the cw93 observed worst_reach was 2.39 m; the computed park is far inside the envelope
    assert max(pp.reach_place_m, pp.reach_preplace_m) < 1.1 < 2.39


# --------------------------------------------------------------------- 6) boot-path wiring (the new stages run)
class _FaithfulGeometry:
    """Verbatim shelfbot wall_grip + a zero-pad approach_contact, so the controller drive runs without the
    shelfbot source (cross-checked when present)."""
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


def _faithful_policy():
    if SHELFBOT.is_dir():
        return WallCartonPolicy(str(SHELFBOT), _tool(), walls=HOST['walls'])
    policy = WallCartonPolicy.__new__(WallCartonPolicy)
    policy.tool, policy.walls, policy.lift = _tool(), dict(HOST['walls']), .03
    policy.geometry = policy.primitives = _FaithfulGeometry()
    return policy


def yaw_q(yaw):
    return (0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.))


def _to_base(world, base=BASE_356):
    dx, dy = world[0] - base[0], world[1] - base[1]
    c, s = math.cos(-base[2]), math.sin(-base[2])
    return (c * dx - s * dy, s * dx + c * dy, world[2] - UB_Z)


def _live_source(t=1.0):
    bx, by, byaw = BASE_356
    dx = bx + math.cos(byaw) * LEGACY_SOURCE_356[0] + 0.18
    dy = by + math.sin(byaw) * LEGACY_SOURCE_356[0]
    cw = (dx, dy, 0.9325 + PROFILE.dimensions[2] / 2.0)
    top = _to_base((cw[0], cw[1], cw[2] + PROFILE.dimensions[2] / 2.0))
    return Observation(f'cart:{t}', 'sensor', t, 'juice_orange', 'rollwagen',
                       (*top, *yaw_q(-math.pi / 2 - BASE_356[2])), 1.0)


def _bay_target(t=1.0):
    return Target(f'shelf:{t}', 'ground_truth', t, 'G3_back_s2', (*_to_base(BAY), *yaw_q(-BASE_356[2])),
                  True, (0.7, 0.5, 0.5))


class _Motion:
    def begin(self, pose, joints, **kw):
        pass

    def begin_joint_goal(self, goal, measured, slow=False, goal_tol=None):
        pass

    def poll(self):
        return 'succeeded'

    def last_fault_was_sim_deadline(self):
        return False

    def stop(self):
        pass

    def update_payload(self, held, profile):
        return True


def _host_controller(regrasp, monkeypatch):
    """Construct the controller through the SAME kwargs ros_host.main derives from a HOST config with the keys
    on (the boot path). transfer_regrasp + transfer_service_carry are the new stages; live_pose + wrist_guard
    are the required transfer workaround env."""
    monkeypatch.setenv(tm.ENV, 'live_pose')
    monkeypatch.setenv('RESTOCK_TRANSFER_CARRY', 'wrist_guard')
    config = dict(HOST)
    config['transfer_regrasp'] = regrasp
    config['transfer_service_carry'] = True
    stow = (0.11, -2.5, 1.7, -0.7, -1.5708, 0.0)
    c = TransferController(
        _Motion(), {'juice_orange': PROFILE}, pose_policy=_faithful_policy(), transport_joints=stow,
        contact_straight_line=True, staged_cart_swing=True,
        cart_pan_joint0=config['cart_pan_joint0'], cart_tuck_shape=config['cart_tuck_shape'],
        carry_lift_shape=config['carry_lift_shape'], carry_wrist1_cap=config['carry_wrist1_cap'],
        carry_wrist2_min=config['carry_wrist2_min'], grasp_track_m=config['grasp_track_m'],
        transfer_service_carry=bool(config.get('transfer_service_carry', False)),
        transfer_regrasp=bool(config.get('transfer_regrasp', False)),
        transfer_ground_truth=True, phase_timeout=120.)
    c.start('transfer', JOB)
    return c


def _drive_to_observe(c):
    """One observe tick: feed a live source + bay target so the controller builds self.poses, then stop
    before dispatch (keep it at the approach boundary)."""
    import time
    src = _live_source()
    tgt = _bay_target()
    for _ in range(5):
        c.step(0.1, Sensors(1.0, (src,), tgt, True, False, 1.0, 1.0, (0.,) * 6, (), 1.0,
                            time.monotonic() + 100., (*src.pose[:3], 0., 0., 0., 1.)))
        if 'approach' in c.poses:
            break
    return c.poses


def test_boot_path_regrasp_on_keeps_the_vertical_pad_down_pick(monkeypatch, capsys):
    c = _host_controller(regrasp=True, monkeypatch=monkeypatch)
    assert c.transfer_regrasp and c._uses_regrasp_pad_down() and c._repositions_before_place()
    poses = _drive_to_observe(c)
    # pad-DOWN: the approach stands directly ABOVE the contact (same x, y; higher z) -- a vertical top grasp,
    # NOT the near-face side reach (approach offset in x from contact) that cuMotion could not IK.
    assert poses['approach'][0] == pytest.approx(poses['contact'][0], abs=1e-9)
    assert poses['approach'][1] == pytest.approx(poses['contact'][1], abs=1e-9)
    assert poses['approach'][2] > poses['contact'][2]
    assert 'TRANSFER_REGRASP pad-down pick' in capsys.readouterr().out


def test_boot_path_regrasp_off_is_byte_identical_near_face_side_pick(monkeypatch, capsys):
    c = _host_controller(regrasp=False, monkeypatch=monkeypatch)
    assert not c.transfer_regrasp and not c._uses_regrasp_pad_down()
    poses = _drive_to_observe(c)
    # off: the near-face side pick stands off in x from the contact (approach reaches in through the mouth)
    assert poses['approach'][0] != pytest.approx(poses['contact'][0], abs=1e-6)
    assert 'TRANSFER_REGRASP' not in capsys.readouterr().out


def test_ros_host_passes_the_regrasp_key_to_the_controller():
    """ros_host.main() must forward config['transfer_regrasp'] to TransferController and log it. On the merged
    code the key does not exist, so these assertions fail there."""
    from twin.restock_manipulation import ros_host
    src = Path(ros_host.__file__).read_text()
    assert "transfer_regrasp=bool(config.get('transfer_regrasp', False))" in src
    assert "'transfer_regrasp': controller.transfer_regrasp" in src          # HOST_EFFECTIVE_CONFIG


def test_transfer_regrasp_is_registered_in_config_preflight():
    from twin.restock_manipulation.config_preflight import KEY_CONSUMER, HOST as _HOST
    assert KEY_CONSUMER.get('transfer_regrasp') == _HOST


def test_collect_and_non_live_pose_are_untouched_by_the_key(monkeypatch):
    # collect is never live_pose => the regrasp pick never arms
    monkeypatch.setenv(tm.ENV, 'live_pose')
    col = TransferController(_Motion(), {'juice_orange': PROFILE}, pose_policy=_faithful_policy(),
                            transfer_regrasp=True)
    col.start('collect', JOB)
    assert not col._uses_regrasp_pad_down()
    # transfer_mode off => not a live_pose transfer => the key is inert
    monkeypatch.delenv(tm.ENV, raising=False)
    off = TransferController(_Motion(), {'juice_orange': PROFILE}, pose_policy=_faithful_policy(),
                            transfer_regrasp=True)
    off.start('transfer', JOB)
    assert not off._uses_regrasp_pad_down()
