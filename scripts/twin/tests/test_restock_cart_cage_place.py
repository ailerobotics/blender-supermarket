"""CPU-only guards for the cart-cage open-face place (firstmate 350).

Two tiers, both GPU-free and dataset-free:
  * the TransferController 2-step place re-chaining + straight-line insert routing (fixture motion), and
  * the pure cage geometry (committed CANONICAL_CAGE + the proof's exact carton-box helpers): the
    carton inserts down the OPEN +x face clear of the three walls, and the WRONG way (into the back
    wall) collides -- proving the open face is +x.

The live clearance/fold numbers against the real URDF/XRDF are produced by scripts/run_cart_cage_place_proof.py
(it needs the dataset CAD); these tests pin the logic and the open-face geometry that gate it.
"""
import importlib.util
from pathlib import Path

import pytest

from scripts.twin.restock_manipulation import (
    Observation, ProductProfile, Sensors, Target, TransferController)

# Load the proof module (committed authoritative cage + the wrist_1-safe shape constants) by file path.
_PROOF = Path(__file__).resolve().parents[2] / 'run_cart_cage_place_proof.py'
_spec = importlib.util.spec_from_file_location('cart_cage_place_proof', _PROOF)
proof = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proof)

PROFILE = ProductProfile('sku-box', (.6, .4, .16), (.5, .5, .5, .5), max_reach=1.3, grasp_mode='top')
JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')


class FixtureMotion:
    """Records every begin()/begin_joint_goal() so the test can assert move routing."""
    def __init__(self):
        self.moves, self.straights, self.joint_spaces, self.payloads = [], [], [], []
        self.joint_goals = []
        self.outcome, self.attach_ok = 'succeeded', True

    def begin(self, pose, joints, straight_line=False, joint_space=False, wrist1_cap=None,
              slow=False, goal_tol=None, near_start=False):
        self.moves.append(pose)
        self.straights.append(straight_line)
        self.joint_spaces.append(joint_space)

    def poll(self):
        return self.outcome

    def begin_joint_goal(self, goal, measured, slow=False):
        self.joint_goals.append(tuple(goal))

    def stop(self):
        pass

    def update_payload(self, held, profile):
        self.payloads.append(held)
        return self.attach_ok


def make_controller(**kwargs):
    motion = FixtureMotion()
    c = TransferController(motion, {'sku-box': PROFILE}, fixture_mode=True, **kwargs)
    c.start('transfer', JOB)
    return c, motion


def packet(c, t=1., **kwargs):
    import time
    obs = Observation(f'source-{t}', 'test_fixture', t, 'sku-box', c.source, (.4, 0., .2, 0, 0, 0, 1), .99)
    # A deck target behind the base (-x), representative of the cart load centre.
    target = Target(f'target-{t}', 'test_fixture', t, c.destination, (-.5, 0., .1, 0, 0, 0, 1), True, (.7, .7, .5))
    values = dict(simulation_time=t, observations=(obs,), target=target, parked=True, contact=c.held,
                  feedback_time=t, collision_time=t, joint_positions=(0.,) * 6,
                  scanned_locations=(c.source, c.destination), scan_time=t,
                  valid_until_steady=time.monotonic() + 100.)
    values.update(kwargs)
    return Sensors(**values)


def run_to(c, phase, t0=1.0):
    for i in range(80):
        if c.phase == phase:
            return t0 + i * .1
        contact = c.held or (c.phase == 'grasp' and c.issued)
        if c.phase == 'release' and c.issued:
            contact = False
        r = c.step(.1, packet(c, t0 + i * .1, contact=contact))
        assert r.status != 'failed', r.reason
    pytest.fail('phase %r not reached' % phase)


# ----------------------------------------------------------------------------- controller logic

def test_cart_cage_place_rechains_place_through_open_face():
    # The place is re-chained: place = sensed deck target lifted insert_gap; preplace/retreat offset
    # the SAME pose out the open +x mouth by mouth_standoff. Orientation is unchanged.
    gap, standoff = 0.03, 0.30
    c, _ = make_controller(cart_cage_place=True, cage_mouth_standoff=standoff, cage_insert_gap=gap)
    run_to(c, 'preplace')
    place, preplace, retreat = c.poses['place'], c.poses['preplace'], c.poses['retreat']
    base_place = (-.5, 0., .1 + PROFILE.dimensions[2])  # pose(target, dims[2]); tcp_surface_offset=0
    assert place[:3] == pytest.approx((base_place[0], base_place[1], base_place[2] + gap))
    # preplace/retreat are the place pose pushed +x (the open face) by the standoff, same height + quat.
    assert preplace[:3] == pytest.approx((place[0] + standoff, place[1], place[2]))
    assert retreat[:3] == pytest.approx(preplace[:3])
    assert place[3:] == pytest.approx(PROFILE.tcp_quaternion)      # orientation unchanged
    assert preplace[3:] == pytest.approx(PROFILE.tcp_quaternion)


def test_cart_cage_insert_is_straight_line_and_preplace_is_collision_aware():
    # PREPLACE routes around the walls -> collision-aware cuMotion (NOT straight-line); the PLACE insert
    # and the empty RETREAT drive straight through the open face (straight_line=True, collision-off).
    c, m = make_controller(cart_cage_place=True, cage_mouth_standoff=0.30)
    run_to(c, 'preplace')
    c.step(.1, packet(c, 9.0)); assert c.phase == 'preplace'   # issue preplace
    assert m.straights[-1] is False                             # cuMotion plan around the cage walls
    c.step(.1, packet(c, 9.1)); assert c.phase == 'place'       # preplace poll -> place
    c.step(.1, packet(c, 9.2))                                  # issue the insert
    assert m.straights[-1] is True                              # straight-line insert through the mouth
    run_to(c, 'retreat', t0=9.3)
    c.step(.1, packet(c, 20.0))                                 # issue the retreat (empty-handed, out +x)
    assert m.straights[-1] is True


def test_cart_cage_place_off_leaves_place_chain_and_routing_untouched():
    off, m_off = make_controller()                              # default: cart_cage_place off
    on, _ = make_controller(cart_cage_place=True, cage_mouth_standoff=0.30)
    run_to(off, 'preplace'); run_to(on, 'preplace')
    # With the flag OFF the place chain is the untouched pose-IK preplace/place; ON it is re-chained.
    assert off.poses['preplace'][:3] != pytest.approx(on.poses['preplace'][:3])
    assert off.poses['place'][2] != pytest.approx(on.poses['place'][2])   # ON lifts place by insert_gap
    # And OFF never routes the place/retreat as a straight-line move (every move is cuMotion-planned).
    run_to(off, 'retreat', t0=9.0)
    c2, m2 = make_controller()
    for t in range(60):
        c2.step(.1, packet(c2, 30 + t * .1,
                            contact=(c2.held or (c2.phase == 'grasp' and c2.issued))
                            and not (c2.phase == 'release' and c2.issued)))
        if c2.phase in ('succeeded', 'failed'):
            break
    assert not any(m2.straights)                               # no straight-line insert when OFF


@pytest.mark.parametrize('bad', [dict(cage_axis=(0., 0., 0.)), dict(cage_mouth_standoff=0.),
                                 dict(cage_mouth_standoff=-0.1), dict(cage_insert_gap=-0.01)])
def test_cart_cage_place_rejects_invalid_params(bad):
    with pytest.raises(ValueError):
        TransferController(FixtureMotion(), {'sku-box': PROFILE}, cart_cage_place=True, **bad)


# ------------------------------------------------- fixed wrist_1-safe preplace (begin_joint_goal)

SHAPE = proof.CAGE_PREPLACE_SHAPE


def test_cage_preplace_shape_routes_begin_joint_goal():
    # run-279: with cart_cage_place + cage_preplace_shape, PREPLACE is a fixed begin_joint_goal to the
    # shape (wrist_1 below the self-collision stop) instead of the pose-IK move that jammed in run-278.
    # The PLACE insert stays a straight-line move to the SENSED target.
    c, m = make_controller(cart_cage_place=True, cage_mouth_standoff=0.30, cage_preplace_shape=SHAPE)
    run_to(c, 'preplace')
    before = len(m.joint_goals)
    c.step(.1, packet(c, 9.0)); assert c.phase == 'preplace'   # issue preplace
    assert len(m.joint_goals) == before + 1                    # routed to begin_joint_goal...
    assert m.joint_goals[-1] == pytest.approx(SHAPE)           # ...with the fixed wrist_1-safe shape
    c.step(.1, packet(c, 9.1)); assert c.phase == 'place'      # preplace poll -> place
    c.step(.1, packet(c, 9.2))                                 # issue the insert
    assert m.straights[-1] is True                             # place still a straight-line insert


def test_cage_preplace_shape_absent_keeps_pose_ik_preplace():
    # cart_cage_place ON but cage_preplace_shape ABSENT => preplace is the pose-IK move (byte-identical
    # to before the fix): a begin() collision-aware move, never begin_joint_goal.
    c, m = make_controller(cart_cage_place=True, cage_mouth_standoff=0.30)
    run_to(c, 'preplace')
    goals_before = len(m.joint_goals)
    c.step(.1, packet(c, 9.0)); assert c.phase == 'preplace'
    assert len(m.joint_goals) == goals_before                  # no begin_joint_goal for preplace
    assert m.straights[-1] is False                            # pose-IK collision-aware preplace


@pytest.mark.parametrize('bad', [(0., 0., 0.), (1.,) * 5, (1.,) * 7,
                                 (1., 2., float('nan'), 4., 5., 6.)])
def test_cage_preplace_shape_rejects_invalid(bad):
    with pytest.raises(ValueError):
        TransferController(FixtureMotion(), {'sku-box': PROFILE}, cart_cage_place=True,
                           cage_preplace_shape=bad)


# ----------------------------------------------------------- authoritative cage + shape (run-279 pin)

def test_canonical_cage_is_the_beside_cart_open_plus_x():
    # run-279 correction: CANONICAL_CAGE is the mission's REAL /Rollwagen BESIDE the robot on -y, open
    # +x -- NOT the stale behind-cart (deck -1.27) from the compose_robot_in_store stage.
    by_id = {s['id']: s for s in proof.CANONICAL_CAGE}
    assert by_id['cart_deck']['center'] == pytest.approx([0.0, -0.75, -0.30], abs=1e-6)
    assert by_id['cart_back']['center'][0] == pytest.approx(-0.38, abs=1e-6)    # back wall at -x
    assert by_id['cart_side0']['center'][1] == pytest.approx(-1.07, abs=1e-6)   # side walls straddle -0.75
    assert by_id['cart_side1']['center'][1] == pytest.approx(-0.43, abs=1e-6)
    assert proof.CAGE_AXIS == (1.0, 0.0, 0.0)                                    # open +x


def test_cage_preplace_shape_wrist1_safe_and_matches_host_config():
    # The committed shape keeps wrist_1 at/below the ur_wrist_2-vs-ur_forearm self-collision stop, and
    # the proof constant matches the host config ros_host actually reads (and the solid cage is dropped).
    import json
    assert proof.CAGE_PREPLACE_SHAPE[3] <= proof.WRIST1_STOP
    cfg = json.loads((Path(__file__).resolve().parents[2]
                      / 'twin/restock_manipulation/ros_host_config.json').read_text())
    assert cfg['cage_preplace_shape'] == pytest.approx(list(proof.CAGE_PREPLACE_SHAPE))
    assert 'world_obstacles' not in cfg                        # the solid cage obstacle is dropped
