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
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.twin.restock_manipulation import (
    Observation, ProductProfile, Sensors, Target, TransferController)

# Load the proof module (pure geometry helpers + the committed authoritative cage) by file path.
_PROOF = Path(__file__).resolve().parents[2] / 'run_cart_cage_place_proof.py'
_spec = importlib.util.spec_from_file_location('cart_cage_place_proof', _PROOF)
proof = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proof)

from scripts.twin.restock_host.world_obstacles import boxes_from_specs  # noqa: E402

PROFILE = ProductProfile('sku-box', (.6, .4, .16), (.5, .5, .5, .5), max_reach=1.3, grasp_mode='top')
JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')


class FixtureMotion:
    """Records every begin() flag so the test can assert the insert is a straight-line move."""
    def __init__(self):
        self.moves, self.straights, self.joint_spaces, self.payloads = [], [], [], []
        self.outcome, self.attach_ok = 'succeeded', True

    def begin(self, pose, joints, straight_line=False, joint_space=False, wrist1_cap=None,
              slow=False, goal_tol=None, near_start=False):
        self.moves.append(pose)
        self.straights.append(straight_line)
        self.joint_spaces.append(joint_space)

    def poll(self):
        return self.outcome

    def begin_joint_goal(self, goal, measured, slow=False):
        pass

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


# ----------------------------------------------------------------------------- cage geometry

def test_open_face_insert_clears_walls_and_deck():
    # The carton (exact side-grip box) inserted down the open +x axis onto the deck clears both side
    # walls + the back wall at every step, and rests insert_gap above the deck.
    boxes = boxes_from_specs(proof.CANONICAL_CAGE)
    quat, dims = (.5, .5, .5, .5), (.6, .4, .16)
    half = proof.carton_world_extents(quat, dims)
    a = np.array(proof.CAGE_AXIS, float)
    place = np.array([-1.27, 0., proof.DECK_TOP_Z + 0.03 + half[2]])
    preplace = place + a * 0.55
    mins = {b.id: math.inf for b in boxes}
    for t in np.linspace(0, 1, 24):
        pts = proof.carton_box_points(preplace + (place - preplace) * t, quat, dims)
        for b in boxes:
            mins[b.id] = min(mins[b.id], float(np.min(b.signed_distance(pts))))
    walls = [b.id for b in boxes if b.id != 'cart_deck']
    assert all(mins[w] > 0 for w in walls), mins          # collision-free insert exists
    deck = next(b for b in boxes if b.id == 'cart_deck')
    rest = float(np.min(deck.signed_distance(proof.carton_box_points(place, quat, dims))))
    assert rest == pytest.approx(0.03, abs=5e-3)          # a few cm above the deck


def test_side_grip_carton_spans_the_narrow_cage_gap():
    # The side grip (quat 0.5,0.5,0.5,0.5) maps the carton's 0.6 m face across the cage's ~0.63 m
    # interior y-gap -> the side-wall clearance is small but POSITIVE (~10 mm). This is the inherent
    # tight fit (smaller-problems-later), not fixable by a shallower deck or closer base.
    half = proof.carton_world_extents((.5, .5, .5, .5), (.6, .4, .16))
    assert half[1] == pytest.approx(0.30, abs=1e-6)       # 0.6 m dimension spans +-y
    boxes = {b.id: b for b in boxes_from_specs(proof.CANONICAL_CAGE)}
    pts = proof.carton_box_points((-1.27, 0., proof.DECK_TOP_Z + 0.03 + half[2]), (.5, .5, .5, .5), (.6, .4, .16))
    side = min(float(np.min(boxes[s].signed_distance(pts))) for s in ('cart_side0', 'cart_side1'))
    assert 0 < side < 0.03, side                          # positive but tight


def test_insert_the_wrong_way_hits_the_back_wall():
    # Driving the carton PAST the deck toward the back wall (-x, the CLOSED face) collides -- proving
    # the only open face is +x.
    boxes = {b.id: b for b in boxes_from_specs(proof.CANONICAL_CAGE)}
    quat, dims = (.5, .5, .5, .5), (.6, .4, .16)
    half = proof.carton_world_extents(quat, dims)
    into_back = np.array([-1.65 + half[0], 0., proof.DECK_TOP_Z + 0.03 + half[2]])  # against the back wall
    pts = proof.carton_box_points(into_back, quat, dims)
    assert float(np.min(boxes['cart_back'].signed_distance(pts))) <= 0.02
