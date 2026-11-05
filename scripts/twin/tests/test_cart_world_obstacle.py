"""CPU proof for the known-pose rollwagen cart as cuMotion world obstacles (no GPU/ROS/USD).

Proves the S0 CPU acceptance: with the cart published as world CollisionObjects, a deck-
penetrating place is REJECTED (negative world-obstacle clearance -> cuMotion INVALID_INITIAL_CSPACE)
while a standoff above the deck is accepted -- i.e. the set-down is now collision-checked against
the real deck/cage instead of relying on the CAGE_COLLISION filter + straight-line inserts.
"""
import math

import numpy as np
import pytest

from twin.restock_manipulation import cart_world_obstacle as cwo
from twin.restock_host.world_obstacles import Box, boxes_from_specs


# A synthetic rollwagen collider set in the cart BODY frame (x -> open +x mouth, origin at the
# cart ground point): a deck + upper shelf (horizontal supports), four cage posts, a drawbar.
# Names follow place_clearance.CAGE_PREFIXES ('post_'/'side_'/'back_') so is_cage_body classifies
# them exactly as the real authored colliders do.
CART_COLLIDERS = {
    'deck':     ((-0.40, -0.30, 0.35), (0.40, 0.30, 0.42)),
    'shelf':    ((-0.40, -0.30, 0.95), (0.40, 0.30, 1.00)),
    'drawbar':  ((-0.52, -0.05, 0.30), (-0.40, 0.05, 0.40)),
    'post_fl':  ((0.35, 0.25, 0.42), (0.40, 0.30, 1.00)),
    'post_fr':  ((0.35, -0.30, 0.42), (0.40, -0.25, 1.00)),
    'post_bl':  ((-0.40, 0.25, 0.42), (-0.35, 0.30, 1.00)),
    'post_br':  ((-0.40, -0.30, 0.42), (-0.35, -0.25, 1.00)),
    'side_l':   ((-0.35, 0.28, 0.45), (0.35, 0.30, 0.95)),
}

# A plausible coupled cart pose in ur_base_link: the cart to the robot's side/below (base y ~ -0.8),
# level, yawed a little.
CART_POSE = (0.10, -0.80, -0.20, 0.3)


def test_specs_group_cage_and_keep_supports_separate():
    specs = cwo.rollwagen_world_obstacle_specs(CART_COLLIDERS, CART_POSE)
    ids = [s['id'] for s in specs]
    assert cwo.CAGE_OBSTACLE_ID in ids
    # Every post_/side_ body is folded into the single cage obstacle; the horizontal supports stay
    # their own obstacle so a place is checked against exactly the deck.
    cage = next(s for s in specs if s['id'] == cwo.CAGE_OBSTACLE_ID)
    assert len(cage['boxes']) == 5  # 4 posts + 1 side bar
    assert 'rollwagen_deck' in ids and 'rollwagen_shelf' in ids and 'rollwagen_drawbar' in ids
    # Every emitted spec is a valid world_obstacles host-key spec (round-trips through the host loader).
    flat = [{'id': s['id'], **box} for s in specs for box in s['boxes']]
    assert boxes_from_specs(flat)


def test_pose_transform_applies_yaw_and_translation():
    # A box at the cart origin maps to the cart translation; a yaw rotates the box frame.
    colliders = {'deck': ((-0.1, -0.1, 0.0), (0.1, 0.1, 0.1))}
    specs = cwo.rollwagen_world_obstacle_specs(colliders, (1.0, 2.0, 0.5, math.pi / 2))
    box = specs[0]['boxes'][0]
    cx, cy, cz = box['center']
    # cart-frame box centre (0, 0, 0.05) under yaw +90 deg + translation (1, 2, 0.5)
    assert cx == pytest.approx(1.0, abs=1e-6)
    assert cy == pytest.approx(2.0, abs=1e-6)
    assert cz == pytest.approx(0.55, abs=1e-6)
    assert box['rpy'][2] == pytest.approx(math.pi / 2, abs=1e-6)


def _deck_box(pose=CART_POSE):
    specs = cwo.rollwagen_world_obstacle_specs(CART_COLLIDERS, pose, include=['deck'])
    flat = [{'id': specs[0]['id'], **specs[0]['boxes'][0]}]
    return boxes_from_specs(flat)[0]


def test_deck_penetrating_place_is_rejected_standoff_accepted():
    """S0 CPU acceptance: a place that penetrates the deck has NEGATIVE world-obstacle clearance
    (cuMotion would reject INVALID_INITIAL_CSPACE); a standoff above the deck is clear."""
    deck = _deck_box()
    T = cwo.cart_pose_matrix(CART_POSE)
    # Carton bottom-centre samples in the cart frame, over the deck centre (deck top z = 0.42):
    deck_top_z = 0.42
    # Penetrating: carton bottom 2 cm BELOW the deck top.
    pene_cart = np.array([[0.0, 0.0, deck_top_z - 0.02]])
    pene_base = pene_cart @ T[:3, :3].T + T[:3, 3]
    assert float(deck.signed_distance(pene_base)[0]) < 0.0
    # Standoff: carton bottom 6 cm ABOVE the deck top -> clear.
    stand_cart = np.array([[0.0, 0.0, deck_top_z + 0.06]])
    stand_base = stand_cart @ T[:3, :3].T + T[:3, 3]
    assert float(deck.signed_distance(stand_base)[0]) > 0.0


def test_padding_inflates_every_extent():
    base = cwo.rollwagen_world_obstacle_specs({'deck': ((-0.1, -0.1, 0.0), (0.1, 0.1, 0.1))},
                                              (0, 0, 0, 0))[0]['boxes'][0]
    padded = cwo.rollwagen_world_obstacle_specs({'deck': ((-0.1, -0.1, 0.0), (0.1, 0.1, 0.1))},
                                                (0, 0, 0, 0), pad_m=0.02)[0]['boxes'][0]
    for b, p in zip(base['size'], padded['size']):
        assert p == pytest.approx(b + 0.04, abs=1e-9)


def test_bad_pose_rejected():
    with pytest.raises(ValueError):
        cwo.cart_pose_matrix((1.0, 2.0))          # wrong length
    with pytest.raises(ValueError):
        cwo.rollwagen_world_obstacle_specs({}, CART_POSE)   # no colliders


# -- Full arm-clearance proof against the real dataset arm (skips where the assets are absent) --
import os  # noqa: E402

_URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'


@pytest.mark.skipif(not (os.path.isfile(_URDF) and os.path.isfile(_XRDF)),
                    reason='dataset arm URDF/XRDF not on this box')
def test_cart_obstacle_clears_arm_at_rest():
    # The cart is to the side/below; its world obstacles must clear the arm at the zero config
    # (a box overlapping the start config rejects every plan, firstmate 295).
    specs = cwo.rollwagen_world_obstacle_specs(CART_COLLIDERS, CART_POSE)
    clearances = cwo.check_cart_obstacle_clears_arm(specs, _URDF, _XRDF, configs=[{}])
    assert clearances and min(clearances.values()) > 0.0
