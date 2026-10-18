"""Offline guards for the TRANSFER place (cart -> shelf G3_back_s2), so the first GPU run after the
host-lifecycle fix exercises a FEASIBLE place rather than re-discovering an IK-infeasible one.

The sim cart-place policy (``SimCartPlacePolicy``) exists because the real-rig wall policy's OUTWARD
place standoff is cuMotion-IK-infeasible onto the cart/shelf (runs 218-222: side-grip preplace/retreat
~1.3 m fail). It re-chains the standoff BASE-WARD and sagittal-mirrors the grip orientation to the
UR10-reachable -y side. These tests pin both properties at the transfer reach distance (the shelf slot
sits ~1.1-1.2 m from the arm base in the parked-at-shelf pose), with no Isaac/CAD/GPU dependency.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock_manipulation.types import Observation, ProductProfile, Target
from twin.restock_manipulation.wall_policy import SimCartPlacePolicy, WallCartonPolicy


class NS:
    def __init__(self, **kw): self.__dict__.update(kw)


def _wall_policy():
    # Minimal WallCartonPolicy fixture (no CAD tool) producing the proven side-grip pick + an
    # OUTWARD place standoff -- the exact pose SimCartPlacePolicy must re-chain base-ward.
    policy = WallCartonPolicy.__new__(WallCartonPolicy)
    policy.tool = NS(validate_profile=lambda profile: None, panel_size=(.1, .1))
    policy.walls, policy.lift = {'rollwagen': 0, 'G3_back_s2': 0}, .03
    policy.geometry = NS(wall_grip=lambda tray, p, wall:
                         (-tray.tray_size[0] / 2 - tray.contact_margin, 0, -tray.approach_offset_x_m, 0, [.5] * 4))
    policy.primitives = NS(approach_contact=lambda x, y, z, rx, ry, q, pad:
                           ([x + rx, y + ry, z, *q], [x, y, z, *q]))
    return policy


def _profile(max_reach=1.3):
    return ProductProfile('juice_orange', (.6, .4, .16), (.5, .5, .5, .5),
                          tcp_surface_offset=.00275, grasp_mode='side', max_reach=max_reach, approach=.12)


def _reach(pose):
    return math.dist(pose[:3], (0., 0., 0.))


@pytest.mark.parametrize('ty', [1.10, 1.15, 1.20])   # shelf slot distance from the arm base (+y side)
def test_transfer_place_poses_are_base_ward_and_ik_reachable(ty):
    policy = SimCartPlacePolicy(_wall_policy())
    profile = _profile()
    # Transfer: pick from the cart (base -x, behind), place on the shelf slot (base +y, G3_back_s2).
    source = Observation('cart', 'sensor', 1., 'juice_orange', 'rollwagen', (-.9, 0., .35, 0, 0, 0, 1), 1)
    target = Target('shelf', 'sensor', 1., 'G3_back_s2', (0., ty, .23, 0, 0, 0, 1), True, (.64, .44, .30))
    poses = policy.build(source, target, profile)

    place, preplace, retreat = poses['place'], poses['preplace'], poses['retreat']
    # (1) The standoff is BASE-WARD: retreat/preplace are strictly closer to the base than place,
    #     by the profile approach in the horizontal plane -- the run-218..222 IK fix.
    assert _reach(retreat) < _reach(place)
    assert math.hypot(retreat[0], retreat[1]) == pytest.approx(math.hypot(place[0], place[1]) - profile.approach, abs=1e-6)
    assert retreat[2] == pytest.approx(place[2])                 # horizontal shelf exit
    assert preplace[2] == pytest.approx(retreat[2] + policy.lift)  # lift above retreat
    # (2) Every place-chain pose is within the arm's reach envelope (what controller._pose_ok gates on),
    #     so cuMotion gets a reachable goal instead of the ~1.3 m IK-fail.
    for name in ('place', 'preplace', 'retreat'):
        assert _reach(poses[name]) <= profile.max_reach + 1e-9, (name, _reach(poses[name]))
    # (3) Orientation is the sagittal mirror of the pick grip (reachable at -y), a unit quaternion.
    q = place[3:]
    assert q == pytest.approx(SimCartPlacePolicy._sagittal_mirror(poses['contact'][3:]))
    assert sum(c * c for c in q) == pytest.approx(1., abs=1e-6)


def test_transfer_place_rejects_a_slot_too_shallow_for_the_carton():
    # The shelf slot must have vertical room for the carton + lift (wall policy fail-closed), so a
    # mis-sensed shallow slot is refused offline rather than driving the arm into the shelf.
    policy = SimCartPlacePolicy(_wall_policy())
    profile = _profile()
    source = Observation('cart', 'sensor', 1., 'juice_orange', 'rollwagen', (-.9, 0., .35, 0, 0, 0, 1), 1)
    shallow = Target('shelf', 'sensor', 1., 'G3_back_s2', (0., 1.15, .23, 0, 0, 0, 1), True, (.64, .44, .17))
    with pytest.raises(ValueError):
        policy.build(source, shallow, profile)
