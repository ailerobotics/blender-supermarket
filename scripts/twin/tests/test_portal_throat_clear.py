"""sim-throat-fixed-collider-v1: the from-rest portal pin is NOT a latent vertical collider.

CPU disproof of harvest item 5 / repeatability-ledger msg 628 ("a VERTICAL collider at
x~=-3.935 against the base body, cart body or cart casters"). Against the store's own static
collision geometry (the committed throat obstacle fixture = the nav manifest's world AABBs in the
portal region), the coupled tow's swept envelope through the portal from the recorded from-rest
pin is collider-clear, the load-bearing drive wheels are off every ``Backroom_floor`` cooked
diagonal, and the one caster over a diagonal is held airborne by ``caster_lift_m``. A positive
control proves the guard fires when a collider IS dropped into the corridor.

Pure geometry; no pxr, no Isaac, no GPU.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.mobile import MobileSpec, Pose2  # noqa: E402
from twin.restock import portal_throat as pt  # noqa: E402
from twin.restock import portal_floor as pf  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "portal_throat_obstacles.json"
LATE = MobileSpec(caster_lift_m=0.03)   # the late-run traction config of the pinned runs
LEGACY = MobileSpec(caster_lift_m=0.0)  # the early co-planar-caster config


@pytest.fixture(scope="module")
def obstacles():
    data = json.loads(FIXTURE.read_text())
    assert data["n_obstacles"] == len(data["obstacles"]) >= 50
    return data["obstacles"]


# --- the headline disproof ------------------------------------------------------------------
def test_throat_is_collider_clear(obstacles):
    c = pt.assess(obstacles, LATE)
    assert c.collider_free is True
    assert c.corridor_clear is True
    assert c.intrusions == ()


def test_nearest_static_obstacle_is_metres_away(obstacles):
    c = pt.assess(obstacles, LATE)
    # The nearest drive-height collider to the pin is the backroom racks, ~2.66 m north --
    # nothing at the pin x=-3.935. A latent collider there would read << 1 m.
    assert c.nearest_obstacle_m is not None and c.nearest_obstacle_m > 1.5
    assert c.nearest_obstacle_id and c.nearest_obstacle_id.startswith("Rack_")


# --- the load path is off every floor edge --------------------------------------------------
def test_drive_wheels_off_both_cooked_diagonals():
    base = Pose2(pt.PIN_BASE_XY[0], pt.PIN_BASE_XY[1], 0.0)
    diags = pt.cooked_diagonals()
    assert len(diags) == 2
    dmin = min(abs(pf.perp_distance(w, d[0], d[1]))
               for w in pt.drive_wheel_world(base, LATE) for d in diags)
    # The arrested load-bearing contacts sit ~0.47 m from the nearest cooked diagonal -- the
    # floor-edge ghost-contact mechanism cannot be the arrest.
    assert dmin > 0.3
    assert pt.assess([], LATE).load_path_off_diagonal is True


# --- the one caster over a diagonal is airborne (and WAS co-planar in the early runs) --------
def test_front_caster_overlies_a_diagonal_but_is_airborne():
    c = pt.assess([], LATE)
    # The front-left cart caster sits ON a cooked diagonal (the original portal_floor observation)
    assert c.caster_on_diagonal_label == "front_left"
    assert c.caster_on_diagonal_perp_m < 0.05
    # ...but caster_lift_m=0.03 holds its wheel 30 mm above the floor -> it bears no load.
    assert c.caster_airborne_m == pytest.approx(0.03, abs=1e-3)
    assert c.caster_airborne_m > 0.0


def test_legacy_coplanar_caster_is_on_the_floor():
    # With the early co-planar config the caster wheel bottom is AT the floor (z~0): then the
    # front caster DID land on the cooked diagonal -- the early-run floor-edge pin mechanism.
    assert pt.caster_wheel_bottom_z(LEGACY) == pytest.approx(0.0, abs=1e-6)
    assert pt.caster_wheel_bottom_z(LATE) == pytest.approx(0.03, abs=1e-6)


# --- the fixed x is hitch kinematics, not a collider ----------------------------------------
def test_fixed_pin_x_is_hitch_offset_from_the_cart_rest_pose():
    ho = pt.hitch_offset(LATE)
    assert ho == pytest.approx(1.27, abs=1e-6)
    base = Pose2(pt.PIN_BASE_XY[0], pt.PIN_BASE_XY[1], 0.0)
    cart = pt.cart_pose_trailing(base, LATE)
    # base_x == cart_x + hitch_offset -> the fixed pin x is set by where the follower seats the
    # cart at its turn-entry pose, not by any collider.
    assert base.x == pytest.approx(cart.x + ho, abs=1e-9)


# --- positive control: the guard MUST fire when a collider is in the corridor ----------------
def test_guard_fires_on_an_injected_corridor_collider(obstacles):
    # A hypothetical jamb footing / duplicated wall segment dropped onto the pin at drive height.
    intruder = dict(id="HYPOTHETICAL_jamb_footing",
                    min=[-4.05, 15.0, 0.0], max=[-3.85, 15.4, 1.2])
    c = pt.assess(list(obstacles) + [intruder], LATE)
    assert c.corridor_clear is False
    assert c.collider_free is False
    assert any("HYPOTHETICAL_jamb_footing" in ids for _, ids in c.intrusions)


def test_a_ceiling_only_collider_does_not_fire(obstacles):
    # A collider at the pin xy but ABOVE the drive band (e.g. an overhead beam) must NOT fire --
    # the train passes under it.
    overhead = dict(id="overhead_beam", min=[-4.05, 15.0, 3.0], max=[-3.85, 15.4, 3.4])
    c = pt.assess(list(obstacles) + [overhead], LATE)
    assert c.corridor_clear is True
    assert c.collider_free is True


# --- SAT geometry sanity --------------------------------------------------------------------
def test_polygon_aabb_overlap_sat():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert pt.polygon_aabb_overlap(square, [0.4, 0.4], [0.6, 0.6]) is True   # inside
    assert pt.polygon_aabb_overlap(square, [0.9, 0.9], [1.5, 1.5]) is True   # corner overlap
    assert pt.polygon_aabb_overlap(square, [2.0, 2.0], [3.0, 3.0]) is False  # far away
