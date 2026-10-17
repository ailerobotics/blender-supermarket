"""The default crowd really crosses the route and approaches the stations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from twin.humans.isaac_people import PeopleStage, ground_capsule_z
from twin.humans.config import HumansConfig
from twin.humans.paths import crosses_segment, min_distance_to_point
from twin.humans.people import default_people

# A representative mission pose set (x, y, yaw) in the map frame.
POSES = {
    'pickup': (2.0, 1.0, 0.0),
    'cart_dock': (3.5, 1.0, 3.14159),
    'shelf_approach': (9.0, 5.0, 1.5708),
    'home': (2.0, 5.0, 0.0),
}
DURATION = 60.0


def test_default_people_ids_and_roles():
    people = default_people(POSES)
    assert {p.person_id for p in people} == {'shopper_aisle', 'staff_shelf', 'shopper_dock'}
    assert {p.role for p in people} == {'shopper', 'staff'}


def test_aisle_shopper_crosses_outbound_route():
    people = {p.person_id: p for p in default_people(POSES)}
    aisle = people['shopper_aisle'].path()
    route = ((POSES['pickup'][0], POSES['pickup'][1]),
             (POSES['shelf_approach'][0], POSES['shelf_approach'][1]))
    assert crosses_segment(aisle, route, DURATION, clearance_m=0.5)


def test_staff_enters_shelf_station():
    people = {p.person_id: p for p in default_people(POSES)}
    staff = people['staff_shelf'].path()
    shelf = (POSES['shelf_approach'][0], POSES['shelf_approach'][1])
    assert min_distance_to_point(staff, shelf, DURATION) < 1.5


def test_dock_shopper_approaches_backstock():
    people = {p.person_id: p for p in default_people(POSES)}
    dock = people['shopper_dock'].path()
    pickup = (POSES['pickup'][0], POSES['pickup'][1])
    assert min_distance_to_point(dock, pickup, DURATION) < 0.5


def test_people_stage_positions_are_deterministic():
    cfg = HumansConfig(enabled=True, people=default_people(POSES))
    stage = PeopleStage(cfg)
    a = stage.world_positions(3.3)
    b = stage.world_positions(3.3)
    assert a == b
    assert set(a) == {'shopper_aisle', 'staff_shelf', 'shopper_dock'}
    # z is a plausible chest height
    for _pid, (_x, _y, z) in a.items():
        assert 0.5 < z < 1.2


def test_ground_capsule_z():
    # 1.75 m person, 0.25 m radius -> centre at 0.875 m (base on the floor)
    assert abs(ground_capsule_z(1.75, 0.25) - 0.875) < 1e-9
