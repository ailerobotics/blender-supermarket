"""USD-authoring tests for the people spawner (need usd-core / pxr).

Run under the CPU-only usd-core venv, e.g.::

    env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        .venv-usd/bin/python -m pytest scripts/twin/humans/tests/test_isaac_people_usd.py
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
pytest.importorskip('pxr')
from pxr import Usd, UsdGeom
from twin.humans.config import HumansConfig, PersonSpec
from twin.humans.isaac_people import HUMANS_ROOT, PeopleStage, ground_capsule_z


def _cfg():
    people = (
        PersonSpec(person_id='p1', role='shopper', waypoints=((0, 0), (4, 0)),
                   speed_mps=1.0, height_m=1.8, radius_m=0.25),
        PersonSpec(person_id='p2', role='staff', waypoints=((1, 1), (1, 5)),
                   speed_mps=0.5),
    )
    return HumansConfig(enabled=True, people=people)


def test_author_creates_capsule_actors_on_the_floor():
    stage = Usd.Stage.CreateInMemory()
    people = PeopleStage(_cfg())
    paths = people.author(stage)
    assert set(paths) == {'p1', 'p2'}
    prim = stage.GetPrimAtPath('/Humans/p1')
    assert prim and prim.IsValid()
    assert prim.GetCustomDataByKey('humans:role') == 'shopper'
    assert 'not_sensor_evidence' in prim.GetCustomDataByKey('humans:provenance')
    body = stage.GetPrimAtPath('/Humans/p1/Body')
    assert body.IsA(UsdGeom.Capsule)
    # authored at t=0 at the first waypoint, base on the floor
    cache = UsdGeom.XformCache()
    m = cache.GetLocalToWorldTransform(prim)
    t = m.ExtractTranslation()
    assert math.isclose(t[0], 0.0, abs_tol=1e-6) and math.isclose(t[1], 0.0, abs_tol=1e-6)
    assert math.isclose(t[2], ground_capsule_z(1.8, 0.25), abs_tol=1e-6)


def test_update_teleports_to_scripted_pose():
    stage = Usd.Stage.CreateInMemory()
    people = PeopleStage(_cfg())
    people.author(stage)
    people.update(2.0)                    # p1 at 1 m/s for 2 s -> x = 2
    cache = UsdGeom.XformCache()
    t = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/Humans/p1')).ExtractTranslation()
    assert math.isclose(t[0], 2.0, abs_tol=1e-6)


def test_refuses_to_overwrite_existing_scope():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, HUMANS_ROOT)
    with pytest.raises(ValueError, match='Humans'):
        PeopleStage(_cfg()).author(stage)


def test_update_before_author_raises():
    with pytest.raises(RuntimeError):
        PeopleStage(_cfg()).update(0.0)
