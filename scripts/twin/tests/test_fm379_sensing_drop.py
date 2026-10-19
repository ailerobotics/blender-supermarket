"""fm379 mitigation 2 (pure, no Isaac): station_camera_specs drop + _unused_sensing_camera_ids."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from twin.restock.sensing_sources import station_camera_specs, _unused_sensing_camera_ids  # noqa: E402


def _design():
    # Two station cameras (source, shelf), each with a housing solid; one support per camera.
    return {
        'cameras': {
            'cam_source': {'id': 'cam_source', 'parent': 'map', 'intrinsics': {'width': 8, 'height': 6}},
            'cam_shelf': {'id': 'cam_shelf', 'parent': 'map', 'intrinsics': {'width': 8, 'height': 6}},
        },
        'solids': [
            {'id': 'cam_source_housing', 'parent_from_solid': [0, 0, 0], 'dimensions_m': [0.1, 0.1, 0.1]},
            {'id': 'cam_shelf_housing', 'parent_from_solid': [0, 0, 0], 'dimensions_m': [0.1, 0.1, 0.1]},
        ],
        'supports': {
            'source': {'camera_id': 'cam_source'},
            'shelf': {'camera_id': 'cam_shelf'},
        },
    }


def test_unused_is_shelf_camera():
    assert _unused_sensing_camera_ids(_design()) == ('cam_shelf',)


def test_unused_empty_when_no_shelf_support():
    d = _design()
    del d['supports']['shelf']
    assert _unused_sensing_camera_ids(d) == ()


def test_unused_empty_when_shelf_camera_shared():
    d = _design()
    d['supports']['shelf']['camera_id'] = 'cam_source'   # shelf shares the source camera -> keep it
    assert _unused_sensing_camera_ids(d) == ()


def test_specs_default_byte_identical():
    specs = station_camera_specs(_design(), {'map': '/World'})
    assert {s['id'] for s in specs} == {'cam_source', 'cam_shelf'}


def test_specs_drop_removes_only_listed():
    specs = station_camera_specs(_design(), {'map': '/World'}, drop_camera_ids=('cam_shelf',))
    assert {s['id'] for s in specs} == {'cam_source'}


def test_specs_drop_empty_is_identical():
    a = station_camera_specs(_design(), {'map': '/World'})
    b = station_camera_specs(_design(), {'map': '/World'}, drop_camera_ids=())
    assert [s['id'] for s in a] == [s['id'] for s in b]
