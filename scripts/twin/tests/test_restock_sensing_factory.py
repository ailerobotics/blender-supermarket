"""CPU contract tests for the reference sensing_factory; no Isaac/ROS, no GPU."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_camera.sensing import build_sensing, support_slots, rendered_context
from scripts.twin.restock_camera.config import camera_config
from scripts.twin.restock_manipulation.types import Sensors, Target


def design():
    scenario = dict(handling_unit=dict(local_wall=3, grasp_mode='side', dimensions_m=[.6, .4, .16]),
        initial_staging=dict(support_center_xy=[-5.73, 17.1], unit_bottom_z=.6,
            target_surface_pose_world=[-3.05, 4.826, .73, 0], cart_load_center_body_xy=[.08, 0],
            cart_support_z_world=.225))
    return camera_config(scenario, dict(spec=dict(arm_mount_z=.5)))


def _sources(monkeypatch, *, target_location='shelf'):
    """A fake assembly whose StationSensors adapters return canned Sensors packets."""
    from scripts.twin.restock_manipulation import isaac_sensors
    target = Target('depth:2', 'sensor', 2., target_location, (0, 0, .2, 0, 0, 0, 1), True, (.62, .42, .25))
    packets = {
        'cart': Sensors(2., (), None, True, False, 2., 2., (0.,) * 6, ('rollwagen',), 2.),
        'shelf': Sensors(2., (), target, True, False, 2., 2., (0.,) * 6, ('shelf',), 2.)}
    calls = []

    def adapter(camera, **kw):
        key = kw['camera_id']
        def capture(now, source, destination, motion, contact):
            calls.append((now, source, destination, motion, contact))
            return packets[key]
        return NS(slots=(NS(location={'cart': 'rollwagen', 'shelf': 'shelf'}[key]),),
                  capture=capture, sensed_target_bboxes=(), evidence=())
    monkeypatch.setattr(isaac_sensors, 'IsaacCartonSensors', adapter)
    motion = NS(parked=lambda: True, collision_time=2., joint_positions=lambda: (0.,) * 6)
    contact = NS(read=lambda: False)
    sources = dict(cameras={'cart': object(), 'shelf': object()},
                   context_providers={'cart': object(), 'shelf': object()},
                   location_cameras={'rollwagen': 'cart', 'shelf': 'shelf'}, profiles={},
                   pose_bridge_root='fixture', motion=motion, contact=contact,
                   recording_camera='mission-cam')
    return NS(sensing_sources=sources), packets, calls, motion, contact


def test_factory_returns_capture_and_recording_camera(monkeypatch):
    assembly, packets, calls, motion, contact = _sources(monkeypatch)
    sensing = build_sensing(app=None, assembly=assembly, plan=None, config={}, output=None)
    # 'set_render_suspended' is the opt-in sensing-gate seam (a no-op when the gate config is unset).
    assert set(sensing) == {'capture', 'recording_camera', 'set_render_suspended'}
    assert sensing['recording_camera'] == 'mission-cam'
    assert callable(sensing['set_render_suspended'])
    result = sensing['capture'](2., 'rollwagen', 'shelf')
    # Aggregated across the two assigned cameras; the closure forwards motion + contact.
    assert result.target is packets['shelf'].target
    assert result.scanned_locations == ('rollwagen', 'shelf')
    assert calls and all(c[3] is motion and c[4] is contact for c in calls)


def test_capture_enforces_station_gates(monkeypatch):
    assembly, packets, calls, _, _ = _sources(monkeypatch)
    sensing = build_sensing(app=None, assembly=assembly, plan=None, config={}, output=None)
    # Unsynchronized station cameras are rejected (not silently averaged).
    packets['cart'] = replace(packets['cart'], scan_time=1.9)
    with pytest.raises(ValueError, match='not synchronized'):
        sensing['capture'](2., 'rollwagen', 'shelf')


def test_factory_fails_closed_on_missing_binding(monkeypatch):
    assembly, *_ = _sources(monkeypatch)
    del assembly.sensing_sources['recording_camera']
    with pytest.raises(ValueError, match='recording_camera'):
        build_sensing(app=None, assembly=assembly, plan=None, config={}, output=None)
    with pytest.raises(ValueError, match='sensing_sources'):
        build_sensing(app=None, assembly=NS(), plan=None, config={}, output=None)


def test_support_slots_from_the_camera_design():
    config = design()
    masks = {name: np.ones((4, 4), bool) for name in ('source', 'rollwagen', 'shelf')}
    slots = support_slots(config, 'cart', label_masks=masks,
                          support_locations={'cart': 'rollwagen'})
    assert len(slots) == 1
    slot = slots[0]
    support = config['supports']['cart']
    assert slot.location == 'rollwagen' and slot.support_id == support['id']
    assert slot.marker_id == support['markers'][0]['id']
    assert slot.auxiliary_id == support['markers'][1]['id']
    assert slot.marker_side_m == support['markers'][0]['side_m']
    assert slot.revision == support['revision']
    assert slot.required_xy == tuple(support['required_xy_m'])
    assert slot.auxiliary_offset_xy == tuple(support['auxiliary_offset_xy'])
    # Only this camera's support is returned.
    assert [s.support_id for s in support_slots(config, 'shelf', label_masks=masks)] \
        == [config['supports']['shelf']['id']]


def test_rendered_context_takes_design_intrinsics_and_depth_bound():
    config = design()
    masks = {'shelf': np.ones((4, 4), bool)}
    slots = support_slots(config, 'shelf', label_masks=masks)
    context = rendered_context(config, 'shelf', slots,
                               measure_context=lambda frame, now: None,
                               clearance_for_slot=lambda *a: None, camera_revision='lens1')
    assert context.camera_id == 'shelf'
    assert context.intrinsics.fx == config['cameras']['shelf']['intrinsics']['fx']
    assert context.depth_error_m == config['required_depth_error_bound_m']
    assert context.camera_revision == 'lens1'
    assert tuple(s.location for s in context.slots) == ('shelf',)
