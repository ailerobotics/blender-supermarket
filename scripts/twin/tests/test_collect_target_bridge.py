"""CollectTargetBridge: fixed-source target -> host Sensors packet + reprojected arm box.

The Isaac-only live arm pose read (`_arm_base_from_camera`) is stubbed; everything else
(freshness/parked gates, packet construction, bbox reprojection) is exercised on CPU.
"""
import numpy as np
import pytest

from scripts.twin.restock_manipulation.isaac_factory import CollectTargetBridge
from scripts.twin.restock_manipulation.types import ProductProfile
from scripts.twin.restock_camera.source_target import SourceTarget


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _bridge(provider):
    cam = NS(get_current_frame=lambda: {'distance_to_image_plane': np.zeros((540, 960))},
             get_intrinsics_matrix=lambda: np.array([[1500., 0, 480.], [0, 1500., 270.], [0, 0, 1.]]))
    b = CollectTargetBridge(stage=None, arm_camera=cam, arm_camera_path='/arm', base_path='/base',
                            profiles={'juice_orange': ProductProfile(sku='juice_orange',
                                      dimensions=(.6, .4, .16), tcp_quaternion=(0., 0., 0., 1.))},
                            motion=NS(parked=lambda: True, collision_time=9.0), contact=NS(read=lambda: False),
                            joints=lambda: (0.,) * 6, source_location='backstock_00')
    b._arm_base_from_camera = lambda: np.eye(4)      # stub the Isaac XformCache read
    b.provider = provider
    return b


def _target(stamp=1.0, valid_until_steady=None):
    return SourceTarget(sku='juice_orange', pose_station=(0., 0., 1.5, 0., 0., 0., 1.),
                        stamp=stamp, provenance={'camera_id': 'source', 'source': 'sensor'},
                        valid_until_steady=valid_until_steady)


def test_sensed_target_produces_a_sensor_packet_and_reprojected_box():
    b = _bridge(lambda now: _target(stamp=now))
    packet = b.capture(1.0, 'backstock_00', 'rollwagen')
    assert packet is not None
    assert len(packet.observations) == 1
    obs = packet.observations[0]
    assert obs.source == 'sensor' and obs.sku == 'juice_orange' and obs.location == 'backstock_00'
    assert obs.frame == 'ur_base_link'
    assert packet.target is None and packet.parked is True
    # the reprojected arm box is published at the collect location
    assert len(b.sensed_target_bboxes) == 1
    box = b.sensed_target_bboxes[0]
    assert box['location'] == 'backstock_00' and len(box['roi']) == 4


def test_held_acquisition_deadline_is_threaded_onto_the_host_packet():
    # Under the sensing gate the held source target carries a fresh sim-clock valid_until_steady;
    # the bridge must put it on the host Sensors so the contact/grasp suction guard can fire (run-140).
    b = _bridge(lambda now: _target(stamp=now, valid_until_steady=now + 18.5))
    packet = b.capture(1.0, 'backstock_00', 'rollwagen')
    assert packet is not None and packet.valid_until_steady == 1.0 + 18.5
    # Not suspended (no deadline on the target): the host packet carries none -- byte-identical.
    b2 = _bridge(lambda now: _target(stamp=now))
    assert b2.capture(1.0, 'backstock_00', 'rollwagen').valid_until_steady is None


def test_none_when_nothing_sensed():
    b = _bridge(lambda now: None)
    assert b.capture(1.0, 'backstock_00', 'rollwagen') is None
    assert b.sensed_target_bboxes == ()


def test_stale_capture_is_rejected():
    b = _bridge(lambda now: _target(stamp=0.0))   # now-stamp = 1.0 > 0.5
    assert b.capture(1.0, 'backstock_00', 'rollwagen') is None


def test_not_parked_is_rejected():
    b = _bridge(lambda now: _target(stamp=now))
    b.motion = NS(parked=lambda: False)
    assert b.capture(1.0, 'backstock_00', 'rollwagen') is None


def test_no_provider_is_none():
    b = _bridge(None)
    assert b.capture(1.0, 'backstock_00', 'rollwagen') is None
