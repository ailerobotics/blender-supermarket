"""Opt-in integration-controlled sensing gate (CPU, no GPU).

During a controller EXECUTION phase after observe has validated the registration, the rendered
decode-camera frame is unused yet its whole-store RTX render is the dominant mission wall cost
(run-126). The gate lets integration SUSPEND that render and have the capture path republish the
last validated registration (PR 80 hold, occluded flag) with fresh motion-completion fields, so
/sensors keeps feeding the controller's parked/contact/joint/collision polling. Bounded by the same
occlusion window; default off and byte-identical when off. These tests exercise the gate semantics
directly on the real ``IsaacCartonSensors`` built by the reach-replay bench.
"""
import pytest

from scripts.twin.restock_manipulation.isaac_sensors import (
    sensing_render_required, _SENSING_SUSPEND_PHASES)

cv2 = pytest.importorskip('cv2')

from scripts.twin.tools.replay_reach import build_source_bench, FrameRecord  # noqa: E402


def test_sensing_render_required_policy():
    # render required for observe + the re-validation / unknown phases ...
    for phase in ('observe', 'preplace', 'verify', 'idle', 'host', 'warm', 'anything'):
        assert sensing_render_required(phase) is True
    # ... and NOT required (suspend) for the validated-reach execution phases.
    for phase in ('approach', 'contact', 'grasp', 'lift', 'place', 'retreat', 'release', 'stow',
                  'couple'):
        assert sensing_render_required(phase) is False
        assert phase in _SENSING_SUSPEND_PHASES


def _clean_observe(adapter, cam, motion, contact, loc, *, frame_id=5, now=1.0):
    """Render + validate one clean observe frame so the gate has a registration to hold."""
    cam.present(FrameRecord(0, frame_id, now, 'h'), occluded=False)
    motion.collision_time = now
    pkt = adapter.capture(now, loc, loc, motion, contact)
    assert pkt is not None and not adapter.evidence[0]['registration_occluded']
    return pkt


class _NoRenderCamera:
    """Stands in for the camera once the render is suspended: any frame fetch is a failure."""
    def get_current_frame(self, **_):
        raise AssertionError('render was fetched while suspended')
    def get_intrinsics_matrix(self):
        raise AssertionError('render was fetched while suspended')


def test_held_publish_needs_no_render_and_flags_occluded():
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=True)
    _clean_observe(adapter, cam, motion, contact, loc, now=1.0)
    adapter.camera = _NoRenderCamera()          # prove the held path never touches the camera
    adapter.set_render_suspended(True)
    motion.collision_time = 1.2
    pkt = adapter.capture(1.2, loc, loc, motion, contact)
    assert pkt.simulation_time == 1.2 and pkt.feedback_time == 1.2   # fresh tick + motion fields
    assert pkt.collision_time == 1.2 and len(pkt.joint_positions) == 6
    assert pkt.valid_until_steady is not None                        # a fresh steady deadline
    assert adapter.evidence and all(e['registration_occluded'] for e in adapter.evidence)
    assert adapter.sensed_target_bboxes == ()                        # no fresh detection claimed
    # resume: a real frame renders again and the republished occlusion clears.
    adapter.camera = cam
    adapter.set_render_suspended(False)
    pkt2 = _clean_observe(adapter, cam, motion, contact, loc, frame_id=9, now=1.3)
    assert pkt2.simulation_time == 1.3


def test_held_publish_refuses_beyond_the_occlusion_window():
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=5.0,
                                                            execution_gate=True)
    _clean_observe(adapter, cam, motion, contact, loc, now=1.0)
    adapter.camera = _NoRenderCamera()
    adapter.set_render_suspended(True)
    motion.collision_time = 1.0 + 5.001
    with pytest.raises(ValueError, match='beyond the occlusion window'):
        adapter.capture(1.0 + 5.001, loc, loc, motion, contact)


def test_held_publish_refuses_without_a_prior_registration():
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=True)
    adapter.camera = _NoRenderCamera()
    adapter.set_render_suspended(True)
    with pytest.raises(ValueError, match='no validated registration to hold'):
        adapter.capture(1.0, loc, loc, motion, contact)


def test_held_publish_refuses_a_moving_base():
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=True)
    _clean_observe(adapter, cam, motion, contact, loc, now=1.0)
    adapter.camera = _NoRenderCamera()
    adapter.set_render_suspended(True)
    motion.parked = lambda: False               # the held pose is not re-measured, so refuse
    motion.collision_time = 1.1
    with pytest.raises(ValueError, match='parked base'):
        adapter.capture(1.1, loc, loc, motion, contact)


def test_gate_off_is_byte_identical_suspension_flag_is_ignored():
    # Default off: set_render_suspended records the flag but capture ignores it -- the normal render
    # path runs, so the camera IS fetched and the result is a fresh (not occluded) frame.
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=False)
    _clean_observe(adapter, cam, motion, contact, loc, now=1.0)
    adapter.set_render_suspended(True)
    assert adapter._render_suspended is True
    pkt = _clean_observe(adapter, cam, motion, contact, loc, frame_id=7, now=1.2)   # renders normally
    assert pkt.simulation_time == 1.2 and not adapter.evidence[0]['registration_occluded']
