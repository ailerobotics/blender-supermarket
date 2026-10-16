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


def test_suspend_disables_the_decode_render_product_and_resume_re_enables_it_on_edges():
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=True)
    _clean_observe(adapter, cam, motion, contact, loc, now=1.0)
    assert cam.render_products_enabled is True and cam.product_toggles == []
    # suspend edge -> product disabled (world.step(render=True) would skip the decode render)
    adapter.set_render_suspended(True)
    assert cam.render_products_enabled is False and cam.product_toggles == [False]
    adapter.set_render_suspended(True)                  # steady-state tick: no extra toggle
    assert cam.product_toggles == [False]
    # resume edge -> product re-enabled
    adapter.set_render_suspended(False)
    assert cam.render_products_enabled is True and cam.product_toggles == [False, True]
    adapter.set_render_suspended(False)                 # steady-state tick: no extra toggle
    assert cam.product_toggles == [False, True]


def test_first_non_suspended_call_enables_the_product_at_startup():
    # run-146..148: with the gate on, the decode render product was left OFF at mission start because
    # the FIRST set_render_suspended(False) (phase=observe, was=False at init) was a no-edge no-op, so
    # the source never rendered -> "no initial sensor frame". The idempotent drive must ENABLE the
    # product on that first non-suspended call regardless of a native default-off or a persisted
    # disable.
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=True)
    cam.render_products_enabled = False                 # native default-off / persisted disable at boot
    cam.product_toggles.clear()
    adapter.set_render_suspended(False)                 # first startup tick, not suspended
    assert cam.render_products_enabled is True and cam.product_toggles == [True]
    adapter.set_render_suspended(False)                 # steady state: idempotent, no re-toggle
    assert cam.product_toggles == [True]


def test_gate_off_does_not_touch_the_product_at_startup():
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=False)
    cam.render_products_enabled = False
    cam.product_toggles.clear()
    adapter.set_render_suspended(False)
    adapter.set_render_suspended(True)
    assert cam.product_toggles == []                    # gate off never manages the product


def test_resume_forces_a_fresh_render_not_a_stale_product_frame():
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=True)
    _clean_observe(adapter, cam, motion, contact, loc, frame_id=5, now=1.0)
    adapter.set_render_suspended(True)                  # hold through a reach (no frames)
    motion.collision_time = 1.2
    adapter.capture(1.2, loc, loc, motion, contact)     # held republish, no render
    adapter.set_render_suspended(False)                 # resume: reuse snapshot dropped
    # A STALE decode-product frame (same id/stamp as before suspension) must NOT be reused: the
    # monotonic-frame gate refuses it until a genuinely new render lands.
    cam.present(FrameRecord(0, 5, 1.0, 'h'), occluded=False)
    motion.collision_time = 1.3
    # refused by a production gate (anti-replay / monotonic frame / stale evidence) -- NOT reused.
    with pytest.raises(ValueError, match='no new synchronized|camera evidence|replayed|stale'):
        adapter.capture(1.3, loc, loc, motion, contact)
    # a fresh render (new id/stamp) is accepted again -- exact-frame identity/gates unchanged.
    pkt = _clean_observe(adapter, cam, motion, contact, loc, frame_id=9, now=1.4)
    assert pkt.simulation_time == 1.4 and not adapter.evidence[0]['registration_occluded']


def test_gate_off_never_toggles_the_render_product():
    adapter, cam, motion, contact, loc = build_source_bench(occlusion_window_s=20.0,
                                                            execution_gate=False)
    _clean_observe(adapter, cam, motion, contact, loc, now=1.0)
    adapter.set_render_suspended(True)
    adapter.set_render_suspended(False)
    assert cam.render_products_enabled is True and cam.product_toggles == []   # untouched when off


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
