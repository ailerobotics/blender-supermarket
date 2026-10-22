"""fm384 phase-aware collect capture: after grasp the capture renders the DESTINATION(cart) camera
ONLY (source observations no longer needed; the single target already comes from the destination and
the destination renders a REAL fresh frame), so the source render product can be freed pre-peak with
NO fabricated freshness. These tests drive the pure StationSensors.capture logic with fake adapters
(no Isaac): the default renders source+cart (byte-identical), release_source() switches to cart-only
with zero source renders, and require_fresh / one-target / sync integrity still pass on genuinely fresh
cart frames -- including over a simulated 300 s hold."""
import types
import pytest

from twin.restock_camera.binding import StationSensors


class _Adapter:
    """Fake per-camera adapter: records capture() calls (render count) and returns a packet."""
    def __init__(self, cam_id, *, target, location):
        self.cam_id, self._target, self.location = cam_id, target, location
        self.calls = 0
        self.slots = (types.SimpleNamespace(location=location),)
        self.sensed_target_bboxes = ()
        self.evidence = ()

    def capture(self, now, source, destination, motion, contact):
        self.calls += 1
        return types.SimpleNamespace(
            target=self._target, scan_time=now, valid_until_steady=now + 5.0,
            observations=(), parked=True, contact=False, feedback_time=now,
            collision_time=now, joint_positions=(0.,)*6, scanned_locations=(self.location,))


def _station(sim_now_ref):
    """A fake StationSensors self: cart provides the target (destination=rollwagen); source does not
    (target=None), mirroring binding's target=targets.get(destination)."""
    src = _Adapter('cam_source', target=None, location='backstock_00')
    cart = _Adapter('cam_cart', target=(1.0, 2.0, 3.0), location='rollwagen')
    s = types.SimpleNamespace(
        location_cameras={'backstock_00': 'cam_source', 'rollwagen': 'cam_cart'},
        adapters={'cam_source': src, 'cam_cart': cart},
        steady_clock=lambda: sim_now_ref[0], wall_clock=lambda: 0.0, wall_backstop_s=60.0,
        sensed_target_bboxes=(), evidence=())
    return s, src, cart


def _capture(s, now):
    return StationSensors.capture(s, now, 'backstock_00', 'rollwagen', motion=None, contact=None)


def test_byte_identical_off_renders_source_and_cart():
    now = [0.0]
    s, src, cart = _station(now)
    out = _capture(s, 1.0)                        # no release_source -> default path
    assert src.calls == 1 and cart.calls == 1     # BOTH cameras rendered, as today
    assert out.target == (1.0, 2.0, 3.0)          # single target from the cart (destination)


def test_release_source_switches_to_cart_only_zero_source_renders():
    now = [0.0]
    s, src, cart = _station(now)
    StationSensors.release_source(s)              # post-grasp
    out = _capture(s, 1.0)
    assert src.calls == 0                          # SOURCE never rendered after release
    assert cart.calls == 1                         # destination rendered (real fresh)
    assert out.target == (1.0, 2.0, 3.0)           # exactly-one-target satisfied by the cart


def test_release_source_sets_flag_idempotent():
    now = [0.0]
    s, _, _ = _station(now)
    assert getattr(s, '_source_released', False) is False
    StationSensors.release_source(s)
    StationSensors.release_source(s)
    assert s._source_released is True


def test_cart_only_packet_passes_require_fresh_and_one_target():
    now = [100.0]
    s, src, cart = _station(now)
    StationSensors.release_source(s)
    # cart deadline = now+5 (genuinely fresh vs steady_clock==100) -> require_fresh passes; one target.
    out = _capture(s, 100.0)
    assert out is not None and out.target == (1.0, 2.0, 3.0)
    assert src.calls == 0


def test_watchdog_satisfied_by_real_cart_frames_over_300s():
    now = [0.0]
    s, src, cart = _station(now)
    StationSensors.release_source(s)
    # advance the sim clock to 300 s; each tick a REAL fresh cart frame (deadline = now+5) -> no raise.
    t = 0.0
    while t <= 300.0:
        now[0] = t
        out = _capture(s, t)                       # would raise ValueError if freshness/one-target failed
        assert out.target == (1.0, 2.0, 3.0)
        t += 10.0
    assert src.calls == 0                          # zero source renders across the whole hold
    assert cart.calls == 31                        # 0,10,...,300


def test_cart_only_rejects_stale_cart_frame():
    # Honesty guard: if the cart frame is NOT fresh, require_fresh still refuses (no fabrication rescue).
    now = [1000.0]
    s, src, cart = _station(now)
    StationSensors.release_source(s)
    cart._target = (1.0, 2.0, 3.0)
    # make the cart deadline stale: capture() sets valid_until_steady = now+5, so steady_clock must be
    # AFTER that. Set steady_clock well past the packet deadline.
    s.steady_clock = lambda: 2000.0
    with pytest.raises(ValueError):
        _capture(s, 1000.0)
