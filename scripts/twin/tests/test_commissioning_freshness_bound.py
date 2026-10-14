"""The commissioning freshness window is opt-in; the live sensing bound stays 0.5 s.

`commission()` may use a wider static-scene acquisition-to-use stall guard
(`commissioning_freshness_s`, default 5 s), but every live sensing path
(`RenderedSupportContext.__call__`, `IsaacCartonSensors.capture`) calls
`frame_deadline` with its unchanged 0.5 s default. These are deterministic
clock-math checks — no renderer, physics or timing flake.
"""
import pytest

from scripts.twin.restock_camera.freshness import frame_deadline, require_fresh


def _frame(stamp=1.0, acquired=None, domain=''):
    return {'rendering_time': stamp, 'acquired_steady': acquired, 'steady_domain': domain}


def test_live_acquisition_to_use_bound_is_unchanged_half_second():
    started = 1000.0
    # now == stamp: the live default deadline is exactly started + 0.5.
    assert frame_deadline(_frame(1.0), 1.0, started) == pytest.approx(started + 0.5)
    # a use 0.6 s after commission start trips the live gate ...
    with pytest.raises(ValueError):
        require_fresh(frame_deadline(_frame(1.0), 1.0, started), lambda: started + 0.6)
    # ... while 0.4 s after start is still fresh.
    require_fresh(frame_deadline(_frame(1.0), 1.0, started), lambda: started + 0.4)
    # the live pre-check still rejects a frame already older than 0.5 s at call time.
    with pytest.raises(ValueError):
        frame_deadline(_frame(1.0), 1.6, started)  # now-stamp = 0.6 > 0.5


def test_commissioning_window_is_wider_and_opt_in():
    started = 1000.0
    deadline = frame_deadline(_frame(1.0), 1.0, started, budget_s=5.0)
    assert deadline == pytest.approx(started + 5.0)
    # the same 0.6 s processing that trips the live gate is fresh under the static-scene budget.
    require_fresh(deadline, lambda: started + 0.6)
    # and the wider budget tolerates a staler now-stamp pre-check that the live default rejects.
    frame_deadline(_frame(1.0), 3.0, started, budget_s=5.0)  # now-stamp = 2.0 ok under 5.0
    with pytest.raises(ValueError):
        frame_deadline(_frame(1.0), 3.0, started)            # now-stamp = 2.0 > 0.5 default


def test_commission_default_bound_is_five_seconds():
    import inspect
    from scripts.twin.restock_camera.binding import RenderedSupportContext
    sig = inspect.signature(RenderedSupportContext.commission)
    assert sig.parameters['commissioning_freshness_s'].default == 5.0
