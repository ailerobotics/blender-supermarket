"""Acquisition-to-use deadlines on the native process monotonic clock."""
import math
import time


# A rendered frame reflects the state just AFTER a physics step, so its render clock can lead
# the consumer's dt-accumulated sim clock by up to one physics step (1/60 s ~= 0.0167 s). Allow
# that sub-step lead as not-from-the-future; genuine future/stale frames are still rejected.
_RENDER_LEAD=0.02


def frame_deadline(frame, now, started, budget_s=.5, *, use_acquired=True):
    """Steady-clock deadline for using a frame acquired at ``now``.

    ``budget_s`` is the acquisition-to-use budget: the live mission capture uses the standing
    0.5 s gate; a static-scene commissioning pass passes its larger ``commissioning_freshness_s``
    so a slow full-resolution decode+calibrate does not trip the live budget.

    ``use_acquired`` folds the frame's wall acquisition-callback stamp (``acquired_steady``) into the
    deadline (queue age). The live mission capture, which measures freshness on the SIM/node clock
    (``started`` is sim seconds) under render pacing, sets it False: the wall acquisition stamp is a
    different clock domain and its wall age is not real staleness — the SIM age ``now - stamp`` (the
    render's sim time) already bounds queue age in the clock the budget is measured in.
    """
    stamp=float(frame['rendering_time'])
    if not all(math.isfinite(v) for v in (now,started,stamp,budget_s)) or not -_RENDER_LEAD<=now-stamp<=budget_s:
        raise ValueError('stale acquisition before processing')
    deadline=started+budget_s-(now-stamp)
    if use_acquired:
        acquired=frame.get('acquired_steady')
        if bool(frame.get('steady_domain')) != (acquired is not None):
            raise ValueError('incomplete acquisition clock provenance')
        if acquired is not None:
            if not math.isfinite(acquired) or acquired>started:
                raise ValueError('invalid or future acquisition clock')
            deadline=min(deadline,acquired+budget_s)
    return deadline


def require_fresh(deadline, clock, *, wall_started=None, wall_backstop_s=None, wall_clock=time.monotonic):
    """Refuse a frame used after its acquisition-to-use ``deadline`` on ``clock`` (read after
    processing, never the caller's old ``now``).

    When ``clock`` is the SIM/node clock (render-paced mission), a long full-resolution render barely
    advances it, so the sim budget holds — the correct semantics (real staleness, not render wall
    time). A generous ``wall_backstop_s`` on the real monotonic clock still hard-stops a genuinely
    hung render whose sim clock is frozen (the sim deadline can never advance to trip then). Mirrors
    the node-clocked status watchdog in ``isaac_ros_graph`` (sim bound + wall backstop).
    """
    current=clock()
    if not math.isfinite(current) or not math.isfinite(deadline) or current>deadline:
        raise ValueError('acquisition-to-use freshness expired after processing')
    if wall_backstop_s is not None and wall_started is not None:
        if not math.isfinite(wall_backstop_s) or wall_clock()-wall_started>wall_backstop_s:
            raise ValueError('acquisition-to-use wall backstop exceeded')
