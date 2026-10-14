"""Acquisition-to-use deadlines on the native process monotonic clock."""
import math


# A rendered frame reflects the state just AFTER a physics step, so its render clock can lead
# the consumer's dt-accumulated sim clock by up to one physics step (1/60 s ~= 0.0167 s). Allow
# that sub-step lead as not-from-the-future; genuine future/stale frames are still rejected.
_RENDER_LEAD=0.02


def frame_deadline(frame, now, started):
    stamp=float(frame['rendering_time'])
    if not all(math.isfinite(v) for v in (now,started,stamp)) or not -_RENDER_LEAD<=now-stamp<=.5:
        raise ValueError('stale acquisition before processing')
    deadline=started+.5-(now-stamp)
    acquired=frame.get('acquired_steady')
    if bool(frame.get('steady_domain')) != (acquired is not None):
        raise ValueError('incomplete acquisition clock provenance')
    if acquired is not None:
        if not math.isfinite(acquired) or acquired>started:
            raise ValueError('invalid or future acquisition clock')
        deadline=min(deadline,acquired+.5)
    return deadline


def require_fresh(deadline, clock):
    current=clock()  # Read after processing; never reuse the caller's old `now`.
    if not math.isfinite(current) or not math.isfinite(deadline) or current>deadline:
        raise ValueError('acquisition-to-use freshness expired after processing')
