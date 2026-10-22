"""Record-first, ordered teardown for the mission launcher (CPU-pure; no Isaac/ROS imports).

A run that ends on a per-op manipulation failure must still leave its structured record
(``mission.json``) and exit without a core dump. Isaac's ``SimulationApp.close()`` can hard-exit
the process -- the renderer shutdown path (libcarb.graphics-vulkan -> librtx.* -> libomni.usd) has
core-dumped on a long / VRAM-pressured failure run -- so the launcher cannot rely on an outer
``finally`` that runs *after* ``app.close()`` to persist the record. Two invariants fix that, both
expressed here as pure functions so they run under a plain CPU pytest:

1. **Record first.** :func:`write_mission_record` is called on EVERY exit path (clean success,
   clean failure, and an exception out of the run) BEFORE any Isaac or ROS teardown begins, so the
   record is already on disk even if the renderer crashes during ``app.close()``.

2. **Ordered teardown.** :func:`ordered_teardown` runs the teardown steps in a fixed order with the
   RTX render products + camera feeds released BEFORE the app close -- the SAME order on the failure
   path as on a clean exit -- so the renderer shutdown has VRAM headroom and a single raising step
   never skips a later one.
"""
import json
from pathlib import Path


def write_mission_record(output_dir, summary, events):
    """Persist ``<output_dir>/mission.json`` from ``summary`` merged with ``events``, robustly.

    Record-first: call this on every exit path BEFORE any teardown. A shallow copy of ``summary``
    is written with ``events`` attached, so the caller's live ``summary`` dict is never mutated.
    Never raises (a teardown-time write failure must not mask the real outcome); returns True iff
    the file was written.
    """
    try:
        record = dict(summary)
        record['events'] = events
        path = Path(output_dir) / 'mission.json'
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        return True
    except Exception:  # noqa: BLE001 -- persistence must never raise into the teardown path
        return False


def ordered_teardown(steps, *, log=None):
    """Run teardown ``steps`` in order, each best-effort; return ``(invoked, errors)``.

    ``steps`` is a sequence of ``(name, callable_or_None)``. A ``None`` callable is skipped (so a
    run with no recorder / no app can pass the same list). Each callable runs inside its own
    try/except: a step that raises records its error and the NEXT step still runs -- teardown must
    always reach ``close_app`` last, even if an earlier step fails.

    The canonical mission order (built by the launcher) is::

        release_render     -> pause/destroy RTX render products + stop camera feeds (free VRAM)
        stop_adapter       -> park base, cancel nav, stop manip, park port
        close_bundle       -> stop recorder/verifier/ROS bridges (finalizes frames.json)
        encode_walkthrough -> encode the VFR H.264 from the on-disk frames
        close_app          -> SimulationApp.close()  (LAST; may hard-exit the process)

    so ``release_render`` (and the camera-feed stops folded into ``close_bundle``) always precede
    ``close_app`` -- the invariant the shutdown-sequence test asserts. Returns the list of step
    names actually invoked (in order) and a ``{name: error_str}`` dict for any that raised.
    """
    invoked = []
    errors = {}
    for name, call in steps:
        if call is None:
            continue
        invoked.append(name)
        try:
            call()
        except Exception as exc:  # noqa: BLE001 -- one failed step must not skip a later one
            errors[name] = str(exc)
            if log is not None:
                log('TEARDOWN_STEP_ERROR step=%s error=%r' % (name, exc))
    return invoked, errors
