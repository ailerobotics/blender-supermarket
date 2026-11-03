"""Record-first + ordered-teardown robustness for the mission launcher (CPU; no Isaac/ROS/GPU).

A run that ends on a per-op manipulation failure must still leave its ``mission.json`` and exit
without a core dump. Isaac's ``SimulationApp.close()`` can hard-exit in the renderer shutdown path
on a long / VRAM-pressured failure run, so the record MUST be written BEFORE any teardown and the
render products released BEFORE the app close. These exercise the two pure primitives
(:mod:`twin.restock.shutdown`) and the ``Bundle.release_render`` wiring, and compose them exactly
as ``run_mobile_restock.run``'s teardown finally does to prove the invariant survives a close that
raises (the core-dump proxy)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.shutdown import ordered_teardown, write_mission_record  # noqa: E402


# --- write_mission_record (record first) -------------------------------------------------------

def test_write_mission_record_merges_events_without_mutating_summary(tmp_path):
    summary = {'state': 'failed', 'reason': 'collision planning or execution failed'}
    ok = write_mission_record(tmp_path, summary, [{'action': 'transfer', 'status': 'failed'}])
    assert ok is True
    record = json.loads((tmp_path / 'mission.json').read_text())
    assert record['state'] == 'failed'
    assert record['reason'] == 'collision planning or execution failed'
    assert record['events'] == [{'action': 'transfer', 'status': 'failed'}]
    assert 'events' not in summary          # the caller's live summary is never mutated


def test_write_mission_record_never_raises_on_a_bad_record(tmp_path):
    # A non-finite value with allow_nan=False makes json.dumps raise; the write must swallow it
    # (a teardown-time persistence failure must not mask the real outcome) and report False.
    assert write_mission_record(tmp_path, {'state': 'failed', 'x': float('nan')}, []) is False
    assert not (tmp_path / 'mission.json').exists()
    # A non-existent output directory likewise degrades to False rather than raising.
    assert write_mission_record(tmp_path / 'missing', {'state': 'failed'}, []) is False


# --- ordered_teardown (shutdown sequence) ------------------------------------------------------

def test_ordered_teardown_runs_in_order_and_skips_none():
    seen = []
    invoked, errors = ordered_teardown([
        ('close_feeds', lambda: seen.append('close_feeds')),
        ('encode_walkthrough', None),                      # a run with no walkthrough: skipped
        ('release_render', lambda: seen.append('release_render')),
        ('close_app', lambda: seen.append('close_app')),
    ])
    assert seen == ['close_feeds', 'release_render', 'close_app']
    assert invoked == ['close_feeds', 'release_render', 'close_app']   # None not invoked
    assert errors == {}


def test_render_and_feeds_are_released_before_the_app_close():
    seen = []
    steps = [
        ('close_feeds', lambda: seen.append('close_feeds')),
        ('encode_walkthrough', lambda: seen.append('encode_walkthrough')),
        ('release_render', lambda: seen.append('release_render')),
        ('close_app', lambda: seen.append('close_app')),
    ]
    ordered_teardown(steps)
    # The invariant: feeds + render products are released before the renderer shuts down.
    assert seen.index('close_feeds') < seen.index('close_app')
    assert seen.index('release_render') < seen.index('close_app')
    assert seen[-1] == 'close_app'


def test_a_raising_step_never_skips_the_app_close():
    seen = []
    logs = []

    def boom():
        seen.append('close_feeds')
        raise RuntimeError('recorder died')

    invoked, errors = ordered_teardown([
        ('close_feeds', boom),
        ('release_render', lambda: seen.append('release_render')),
        ('close_app', lambda: seen.append('close_app')),
    ], log=logs.append)
    assert seen == ['close_feeds', 'release_render', 'close_app']   # app close still reached
    assert errors == {'close_feeds': 'recorder died'}
    assert any('close_feeds' in m for m in logs)


# --- composed as the launcher does it ----------------------------------------------------------

def test_failing_mission_leaves_the_record_even_when_app_close_core_dumps(tmp_path):
    """Compose write_mission_record + ordered_teardown exactly as run()'s finally does, with a
    close_app that raises (the renderer core-dump proxy). The record must already be on disk, and
    the render release must have run, before the app close is attempted."""
    summary = {'state': 'failed', 'reason': 'MANIP_HOST_FAILED op=transfer', 'physics_verified': False}
    events = [{'action': 'transfer', 'status': 'failed'}]
    released = []

    def close_app():
        # At the point the renderer shuts down, the record is already persisted and the render
        # products released -- prove it, then simulate the native hard-exit.
        assert (tmp_path / 'mission.json').exists()
        assert released == ['release_render']
        raise SystemExit('renderer SIGSEGV (core dump proxy)')

    # RECORD FIRST, then the ordered teardown -- the exact launcher order.
    write_mission_record(tmp_path, summary, events)
    with pytest.raises(SystemExit):
        ordered_teardown([
            ('close_feeds', lambda: None),
            ('release_render', lambda: released.append('release_render')),
            ('close_app', close_app),
        ])

    record = json.loads((tmp_path / 'mission.json').read_text())
    assert record['state'] == 'failed'
    assert record['reason'] == 'MANIP_HOST_FAILED op=transfer'
    assert record['events'] == events


def test_bundle_release_render_is_best_effort_and_never_raises():
    from twin.restock.isaac_bundle import Bundle, _render_closers
    from types import SimpleNamespace as NS

    class _Cam:                       # a raw camera with no hydra pause (the arm-feed shape)
        _render_product = None

    assembly = NS(sensing_sources={'cameras': {'src': _Cam(), 'cart': _Cam()},
                                   'location_cameras': {'G3_back': _Cam()}})
    closers = _render_closers(assembly)
    assert len(closers) == 3          # one per camera, station + location
    bundle = Bundle(adapter=NS(stop=lambda: None), _render_closers=closers)
    assert bundle.release_render() == 3     # attempts every closer, swallows the toggle AttributeError

    # No cameras (a partial/headless assembly) => no closers, release is a no-op.
    assert Bundle(adapter=NS(stop=lambda: None)).release_render() == 0
    assert _render_closers(NS(sensing_sources=None)) == ()
