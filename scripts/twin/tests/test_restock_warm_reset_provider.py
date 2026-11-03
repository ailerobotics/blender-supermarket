"""Warm-session reset PROVIDER: sequence, census, fallback paths, live-op wiring (CPU; no GPU).

``twin.restock.warm_reset_provider:build`` is the production provider ``config['warm_reset']`` names.
These exercise the full orchestration with an injected :class:`ResetOps` (the GPU-side re-seat is
validated by the integration lane): the ordered fail-closed sequence, the prim census that feeds the
``stage_clean`` gate, every step's named cold-boot refusal, the host-relaunch-hook contract, and the
real production ops bound to fake stage/world/app handles. No renderer/physics/ROS is involved.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import warm_reset_provider as wrp  # noqa: E402
from twin.restock import warm_session as ws  # noqa: E402


# --------------------------------------------------------------------------- helpers

def _ops(calls, *, list_prims=None, fail=None):
    """A fake ResetOps that records call order into ``calls`` and raises at the step named in ``fail``."""
    def _step(name, takes_kw=False):
        def fn(**kw):
            calls.append(name)
            if fail == name:
                raise RuntimeError(f'boom at {name}')
        return fn
    return wrp.ResetOps(
        list_prims=list_prims if list_prims is not None else (lambda: ['/A', '/B']),
        stop_runtime=_step('stop_runtime'),
        reset_stage=lambda leaked: (calls.append(('reset_stage', tuple(leaked))),
                                    (_ for _ in ()).throw(RuntimeError('boom at reset_stage'))
                                    if fail == 'reset_stage' else None),
        reseat=_step('reseat', True),
        relaunch_host=_step('relaunch_host', True),
        verify=_step('verify', True))


def _provider(calls, *, baseline=('/A', '/B'), list_prims=None, fail=None):
    return wrp.WarmResetProvider(ops=_ops(calls, list_prims=list_prims, fail=fail),
                                 baseline_prims=frozenset(baseline))


# --------------------------------------------------------------------------- census

def test_census_clean_when_present_matches_baseline():
    p = _provider([], baseline=('/A', '/B'), list_prims=lambda: ['/A', '/B'])
    st = p.stage_census()
    assert st.leaked() == () and st.missing() == () and not st.dirty


def test_census_names_leaked_mission_prims():
    p = _provider([], baseline=('/A', '/B'), list_prims=lambda: ['/A', '/B', '/World/carton', '/weld'])
    st = p.stage_census()
    assert st.leaked() == ('/World/carton', '/weld')
    # The leak makes the stage_clean gate cold-boot (via the pure core).
    d = ws.decide_reset({}, {}, stage_state=st, vram_used_mib=10, post_boot_baseline_mib=10)
    assert d.cold_boot and 'leaked prims' in d.reason


def test_census_failure_is_a_dirty_stage_not_a_crash():
    def boom():
        raise RuntimeError('stage gone')
    p = _provider([], list_prims=boom)
    st = p.stage_census()
    assert st.dirty and 'stage gone' in st.note


# --------------------------------------------------------------------------- sequence

def test_sequence_runs_every_step_in_order_on_success():
    calls = []
    p = _provider(calls, baseline=('/A', '/B'), list_prims=lambda: ['/A', '/B', '/leak'])
    assert p(attempt=1, now=1.0) is True
    names = [c[0] if isinstance(c, tuple) else c for c in calls]
    assert names == ['stop_runtime', 'reset_stage', 'reseat', 'relaunch_host', 'verify']
    # reset_stage was handed the leaked set from the census.
    reset = next(c for c in calls if isinstance(c, tuple) and c[0] == 'reset_stage')
    assert reset[1] == ('/leak',)


@pytest.mark.parametrize('bad', ['stop_runtime', 'reset_stage', 'reseat', 'relaunch_host', 'verify'])
def test_sequence_stops_at_first_refusal_and_names_the_step(bad):
    calls = []
    p = _provider(calls, fail=bad)
    with pytest.raises(wrp.WarmResetRefused) as ei:
        p(attempt=1, now=1.0)
    assert bad in str(ei.value)
    names = [c[0] if isinstance(c, tuple) else c for c in calls]
    # Every step up to and including the failing one ran; none after it.
    assert names[-1] == bad
    assert wrp.STEP_NAMES.index(bad) == len(names) - 1


# --------------------------------------------------------------------------- integration with the core

def test_provider_as_reset_step_reuses_a_clean_session():
    calls = []
    p = _provider(calls, baseline=('/A', '/B'), list_prims=lambda: ['/A', '/B'])
    step = ws.SequenceStep('reset_stage_and_reseed', lambda: p(attempt=2, now=2.0))
    out = ws.plan_and_run_reset({}, {}, [step], stage_state=p.stage_census(),
                                vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert out.reused and out.sequence.ok


def test_provider_refusal_becomes_a_named_cold_boot_through_the_core():
    calls = []
    p = _provider(calls, fail='relaunch_host')
    step = ws.SequenceStep('reset_stage_and_reseed', lambda: p(attempt=2, now=2.0))
    out = ws.plan_and_run_reset({}, {}, [step], stage_state=p.stage_census(),
                                vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert out.cold_boot and 'relaunch_host' in out.reason


# --------------------------------------------------------------------------- build()

def test_build_snapshots_the_post_boot_baseline_from_list_prims():
    ops = _ops([], list_prims=lambda: ['/Supermarket', '/Rollwagen'])
    p = wrp.build(app=None, assembly=None, parts={}, plan=None, config={}, output='/x', _ops=ops)
    assert p.baseline_prims == frozenset({'/Supermarket', '/Rollwagen'})
    assert callable(p.stage_census)


def test_build_refuses_when_the_baseline_census_fails():
    def boom():
        raise RuntimeError('no stage')
    ops = _ops([], list_prims=boom)
    with pytest.raises(wrp.WarmResetRefused):
        wrp.build(app=None, assembly=None, parts={}, plan=None, config={}, output='/x', _ops=ops)


# --------------------------------------------------- production ops bound to fake live handles

class _FakeStage:
    def __init__(self, paths):
        self._paths = list(paths)

    def Traverse(self):
        class _P:
            def __init__(s, path):
                s._p = path
            def GetPath(s):
                class _Pp:
                    pathString = s._p
                return _Pp()
        return [_P(p) for p in self._paths]

    def RemovePrim(self, path):
        self._paths.remove(path)


class _FakeScene:
    def __init__(self):
        self.removed = []

    def remove_object(self, name):
        self.removed.append(name)


class _FakeWorld:
    def __init__(self):
        self.current_time = 7.5
        self.reset_calls = 0
        self.scene = _FakeScene()

    def reset(self):
        self.reset_calls += 1
        self.current_time = 0.0


class _FakeApp:
    def __init__(self, running=True):
        self._running = running

    def is_running(self):
        return self._running


class _FakeAssembly:
    def __init__(self, stage, world):
        self.stage = stage
        self.world = world


def _prod(config, *, hooks=None, app=None, world=None, stage=None):
    stage = stage if stage is not None else _FakeStage(['/Supermarket', '/Rollwagen'])
    world = world if world is not None else _FakeWorld()
    assembly = _FakeAssembly(stage, world)
    return wrp._build_production_ops(app=app or _FakeApp(), assembly=assembly, parts={}, plan=None,
                                     config=config, output='/x', scenario={}, hooks=hooks or {})


def test_production_list_prims_traverses_the_stage():
    ops = _prod({})
    assert set(ops.list_prims()) == {'/Supermarket', '/Rollwagen'}


def test_production_reset_stage_prunes_resets_and_checks_the_clock():
    world = _FakeWorld()
    ops = _prod({}, world=world, stage=_FakeStage(['/Supermarket', '/World/carton']))
    ops.reset_stage(['/World/carton'])
    assert world.reset_calls == 1 and world.current_time == 0.0


def test_production_reset_stage_refuses_when_clock_not_reset():
    class _StuckWorld(_FakeWorld):
        def reset(self):
            self.reset_calls += 1  # clock stays non-zero
    with pytest.raises(wrp.WarmResetRefused):
        _prod({}, world=_StuckWorld()).reset_stage([])


def test_production_stop_runtime_refuses_if_kit_died():
    ops = _prod({}, app=_FakeApp(running=False))
    with pytest.raises(wrp.WarmResetRefused):
        ops.stop_runtime()


def test_production_relaunch_host_refuses_without_a_lane_hook():
    with pytest.raises(wrp.WarmResetRefused):
        _prod({}).relaunch_host(attempt=1, now=1.0)


def test_production_relaunch_host_uses_the_lane_hook():
    seen = {}
    hooks = {'relaunch_host': lambda **k: seen.update(k)}
    _prod({}, hooks=hooks).relaunch_host(attempt=3, now=9.0)
    assert seen == {'attempt': 3, 'now': 9.0}


def test_production_relaunch_host_accepts_a_reset_host_state_hook():
    seen = {}
    hooks = {'reset_host_state': lambda **k: seen.update(k)}
    _prod({}, hooks=hooks).relaunch_host(attempt=2, now=4.0)
    assert seen == {'attempt': 2, 'now': 4.0}


def test_malformed_warm_reset_hooks_is_refused():
    with pytest.raises(ValueError):
        wrp._resolve_hooks({'warm_reset_hooks': 'not-a-ref'}, app=None, assembly=None, parts={},
                           plan=None, output='/x')
