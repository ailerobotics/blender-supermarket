"""Production warm-reset host-lifecycle hooks: ordering, live-handle wiring, the loud cold-boot
fallback, and the provider wiring that makes ``relaunch_host`` ACCEPT the in-repo default (CPU; no
Isaac/ROS/GPU). ``twin.restock.warm_reset_hooks:build`` is the factory ``config['warm_reset_hooks']``
resolves (and the provider default); these drive it with fake host handles so the whole orchestration --
the ordered sequence, the state reset, and the named ``WARM_RESET_HOOK_FALLBACK`` fallback -- is proven
offline. The GPU-side effect is validated by the integration lane at acceptance.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import warm_reset_hooks as wrh  # noqa: E402
from twin.restock import warm_reset_provider as wrp  # noqa: E402
from twin.restock import warm_session as ws  # noqa: E402


# --------------------------------------------------------------------------- fakes

def _ops(calls, *, fail=None, faulted_after=False, app_running=True):
    """A fake HostOps that records call order into ``calls`` and raises at the step named in ``fail``."""
    def _step(name):
        def fn():
            calls.append(name)
            if fail == name:
                raise RuntimeError(f'boom at {name}')
        return fn
    return wrh.HostOps(
        warm_safe_halt=_step('warm_safe_halt'),
        release_sim_attachments=_step('release_sim_attachments'),
        restore_sensing=_step('restore_sensing'),
        arm_to_start=_step('arm_to_start'),
        clear_manipulation=_step('clear_manipulation'),
        rearm_heartbeats=_step('rearm_heartbeats'),
        arm_faulted=lambda: faulted_after,
        app_running=lambda: app_running)


def _hooks(calls, **kw):
    return wrh.HostLifecycleHooks(ops=_ops(calls, **kw))


class _FakeGate:
    """A stand-in for ArmCommandGate: hold() latches faulted, resume() clears it (both recorded)."""
    def __init__(self, faulted=True):
        self.faulted = faulted
        self.events = []

    def hold(self):
        self.faulted = True
        self.events.append('hold')

    def resume(self):
        self.faulted = False
        self.events.append('resume')


class _FakeMobile:
    def __init__(self):
        self.parked = 0

    def park(self):
        self.parked += 1


class _FakeNav:
    def __init__(self):
        self.cancelled = 0

    def cancel(self):
        self.cancelled += 1


class _FakeManip:
    """A manipulation runtime whose stop() MUST NOT be called by the warm-safe halt (it closes the
    station-level sensing the next run reuses). reset_crash_budget is the between-op latch clear."""
    def __init__(self, with_budget=True):
        self.stop_calls = 0
        self.budget_resets = 0
        if with_budget:
            self.reset_crash_budget = self._reset_budget

    def stop(self):
        self.stop_calls += 1

    def _reset_budget(self):
        self.budget_resets += 1


class _FakePort:
    def __init__(self, gate, *, hold_arm=True, via_transport=False):
        self._hold_count = 0
        if hold_arm:
            self.hold_arm = self._hold_arm
        if via_transport:
            self.native_transport = type('NT', (), {'command_gate': gate})()
        else:
            self.command_gate = gate

    def _hold_arm(self):
        self._hold_count += 1


class _FakeApp:
    def __init__(self, running=True):
        self._running = running

    def is_running(self):
        return self._running


def _parts(*, gate=None, manip=None, hold_arm=True, via_transport=False, with_budget=True):
    gate = gate if gate is not None else _FakeGate()
    return {
        'port': _FakePort(gate, hold_arm=hold_arm, via_transport=via_transport),
        'mobile': _FakeMobile(),
        'navigation': _FakeNav(),
        'manipulation': manip if manip is not None else _FakeManip(with_budget=with_budget),
        'adapter': object(),
    }, gate


# --------------------------------------------------------------------------- reset_host_state ordering

def test_reset_host_state_runs_every_step_in_order():
    calls = []
    assert _hooks(calls).reset_host_state(attempt=2, now=2.0) is True
    assert calls == list(wrh.RESET_STEP_NAMES) == [
        'release_sim_attachments', 'restore_sensing',
        'arm_to_start', 'clear_manipulation', 'rearm_heartbeats']


@pytest.mark.parametrize('bad', ['release_sim_attachments', 'restore_sensing',
                                 'arm_to_start', 'clear_manipulation', 'rearm_heartbeats'])
def test_reset_host_state_stops_at_first_refusal_and_names_the_step(bad, capsys):
    calls = []
    with pytest.raises(wrp.WarmResetRefused) as ei:
        _hooks(calls, fail=bad).reset_host_state(attempt=1, now=1.0)
    assert bad in str(ei.value)
    # Every step up to and including the failing one ran; none after it.
    assert calls[-1] == bad
    assert wrh.RESET_STEP_NAMES.index(bad) == len(calls) - 1
    # The LOUD, named cold-boot fallback line was emitted (never a silent half-reset).
    out = capsys.readouterr().out
    assert 'WARM_RESET_HOOK_FALLBACK' in out and f'step={bad}' in out and 'cold boot' in out


def test_stop_runtime_loud_fallback_on_failed_halt(capsys):
    calls = []
    with pytest.raises(wrp.WarmResetRefused):
        _hooks(calls, fail='warm_safe_halt').stop_runtime()
    out = capsys.readouterr().out
    assert 'WARM_RESET_HOOK_FALLBACK stop_runtime refused' in out and 'cold boot' in out


def test_verify_refuses_when_app_down_or_gate_faulted(capsys):
    with pytest.raises(wrp.WarmResetRefused):
        _hooks([], app_running=False).verify(now=1.0)
    assert 'SimulationApp not running' in capsys.readouterr().out
    with pytest.raises(wrp.WarmResetRefused):
        _hooks([], faulted_after=True).verify(now=1.0)
    assert 'arm gate still faulted' in capsys.readouterr().out


def test_verify_passes_on_a_clean_reset():
    _hooks([], app_running=True, faulted_after=False).verify(now=1.0)  # no raise


# --------------------------------------------------- live ops bound to fake host handles

def test_warm_safe_halt_parks_cancels_and_holds_without_closing_sensing():
    parts, gate = _parts()
    ops = wrh._build_live_ops(app=_FakeApp(), assembly=None, parts=parts)
    ops.warm_safe_halt()
    assert parts['mobile'].parked == 1 and parts['navigation'].cancelled == 1
    assert parts['port']._hold_count == 1               # arm held via port.hold_arm
    assert parts['manipulation'].stop_calls == 0        # station-level sensing left alive (warm-safe)


def test_warm_safe_halt_falls_back_to_gate_hold_when_no_hold_arm():
    parts, gate = _parts(hold_arm=False)
    wrh._build_live_ops(app=_FakeApp(), assembly=None, parts=parts).warm_safe_halt()
    assert gate.events == ['hold'] and gate.faulted


def test_warm_safe_halt_refuses_with_no_arm_hold_path():
    port = type('P', (), {})()   # no hold_arm, no command_gate, no native_transport
    parts = {'port': port, 'mobile': _FakeMobile(), 'navigation': _FakeNav()}
    with pytest.raises(wrp.WarmResetRefused):
        wrh._build_live_ops(app=_FakeApp(), assembly=None, parts=parts).warm_safe_halt()


def test_arm_to_start_re_arms_the_gate_clearing_a_latched_fault():
    parts, gate = _parts(gate=_FakeGate(faulted=True))
    wrh._build_live_ops(app=_FakeApp(), assembly=None, parts=parts).arm_to_start()
    assert gate.events == ['resume'] and gate.faulted is False


def test_arm_to_start_reaches_the_gate_behind_native_transport():
    parts, gate = _parts(gate=_FakeGate(faulted=True), via_transport=True)
    wrh._build_live_ops(app=_FakeApp(), assembly=None, parts=parts).arm_to_start()
    assert gate.faulted is False


def test_arm_to_start_refuses_when_no_gate_is_reachable():
    port = type('P', (), {})()
    parts = {'port': port}
    with pytest.raises(wrp.WarmResetRefused):
        wrh._build_live_ops(app=_FakeApp(), assembly=None, parts=parts).arm_to_start()


def test_clear_manipulation_resets_the_crash_budget_best_effort():
    manip = _FakeManip(with_budget=True)
    parts, _ = _parts(manip=manip)
    wrh._build_live_ops(app=_FakeApp(), assembly=None, parts=parts).clear_manipulation()
    assert manip.budget_resets == 1


def test_clear_manipulation_skips_an_absent_budget_handle_without_raising():
    manip = _FakeManip(with_budget=False)
    parts, _ = _parts(manip=manip)
    # No reset_crash_budget anywhere => a named skip, not a crash.
    wrh._build_live_ops(app=_FakeApp(), assembly=None, parts=parts).clear_manipulation()


def test_reset_host_state_against_live_handles_resets_state_in_order():
    parts, gate = _parts(gate=_FakeGate(faulted=True))
    ops = wrh._build_live_ops(app=_FakeApp(), assembly=None, parts=parts)
    hooks = wrh.HostLifecycleHooks(ops=ops)
    assert hooks.reset_host_state(attempt=2, now=2.0) is True
    assert gate.faulted is False and gate.events == ['resume']
    assert parts['manipulation'].budget_resets == 1
    hooks.verify(now=2.0)  # clean reset: app running + gate un-faulted


def test_build_is_lazy_with_none_handles():
    # The provider builds hooks with app=None/assembly=None/parts={} in its own CPU tests; constructing
    # must not touch any (absent) handle.
    hooks = wrh.build(app=None, assembly=None, parts={}, plan=None, config={}, output='/x')
    assert isinstance(hooks, wrh.HostLifecycleHooks)


# --------------------------------------------------- provider wiring: relaunch_host now ACCEPTS

def test_provider_default_hooks_resolve_to_the_in_repo_factory():
    # With no config['warm_reset_hooks'], the provider defaults to the in-repo production factory, which
    # supplies reset_host_state (+ stop_runtime + verify).
    hooks = wrp._resolve_hooks({}, app=None, assembly=None, parts={}, plan=None, output='/x')
    assert callable(hooks.get('reset_host_state'))
    assert callable(hooks.get('stop_runtime')) and callable(hooks.get('verify'))
    assert 'relaunch_host' not in hooks   # reset_host_state satisfies the step; no OS-process relaunch


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


class _FakeWorld:
    def __init__(self):
        self.current_time = 3.0
        self.scene = type('S', (), {'remove_object': lambda self, n: None})()

    def reset(self):
        self.current_time = 0.0


class _FakeAssembly:
    def __init__(self):
        self.stage = _FakeStage(['/Supermarket', '/Rollwagen'])
        self.world = _FakeWorld()


def _provider_prod_ops(parts, *, config=None):
    """Build the REAL provider production ops with the REAL default hooks resolved for a fake host, so
    the provider's relaunch_host step runs the in-repo reset_host_state end to end."""
    config = config or {}
    app = _FakeApp()
    assembly = _FakeAssembly()
    hooks = wrp._resolve_hooks(config, app=app, assembly=assembly, parts=parts, plan=None, output='/x')
    return wrp._build_production_ops(app=app, assembly=assembly, parts=parts, plan=None,
                                     config=config, output='/x', scenario={}, hooks=hooks), gate_of(parts)


def gate_of(parts):
    port = parts['port']
    return getattr(port, 'command_gate', None) or getattr(getattr(port, 'native_transport', None),
                                                           'command_gate', None)


def test_provider_relaunch_host_now_accepts_via_default_reset_host_state():
    parts, gate = _parts(gate=_FakeGate(faulted=True))
    ops, _ = _provider_prod_ops(parts)
    # Before PR sim-warm-reset-hooks-v1 this refused ("no host relaunch/reset hook wired"); now it runs
    # the in-repo reset_host_state and re-arms the gate.
    ops.relaunch_host(attempt=2, now=2.0)
    assert gate.faulted is False


def test_provider_stop_runtime_uses_the_warm_safe_hook_not_adapter_stop():
    manip = _FakeManip()
    parts, _ = _parts(manip=manip)
    # The adapter would close sensing; the default stop_runtime hook must be used instead.
    parts['adapter'] = type('A', (), {'stop': lambda self: manip.stop()})()
    ops, _ = _provider_prod_ops(parts)
    ops.stop_runtime()
    assert manip.stop_calls == 0 and parts['mobile'].parked == 1


def test_full_provider_sequence_reuses_a_clean_session_through_the_core():
    parts, gate = _parts(gate=_FakeGate(faulted=True))
    app = _FakeApp()
    assembly = _FakeAssembly()
    provider = wrp.build(app=app, assembly=assembly, parts=parts, plan=None, config={}, output='/x',
                         scenario={})
    step = ws.SequenceStep('warm_reset_provider', lambda: provider(attempt=2, now=2.0))
    out = ws.plan_and_run_reset({}, {}, [step], stage_state=provider.stage_census(),
                                vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert out.reused and out.sequence.ok
    assert gate.faulted is False   # the host state was reset through the full provider sequence


def test_full_provider_sequence_cold_boots_loudly_on_a_missing_gate(capsys):
    # A port with no reachable gate => arm_to_start refuses => the provider relaunch_host step fails =>
    # a NAMED cold-boot fallback through the warm_session core (never a silent half-reset).
    port = _FakePort(_FakeGate(), hold_arm=True)
    del port.command_gate   # make the gate unreachable for arm_to_start
    parts = {'port': port, 'mobile': _FakeMobile(), 'navigation': _FakeNav(),
             'manipulation': _FakeManip(), 'adapter': object()}
    app = _FakeApp()
    assembly = _FakeAssembly()
    provider = wrp.build(app=app, assembly=assembly, parts=parts, plan=None, config={}, output='/x',
                         scenario={})
    step = ws.SequenceStep('warm_reset_provider', lambda: provider(attempt=2, now=2.0))
    out = ws.plan_and_run_reset({}, {}, [step], stage_state=provider.stage_census(),
                                vram_used_mib=8300, post_boot_baseline_mib=8200)
    assert out.cold_boot and 'relaunch_host' in out.reason and 'arm_to_start' in out.reason
    assert 'WARM_RESET_HOOK_FALLBACK' in capsys.readouterr().out
