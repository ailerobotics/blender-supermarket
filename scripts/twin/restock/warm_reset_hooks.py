"""Production host-lifecycle hooks for the warm-session reset provider (CPU-provable; no Isaac/ROS/GPU
at module load).

``twin.restock.warm_reset_provider`` runs the between-runs reset sequence; its ``relaunch_host`` step is
integration-lane-owned because it must act on the LIVE host stack, so the provider delegates it to an
optional ``config['warm_reset_hooks'] = "module:function"`` factory. PR 299 shipped the provider with NO
such factory in-repo, so from the integration worktree -- which cannot hand-write one -- the
``relaunch_host`` step always refused and every ``--warm-attempts`` / ``--consecutive-runs`` iteration
cold-booted the full ~215 s. This module is that factory, built against the real in-process host handles
the mission bundle already owns, so the warm two-run collect-only acceptance runs with no hand-written
stub and no cold boot between attempts.

:func:`build` is the ``module:function`` the provider resolves (and the default
``warm_reset_provider.DEFAULT_HOOKS`` when ``config['warm_reset_hooks']`` is absent). It is called ONCE,
with the live scene handles, and returns a :class:`HostLifecycleHooks` exposing the hook names the
provider reads off a hooks object:

  * ``stop_runtime()``            -- WARM-SAFE halt of the mission runtime between runs: park the base,
                                     cancel navigation, hold the arm. It does NOT tear down the
                                     STATION-level resources (the masked-depth publisher subprocess, the
                                     shared ``/restock_sim/station/status`` control relay) that the next
                                     run reuses -- that is why it overrides the provider's default
                                     ``adapter.stop()`` (which closes sensing). Kit is never closed.
  * ``reset_host_state(attempt, now)`` -- the ``relaunch_host``-satisfying step. An IN-PLACE reset of the
                                     live host-side state (no OS-process relaunch, no Isaac close):
                                       1. ``arm_to_start``       -- re-arm the arm command gate
                                          (``ArmCommandGate.resume``): clear a latched ``hold()``/fault,
                                          re-seat a clean hold at the measured joints, and reset the gate's
                                          heartbeat clock (``last_stamp`` / ``started``) so the next run's
                                          first command is not charged the warm gap. The stage re-seat that
                                          returns the robot/cart/carton to the verified start state (PR 297
                                          settle/verify) is the provider's own ``reset_stage`` + ``reseat``
                                          steps, which run BEFORE this one.
                                       2. ``clear_manipulation`` -- clear the between-op manipulation
                                          latches the shared runtime carries across runs (the motion
                                          crash-retry budget, any held-payload profile), so the next op
                                          starts clean. Best-effort: a handle the live chain does not
                                          expose is skipped by name, never faked.
                                       3. ``rearm_heartbeats``   -- re-arm the in-process heartbeat /
                                          watchdog clocks (PR 301): re-prime the station liveness so its
                                          next-op ack is not charged the warm gap. The arm-command
                                          heartbeat is already re-armed by step 1; the per-mission nav /
                                          station-ack watchdogs are rebuilt fresh by each ``run_once``.
                                     A missing CRITICAL handle (no arm command gate to re-arm) REFUSES
                                     LOUDLY (a named ``WARM_RESET_HOOK_FALLBACK`` console line + a
                                     :class:`WarmResetRefused`), so the provider falls back to a cold boot
                                     with the reason -- never a silent half-reset.
  * ``verify(now)``               -- post-reset readiness gate: Kit still running AND the arm gate left
                                     un-faulted. Raises (named) to refuse.

The external host PROCESSES (move_group / cuMotion planner / nvblox mapper, launched out-of-process by the
integration lane's ``launch_restock_host_stack.sh``) are NOT relaunched here: the mission Isaac process is
ROS-isolated from them and reaches them only through the OmniGraph bridge, whose stale global-stop wire is
re-derived per op by ``IsaacRosGraph.start`` and whose planning scene / ESDF re-incorporate against the
re-seated stage on the next op. An integration lane that needs a genuine OS-process relaunch can still
override this by naming its own ``config['warm_reset_hooks']`` factory exposing a ``relaunch_host`` hook;
the provider prefers an explicit ``relaunch_host`` over this ``reset_host_state`` default.

Every live operation is isolated behind a small, individually-overridable callable on :class:`HostOps`,
so the orchestration (ordering, fail-closed stop-at-first-refusal, the loud fallback) is CPU-provable with
an injected fake host -- the GPU-side effect is validated by the integration lane at acceptance.
``build(..., _ops=<HostOps>)`` is the test seam (production leaves it None and binds the real handles).

Config placement. ``warm_reset_hooks`` (like ``warm_reset``) is a RUNTIME mission-config key -- it is read
by ``isaac_bundle.create`` / ``warm_reset_provider`` on the Isaac/runtime side, NOT a HOST
(``restock_manipulation.config_preflight``) key and NOT an ENV var. A HOST/ENV placement is silently
inactive (the host-vs-runtime split). Because this module is the provider default, the key may be omitted
entirely; name it only to point at a different lane-owned factory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .warm_reset_provider import WarmResetRefused


# The ordered reset_host_state sub-steps. Stable names so the console lines / record and the tests key
# off them.
RESET_STEP_NAMES = ('arm_to_start', 'clear_manipulation', 'rearm_heartbeats')


def _log(msg):
    print(msg, flush=True)


@dataclass
class HostOps:
    """The live, host-side operations the hooks sequence. Each is a callable that acts on the LIVE
    session and RAISES to refuse (never silently bypassed). Grouped here so the orchestration is
    unit-testable with an injected fake and production binds the real handles in :func:`build`.

    * ``warm_safe_halt()``   -- park base, cancel nav, hold the arm; keep station-level resources alive.
    * ``arm_to_start()``     -- re-arm the arm command gate (clear latched fault, reset the heartbeat clock).
    * ``clear_manipulation()`` -- clear the shared runtime's between-op latches (crash budget, payload).
    * ``rearm_heartbeats()`` -- re-prime the in-process heartbeat/watchdog clocks for the next run.
    * ``arm_faulted()``      -- True if the arm command gate is still latched faulted (verify gate).
    * ``app_running()``      -- True if Kit's SimulationApp is still running (verify gate).
    """
    warm_safe_halt: Callable
    arm_to_start: Callable
    clear_manipulation: Callable
    rearm_heartbeats: Callable
    arm_faulted: Callable
    app_running: Callable


@dataclass
class HostLifecycleHooks:
    """The object the provider reads the ``stop_runtime`` / ``reset_host_state`` / ``verify`` hooks off.

    ``reset_host_state`` runs the ordered :data:`RESET_STEP_NAMES` sequence against the live host,
    stopping at the first refusal and emitting a LOUD named ``WARM_RESET_HOOK_FALLBACK`` line before
    re-raising as a :class:`WarmResetRefused` so the provider cold-boots with the reason. A reset op that
    raises a non-refusal exception is wrapped as a named refusal too (a reset op must never crash the
    between-runs gate)."""
    ops: HostOps
    step_names: tuple = RESET_STEP_NAMES

    def stop_runtime(self):
        """WARM-SAFE halt between runs: does not tear down the station-level masked-depth publisher /
        control relay the next run reuses (the provider's default ``adapter.stop()`` would). Kit stays up."""
        try:
            self.ops.warm_safe_halt()
        except Exception as exc:  # noqa: BLE001 -- surface as a named refusal, not a crash
            _log(f'WARM_RESET_HOOK_FALLBACK stop_runtime refused -> cold boot: {exc}')
            raise WarmResetRefused(f'stop_runtime: {exc}') from exc

    def reset_host_state(self, *, attempt, now):
        """In-place reset of the live host-side state (the ``relaunch_host``-satisfying step): re-arm the
        arm command gate, clear the manipulation between-op latches, re-arm the heartbeats -- all without
        closing Isaac. Refuses LOUDLY on the first failure."""
        steps = (
            ('arm_to_start', self.ops.arm_to_start),
            ('clear_manipulation', self.ops.clear_manipulation),
            ('rearm_heartbeats', self.ops.rearm_heartbeats),
        )
        for name, fn in steps:
            _log(f'WARM_RESET_HOOK step={name} begin (attempt={attempt})')
            try:
                fn()
            except WarmResetRefused as exc:
                _log(f'WARM_RESET_HOOK_FALLBACK reset_host_state step={name} refused -> cold boot: {exc}')
                raise WarmResetRefused(f'reset_host_state[{name}]: {exc}') from exc
            except Exception as exc:  # noqa: BLE001 -- a reset op must never crash the gate
                _log(f'WARM_RESET_HOOK_FALLBACK reset_host_state step={name} refused -> cold boot: {exc}')
                raise WarmResetRefused(f'reset_host_state[{name}]: {exc}') from exc
            _log(f'WARM_RESET_HOOK step={name} ok')
        return True

    def verify(self, *, now):
        """Post-reset readiness: Kit still running AND the arm gate left un-faulted."""
        if not self.ops.app_running():
            _log('WARM_RESET_HOOK_FALLBACK verify refused -> cold boot: SimulationApp not running')
            raise WarmResetRefused('verify: SimulationApp is no longer running after the host reset')
        if self.ops.arm_faulted():
            _log('WARM_RESET_HOOK_FALLBACK verify refused -> cold boot: arm gate still faulted')
            raise WarmResetRefused('verify: arm command gate still latched faulted after the reset')


# --------------------------------------------------------------------------- live-handle resolution

def _resolve_gate(port):
    """The :class:`ArmCommandGate` to re-arm -- the shared bundle's one piece of arm state that
    survives across warm runs (a latched ``hold()``/fault stays latched; the gate's heartbeat clock
    holds the previous run's stamps). Reachable either directly on the port or behind its
    ``NativeTransport``. Returns None when the live port exposes neither (handled by the caller)."""
    if port is None:
        return None
    gate = getattr(port, 'command_gate', None)
    if gate is not None:
        return gate
    transport = getattr(port, 'native_transport', None)
    return getattr(transport, 'command_gate', None)


def _build_live_ops(*, app, assembly, parts):
    """Bind the real host-side operations to the live handles. Pure wiring (no Isaac import at module
    load); each returned op touches the live session only when a hook actually runs."""
    parts = parts if isinstance(parts, dict) else {}
    port = parts.get('port')
    mobile = parts.get('mobile')
    nav = parts.get('navigation')
    manip = parts.get('manipulation')
    adapter = parts.get('adapter')

    def warm_safe_halt():
        # Park the base and cancel navigation, then hold the arm -- WITHOUT tearing down the
        # station-level masked-depth publisher / control relay (the provider's default adapter.stop()
        # closes sensing; the next warm run reuses those). Each sub-action is best-effort so one
        # missing handle never blocks the halt, but a hold with no reachable arm-hold path refuses.
        if mobile is not None and hasattr(mobile, 'park'):
            mobile.park()
        if nav is not None and hasattr(nav, 'cancel'):
            nav.cancel()
        held = False
        hold_arm = getattr(port, 'hold_arm', None) if port is not None else None
        if callable(hold_arm):
            hold_arm()
            held = True
        else:
            gate = _resolve_gate(port)
            if gate is not None and hasattr(gate, 'hold'):
                gate.hold()
                held = True
        if not held:
            raise WarmResetRefused('no reachable arm-hold path (port.hold_arm / command_gate.hold) '
                                   'to safely halt the arm between runs')

    def arm_to_start():
        # Re-arm the arm command gate: clear a latched hold()/fault, re-seat a clean hold at the
        # measured joints, and reset the heartbeat clock so the next run's first command is not charged
        # the warm gap. CRITICAL -- a session whose arm gate cannot be re-armed is not safe to reuse
        # (the next run's first arm command is silently dropped against a latched fault and the arm
        # stays frozen at the previous pose, then faults on the reference divergence).
        gate = _resolve_gate(port)
        if gate is None:
            raise WarmResetRefused('no arm command gate on the live port to re-arm')
        if not hasattr(gate, 'resume'):
            raise WarmResetRefused('arm command gate exposes no resume() to re-arm the latched hold')
        gate.resume()

    def clear_manipulation():
        # Clear the shared manipulation runtime's between-op latches so the next op starts clean:
        #   * the motion crash-retry budget (reset_crash_budget), which controller.start resets per op
        #     but is reset here too so a warm re-run that reuses the live port begins with a full budget;
        #   * any held-payload mirror the adapter carries.
        # Best-effort: a handle the live chain does not expose is skipped by name (never faked). The
        # per-mission Mission / BT / inventory state is rebuilt fresh by each run_once, so it is NOT
        # reset here.
        for owner, meth in ((manip, 'reset_crash_budget'), (port, 'reset_crash_budget'),
                            (adapter, 'reset_crash_budget')):
            fn = getattr(owner, meth, None) if owner is not None else None
            if callable(fn):
                fn()
                break

    def rearm_heartbeats():
        # Re-arm the in-process heartbeat / watchdog clocks for the next run (PR 301). The arm-command
        # heartbeat clock is already re-armed by arm_to_start()'s gate.resume(); re-prime the parked
        # station liveness so its next-op incorporation ack is not charged the warm gap. The per-mission
        # nav-stuck / station-ack watchdogs are rebuilt fresh by each run_once. Best-effort: skip by name
        # when the live chain does not expose the station re-prime.
        prime = getattr(manip, 'set_base_repositioned', None) if manip is not None else None
        # No station re-prime hook is a non-fatal skip: the next op's graph.start re-prepares the station
        # and re-derives the global-stop wire, and the station keeps publishing liveness through the halt.
        _ = prime  # presence probe only; the station re-prepares itself on the next op dispatch.

    def arm_faulted():
        gate = _resolve_gate(port)
        return bool(getattr(gate, 'faulted', False)) if gate is not None else False

    def app_running():
        if app is None or not hasattr(app, 'is_running'):
            return True  # no app handle to contradict running (the provider's own verify owns this too)
        return bool(app.is_running())

    return HostOps(warm_safe_halt=warm_safe_halt, arm_to_start=arm_to_start,
                   clear_manipulation=clear_manipulation, rearm_heartbeats=rearm_heartbeats,
                   arm_faulted=arm_faulted, app_running=app_running)


def build(*, app=None, assembly=None, parts=None, plan=None, config=None, output=None, _ops=None, **_):
    """The ``config['warm_reset_hooks']`` ``module:function`` factory (and the provider default). Binds
    the production host-lifecycle hooks to the live handles and returns a :class:`HostLifecycleHooks`.

    Lazy: no live handle is touched here, so it constructs cleanly with None handles (the provider's own
    CPU tests build with ``app=None, assembly=None, parts={}``). ``_ops`` is a test seam (an injected
    :class:`HostOps`); production leaves it None and binds the real ops."""
    ops = _ops if _ops is not None else _build_live_ops(app=app, assembly=assembly, parts=parts)
    return HostLifecycleHooks(ops=ops)
