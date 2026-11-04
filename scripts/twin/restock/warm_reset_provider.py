"""Production warm-session reset provider: the ``<module>:build`` the launcher drives.

PR 278 shipped the warm-session SAFETY CORE (``twin.restock.warm_session``) -- the conservative
reuse-vs-cold-boot decision and the fail-closed reset sequence -- but NO production provider, so
``config['warm_reset']`` had nothing to name and every ``--warm-attempts`` / ``--consecutive-runs``
run paid the full ~215 s cold boot again. This module is that provider.

``build(...)`` is the lane-owned builder ``isaac_bundle.create`` resolves from
``config['warm_reset'] = "twin.restock.warm_reset_provider:build"``. It is called ONCE, with the live
scene handles, right after the mission bundle is assembled, and returns a :class:`WarmResetProvider`.
The provider is the object the launcher's consecutive-run loop drives between missions:

* it is CALLABLE as ``reset(attempt=, now=)`` -- the per-attempt contract ``bundle.warm_reset`` already
  exposes -- and running it executes the ordered, fail-closed in-session reset sequence;
* it exposes :meth:`WarmResetProvider.stage_census` -- wired onto ``bundle.warm_stage_state`` -- which
  the loop reads to feed the ``stage_clean`` gate of ``warm_session.decide_reset``.

The reset SEQUENCE, in order (each step RAISES :class:`WarmResetRefused` to refuse; a refusal is a
cold-boot fallback with the step named, never a silent half-reset -- the captain's rule):

  1. ``stop_runtime``       -- stop the mission runtime + host stack cleanly, WITHOUT exiting Kit
                               (never ``app.close()``); the live ``SimulationApp`` + loaded USD stage
                               (the ~205 s of the boot) are preserved.
  2. ``reset_stage``        -- prune the mission-spawned prims (anything present now that was not in the
                               post-boot baseline census: the carton, welds, markers, a respawned cart),
                               then ``world.reset()`` to re-seat every registered body to its post-boot
                               default pose AND reset the sim clock to t=0.
  3. ``reseat``             -- re-establish the next run's start precondition + verify it: for a
                               ``start_at`` run, re-run ``shortcut.apply_start_state`` (which re-seats the
                               robot/cart/carton and runs PR 297's sim-time settle + rest/heading verify,
                               raising on a mis-seated train); for a cold dock-aligned run,
                               ``shortcut.verify_dock_seat``; otherwise a no-op (``world.reset`` re-seated).
  4. ``relaunch_host``      -- re-launch / re-sync the host stack (planner / mapper) so it serves the
                               fresh stage and ``bundle.adapter`` is usable for the next ``Mission``.
                               This is integration-lane-owned (the host processes are not built in this
                               repo), so it is delegated to a lane HOOK; absent => REFUSE (cold boot).
  5. ``verify``             -- the live app is still running and an optional lane readiness hook passes.
                               (The launcher runs the runtime-input ``preflight_verify`` as its own next
                               step, so the full "existing preflight gates + PR 297 verify" the brief
                               asks for is covered across the provider + the loop.)

Every live operation is isolated behind a small, individually-overridable callable on :class:`ResetOps`
so the ORCHESTRATION (ordering, fail-closed stop-at-first-refusal, the census diff, every fallback
reason) is CPU-provable with injected fakes -- the GPU-side re-seat itself is validated by the
integration lane at acceptance. ``build(..., _ops=<ResetOps>)`` is the test seam (production leaves it
None and binds the real Isaac ops).

Lane hooks. The two integration-owned steps (``stop_runtime`` host teardown, ``relaunch_host``) are
resolved from an optional ``config['warm_reset_hooks'] = "module:function"`` factory, called
``fn(app=, assembly=, parts=, plan=, config=, output=)`` and returning an object/mapping with any of
``stop_runtime`` / ``relaunch_host`` / ``reset_host_state`` / ``verify`` (each a zero-/kw-arg callable
that raises to refuse). ``relaunch_host`` is satisfied by EITHER a ``relaunch_host`` or a
``reset_host_state`` hook. When ``config['warm_reset_hooks']`` is ABSENT the in-repo production factory
:data:`DEFAULT_HOOKS` (``twin.restock.warm_reset_hooks:build``) is used, which supplies a
``reset_host_state`` (an IN-PLACE reset of the live host-side state -- re-arm the arm command gate, clear
the manipulation between-op latches, re-arm the heartbeats -- without an OS-process relaunch or an Isaac
close) plus a warm-safe ``stop_runtime`` and a ``verify``, so a warm reset works with no hand-written
stub. A custom factory that supplies NEITHER a ``relaunch_host`` nor a ``reset_host_state`` hook still
refuses that step and the loop cold-boots the next run with the reason named -- strictly safer than
reusing a host still carrying the previous run's state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from . import warm_session as ws


class WarmResetRefused(RuntimeError):
    """A reset step refused the warm reuse (=> cold-boot fallback). Its message names the step."""


# Isaac-managed RTX render-target prims. A sensing camera's render product lives under this namespace
# (``/Render/OmniverseKit/HydraTextures/Replicator_<NN>/...``); Isaac allocates it when the camera starts
# rendering and the fm380/fm381 teardown destroys it to free VRAM, picking a fresh ``Replicator_<NN>``
# index when it is RE-created. So these prims are (a) volatile across a run's own teardown and (b)
# re-created ON DEMAND the instant the owning camera's render product is re-initialized -- which the warm
# reset does (``warm_reset_hooks.reset_host_state`` -> ``CameraRig.recreate_destroyed``). Matching their
# exact paths against the post-boot baseline is therefore both brittle (the index churns) and redundant
# (the STABLE ``/World/restock_camera_*/Camera`` prim they hang off IS matched and restored). Per the
# captain's rule a render prim may be excluded from the census ONLY when it is re-created on demand by the
# next attempt and that is proven -- which is exactly this subtree; the restock-camera prims are NOT
# excluded (a torn-down camera is a real ``missing`` the reset must restore). The exclusion is scoped
# tightly to this RTX render-target namespace so a leaked WORLD prim still fails the census.
_VOLATILE_RENDER_PREFIX = '/Render/OmniverseKit/HydraTextures/'


def _census_prims(paths):
    """The set of prim paths the warm stage census compares, with the volatile Isaac RTX render-target
    subtree (:data:`_VOLATILE_RENDER_PREFIX`) excluded (re-created on demand; see the constant). Applied
    to BOTH the post-boot baseline and the live present set so the exclusion cannot itself create a
    spurious leak/missing."""
    return frozenset(str(p) for p in paths if not str(p).startswith(_VOLATILE_RENDER_PREFIX))


# The ordered reset steps. Stable names so the record / dry-run report and the tests key off them.
STEP_NAMES = ('stop_runtime', 'reset_stage', 'reseat', 'relaunch_host', 'verify')

# The in-repo production host-lifecycle hooks factory, used when ``config['warm_reset_hooks']`` is absent.
# PR 299 shipped the provider with NO in-repo factory, so the integration worktree -- which cannot
# hand-write one -- cold-booted every warm attempt. ``twin.restock.warm_reset_hooks:build`` is that
# factory (an IN-PLACE reset of the live host-side state satisfying the ``relaunch_host`` step without an
# OS-process relaunch or an Isaac close); naming the key points at a different lane-owned factory instead.
DEFAULT_HOOKS = 'twin.restock.warm_reset_hooks:build'


@dataclass
class ResetOps:
    """The live, Isaac-side operations the provider sequences. Each is a callable that performs one
    step against the LIVE session and RAISES to refuse (never silently bypassed). Grouped here so the
    orchestration can be unit-tested with an injected fake and production binds the real ops in
    :func:`build`.

    * ``list_prims()`` -> iterable of stage prim-path strings (the census source).
    * ``stop_runtime()`` -> stop the mission runtime + host stack without exiting Kit.
    * ``reset_stage(leaked)`` -> prune the ``leaked`` prim paths, ``world.reset()``, reset the sim clock.
    * ``reseat(attempt, now)`` -> re-establish + verify the next run's start precondition.
    * ``relaunch_host(attempt, now)`` -> re-launch / re-sync the host stack + leave the adapter ready.
    * ``verify(now)`` -> final readiness gate (app running + optional lane hook).
    """
    list_prims: Callable
    stop_runtime: Callable
    reset_stage: Callable
    reseat: Callable
    relaunch_host: Callable
    verify: Callable


@dataclass
class WarmResetProvider:
    """The object ``bundle.warm_reset`` points at: callable as ``reset(attempt=, now=)`` to run the
    in-session reset sequence, plus :meth:`stage_census` for ``bundle.warm_stage_state``."""
    ops: ResetOps
    baseline_prims: frozenset
    step_names: tuple = STEP_NAMES

    def stage_census(self) -> ws.StageState:
        """A post-teardown prim census vs the post-boot baseline, as a ``warm_session.StageState`` the
        ``stage_clean`` gate consumes. A census that itself fails is reported as a DIRTY stage (a doubt
        => cold boot), never swallowed."""
        try:
            present = _census_prims(self.ops.list_prims())
        except Exception as exc:  # noqa: BLE001 -- a failed census is a doubt, not a crash
            return ws.StageState(dirty=True, note=f'prim census failed: {exc}')
        return ws.StageState(present_prims=present, expected_prims=self.baseline_prims)

    def _leaked(self) -> tuple:
        """The prim paths present now but not in the post-boot baseline -- what ``reset_stage`` prunes."""
        return self.stage_census().leaked()

    def __call__(self, *, attempt, now):
        """Run the ordered reset sequence against the live session. Returns True on a complete reset;
        RAISES :class:`WarmResetRefused` (step named) on the first refusal so the launcher's reset
        sequence records a cold-boot fallback with the reason. Any non-refusal exception from a live op
        is wrapped as a named refusal too (a reset op must never crash the between-runs gate)."""
        steps = (
            ('stop_runtime', lambda: self.ops.stop_runtime()),
            ('reset_stage', lambda: self.ops.reset_stage(self._leaked())),
            ('reseat', lambda: self.ops.reseat(attempt=attempt, now=now)),
            ('relaunch_host', lambda: self.ops.relaunch_host(attempt=attempt, now=now)),
            ('verify', lambda: self.ops.verify(now=now)),
        )
        for name, fn in steps:
            try:
                fn()
            except WarmResetRefused as exc:
                # Already named by the op; re-raise as-is (its message leads with the cause).
                raise WarmResetRefused(f'{name}: {exc}') from exc
            except Exception as exc:  # noqa: BLE001 -- surface as a named refusal, not a crash
                raise WarmResetRefused(f'{name}: {exc}') from exc
        return True


# --------------------------------------------------------------------------- lane-hook resolution

def _resolve_hooks(config, *, app, assembly, parts, plan, output):
    """Resolve the host-lifecycle hooks named by ``config['warm_reset_hooks']`` (a ``module:function``
    factory). Returns a mapping of the hooks it supplied (any of ``stop_runtime`` / ``relaunch_host`` /
    ``reset_host_state`` / ``verify``). When the key is absent the in-repo production default
    :data:`DEFAULT_HOOKS` is used, so a warm reset works with no hand-written stub; name the key only to
    point at a different lane-owned factory. A malformed key is refused (never silently ignored)."""
    ref = config.get('warm_reset_hooks') if isinstance(config, dict) else None
    if ref is None:
        ref = DEFAULT_HOOKS
    if not isinstance(ref, str) or ':' not in ref:
        raise ValueError('config.warm_reset_hooks must name a lane-owned builder as "module:function"')
    import importlib
    module, name = ref.split(':', 1)
    factory = getattr(importlib.import_module(module), name)
    hooks = factory(app=app, assembly=assembly, parts=parts, plan=plan, config=config, output=output)
    if hooks is None:
        return {}
    if isinstance(hooks, dict):
        return {k: v for k, v in hooks.items() if callable(v)}
    return {k: getattr(hooks, k) for k in ('stop_runtime', 'relaunch_host', 'reset_host_state', 'verify')
            if callable(getattr(hooks, k, None))}


# --------------------------------------------------------------------------- production live ops

def _build_production_ops(*, app, assembly, parts, plan, config, output, scenario, hooks):
    """Bind the real Isaac-side operations to the live handles. Pure wiring (no Isaac import at module
    load); each returned op touches the live stage only when the provider actually runs."""
    adapter = parts.get('adapter') if isinstance(parts, dict) else None

    def list_prims():
        # Traverse the live USD stage once and return every prim path. Between-runs only (cheap enough);
        # the diff vs the baseline is what names a leak. Falls back to the stage's direct children if a
        # full traverse is unavailable on this handle.
        stage = getattr(assembly, 'stage', None)
        if stage is None:
            raise WarmResetRefused('no live stage handle to census')
        if hasattr(stage, 'Traverse'):
            return [p.GetPath().pathString for p in stage.Traverse()]
        if hasattr(stage, 'GetPseudoRoot'):
            return [c.GetPath().pathString for c in stage.GetPseudoRoot().GetChildren()]
        raise WarmResetRefused('live stage handle exposes no Traverse/GetPseudoRoot')

    def stop_runtime():
        # Stop the mission runtime + host stack cleanly, WITHOUT exiting Kit. Never app.close(): the live
        # SimulationApp + loaded stage (the expensive ~205 s) must survive. The adapter stop holds the arm
        # and does not release a held payload (RuntimeAdapter.stop); the host teardown is a lane hook.
        if callable(hooks.get('stop_runtime')):
            hooks['stop_runtime']()
        elif adapter is not None and hasattr(adapter, 'stop'):
            adapter.stop()
        if app is not None and hasattr(app, 'is_running') and not app.is_running():
            raise WarmResetRefused('SimulationApp stopped during runtime teardown (Kit must stay up)')

    def reset_stage(leaked):
        # Prune the mission-spawned prims, then re-seat every registered body + reset the sim clock via
        # world.reset(). RemovePrim mirrors native_cameras' teardown; world.reset() is Isaac's own
        # default-state re-seat (robot joints, cart, the start-state carton all return to their add-time
        # pose) and resets current_time to 0.
        stage = getattr(assembly, 'stage', None)
        for path in leaked:
            if stage is not None and hasattr(stage, 'RemovePrim'):
                stage.RemovePrim(path)
            else:
                raise WarmResetRefused(f'cannot prune leaked prim {path}: no RemovePrim on the stage')
        world = getattr(assembly, 'world', None)
        if world is None or not hasattr(world, 'reset'):
            raise WarmResetRefused('no live World handle to reset physics / sim clock')
        world.reset()
        cur = getattr(world, 'current_time', None)
        if cur is not None and float(cur) > 1e-6:
            raise WarmResetRefused(f'world.reset() did not reset the sim clock (current_time={cur})')

    def reseat(*, attempt, now):
        # Re-establish + verify the next run's start precondition. apply_start_state / verify_dock_seat
        # run PR 297's sim-time settle + rest/heading verify and RAISE on a mis-seated train, which is
        # exactly the refuse-to-release gate we want. world.reset() above returned the start-state carton
        # to its add-time pose, so clear the stale scene registration before apply_start_state re-adds it.
        if config.get('start_at'):
            world = getattr(assembly, 'world', None)
            scene = getattr(world, 'scene', None)
            if scene is not None and hasattr(scene, 'remove_object'):
                try:
                    scene.remove_object('shortcut_carton')
                except Exception:
                    pass  # absent on the first re-seat of a run that never spawned it; not an error
            from .shortcut import apply_start_state
            apply_start_state(assembly, config, scenario)
        elif config.get('couple_dock_align'):
            from .shortcut import verify_dock_seat
            verify_dock_seat(assembly, config, scenario)
        # A plain cold run has no synthesized start precondition; world.reset() already re-seated it.

    def relaunch_host(*, attempt, now):
        # Re-launch / re-sync the host stack (planner / mapper) so it serves the fresh stage and leaves
        # bundle.adapter ready for the next Mission. The host processes are integration-lane-owned (not
        # built in this repo), so this is a lane hook: either an explicit relaunch_host or an in-place
        # reset_host_state satisfies it. With neither wired we REFUSE (cold boot) rather than reuse a
        # host still carrying the previous run's planning scene / nvblox ESDF / station acks.
        if callable(hooks.get('relaunch_host')):
            hooks['relaunch_host'](attempt=attempt, now=now)
        elif callable(hooks.get('reset_host_state')):
            hooks['reset_host_state'](attempt=attempt, now=now)
        else:
            raise WarmResetRefused(
                'no host relaunch/reset hook wired (config.warm_reset_hooks must supply relaunch_host '
                'or reset_host_state); the live host stack still carries the previous run state')

    def verify(*, now):
        if app is not None and hasattr(app, 'is_running') and not app.is_running():
            raise WarmResetRefused('SimulationApp is no longer running after the reset')
        if callable(hooks.get('verify')):
            hooks['verify'](now=now)

    return ResetOps(list_prims=list_prims, stop_runtime=stop_runtime, reset_stage=reset_stage,
                    reseat=reseat, relaunch_host=relaunch_host, verify=verify)


def build(*, app, assembly, parts, plan, config, output, scenario=None, _ops=None, **_):
    """Lane-owned builder (the ``config['warm_reset']`` ``module:function`` contract). Returns the
    :class:`WarmResetProvider` the launcher drives between consecutive missions, or raises to refuse.

    Captures the post-boot prim baseline NOW (after the bundle is assembled + any ``start_at`` start
    state is applied), so the between-runs census names exactly the prims a mission added. ``_ops`` is a
    test seam (an injected :class:`ResetOps`); production leaves it None and binds the real Isaac ops.
    """
    hooks = _resolve_hooks(config, app=app, assembly=assembly, parts=parts, plan=plan, output=output)
    ops = _ops if _ops is not None else _build_production_ops(
        app=app, assembly=assembly, parts=parts, plan=plan, config=config, output=output,
        scenario=scenario, hooks=hooks)
    try:
        baseline = _census_prims(ops.list_prims())
    except Exception as exc:  # noqa: BLE001
        raise WarmResetRefused(f'cannot snapshot the post-boot prim baseline: {exc}') from exc
    return WarmResetProvider(ops=ops, baseline_prims=baseline)
