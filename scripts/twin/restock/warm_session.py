"""Warm-session reset decision, sequence and cold-boot fallback (CPU; no Isaac/ROS/GPU).

A cold run of the Isaac restock mission pays a FIXED ~215 s boot (Kit process start + stage
load + host/planner/mapper launch + first ESDF/frame readiness) BEFORE a single mission frame
is rendered -- measured at 211-222 s across today's runs (coldwin22 216 s, coldwin23 211 s,
collect2 222 s; see ``data/sim-warm-session-reset-v1/boot-vs-mission.txt``). That is ~8.5-10.5 %
of a full ~42 min cycle but ~42 % of a short fast-failing debug run, and it recurs on EVERY run,
so eliminating it compounds across the many proof runs a debugging program needs.

A *warm session* keeps the live ``SimulationApp`` + host stack up between missions and re-seats
the stage to its commissioned initial state (via the lane-owned ``bundle.warm_reset`` provider),
skipping the Kit boot + stage load entirely. The next mission then starts in the time of a stage
re-seat + re-commission + preflight, not a full boot.

This module is the SAFETY CORE around that: it is pure control flow + decision logic (no renderer,
no physics, no ROS), so it is fully CPU-provable. It answers one question conservatively --

    *Is it safe to reuse this live session for the next run, or must we fall back to a cold boot?*

and NAMES the reason either way (the brief's "any doubt ... falls back to a cold boot and says
why"). It never performs the GPU-side re-seat itself; the live ``World.reset`` / prim re-seat /
camera re-commission live where the stage and cameras do (the integration + camera/stations lanes,
behind ``bundle.warm_reset``). This core gates and sequences them.

The decision is deliberately CONSERVATIVE: a warm reset is reused ONLY when every check passes.
Any doubt -- a HOST-level config change the live host stack cannot re-apply, a dirty stage / leaked
prims, VRAM that has not returned to the post-boot baseline, or a caller-supplied doubt -- yields a
cold-boot fallback with the reason, never a silent reuse of a possibly-corrupt session.

Per the captain's SIM-clock rule: this is a BETWEEN-missions gate, not a mission guard, so it adds
no wall-timed watchdog to the running simulation. Wall duration of the reset is MEASURED and
reported (the hand-off's "second run starts within a stated time" acceptance), never imposed as a
deadline that could false-trip under render pacing. The commissioned-baseline lifetime check stays
on the sim clock inside ``bundle.warm_reset`` (unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# HOST/RUNTIME classification of every mission batch key -- the single source of truth the
# activation preflight (run-277) already maintains. A warm session booted the host stack with the
# ORIGINAL host config; the live ROS host cannot re-apply a HOST-key change without a relaunch, so
# any HOST-key delta forces a cold boot.
from twin.restock_manipulation.config_preflight import HOST, KEY_CONSUMER

# Boot-fixed keys outside KEY_CONSUMER whose value is baked into the Kit boot / host launch and so
# cannot change on a warm reuse: the scenario identity, host-config path, package roots and the
# render resolution / low-VRAM render passes (SimulationApp boot args). A delta on any of these is a
# cold boot, exactly like a HOST key.
BOOT_FIXED_KEYS = (
    'scenario',
    'host_config',
    'package_roots',
    'commissioning_dir',   # the station baselines are loaded at boot; a different dir is a cold boot
    'render_width',
    'render_height',
    'low_vram_render',
)

# Default tolerance (MiB) for "VRAM returned to the post-boot baseline". The 12 GiB card completes
# the full capstone within ~500 MiB of its top, so a run that leaked more than a few hundred MiB
# after teardown is not safe to reuse. Conservative and overridable.
DEFAULT_VRAM_TOL_MIB = 400.0


@dataclass(frozen=True)
class StageState:
    """A cheap post-teardown snapshot of the live stage the integration lane produces (prim census).

    ``expected_prims`` is the post-boot baseline set (names or paths) the clean initial stage carries;
    ``present_prims`` is what the live stage carries now. A warm reuse requires the two to match
    exactly: any prim present-but-unexpected is a leak, any expected-but-missing is a torn-down stage.
    ``dirty`` is an explicit escape hatch the lane can set when it knows the stage is unusable for a
    reason a census cannot see (e.g. a physics scene left in a faulted state).
    """
    present_prims: frozenset = frozenset()
    expected_prims: frozenset = frozenset()
    dirty: bool = False
    note: str = ''

    def leaked(self) -> tuple:
        return tuple(sorted(set(self.present_prims) - set(self.expected_prims)))

    def missing(self) -> tuple:
        return tuple(sorted(set(self.expected_prims) - set(self.present_prims)))


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ''


@dataclass(frozen=True)
class ResetDecision:
    """The verdict: ``reuse`` the live session for the next run, or fall back to a cold boot.

    ``reason`` is ALWAYS populated (the "says why" requirement) -- the single human-readable line
    the launcher records on the summary and prints. ``checks`` is the full per-check breakdown for
    the record / dry-run report. ``cold_boot`` is the inverse of ``reuse`` for readability at call
    sites (a cold-boot fallback is a decision, not an error)."""
    reuse: bool
    reason: str
    checks: tuple = ()

    @property
    def cold_boot(self) -> bool:
        return not self.reuse


def host_level_changes(prev_config: dict, next_config: dict) -> list:
    """Return the sorted keys whose value differs between runs AND whose consumer is the HOST
    config or a boot-fixed boot arg -- i.e. changes a warm session cannot re-apply without a
    relaunch. An unregistered key that changed is reported too (unknown placement => treat as a
    cold-boot trigger, never a silent reuse)."""
    prev = prev_config or {}
    nxt = next_config or {}
    changed = []
    host_like = {k for k, side in KEY_CONSUMER.items() if side == HOST} | set(BOOT_FIXED_KEYS)
    for key in set(prev) | set(nxt):
        if prev.get(key) == nxt.get(key):
            continue
        if key in host_like:
            changed.append(key)
        elif key not in KEY_CONSUMER and key not in _RUNTIME_SAFE_IGNORE:
            # A changed key we do not recognise: we cannot prove it is RUNTIME-only, so be
            # conservative and force a cold boot. Register known RUNTIME keys in KEY_CONSUMER.
            changed.append(key)
    return sorted(changed)


# Launcher/bookkeeping keys that legitimately differ between two consecutive runs and are NOT a
# simulator input (so a delta on them must not force a cold boot). The output dir and the launch
# environment snapshot are injected per run by the launcher itself.
_RUNTIME_SAFE_IGNORE = frozenset({'_launch_environment', 'output', 'stages', 'start_at',
                                  'start_staging', 'stop_after'})


def decide_reset(
    prev_config: dict,
    next_config: dict,
    *,
    stage_state: Optional[StageState] = None,
    vram_used_mib: Optional[float] = None,
    post_boot_baseline_mib: Optional[float] = None,
    vram_tol_mib: float = DEFAULT_VRAM_TOL_MIB,
    extra_doubts: Sequence[str] = (),
) -> ResetDecision:
    """Decide whether the live session may be reused for the next run (conservative; any doubt =>
    cold boot). Returns a :class:`ResetDecision` whose ``reason`` always names the outcome.

    Checks, in order (all must pass to reuse):
      1. **config_host_stable** -- no HOST-level / boot-fixed key changed between the booted config
         and the next run's config (``host_level_changes``). The live host stack + Kit cannot
         re-apply such a change, so a delta is a cold boot.
      2. **stage_clean** -- the post-teardown stage census matches the post-boot baseline exactly
         (no leaked prims, no missing prims) and is not explicitly flagged dirty.
      3. **vram_returned** -- live used VRAM is within ``vram_tol_mib`` of the post-boot baseline
         (GPU memory returned after the previous run's teardown). Skipped only when no reading is
         supplied (the caller then owns that risk; the dry-run always supplies one).
      4. **no_extra_doubt** -- the caller passed no explicit doubt string.
    """
    checks: list = []

    changed = host_level_changes(prev_config, next_config)
    checks.append(Check('config_host_stable', not changed,
                        '' if not changed else 'HOST/boot-fixed keys changed: ' + ', '.join(changed)))

    if stage_state is None:
        checks.append(Check('stage_clean', False, 'no stage census supplied (cannot prove the stage is clean)'))
    else:
        leaked, missing = stage_state.leaked(), stage_state.missing()
        problems = []
        if stage_state.dirty:
            problems.append('stage flagged dirty' + (f' ({stage_state.note})' if stage_state.note else ''))
        if leaked:
            problems.append('leaked prims: ' + ', '.join(leaked))
        if missing:
            problems.append('missing prims: ' + ', '.join(missing))
        checks.append(Check('stage_clean', not problems, '; '.join(problems)))

    if vram_used_mib is None or post_boot_baseline_mib is None:
        checks.append(Check('vram_returned', True, 'no VRAM reading supplied (check skipped)'))
    else:
        delta = float(vram_used_mib) - float(post_boot_baseline_mib)
        ok = delta <= float(vram_tol_mib)
        checks.append(Check('vram_returned', ok,
                            f'used {vram_used_mib:.0f} MiB vs post-boot baseline '
                            f'{post_boot_baseline_mib:.0f} MiB (+{delta:.0f}, tol {vram_tol_mib:.0f})'))

    doubts = [d for d in extra_doubts if d]
    checks.append(Check('no_extra_doubt', not doubts,
                        '' if not doubts else '; '.join(doubts)))

    failed = [c for c in checks if not c.ok]
    if failed:
        reason = 'cold boot: ' + '; '.join(f'{c.name}: {c.detail}' for c in failed)
        return ResetDecision(reuse=False, reason=reason, checks=tuple(checks))
    return ResetDecision(reuse=True, reason='warm reuse: all reset checks passed',
                         checks=tuple(checks))


@dataclass
class SequenceStep:
    """One ordered step of the warm-reset sequence. ``fn`` is a no-arg callable that performs the
    step against the LIVE session; it returns None/truthy on success and RAISES to refuse (never
    bypassed). The step names are stable so the record / dry-run report can key off them."""
    name: str
    fn: Callable


@dataclass
class SequenceResult:
    ok: bool
    reason: str
    completed: tuple = ()           # step names that ran successfully, in order
    failed_step: Optional[str] = None
    wall_seconds: Optional[float] = None
    logs: tuple = ()


def run_reset_sequence(steps: Sequence[SequenceStep], *, clock: Optional[Callable] = None,
                       on_log: Optional[Callable] = None) -> SequenceResult:
    """Run the warm-reset sequence in order; stop at the first failure (fail-closed).

    Each step raises to refuse (e.g. the baseline lifetime guard, a preflight failure, a VRAM
    check). A refusal is NOT an error to swallow -- it is a cold-boot fallback with the step named,
    exactly like ``decide_reset`` returning ``cold_boot``. ``clock`` (default ``time.monotonic``)
    measures the wall duration purely to REPORT how long the reset took (the hand-off acceptance);
    it imposes no deadline. ``on_log`` receives one line per step for the console / report.
    """
    if clock is None:
        import time
        clock = time.monotonic
    log = on_log if on_log is not None else (lambda _m: None)
    lines = []

    def _emit(msg):
        lines.append(msg)
        log(msg)

    start = clock()
    completed = []
    for step in steps:
        _emit(f'WARM_RESET step={step.name} begin')
        try:
            step.fn()
        except Exception as exc:  # noqa: BLE001 -- a refusal is a cold-boot fallback, not a crash
            dur = clock() - start
            _emit(f'WARM_RESET step={step.name} REFUSED: {exc}')
            return SequenceResult(
                ok=False,
                reason=f'cold boot: reset step {step.name!r} refused: {exc}',
                completed=tuple(completed), failed_step=step.name,
                wall_seconds=round(dur, 3), logs=tuple(lines))
        completed.append(step.name)
        _emit(f'WARM_RESET step={step.name} ok')
    dur = clock() - start
    _emit(f'WARM_RESET sequence complete in {dur:.2f}s ({len(completed)} steps)')
    return SequenceResult(ok=True, reason='warm reset sequence complete',
                          completed=tuple(completed), failed_step=None,
                          wall_seconds=round(dur, 3), logs=tuple(lines))


@dataclass
class WarmOutcome:
    """The combined verdict of the decision gate + (when reused) the reset sequence."""
    reused: bool                    # True => the next run proceeds on the live session
    reason: str
    decision: ResetDecision
    sequence: Optional[SequenceResult] = None

    @property
    def cold_boot(self) -> bool:
        return not self.reused

    def record(self) -> dict:
        """A JSON-serialisable record for mission.json / warm-session.json."""
        rec = {
            'reused': self.reused,
            'reason': self.reason,
            'decision': {
                'reuse': self.decision.reuse,
                'reason': self.decision.reason,
                'checks': [{'name': c.name, 'ok': c.ok, 'detail': c.detail}
                           for c in self.decision.checks],
            },
        }
        if self.sequence is not None:
            rec['sequence'] = {
                'ok': self.sequence.ok,
                'reason': self.sequence.reason,
                'completed': list(self.sequence.completed),
                'failed_step': self.sequence.failed_step,
                'wall_seconds': self.sequence.wall_seconds,
            }
        return rec


def plan_and_run_reset(
    prev_config: dict,
    next_config: dict,
    steps: Sequence[SequenceStep],
    *,
    stage_state: Optional[StageState] = None,
    vram_used_mib: Optional[float] = None,
    post_boot_baseline_mib: Optional[float] = None,
    vram_tol_mib: float = DEFAULT_VRAM_TOL_MIB,
    extra_doubts: Sequence[str] = (),
    clock: Optional[Callable] = None,
    on_log: Optional[Callable] = None,
) -> WarmOutcome:
    """Gate then (only if the gate passes) run the warm-reset sequence. Returns a
    :class:`WarmOutcome` whose ``reused`` is True ONLY when the decision allowed reuse AND every
    reset step completed. On any cold-boot decision the sequence is NOT run (no stage is touched);
    on any step refusal ``reused`` is False with the step named. Never raises for a refusal."""
    decision = decide_reset(
        prev_config, next_config, stage_state=stage_state,
        vram_used_mib=vram_used_mib, post_boot_baseline_mib=post_boot_baseline_mib,
        vram_tol_mib=vram_tol_mib, extra_doubts=extra_doubts)
    if not decision.reuse:
        if on_log is not None:
            on_log('WARM_RESET decision=cold_boot ' + decision.reason)
        return WarmOutcome(reused=False, reason=decision.reason, decision=decision)
    seq = run_reset_sequence(steps, clock=clock, on_log=on_log)
    return WarmOutcome(reused=seq.ok, reason=(decision.reason if seq.ok else seq.reason),
                       decision=decision, sequence=seq)
