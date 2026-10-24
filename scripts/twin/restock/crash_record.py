"""Crash-safe mission recording (sim-robust-vram-headroom-v1).

The Isaac restock mission runs the simulator IN-PROCESS inside ``run_mobile_restock.py``. A native
fault in the embedded kit (coldwin13: ``carb.crashreporter`` SIGSEGV, exit 139, core dumped, during
the transfer approach) kills the whole python process BEFORE the ``finally`` that writes
``mission.json`` can run -- so a native crash leaves a run dir with ``host-stack.log`` /
``nav-trace.jsonl`` / ``bt_events.jsonl`` / a re-encoded ``walkthrough.mp4`` but **no verdict**, and
the record-first teardown + the evidence tooling (the ledger, ``check_walkthrough_run.py``) have
nothing to grade.

This module makes a crashed run still leave a ``mission.json`` with ``state="crashed"``, using two
pieces that are pure + CPU-tested here and wired by the bundle / shell elsewhere:

1. :class:`ProgressWriter` -- an INCREMENTAL, durable stage log the mission appends to on every stage
   transition (begin / succeeded / failed). Append-only + ``flush`` + ``fsync`` means the already
   written lines survive a native crash mid-tick. ~7 lines per run, never raises, and it only adds a
   NEW sidecar (``mission-progress.jsonl``) -- ``mission.json`` on the success path is byte-identical.

2. :func:`write_if_missing` -- the OUT-OF-PROCESS finalizer (run by the shell parent, which survives
   the python SIGSEGV, after the mission process exits). When the process died by a signal / wall-cap
   timeout AND ``mission.json`` is absent, it reconstructs the verdict from whatever durable artifacts
   exist -- preferring ``mission-progress.jsonl``, falling back to ``nav-trace.jsonl`` (last
   ``action``/``t``) + ``bt_events.jsonl`` (the stage ladder + last reason) so it ALSO recovers legacy
   runs (coldwin13) that predate the progress writer -- and writes ``state="crashed"`` with the crash
   signal, last stage, sim time, the stage ladder (done / current / pending) and the last sampled VRAM
   headroom (``vram-headroom.json``). If ``mission.json`` already exists it is a no-op.

Pure: stdlib only (no Isaac / ROS / numpy import), so the whole recovery is unit-tested off a run dir.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# The mission stage ladder, in order. Mirrors twin.restock.mission.ACTIONS; duplicated here as a plain
# tuple so the finalizer imports nothing from the Isaac-coupled mission module (it runs CPU-only, after
# the mission process is already dead).
ACTIONS = ('collect', 'couple', 'outbound', 'park', 'transfer', 'verify', 'return')

PROGRESS_LOG = 'mission-progress.jsonl'
NAV_TRACE = 'nav-trace.jsonl'
BT_EVENTS = 'bt_events.jsonl'
VRAM_FILE = 'vram-headroom.json'
MISSION_JSON = 'mission.json'


def signal_name(signal):
    """Human name for a crash signal number (e.g. 11 -> 'SIGSEGV'); passthrough for unknown/None."""
    names = {4: 'SIGILL', 6: 'SIGABRT', 7: 'SIGBUS', 8: 'SIGFPE', 9: 'SIGKILL', 11: 'SIGSEGV',
             15: 'SIGTERM'}
    if signal is None:
        return None
    return names.get(int(signal), f'SIG{int(signal)}')


# --------------------------------------------------------------------------- incremental writer

class ProgressWriter:
    """Durable, append-only incremental record of the mission's stage ladder + sim time.

    One line per transition; each ``append`` ``flush``es and ``fsync``s so a native crash mid-tick
    cannot lose an already-recorded transition. Best-effort: any I/O error is swallowed (the recorder
    must never take down the mission), and the file is a NEW sidecar so the success path's
    ``mission.json`` stays byte-identical.
    """

    def __init__(self, run_dir, *, actions=ACTIONS, log=None):
        self.path = Path(run_dir) / PROGRESS_LOG
        self.actions = tuple(actions)
        self._log = log
        self._done = []          # stages a 'succeeded' line was written for, in order

    def _append(self, record):
        try:
            with self.path.open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:  # noqa: BLE001 -- recording must never break the mission
            if self._log is not None:
                self._log('PROGRESS_WRITE skipped: %r' % (exc,))

    def transition(self, event, stage, sim_time, *, attempt=None, reason=None):
        """Record a stage ``event`` ('begin' | 'succeeded' | 'failed') at ``sim_time`` (sim seconds)."""
        if event == 'succeeded' and stage not in self._done:
            self._done.append(stage)
        rec = {'event': str(event), 'stage': stage,
               'sim_time': (float(sim_time) if sim_time is not None else None),
               'done': list(self._done)}
        if attempt is not None:
            rec['attempt'] = int(attempt)
        if reason:
            rec['reason'] = str(reason)
        self._append(rec)

    # Convenience aliases mirroring the mission seams.
    def begin(self, stage, sim_time, *, attempt=None):
        self.transition('begin', stage, sim_time, attempt=attempt)

    def succeeded(self, stage, sim_time, *, attempt=None):
        self.transition('succeeded', stage, sim_time, attempt=attempt)

    def failed(self, stage, sim_time, *, attempt=None, reason=None):
        self.transition('failed', stage, sim_time, attempt=attempt, reason=reason)


# --------------------------------------------------------------------------- recovery (pure)

def _iter_jsonl_tolerant(path):
    """Yield parsed JSON objects from a .jsonl file, SKIPPING a truncated / malformed trailing line.

    A native crash can leave the last appended line half-written; a truncated last line must not sink
    the whole recovery, so every undecodable line is skipped rather than raised."""
    p = Path(path)
    if not p.is_file():
        return
    try:
        text = p.read_text(errors='replace')
    except Exception:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue  # truncated / corrupt line (typically the last one before a crash)


@dataclass
class Progress:
    last_stage: str | None = None
    sim_time: float | None = None
    stages_done: tuple = ()
    stages_pending: tuple = ()
    last_reason: str | None = None
    last_vram: dict | None = None
    attempt: int | None = None
    sources: tuple = ()           # which artifacts contributed ('progress','bt','nav','vram')

    @property
    def ladder(self):
        """The full stage ladder as {done: [...], current: stage|None, pending: [...]}."""
        return {'done': list(self.stages_done),
                'current': self.last_stage,
                'pending': list(self.stages_pending)}


def _max_sim(a, b):
    vals = [v for v in (a, b) if v is not None]
    return max(vals) if vals else None


def scan_progress_log(run_dir):
    """Recover from ``mission-progress.jsonl`` (the richest source). Returns a dict or None."""
    rows = list(_iter_jsonl_tolerant(Path(run_dir) / PROGRESS_LOG))
    if not rows:
        return None
    done = []
    last = rows[-1]
    sim = None
    stage = None
    attempt = None
    reason = None
    for r in rows:
        if r.get('event') == 'succeeded':
            s = r.get('stage')
            if s and s not in done:
                done.append(s)
        sim = _max_sim(sim, r.get('sim_time'))
    stage = last.get('stage')
    attempt = last.get('attempt')
    reason = last.get('reason')
    # The current stage is the latest stage that has NOT been recorded succeeded.
    if stage in done:
        stage = None
    return {'last_stage': stage, 'sim_time': sim, 'done': tuple(done),
            'attempt': attempt, 'reason': reason}


def scan_bt_events(run_dir):
    """Recover the stage ladder + last reason from ``bt_events.jsonl`` (BT-driver runs). None if absent."""
    rows = list(_iter_jsonl_tolerant(Path(run_dir) / BT_EVENTS))
    if not rows:
        return None
    done = []
    current = None
    sim = None
    reason = None
    attempt = None
    for r in rows:
        data = r.get('data') or {}
        sim = _max_sim(sim, data.get('sim_time'))
        typ = r.get('type')
        if typ == 'stage_begin':
            current = data.get('stage')
            attempt = data.get('attempt')
        elif typ == 'stage_succeeded':
            s = data.get('stage')
            if s and s not in done:
                done.append(s)
            if s == current:
                current = None
        elif typ in ('stage_failed', 'seal_failed', 'coupling_failed'):
            if data.get('reason'):
                reason = data.get('reason')
    if current in done:
        current = None
    return {'last_stage': current, 'sim_time': sim, 'done': tuple(done),
            'attempt': attempt, 'reason': reason}


def scan_nav_trace(run_dir):
    """Recover the last stage + sim time from ``nav-trace.jsonl`` (written by every run, per tick).

    Reads only the tail for speed (the file can be tens of MB). None if absent/empty."""
    p = Path(run_dir) / NAV_TRACE
    if not p.is_file():
        return None
    try:
        size = p.stat().st_size
        with p.open('rb') as f:
            if size > 65536:
                f.seek(-65536, os.SEEK_END)
            tail = f.read().decode('utf-8', errors='replace')
    except Exception:
        return None
    action = None
    t = None
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue  # partial first line from the seek, or a truncated last line
        if 'action' in row:
            action = row.get('action') or action
        t = _max_sim(t, row.get('t'))
    if action is None and t is None:
        return None
    return {'last_stage': action, 'sim_time': t}


def read_vram(run_dir):
    """The last sampled VRAM headroom (``vram-headroom.json``) if the monitor wrote one, else None."""
    p = Path(run_dir) / VRAM_FILE
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def scan_progress(run_dir, *, actions=ACTIONS):
    """Merge every available durable artifact into a single :class:`Progress`.

    Precedence for the ladder/current stage: ``mission-progress.jsonl`` > ``bt_events.jsonl``; the last
    stage + sim time additionally fall back to ``nav-trace.jsonl`` (present on every run). The sim time
    is the MAX across sources. Pure."""
    run_dir = Path(run_dir)
    prog = scan_progress_log(run_dir)
    bt = scan_bt_events(run_dir)
    nav = scan_nav_trace(run_dir)
    vram = read_vram(run_dir)

    sources = []
    done = ()
    last_stage = None
    sim = None
    reason = None
    attempt = None
    # Prefer the progress log, then bt_events, for the ladder.
    for src, name in ((prog, 'progress'), (bt, 'bt')):
        if not src:
            continue
        sources.append(name)
        if not done and src.get('done'):
            done = tuple(src['done'])
        if last_stage is None and src.get('last_stage'):
            last_stage = src['last_stage']
        if reason is None and src.get('reason'):
            reason = src['reason']
        if attempt is None and src.get('attempt') is not None:
            attempt = src['attempt']
        sim = _max_sim(sim, src.get('sim_time'))
    if nav:
        sources.append('nav')
        if last_stage is None and nav.get('last_stage'):
            last_stage = nav['last_stage']
        sim = _max_sim(sim, nav.get('sim_time'))
    if vram is not None:
        sources.append('vram')

    # Derive done from the ladder position if we only know the current stage (nav-only recovery): every
    # stage BEFORE the current one in the ordered ladder is implicitly complete.
    actions = tuple(actions)
    if not done and last_stage in actions:
        done = actions[:actions.index(last_stage)]
    # Pending = the stages after the current one (or after the last done stage).
    pivot = last_stage if last_stage in actions else (done[-1] if done else None)
    if pivot in actions:
        pending = actions[actions.index(pivot) + 1:]
    else:
        pending = ()

    return Progress(last_stage=last_stage, sim_time=sim, stages_done=tuple(done),
                    stages_pending=tuple(pending), last_reason=reason, last_vram=vram,
                    attempt=attempt, sources=tuple(sources))


def build_crashed_summary(progress, signal, *, modes=None, extra=None):
    """Build the ``state="crashed"`` ``mission.json`` dict from a recovered :class:`Progress`.

    ``signal`` is the crash signal number (11 for SIGSEGV) or None (e.g. a wall-cap timeout)."""
    sig = None if signal is None else int(signal)
    sig_name = signal_name(sig)
    where = progress.last_stage or 'unknown stage'
    at = ('' if progress.sim_time is None else f' at sim t={progress.sim_time:.3f}')
    cause = (f'native crash (signal {sig} {sig_name})' if sig is not None
             else 'process terminated without a verdict (wall-cap timeout or unknown)')
    summary = {
        'state': 'crashed',
        'reason': f'{cause} during {where}{at}; mission.json reconstructed out-of-process from the '
                  f'durable run artifacts (the record-first teardown could not run on a native crash).',
        'physics_verified': False,
        'crash_signal': sig,
        'crash_signal_name': sig_name,
        'last_stage': progress.last_stage,
        'sim_time': progress.sim_time,
        'stage_ladder': progress.ladder,
        'recovery_sources': list(progress.sources),
        'note': 'CRASHED RUN: this verdict was reconstructed after the Isaac mission process died '
                'natively; it is NOT a completed or sensor-verified mission. See stage_ladder for how '
                'far the run got.',
    }
    if progress.last_reason:
        summary['last_fault'] = progress.last_reason
    if progress.attempt is not None:
        summary['last_attempt'] = progress.attempt
    if progress.last_vram is not None:
        summary['last_vram'] = progress.last_vram
    if modes:
        summary.update(modes)
    if extra:
        summary.update(extra)
    return summary


def write_if_missing(run_dir, signal, *, modes=None, extra=None, actions=ACTIONS, log=None):
    """Write a reconstructed ``state="crashed"`` ``mission.json`` IFF one does not already exist.

    Returns the path written, or None when ``mission.json`` already exists (clean success/failure) so
    the normal path is byte-identical. The out-of-process finalizer (shell parent / orchestrator).
    """
    run_dir = Path(run_dir)
    out = run_dir / MISSION_JSON
    if out.exists():
        if log is not None:
            log(f'FINALIZE_CRASH skip: {out} already exists (clean verdict kept)')
        return None
    progress = scan_progress(run_dir, actions=actions)
    summary = build_crashed_summary(progress, signal, modes=modes, extra=extra)
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if log is not None:
        log(f'FINALIZE_CRASH wrote {out}: state=crashed last_stage={progress.last_stage} '
            f'sim_time={progress.sim_time} signal={signal_name(signal)} sources={progress.sources}')
    return out
