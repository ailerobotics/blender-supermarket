"""move_group respawn supervision policy -- pure, import-light, CPU-testable.

The host-stack launch (``twin.restock_host.launch``) relaunches move_group on an abnormal
process EXIT via ROS 2 launch's native respawn (``respawn=True``). This module owns the two
knobs that launch is given (a BOUNDED retry count + a delay) and the observability the raw
launch respawn lacks: each respawn and the final give-up are logged BY NAME with the attempt
count. It imports nothing from ROS/launch so the policy is unit-tested with no GPU boot.

Audit (sim-robust-movegroup-respawn-reliability-v1): ROS 2 launch respawn fires only on a
real process EXIT (``ProcessExited`` in ``launch/actions/execute_local.py``). A move_group
that EXITS on the cuMotion SIGABRT (exit -6: dblproof, coldwin7) is respawned reliably; a
move_group that HANGS in glibc's abort path (coldliveapproach: ``malloc_consolidate():
invalid chunk size`` with NO following ``process has died`` line) never exits, so NO launch
respawn can fire -- that case is owned by the client-side crash-retry (ros_adapter) and its
``RESPAWN_WAIT_CAP`` fast-fail (PR 234 / PR 242), not here.

The supervisor here is deliberately OBSERVE-ONLY: it never respawns or kills a process, so it
cannot fight the launch respawn. The single respawner stays ROS 2 launch.
"""
import os

# Bounded respawn budget + delay the launch hands move_group when the host config carries no
# ``movegroup_respawn`` block. Default 3 lifetime respawns: coldwin7 (a full honest cycle)
# crashed move_group once, so 3 is generous headroom for a healthy-but-crashy run while still
# bounding a FLAPPING planner (crash-on-every-plan) to ~3 respawns before the client fast-fails
# by name instead of masking the fault behind an unbounded respawn loop. Env-overridable;
# clamped non-negative. A move_group that never dies never consults this (byte-identical).
DEFAULT_MAX_RETRIES = max(0, int(os.environ.get('RESTOCK_MOVEGROUP_RESPAWN_MAX', 3)))
# The short delay lets the old process release its endpoints before the replacement binds
# them (see launch.py). Unchanged from the pre-audit 3.0 s.
DEFAULT_RESPAWN_DELAY_S = max(0.0, float(os.environ.get('RESTOCK_MOVEGROUP_RESPAWN_DELAY_S', 3.0)))


def resolve_respawn(config):
    """Resolve ``(max_retries, delay)`` from a host config's optional ``movegroup_respawn``
    block (keys ``max_retries`` and ``delay``/``respawn_delay``), clamped non-negative. An
    absent or empty block => the module defaults, so a config that never mentions respawn runs
    the bounded default. Pure (dict in, tuple out) for CPU testing."""
    block = (config or {}).get('movegroup_respawn') or {}
    max_retries = block.get('max_retries', DEFAULT_MAX_RETRIES)
    delay = block.get('delay', block.get('respawn_delay', DEFAULT_RESPAWN_DELAY_S))
    return max(0, int(max_retries)), max(0.0, float(delay))


def respawn_decision(death_count, max_retries):
    """Pure policy mirroring ROS 2 launch's own respawn budget test: given the ``death_count``th
    process EXIT (1-based) and the ``max_retries`` budget, launch respawns while
    ``death_count <= max_retries`` and gives up otherwise. Returns
    ``('respawn', attempt, max_retries)`` or ``('give_up', max_retries, max_retries)``."""
    if death_count <= max_retries:
        return ('respawn', death_count, max_retries)
    return ('give_up', max_retries, max_retries)


def respawn_log_line(death_count, max_retries, delay, pid=None, returncode=None):
    """The by-name, greppable host-stack.log line for the ``death_count``th move_group exit:
    ``MOVE_GROUP_RESPAWN ... attempt N/M`` while within budget, else
    ``MOVE_GROUP_RESPAWN_GIVE_UP ...``. Pure (CPU-testable); the launch handler prints it."""
    decision, attempt, limit = respawn_decision(death_count, max_retries)
    where = ''
    if pid is not None:
        where += ' pid=%s' % pid
    if returncode is not None:
        where += ' exit=%s' % returncode
    if decision == 'respawn':
        return ('MOVE_GROUP_RESPAWN move_group died%s; relaunching move_group '
                '(attempt %d/%d, respawn_delay=%.1fs)' % (where, attempt, limit, delay))
    return ('MOVE_GROUP_RESPAWN_GIVE_UP move_group died%s and the bounded respawn budget is '
            'exhausted (%d/%d respawns used); move_group will NOT be relaunched -- the planning '
            'client fast-fails the stage by name (ros_adapter RESPAWN_WAIT_CAP)'
            % (where, limit, limit))


class RespawnSupervisor:
    """Observe-only move_group respawn watcher. Counts process EXITs and renders the by-name
    respawn / give-up log line with the attempt count. It NEVER respawns or kills a process
    (no such method exists), so it cannot fight ROS 2 launch's native respawn -- the single
    respawner stays launch. One instance per launch; its :meth:`on_exit` is the
    ``OnProcessExit`` callback. Import-light for CPU testing: ``emit`` is injected (print in
    production, a capture list in tests)."""

    def __init__(self, max_retries, delay, emit):
        self.max_retries = max(0, int(max_retries))
        self.delay = max(0.0, float(delay))
        self._emit = emit        # callable(str) -> None (print/log); injected
        self.deaths = 0

    def note_exit(self, pid=None, returncode=None):
        """Record one move_group EXIT, emit the by-name line, and return it. Returns the
        decision tuple's verb alongside for callers/tests that want the branch."""
        self.deaths += 1
        line = respawn_log_line(self.deaths, self.max_retries, self.delay, pid, returncode)
        self._emit(line)
        return line

    @property
    def gave_up(self):
        """True once the respawn budget is spent (the next/last exit is terminal)."""
        return self.deaths > self.max_retries

    def on_exit(self, event, context):
        """``OnProcessExit`` callback. Logs the exit by name with the attempt count and returns
        no launch actions -- observe-only, so it adds no second teardown. ``event`` carries
        ``pid`` and ``returncode`` (ProcessExited); both optional so a fixture event works."""
        self.note_exit(getattr(event, 'pid', None), getattr(event, 'returncode', None))
        return None
