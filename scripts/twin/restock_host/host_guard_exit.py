"""sim-robust-pick-wrist-guard-cspace-fix-v1: name a shutdown-guard process death.

When a guarded host component dies (e.g. the ``component_container_mt`` holding cuMotion, which
exited -6 on a ``cumotion::FatalException CUDA error: out of memory`` after the PR254 malformed
c-space target reject storm), ROS 2 launch tears the whole host stack down via an ``OnProcessExit``
``Shutdown``. The trajectory executor then only saw that it was being SIGINT'd and faulted with the
bare ``executor stopped``, masking the real cause.

These helpers name the death -- the dying process, its exit code, and its last error line -- and
bridge it across the process boundary via a small marker file (path in ``RESTOCK_HOST_GUARD_MARKER``),
so the arm host's last-resort fault carries the cause instead of ``executor stopped``. Pure/IO-only
and import-light (stdlib only) so launch.py, trajectory_host.py and the CPU tests can all use them.
"""
import json
import re

MARKER_ENV = 'RESTOCK_HOST_GUARD_MARKER'

# Strip ANSI colour escapes (the cuMotion FatalException line is wrapped in them in host-stack.log).
_ANSI = re.compile(r'\x1b\[[0-9;]*m')


def clean_line(text):
    """A single trimmed, ANSI-stripped line from a process-output chunk, or '' if nothing meaningful.

    Keeps the LAST non-empty line of a possibly multi-line chunk, so a terminal error line is captured
    even when several arrive together. Pure."""
    if not text:
        return ''
    if isinstance(text, (bytes, bytearray)):
        text = text.decode('utf-8', 'replace')
    best = ''
    for raw in str(text).splitlines():
        line = _ANSI.sub('', raw).strip()
        if line:
            best = line
    return best


def guard_exit_reason(process_name, returncode, last_error_line=None):
    """A named host-component-death reason string. Pure."""
    reason = "host component '%s' exited (exit code %s)" % (process_name, returncode)
    detail = clean_line(last_error_line)
    if detail:
        reason += ': ' + detail
    return reason


def guard_exit_record(process_name, returncode, last_error_line=None):
    """The dict persisted to / read from the marker. Pure."""
    return {'process_name': process_name, 'returncode': returncode,
            'last_error_line': clean_line(last_error_line) or None,
            'reason': guard_exit_reason(process_name, returncode, last_error_line)}


def write_guard_exit(path, process_name, returncode, last_error_line=None):
    """Record a guarded-process death to the marker file. Best-effort: never raises (a teardown must not
    crash on a full/readonly fs). Writes nothing when ``path`` is falsy."""
    if not path:
        return
    try:
        with open(path, 'w') as handle:
            json.dump(guard_exit_record(process_name, returncode, last_error_line), handle)
    except OSError:
        pass


def read_guard_exit(path):
    """Return the recorded guard-exit dict, or None when the marker is absent/empty/malformed."""
    if not path:
        return None
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get('process_name') is None:
        return None
    return data


def guard_exit_fault_reason(marker, default):
    """The honest executor fault reason: the recorded guard-exit reason if a guarded component died,
    else ``default`` (the generic last-resort marker). Pure."""
    if isinstance(marker, dict) and marker.get('reason'):
        return marker['reason']
    return default
