"""No boot until exclusive GPU ownership, protected-session and resource checks pass."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import socket

from twin.gpu_guard import query_and_assess


def available_ram_mb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    raise RuntimeError('cannot determine available RAM')


def isolation():
    required = {'ROS_DOMAIN_ID': '88', 'ROS_LOCALHOST_ONLY': '1',
                'ROS_AUTOMATIC_DISCOVERY_RANGE': 'LOCALHOST'}
    for key, value in required.items():
        if os.environ.get(key, value) != value:
            raise RuntimeError(f'{key} must be {value}')
    os.environ.update(required)


def protected_blender():
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            if 'blender' in path.read_text().lower():
                return True
        except FileNotFoundError:
            pass  # process exited during enumeration
        except PermissionError:
            raise RuntimeError('cannot inspect protected process state')
    with socket.socket() as sock:
        sock.settimeout(.2)
        return sock.connect_ex(('127.0.0.1', 9876)) == 0


@contextmanager
def gpu_slot(*, heavy_threshold_mb=1500):
    """Held for the entire simulator lifetime. No force option exists."""
    if heavy_threshold_mb not in (1500, 2500):
        raise ValueError('heavy threshold must use default 1500 or explicit opt-in 2500')
    isolation()
    with open('/tmp/aile-gpu.lock', 'a') as lock:
        # A parent orchestrator may hold /tmp/aile-gpu.lock across its WHOLE sequence
        # (commissioning boot + mission boot, no gap) so a concurrent bench cannot slip into the
        # gap, inheriting that hold to us via AILE_GPU_LOCK_INHERITED. Re-acquiring here would
        # false-fail on the parent's own exclusive lock, so trust the parent's hold and skip the
        # acquire (every other guard below still runs). Unset -> the original exclusive acquire.
        if not os.environ.get('AILE_GPU_LOCK_INHERITED'):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if protected_blender():
            raise RuntimeError('protected Blender process or MCP socket is present')
        if available_ram_mb() < 12*1024:
            raise RuntimeError('need at least 12 GiB available RAM')
        decision = query_and_assess(required_free_mb=8000,
                                    heavy_threshold_mb=heavy_threshold_mb)
        if not decision.ok:
            raise RuntimeError(decision.message)
        # DDS domain 88 discovery port. Occupancy refuses; never switch to rig domains.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 7400+250*88))
        yield decision
