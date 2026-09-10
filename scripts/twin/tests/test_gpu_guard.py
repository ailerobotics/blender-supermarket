"""GPU guard parsing + refusal logic — the safety gate that protects Blender."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from twin import gpu_guard as GG  # noqa: E402


# A captured nvidia-smi Processes table (this box), incl. the protected Blender GUI.
SMI = """\
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|=========================================================================================|
|    0   N/A  N/A            1189      G   /usr/lib/xorg/Xorg                      149MiB |
|    0   N/A  N/A         2504793      C   ./venv/bin/python                      2184MiB |
|    0   N/A  N/A         2996355    C+G   ...nishsaini/opt/blender/blender        171MiB |
+-----------------------------------------------------------------------------------------+
"""


def test_parse_meminfo():
    assert GG.parse_meminfo("12288, 6096, 5815") == (12288, 6096, 5815)


def test_parse_processes_captures_graphics_and_compute():
    procs = GG.parse_processes(SMI)
    names = {p.name for p in procs}
    assert any("blender" in n for n in names)         # graphics C+G row captured
    assert any("venv/bin/python" in n for n in names)  # heavy compute row captured
    blender = next(p for p in procs if "blender" in p.name)
    assert blender.pid == 2996355 and blender.mem_mib == 171


def test_protected_blender_is_hard_refusal_even_with_force():
    procs = GG.parse_processes(SMI)
    d = GG.assess(free_mb=9000, total_mb=12288, processes=procs, our_pid=999,
                  force=True)  # generous headroom + force
    assert d.ok is False
    assert d.hard is True                       # not overridable
    assert "blender" in d.message.lower()
    assert "close" in d.message.lower()


def test_low_headroom_refused_but_forceable():
    d_no = GG.assess(free_mb=5815, total_mb=12288, processes=[], our_pid=1,
                     required_free_mb=8000, force=False)
    assert d_no.ok is False and d_no.hard is False
    d_force = GG.assess(free_mb=5815, total_mb=12288, processes=[], our_pid=1,
                        required_free_mb=8000, force=True)
    assert d_force.ok is True
    assert "force" in d_force.message.lower()


def test_heavy_other_consumer_refused():
    procs = [GG.GpuProcess(pid=42, ptype="C", name="./venv/bin/python", mem_mib=2184)]
    d = GG.assess(free_mb=9000, total_mb=12288, processes=procs, our_pid=1,
                  heavy_threshold_mb=1500, force=False)
    assert d.ok is False and d.hard is False
    assert "2184" in d.message


def test_our_own_process_is_not_a_blocker():
    procs = [GG.GpuProcess(pid=7, ptype="C", name="python", mem_mib=4000)]
    d = GG.assess(free_mb=9000, total_mb=12288, processes=procs, our_pid=7)
    assert d.ok is True


def test_clean_gpu_passes():
    d = GG.assess(free_mb=11000, total_mb=12288, processes=[], our_pid=1)
    assert d.ok is True and "passed" in d.message.lower()
