"""Refuse to boot Isaac Sim when another heavy GPU consumer is already resident.

The workstation is tight (RTX 3060, 12 GB) and the captain's live Blender GUI
holds the scene on the same GPU; Isaac Sim has OOM-crashed here before. Booting a
56k-prim RTX scene beside Blender risks an OOM kill of a window with unsaved work.

This module PARSES ``nvidia-smi`` and DECIDES; the pure decision (assess) is unit
tested with captured text. Two gates:
  * a protected process (Blender) present  -> hard refuse, NOT overridable by --force
    (closing Blender first is the whole point of this task);
  * low free VRAM or any other heavy consumer -> refuse, overridable by --force
    (the captain's explicit call).
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# VRAM-conscious defaults (MiB). Isaac Sim + the RTX scene wants real headroom.
DEFAULT_REQUIRED_FREE_MB = 8000
DEFAULT_HEAVY_THRESHOLD_MB = 1500
PROTECTED_NAME_SUBSTRINGS = ("blender",)

_PROC_RE = re.compile(
    r"^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(\S+)\s+(.*?)\s+(\d+)\s*MiB\s*\|\s*$")


@dataclass
class GpuProcess:
    pid: int
    ptype: str      # C, G, C+G
    name: str
    mem_mib: int


@dataclass
class GpuDecision:
    ok: bool
    message: str
    total_mb: Optional[int] = None
    free_mb: Optional[int] = None
    hard: bool = False          # True => not overridable by --force
    consumers: List[GpuProcess] = field(default_factory=list)


def parse_meminfo(csv_line: str) -> tuple[int, int, int]:
    """Parse 'total, used, free' from nvidia-smi --query-gpu ... --format=csv,noheader,nounits."""
    parts = [p.strip() for p in csv_line.split(",")]
    return int(parts[0]), int(parts[1]), int(parts[2])


def parse_processes(smi_text: str) -> List[GpuProcess]:
    """Parse the Processes table of full ``nvidia-smi`` text (captures BOTH compute
    and graphics rows; --query-compute-apps misses graphics-only memory)."""
    out: List[GpuProcess] = []
    for line in smi_text.splitlines():
        m = _PROC_RE.match(line)
        if m:
            out.append(GpuProcess(pid=int(m.group(1)), ptype=m.group(2),
                                  name=m.group(3).strip(), mem_mib=int(m.group(4))))
    return out


def _is_protected(name: str, protected: Sequence[str]) -> bool:
    low = name.lower()
    return any(s in low for s in protected)


def assess(
    free_mb: int,
    total_mb: int,
    processes: Sequence[GpuProcess],
    our_pid: int,
    required_free_mb: int = DEFAULT_REQUIRED_FREE_MB,
    heavy_threshold_mb: int = DEFAULT_HEAVY_THRESHOLD_MB,
    protected: Sequence[str] = PROTECTED_NAME_SUBSTRINGS,
    force: bool = False,
) -> GpuDecision:
    """Decide whether it is safe to boot Isaac Sim. Pure — no subprocess."""
    others = [p for p in processes if p.pid != our_pid]
    protected_hits = [p for p in others if _is_protected(p.name, protected)]
    heavy = [p for p in others if p.mem_mib >= heavy_threshold_mb
             and not _is_protected(p.name, protected)]

    # Gate 1 (hard): the protected session is live.
    if protected_hits:
        who = ", ".join(f"{p.name} (pid {p.pid}, {p.mem_mib} MiB)" for p in protected_hits)
        return GpuDecision(
            ok=False, hard=True, total_mb=total_mb, free_mb=free_mb, consumers=protected_hits,
            message=(
                f"REFUSING to boot Isaac Sim: a protected GPU session is running — {who}.\n"
                "  This is almost certainly the captain's live Blender GUI holding "
                "scenes/supermarket.blend (MCP :9876) with unsaved work.\n"
                "  Booting a 56k-prim RTX scene beside it risks an OOM kill of that window.\n"
                "  Close/save Blender FIRST (the captain's decision), then re-run. "
                "--force does NOT override this gate."))

    # Gate 2 (soft): low headroom.
    if free_mb < required_free_mb and not force:
        return GpuDecision(
            ok=False, hard=False, total_mb=total_mb, free_mb=free_mb,
            message=(
                f"REFUSING to boot Isaac Sim: only {free_mb} MiB VRAM free of {total_mb} "
                f"(need >= {required_free_mb}).\n"
                "  Free the GPU or, if you accept the OOM risk, re-run with --force."))

    # Gate 3 (soft): another heavy consumer.
    if heavy and not force:
        who = ", ".join(f"{p.name} (pid {p.pid}, {p.mem_mib} MiB)" for p in heavy)
        return GpuDecision(
            ok=False, hard=False, total_mb=total_mb, free_mb=free_mb, consumers=heavy,
            message=(
                f"REFUSING to boot Isaac Sim: heavy GPU consumer(s) present — {who}.\n"
                "  Free the GPU or re-run with --force to accept the risk."))

    note = ""
    if force and free_mb < required_free_mb:
        note = f" (--force: proceeding with only {free_mb} MiB free — OOM risk accepted)"
    return GpuDecision(ok=True, hard=False, total_mb=total_mb, free_mb=free_mb,
                       message=f"GPU check passed: {free_mb} MiB free of {total_mb}." + note)


def query_and_assess(force: bool = False, **kw) -> GpuDecision:
    """Run nvidia-smi and assess. Missing nvidia-smi -> hard refuse (can't verify)."""
    try:
        mem = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()[0]
        total, _used, free = parse_meminfo(mem)
        full = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                              timeout=15, check=True).stdout
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        return GpuDecision(ok=False, hard=True,
                           message=f"REFUSING: could not query nvidia-smi ({exc!r}); "
                                   "cannot verify the GPU is free.")
    procs = parse_processes(full)
    return assess(free, total, procs, os.getpid(), force=force, **kw)
