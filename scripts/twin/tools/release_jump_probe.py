"""Pure core for the fm378 release-jump VRAM probe.

Tests the hypothesis that the full mission's discrete ~+2.68 GB VRAM jump (sim ~223-226 s, runs
285/297/299) is triggered by the carton RELEASE (SIM_ATTACH_FALLBACK detach) turning the carton from
a welded/kinematic body into a FREE DYNAMIC body -- an RTX acceleration-structure / instancing rebuild.

This module is PURE (no Isaac/omni imports) so it is CPU-importable and unit-testable: it holds the
VRAM sampler, the sample trace with phase labels, and the jump verdict. The Isaac driver that boots the
scene, welds the carton and calls the real ``contact._release_fallback()`` lives in the guarded runner
``scripts/run_release_jump_probe.py`` (it imports this core).
"""
import subprocess

# The sample phases, in order. 'detach' marks the single tick the release is issued.
PHASES = ('pre_weld', 'welded', 'detach', 'post_detach', 'post_destroy')


def vram_used_mib():
    """Current GPU ``memory.used`` in MiB via nvidia-smi, or ``None`` when unavailable.

    Deliberately shells out (same source as tools/vram_guard --sample) so the probe measures the whole
    card exactly as the orchestrator's sampler does -- a reallocation inside the RTX renderer shows here
    whether or not it is visible to any in-process API."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=15)
        line = out.stdout.strip().splitlines()[0].strip()
        return int(line)
    except Exception:
        return None


class ReleaseTrace:
    """Ordered VRAM samples tagged by phase (one of :data:`PHASES`) and probe-relative seconds."""

    def __init__(self):
        self.samples = []

    def add(self, phase, t_s, used_mib):
        if phase not in PHASES:
            raise ValueError(f'unknown phase {phase!r}')
        self.samples.append({'phase': phase, 't_s': round(float(t_s), 2), 'used_mib': used_mib})

    def used(self, phase):
        """The non-None used_mib samples recorded in ``phase``."""
        return [s['used_mib'] for s in self.samples if s['phase'] == phase and s['used_mib'] is not None]


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else None


def verdict(trace, *, threshold_mib=1000):
    """Compare the WELDED baseline (median of welded-phase samples) to the PEAK of the post-detach
    samples. ``detected`` iff that rise is >= ``threshold_mib`` -- i.e. a discrete jump appeared within
    the post-release window, reproducing the mission signature.

    Falls back to the pre_weld median for the baseline only if no welded sample was taken (so a verdict
    is still computable), flagged via ``baseline_phase``.
    """
    welded = trace.used('welded')
    baseline_phase = 'welded'
    if not welded:
        welded = trace.used('pre_weld')
        baseline_phase = 'pre_weld'
    post = trace.used('post_detach')
    baseline = _median(welded)
    peak_after = max(post) if post else None
    delta = (peak_after - baseline) if (baseline is not None and peak_after is not None) else None
    detected = bool(delta is not None and delta >= threshold_mib)
    return {'detected': detected, 'baseline_mib': baseline, 'baseline_phase': baseline_phase,
            'peak_after_detach_mib': peak_after, 'delta_mib': delta, 'threshold_mib': threshold_mib}


def format_verdict(v):
    return ('RELEASE_JUMP detected=%s baseline=%s peak_after_detach=%s delta=%s (baseline_phase=%s '
            'threshold=%s MiB)' % (v['detected'], v['baseline_mib'], v['peak_after_detach_mib'],
                                   v['delta_mib'], v['baseline_phase'], v['threshold_mib']))
