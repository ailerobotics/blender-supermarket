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


# --------------------------------------------------------------------- residual-bind probe
# sim-place-release-residual-bind-v1 (coldwin106): after the REAL contact._release_fallback(), is the
# "freed" carton actually free? The driver (run_release_jump_probe.py --residual-bind) samples EVERY
# tick across three phases and feeds the per-tick dicts here for a PURE (CPU-testable) reading of which
# bind survives. A sample dict carries:
#   phase            one of BIND_PHASES
#   tick             int, monotonic within the run
#   carton_pos       [x, y, z]   world metres
#   carton_vel       [vx, vy, vz] world m/s (linear)
#   pad_pos          [x, y, z]   world metres of the gripper TCP/pad (the arm reference for the lift)
#   kinematic_enabled  bool|None  carton RigidBodyAPI physics:kinematicEnabled
#   collision_enabled  bool|None  any carton-subtree CollisionAPI physics:collisionEnabled is True
#   joint_refs       list of {path, enabled} for every stage joint whose body0/body1 targets the carton
#   contacts         list of prim-path strings the carton touched this tick (PhysX contact report)
BIND_PHASES = ('pre_release', 'post_release_still', 'lift')
_GRAV_FALL_M = 0.010   # a free carton drops >= this under gravity across the still-hold window
_HOVER_M = 0.003       # a bound carton's z barely moves while the arm is still
_FOLLOW_RATIO = 0.5    # carton_disp / pad_disp >= this during the lift => the carton tracks the arm


def _phase(samples, phase):
    return [s for s in samples if s.get('phase') == phase]


def _disp(a, b):
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) ** 0.5


def classify_bind(samples):
    """PURE reading of which release bind survives, from the per-tick BIND samples.

    Returns a dict with the ``verdict`` (one of 'free', 'kinematic_flag', 'surviving_constraint',
    'pose_follow', 'inconclusive'), the measured evidence, and a human ``summary``. The decision order
    puts the DIRECT scene-graph evidence first (a still-True kinematic flag or a still-bound joint names
    the mechanism outright), then the kinematic behaviour (does the still carton fall or hover; does the
    lifted carton track the arm), so a run with no surviving flag/joint but a carton that hovers+tracks
    still reports a non-contact bind rather than 'free'.
    """
    still = _phase(samples, 'post_release_still')
    lift = _phase(samples, 'lift')
    ev = {}

    # (1) direct scene-graph state after release (sampled every still tick; take the last)
    kin = next((s.get('kinematic_enabled') for s in reversed(still)
                if s.get('kinematic_enabled') is not None), None)
    bound_joints = sorted({r['path'] for s in still for r in (s.get('joint_refs') or [])
                           if r.get('enabled')})
    coll = next((s.get('collision_enabled') for s in reversed(still)
                 if s.get('collision_enabled') is not None), None)
    ev['kinematic_enabled'] = kin
    ev['bound_joint_paths'] = bound_joints
    ev['collision_enabled'] = coll

    # (2) still-hold kinematic behaviour: did the carton fall (free) or hover (bound)?
    still_drop_m = still_peak_speed = None
    if still:
        z0 = float(still[0]['carton_pos'][2])
        zs = [float(s['carton_pos'][2]) for s in still]
        still_drop_m = z0 - min(zs)                     # positive => fell
        still_peak_speed = max((sum(float(v) ** 2 for v in s['carton_vel']) ** 0.5) for s in still)
    ev['still_drop_m'] = still_drop_m
    ev['still_peak_speed_mps'] = still_peak_speed
    ev['still_fell'] = (still_drop_m is not None and still_drop_m >= _GRAV_FALL_M)
    ev['still_hovered'] = (still_drop_m is not None and still_drop_m < _HOVER_M)

    # (3) lift behaviour: does the carton track the arm (pad)? Use the DIRECTIONAL projection of the
    #     carton displacement onto the pad displacement (dot / |pad|^2), so a carton that FELL during the
    #     lift (large magnitude, opposite direction) reads ~<=0, not a spurious "track".
    follow_ratio = pad_disp = carton_disp = None
    if len(lift) >= 2:
        pad_vec = [float(lift[-1]['pad_pos'][i]) - float(lift[0]['pad_pos'][i]) for i in range(3)]
        carton_vec = [float(lift[-1]['carton_pos'][i]) - float(lift[0]['carton_pos'][i]) for i in range(3)]
        pad_disp = sum(c * c for c in pad_vec) ** 0.5
        carton_disp = sum(c * c for c in carton_vec) ** 0.5
        if pad_disp > 1e-6:
            follow_ratio = sum(pad_vec[i] * carton_vec[i] for i in range(3)) / (pad_disp * pad_disp)
    ev['lift_pad_disp_m'] = pad_disp
    ev['lift_carton_disp_m'] = carton_disp
    ev['lift_follow_ratio'] = follow_ratio       # projection: ~1 tracks the arm, <=0 fell/diverged
    tracks = (follow_ratio is not None and follow_ratio >= _FOLLOW_RATIO)
    ev['lift_tracks_arm'] = tracks

    # (4) decide
    if kin is True:
        verdict = 'kinematic_flag'
    elif bound_joints:
        verdict = 'surviving_constraint'
    elif ev['still_fell'] and not tracks:
        verdict = 'free'
    elif ev['still_hovered'] or tracks:
        verdict = 'pose_follow'
    else:
        verdict = 'inconclusive'
    ev['verdict'] = verdict
    ev['summary'] = _bind_summary(verdict, ev)
    return ev


def _bind_summary(verdict, ev):
    if verdict == 'kinematic_flag':
        return ('carton RigidBody still physics:kinematicEnabled=True after release -> a kinematic flag '
                'survives: the body ignores gravity and pose-follows the tool (not a contact/constraint)')
    if verdict == 'surviving_constraint':
        return ('a stage joint still targets the carton after release (%s) -> a surviving '
                'joint/constraint binds it to the tool' % ', '.join(ev['bound_joint_paths']))
    if verdict == 'free':
        return ('carton fell %.1f mm under gravity while the arm held still and did NOT track the lift '
                '(follow_ratio=%s) -> the released carton IS free in the bench; the mission bind does not '
                'reproduce here (env/scene-specific)' % ((ev['still_drop_m'] or 0) * 1000,
                                                         ev['lift_follow_ratio']))
    if verdict == 'pose_follow':
        return ('no surviving kinematic flag or joint, yet the still carton hovered (drop %.1f mm) and/or '
                'tracked the lift (follow_ratio=%s) -> a non-contact POSE-FOLLOW bind survives'
                % ((ev['still_drop_m'] or 0) * 1000, ev['lift_follow_ratio']))
    return 'inconclusive: the still carton neither clearly fell nor clearly hovered/tracked'


def format_bind(ev):
    return ('RESIDUAL_BIND verdict=%s kinematic_enabled=%s bound_joints=%s collision_enabled=%s '
            'still_drop_mm=%s still_peak_speed=%s lift_pad_disp_mm=%s lift_carton_disp_mm=%s '
            'follow_ratio=%s' % (
                ev['verdict'], ev['kinematic_enabled'], ev['bound_joint_paths'] or '[]',
                ev['collision_enabled'],
                None if ev['still_drop_m'] is None else round(ev['still_drop_m'] * 1000, 2),
                None if ev['still_peak_speed_mps'] is None else round(ev['still_peak_speed_mps'], 4),
                None if ev['lift_pad_disp_m'] is None else round(ev['lift_pad_disp_m'] * 1000, 2),
                None if ev['lift_carton_disp_m'] is None else round(ev['lift_carton_disp_m'] * 1000, 2),
                None if ev['lift_follow_ratio'] is None else round(ev['lift_follow_ratio'], 3)))
