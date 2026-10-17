"""Boot-once arm-carry bench: prove the carton-carrying arm MOVES (tuck fold, pan,
pre-place, place, retreat) in minutes instead of a 15-minute mission run, and sweep the
carry drive gains, the velocity/acceleration scaling of the carrying moves and the
sim-only tracking/settle tolerances to recommend the most robust set.

WHAT IS EXERCISED IDENTICALLY TO THE LIVE MISSION
    * The real Isaac arm position drive (gains applied through the SAME
      ``articulation_controller.set_gains`` / ``set_max_efforts`` path
      ``physics_setup`` uses) -- so a limit cycle / drive stall reproduces.
    * The real ``ArmCommandGate.command`` path (same speed + tracking envelope the
      mission drives the UR10 with), fed the sampled trajectory exactly as
      ``NativeTransport.tick`` feeds it.
    * The real ``trajectory_host`` Plan interpolation, ``tracking_fault`` debounce and
      the terminal settle gate (GOAL_POSITION_TOL / GOAL_VELOCITY_TOL /
      GOAL_SETTLE_MARGIN_S / TRACKING_TOL / TRACKING_HARD_TOL / TRACKING_DEBOUNCE),
      imported from ``trajectory_host`` so the tolerances are the real ones.
    * The carton attached at the tool through the mission's own SIM_ATTACH_FALLBACK
      fixed-joint weld, welded statically at the measured grasp pose; the base held in
      the mission's airborne-carry braked state (yaw-lock D6 brake).

WHAT HONESTLY DIFFERS FROM THE LIVE MISSION (state this with every result)
    * PLANNING PATH: the carrying move trajectories are generated analytically here
      (rest-to-rest cubic per joint goal) instead of by MoveIt/cuMotion. The joint-space
      goals, the carton load and the drive are identical; only the planner that produced
      the waypoints differs. The ``slow`` velocity/acceleration SCALING is reproduced as
      an explicit ``peak_speed``/``peak_accel`` per carrying move (a sweep axis).
    * TRANSPORT/TIMING: the executor loop runs in-process against the live articulation
      on the sim clock rather than over the ROS /clock+/joint_states+/arm_joint_command
      relay round-trip. The sim-clock-based tolerances (the ones that decide pass/fail)
      are preserved; the wall-clock backstops are not the subject of this bench.

This module keeps the pure logic (trajectory build, executor loop, metrics, sweep grid)
Isaac-free so it is CPU unit-testable; the only Isaac-dependent part is the thin
``IsaacCarryStage`` glue, driven by ``scripts/run_arm_carry_bench.py`` inside a guarded,
already-booted ``SimulationApp``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

# Reuse the REAL executor logic + tolerances so the bench grades carrying moves exactly
# as trajectory_host does. These imports are ROS/Isaac-free (module-level).
from twin.restock.trajectory_host import (
    Plan, tracking_fault,
    GOAL_POSITION_TOL, GOAL_VELOCITY_TOL, GOAL_SETTLE_MARGIN_S,
    TRACKING_TOL, TRACKING_DEBOUNCE, TRACKING_HARD_TOL)
from twin.restock_manipulation.cad_tool import ARM_JOINTS

WRIST_1 = ARM_JOINTS.index('ur_wrist_1_joint')  # index 3 -- the carry-fold's binding joint


# --------------------------------------------------------------------------- specs

@dataclass(frozen=True)
class Tolerances:
    """The sim-only executor tolerances. Defaults ARE trajectory_host's live constants;
    override to sweep them. ``goal_settle_margin_s`` bounds how long after the planned
    end the move may ring before a GOAL_MISS."""
    goal_position_tol: float = GOAL_POSITION_TOL
    goal_velocity_tol: float = GOAL_VELOCITY_TOL
    goal_settle_margin_s: float = GOAL_SETTLE_MARGIN_S
    tracking_tol: float = TRACKING_TOL
    tracking_debounce: int = TRACKING_DEBOUNCE
    tracking_hard_tol: float = TRACKING_HARD_TOL

    def validate(self):
        if not (0 < self.goal_position_tol and 0 < self.goal_velocity_tol
                and 0 < self.goal_settle_margin_s and 0 < self.tracking_tol
                and self.tracking_hard_tol > self.tracking_tol
                and int(self.tracking_debounce) >= 1):
            raise ValueError('invalid executor tolerances (hard must exceed tol; debounce>=1)')
        return self


# UR10 physically-grounded per-joint effort limits (Nm), ARM_JOINTS order
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3].
UR10_EFFORTS = (330.0, 330.0, 330.0, 56.0, 56.0, 56.0)


@dataclass(frozen=True)
class DriveGains:
    """Isaac arm position-drive gains. ``drive_type`` selects the PhysX drive mode: 'acceleration'
    (the mission default -- stiffness/damping are MASS-NORMALISED, so Kp 1e5-1e6 is enormous) or
    'force' (stiffness/damping/effort in physical Nm units, sizeable to the UR10). When
    ``per_joint_effort`` is True the max effort is the UR10 per-joint vector (wrists ~56 Nm) instead
    of a uniform ceiling; stiffness/damping are likewise scaled per joint by the effort ratio so the
    wrists are not over-stiffened relative to their torque authority."""
    stiffness: float            # Kp (accel units if acceleration drive; Nm/rad if force drive)
    damping: float              # Kd
    max_effort: float           # per-joint maxForce ceiling (Nm); base value when per_joint_effort
    drive_type: str = 'acceleration'
    per_joint_effort: bool = False

    def validate(self):
        if self.drive_type not in ('acceleration', 'force'):
            raise ValueError("drive_type must be 'acceleration' or 'force'")
        if not (math.isfinite(self.stiffness) and self.stiffness > 0
                and math.isfinite(self.damping) and self.damping >= 0
                and math.isfinite(self.max_effort) and self.max_effort > 0):
            raise ValueError('invalid drive gains')
        return self

    def per_joint(self):
        """(stiffness, damping, max_effort) 6-vectors in ARM_JOINTS order. For force drive with
        per_joint_effort, the wrists get the UR10 56 Nm cap and proportionally lower stiffness."""
        if self.per_joint_effort:
            scale = [e / max(UR10_EFFORTS) for e in UR10_EFFORTS]  # wrists ~0.17, others 1.0
            return ([self.stiffness * s for s in scale],
                    [self.damping * s for s in scale],
                    list(UR10_EFFORTS))
        return ([self.stiffness] * 6, [self.damping] * 6, [self.max_effort] * 6)


@dataclass(frozen=True)
class CarryProfile:
    """Velocity/acceleration SCALING of a carrying move, as the generated rest-to-rest
    trajectory's per-joint peak speed (rad/s) and peak acceleration (rad/s^2) caps."""
    peak_speed: float
    peak_accel: float

    def validate(self):
        if not (math.isfinite(self.peak_speed) and self.peak_speed > 0
                and math.isfinite(self.peak_accel) and self.peak_accel > 0):
            raise ValueError('invalid carry velocity/acceleration profile')
        return self


@dataclass(frozen=True)
class TrialSpec:
    label: str
    gains: DriveGains
    carry: CarryProfile
    tolerances: Tolerances = field(default_factory=Tolerances)

    def validate(self):
        if not self.label:
            raise ValueError('trial needs a label')
        self.gains.validate(); self.carry.validate(); self.tolerances.validate()
        return self


# ------------------------------------------------------- analytic carry trajectory

class _Point:
    """Duck-typed trajectory_msgs/JointTrajectoryPoint for the real Plan/validate path."""
    __slots__ = ('positions', 'velocities', 'effort', 'time_from_start')

    def __init__(self, positions, velocities, t):
        self.positions = [float(v) for v in positions]
        self.velocities = [float(v) for v in velocities]
        self.effort = []
        sec = int(math.floor(t))
        self.time_from_start = _Duration(sec, int(round((t - sec) * 1e9)))


class _Duration:
    __slots__ = ('sec', 'nanosec')

    def __init__(self, sec, nanosec):
        self.sec, self.nanosec = sec, nanosec


class _Trajectory:
    __slots__ = ('joint_names', 'points')

    def __init__(self, joint_names, points):
        self.joint_names = list(joint_names)
        self.points = points


def move_duration(start, goal, peak_speed, peak_accel):
    """Duration (s) of a rest-to-rest cubic that respects BOTH the per-joint peak-speed and
    peak-acceleration caps for the binding joint. For a cubic q(t) from rest to rest over T
    with travel d: peak speed = 1.5 d / T, peak |accel| = 6 d / T^2. Solve each for T and take
    the larger (plus the other joints' needs), so the generated move never violates either cap."""
    delta = np.abs(np.asarray(goal, float) - np.asarray(start, float))
    d = float(np.max(delta))
    if d <= 1e-9:
        return 0.0
    t_speed = 1.5 * d / peak_speed
    t_accel = math.sqrt(6.0 * d / peak_accel)
    return max(t_speed, t_accel)


def build_carry_trajectory(start, goal, carry: CarryProfile):
    """A single rest-to-rest joint move from ``start`` to ``goal`` (both 6-vectors in
    ARM_JOINTS order), time-parameterised to the carry profile. Returns ``(trajectory, plan)``
    where ``plan`` is the REAL trajectory_host.Plan (so the bench samples/validates exactly as
    the executor does). ``plan`` is None for a zero-travel move."""
    start = [float(v) for v in start]
    goal = [float(v) for v in goal]
    if len(start) != 6 or len(goal) != 6:
        raise ValueError('carry start/goal must be six joints each')
    carry.validate()
    T = move_duration(start, goal, carry.peak_speed, carry.peak_accel)
    traj = _Trajectory(ARM_JOINTS, [
        _Point(start, [0.0] * 6, 0.0),
        _Point(goal, [0.0] * 6, max(T, 1e-3))])
    return traj, T


# --------------------------------------------------------------------- executor

@dataclass
class MoveMetrics:
    """Per carrying-move metrics (brief item 1 + the redirect's contact/clearance diagnosis)."""
    label: str
    outcome: str = 'pending'                 # 'settled' | 'tracking_fault' | 'goal_miss' | 'gate_fault' | 'step_cap'
    reason: str = ''
    duration_s: float = 0.0                  # planned trajectory duration
    settle_time_s: float | None = None       # sim seconds from planned-end to the settle gate passing
    peak_tracking_error: list = field(default_factory=lambda: [0.0] * 6)   # per joint (rad)
    peak_joint_velocity: list = field(default_factory=lambda: [0.0] * 6)   # per joint (rad/s)
    terminal_pos_error: list = field(default_factory=lambda: [0.0] * 6)    # per joint at end (rad)
    terminal_velocity: list = field(default_factory=lambda: [0.0] * 6)     # per joint at end (rad/s)
    peak_carton_drift_m: float = 0.0         # carton drift relative to the tool
    peak_base_displacement_m: float = 0.0    # base translation from move start
    sim_elapsed_s: float = 0.0
    # --- redirect: rigid-contact diagnosis (PhysX contact + geometric clearance of the CARTON) ---
    min_clearance_m: dict = field(default_factory=dict)    # obstacle name -> min AABB gap over the move (m)
    clearance_at_wrist1: dict = field(default_factory=dict)  # obstacle name -> wrist_1 (rad) at that min gap
    contact_prims: list = field(default_factory=list)      # distinct prims the carton physically contacted
    first_contact: dict | None = None        # {sim_time, fold_fraction, wrist1_rad, prim, position} at first touch

    @property
    def ok(self):
        return self.outcome == 'settled'

    def worst_tracking(self):
        j = int(np.argmax(self.peak_tracking_error))
        return j, self.peak_tracking_error[j]


class ArmStage:
    """Abstract interface the executor drives. The Isaac glue implements it against the live
    articulation; CPU tests implement it against a simulated spring-mass arm.

    Contract:
      measured()            -> np.ndarray(6)   arm joint POSITIONS (ARM_JOINTS order)
      measured_velocity()   -> np.ndarray(6)   arm joint VELOCITIES
      command(q, v, stamp)  -> drive the arm toward (q, v); may raise on gate fault
      step(dt)              -> advance physics one tick
      sim_time()            -> float           sim clock seconds
      carton_in_tool()      -> np.ndarray(3)   carton origin in the tool frame (m)
      base_xy()             -> np.ndarray(2)   base world translation (m)
    """

    def measured(self): raise NotImplementedError
    def measured_velocity(self): raise NotImplementedError
    def command(self, q, v, stamp): raise NotImplementedError
    def step(self, dt): raise NotImplementedError
    def sim_time(self): raise NotImplementedError
    def carton_in_tool(self): return np.zeros(3)
    def base_xy(self): return np.zeros(2)
    # --- redirect diagnosis hooks (default empty; the Isaac glue implements them) ---
    def clearance_snapshot(self):
        """{obstacle_name: signed gap (m) between the carton AABB and that obstacle} THIS tick.
        Negative = the carton's bounding volume overlaps the obstacle's."""
        return {}
    def probe_contacts(self):
        """List of {'prim': str, 'position': [x,y,z]} the carton is PHYSICALLY contacting THIS
        tick (PhysX contact reports, hollow-geometry-aware)."""
        return []


def run_move(stage: ArmStage, start, goal, carry: CarryProfile, tol: Tolerances,
             label, *, dt=1.0 / 60.0, max_sim_s=60.0):
    """Execute one carrying move and grade it EXACTLY as trajectory_host does: sample the real
    Plan, command the real gate, apply the real tracking-fault debounce and terminal settle gate.

    Returns MoveMetrics. ``outcome`` is 'settled' on success; a failing move yields a non-'settled'
    outcome (the runner turns any non-'settled' trial into a non-zero exit)."""
    tol.validate()
    traj, duration = build_carry_trajectory(start, goal, carry)
    m = MoveMetrics(label=label, duration_s=duration)
    limits = [(-2 * math.pi, 2 * math.pi)] * 6   # generous; the generated move stays well within
    # Plan guards the generated trajectory; set its speed/accel ceilings just above the
    # generated peaks (the generator already respects the carry caps).
    delta = float(np.max(np.abs(np.asarray(goal, float) - np.asarray(start, float))))
    if duration > 0:
        peak_speed = 1.5 * delta / duration
        peak_accel = 6.0 * delta / (duration * duration)
    else:
        peak_speed = peak_accel = 1e-6
    plan = Plan(traj, np.asarray(start, float), limits,
                max_speed=peak_speed * 1.05 + 1e-6, max_acceleration=peak_accel * 1.05 + 1e-6)

    start_sim = stage.sim_time()
    base0 = np.asarray(stage.base_xy(), float)
    carton0 = np.asarray(stage.carton_in_tool(), float)
    peak_track = np.zeros(6)
    peak_vel = np.zeros(6)
    miss_count = 0
    miss_stamp = None
    steps = 0
    max_steps = int(max_sim_s / dt) + 1
    dur = max(plan.times[-1], 1e-9)
    while True:
        now = stage.sim_time()
        elapsed = max(0.0, now - start_sim)
        measured = np.asarray(stage.measured(), float)
        meas_vel = np.asarray(stage.measured_velocity(), float)
        q, v = plan.sample(elapsed)
        reference, _ = plan.sample(elapsed)   # feedback stamp == now in-process (no relay lag)
        track = np.abs(reference - measured)
        peak_track = np.maximum(peak_track, track)
        peak_vel = np.maximum(peak_vel, np.abs(meas_vel))
        track_err = float(np.max(track))
        # tracking-fault debounce over distinct feedback samples (each tick is a distinct sim stamp)
        if track_err > tol.tracking_tol:
            if now != miss_stamp:
                miss_stamp = now
                miss_count += 1
            if tracking_fault(track_err, miss_count, tol=tol.tracking_tol,
                              debounce=tol.tracking_debounce, hard=tol.tracking_hard_tol):
                m.outcome, m.reason = 'tracking_fault', (
                    f'tracking error {track_err:.3f} exceeds {tol.tracking_tol} rad '
                    f'for {miss_count} samples (worst joint {int(np.argmax(track))})')
                break
        else:
            miss_count, miss_stamp = 0, None
        # drive the real gate exactly as NativeTransport.tick does (stamp == now in-process)
        try:
            stage.command(q, v, now)
        except Exception as exc:
            m.outcome, m.reason = 'gate_fault', f'arm command gate faulted: {exc}'
            break
        # metrics that need the current contact/base state
        m.peak_carton_drift_m = max(m.peak_carton_drift_m,
                                    float(np.linalg.norm(np.asarray(stage.carton_in_tool(), float) - carton0)))
        m.peak_base_displacement_m = max(m.peak_base_displacement_m,
                                         float(np.linalg.norm(np.asarray(stage.base_xy(), float) - base0)))
        stage.step(dt)
        steps += 1
        # --- redirect: carton clearance + physical contact against the scene obstacles ---
        wrist1 = float(measured[WRIST_1])
        fold_fraction = min(1.0, elapsed / dur)
        for name, gap in stage.clearance_snapshot().items():
            if name not in m.min_clearance_m or gap < m.min_clearance_m[name]:
                m.min_clearance_m[name] = float(gap)
                m.clearance_at_wrist1[name] = wrist1
        for hit in stage.probe_contacts():
            prim = hit.get('prim')
            if prim and prim not in m.contact_prims:
                m.contact_prims.append(prim)
            if m.first_contact is None and prim:
                m.first_contact = dict(sim_time=float(stage.sim_time()), fold_fraction=fold_fraction,
                                       wrist1_rad=wrist1, prim=prim, position=hit.get('position'))
        # terminal settle gate (sim-clocked), mirroring trajectory_host.execute()
        if elapsed >= plan.times[-1]:
            pos_err = float(np.max(np.abs(measured - plan.end)))
            vel_mag = float(np.max(np.abs(meas_vel)))
            if pos_err <= tol.goal_position_tol and vel_mag <= tol.goal_velocity_tol:
                m.outcome = 'settled'
                m.settle_time_s = elapsed - plan.times[-1]
                break
            if elapsed > plan.times[-1] + tol.goal_settle_margin_s:
                m.outcome, m.reason = 'goal_miss', (
                    f'goal tolerance not reached: pos_err={pos_err:.4f} vel={vel_mag:.4f}')
                break
        if steps >= max_steps:
            m.outcome, m.reason = 'step_cap', f'move exceeded {max_sim_s}s sim budget'
            break
    final = np.asarray(stage.measured(), float)
    final_v = np.asarray(stage.measured_velocity(), float)
    m.terminal_pos_error = (final - plan.end).tolist()
    m.terminal_velocity = final_v.tolist()
    m.peak_tracking_error = peak_track.tolist()
    m.peak_joint_velocity = peak_vel.tolist()
    m.sim_elapsed_s = max(0.0, stage.sim_time() - start_sim)
    return m


# --------------------------------------------------------------------- sequence

# The staged carrying sequence (controller.py / handoff (b)). Joint goals for joints 1-5 are
# fixed by cart_tuck_shape; joint0 is the measured lift pan for 'tuck', cart_pan_joint0 for 'pan'.
def carry_goals(lift_joints, cart_tuck_shape, cart_pan_joint0, extra_goals=None):
    """Return the ordered list of (phase, goal_joints) the carrying arm folds through.

    tuck: fold joints 1-5 to the carry shape at the MEASURED lift pan.
    pan:  rotate joint0 to the cart side at that tucked radius (the mirrored side pose).
    extra_goals: optional {phase: 6-vector} for pre-place/place/retreat (the downstream lane
                 supplies these once its cuMotion place IK is known); appended in order."""
    lift_joints = [float(v) for v in lift_joints]
    shape = [float(v) for v in cart_tuck_shape]
    if len(lift_joints) != 6 or len(shape) != 5:
        raise ValueError('lift_joints must be 6 and cart_tuck_shape 5 (joints 1-5)')
    goals = [('tuck', [lift_joints[0], *shape]),
             ('pan', [float(cart_pan_joint0), *shape])]
    for phase, goal in (extra_goals or {}).items():
        goal = [float(v) for v in goal]
        if len(goal) != 6:
            raise ValueError(f'extra goal {phase!r} must be six joints')
        goals.append((phase, goal))
    return goals


def tuck_shape_variant(base_shape, deltas):
    """A carry-shape variant: base_shape is joints 1-5 (shoulder_lift, elbow, wrist_1, wrist_2,
    wrist_3); deltas is a 5-vector added to it. Used to sweep carry POSES that raise the carton
    higher / fold it less into the body so it clears the cart cage + rack (the redirect)."""
    base_shape = [float(v) for v in base_shape]
    deltas = [float(v) for v in deltas]
    if len(base_shape) != 5 or len(deltas) != 5:
        raise ValueError('tuck shape and deltas are joints 1-5 (five each)')
    return [a + b for a, b in zip(base_shape, deltas)]


# --------------------------------------------------------------------- sweep grid

def sweep_grid(stiffnesses, dampings, max_efforts, carries, tolerances=None, *,
               drive_types=('acceleration',), per_joint_effort=False, label_prefix=''):
    """Cartesian product of candidate drive types x gains x carry profiles x tolerances -> TrialSpecs.

    Physically grounded per-joint effort limits (UR10 ~330 Nm base/shoulder/elbow, ~56 Nm wrists)
    are ONE candidate family (``per_joint_effort``), never a mandate (brief item 2); drive type
    ('force' vs 'acceleration') is a first-class sweep axis (integration msg 006)."""
    tolerances = tolerances or [Tolerances()]
    specs = []
    for dtype in drive_types:
        for kp in stiffnesses:
            for kd in dampings:
                for eff in max_efforts:
                    for carry in carries:
                        for ti, tol in enumerate(tolerances):
                            label = (f'{label_prefix}{dtype[:3]}_kp{kp:g}_kd{kd:g}_eff{eff:g}'
                                     f'_v{carry.peak_speed:g}')
                            if len(tolerances) > 1:
                                label += f'_tol{ti}'
                            specs.append(TrialSpec(
                                label=label,
                                gains=DriveGains(kp, kd, eff, drive_type=dtype,
                                                 per_joint_effort=per_joint_effort),
                                carry=carry, tolerances=tol).validate())
    return specs


def summarize(trial_results):
    """Rank trials most-robust-first. A robust carry: EVERY move settled AND the carton made NO
    physical contact with the scene. The score prefers larger obstacle CLEARANCE (the redirect's
    primary axis) when clearance was probed, else the headroom to the settle/tracking tolerances.

    ``trial_results`` is a list of dicts {label, spec, moves:[MoveMetrics...]}.
    Returns the list sorted most-robust-first with 'robust' bool, 'score', 'min_clearance_m' and
    'hit' (contacted prims) per trial."""
    ranked = []
    for r in trial_results:
        moves = r['moves']
        all_settled = bool(moves) and all(m.ok for m in moves)
        hit = sorted({p for m in moves for p in m.contact_prims})
        mc = _min_clearance(moves)
        robust = all_settled and not hit
        if not robust:
            score = float('-inf')
        elif mc is not None:
            score = mc[1]                       # larger clearance = more robust
        else:
            tol = r['spec'].tolerances
            track_margin = min(tol.tracking_tol - max(m.peak_tracking_error) for m in moves)
            vel_margin = min(tol.goal_velocity_tol - max(abs(x) for x in m.terminal_velocity)
                             for m in moves)
            score = min(track_margin, vel_margin)
        ranked.append(dict(r, robust=robust, score=score,
                           min_clearance_m=(mc[1] if mc else None), hit=hit))
    ranked.sort(key=lambda r: r['score'], reverse=True)
    return ranked


def _min_clearance(moves):
    """Smallest clearance to any obstacle across a trial's moves, and which obstacle; None if
    no clearance was probed."""
    best = None
    for m in moves:
        for name, gap in m.min_clearance_m.items():
            if best is None or gap < best[1]:
                best = (name, gap)
    return best


def contact_summary(moves):
    """Human-readable 'what the carton hit and where along the fold' for a trial's moves."""
    lines = []
    for m in moves:
        if m.first_contact:
            fc = m.first_contact
            lines.append('%s: FIRST CONTACT prim=%s at fold_fraction=%.2f wrist1=%.3f rad pos=%s'
                         % (m.label, fc['prim'], fc['fold_fraction'], fc['wrist1_rad'], fc['position']))
        if m.contact_prims:
            lines.append('%s: contacted prims=%s' % (m.label, m.contact_prims))
        for name, gap in sorted(m.min_clearance_m.items()):
            lines.append('%s: min clearance to %s = %.1f mm (at wrist1=%.3f rad)'
                         % (m.label, name, gap * 1000, m.clearance_at_wrist1.get(name, float('nan'))))
    return '\n'.join(lines) if lines else '(no contact/clearance probed)'


def metrics_table(trial_results):
    """A compact text metrics table (report evidence). One row per trial. Includes the
    rigid-contact diagnosis columns (min clearance + whether the carton physically contacted)."""
    rows = [('trial', 'outcome', 'worst_track', 'wrist1_track', 'term_vel_max',
             'settle_s', 'carton_mm', 'base_mm', 'min_clear_mm', 'hit')]
    for r in trial_results:
        moves = r['moves']
        outcome = 'PASS' if (moves and all(m.ok for m in moves)) else 'FAIL'
        worst_track = max((max(m.peak_tracking_error) for m in moves), default=0.0)
        wrist1 = max((m.peak_tracking_error[WRIST_1] for m in moves), default=0.0)
        term_vel = max((max(abs(x) for x in m.terminal_velocity) for m in moves), default=0.0)
        settle = max((m.settle_time_s or 0.0 for m in moves), default=0.0)
        carton = max((m.peak_carton_drift_m for m in moves), default=0.0) * 1000
        base = max((m.peak_base_displacement_m for m in moves), default=0.0) * 1000
        mc = _min_clearance(moves)
        clear = f'{mc[1] * 1000:.1f}' if mc else '-'
        hit = ';'.join(sorted({p.rsplit('/', 1)[-1] for m in moves for p in m.contact_prims})) or '-'
        rows.append((r['label'], outcome, f'{worst_track:.3f}', f'{wrist1:.3f}',
                     f'{term_vel:.3f}', f'{settle:.2f}', f'{carton:.1f}', f'{base:.1f}', clear, hit))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return '\n'.join('  '.join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)
