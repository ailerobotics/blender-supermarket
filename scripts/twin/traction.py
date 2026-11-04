"""Coupled-tow traction: the loaded-tow margin grader + a quasi-static drive-authority budget.

Pure, Isaac-free core for the sim-robust-coupled-traction investigation. Two independent pieces:

1. ``coupled_tow_margin`` -- the ACCEPTANCE grader. Given a run's recorded outbound nav-trace rows and
   the lane goal, it decides whether the coupled tow passed the way warm8/warm11 did (reached the goal,
   never pinned for more than a few sim seconds) or stalled the way outbound1/warm9/warm10 did (pinned
   at x~=-3.93 and faulted ``coupled_no_progress``). This is the deliverable's pass/fail gate: the
   integration lane's bounded GPU proof trace is fed to the SAME grader (see
   ``scripts/check_tow_traction.py``). It mirrors the PR-201 net-progress monitor's own window math
   (anchor on net displacement >= min_m; otherwise accumulate SIM time since the anchor) so a run the
   grader passes is a run the live monitor would not fault.

2. ``static_traction_budget`` -- a quasi-static force budget used ONLY to justify the DIRECTION of the
   physical levers (caster lift, wheel friction, drive force), never as an efficacy proof. The observed
   stall is a dynamic re-breakaway-FROM-REST phenomenon (warm8 crossed the same clear geometry carrying
   entry momentum; outbound1/warm9/warm10 stalled from a near-stop) that a steady budget over-estimates --
   the same reason the base-drive notes repeat "efficacy is a GPU-measured sweep, not offline-provable".
   So this budget is monotone in the levers and is asserted only for monotonicity and strict dominance
   of the recommended config over the stuck one; the real verdict is the GPU bench.

Coordinates/units: SI (metres, seconds, radians, newtons, kg). Sim time comes from the trace's
``world_t`` (the node/sim clock), never wall, so render pacing cannot distort the window -- the same clock
discipline the restock freshness/progress monitors use.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


# ---- (1) recorded-trace loaded-tow margin (the acceptance gate) ---------------------------------

def _sim_now(row):
    """The SIM clock for a trace row: the node/world time, falling back to the controller-elapsed ``t``
    only if a trace predates ``world_t``. Never wall (``wall``), which render pacing distorts."""
    v = row.get('world_t', row.get('t'))
    return float(v)


@dataclass(frozen=True)
class TowMargin:
    reached_goal: bool          # the base came within goal_tol_m of the lane goal xy
    longest_pin_s: float        # longest contiguous SIM-time window with < min_progress_m net advance
    max_advance_m: float        # furthest net +x advance from the first coupled row
    drive_cruise_mps: float     # median achieved forward speed while coupled and actively driving
    coupled_ticks: int
    passed: bool                # reached_goal AND longest_pin_s < pin_fault_s
    reason: str


def coupled_tow_margin(rows, goal, *, goal_tol_m=0.12, min_progress_m=0.05,
                       pin_fault_s=5.0, drive_speed_floor_mps=0.03):
    """Grade a coupled outbound tow from its recorded nav-trace rows.

    ``rows``: the outbound rows (dicts with x, y, coupled, parked, lin, world_t/t) in time order.
    ``goal``: the lane goal (x, y[, yaw]); only xy is used.

    A tow PASSES iff it reached within ``goal_tol_m`` of the goal AND never pinned (net advance
    < ``min_progress_m``) for ``pin_fault_s`` contiguous SIM seconds while coupled. The pin window is
    the PR-201 monitor's own math: anchor on >= ``min_progress_m`` net displacement, otherwise accrue
    SIM time since the anchor; a gap in coupling re-anchors (a legitimate wait is never charged). Returns
    a :class:`TowMargin`. Empty/degenerate input is a FAIL (never a silent pass)."""
    coupled = [r for r in rows if r.get('coupled')]
    if not coupled:
        return TowMargin(False, 0.0, 0.0, 0.0, 0, False, 'no_coupled_rows')
    if not (goal_tol_m > 0 and min_progress_m > 0 and pin_fault_s > 0):
        raise ValueError('goal_tol_m, min_progress_m and pin_fault_s must be positive')
    gx, gy = float(goal[0]), float(goal[1])
    x0, y0 = float(coupled[0]['x']), float(coupled[0]['y'])
    reached = any(math.hypot(float(r['x']) - gx, float(r['y']) - gy) <= goal_tol_m for r in coupled)
    max_advance = max(float(r['x']) - x0 for r in coupled)   # the lane runs +x; net forward advance
    anchor = None
    longest = 0.0
    prev_now = None
    for r in rows:
        if not r.get('coupled'):
            anchor = None                                    # coupling gap re-anchors
            continue
        now = _sim_now(r)
        gap = anchor is None or prev_now is None or (now - prev_now) > 1.0
        if gap or math.hypot(float(r['x']) - anchor[1], float(r['y']) - anchor[2]) >= min_progress_m:
            anchor = (now, float(r['x']), float(r['y']))
        else:
            longest = max(longest, now - anchor[0])
        prev_now = now
    driving = [float(r['lin']) for r in coupled
               if not r.get('parked') and float(r.get('lin', 0.0)) >= drive_speed_floor_mps]
    cruise = _median(driving) if driving else 0.0
    passed = bool(reached and longest < pin_fault_s)
    reason = ('passed' if passed
              else 'never_reached_goal' if not reached
              else 'pinned_%.1fs' % longest)
    return TowMargin(reached, round(longest, 2), round(max_advance, 3),
                     round(cruise, 4), len(coupled), passed, reason)


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def throat_single_tick_stops(rows, *, before_x=-3.90, drop_from_mps=0.12, to_mps=0.03):
    """Count the coupled-tow's single-tick velocity COLLAPSES WEST of ``before_x`` (the throat momentum
    signature). sim-robust-throat-momentum-launch-v1: in every pinning run the base went from ~0.18 m/s to
    ~0.006 m/s in ONE trace tick at x=-3.935 (a ~10 m/s^2 impulse) then never advanced; the passers rolled
    through the same spot at cruise with no such collapse. A momentum-launch run that threaded the pin has
    ZERO such collapses before ``before_x``. Keyed on the recorded robot linear speed (``lin``) between
    adjacent coupled rows; returns (count, first_pose) where first_pose is (x, y) of the first collapse or
    None. A collapse is a drop from >= ``drop_from_mps`` to <= ``to_mps`` across one tick while still west of
    ``before_x`` (the base has not yet carried momentum past the pin)."""
    coupled = [r for r in rows if r.get('coupled')]
    count = 0
    first = None
    for a, b in zip(coupled, coupled[1:]):
        va = float(a.get('lin', 0.0))
        vb = float(b.get('lin', 0.0))
        if float(b['x']) <= before_x and va >= drop_from_mps and vb <= to_mps:
            count += 1
            if first is None:
                first = (round(float(b['x']), 3), round(float(b['y']), 3))
    return count, first


# ---- (1b) chassis pitch see-saw + compliant caster suspension (sim-chassis-pitch-stability-v1) ---
#
# The named cause of the coldwin33-37 collect set-down CARTON_TILT +2.06 deg trips: the two-drive-wheel
# base (wheels on the lateral axle at x=0) with BOTH passive casters recessed ``caster_lift_m`` ABOVE the
# drive-wheel ground plane balances on the drive axle with ZERO pitch stiffness. It free-pitches until a
# caster touches down -- +-atan(caster_lift_m / caster_spacing) -- and the arm posture (the UR10's own
# mass out at reach, NOT the 0.45 kg carton) decides which end stop it sits on. The welded carton is rigid
# to the arm and the arm rigid to the chassis, so after a LEVEL descent was planned the carton pitches by
# exactly the chassis rock-back: that rock-back IS the 2.06. BASE_ATT (isaac_port, ~1 Hz) measured the two
# end stops at -4.1 / +4.9 deg in coldwin37 (predicted +-4.52) and the set-down rock-back 4.8 -> 2.3 deg.
#
# The fix is pitch stiffness. A compliant caster SUSPENSION keeps both casters in LIGHT ground contact
# through a sprung prismatic caster-fork joint (URDF joint + PhysX linear drive stiffness/damping/preload
# authored like the wheel drives). The functions below are the pure analytical design: the free-pitch
# geometry, the sprung chassis pitch under an arm moment (a BILINEAR spring model -- both casters loaded,
# then one lifting off at the preload limit), and the preload-based drive-wheel share so the traction
# budget can prove the suspension strictly dominates the stuck pre-PR-207 rig. No simulator; GPU is truth.

# Caster fore/aft offset from the drive axle (base x=0): mobile_assets authors the casters at
# ``+-(length/2 - 0.12)`` m. For the default 1.0 m base this is 0.38 m.
def caster_spacing_m(length_m=1.0, caster_inset_m=0.12):
    """Fore/aft distance (m) of each passive caster contact from the drive axle, matching
    ``mobile_assets.build_mobile_urdf`` (``caster x = length/2 - 0.12``)."""
    d = length_m / 2.0 - caster_inset_m
    if not (d > 0 and math.isfinite(d)):
        raise ValueError('caster spacing must be a positive finite distance')
    return d


def free_pitch_travel_deg(caster_lift_m, caster_spacing):
    """The UNSPRUNG chassis free-pitch travel (degrees, one-sided) before a recessed caster touches
    down: ``atan(caster_lift_m / caster_spacing)``. 0 when the casters are co-planar (caster_lift_m=0).
    This is the see-saw amplitude; a rig whose free travel exceeds the carton bound can rock the welded
    carton off level with no arm fault (coldwin33-37)."""
    if not (caster_spacing > 0 and math.isfinite(caster_spacing)):
        raise ValueError('caster_spacing must be a positive finite distance')
    if not (math.isfinite(caster_lift_m) and caster_lift_m >= 0):
        raise ValueError('caster_lift_m must be a finite non-negative distance')
    return math.degrees(math.atan2(caster_lift_m, caster_spacing))


@dataclass(frozen=True)
class CasterSuspension:
    """A compliant caster suspension design: a sprung prismatic caster-fork joint per caster. The
    spring (``spring_n_per_m`` k) resists chassis pitch; the ``preload_n`` F0 holds both casters in
    light ground contact at rest so the drive wheels keep most of the weight (traction). ``travel_m``
    bounds the prismatic stroke; ``damping_n_s_per_m`` is the drive damping (ring suppression)."""
    spring_n_per_m: float
    preload_n: float
    damping_n_s_per_m: float = 2000.0
    travel_m: float = 0.02

    def __post_init__(self):
        if not (self.spring_n_per_m > 0 and math.isfinite(self.spring_n_per_m)):
            raise ValueError('suspension spring_n_per_m must be positive')
        if not (self.preload_n >= 0 and math.isfinite(self.preload_n)):
            raise ValueError('suspension preload_n must be finite non-negative')
        if not (self.damping_n_s_per_m >= 0 and math.isfinite(self.damping_n_s_per_m)):
            raise ValueError('suspension damping_n_s_per_m must be finite non-negative')
        if not (self.travel_m > 0 and math.isfinite(self.travel_m)):
            raise ValueError('suspension travel_m must be positive')


# sim-chassis-pitch-stability-v1 (coldwin38 FAIL): the sprung caster mass below the spring, used for the
# numerical-stability ratio. The suspender link + its fork + wheel all translate with the prismatic DOF.
CASTER_FORK_WHEEL_MASS_KG = 0.8   # fork 0.3 + wheel 0.5 (mobile_assets.build_mobile_urdf)
# The PhysX solver tolerates a joint-drive stiffness-to-sprung-mass-per-step ratio only up to ~O(40) on
# this rig: the arm's own stiff position drives run at k*dt^2/m ~= 1e6*(1/60)^2/7 ~= 40 STABLY on the same
# 1/60 s solver, while coldwin38's sprung caster at k*dt^2/m ~= 3e5*(1/60)^2/1.0 ~= 83 EXPLODED at reset
# (BASE_ATT -7.4..+12.2 deg bounce before any arm motion). The ceiling is set conservatively BELOW the
# observed-unstable 83, with margin under the arm's stable ~40. A raised physics_dt relaxes it (the ratio
# scales dt^2); the mission default dt is 1/60.
SUSPENSION_STABILITY_RATIO_MAX = 30.0
MISSION_PHYSICS_DT_S = 1.0 / 60.0


def suspension_sprung_mass_kg(suspender_mass_kg):
    """Effective sprung mass on the prismatic caster-lift DOF: the suspender link + the fork + wheel that
    translate with it."""
    if not (suspender_mass_kg > 0 and math.isfinite(suspender_mass_kg)):
        raise ValueError('caster suspender mass must be positive')
    return suspender_mass_kg + CASTER_FORK_WHEEL_MASS_KG


def suspension_solver_stability_ratio(spring_n_per_m, suspender_mass_kg, dt_s=MISSION_PHYSICS_DT_S):
    """``k * dt^2 / m_sprung`` -- the explicit-ish joint-drive stability number at the solver step. Above
    ~``SUSPENSION_STABILITY_RATIO_MAX`` the sprung caster drive is numerically unstable at reset (the
    coldwin38 bounce). Calibrated to GPU: the arm's stiff drives run ~40 stable, the shipped caster ran
    ~83 unstable."""
    if not (spring_n_per_m > 0 and dt_s > 0 and math.isfinite(spring_n_per_m) and math.isfinite(dt_s)):
        raise ValueError('spring and dt must be positive finite')
    return spring_n_per_m * dt_s * dt_s / suspension_sprung_mass_kg(suspender_mass_kg)


def suspension_solver_stable(spring_n_per_m, suspender_mass_kg, dt_s=MISSION_PHYSICS_DT_S,
                             ratio_max=SUSPENSION_STABILITY_RATIO_MAX):
    """True iff the sprung caster drive is within the solver-stable ratio band at ``dt_s``."""
    return suspension_solver_stability_ratio(spring_n_per_m, suspender_mass_kg, dt_s) <= ratio_max


def min_suspender_mass_kg(spring_n_per_m, dt_s=MISSION_PHYSICS_DT_S,
                          ratio_max=SUSPENSION_STABILITY_RATIO_MAX):
    """The least suspender-link mass (kg) that keeps ``spring_n_per_m`` solver-stable at ``dt_s`` (so a
    spec can be auto-sized rather than guessed)."""
    return max(0.0, spring_n_per_m * dt_s * dt_s / ratio_max - CASTER_FORK_WHEEL_MASS_KG)


def sprung_chassis_pitch_deg(moment_nm, suspension: CasterSuspension, caster_spacing):
    """Quasi-static chassis pitch (degrees) under an arm pitch ``moment_nm`` about the drive axle for a
    compliant caster suspension, from a BILINEAR spring model (two casters at +-``caster_spacing`` from
    the axle, each a linear spring ``k`` preloaded ``F0``):

      * |M| <= M_liftoff (= 2*F0*d): BOTH casters resist, rotational stiffness K = 2*k*d^2.
      * |M| >  M_liftoff: the unloading caster lifts off at F0; only the compressing caster resists,
        incremental stiffness K = k*d^2 (half). The nearer caster carries the excess.

    theta = M/(2kd^2) up to the knee, then + (|M|-M_liftoff)/(k d^2). Signed like ``moment_nm``. This is
    the analytical pitch the suspension holds the chassis to; the GPU bench is the efficacy truth."""
    k, d = suspension.spring_n_per_m, caster_spacing
    if not (d > 0 and math.isfinite(d)):
        raise ValueError('caster_spacing must be a positive finite distance')
    if not math.isfinite(moment_nm):
        raise ValueError('moment_nm must be finite')
    both = 2.0 * k * d * d
    single = k * d * d
    m = abs(moment_nm)
    m_liftoff = 2.0 * suspension.preload_n * d
    if m <= m_liftoff:
        theta = m / both
    else:
        theta = m_liftoff / both + (m - m_liftoff) / single
    return math.copysign(math.degrees(theta), moment_nm)


def suspension_meets_pitch_bound(moment_nm, suspension: CasterSuspension, caster_spacing,
                                 bound_deg=0.5):
    """True iff the sprung chassis pitch at the worst-case ``moment_nm`` stays within ``bound_deg``
    (the design criterion (a): <= 0.5 deg under the UR10's own mass at full reach + the carton)."""
    return abs(sprung_chassis_pitch_deg(moment_nm, suspension, caster_spacing)) <= bound_deg


def suspension_drive_wheel_share(preload_n, robot_mass_kg=None):
    """Fraction of the ROBOT weight borne by the two drive wheels when the casters are held in light
    contact by a per-caster ``preload_n``: ``1 - 2*F0 / (m g)`` (the casters carry ~2*F0, the drive
    wheels the rest). Monotone DECREASING in the preload -- a lighter preload keeps more weight (hence
    traction) on the drive wheels, which is why the design uses a STIFF spring + LIGHT preload rather
    than a soft spring + heavy preload. Clamped to [0, 1]."""
    m = DEFAULT_ROBOT_MASS_KG if robot_mass_kg is None else robot_mass_kg
    if not (m > 0 and math.isfinite(m)):
        raise ValueError('robot_mass_kg must be positive')
    if not (preload_n >= 0 and math.isfinite(preload_n)):
        raise ValueError('preload_n must be finite non-negative')
    return max(0.0, min(1.0, 1.0 - 2.0 * preload_n / (m * G)))


# ---- (2) quasi-static drive-authority budget (lever DIRECTION only) -----------------------------

# Robot-side body masses carried by the two drive wheels (the composed mission articulation; the cart is
# a separate trailer on its own casters and does NOT load the drive wheels). Sum of the Isaac per-body
# masses physics_setup authors (base_link 180 kg + arm column + UR10 + gripper/payload) ~= 250 kg.
DEFAULT_ROBOT_MASS_KG = 250.0
DEFAULT_CART_MASS_KG = 35.0
G = 9.81


def drive_wheel_normal_share(caster_lift_m, *, coplanar_share=0.5, unloaded_share=0.95,
                             lift_unload_m=0.02):
    """Fraction of the ROBOT weight borne by the two drive wheels, as a function of the caster recess.

    Co-planar (lift ~ 0) the diamond of two drive wheels (on the lateral axle line) and two fore/aft
    casters shares the weight roughly evenly, so the drive wheels carry ~``coplanar_share`` -- the
    "starved drive-wheel normal force" the uncoupled base-drive bench measured (tracking 0.097). Recessing
    the casters transfers their share onto the drive wheels; past ``lift_unload_m`` the casters are fully
    unloaded and the drive wheels carry ``unloaded_share`` (the residual going to whichever caster the COM
    pitches onto). Linear, monotone non-decreasing, clamped to [coplanar_share, unloaded_share]. This is a
    quasi-static contact model, not a PhysX result; it exists to show the lever's DIRECTION and sign."""
    if not (0.0 <= coplanar_share <= unloaded_share <= 1.0 and lift_unload_m > 0):
        raise ValueError('invalid share model parameters')
    if not (isinstance(caster_lift_m, (int, float)) and math.isfinite(caster_lift_m) and caster_lift_m >= 0):
        raise ValueError('caster_lift_m must be a finite non-negative distance')
    frac = min(1.0, caster_lift_m / lift_unload_m)
    return coplanar_share + (unloaded_share - coplanar_share) * frac


def static_traction_budget(*, wheel_friction, caster_lift_m, drive_max_force_nm=80.0,
                           wheel_radius_m=0.16, robot_mass_kg=DEFAULT_ROBOT_MASS_KG,
                           cart_mass_kg=DEFAULT_CART_MASS_KG, target_accel_mps2=0.2,
                           crr_robot=0.03, crr_cart=0.04, grade_rad=0.0,
                           suspension_preload_n=None, **share_kw):
    """Quasi-static coupled-tow force budget (lever DIRECTION tool, NOT an efficacy proof).

    Available tractive force is the lesser of the friction limit (mu * drive-wheel normal force) and the
    drive torque cap (2 * maxForce / wheel_radius). Demand is accelerate-the-train + rolling resistance of
    both bodies + any grade. ``margin = available / demand``. Monotone increasing in wheel_friction,
    caster_lift_m and drive_max_force_nm -- the three physical levers -- so a recommended config strictly
    dominates the stuck config. The absolute margin OVER-estimates robustness (it is steady-state; the real
    failure is dynamic re-breakaway from rest), so only the ordering is asserted; the GPU bench is truth.

    ``suspension_preload_n`` (sim-chassis-pitch-stability-v1): when given, the compliant caster suspension
    is active, so the drive-wheel share comes from the per-caster PRELOAD (``suspension_drive_wheel_share``)
    rather than the caster recess -- the casters are now in light CONTACT, not airborne. A light preload
    keeps most of the weight (hence traction) on the drive wheels; the budget is monotone DECREASING in the
    preload, so a light-preload suspension strictly dominates the stuck coplanar rig and sits just below the
    caster_lift 0.03 config. ``caster_lift_m`` is ignored for the share when a preload is supplied (the two
    mechanisms are mutually exclusive -- a sprung caster is not also recessed)."""
    if not (wheel_friction > 0 and drive_max_force_nm > 0 and wheel_radius_m > 0
            and robot_mass_kg > 0 and cart_mass_kg > 0 and target_accel_mps2 >= 0
            and crr_robot >= 0 and crr_cart >= 0):
        raise ValueError('invalid budget inputs')
    if suspension_preload_n is not None:
        share = suspension_drive_wheel_share(suspension_preload_n, robot_mass_kg)
    else:
        share = drive_wheel_normal_share(caster_lift_m, **share_kw)
    n_drive = robot_mass_kg * G * share
    f_friction = wheel_friction * n_drive
    f_torque_cap = 2.0 * drive_max_force_nm / wheel_radius_m
    f_avail = min(f_friction, f_torque_cap)
    total_mass = robot_mass_kg + cart_mass_kg
    f_demand = (total_mass * target_accel_mps2
                + crr_robot * robot_mass_kg * G
                + crr_cart * cart_mass_kg * G
                + total_mass * G * math.sin(grade_rad))
    binding = 'friction' if f_friction <= f_torque_cap else 'drive_torque'
    return dict(drive_wheel_share=round(share, 4), n_drive_n=round(n_drive, 1),
                suspension_preload_n=(None if suspension_preload_n is None
                                      else round(float(suspension_preload_n), 1)),
                f_friction_n=round(f_friction, 1), f_torque_cap_n=round(f_torque_cap, 1),
                f_available_n=round(f_avail, 1), f_demand_n=round(f_demand, 1),
                margin=round(f_avail / f_demand, 3) if f_demand > 0 else float('inf'),
                binding_cap=binding)
