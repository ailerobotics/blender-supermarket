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
                           crr_robot=0.03, crr_cart=0.04, grade_rad=0.0, **share_kw):
    """Quasi-static coupled-tow force budget (lever DIRECTION tool, NOT an efficacy proof).

    Available tractive force is the lesser of the friction limit (mu * drive-wheel normal force) and the
    drive torque cap (2 * maxForce / wheel_radius). Demand is accelerate-the-train + rolling resistance of
    both bodies + any grade. ``margin = available / demand``. Monotone increasing in wheel_friction,
    caster_lift_m and drive_max_force_nm -- the three physical levers -- so a recommended config strictly
    dominates the stuck config. The absolute margin OVER-estimates robustness (it is steady-state; the real
    failure is dynamic re-breakaway from rest), so only the ordering is asserted; the GPU bench is truth."""
    if not (wheel_friction > 0 and drive_max_force_nm > 0 and wheel_radius_m > 0
            and robot_mass_kg > 0 and cart_mass_kg > 0 and target_accel_mps2 >= 0
            and crr_robot >= 0 and crr_cart >= 0):
        raise ValueError('invalid budget inputs')
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
                f_friction_n=round(f_friction, 1), f_torque_cap_n=round(f_torque_cap, 1),
                f_available_n=round(f_avail, 1), f_demand_n=round(f_demand, 1),
                margin=round(f_avail / f_demand, 3) if f_demand > 0 else float('inf'),
                binding_cap=binding)
