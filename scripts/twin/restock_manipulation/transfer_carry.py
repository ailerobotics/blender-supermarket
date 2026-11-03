"""Opt-in, SIM-ONLY transfer CARRY switches (default off => every caller byte-identical).

ONE env var, ``RESTOCK_TRANSFER_CARRY`` (a comma-separated flag list), read by BOTH mission processes -- the
host ``TransferController`` (which moves the arm) and the Isaac side (``stamp_summary`` -> mission.json +
the burned-in caption) -- so it cannot be split across the host/runtime configs (the run-277 trap), exactly
like ``RESTOCK_TRANSFER_MODE``. An unknown flag RAISES at boot instead of silently running the default.

``wrist_guard`` -- the warm2 transfer-carry blocker (2026-10-22). The real-arm transfer grasped and lifted
  the carton (held by the SIM_ATTACH_FALLBACK weld the whole time -- it was never dropped), tucked, then the
  PREPLACE joint-space move jammed: the seeded-KDL IK picked the self-colliding wrist branch (wrist_1
  0.65 -> 2.08, wrist_2 0.80 -> -1.02), so the arm pinned at the ``ur_wrist_2``-vs-``ur_forearm`` stop
  (wrist_1 0.678) while the reference ran on; the trajectory host faulted at 1.304 > 1.3 rad tracking error
  and published the global stop ('host emergency stop'). That arm-vs-arm pair is ignored by cuMotion's
  sphere model and MoveIt's SRDF, so neither the IK nor the planner can see it (the run-278 cage-preplace
  and run-300d/305/311 loaded-lift jams, fixed with fixed clear joint shapes). With the flag on, every
  TRANSFER joint-space KDL move (preplace, place, retreat) must land in the bench-proven CLEAR wrist region
  -- wrist_1 <= ``carry_wrist1_cap`` and wrist_2 >= ``carry_wrist2_min`` (angles wrapped to (-pi, pi]) --
  and an IK solution outside it is re-seeded from its WRIST-FLIPPED twin (wrist_1 - pi, -wrist_2,
  wrist_3 + pi), which reaches the SAME tool pose in the clear branch (CPU: the warm2 preplace pose solves
  at wrist_1 -0.50, wrist_2 +1.02). No clear solution => the move fails honestly ('IK outside the carry
  wrist region') instead of driving the arm into the stop. Collect is never affected.

``weld`` -- name (and guarantee) what holds the carton during the transfer carry. warm2's carton was held by
  the labelled SIM_ATTACH_FALLBACK fixed-joint weld (runtime config ``sim_attach_fallback``), the same
  captain-approved pattern collect uses (answer #1): it forms at the <=0.5 mm flush grasp contact with
  suction on, holds through status flickers, and releases when the gripper opens at the place pose; the
  native suction never sealed (gripped_n=0 throughout). The flag records ``transfer_attach: weld`` in
  mission.json + the caption, and turns the weld path on if the runtime config left it off (logged), so a
  video can never present the weld as suction physics.
"""
from __future__ import annotations

import math
import os

ENV = 'RESTOCK_TRANSFER_CARRY'
WRIST_GUARD = 'wrist_guard'
WELD = 'weld'
FLAGS = (WRIST_GUARD, WELD)
DESCRIPTIONS = {
    WRIST_GUARD: ('transfer place moves (preplace/place/retreat) keep the arm IK in the clear wrist branch '
                  '(wrist_1 <= carry cap, wrist_2 >= carry min; wrist-flipped re-seed) so the carry never '
                  'drives into the ur_wrist_2-vs-forearm self-collision stop'),
    WELD: ('the transfer carton is held by the SIM fixed-joint weld (SIM_ATTACH_FALLBACK, the captain-approved '
           'collect pattern) from the grasp contact to the release at the place pose, not native suction physics'),
}
CAPTIONS = {
    WRIST_GUARD: 'TRANSFER CARRY: wrist-guarded IK (clear wrist branch)',
    WELD: 'TRANSFER CARRY: weld attach (sim fixed joint), not suction',
}
# The bench-proven clear carry region (fm403, data/sim-restock-armbench-v1): wrist_1 <= ~0.65 AND wrist_2
# positive. Used only when the host config leaves carry_wrist1_cap / carry_wrist2_min unset.
DEFAULT_WRIST1_CAP = 0.65
DEFAULT_WRIST2_MIN = 0.0
# The transfer moves the guard covers: every seeded-KDL joint-space move of the place half. The pick half
# (approach/contact task-space, lift/tuck fixed joint shapes) is untouched.
GUARDED_PHASES = ('preplace', 'place', 'retreat')
_OFF = ('', '0', 'off', 'none', 'false')


def carry_flags(environ=None):
    """The active carry flags as a frozenset (unset/empty/'0'/'off' => empty => off). An unknown flag
    RAISES so a typo fails loudly at boot instead of silently running the unguarded transfer."""
    raw = str((os.environ if environ is None else environ).get(ENV, '')).strip().lower()
    if raw in _OFF:
        return frozenset()
    flags = frozenset(f.strip() for f in raw.split(',') if f.strip())
    unknown = sorted(flags - set(FLAGS))
    if unknown:
        raise ValueError(f'{ENV}={raw!r}: unknown transfer carry flag(s) {unknown} (known: {", ".join(FLAGS)})')
    return flags


def weld_attach(config, environ=None):
    """The ``sim_attach_fallback`` value for the mission's gripper contact: the runtime config's flag, or True
    when RESTOCK_TRANSFER_CARRY carries ``weld`` (logged when that turns an unset/false config on). One weld
    path for both stages -- the same NativePanelContact holds collect and transfer -- exactly as the config
    flag already does."""
    configured = bool((config or {}).get('sim_attach_fallback', False))
    try:
        weld = WELD in carry_flags(environ)
    except ValueError:
        weld = False          # an invalid flag list fails loudly at the controller / summary, not here
    if weld and not configured:
        print('TRANSFER_CARRY weld: enabling the SIM_ATTACH_FALLBACK weld (runtime config sim_attach_fallback '
              'was off)', flush=True)
    return configured or weld


def wrap_angle(value):
    """``value`` wrapped to (-pi, pi]: the physical joint angle, whatever 2*pi branch the IK unwrapped to."""
    wrapped = math.atan2(math.sin(value), math.cos(value))
    return math.pi if wrapped == -math.pi else wrapped


def in_clear_region(q, wrist1_cap, wrist2_min):
    """True when the arm config ``q`` (ARM order: pan, lift, elbow, wrist_1, wrist_2, wrist_3) has its
    PHYSICAL wrist_1 <= wrist1_cap and wrist_2 >= wrist2_min -- the clear carry region, below the
    ur_wrist_2-vs-ur_forearm stop. Wrapped, so a far 2*pi branch of a colliding angle is not 'clear'."""
    return wrap_angle(q[3]) <= wrist1_cap and wrap_angle(q[4]) >= wrist2_min


def clear_branch(q, wrist1_cap, lower, upper, *, tol=1e-9):
    """Re-express the arm config ``q`` so the COMMANDED ur_wrist_1 (index 3) lies in the clear band
    (raw value <= ``wrist1_cap``) by shifting it a whole number of turns -- an exact +/-2*pi shift of a
    single revolute joint is the IDENTICAL tool pose, so the TCP is unchanged. Returns a new list (or the
    same values) with wrist_1 set to the largest in-limit turn <= the cap, or ``None`` when no in-limit turn
    satisfies the cap (the pose has no clear-branch COMMAND).

    This is the fix for the wrap blind spot in ``in_clear_region``: that predicate wraps wrist_1 to
    (-pi, pi], so an IK solution whose RAW wrist_1 is e.g. +4.12 (wraps to -2.16 <= cap) reads as "clear"
    even though a joint starting below the ~0.69 ur_wrist_1-vs-ur_forearm stop has to sweep UP through the
    stop to reach +4.12. Commanding the equivalent branch (-2.16) reaches the same clear orientation without
    crossing the stop. wrist_2/wrist_3 are left untouched -- their branch is chosen by the selection guard
    (``in_clear_region``), and shifting wrist_2 by 2*pi would not change its colliding orientation."""
    q = [float(v) for v in q]
    w1 = q[3]
    lo, hi = float(lower[3]), float(upper[3])
    cands = [w1 + k * 2.0 * math.pi for k in range(-3, 4)]
    cands = [c for c in cands if lo - tol <= c <= hi + tol and c <= wrist1_cap + tol]
    if not cands:
        return None
    q[3] = max(cands)
    return q


def wrist_flip_seed(q):
    """The wrist-flipped twin of ``q`` as an IK SEED: wrist_1 - pi, -wrist_2, wrist_3 + pi (shoulder/elbow
    unchanged). For a UR it is exact only when the wrist_2 offset is zero; with the real 0.116 m offset the
    flip moves the wrist centre, so the IK converges from this seed to the nearby exact clear solution
    (CPU on the mission URDF: shoulder_lift/elbow shift ~0.2-0.4 rad, tool pose identical)."""
    q = [float(v) for v in q]
    return (q[0], q[1], q[2], wrap_angle(q[3] - math.pi), -q[4], wrap_angle(q[5] + math.pi))


# sim-robust-transfer-preplace-reach-v1: the live_pose TRANSFER preplace standoff has (at least) two
# wrist-CLEAR IK branches -- a HOLDABLE UPRIGHT one (ur_shoulder_lift ~ -1.9, the arm barely folded) and a
# DEEP fold (ur_shoulder_lift ~ -4.0) the arm physically CANNOT reach (coldwin6: the arm froze 0.35 rad
# short at -3.65 and faulted PATH_TOLERANCE_VIOLATED). The wrist-flip re-seed that clears the wrist
# (wrist_flip_seed) is INEXACT (0.116 m offset), so from it KDL can re-converge into the deep-fold branch;
# in_clear_region then ACCEPTS a clear-but-infeasible goal (coldwin6 accepted ur_shoulder_lift -4.003).
# fold_droop / reachable_fold let the guard ALSO require the holdable fold, so a deep-fold clear solution is
# rejected and a holdable (upright-seeded) branch is chosen instead. Same droop metric the collect-place
# lane uses (sim-robust-cart-place-goal-v1, tests/test_restock_cart_place_goal._droop).
DEFAULT_MAX_FOLD_DROOP = 2.0   # upright branch droop ~0.38 rad, deep fold ~3.85 rad -> 2.0 separates them
# A generic holdable UPRIGHT clear-branch IK seed (ARM order). CPU-verified on the coldwin6 / run-356
# mobile-import.urdf: the seeded-KDL IK converges from here to the upright clear branch (ur_shoulder_lift
# ~ -1.95, in_clear_region True) for the transfer preplace standoff, regardless of its shoulder_pan.
DEFAULT_UPRIGHT_SEED = (0.0, -1.9, 2.0, -0.1, 1.6, 0.0)


def fold_droop(q):
    """The arm's shoulder-fold 'droop' proxy: ``|wrap(ur_shoulder_lift) - (-pi/2)|`` (rad). -pi/2 is the
    upper arm horizontal/upright; the HOLDABLE upright branch sits ~0.38 rad from it, the un-holdable DEEP
    fold ~3.85 rad (near fully folded). Wrapped so a far 2*pi branch of the same physical angle reads the
    same. ``q`` is ARM order (pan, lift, elbow, wrist_1, wrist_2, wrist_3)."""
    return abs(wrap_angle(q[1]) - (-math.pi / 2.0))


def reachable_fold(q, max_droop):
    """True when the arm config ``q`` keeps its shoulder fold within the holdable envelope
    (``fold_droop(q) <= max_droop``) -- i.e. NOT the deep, physically-unreachable fold. A clear-wrist KDL
    solution must ALSO pass this to be accepted as a transfer preplace goal; see
    ``RosManipulationRuntime._guard_rejects``."""
    return fold_droop(q) <= float(max_droop)


def resolve_reach_guard(spec):
    """Normalise a ``transfer_preplace_reach_guard`` config value into ``(max_droop, upright_seed)`` or
    None (off => the plain wrist-guard, byte-identical). Accepts:

      * absent / None / False / '0' / 'off'  => None (off),
      * True / {} / 'on'                      => the defaults (DEFAULT_MAX_FOLD_DROOP, DEFAULT_UPRIGHT_SEED),
      * a mapping with optional ``max_droop`` (finite, > 0) and ``upright_seed`` (6 finite joints, ARM
        order; null/absent keeps the default generic seed; false drops the seed, leaving only the fold
        acceptance test).

    Raises ValueError on a malformed value so a typo fails loudly at boot, never silently inert."""
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        v = spec.strip().lower()
        if v in ('', '0', 'off', 'false', 'none'):
            return None
        if v in ('1', 'on', 'true'):
            spec = True
        else:
            raise ValueError(f'transfer_preplace_reach_guard: unknown value {spec!r}')
    if spec is True:
        return (DEFAULT_MAX_FOLD_DROOP, tuple(float(v) for v in DEFAULT_UPRIGHT_SEED))
    if not isinstance(spec, dict):
        raise ValueError('transfer_preplace_reach_guard must be a bool or a mapping')
    max_droop = spec.get('max_droop', DEFAULT_MAX_FOLD_DROOP)
    if not (isinstance(max_droop, (int, float)) and math.isfinite(max_droop) and max_droop > 0):
        raise ValueError('transfer_preplace_reach_guard.max_droop must be a finite positive number')
    seed = spec.get('upright_seed', DEFAULT_UPRIGHT_SEED)
    if seed is False or seed is None:
        upright = None
    else:
        seed = tuple(seed)
        if len(seed) != 6 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in seed):
            raise ValueError('transfer_preplace_reach_guard.upright_seed must be 6 finite joint values')
        upright = tuple(float(v) for v in seed)
    return (float(max_droop), upright)


def stamp_summary(summary, environ=None):
    """Record the active carry flags in a mission.json summary dict (in place; returned). Off => untouched
    (byte-identical). Never raises: a bad value is recorded, not hidden."""
    try:
        flags = carry_flags(environ)
    except ValueError as exc:
        summary['transfer_carry'] = 'invalid'
        summary['transfer_carry_error'] = str(exc)
        return summary
    if not flags:
        return summary
    ordered = [f for f in FLAGS if f in flags]
    summary['transfer_carry'] = ordered
    if WELD in flags:
        summary['transfer_attach'] = 'weld'
    workarounds = summary.setdefault('workarounds', [])
    note = summary.get('note') or ''
    for flag in ordered:
        label = f'transfer_carry={flag}'
        if label not in workarounds:
            workarounds.append(label)
        tag = f' TRANSFER_CARRY={flag}: {DESCRIPTIONS[flag]}.'
        if tag not in note:
            note += tag
    summary['note'] = note
    summary['walkthrough_caption'] = carry_caption(summary.get('walkthrough_caption'), environ)
    return summary


def carry_caption(base=None, environ=None):
    """The walkthrough caption with the active carry flags appended (burned into the video by the Recorder;
    mirrored into mission.json). Off/invalid => ``base`` unchanged (byte-identical)."""
    try:
        flags = carry_flags(environ)
    except ValueError:
        return base
    caption = base
    for flag in FLAGS:
        if flag not in flags:
            continue
        label = CAPTIONS[flag]
        if caption and label in caption:
            continue
        caption = f'{caption} | {label}' if caption else label
    return caption
