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


def wrist_flip_seed(q):
    """The wrist-flipped twin of ``q`` as an IK SEED: wrist_1 - pi, -wrist_2, wrist_3 + pi (shoulder/elbow
    unchanged). For a UR it is exact only when the wrist_2 offset is zero; with the real 0.116 m offset the
    flip moves the wrist centre, so the IK converges from this seed to the nearby exact clear solution
    (CPU on the mission URDF: shoulder_lift/elbow shift ~0.2-0.4 rad, tool pose identical)."""
    q = [float(v) for v in q]
    return (q[0], q[1], q[2], wrap_angle(q[3] - math.pi), -q[4], wrap_angle(q[5] + math.pi))


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
