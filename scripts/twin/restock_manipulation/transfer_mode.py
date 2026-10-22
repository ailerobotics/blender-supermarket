"""Opt-in, SIM-ONLY transfer workaround modes (default off => every caller byte-identical).

ONE switch, the env var ``RESTOCK_TRANSFER_MODE``, read by BOTH mission processes -- the Isaac capture
(``isaac_factory``) and the host ``TransferController`` -- so it cannot be split across the host/runtime
configs (the run-277 trap: a key in the wrong file silently never activates). Each side logs its
activation (``TRANSFER_MODE ...`` in the Isaac console, ``transfer_mode`` in ``HOST_EFFECTIVE_CONFIG``),
and ``stamp_summary`` records it in mission.json.

``live_pose`` -- the real-arm transfer reach fix (run-354/356 'approach outside reach envelope'):
  * PICK from the carton's LIVE pose -- the handling-unit prim's centre + yaw, presented as the observed
    TOP contact surface (the ``types.Observation.pose`` contract) -- instead of the deck-top AABB centre,
    which is neither where a warm-start carton rests (upper shelf z 0.93, cart body +0.18 m) nor its top
    (the side grip landed 0.16 m low, 1.27 m behind the arm base).
  * NEAR-FACE grip: a rectangular sealed carton and the bay support are both symmetric under a 180 deg yaw,
    so the host picks the yaw representative (theta / theta+pi) of the carton AND of the bay place that
    minimises the worst station reach -- i.e. keeps the configured side-grip wall facing the arm (bay place
    yaw pi on the backroom bay; the ground-truth verify checks position only).
  * NO base reposition inside the transfer: ``reposition_before_place``'s TRANSIT hold is a collect-only
    handshake (the runtime never drives a transfer reposition), so a transfer that passed the pick would
    hold in TRANSIT until the phase timeout. live_pose routes lift -> tuck -> preplace (the collect-cart pan
    is skipped too) and reach-gates EVERY station, place side included, at observe.
  * REQUIRES the wrist_guard: live_pose fixes the PICK reach but its preplace/place/retreat are still
    seeded-KDL joint-space moves whose IK can jam the ur_wrist_2-vs-ur_forearm self-collision branch
    (audit transfer-preplace-wrist-branch-jam). The controller fails a live_pose transfer FAST with a named
    reason at observe unless ``RESTOCK_TRANSFER_CARRY=wrist_guard`` is also set, rather than silently
    running the jam-prone planner (which dies as a generic 'arm command heartbeat lost' e-stop).
"""
from __future__ import annotations

import math
import os
from dataclasses import replace

ENV = 'RESTOCK_TRANSFER_MODE'
LIVE_POSE = 'live_pose'
SCRIPTED = 'scripted'
MODES = (LIVE_POSE, SCRIPTED)
DESCRIPTIONS = {
    LIVE_POSE: ('real-arm transfer picks the carton from its live (controlled-asset ground-truth) pose with '
                'a near-face side grip, bay place yaw chosen for reach, and no base reposition'),
    SCRIPTED: ('SCRIPTED transfer fallback: the carton is moved from the rollwagen to the ground-truth bay '
               'target along a scripted kinematic path (arm still, or opted in: the scripted arm path through '
               'the host trajectory executor; not cuMotion-planned, not physics-grasped); placement checked '
               'against the ground-truth target'),
}
CAPTIONS = {
    LIVE_POSE: 'TRANSFER: live-pose pick (ground-truth carton pose)',
    SCRIPTED: 'SCRIPTED TRANSFER (workaround): scripted carton path, not planned or grasped',
}
_OFF = ('', '0', 'off', 'none', 'false')


def transfer_mode(environ=None):
    """The active transfer mode, or None (unset/empty/'0'/'off' => off). An unknown value RAISES so a
    typo fails loudly at boot instead of silently running the default transfer."""
    raw = (os.environ if environ is None else environ).get(ENV, '')
    value = str(raw).strip().lower()
    if value in _OFF:
        return None
    if value not in MODES:
        raise ValueError(f'{ENV}={raw!r}: unknown transfer mode (known: {", ".join(MODES)})')
    return value


def stamp_summary(summary, environ=None):
    """Record the active transfer mode in a mission.json summary dict (in place; returned). Absent =>
    the summary is untouched (byte-identical). Never raises: a bad value is recorded, not hidden."""
    try:
        mode = transfer_mode(environ)
    except ValueError as exc:
        summary['transfer_mode'] = 'invalid'
        summary['transfer_mode_error'] = str(exc)
        return summary
    if mode is None:
        return summary
    summary['transfer_mode'] = mode
    workarounds = summary.setdefault('workarounds', [])
    label = f'transfer_mode={mode}'
    if label not in workarounds:
        workarounds.append(label)
    note = summary.get('note') or ''
    tag = f' TRANSFER_MODE={mode}: {DESCRIPTIONS[mode]}.'
    if tag not in note:
        summary['note'] = note + tag
    summary['walkthrough_caption'] = transfer_caption(summary.get('walkthrough_caption'), environ)
    return summary


def transfer_caption(base=None, environ=None):
    """The walkthrough caption with the active transfer mode appended (burned into the video by the
    Recorder; mirrored into mission.json). Off/invalid => ``base`` unchanged (byte-identical)."""
    try:
        mode = transfer_mode(environ)
    except ValueError:
        return base
    if mode is None:
        return base
    label = CAPTIONS[mode]
    if base and label in base:
        return base
    return f'{base} | {label}' if base else label


def flip_yaw(quaternion):
    """The same box rotated 180 deg about its OWN z axis: q * (0, 0, 1, 0), quaternions (x, y, z, w)."""
    x, y, z, w = quaternion
    return (y, -x, w, -z)


def _worst_reach(poses):
    return max(math.dist(p[:3], (0., 0., 0.)) for p in poses.values())


def near_face_poses(policy, source, target, profile):
    """Build the pick/place stations with the yaw representatives (source theta|theta+pi, target
    theta|theta+pi) that minimise the WORST station reach from the arm base. Returns ``(poses, choice)``
    where choice = dict(source_flipped, target_flipped, worst_reach). Ties keep the unflipped poses, so a
    layout that already faces the arm is unchanged. A combination the policy refuses is skipped; if all
    four are refused the first refusal is re-raised (fail closed, never a silent default)."""
    best, first_error = None, None
    for flip_source in (False, True):
        for flip_target in (False, True):
            s = replace(source, pose=(*source.pose[:3], *flip_yaw(source.pose[3:]))) if flip_source else source
            t = replace(target, pose=(*target.pose[:3], *flip_yaw(target.pose[3:]))) if flip_target else target
            try:
                poses = policy.build(s, t, profile)
            except ValueError as exc:
                first_error = first_error or exc
                continue
            worst = _worst_reach(poses)
            if best is None or worst < best[0] - 1e-9:
                best = (worst, poses, dict(source_flipped=flip_source, target_flipped=flip_target,
                                           worst_reach=worst))
    if best is None:
        raise first_error
    return best[1], best[2]
