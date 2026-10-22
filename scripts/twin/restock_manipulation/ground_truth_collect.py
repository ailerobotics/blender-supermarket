"""GROUND-TRUTH COLLECT (collect-gt): a captain-sanctioned, labelled demo workaround (default OFF).

Why: cold collect never completed (runs 343-348). The FIXED source camera never produced a
``backstock_00`` carton target (``COLLECT_CAPTURE target=None`` / ``unknown or undecoded label in
calibrated slot``), so the Isaac side published ZERO ``/sensors`` packets and the host failed
``no initial sensor frame`` once its WALL startup grace (``RESTOCK_STARTUP_GRACE_S=120``) expired ->
stop -> ``StationAckStreamStalled``. Not the fm476 stale-packet reject: no packet was ever sent.

What: with ``RESTOCK_COLLECT_GROUND_TRUTH=1`` the collect PICK target comes from the simulated pose of
the controlled backstock carton asset (``/RestockScenario/HandlingUnit``) instead of the refused camera
observation -- the fm461 ground-truth verify / fm462 ground_truth_pick pattern. Labelled end to end: the
pick Observation is ``source='ground_truth'`` (never ``'sensor'``), the host evidence records
``collect_mode='ground_truth'``, and mission.json carries ``collect_mode`` plus a note. It never
certifies inventory: ``verify`` stays the stock-commit gate.

One switch, read by every process that needs it. The env propagates from the orchestrate run env to
the Isaac mission process AND the host manipulation stack, like ``RESTOCK_STARTUP_GRACE_S``. Readers:
``isaac_factory.CollectTargetBridge`` (serves the target), ``restock.ros_manipulation`` (packet
integrity gate), ``controller.TransferController`` (provenance + fm449 gate) and
``run_mobile_restock`` / ``restock.bt.integration`` (mission.json). Unset => every path byte-identical.

Pure (no Isaac/ROS/pxr imports) so every gate is CPU-tested.
"""
import math
import os

ENV = 'RESTOCK_COLLECT_GROUND_TRUTH'
SOURCE = 'ground_truth'          # Observation.source of the ground-truth pick (never 'sensor')
MODE = 'ground_truth'            # collect_mode recorded in the host evidence + mission.json
PLACE_CHECK = 'ground_truth_place_check'   # controller's fm389 sim-measured place-verify record
# Opt-in deck-slot shift (metres along the cart BODY +x, i.e. toward the hitch/robot side) applied to
# the deck-interior load centre. Clamped so the carton stays on the 0.8 m deck whichever way it lies.
DECK_SLOT_ENV = 'RESTOCK_COLLECT_DECK_SLOT_X'
DECK_SLOT_MAX_M = 0.10

NOTE = (' GROUND-TRUTH COLLECT (RESTOCK_COLLECT_GROUND_TRUTH=1): the collect pick target is the '
        'simulated pose of the controlled backstock carton asset, not a source-camera observation '
        '(labelled demo workaround); see collect_mode.')


def enabled(env=None):
    """True only for an explicit truthy ``RESTOCK_COLLECT_GROUND_TRUTH``; unset/0/garbage => False."""
    e = os.environ if env is None else env
    return str(e.get(ENV, '')).strip().lower() in ('1', 'true', 'yes', 'on')


def mission_labels(env=None, *, skipped=()):
    """mission.json fields recording the workaround; {} when off (byte-identical summary) or when a
    start-phase shortcut skipped collect (nothing was picked, so there is no collect mode to label)."""
    return {'collect_mode': MODE} if enabled(env) and 'collect' not in tuple(skipped or ()) else {}


def label_summary(summary, shortcut=None, env=None):
    """Record the workaround in a mission.json summary in place: collect_mode + the caption note."""
    labels = mission_labels(env, skipped=(shortcut or {}).get('skipped', ()))
    if labels:
        summary.update(labels)
        summary['note'] = summary.get('note', '') + NOTE
    return summary


def deck_slot_x(env=None):
    """The opt-in deck-slot body-x shift in metres, clamped to +/-DECK_SLOT_MAX_M; 0.0 when unset."""
    e = os.environ if env is None else env
    try:
        v = float(e.get(DECK_SLOT_ENV, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(-DECK_SLOT_MAX_M, min(DECK_SLOT_MAX_M, v))


def carton_top_pose(world_from_unit, world_from_base, local_center, height):
    """The ground-truth pick pose in the camera observation's convention.

    ``world_from_unit`` / ``world_from_base`` are 4x4 (row-major, column vectors) local-to-world
    transforms of the carton prim and ``ur_base_link``; ``local_center`` the carton-local centre
    (x,y,z) and ``height`` the profile height. Returns ``(x, y, z, qx, qy, qz, qw)``:

    * position = the TOP-face centre (``Observation.pose`` is the observed top contact surface; the
      wall policy derives the grasp centre as ``z - height/2``) in ``ur_base_link``, the frame the
      fixed source camera reports in (run-326 camera pose (-0.001, 1.1, 0.261) for this carton);
    * orientation = the carton's WORLD yaw (map), exactly what the camera ``pose_station`` carries --
      the bridge's ``_map_pose_to_base`` re-expresses it through the live chassis yaw like the camera.
    """
    import numpy as np
    U = np.asarray(world_from_unit, float)
    B = np.asarray(world_from_base, float)
    if U.shape != (4, 4) or B.shape != (4, 4) or not np.isfinite(U).all() or not np.isfinite(B).all():
        raise ValueError('carton/base transforms must be finite 4x4 matrices')
    c = U @ np.array([float(local_center[0]), float(local_center[1]), float(local_center[2]), 1.0])
    up = U[:3, 2] / (np.linalg.norm(U[:3, 2]) or 1.0)          # carton local +z (height axis) in world
    top = c[:3] + up * (float(height) / 2.0)
    p = np.linalg.inv(B) @ np.array([top[0], top[1], top[2], 1.0])
    yaw = math.atan2(float(U[1, 0]), float(U[0, 0]))
    return (float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def stage_evidence_from_place_check(last, *, sku, quantity, expected_location):
    """Turn the controller's final fm389 ``ground_truth_place_check`` record into stage evidence.

    With ``RESTOCK_VERIFY_PLACE`` the collect place is verified from the sim-measured carton pose (no
    fixed camera frames the cart), so the stage's final evidence is that ground-truth record. It has no
    sku/observation_id and is not 'sensor', so the runtime refused it ('transfer evidence mismatch') and
    the mission would refuse it ('action requires sensor evidence') -- latent until a collect completes.
    This keeps it HONEST: source 'ground_truth' + an explicit ``ground_truth`` flag, the verify mode and
    the measured offsets carried through. Raises ValueError on any location/clock mismatch.
    """
    if not isinstance(last, dict) or last.get('source') != PLACE_CHECK:
        raise ValueError('not a ground-truth place-check record')
    if last.get('location') != expected_location:
        raise ValueError('ground-truth place check at %r, expected %r'
                         % (last.get('location'), expected_location))
    t = last.get('simulation_time')
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        raise ValueError('ground-truth place check lacks a finite simulation_time')
    out = {'source': SOURCE, 'ground_truth': True,
           'observation_id': 'ground_truth_place:%.3f' % float(t),
           'simulation_time': float(t), 'sku': sku, 'location': expected_location,
           'quantity': quantity, 'verify_mode': last.get('verify_mode', 'ground_truth_place')}
    for key in ('collect_mode', 'xy_offset', 'bottom_gap', 'carton_pose_station', 'support_pose_station'):
        if key in last:
            out[key] = last[key]
    return out
