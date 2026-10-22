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
``collect_mode='ground_truth'``, and mission.json + the burned-in walkthrough caption name it. It
never certifies inventory: ``verify`` stays the stock-commit gate.

Where the carton ends (the ONE spot, decided with sim-video-transfer-fallback-v1): the warm start_at
shortcut's slot -- rollwagen UPPER shelf, cart body (+0.18, 0), support z 0.9325 (carton centre 1.0125),
yaw -90 deg cart-relative (0.6 m side across the cart, gripped face toward the hitch). The PLACE target
reads the SAME scenario ``initial_staging`` fields the shortcut uses (``warm_slot``), so cold and warm
runs hand the transfer an identical carton pose; the deck target (RESTOCK_PLACE_ON_DECK) is not used.

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
    """Record the workaround in a mission.json summary in place: collect_mode, the note and the
    walkthrough caption (appended to a start-phase shortcut's caption, if any)."""
    labels = mission_labels(env, skipped=(shortcut or {}).get('skipped', ()))
    if labels:
        summary.update(labels)
        summary['note'] = summary.get('note', '') + NOTE
        base = summary.get('walkthrough_caption')
        summary['walkthrough_caption'] = f'{base} | {CAPTION}' if base else CAPTION
    return summary


CAPTION = 'collect: GROUND-TRUTH pick (simulated carton pose, not camera)'


def join_caption(base, config=None, env=None):
    """The walkthrough caption the encode burns into walkthrough.mp4 (restock.recording), with the
    ground-truth collect label appended when the switch is on and collect actually runs. collect is
    the first mission stage, so any other ``start_at`` skips it. Off => ``base`` unchanged."""
    start_at = config.get('start_at') if isinstance(config, dict) else None
    if not enabled(env) or (start_at and start_at != 'collect'):
        return base
    return f'{base} | {CAPTION}' if base else CAPTION


def warm_slot(scenario):
    """The ONE carton spot on the rollwagen: where the warm ``start_at`` shortcut stages it (scenario
    ``initial_staging``: ``cart_load_center_body_xy``, ``cart_support_z_world`` and the cart-relative
    yaw ``unit_yaws_world.cart - cart_body_pose[2]``), which the transfer pick expects. Reading the SAME
    fields as ``restock.shortcut.plan_start_state`` keeps cold and warm placement equal by construction.
    Returns ``{'load_center_body_xy': (x, y), 'support_z_world': z, 'yaw_cart_rel': rad}``; raises
    ValueError on missing/malformed staging (fail closed -- never an invented spot)."""
    st = scenario.get('initial_staging') if isinstance(scenario, dict) else None
    if not isinstance(st, dict):
        raise ValueError('scenario.initial_staging is required for the ground-truth collect place slot')
    xy, z = st.get('cart_load_center_body_xy'), st.get('cart_support_z_world')
    cart, yaw = st.get('cart_body_pose'), (st.get('unit_yaws_world') or {}).get('cart')

    def finite(*v):
        return all(isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a) for a in v)
    if not (isinstance(xy, (list, tuple)) and len(xy) == 2 and finite(*xy)):
        raise ValueError('initial_staging.cart_load_center_body_xy must be finite x/y')
    if not finite(z):
        raise ValueError('initial_staging.cart_support_z_world must be finite')
    if not (isinstance(cart, (list, tuple)) and len(cart) == 3 and finite(*cart)) or not finite(yaw):
        raise ValueError('initial_staging needs cart_body_pose x/y/yaw and unit_yaws_world.cart')
    rel = (float(yaw) - float(cart[2]) + math.pi) % (2 * math.pi) - math.pi
    return {'load_center_body_xy': (float(xy[0]), float(xy[1])), 'support_z_world': float(z),
            'yaw_cart_rel': rel}


def slot_pose(world_from_cart, world_from_base, slot):
    """The place-target pose for ``slot`` (see warm_slot) on the cart at its LIVE pose: the support
    point (cart body xy, world support z) in ``ur_base_link`` plus the carton's intended WORLD yaw
    (cart yaw + cart-relative yaw) -- the same convention as carton_top_pose, so the bridge's
    ``_map_pose_to_base`` turns it base-relative exactly like the pick. Pure numpy."""
    import numpy as np
    C = np.asarray(world_from_cart, float)
    B = np.asarray(world_from_base, float)
    if C.shape != (4, 4) or B.shape != (4, 4) or not np.isfinite(C).all() or not np.isfinite(B).all():
        raise ValueError('cart/base transforms must be finite 4x4 matrices')
    lx, ly = slot['load_center_body_xy']
    w = C @ np.array([float(lx), float(ly), 0.0, 1.0])
    p = np.linalg.inv(B) @ np.array([w[0], w[1], float(slot['support_z_world']), 1.0])
    yaw = math.atan2(float(C[1, 0]), float(C[0, 0])) + float(slot['yaw_cart_rel'])
    return (float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


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
