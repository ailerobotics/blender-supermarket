"""Fail-fast activation preflight for mission batch config keys (CPU, no GPU).

run-277 burned a full ~35-min GPU run because the cart-cage batch keys were authored
into the RUNTIME config, but `ros_host` reads them from the HOST config
(`ros_host_config.json`) -- the feature silently never activated and the run repeated
an already-known failure. A config-placement slip produces a byte-identical, silently
inert run, not an error: the most expensive failure mode on a GPU-bounded lane.

This module proves, BEFORE any GPU boot, that every batch key a run declares it needs
is present in the file its CONSUMER actually reads -- and is not merely mis-placed in
the other config file (the exact run-277 slip). The orchestrator runs `main()` and
refuses to boot (exit 2) when a required key is missing or mis-placed.

The two mission config files and their consumers:
  * HOST config (`restock_manipulation/ros_host_config.json`): read by `ros_host.py` /
    `TransferController` -- place/cage/staged-swing/contact keys.
  * RUNTIME config (`mission-runtime-*.json`): read by the Isaac side
    (`restock.isaac_bundle` / `isaac_port` / `physics_setup`) -- drive gains, phase
    drive, stage timeout, commissioning dir.

To register a NEW batch key, add it to `KEY_CONSUMER` with the side that reads it. The
orchestrator declares the keys a given run requires via `--require`.
"""
from __future__ import annotations

import argparse
import json
import sys

HOST = 'host'
RUNTIME = 'runtime'

# Which config file's consumer reads each batch key. Extend this when a batch adds a key.
KEY_CONSUMER = {
    # --- HOST config (ros_host.py / TransferController) ---
    'cart_cage_place': HOST,
    'world_obstacles': HOST,
    'cage_axis': HOST,
    'cage_mouth_standoff': HOST,
    'cage_insert_gap': HOST,
    'staged_cart_swing': HOST,
    'sim_cart_place_policy': HOST,
    'reposition_before_place': HOST,   # fm390: ros_host builds the controller with it (base drives to place_approach)
    'cage_preplace_shape': HOST,
    # sim-robust-cart-place-goal-v1: non-cage collect place-on-cart goal guards (ros_host -> TransferController)
    'collect_place_preplace_shape': HOST,   # fixed UPRIGHT clear-region preplace joint shape (begin_joint_goal)
    'collect_place_wrist_guard': HOST,       # clear-branch wrist-guard extended to the collect place/retreat pose-IK
    'collect_place_straight_line': HOST,     # route the collect deck place+retreat collision-off straight-line (attached-carton self-collision)
    'collect_retreat_liftoff': HOST,         # sim-collect-retreat-abort-v2: lift the empty gripper straight up before the collect deck retreat folds to the preplace shape (default ON)
    'collect_retreat_liftoff_dz': HOST,      # sim-collect-retreat-abort-v2: the straight-up TCP rise (m) for that lift-off
    'transfer_preplace_reach_guard': HOST,   # sim-robust-transfer-preplace-reach-v1: steer the live_pose transfer preplace to the holdable upright clear branch
    'transfer_pick_wrist_guard': HOST,       # sim-robust-transfer-settle-oscillation-v1: extend the clear-wrist-region guard to the live_pose transfer PICK approach+contact (coldwin12 contact limit cycle)
    'transfer_service_carry': HOST,          # sim-service-leg-carry-v1: TRANSIT carry-hold for the shelf-staging live_pose transfer service leg (pick@staging -> carry -> place); pair with runtime shelf_staging
    'cart_pan_joint0': HOST,
    'cart_tuck_shape': HOST,
    'base_collision_object': HOST,
    'contact_straight_line': HOST,
    'pose_goal': HOST,
    # fm402: the controller reads these cap/limit keys OPTIONALLY via config.get(...) with a silent
    # default, so an absent key silently un-caps/limits the arm (the run-277 class). carry_wrist1_cap
    # absent = the loaded LIFT drives wrist_1 past the wrist_2-vs-forearm self-collision stop and jams
    # (runs 300d/305/311). They are in BASELINE_REQUIRED so the preflight fails LOUDLY if any is missing.
    'carry_wrist1_cap': HOST,
    'carry_lift_shape': HOST,     # fm403: fixed clear-region joint goal for the loaded lift
    'carry_wrist2_min': HOST,     # fm403: wrist_2 lower-bound guard on task-space carry moves
    'contact_overpress_cap_m': HOST,
    'contact_overpress_step_m': HOST,
    'arm_max_speed': HOST,
    'plan_wall_cap_s': HOST,
    'grasp_track_m': HOST,        # fm410: lift-probe carton-tracking physics confirm threshold (m)
    # sim-robust-cart-place-descent-v1: SIM-clock release-detach budget; ros_host -> TransferController
    'release_detach_timeout_s': HOST,
    # sim-robust-movegroup-collision-meshes-v1: rewrite the resolved URDF's bare absolute <mesh>
    # paths to file:// URIs so move_group's resource_retriever loads real arm link collision
    # geometry (restock_host.launch reads it from ros_host_config.json). Default off => the
    # robot_description is the file verbatim (byte-identical, the historical empty-geometry path).
    'moveit_mesh_uris': HOST,
    'esdf_boot_ack_grace_s': HOST,  # sim-robust-esdf-boot-gate-v1: SIM-clock boot grace for the first valid ESDF ack
    'carton_level': HOST,  # sim-carton-level-carry-v1: level the carried carton (fold goals) + CARTON_TILT trip; HOST (the controller drives the arm)
    # --- RUNTIME config (isaac_bundle / isaac_port / physics_setup) ---
    'phase_drive': RUNTIME,
    'arm_stiffness_abs': RUNTIME,
    'arm_damping_abs': RUNTIME,
    'arm_max_effort': RUNTIME,
    'arm_gravity_ff': RUNTIME,
    # sim-robust-executor-shoulder-sag-v1: load-aware gravity FF for the executor LOADED pick. All three
    # are read by physics_setup.assemble (RUNTIME). Opt-in, default-off => byte-identical.
    'arm_gravity_ff_loaded_carry': RUNTIME,
    'arm_gravity_ff_payload_kg': RUNTIME,
    'arm_gravity_ff_payload_link': RUNTIME,
    'stage_timeout_s': RUNTIME,
    'commissioning_dir': RUNTIME,
    # couple/outbound/park activation (fm356): the nav stack + mode that drive the post-collect stages
    # (coupling planner PR150, outbound tow, park) -- asserted present so a run that finally clears
    # collect actually attempts them instead of stopping configured-off.
    'nav_factory': RUNTIME,
    'nav_mode': RUNTIME,
    'navigation': RUNTIME,
    'collect_observe_force_render_ticks': RUNTIME,   # fm452: COLD force-render window; isaac_bundle reads it
    'cage_collision': RUNTIME,   # sim-only carton-vs-cage contact filter; physics_setup -> scene.build reads it
    # sim-chassis-pitch-stability-v1 FALLBACK: sim-only station pitch hold. physics_setup.assemble reads it
    # (RUNTIME) to build the MobileController with the pitch/roll-locking station brake. Default off =>
    # byte-identical. The preferred source fix (compliant caster suspension) is a MobileSpec field in the
    # mobile manifest, NOT a batch config key.
    'chassis_station_pitch_hold': RUNTIME,
    # coldwin39 fix: the station-hold target attitude (degrees); physics_setup.assemble reads them
    # (RUNTIME). Absent => default rig rest pitch -1.17 / roll 0 (not the instantaneous end stop).
    'chassis_station_pitch_hold_pitch_deg': RUNTIME,
    'chassis_station_pitch_hold_roll_deg': RUNTIME,
    # sim-collect-retreat-collider-v3: defer re-enabling the tool-side gripper colliders past the empty
    # retreat (DEFAULT ON). isaac_factory.parse_manipulation_config reads it (RUNTIME); env
    # RESTOCK_DEFER_TOOL_COLLISION is the alias. Registered so a run can --require it (assert ON).
    'defer_tool_collision': RUNTIME,
    'sim_speed': RUNTIME,        # sim-video-speedup-v1 speed profile; isaac_bundle/isaac_port + bt view read it
    # sim-robust-cart-place-descent-v1: bounded base-settle tolerance for the loaded collect place;
    # isaac_bundle builds the RuntimeAdapter with it (orchestrate process), NOT ros_host.
    'collect_place_base_settle_m': RUNTIME,
    # sim-shelf-staging-v1 / sim-service-leg-carry-v1: the sales-floor re-sequence + solo carton-carry
    # service leg. isaac_bundle builds the RuntimeAdapter with these (RUNTIME). Pair `shelf_staging: true`
    # (runtime) with the HOST `transfer_service_carry: true` so the live_pose transfer produces the carry-
    # hold the runtime drives; a split (one side only) is the run-277 trap these registrations guard against.
    'shelf_staging': RUNTIME,
    'carry_drive_profile': RUNTIME,   # drive-with-payload base speed/accel caps for the carry→endcap leg
    # sim-cart-side-bay-v1: park the cart in an OFF-AXIS side bay (extends shelf_staging); isaac_bundle
    # builds the RuntimeAdapter with it (RUNTIME). Pair `cart_side_bay: true` with `shelf_staging: true`.
    'cart_side_bay': RUNTIME,
    # sim-service-place-target-v1: the TRANSFER place target + its slot id + the mission poses are
    # read orchestrate-side (isaac_bundle / restock_camera.sensing / ground_truth_verify => RUNTIME).
    # resolve_transfer_place_target reads them to select the endcap in the shelf_staging flow and to
    # fail loud on a mis-filed (unreachable) target -- the coldwinint1 backroom-bay trap.
    'transfer_ground_truth': RUNTIME,
    'transfer_shelf_location': RUNTIME,
    'mission_poses': RUNTIME,
    'verify_ground_truth': RUNTIME,    # ground_truth_verify.from_config (orchestrate side); the verify
                                       # bay is kept consistent with the resolved place target
}

# fm402: cap/limit keys the controller reads OPTIONALLY (config.get with a silent default). These are
# ALWAYS asserted present (unioned into every run's --require), so a missing cap fails the launch loudly
# instead of silently un-capping the arm -- the carry_wrist1_cap slip that let the loaded lift drive
# wrist_1 into the self-collision stop (runs 300d/305/311). Add any future optional cap/limit key here.
BASELINE_REQUIRED = (
    'carry_wrist1_cap',
    'carry_lift_shape',
    'carry_wrist2_min',
    'contact_overpress_cap_m',
    'contact_overpress_step_m',
    'arm_max_speed',
    'plan_wall_cap_s',
    'grasp_track_m',          # fm410: physics grasp-confirm threshold; absent => the camera-independent
                              # lift-probe confirm silently de-activates (run-316 phantom-load risk).
)


def check_activation(host_cfg, runtime_cfg, required):
    """Return a list of human-readable problems ([] == every required key is loadable).

    For each required key: it must be registered, present in the config its consumer
    reads, and not present ONLY in the wrong file (the run-277 mis-placement slip).
    """
    problems = []
    side_cfg = {HOST: host_cfg, RUNTIME: runtime_cfg}
    other = {HOST: RUNTIME, RUNTIME: HOST}
    for key in required:
        consumer = KEY_CONSUMER.get(key)
        if consumer is None:
            problems.append(
                f"{key!r}: unregistered batch key -- add it to config_preflight.KEY_CONSUMER "
                f"with the side ({HOST!r}/{RUNTIME!r}) whose code reads it")
            continue
        in_consumer = key in side_cfg[consumer]
        in_other = key in side_cfg[other[consumer]]
        if in_consumer:
            continue
        if in_other:
            problems.append(
                f"{key!r}: MIS-PLACED -- present in the {other[consumer]} config but its "
                f"consumer reads the {consumer} config; it will silently never activate "
                f"(the run-277 slip). Move it to the {consumer} config.")
        else:
            problems.append(
                f"{key!r}: MISSING from the {consumer} config its consumer reads.")
    return problems


def check_flow_consistency(host_cfg, runtime_cfg):
    """sim-service-carry-drive-to-shelf-v1: the shelf-staging service leg needs BOTH halves together,
    across the two config layers, or it silently does nothing.

    The RUNTIME ``shelf_staging`` re-sequence (isaac_bundle -> RuntimeAdapter) is what DRIVES the base
    carry to ``shelf_approach``, but it only fires once the arm has folded to TRANSIT -- and the arm
    only folds to TRANSIT when the HOST ``transfer_service_carry`` is on (controller
    ``_repositions_before_place``). With ``shelf_staging`` on but ``transfer_service_carry`` off
    (transferobs1: the HOST config left the key absent -> default False), the live_pose transfer goes
    tuck -> preplace, the base NEVER carries the carton, and the controller gates the endcap place
    reach from the side-bay PARK pose at observe -> 'unreachable or undersized target' -> emergency
    stop. That is a cross-LAYER split, invisible to :func:`check_activation` (both keys are in the
    right FILE; one is just false), so name it here and REFUSE at plan time, before any GPU boot.

    One-directional on purpose: ``shelf_staging`` ON requires ``transfer_service_carry`` ON. A
    ``transfer_service_carry`` left on with ``shelf_staging`` off does not drive a base carry and, for
    a live_pose transfer with no runtime drive, fails BOUNDED (the TRANSIT phase times out by name) --
    it is not the silent skip, and refusing it would false-block every non-staging run that shares the
    repo HOST config (collect-only, legacy transfer)."""
    if bool(runtime_cfg.get('shelf_staging')) and not bool(host_cfg.get('transfer_service_carry')):
        return [("shelf-staging service leg SPLIT across config layers: runtime 'shelf_staging' is ON "
                 "but host 'transfer_service_carry' is OFF. The live_pose transfer will not fold to "
                 "TRANSIT, so the base never carries the carton to shelf_approach and the controller "
                 "gates the endcap place reach from the side-bay park pose at observe "
                 "('unreachable or undersized target' -> emergency stop; transferobs1). Set "
                 "'transfer_service_carry': true in the HOST config (ros_host_config.json) alongside "
                 "the runtime 'shelf_staging': true.")]
    return []


def check_carton_level(host_cfg):
    """sim-place-descent-residual-v1/-v2/-v3: resolve the HOST ``carton_level`` block so a malformed value
    (incl. the place-clearance keys ``preplace_above_place`` / ``preplace_standoff_m`` /
    ``place_clearance_check`` / ``place_clearance_margin_m``, the v2 keys ``place_clearance_drooped`` /
    ``closed_loop_level`` / ``closed_loop_authority_deg`` / ``closed_loop_rate_deg_per_s`` /
    ``closed_loop_gain``, and the v3 keys ``place_contact_release`` / ``place_contact_gap_m`` /
    ``closed_loop_sensor``, read from ros_host_config.json) FAILS the pre-flight loudly instead of at a GPU
    boot. ``resolve_level_config``/``LevelConfig.validate`` enforce the ranges and the
    ``closed_loop_sensor`` -> ``closed_loop_level`` dependency; a violation surfaces here as a malformed
    refusal. Also flags incoherent combinations -- the LEVEL clearance gate armed with no remedy, and the v1
    ``preplace_above_place`` remedy left ON (sim-place-descent-residual-v2 found it regresses the set-down)
    -- as refusals. No-op ([]) when ``carton_level`` is absent/off (byte-identical)."""
    from twin.restock_manipulation import carton_level as cl
    spec = (host_cfg or {}).get('carton_level')
    try:
        resolved = cl.resolve_level_config(spec)
    except ValueError as exc:
        return [f"carton_level (HOST ros_host_config.json): malformed -- {exc}"]
    if resolved is None:
        return []
    problems = []
    # sim-place-descent-residual-v2: coldwin41 proved the v1 remedy preplace_above_place REGRESSES the
    # set-down (its IK drifts the carton ~93 mm toward the +x mouth -> post_2 edge pivot -> 4.94 deg). The
    # v2 coherent fix is the LEAST-EXTENDED approach (external place load-centre + base pose) + the
    # closed-loop backstop (closed_loop_level) + the DROOP-AWARE gate (place_clearance_drooped), with
    # preplace_above_place OFF. So the v1 LEVEL gate's gate-without-remedy refusal only applies when the
    # v2 droop-aware gate is NOT the one in use.
    if resolved.place_clearance_check and not resolved.place_clearance_drooped \
            and not resolved.preplace_above_place:
        problems.append(
            "carton_level place-clearance LEVEL gate armed (place_clearance_check) but no remedy is on: "
            "either set 'place_clearance_drooped': true (the v2 droop-aware gate, used with the "
            "least-extended approach + 'closed_loop_level') or, for the v1 path, 'preplace_above_place': "
            "true. The idealised level sweep alone (place_clearance_check) describes a pose the arm never "
            "holds (coldwin41).")
    if resolved.preplace_above_place:
        problems.append(
            "carton_level 'preplace_above_place' is ON: sim-place-descent-residual-v2 found this v1 remedy "
            "REGRESSES the collect set-down -- its above-place IK drifts the welded carton ~93 mm toward "
            "the +x mouth so the pad strikes post_2 at an edge (coldwin41 CARTON_TILT 4.94 deg). Turn it "
            "OFF and use the least-extended approach + 'closed_loop_level' + 'place_clearance_drooped'.")
    return problems


def check_carton_dims(host_cfg, scenario_cfg, *, tol_m=1e-3):
    """sim-collect-press-regression-v1: refuse a host carton-profile dimension that disagrees with the
    scenario single source of truth (``handling_unit.dimensions_m``, PR 268), before any GPU boot.

    The HOST controller derives the side-grasp PICK pose from ``profile.dimensions`` --
    ``wall_policy.WallCartonPolicy``/``shelfbot ... geometry.wall_grip`` stands the pad off the carton
    CENTRE by ``dimensions[wall-axis]/2 + contact_margin``. So a host dimension LARGER than the real
    stage carton places the pad that overshoot OUTSIDE the true face, and the capped contact over-press
    (``contact_overpress_cap_m``) can never close the remainder: the grasp fails
    'contact not reached within capped over-press travel'. coldwin32 regressed exactly this way -- the
    HOST config drifted to 0.6/0.4/0.16 (odd backstock wall uses dims[1]=0.40 vs the scenario 0.30 => a
    0.05 m over-reach, minus the 0.02 m over-press cap => ~0.03 m short, measured gap 34.96 mm on the
    GPU) while the scenario single source stayed 0.45/0.30/0.18. The two values drifted apart silently
    in separate files; name the mismatch here so the drift class cannot reach a GPU boot again.

    No-op (``[]``) when the scenario or its ``handling_unit`` dims are absent (``--scenario`` not passed),
    so existing preflight invocations are unchanged. Compares per-SKU when the scenario names one."""
    hu = (scenario_cfg or {}).get('handling_unit') or {}
    want = hu.get('dimensions_m') or hu.get('dimensions')
    if not want:
        return []
    sku = hu.get('sku')
    problems = []
    for prof in (host_cfg.get('profiles') or []):
        if sku is not None and prof.get('sku') != sku:
            continue
        dims = prof.get('dimensions')
        if dims is None:
            continue
        if len(dims) != len(want) or any(
                abs(float(a) - float(b)) > tol_m for a, b in zip(dims, want)):
            problems.append(
                f"carton-dim DRIFT: HOST profile {prof.get('sku')!r} dimensions {list(dims)} != "
                f"scenario handling_unit.dimensions_m {list(want)} (the PR 268 single source). The HOST "
                f"side-grasp pick standoff derives from the host dims, so a larger host dimension "
                f"over-reaches the pad and the capped over-press cannot reach contact "
                f"('contact not reached within capped over-press travel'; coldwin32). Restore the host "
                f"profile dimensions in ros_host_config.json to the scenario values.")
    return problems


#: sim-chassis-pitch-stability-v1. The carton-level bound is captain-fixed at 2.0 deg; a rig whose
#: UNSPRUNG free pitch travel exceeds (bound - margin) can rock the welded carton off level with no arm
#: fault (coldwin33-37: caster_lift 0.03 -> +-4.5 deg see-saw -> trip at 2.06). Refuse such a rig unless a
#: chassis stabiliser (compliant suspension or the sim-only station pitch hold) is active.
CARTON_TILT_BOUND_DEG = 2.0
PITCH_TRAVEL_MARGIN_DEG = 0.5
#: config keys that assert a station pitch-hold stabiliser is active (either layer).
STATION_PITCH_HOLD_KEYS = ('chassis_station_pitch_hold',)


def check_pitch_stability(mobile_spec, host_cfg, runtime_cfg, *,
                          bound_deg=CARTON_TILT_BOUND_DEG, margin_deg=PITCH_TRAVEL_MARGIN_DEG):
    """Refuse a chassis whose predicted free pitch travel exceeds ``bound_deg - margin_deg`` unless a
    chassis stabiliser is active (sim-chassis-pitch-stability-v1).

    ``mobile_spec``: the ``spec`` dict from the mobile manifest (``caster_lift_m``, ``length``,
    ``caster_suspension`` ...). The free pitch travel is ``atan(caster_lift_m / caster_spacing)`` where
    ``caster_spacing = length/2 - 0.12`` (matching ``mobile_assets``). A stabiliser is active when the
    spec enables the compliant suspension (``caster_suspension``) OR either config layer sets a station
    pitch-hold key. Returns a list of problems ([] == safe or stabilised). No-op ([]) when no mobile
    spec is supplied (existing preflight invocations are unchanged)."""
    from twin.traction import free_pitch_travel_deg, caster_spacing_m
    if not mobile_spec:
        return []
    length = float(mobile_spec.get('length', 1.0))
    lift = float(mobile_spec.get('caster_lift_m', 0.0))
    spacing = caster_spacing_m(length)
    travel = free_pitch_travel_deg(lift, spacing)
    threshold = bound_deg - margin_deg
    suspension = bool(mobile_spec.get('caster_suspension'))
    station_hold = any(bool((host_cfg or {}).get(k)) or bool((runtime_cfg or {}).get(k))
                       for k in STATION_PITCH_HOLD_KEYS)
    stabilised = suspension or station_hold
    if travel > threshold and not stabilised:
        return [
            f"chassis PITCH-STABILITY: free pitch travel {travel:.2f} deg (caster_lift_m={lift} m, "
            f"caster_spacing={spacing:.2f} m) exceeds the carton bound {bound_deg:.1f} deg minus "
            f"{margin_deg:.1f} deg margin ({threshold:.2f} deg), and NO chassis stabiliser is active. "
            f"An unsprung chassis is a see-saw that rocks the welded carton off level with no arm fault "
            f"(coldwin33-37 CARTON_TILT +2.06). Enable the compliant caster suspension "
            f"(spec caster_suspension + caster_spring_n_per_m/caster_preload_n) or the sim-only station "
            f"pitch hold (chassis_station_pitch_hold)."]
    return []


def check_support_height(runtime_cfg, scenario_cfg, *, resolved_top_z, tol_m=0.010):
    """sim-place-support-z-consistency-v1: for a collect-on-DECK run, refuse any configured place
    support height that disagrees with the resolved deck prim top by more than ``tol_m`` (10 mm).

    The cold ground-truth collect places the carton on the Rollwagen DECK
    (``/Rollwagen/chassis/deck``, top face ~0.2565 world). The place height is carried in two config
    layers that must agree with that prim top:
      * RUNTIME ``cart_placement.support_z_world`` (the legacy deck-interior constant path), and
      * SCENARIO ``initial_staging.cart_support_z_world`` -- which is ALSO the COLLECT_GROUND_TRUTH
        ``place_slot`` support_z_world (``ground_truth_collect.warm_slot`` reads this same field).

    Both historically carried the legacy SHELF constant 0.9325 (~0.68 m above the deck); the runtime
    resolver now overrides the GT slot onto the live deck top, but a stale shelf number left in either
    config could still feed a non-deck path. Name the drift here -- before a GPU boot -- so no stale
    height can feed the collect descent again. ``resolved_top_z`` is the deck prim's resolved top-face
    world z (the orchestrator computes it from the composed cart asset and passes ``--support-top-z-world``).
    No-op ([]) when neither config carries a support height (nothing to check)."""
    from twin.restock_manipulation import cart_support
    heights = {}
    cp = (runtime_cfg or {}).get('cart_placement') or {}
    if isinstance(cp, dict) and 'support_z_world' in cp:
        heights['runtime cart_placement.support_z_world'] = cp.get('support_z_world')
    st = (scenario_cfg or {}).get('initial_staging') or {}
    if isinstance(st, dict) and 'cart_support_z_world' in st:
        heights['scenario initial_staging.cart_support_z_world '
                '(= the COLLECT_GROUND_TRUTH place_slot support_z_world)'] = st.get('cart_support_z_world')
    if not heights:
        return []
    return cart_support.height_consistency_problems(resolved_top_z, heights, tol_m=tol_m)


def _load(path):
    with open(path) as stream:
        return json.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-config', required=True,
                        help='restock_manipulation/ros_host_config.json (read by ros_host)')
    parser.add_argument('--runtime-config', required=True,
                        help='mission-runtime-*.json (read by the Isaac side)')
    parser.add_argument('--require', default='',
                        help='comma-separated batch keys this run needs ACTIVE')
    parser.add_argument('--scenario', default=None,
                        help='mission-scenario.json: when given, refuse a host carton-profile dimension '
                             'that disagrees with handling_unit.dimensions_m (the PR 268 single source)')
    parser.add_argument('--mobile-manifest', default=None,
                        help='mobile-manifest.json: when given, refuse a chassis whose free pitch travel '
                             'exceeds the carton bound minus margin unless a chassis stabiliser is active '
                             '(sim-chassis-pitch-stability-v1)')
    parser.add_argument('--place-on-deck', action='store_true',
                        help='sim-place-support-z-consistency-v1: this is a collect-on-DECK run '
                             '(RESTOCK_PLACE_ON_DECK=1). With --support-top-z-world, refuse a configured '
                             'place support height (runtime cart_placement.support_z_world / scenario '
                             'initial_staging.cart_support_z_world = the GT place_slot) that disagrees with '
                             'the resolved deck prim top by more than 10 mm, so no stale shelf constant '
                             '(0.9325) can feed the collect descent.')
    parser.add_argument('--support-top-z-world', type=float, default=None,
                        help='sim-place-support-z-consistency-v1: the resolved support prim '
                             '(/Rollwagen/chassis/deck) TOP-face world z. Required with --place-on-deck.')
    args = parser.parse_args(argv)
    declared = [k.strip() for k in args.require.split(',') if k.strip()]
    # fm402: always assert the cap/limit baseline, even if a run forgets to --require it -- a missing
    # cap must never silently un-cap the arm. Dedupe while preserving order (baseline first).
    required = list(dict.fromkeys(list(BASELINE_REQUIRED) + declared))
    host_cfg = _load(args.host_config)
    runtime_cfg = _load(args.runtime_config)
    problems = check_activation(host_cfg, runtime_cfg, required)
    # sim-service-carry-drive-to-shelf-v1: cross-layer flow consistency (always, not just --require'd
    # keys) so a shelf_staging/transfer_service_carry split cannot reach a GPU boot.
    problems += check_flow_consistency(host_cfg, runtime_cfg)
    # sim-place-descent-residual-v1: resolve the HOST carton_level block (incl. the place-clearance keys)
    # so a malformed value or the gate-without-remedy combination fails pre-flight, not a GPU boot.
    problems += check_carton_level(host_cfg)
    # sim-collect-press-regression-v1: carton-dim drift between the HOST profile and the scenario single
    # source silently over-reaches the side-grasp pick standoff (coldwin32). Only when --scenario is given.
    if args.scenario:
        problems += check_carton_dims(host_cfg, _load(args.scenario))
    # sim-chassis-pitch-stability-v1: refuse an unsprung see-saw chassis unless a stabiliser is active.
    if args.mobile_manifest:
        problems += check_pitch_stability((_load(args.mobile_manifest) or {}).get('spec') or {},
                                          host_cfg, runtime_cfg)
    # sim-place-support-z-consistency-v1: a collect-on-deck run must carry place heights that agree with
    # the resolved deck prim top (no stale shelf constant lurking in scenario / runtime / GT heights).
    if args.place_on_deck:
        if args.support_top_z_world is None:
            problems.append('--place-on-deck requires --support-top-z-world <deck top-face world z> so the '
                            'preflight can refuse a stale place height (sim-place-support-z-consistency-v1).')
        else:
            scenario_cfg = _load(args.scenario) if args.scenario else {}
            problems += check_support_height(runtime_cfg, scenario_cfg,
                                             resolved_top_z=args.support_top_z_world)
    # Echo the effective placement so the log proves what the preflight saw.
    for key in required:
        consumer = KEY_CONSUMER.get(key, '?')
        present = key in (host_cfg if consumer == HOST else runtime_cfg if consumer == RUNTIME else {})
        print(f'CONFIG_PREFLIGHT: {key} -> {consumer} config: '
              f'{"PRESENT" if present else "ABSENT"}')
    if problems:
        print('CONFIG_PREFLIGHT: FAIL -- refusing GPU boot:', file=sys.stderr)
        for p in problems:
            print('  - ' + p, file=sys.stderr)
        return 2
    print(f'CONFIG_PREFLIGHT: PASS -- {len(required)} required key(s) load from the right config')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
