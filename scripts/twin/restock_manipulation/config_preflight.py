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

A key can also be mis-placed WITHIN a file, not just between files: the Isaac factory reads
the collect placement from the NESTED block `config['manipulation']['cart_placement']`, and
a TOP-LEVEL `config['cart_placement']` is silently ignored (coldwin42-47: five GPU runs on a
block the sim never read, while the pre-flight validated that same ignored block). So the
gate and the factory resolve the placement through ONE shared accessor (`config_access`), and
`check_cart_placement_source` refuses a stale top-level duplicate -- see
sim-preflight-nested-cart-placement-v1.
"""
from __future__ import annotations

import argparse
import json
import math
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
    # sim-arm-reactive-replan-v1 (P1): opt-in human-aware RUN/SLOW/STOP/BACKSTOP authority over the free-space
    # arm legs (ros_host -> TransferController). reactive_arm_human toggles it; reactive_arm is the optional
    # tuning dict (gate dwell/fractions, backstop params, replannable_phases). Absent/False => byte-identical.
    'reactive_arm_human': HOST,
    'reactive_arm': HOST,
    'carton_level': HOST,  # sim-carton-level-carry-v1: level the carried carton (fold goals) + CARTON_TILT trip; HOST (the controller drives the arm)
    # sim-preplace-per-joint-gains-v1: preplace frozen-joint convergence guard + wrist-first sequencing
    # fallback (ros_host -> TransferController). Both opt-in, default off => byte-identical.
    'preplace_convergence_stall_s': HOST,
    'preplace_wrist_first': HOST,
    # sim-preplace-leg2-level-v1: the held chassis attitude mirror (HOST keys for the carton-level FK so it
    # levels/measures the preplace in the WORLD frame) + the plan-derived phase budget + its margin.
    # station_pitch_hold/_pitch_deg/_roll_deg MIRROR the RUNTIME chassis_station_pitch_hold_* keys; absent =>
    # the base frame + the 120 s phase timeout (byte-identical).
    'station_pitch_hold': HOST,
    'station_hold_pitch_deg': HOST,
    'station_hold_roll_deg': HOST,
    'phase_timeout_s': HOST,
    'preplace_budget_margin_s': HOST,
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
    # sim-station-full-brake-place-v1: FULL-lock the base (translation + rotation) over the loaded
    # collect place descent so the welded from-above reaction cannot shove the translation-free base
    # (coldwin49). physics_setup.assemble reads them (RUNTIME) to build the MobileController. Default off
    # => byte-identical. check_station_full_brake refuses the brake without chassis_station_pitch_hold.
    'station_place_full_brake': RUNTIME,
    'station_place_full_brake_effort_limit': RUNTIME,
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
    ``closed_loop_gain``, the v3 keys ``place_contact_release`` / ``place_contact_gap_m`` /
    ``closed_loop_sensor``, and the sim-place-preplace-above-support-v1 keys ``preplace_above_support`` /
    ``preplace_support_clearance_m``, read from ros_host_config.json) FAILS the pre-flight loudly instead of at a GPU
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
    # sim-preplace-level-gate-v1: the preplace carton-level GATE only ENGAGES when the collect preplace
    # standoff is re-targeted above the support (_preplace_above_place_goal -> preplace_above_support /
    # preplace_above_place). A gate ON with neither above-flag is SILENTLY INERT (the host-vs-runtime /
    # key-in-the-wrong-place class of bug), so refuse it loudly at pre-flight rather than ship a run that
    # believes it is gated but drives the ungated single-goal approach.
    if getattr(resolved, 'preplace_level_gate', False) \
            and not (getattr(resolved, 'preplace_above_support', False) or resolved.preplace_above_place):
        problems.append(
            "carton_level 'preplace_level_gate' is ON but neither 'preplace_above_support' nor "
            "'preplace_above_place' is set: the gate engages only on the re-targeted above-support preplace "
            "standoff (_preplace_above_place_goal), so with neither it is silently inert. Set "
            "'preplace_above_support': true (the shelf-correct lane) so the droop-compensated, densified, "
            "speed/accel-capped carried approach actually drives the preplace.")
    # sim-preplace-approach-seeded-v1: the branch-seeded interior solve only changes the WORLD-level pre-place
    # approach (plan_level_world_approach anchor 'seeded'), which is driven only when the preplace level gate
    # engages. 'preplace_seeded_interior' ON with 'preplace_level_gate' OFF is silently inert (the same
    # key-in-the-wrong-place class), so refuse it loudly rather than ship a run that believes the interior is
    # seeded while it drives the ungated approach.
    if getattr(resolved, 'preplace_seeded_interior', False) \
            and not getattr(resolved, 'preplace_level_gate', False):
        problems.append(
            "carton_level 'preplace_seeded_interior' is ON but 'preplace_level_gate' is OFF: the branch-seeded "
            "interior solve only changes the world-level pre-place approach the gate drives, so with the gate "
            "off it is silently inert. Set 'preplace_level_gate': true (with 'preplace_above_support': true) "
            "so the seeded interior actually plans the preplace approach.")
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


def check_phase_drive_carried_soft(runtime_cfg):
    """sim-preplace-sweep-tilt-v1: validate the opt-in ``phase_drive.carried_soft_phases`` override.

    The RUNTIME ``phase_drive`` block (isaac_port / physics_setup) selects a STIFF arm drive profile for
    the loaded pan/preplace/place. On the carried PREPLACE approach the stiff profile swings the welded
    carton off level (coldwin50/51/52, CARTON_TILT 3-8 deg); the fix force-routes named carried phases
    back to the SOFT profile via ``phase_drive['carried_soft_phases']``. This refuses, before any GPU
    boot: a non-list value; a listed phase that is not one of physics_setup's default STIFF phases
    (``_PHASE_DRIVE_STIFF_PHASES`` -- forcing an already-soft phase soft is a typo, not a no-op worth
    shipping); and a ``carried_soft_phases`` set while no ``soft`` gain block is present (the override
    would select a missing profile). No-op ([]) when the override is absent (byte-identical)."""
    rc = runtime_cfg or {}
    pd = rc.get('phase_drive')
    if not isinstance(pd, dict):
        return []
    soft_override = pd.get('carried_soft_phases')
    if not soft_override:
        return []
    from twin.restock.physics_setup import _PHASE_DRIVE_STIFF_PHASES
    problems = []
    if isinstance(soft_override, str) or not isinstance(soft_override, (list, tuple)):
        problems.append('phase_drive.carried_soft_phases must be a list of phase names (e.g. '
                        '["preplace"]); got %r.' % (soft_override,))
        return problems
    bad = [p for p in soft_override if p not in _PHASE_DRIVE_STIFF_PHASES]
    if bad:
        problems.append('phase_drive.carried_soft_phases may only force-soft a phase that is otherwise '
                        'STIFF %s; unknown/non-stiff %s (a typo would silently drop the override).'
                        % (list(_PHASE_DRIVE_STIFF_PHASES), bad))
    if not pd.get('soft'):
        problems.append('phase_drive.carried_soft_phases is set but phase_drive has no "soft" gain '
                        'block for the override to select.')
    return problems


def check_phase_drive_carried_joint_profile(runtime_cfg):
    """sim-preplace-per-joint-gains-v1: validate the opt-in ``phase_drive.carried_joint_profile`` per-joint
    drive override.

    The carried PREPLACE approach runs SOFT (``carried_soft_phases`` kills the stiff-drive carton swing),
    but under the soft eff 400 ceiling shoulder_pan cannot HOLD its goal against the 1.64 rad wrist_2 sweep
    reaction (coldwin53 drifted it 0.56 rad -> phase timeout). The per-joint override force-stiffens the
    HOLD joints (shoulder_pan, and optionally shoulder_lift / elbow) while the tilt-injecting WRISTS stay
    soft. This refuses, before any GPU boot:
      * a non-mapping value, an unknown phase (not a default STIFF phase), or an unknown arm joint name;
      * a malformed / non-positive gain block;
      * a WRIST override on a carried-SOFT phase that would RE-STIFFEN the wrist above the soft eff ceiling
        (that reintroduces the swing the soft profile was chosen to kill -- the brief's explicit guard).
    No-op ([]) when the override is absent (byte-identical)."""
    rc = runtime_cfg or {}
    pd = rc.get('phase_drive')
    if not isinstance(pd, dict):
        return []
    prof = pd.get('carried_joint_profile')
    if not prof:
        return []
    from twin.restock.physics_setup import (_PHASE_DRIVE_STIFF_PHASES, _WRIST_JOINT_NAMES,
                                            resolve_joint_overrides, resolve_stiff_phases)
    problems = []
    if not isinstance(prof, dict):
        return ['phase_drive.carried_joint_profile must be a mapping {phase: {joint_name: {kp,kd,eff}}}; '
                'got %r.' % (prof,)]
    soft_eff = None
    if isinstance(pd.get('soft'), dict):
        try:
            soft_eff = float(pd['soft']['eff'])
        except (KeyError, TypeError, ValueError):
            soft_eff = None
    carried_soft = set()
    try:
        carried_soft = set(_PHASE_DRIVE_STIFF_PHASES) - set(resolve_stiff_phases(pd))
    except ValueError:
        pass  # a bad carried_soft_phases is reported by check_phase_drive_carried_soft
    for phase in prof:
        if phase not in _PHASE_DRIVE_STIFF_PHASES:
            problems.append('phase_drive.carried_joint_profile has unknown phase %r; it may only override a '
                            'default STIFF phase %s.' % (phase, list(_PHASE_DRIVE_STIFF_PHASES)))
            continue
        try:
            overrides = resolve_joint_overrides(phase, pd)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        # The swing guard: on a phase forced SOFT (carried_soft_phases), a wrist override must keep the
        # wrist at or below the soft effort ceiling, or the swing the soft profile killed returns.
        if phase in carried_soft and soft_eff is not None:
            for jn in _WRIST_JOINT_NAMES:
                g = overrides.get(jn)
                if g is not None and g['eff'] > soft_eff + 1e-9:
                    problems.append(
                        'phase_drive.carried_joint_profile[%r][%r] raises the wrist effort to %g N*m above '
                        'the soft ceiling %g N*m: a carried-soft phase must keep the tilt-injecting wrists '
                        'SOFT (re-stiffening a wrist reintroduces the carton swing the soft profile kills). '
                        'Stiffen shoulder_pan (and optionally shoulder_lift / elbow), not the wrists.'
                        % (phase, jn, g['eff'], soft_eff))
    return problems


def check_preplace_convergence_keys(host_cfg):
    """sim-preplace-per-joint-gains-v1: validate the opt-in HOST preplace convergence guard + sequencing
    fallback keys (read by ros_host -> TransferController).

      * ``preplace_convergence_stall_s`` (SIM seconds): a frozen-joint convergence guard that FAILS the
        preplace by name (PHASE_CONVERGENCE) when the goal error stops shrinking, instead of running the
        coarse 120 s phase timeout. Must be a finite non-negative number (0 => off). It is a TIGHTER bound
        than the phase timeout, so a value >= 120 is a no-op-with-a-warning-shape -> refuse it as a slip.
      * ``preplace_wrist_first`` (bool): sequence the carried preplace so the wrists sweep first (at the
        low-reaction tucked pose) and the HOLD joints move with the wrists already set. Must be a bool.
    No-op ([]) when both are absent (byte-identical)."""
    hc = host_cfg or {}
    problems = []
    stall = hc.get('preplace_convergence_stall_s')
    if stall is not None:
        try:
            v = float(stall)
        except (TypeError, ValueError):
            problems.append('preplace_convergence_stall_s must be a number (SIM seconds; 0 => off); got %r.'
                            % (stall,))
        else:
            if not math.isfinite(v) or v < 0:
                problems.append('preplace_convergence_stall_s must be a finite non-negative number '
                                '(SIM seconds; 0 => off); got %r.' % (stall,))
            elif v >= 120.0:
                problems.append('preplace_convergence_stall_s=%g is >= the 120 s phase timeout, so it would '
                                'never fire before the coarse timeout it is meant to pre-empt. Use a tighter '
                                'sim-second bound (e.g. 25-40 s).' % (v,))
    wf = hc.get('preplace_wrist_first')
    if wf is not None and not isinstance(wf, bool):
        problems.append('preplace_wrist_first must be a boolean; got %r.' % (wf,))
    return problems


def check_preplace_phase_budget(host_cfg):
    """sim-preplace-leg2-level-v1 (part 2): validate the held-chassis mirror keys + the plan-derived preplace
    phase budget, and REPORT the computed budget (PREPLACE_PHASE_BUDGET).

    When ``preplace_wrist_first`` is on the two-leg sequence needs ~150 s -- over the legacy 120 s -- so the
    phase timeout is derived from the planned leg durations at the active caps + ``preplace_budget_margin_s``
    (:func:`carton_level.preplace_phase_budget`). This reports the computed budget and REFUSES when an explicit
    ``phase_timeout_s`` is set BELOW the planned preplace duration (a plan that would time out). It also
    validates ``station_pitch_hold`` / ``station_hold_pitch_deg`` / ``station_hold_roll_deg`` (the HOST mirror
    of the RUNTIME chassis_station_pitch_hold_* keys used by the world-frame leveling). No-op ([]) when
    wrist-first is off AND no station-hold keys are set (byte-identical)."""
    from twin.restock_manipulation import carton_level
    hc = host_cfg or {}
    problems = []
    sph = hc.get('station_pitch_hold')
    if sph is not None and not isinstance(sph, bool):
        problems.append('station_pitch_hold must be a boolean (mirrors the RUNTIME chassis_station_pitch_hold); got %r.' % (sph,))
    for nm in ('station_hold_pitch_deg', 'station_hold_roll_deg'):
        v = hc.get(nm)
        if v is not None:
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    raise ValueError
            except (TypeError, ValueError):
                problems.append('%s must be a finite number of degrees (mirrors the RUNTIME '
                                'chassis_station_pitch_hold value); got %r.' % (nm, v))
    margin = hc.get('preplace_budget_margin_s')
    if margin is not None:
        try:
            if not (math.isfinite(float(margin)) and float(margin) >= 0):
                raise ValueError
        except (TypeError, ValueError):
            problems.append('preplace_budget_margin_s must be a finite non-negative number of seconds; got %r.' % (margin,))
    explicit = hc.get('phase_timeout_s')
    if explicit is not None:
        try:
            if not (math.isfinite(float(explicit)) and float(explicit) > 0):
                raise ValueError
        except (TypeError, ValueError):
            problems.append('phase_timeout_s must be a finite positive number of seconds; got %r.' % (explicit,))
            explicit = None
    if not bool(hc.get('preplace_wrist_first', False)):
        return problems
    carry = hc.get('carry_lift_shape'); standoff = hc.get('collect_place_preplace_shape')
    lc = None
    try:
        lc = carton_level.resolve_level_config(hc.get('carton_level'))
    except ValueError as exc:
        problems.append('carton_level does not resolve for the preplace budget: %s' % exc)
    if lc is not None and carry is not None and standoff is not None:
        try:
            m = float(margin) if margin is not None else 90.0
            budget, plan_s, l1, l2 = carton_level.preplace_phase_budget(
                carry, standoff, speed=lc.preplace_approach_speed, accel=lc.preplace_approach_accel,
                max_segment_rad=lc.preplace_max_segment_rad, margin_s=m)
            eff = float(explicit) if explicit is not None else budget
            print('CONFIG_PREFLIGHT: PREPLACE_PHASE_BUDGET wrist_first plan=%.1f s '
                  '(leg1=%.1f leg2=%.1f) margin=%.1f -> derived budget=%.1f s; effective phase_timeout=%.1f s'
                  % (plan_s, l1, l2, m, budget, eff))
            # sim-preplace-wrist-continuity-v1: report the wrist-continuity threshold + the reaction-natural
            # branch selector (the coldwin57 leg-2 wrist_2-divergence fix). The actual per-waypoint max wrist
            # step is measured at runtime by the level gate (CARTON_LEVEL_PREPLACE_GATE max_wrist_step=...),
            # which needs the FK model; the preflight reports the configured threshold + the selector state.
            print('CONFIG_PREFLIGHT: PREPLACE_WRIST_CONTINUITY max_wrist_step<=%.3f rad; reaction_branch=%s '
                  '(min-|wrist_2| settling-branch standoff; soft-wrist effort %.0f N*m for the reported '
                  'margin; branch-select CPU budget=%.1f s off the host hot path -- coldwin59)'
                  % (lc.preplace_max_wrist_step_rad,
                     'ON' if getattr(lc, 'preplace_reaction_branch', False) else 'off',
                     lc.preplace_soft_effort_nm,
                     getattr(lc, 'preplace_branch_select_budget_s', 2.0)))
            if eff < plan_s:
                problems.append('phase_timeout_s=%.1f is BELOW the planned wrist-first preplace duration '
                                '%.1f s (leg1=%.1f + leg2=%.1f) -- the preplace would time out before leg 2 '
                                'finishes. Raise it to >= plan + margin (derived budget %.1f s) or omit it to '
                                'use the derived budget.' % (eff, plan_s, l1, l2, budget))
        except (ValueError, TypeError) as exc:
            problems.append('could not derive the wrist-first preplace budget from carry_lift_shape / '
                            'collect_place_preplace_shape: %s' % exc)
    return problems


def check_station_full_brake(runtime_cfg):
    """sim-station-full-brake-place-v1: validate the opt-in loaded-place full base brake.

    The full D6 lock anchors its rotation at the station pitch/roll hold TARGET
    (``chassis_station_pitch_hold_pitch_deg`` / ``_roll_deg``) and is layered on the rotation-only
    station hold, so ``station_place_full_brake`` REQUIRES ``chassis_station_pitch_hold`` -- otherwise the
    brake would have no target to anchor against (a key-in-the-wrong-place / split-config slip, like
    run-277). Also refuse a non-positive / non-finite ``station_place_full_brake_effort_limit`` (the
    weld-loop effort guard limit, N·m). No-op ([]) when the full brake is off (byte-identical)."""
    rc = runtime_cfg or {}
    if not rc.get('station_place_full_brake'):
        return []
    problems = []
    if not rc.get('chassis_station_pitch_hold'):
        problems.append(
            'station_place_full_brake is set but chassis_station_pitch_hold is NOT: the loaded-place '
            'full D6 base lock anchors its rotation at the pitch/roll hold TARGET and is layered on the '
            'rotation-only station hold, so the station pitch hold must be active. Enable '
            'chassis_station_pitch_hold (+ chassis_station_pitch_hold_pitch_deg / _roll_deg) in the same '
            'RUNTIME config.')
    lim = rc.get('station_place_full_brake_effort_limit')
    if lim is not None:
        try:
            f = float(lim)
        except (TypeError, ValueError):
            f = None
        if f is None or not (f > 0 and math.isfinite(f)):
            problems.append('station_place_full_brake_effort_limit must be a positive finite N·m (the '
                            'weld-loop effort guard limit); got %r.' % (lim,))
    return problems


def check_support_height(runtime_cfg, scenario_cfg, *, resolved_top_z, tol_m=0.010):
    """sim-place-support-z-consistency-v1: for a collect-on-DECK run, refuse any configured place
    support height that disagrees with the resolved deck prim top by more than ``tol_m`` (10 mm).

    The cold ground-truth collect places the carton on the Rollwagen DECK
    (``/Rollwagen/chassis/deck``, top face ~0.2565 world). The place height is carried in two config
    layers that must agree with that prim top:
      * RUNTIME ``manipulation.cart_placement.support_z_world`` (the NESTED block the factory reads,
        via ``config_access``; the legacy deck-interior constant path), and
      * SCENARIO ``initial_staging.cart_support_z_world`` -- which is ALSO the COLLECT_GROUND_TRUTH
        ``place_slot`` support_z_world (``ground_truth_collect.warm_slot`` reads this same field).

    Both historically carried the legacy SHELF constant 0.9325 (~0.68 m above the deck); the runtime
    resolver now overrides the GT slot onto the live deck top, but a stale shelf number left in either
    config could still feed a non-deck path. Name the drift here -- before a GPU boot -- so no stale
    height can feed the collect descent again. ``resolved_top_z`` is the deck prim's resolved top-face
    world z (the orchestrator computes it from the composed cart asset and passes ``--support-top-z-world``).
    No-op ([]) when neither config carries a support height (nothing to check)."""
    from twin.restock_manipulation import cart_support, config_access
    heights = {}
    # Read the SAME nested block the Isaac factory reads (config['manipulation']['cart_placement']),
    # never the ignored top-level block -- see check_cart_placement_source / config_access.
    cp = config_access.cart_placement_block(runtime_cfg) or {}
    if isinstance(cp, dict) and 'support_z_world' in cp:
        heights['runtime manipulation.cart_placement.support_z_world'] = cp.get('support_z_world')
    st = (scenario_cfg or {}).get('initial_staging') or {}
    if isinstance(st, dict) and 'cart_support_z_world' in st:
        heights['scenario initial_staging.cart_support_z_world '
                '(= the COLLECT_GROUND_TRUTH place_slot support_z_world)'] = st.get('cart_support_z_world')
    if not heights:
        return []
    return cart_support.height_consistency_problems(resolved_top_z, heights, tol_m=tol_m)


def _resolve_stock_supports(stock_supports):
    """The cart's stock_supports list (``[deck, shelf]``), from the given override or authored in-memory
    (no Isaac). None when usd-core is absent so the caller can skip the membership check."""
    if stock_supports:
        return list(stock_supports)
    try:
        from pxr import Usd
        from twin.mobile import MobileSpec
        from twin.mobile_assets import author_cart
        stage = Usd.Stage.CreateInMemory()
        return list(author_cart(stage, spec=MobileSpec())['stock_supports'])
    except Exception:                               # noqa: BLE001 -- usd-core absent; caller skips membership
        return None


def check_place_preplace_standoff(host_cfg, runtime_cfg, *, resolved_top_z=None, stock_supports=None):
    """sim-place-preplace-above-support-v1: refuse the coldwin48 config -- a NON-deck place support selected
    while the collect pre-place standoff is still the DECK-tuned fixed shape (no ``preplace_above_support``).

    The RUNTIME selects the place support (``manipulation.cart_placement.support_prim``). When it is the
    DECK default (absent / ``stock_supports[0]``) the fixed ``collect_place_preplace_shape`` is correct
    (byte-identical). When it is a NON-deck support (the UPPER shelf), the deck-tuned fixed standoff parks
    the carried carton bottom BELOW the higher support top, so ``place_descent_drop_m`` reads 0.0, the
    contact-triggered weld release fires in mid-air and the carton falls (coldwin48: verify_place 'carton
    not resting', bottom_gap -0.809). The HOST ``carton_level.preplace_above_support`` is the fix (derive
    the standoff from the resolved support top + a clearance). Refuse the mismatch here, before a GPU boot.

    When ``preplace_above_support`` IS on and ``resolved_top_z`` is given, also compute + report the
    standoff carton-bottom height and the descent length and refuse a non-positive descent or an
    above-reach standoff (``cart_support.place_descent_problems``). No-op ([]) when no support prim is
    selected. Reads the NESTED cart_placement block the factory reads (``config_access``)."""
    from twin.restock_manipulation import cart_support, config_access, carton_level as cl
    cp = config_access.cart_placement_block(runtime_cfg) or {}
    if not isinstance(cp, dict) or 'support_prim' not in cp:
        return []                                    # deck default -> fixed shape is correct (byte-identical)
    supports = _resolve_stock_supports(stock_supports)
    if supports is None:
        return []                                    # usd-core absent: check_support_selection owns the shape
    try:
        resolved = cart_support.resolve_support_prim(supports, cp.get('support_prim'))
    except ValueError:
        return []                                    # bad selector -> check_support_selection names it
    if resolved == supports[0]:
        return []                                    # the DECK default surface -> fixed shape is correct
    # A NON-deck support (e.g. the UPPER shelf) is selected: the fixed deck-tuned standoff is wrong.
    try:
        lc = cl.resolve_level_config((host_cfg or {}).get('carton_level'))
    except ValueError:
        return []                                    # malformed -> check_carton_level names it
    if lc is None or not getattr(lc, 'preplace_above_support', False):
        return [
            f"collect place standoff MIS-CONFIGURED for the selected support: RUNTIME "
            f"cart_placement.support_prim={cp.get('support_prim')!r} selects {resolved} (a NON-deck "
            f"support), but the HOST carton_level 'preplace_above_support' is OFF, so the DECK-tuned fixed "
            f"collect_place_preplace_shape is used. That shape parks the carried carton BELOW the higher "
            f"support top, so place_descent_drop_m reads 0.0 and the contact-triggered weld release fires "
            f"in mid-air (coldwin48: verify_place 'carton not resting', bottom_gap -0.809). Set HOST "
            f"carton_level 'preplace_above_support': true (ros_host_config.json) so the standoff is derived "
            f"from the resolved support top + the clearance."]
    # preplace_above_support ON: report + geometry-check the standoff when the resolved top is available.
    if resolved_top_z is None:
        return []
    clearance = lc.preplace_support_clearance()
    standoff_bottom, descent = cart_support.place_descent_from_support(resolved_top_z, clearance)
    print('CONFIG_PREFLIGHT: collect place preplace-above-support -> support %s top=%.4f m, clearance=%.4f '
          'm => standoff carton bottom=%.4f m, descent length=%.4f m'
          % (resolved, float(resolved_top_z), float(clearance), standoff_bottom, descent))
    return cart_support.place_descent_problems(resolved_top_z, clearance)


def check_cart_placement_source(runtime_cfg):
    """sim-preflight-nested-cart-placement-v1: refuse a STALE top-level ``cart_placement`` duplicate.

    The Isaac factory reads the collect placement from the NESTED block
    ``config['manipulation']['cart_placement']`` (via ``config_access``); a TOP-LEVEL
    ``config['cart_placement']`` is IGNORED. coldwin42-47 edited the top-level block, so the
    factory kept using the stale nested block (coldwin47: ``PLACE_SUPPORT_SELECT selector=None``,
    placed on the deck though the shelf was configured) and five GPU runs ran on settings the sim
    never read -- and the pre-flight passed because it too validated the top-level block.

    Refuse when a top-level ``cart_placement`` is present AND does not match the nested block the
    factory reads (a divergent or orphan duplicate). An IDENTICAL top-level duplicate is harmless
    (the factory reads the nested block, which equals it) and passes -- byte-identical. The message
    names BOTH blocks: the one that is IGNORED and the one that is READ. No-op ([]) when there is no
    top-level ``cart_placement`` (the correct, nested-only shape)."""
    from twin.restock_manipulation import config_access
    top = config_access.top_level_cart_placement(runtime_cfg)
    if top is None:
        return []
    nested = config_access.cart_placement_block(runtime_cfg)
    if nested == top:
        return []                                    # harmless identical duplicate (coldwin48)
    nested_state = ('ABSENT' if nested is None else f'STALE/DIFFERENT -> {nested!r}')
    return [
        f"cart_placement MIS-PLACED across blocks: a top-level 'cart_placement' "
        f"({top!r}) is present in the RUNTIME config, but the Isaac factory "
        f"(isaac_factory.parse_manipulation_config) reads '{config_access.CART_PLACEMENT_SOURCE}' "
        f"-- the NESTED block, which is {nested_state}. The top-level block is IGNORED "
        f"(coldwin42-47: five GPU runs on settings the sim never read; coldwin47 logged "
        f"'PLACE_SUPPORT_SELECT selector=None support_z 0.9325' and placed on the deck though the "
        f"shelf was configured). Move the placement into '{config_access.CART_PLACEMENT_SOURCE}' "
        f"(or make the two blocks identical)."]


def check_support_selection(runtime_cfg, *, stock_supports=None):
    """sim-cart-support-select-v1: validate the RUNTIME ``cart_placement.support_prim`` selector resolves
    against the composed cart's ``stock_supports`` so a mis-set selector (bad index, unknown prim/name)
    fails pre-flight, not at a GPU boot. ``stock_supports`` defaults to the names ``author_cart`` exposes
    (``/Rollwagen/chassis/deck``, ``/Rollwagen/chassis/shelf``) -- a pure in-memory authoring, no Isaac.
    No-op ([]) when the selector is absent (legacy deck default). Returns problems ([] == ok).

    Reads the SAME nested block the Isaac factory reads (``config['manipulation']['cart_placement']``
    via ``config_access``), never the ignored top-level block, so the gate validates the selector the
    factory actually uses (sim-preflight-nested-cart-placement-v1)."""
    from twin.restock_manipulation import cart_support, config_access
    cp = config_access.cart_placement_block(runtime_cfg) or {}
    if not isinstance(cp, dict) or 'support_prim' not in cp:
        return []
    supports = list(stock_supports) if stock_supports else None
    if supports is None:
        try:
            from pxr import Usd
            from twin.mobile import MobileSpec
            from twin.mobile_assets import author_cart
            stage = Usd.Stage.CreateInMemory()
            supports = list(author_cart(stage, spec=MobileSpec())['stock_supports'])
        except Exception as exc:                 # usd-core absent: validate shape only, not membership
            return ([] if isinstance(cp.get('support_prim'), (int, str))
                    and not isinstance(cp.get('support_prim'), bool)
                    else [f"cart_placement.support_prim={cp.get('support_prim')!r} must be a prim path, "
                          f"a prim name, or an index (usd-core absent: {type(exc).__name__})"])
    try:
        resolved = cart_support.resolve_support_prim(supports, cp.get('support_prim'))
    except ValueError as exc:
        return [f'cart_placement.support_prim invalid: {exc}']
    print(f'CONFIG_PREFLIGHT: cart_placement.support_prim={cp.get("support_prim")!r} -> {resolved}')
    return []


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
    # sim-preflight-nested-cart-placement-v1: refuse a stale top-level cart_placement duplicate (the
    # coldwin42-47 slip -- the factory reads manipulation.cart_placement, the top-level is ignored).
    problems += check_cart_placement_source(runtime_cfg)
    # sim-cart-support-select-v1: a mis-set place-support selector (bad index / unknown prim) must fail
    # here, not silently at the GPU boot (always, not just --require'd). Reads the nested block (the one
    # the factory reads) via config_access.
    problems += check_support_selection(runtime_cfg)
    # sim-place-preplace-above-support-v1: a NON-deck place support selected while the collect pre-place
    # standoff is still the deck-tuned fixed shape (no preplace_above_support) reproduces coldwin48 (the
    # from-above release fires in mid-air). With --support-top-z-world it also reports + checks the
    # standoff/descent geometry. Always on (not just --require'd); reads the nested cart_placement block.
    problems += check_place_preplace_standoff(host_cfg, runtime_cfg,
                                              resolved_top_z=args.support_top_z_world)
    # sim-collect-press-regression-v1: carton-dim drift between the HOST profile and the scenario single
    # source silently over-reaches the side-grasp pick standoff (coldwin32). Only when --scenario is given.
    if args.scenario:
        problems += check_carton_dims(host_cfg, _load(args.scenario))
    # sim-station-full-brake-place-v1: the opt-in loaded-place full base brake requires the station
    # pitch hold (it anchors at that target) + a positive effort-guard limit. Always on (not --require'd);
    # no-op when the key is off.
    problems += check_station_full_brake(runtime_cfg)
    # sim-preplace-sweep-tilt-v1: validate the opt-in phase_drive.carried_soft_phases override (run the
    # carried preplace approach on the SOFT profile). Always on (not --require'd); no-op when absent.
    problems += check_phase_drive_carried_soft(runtime_cfg)
    # sim-preplace-per-joint-gains-v1: validate the opt-in phase_drive.carried_joint_profile per-joint drive
    # override (stiffen shoulder_pan so it holds against the wrist sweep; keep the tilt-injecting wrists
    # soft). Always on (not --require'd); no-op when absent.
    problems += check_phase_drive_carried_joint_profile(runtime_cfg)
    # sim-preplace-per-joint-gains-v1: validate the opt-in HOST preplace convergence/sequencing keys.
    problems += check_preplace_convergence_keys(host_cfg)
    # sim-preplace-leg2-level-v1: validate the held-chassis mirror keys + REPORT the plan-derived preplace
    # phase budget; refuse an explicit phase_timeout_s below the planned preplace duration.
    problems += check_preplace_phase_budget(host_cfg)
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
