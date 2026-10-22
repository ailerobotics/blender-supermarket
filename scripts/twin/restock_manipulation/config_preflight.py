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
    # --- RUNTIME config (isaac_bundle / isaac_port / physics_setup) ---
    'phase_drive': RUNTIME,
    'arm_stiffness_abs': RUNTIME,
    'arm_damping_abs': RUNTIME,
    'arm_max_effort': RUNTIME,
    'arm_gravity_ff': RUNTIME,
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
    'sim_speed': RUNTIME,        # sim-video-speedup-v1 speed profile; isaac_bundle/isaac_port + bt view read it
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
    args = parser.parse_args(argv)
    declared = [k.strip() for k in args.require.split(',') if k.strip()]
    # fm402: always assert the cap/limit baseline, even if a run forgets to --require it -- a missing
    # cap must never silently un-cap the arm. Dedupe while preserving order (baseline first).
    required = list(dict.fromkeys(list(BASELINE_REQUIRED) + declared))
    host_cfg = _load(args.host_config)
    runtime_cfg = _load(args.runtime_config)
    problems = check_activation(host_cfg, runtime_cfg, required)
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
