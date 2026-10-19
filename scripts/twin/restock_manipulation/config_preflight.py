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
}


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
    required = [k.strip() for k in args.require.split(',') if k.strip()]
    if not required:
        print('CONFIG_PREFLIGHT: no --require keys declared; nothing to assert')
        return 0
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
