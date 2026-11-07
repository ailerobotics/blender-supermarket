#!/usr/bin/env python3
"""CPU geometry proof (sim-place-seat-geometry-v1): establish the PHYSICAL cart/shelf set-down seat from the
simulation geometry itself, and classify each recorded run's set-down against it -- the evidence the brief
asks for before any GPU boot, reproducible with usd-core + the run's own nav-trace.

No Isaac / ROS / GPU. Uses the composed ``mission.usda`` (usd-core, run it with the ``.venv-usd`` python) for
the slab FLAT-TOP vs world-AABB-CORNER geometry + the carton half-height, and the run's ``nav-trace.jsonl``
(carton WORLD poses) for the measured rested / stalled carton height. See
``scripts/twin/restock_manipulation/place_seat_geometry.py`` for the derivation and the shared
``classify_set_down`` predicate.

    .venv-usd/bin/python scripts/check_place_seat_geometry.py \
        /home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin87 [more run dirs...] [--json OUT]

Exit 0 always (a measurement, not a gate). Prints, per run: the derived shelf flat-top (true seat) vs the
model's AABB-corner support top, the carton half-height, and whether the run's set-down is SEATED (at the
seat), OBSTRUCTED (a contact well above the seat -- the coldwin87 14 cm stall) or a fall.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'twin'))

from twin.restock_manipulation import place_seat_geometry as psg  # noqa: E402

# The resolved MODEL support top the runs used (COLLECT_PLACE_SUPPORT_Z, world): the tilted shelf's AABB
# corner at the un-settled build chassis. The brief's "model says the seat is here".
MODEL_SUPPORT_TOP_WORLD = 0.9707
CARTON_H_M = 0.18
# The TRUE physical seat: the shelf FLAT TOP of the SETTLED cart, i.e. the carton BOTTOM world z where a
# RELEASED carton actually comes to rest. Measured directly from coldwin83 (released + rested + verify
# PASSED): carton final centre 1.0225 world -> bottom 0.9325 = settled shelf flat top. It is 0.0314 m below
# the build-pose flat top (0.9639) -- the dynamic cart settles ~3.1 cm (chassis 0.2315 -> ~0.20) -- and
# 0.0382 m below the model support top 0.9707 (that is the +7.4 mm AABB corner + the +31.4 mm settling).
TRUE_SETTLED_SUPPORT_TOP_WORLD = 0.9325


def _carton_world_z_stats(nav_trace_path, lo_t=None, hi_t=None, tail=1500):
    """From a nav-trace.jsonl, the carton WORLD centre z: (start, min, final) over the window [lo_t, hi_t]
    (world_t), and (min,max) over the final ``tail`` rows. The nav-trace 'carton' field is [x,y,z,yaw_deg]
    in WORLD; z is the carton CENTRE."""
    rows = []
    with open(nav_trace_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            c = d.get('carton')
            wt = d.get('world_t')
            if not c or wt is None:
                continue
            if lo_t is not None and wt < lo_t:
                continue
            if hi_t is not None and wt > hi_t:
                continue
            rows.append((float(wt), float(c[2]), float(c[0]), float(c[1])))
    if not rows:
        return None
    zs = [r[1] for r in rows]
    imin = min(range(len(rows)), key=lambda i: rows[i][1])
    tail_zs = zs[-tail:]
    return {
        'start_z': zs[0], 'min_z': min(zs), 'final_z': zs[-1],
        'min_at_t': rows[imin][0], 'final_t': rows[-1][0],
        'final_xy': (rows[-1][2], rows[-1][3]),
        'tail_min': min(tail_zs), 'tail_max': max(tail_zs), 'n': len(rows),
    }


def _place_start_sim_time(run_dir):
    """The sim_time of the 'place' stage_step from mission.json, so the carton-height stats cover the PLACE
    descent (not the low source-pick phase). None if unavailable."""
    mj = os.path.join(run_dir, 'mission.json')
    if not os.path.exists(mj):
        return None
    try:
        with open(mj) as f:
            data = json.load(f)
    except (ValueError, OSError):
        return None
    for ev in data.get('stage_timeline', []):
        d = ev.get('data', {})
        if d.get('label') == 'place' or d.get('step') == 5:
            return float(d.get('sim_time', 0.0))
    return None


def analyse_run(run_dir, *, expect=None):
    """Derive the seat geometry from the run's mission.usda + its nav-trace, and classify the set-down."""
    usd = os.path.join(run_dir, 'mission.usda')
    nav = os.path.join(run_dir, 'nav-trace.jsonl')
    name = os.path.basename(os.path.normpath(run_dir))
    out = {'run': name, 'expect': expect}
    if not os.path.exists(usd):
        out['error'] = f'no mission.usda in {run_dir}'
        return out
    der = psg.derive_from_stage(usd)
    out['shelf_flat_top_world'] = round(der.shelf_flat_top_z, 4)
    out['shelf_aabb_corner_world'] = round(der.shelf_aabb_top_z, 4)
    out['aabb_corner_inflation_mm'] = round(der.aabb_corner_inflation_m * 1000, 1)
    out['carton_half_m'] = round(der.carton_half_m, 4)
    out['ur_base_z_build'] = round(der.ur_base_z, 4)
    out['chassis_z_build'] = round(der.chassis_z, 4)
    # The MODEL seat (what the runs used): AABB corner + half. The GEOMETRIC seat (flat top of the composed,
    # un-settled cart) + half. The TRUE seat comes from a verified rest (nav-trace), below.
    out['model_support_top_world'] = MODEL_SUPPORT_TOP_WORLD
    out['model_seat_center_world'] = round(MODEL_SUPPORT_TOP_WORLD + der.carton_half_m, 4)
    out['geom_flat_seat_center_world'] = round(der.shelf_flat_top_z + der.carton_half_m, 4)
    # TRUE seat model (settled flat top): the honest seat the release must fire at.
    true_seat = psg.seat_from_support_top(TRUE_SETTLED_SUPPORT_TOP_WORLD, CARTON_H_M)
    out['true_support_top_world'] = TRUE_SETTLED_SUPPORT_TOP_WORLD
    out['true_seat_center_world'] = round(true_seat.seat_center_z, 4)
    out['model_error_mm'] = round((MODEL_SUPPORT_TOP_WORLD - TRUE_SETTLED_SUPPORT_TOP_WORLD) * 1000, 1)
    if os.path.exists(nav):
        place_start = _place_start_sim_time(run_dir)
        stats = _carton_world_z_stats(nav, lo_t=place_start)
        out['place_start_sim_t'] = place_start
        if stats:
            out['carton_final_center_world'] = round(stats['final_z'], 4)
            out['carton_final_bottom_world'] = round(stats['final_z'] - der.carton_half_m, 4)
            out['carton_min_center_world'] = round(stats['min_z'], 4)
            out['carton_final_xy'] = tuple(round(v, 3) for v in stats['final_xy'])
            # End-state verdict against the TRUE seat. The carton BOTTOM final gap:
            #   |gap| <= tol               -> SEATED   (rested on the shelf; cw83)
            #   gap >  tol                 -> STUCK ABOVE SEAT (an OBSTRUCTION -- the arm could not set it
            #                                 down; releasing here drops the carton; cw87/cw89)
            #   gap <  -fell_off threshold -> FELL OFF (released too high and swept/fell off the shelf;
            #                                 cw85/86/88)
            final_gap = true_seat.carton_bottom_z(stats['final_z']) - TRUE_SETTLED_SUPPORT_TOP_WORLD
            out['final_bottom_gap_vs_true_seat_mm'] = round(final_gap * 1000, 1)
            tol = psg.DEFAULT_SEAT_TOL_M
            if abs(final_gap) <= tol:
                out['end_state'] = 'SEATED'
            elif final_gap > tol:
                out['end_state'] = 'STUCK_ABOVE_SEAT (obstruction)'
            else:
                out['end_state'] = 'FELL_OFF'
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+', help='mission run directories (each with mission.usda + nav-trace.jsonl)')
    ap.add_argument('--json', help='write the full derivation as JSON here')
    args = ap.parse_args(argv)

    results = [analyse_run(d) for d in args.run_dirs]
    print('=' * 100)
    print('PLACE SEAT GEOMETRY (sim-place-seat-geometry-v1)')
    print('=' * 100)
    for r in results:
        print(f"\n--- {r['run']} ---")
        if 'error' in r:
            print('  ', r['error'])
            continue
        print(f"  shelf FLAT TOP (true support face, build pose) = {r['shelf_flat_top_world']} world")
        print(f"  shelf AABB CORNER (model over-read)           = {r['shelf_aabb_corner_world']} world "
              f"(+{r['aabb_corner_inflation_mm']} mm tilt inflation over the flat face)")
        print(f"  model support top (COLLECT_PLACE_SUPPORT_Z)   = {r['model_support_top_world']} world")
        print(f"  carton half-height                            = {r['carton_half_m']} m  (bottom = centre - half)")
        print(f"  ur_base_link z (build)                        = {r['ur_base_z_build']} world")
        print(f"  TRUE seat (settled flat top, from cw83 rest)  = {r['true_support_top_world']} world  "
              f"(model is +{r['model_error_mm']} mm too high)")
        if 'carton_final_center_world' in r:
            print(f"  MEASURED carton final: centre {r['carton_final_center_world']} / bottom "
                  f"{r['carton_final_bottom_world']} world  xy={r['carton_final_xy']}  (min centre "
                  f"{r['carton_min_center_world']})")
            print(f"  --> final carton bottom = {r['final_bottom_gap_vs_true_seat_mm']:+} mm vs the true "
                  f"seat  => END STATE = {r['end_state']}")
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
