#!/usr/bin/env python3
"""fm390 offline proof: is the cart DECK interior reachable when the base drives IN FRONT of the
rollwagen's open (+x) face (place_approach), instead of reaching around from the collect station pose?

Actual-URDF CPU IK (reuses scripts/check_restock_reach.check), plus the two gates the live run also
applies: the controller's ``max_reach`` pose gate (ur_base-frame magnitude of the raised deck place
pose) and the drawbar clearance (the 1.0 m base, facing the mouth, has its west edge at bx-0.5, which
must stay EAST of the rollwagen drawbar tip at world x=-5.08). Prints the feasible place_approach
window. NOT a collision planner (collision_checked=False), NOT physics.

Run:  .venv-usd/bin/python -I scripts/check_place_approach_reach.py \
        --mobile-manifest <mobile_manifest.json> --scenario scripts/restock_scenario.json
"""
import argparse, copy, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> check_restock_reach, twin.*
from check_restock_reach import check


def feasible_window(manifest, scenario, *, cart_xy=(-5.73, 15.25), deck_z=0.225,
                    drawbar_tip_x=-5.08, max_reach=1.3, base_len=1.0, arm_z=0.5,
                    bxs=(-4.30, -4.40, -4.45, -4.50, -4.55, -4.58, -4.60)):
    carton_h = scenario['handling_unit']['dimensions_m'][2]
    rows = []
    for bx in bxs:
        s = copy.deepcopy(scenario); st = s['initial_staging']
        st['robot_base_pose'] = [bx, cart_xy[1], math.pi]        # east of cart, facing the mouth (-x)
        st['cart_body_pose'] = [cart_xy[0], cart_xy[1], 0.0]
        st['cart_support_z_world'] = deck_z
        r = check(manifest['mobile_urdf'], manifest, s)
        kin = all(r['poses']['cart_' + p]['reachable'] for p in ('contact', 'approach', 'lift', 'preplace'))
        west_edge = bx - base_len / 2.0
        drawbar_clr = west_edge - drawbar_tip_x
        place_mag = math.hypot(cart_xy[0] - bx, (deck_z - arm_z) + carton_h)   # what _pose_ok checks
        rows.append(dict(bx=bx, west_edge=round(west_edge, 3), drawbar_clr=round(drawbar_clr, 3),
                         place_pose_mag=round(place_mag, 3), within_reach_gate=place_mag <= max_reach,
                         kinematically_reachable=bool(kin),
                         feasible=bool(kin and place_mag <= max_reach and drawbar_clr > 0.0)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mobile-manifest', type=Path, required=True)
    p.add_argument('--scenario', type=Path, default=Path('scripts/restock_scenario.json'))
    a = p.parse_args()
    manifest = json.loads(a.mobile_manifest.read_text())
    scenario = json.loads(a.scenario.read_text())
    rows = feasible_window(manifest, scenario)
    print(json.dumps(rows, indent=2))
    feasible = [r for r in rows if r['feasible']]
    if feasible:
        print('FEASIBLE place_approach base x in [%.2f, %.2f] (y=%.2f, yaw=pi)'
              % (min(r['bx'] for r in feasible), max(r['bx'] for r in feasible), 15.25))
    else:
        print('NO feasible place_approach in the swept window')
    return 0 if feasible else 1


if __name__ == '__main__':
    raise SystemExit(main())
