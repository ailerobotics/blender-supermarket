#!/usr/bin/env python3
"""Build the compact committed fixture for test_outbound_heading_hold_bound.py from the real
coldwin17 (FAIL@throat), coldwin16 (COMPLETE) and coldwin11 (PASS) nav-traces: the coupled-outbound
APPROACH rows (map frame; x,y,yaw_deg,lin,cw) from the post-couple creep up to the throat lip / pin.
Rows are SI, map frame (travel/portal axis ~ 0 deg east, throat x = -3.875). Every 2nd advancing row.
"""
import json
from pathlib import Path

MISSION = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission')
OUT = Path(__file__).resolve().parents[2] / 'scripts/twin/tests/fixtures/outbound_throat_yaw_bound_nav.json'


def approach(run, keep_past_throat=24):
    rows = []
    for ln in (MISSION / run / 'nav-trace.jsonl').open():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        rows.append(r)
    band = [r for r in rows if r.get('coupled') and -4.75 <= r['x'] <= -3.40 and r['y'] > 14.5]
    # advancing approach: keep until the base crosses the lip (x >= -3.84, a few cm past -3.875) OR, if it
    # never crosses (coldwin17 pins at ~-3.93), until it has been stalled near its max-x for keep_past_throat
    # rows. The stall counter only arms once the base is genuinely near the throat (x > -4.0), so the brief
    # post-couple dip at the seat (x ~ -4.5) is never mistaken for the pin.
    out = []
    xmax = -1e9
    stall = 0
    for r in band:
        out.append(r)
        if r['x'] >= -3.84:
            break
        if r["x"] > xmax - 1e-3:
            xmax = max(xmax, r['x'])
            stall = 0
        elif xmax > -4.0:
            stall += 1
        if stall >= keep_past_throat:
            break
    # compact: every 2nd row, only the fields the replay needs
    compact = []
    for i, r in enumerate(out):
        if i % 2:
            continue
        compact.append({'x': round(r['x'], 3), 'y': round(r['y'], 3), 'yaw_deg': round(r['yaw_deg'], 2),
                        'lin': round(r['lin'], 4), 'cw': round(r.get('vel', {}).get('cw', 0.0), 4),
                        'rw': round(r.get('vel', {}).get('rw', 0.0), 4),
                        'world_t': round(r['world_t'], 4)})
    return compact[:170]   # full crossing for the passers; approach+short pin for coldwin17


doc = {
    '_note': ('Real coupled-OUTBOUND approach creeps (SI, map frame; travel/portal axis ~0 deg east, '
              'throat x=-3.875) for sim-robust-heading-hold-bound-v1. coldwin17 FAILED@throat (wedged '
              'at ~-3.93, base yaw grew to +5.5 deg); coldwin16 COMPLETE (entered ~+2 deg); coldwin11 '
              'PASS (entered +5.3 deg but kept momentum). The authored lane outbound_dock_to_bay '
              'descends y 15.5->15.25 so the base seats ~0.25 m SOUTH of the centreline and the '
              'follower lateral term drives a CCW yaw drift. Every 2nd advancing row to the lip/pin.'),
    'throat_x': -3.875,
    'lane': {'name': 'outbound_dock_to_bay', 'start': [-4.46, 15.5, 0.0], 'goal': [-2.0, 15.25, 0.0],
             'start_cart_yaw': 0.0, 'goal_articulation': 0.3, 'start_tol_cart_yaw': 0.35,
             'lead_in_max_m': 3.0},
}
for run in ('coldwin17', 'coldwin16', 'coldwin11'):
    doc[run] = {'rows': approach(run)}

OUT.write_text(json.dumps(doc, indent=1))
print('wrote', OUT, 'coldwin17', len(doc['coldwin17']['rows']), 'coldwin16',
      len(doc['coldwin16']['rows']), 'coldwin11', len(doc['coldwin11']['rows']), 'rows')
