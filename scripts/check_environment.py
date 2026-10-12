"""Headless Blender geometry checks. See RUNBOOK environment section."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--baseline', help='Optional pre-environment build for product invariance check')
parser.add_argument('--radius', type=float, default=.65,
                    help='Uncoupled transport envelope radius in metres (mobile default: 0.65)')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
data = json.loads(Path(args.manifest).read_text())
assert bpy.data.objects.get('Wall_back') is None, 'Solid rear wall still seals portal'
assert bpy.data.objects.get('Baseboard') is None, 'Old skirting crosses portal'
assert abs(bpy.data.objects['Backroom_floor'].location.z + .12) < 1e-6
assert len(data['static_obstacles']) > 600

# Independent checks against generated meshes, not just nominal layout constants.
obstacles = data['static_obstacles']
radius = args.radius
assert math.isfinite(radius) and radius >= .65, 'Radius must cover the configured 0.65 m mobile envelope'
points = data['route_hints']
for a, b in zip(points, points[1:]):
    count = max(1, math.ceil(math.dist(a, b) / .05))
    for i in range(count + 1):
        p = [a[j] + (b[j] - a[j]) * i / count for j in range(2)]
        for ob in obstacles:
            distance = math.hypot(*[max(ob['min'][j] - p[j], 0, p[j] - ob['max'][j]) for j in range(2)])
            assert distance >= radius, f"Route footprint intersects {ob['id']} at {p}: {distance}"
for stock in data['backroom_stocks']:
    cases = sorted(o for o in bpy.data.objects.keys() if o.startswith(stock['id'] + '_case_'))
    assert len(cases) == len(stock['case_positions']) == 6
    for name, pos in zip(cases, stock['case_positions']):
        assert math.dist(bpy.data.objects[name].matrix_world.translation, pos) < 1e-5
        assert bpy.data.objects[name]['units'] == stock['units_per_case']
    assert stock['units'] == len(cases) * stock['units_per_case']
for s in data['stocking_stations']:
    assert bpy.data.objects.get(s['id']) is not None
    assert abs(bpy.data.objects[s['id']].rotation_euler.z - s['pose'][2]) < 1e-5

# Stable digest can be compared against a baseline build independently.
products = sorted((o.name, o.get('sku'), o.get('slot'), list(o.location), list(o.rotation_euler), list(o.scale))
                  for o in bpy.data.objects if 'slot' in o and 'sku' in o)
result = dict(status='passed', static_obstacles=len(obstacles), stocking_stations=len(data['stocking_stations']),
              route_checked_radius_m=radius, route_sample_spacing_m=.05,
              limitation='Uncoupled circular clearance only; excludes cart sweep, controller feasibility and dynamics.',
              product_digest=hashlib.sha256(json.dumps(products).encode()).hexdigest(),
              product_and_tag_count=len(products))
if args.baseline:
    after_products = {row[0]: row for row in products}
    # Undo only the documented Pallet_3 relocation before comparison.
    for row in after_products.values():
        if row[2] == 'Pallet_3':
            row[3][0] += 5
            row[3][1] -= 6.8
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.baseline).resolve()))
    before_products = {o.name: (o.name, o.get('sku'), o.get('slot'), list(o.location),
                              list(o.rotation_euler), list(o.scale))
                       for o in bpy.data.objects if 'slot' in o and 'sku' in o}
    assert before_products.keys() == after_products.keys()
    for name, before in before_products.items():
        after = after_products[name]
        assert before[:3] == after[:3]
        for a, b in zip(before[3:], after[3:]):
            assert math.dist(a, b) < 1e-5, name
    result['baseline_products_preserved'] = True
Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
