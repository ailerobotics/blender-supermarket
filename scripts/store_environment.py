"""Deterministic service-area geometry and mission annotations (metres, Z up).

The manifest is semantic ground truth for scenario setup, never sensor evidence.
Blender is imported only by build_environment; layout() remains CPU-testable.
"""
import json
import math
from pathlib import Path

PORTAL_X = -3.875
PORTAL_WIDTH = 2.5
BACKROOM = (-12.0, 1.0, 11.25, 19.25)


def layout(plan):
    """Derive stocking targets from existing slots without changing the planogram."""
    stations = []
    for g in plan['gondolas']:
        for side, direction in [('L', -1), ('R', 1)]:
            # Outer aisles are narrower and not designated towing routes.
            if (g == plan['gondolas'][0] and side == 'L') or (g == plan['gondolas'][-1] and side == 'R'):
                continue
            for bay in g['sides'][side]['bays']:
                y = g['y0'] + (bay['index'] + .5) * plan['bay_w']
                station_id = f"stock_{g['id']}{side}_b{bay['index']}"
                stations.append(dict(id=station_id,
                    pose=[g['x'] + direction * 1.525, y, -math.pi / 2],
                    pose_body='robot_base',
                    arm_shelf_facing_yaw=math.pi if direction == 1 else 0,
                    slots=[dict(id=f"{g['id']}{side}_b{bay['index']}_s{i}",
                                surface_z=s['z'],
                                sku_ids=[item[0] for item in s['items']])
                           for i, s in enumerate(bay['shelves'])],
                    approach='Chassis aligned down aisle; return requires uncoupling or a separately checked forward route.'))
    endcap = next(e for e in plan['endcaps'] if e['id'] == 'G3_back')
    first_cycle = dict(id='stock_G3_back', pose=[endcap['x'], 6.0, math.pi],
        pose_body='robot_base', arm_shelf_facing_yaw=-math.pi / 2,
        slots=[dict(id=f"G3_back_s{i}", surface_z=s['z'],
                    sku_ids=[item[0] for item in s['items']])
               for i, s in enumerate(endcap['shelves'])],
        approach='Forward straight westbound pull-through; arrival and return turns require articulated planning.',
        pull_through_robot_poses=[[-1.525, 6.0, math.pi], [-7.0, 6.0, math.pi]],
        cart_centre_separation=1.27,
        envelope_reference='MobileSpec defaults; independently predicted cart poses required outside the straight segment')
    stations.append(first_cycle)
    sku_ids = sorted({sku for s in stations for slot in s['slots'] for sku in slot['sku_ids']})
    sku_ids = [endcap['sku']] + [sku for sku in sku_ids if sku != endcap['sku']]
    stocks = [dict(id=f'backstock_{i:02d}', sku=sku_ids[i],
                   position=[-10.8 + i * 2.0, 18.45, 1.12],
                   case_positions=[[-10.8 + i * 2.0 + (j - .5) * .75, 18.45, z + .04]
                                   for z in [.16, 1.08, 2.0] for j in range(2)],
                   size=[.6, .4, .32], units=72, units_per_case=12, role='static_case_visual') for i in range(5)]
    return dict(schema_version=2, frame='map', units='metres', up_axis='Z',
        source='illustrative German grocery layout, not a measured REWE scan',
        ground_truth_only=True, floor_z=0,
        backroom_bounds=list(BACKROOM),
        map_bounds=[-13.25, 13.25, -11.25, 19.5],
        poses=dict(home=[PORTAL_X, 14, 0], pickup=[-5.73, 16, 0],
                   robot_dock=[-5.73, 16, 0], cart_dock=[-7, 16, 0],
                   shelf_approach=first_cycle['pose']),
        pose_bodies=dict(home='robot_base', pickup='robot_base', robot_dock='robot_base',
                         cart_dock='cart_body', shelf_approach='robot_base'),
        first_cycle_station=first_cycle['id'],
        first_cycle_source_bin='backstock_00',
        first_cycle_sku=endcap['sku'],
        portal=dict(id='sales_backroom_portal', centre=[PORTAL_X, 11.125, 0],
                    clear_width=PORTAL_WIDTH, clear_height=2.8),
        zones=[dict(id='receiving_staging', centre=[-9, 14, 0], size=[3, 3]),
               dict(id='rollwagen_pickup', centre=[-7, 16, 0], size=[1.2, 2]),
               dict(id='rollwagen_parking', centre=[-9, 16, 0], size=[1.2, 2]),
               dict(id='backroom_turning', centre=[PORTAL_X, 14, 0], size=[3.2, 3.2])],
        route_hints=[[-5.73, 16], [PORTAL_X, 16], [PORTAL_X, 14], [PORTAL_X, 11.125],
                     [PORTAL_X, 7.5], [-1.525, 7.5], [-1.525, .125]],
        route_policy='Hints only; navigation must plan and collision-check the actual coupled footprint.',
        stocking_stations=stations, backroom_stocks=stocks,
        collision=dict(static_groups=['GRP_Structure', 'GRP_Fixtures'],
                       annotation='userProperties:collision_role',
                       decorative_group='GRP_EnvironmentDecor',
                       movable_stock='Cases are static scenery; promote selected SKU to a dynamic actor in scenario composition.'))


def write_manifest(plan, path, scene=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = layout(plan)
    if scene is not None:
        from mathutils import Vector
        scene.view_layers[0].update()
        obstacles = []
        for ob in scene.objects:
            if ob.type != 'MESH':
                continue
            root = ob
            while root.parent:
                root = root.parent
            if root.name not in {'GRP_Structure', 'GRP_Fixtures'}:
                continue
            corners = [ob.matrix_world @ Vector(v) for v in ob.bound_box]
            lo = [min(p[i] for p in corners) for i in range(3)]
            hi = [max(p[i] for p in corners) for i in range(3)]
            # Floors/overhead beams do not occupy the mobile robot height band.
            if hi[2] <= .02 or lo[2] >= 2.6:
                continue
            obstacles.append(dict(id=ob.name, min=lo, max=hi,
                                  representation='conservative_world_aabb'))
        data['static_obstacles'] = sorted(obstacles, key=lambda o: o['id'])
        data['obstacle_height_band'] = [.02, 2.6]
    path.write_text(json.dumps(data, indent=2) + '\n')


def build_environment(api, materials):
    """Extend a generated scene through the builder's mesh/material helpers."""
    import bpy
    data = layout(api['PLAN'])
    coll, box_mesh, add = api['coll'], api['box_mesh'], api['add_obj']
    structure = coll('Structure')
    fixtures = coll('Fixtures_Backroom', coll('Fixtures'))
    decor = coll('EnvironmentDecor')
    semantic = coll('EnvironmentMarkers')
    red = materials['plastic_red']
    yellow = api['principled']('Service_safety_yellow', api['col']('#e6b632'), .7)
    white = materials['ceiling']

    def box(name, size, pos, mat, collection=fixtures, role='static'):
        return add(name, box_mesh(name, *size, mat), collection, loc=pos,
                   props=dict(collision_role=role, environment_version=2))

    def text(name, body, pos, size=.22, mat=materials['steel_dark']):
        curve = bpy.data.curves.new(name, 'FONT'); curve.body = body
        curve.size = size; curve.align_x = 'CENTER'; curve.extrude = .001
        curve.materials.append(mat)
        return add(name, curve, decor, loc=pos, rot=(math.pi / 2, 0, 0),
                   props=dict(collision_role='none'))

    # Move the existing promotional pallet AND its cases into receiving staging.
    # Its old (-4, 7.2) position blocks the new service route. Slot/SKU IDs survive.
    pallet = bpy.data.objects.get('Pallet_3')
    if pallet:
        dx, dy = -9 - pallet.location.x, 14 - pallet.location.y
        for ob in bpy.data.objects:
            if ob == pallet or ob.get('slot') == 'Pallet_3':
                ob.location.x += dx; ob.location.y += dy

    # Replace the solid rear wall and its continuous skirting with a true opening.
    for name in ['Wall_back', 'Baseboard']:
        ob = bpy.data.objects.get(name)
        if ob: bpy.data.objects.remove(ob, do_unlink=True)
    left, right = PORTAL_X - PORTAL_WIDTH / 2, PORTAL_X + PORTAL_WIDTH / 2
    for suffix, lo, hi in [('west', -13.25, left), ('east', right, 13.25)]:
        box('Wall_back_' + suffix, (hi - lo, .25, 4.3), ((lo + hi) / 2, 11.125, 0), white, structure)
        box('Skirting_back_' + suffix, (hi - lo, .025, .1), ((lo + hi) / 2, 10.99, 0), materials['kick'], structure)
    for x in [-12.99, 12.99]:
        box('Skirting_side_' + str(x), (.025, 22, .1), (x, 0, 0), materials['kick'], structure)
    box('Portal_lintel', (PORTAL_WIDTH, .25, 1.5), (PORTAL_X, 11.125, 2.8), white, structure)
    for x in [left - .035, right + .035]:
        box('Portal_jamb_' + str(x), (.07, .3, 2.8), (x, 11.125, 0), materials['steel_dark'], structure)
    box('Backroom_floor', (13, 8.25, .12), (-5.5, 15.125, -.12), materials['concrete'], structure)
    for name, size, pos in [('west', (.25, 8.25, 4.3), (-12.125, 15.125, 0)),
                            ('east', (.25, 8.25, 4.3), (1.125, 15.125, 0)),
                            ('north', (13.5, .25, 4.3), (-5.5, 19.375, 0))]:
        box('Backroom_wall_' + name, size, pos, white, structure)
    box('Backroom_ceiling', (13.5, 8.5, .05), (-5.5, 15.25, 4), white, structure)
    box('Service_header', (2.5, .04, .42), (PORTAL_X, 10.95, 2.9), red)
    text('Service_label', 'LAGER / WARENEINGANG', (PORTAL_X, 10.92, 3.03), .14, white)
    # Industrial bays with separate uprights and decks: openings stay collision-open.
    for i, stock in enumerate(data['backroom_stocks']):
        x = stock['position'][0]
        for dx in [-.85, .85]:
            for y in [17.95, 18.95]:
                box(f'Rack_{i}_post_{dx}_{y}', (.075, .075, 2.8), (x + dx, y, 0), materials['steel_dark'])
        for level, z in enumerate([.16, 1.08, 2.0]):
            box(f'Rack_{i}_deck_{level}', (1.8, 1.1, .04), (x, 18.45, z), materials['wood_light'])
            box(f'Rack_{i}_beam_{level}', (1.8, .06, .10), (x, 17.9, z - .04), red)
            for j in range(2):
                ob = box(f"{stock['id']}_case_{level}_{j}", stock['size'],
                         (x + (j - .5) * .75, 18.45, z + .04), materials['cardboard'])
                ob['sku'] = stock['sku']; ob['stock_id'] = stock['id']; ob['units'] = stock['units_per_case']
                box(f'Tape_{i}_{level}_{j}', (.05, .405, .002),
                    (x + (j - .5) * .75, 18.45, z + .362), white, decor, 'none')
        text(f'Rack_label_{i}', f"R{i+1:02d}  /  {stock['sku'].upper()}", (x, 17.86, 2.48), .10)
    for zone in data['zones']:
        x, y, _ = zone['centre']; w, d = zone['size']
        for dx in [-w / 2, w / 2]:
            box(zone['id'] + '_line_x' + str(dx), (.045, d, .002), (x + dx, y, .002), yellow, decor, 'none')
        for dy in [-d / 2, d / 2]:
            box(zone['id'] + '_line_y' + str(dy), (w, .045, .002), (x, y + dy, .002), yellow, decor, 'none')
    text('Receiving_label', 'WARENEINGANG   /   LEERGUT', (-8.7, 19.20, 3.25), .25)
    box('Parking_header', (2.8, .04, .4), (-8, 17.24, 2.78), red)
    for x in [-9.2, -6.8]:
        box('Parking_hanger_' + str(x), (.015, .015, .82), (x, 17.24, 3.18), materials['steel_dark'])
    text('Parking_label', 'ROLLWAGEN', (-8, 17.2, 2.9), .24, white)
    # Sales-floor fascia brings a consistent red/neutral German grocery palette.
    for g in api['PLAN']['gondolas']:
        length = g['n_bays'] * api['BAY_W']
        for direction in [-1, 1]:
            box(f"Fascia_{g['id']}_{direction}", (.035, length, .14),
                (g['x'] + direction * .065, g['y0'] + length / 2, 2.03), red)
        # Steel corner guards below existing endcap shelves.
        for y in [g['y0'] - .53, g['y0'] + length + .53]:
            box(f"Guard_{g['id']}_{y}", (1.08, .06, .10), (g['x'], y, .03), materials['steel_dark'])
    for item in data['stocking_stations'] + data['zones'] + data['backroom_stocks']:
        pos = item['pose'][:2] + [0] if 'pose' in item else item.get('centre', item.get('position'))[:3]
        ob = add(item['id'], None, semantic, loc=pos,
                 rot=(0, 0, item['pose'][2] if 'pose' in item else 0),
                 props=dict(semantic_id=item['id'], collision_role='none', ground_truth_only=True))
        ob.empty_display_size = .15
    for x in [-9, -5, -1]:
        for y in [13, 16, 18.5]:
            light = bpy.data.lights.new(f'ServiceLight_{x}_{y}', 'AREA')
            light.energy = 100; light.shape = 'RECTANGLE'; light.size = 1.2; light.size_y = .3
            add(light.name, light, coll('Lighting'), loc=(x, y, 3.85))
            box(f'ServiceLED_{x}_{y}', (1.2, .3, .025), (x, y, 3.9), materials['led'], decor, 'none')
    cd = bpy.data.cameras.new('Cam_Backroom'); cd.lens = 22
    camera = add('Cam_Backroom', cd, coll('Cameras'), loc=(-.1, 12, 2.4))
    api['look_at'](camera, (-7.5, 17.3, 1.25))
