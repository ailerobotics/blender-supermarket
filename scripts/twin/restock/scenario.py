"""Validate environment semantics without treating them as perception evidence."""
import math
from .inventory import Job


def poses_for_job(manifest, job, *, return_pose='robot_dock'):
    """Convert schema-2 body frames to mission robot goals, never cart centres."""
    poses, bodies = manifest['poses'], manifest['pose_bodies']
    for name in ('pickup', 'robot_dock', return_pose):
        if bodies.get(name) != 'robot_base':
            raise ValueError(f'{name} must denote the robot base')
    station = next((s for s in manifest['stocking_stations']
                    if any(slot['id'] == job.shelf_id for slot in s['slots'])), None)
    if station is None or station.get('pose_body') != 'robot_base':
        raise ValueError('shelf station must specify a robot-base pose')
    return dict(pickup=poses['pickup'], cart_dock=poses['robot_dock'],
                shelf_approach=station['pose'], home=poses[return_pose])


def validate(manifest, scenario):
    errors = []
    if (manifest.get('frame'), manifest.get('units'), manifest.get('up_axis')) != ('map', 'metres', 'Z'):
        errors.append('environment must use map frame, metres and Z up')
    if manifest.get('ground_truth_only') is not True:
        errors.append('environment provenance must be marked ground_truth_only')
    try:
        job = Job(**scenario['job'])
    except (TypeError, KeyError, ValueError) as exc:
        return {'ok': False, 'errors': errors + [str(exc)]}
    sources = {s['id']: s for s in manifest.get('backroom_stocks', [])}
    source = sources.get(job.source_bin)
    if not source or source.get('sku') != job.sku:
        errors.append('source bin/SKU does not exist in environment')
    slots = {s['id']: s for station in manifest.get('stocking_stations', []) for s in station['slots']}
    if job.shelf_id not in slots or job.sku not in slots[job.shelf_id]['sku_ids']:
        errors.append('shelf slot does not accept requested SKU')
    if job.quantity != 1:
        errors.append('initial manipulation supports one sealed handling unit only')
    if scenario.get('inventory_unit') != 'sealed_handling_unit':
        errors.append('inventory must count handling units, not visual-case retail contents')
    inventory = scenario.get('initial_inventory', [])
    seen = set()
    available = 0
    if not isinstance(inventory, list):
        inventory = []
    for row in inventory:
        if not isinstance(row, dict):
            errors.append('invalid initial inventory row')
            continue
        key = (row.get('location'), row.get('sku'))
        quantity = row.get('quantity')
        if (not all(isinstance(v, str) and v for v in key) or
                type(quantity) is not int or quantity < 0):
            errors.append('invalid initial inventory row')
            continue
        if key in seen:
            errors.append('duplicate initial inventory row')
        seen.add(key)
        if key == (job.source_bin, job.sku):
            available += quantity
    if available < job.quantity:
        errors.append('initial inventory cannot satisfy job')
    if scenario.get('dynamic_stock_ready') is not True:
        errors.append('static backstock cases must be promoted to dynamic actors')
    if manifest.get('schema_version') == 2:
        try:
            poses_for_job(manifest, job, return_pose=scenario.get('return_pose', 'robot_dock'))
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
    if not manifest.get('static_obstacles'):
        errors.append('missing measured static collision map')
    # Environment stores [xmin,xmax,ymin,ymax]; navigation uses [xmin,ymin,xmax,ymax].
    raw = manifest.get('map_bounds', ())
    bounds = None
    if len(raw) == 4 and all(isinstance(x, (int,float)) and math.isfinite(x) for x in raw):
        bounds = [raw[0], raw[2], raw[1], raw[3]]
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            errors.append('invalid map bounds')
    else:
        errors.append('missing/invalid map bounds')
    radius = scenario.get('coupled_navigation_radius_m')
    if not isinstance(radius, (int,float)) or isinstance(radius,bool) or not math.isfinite(radius) or radius <= 0:
        errors.append('missing conservative coupled navigation radius')
    elif 2*radius >= manifest.get('portal', {}).get('clear_width', 0):
        errors.append('coupled navigation envelope cannot fit backroom portal')
    return {'ok': not errors, 'errors': errors, 'navigation_bounds': bounds,
            'provenance': 'offline_semantic_validation_not_physics'}
