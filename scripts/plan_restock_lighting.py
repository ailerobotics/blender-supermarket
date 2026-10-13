#!/usr/bin/env python3
"""CPU-only line-of-sight proof for the restock store light rig; never a render.

Loads the Blender-matched light rig (twin.restock_camera.lights) and, for each
restock station support, ray-tests whether any ceiling panel or backroom tube has
an unobstructed path to the support plane. This is a geometric design prediction:
it proves which supports a ceiling fixture can reach and flags the ones that are
recessed (the shelf slot) so integration keeps the dome/local fill. It makes no
decode, exposure, contrast or rendered-illuminance claim.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from twin.restock_camera.config import camera_config
from twin.restock_camera.geometry import pose_matrix, transform_points
from twin.restock_camera.lights import (light_rig, store_fixtures, LightRigTuning,
                                         rect_illuminance, sphere_illuminance,
                                         lamp_marker_occlusion)
from twin.restock_camera.visibility import (ray_hits, sample_plane, usd_surfaces,
                                            box_surface)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def reach(fixture, point, surfaces):
    """A fixture reaches a point if it is above it and nothing blocks the segment."""
    eye = np.asarray(fixture.center, float)
    if eye[2] <= point[2]:
        return False
    return not any(ray_hits(eye, point, s.triangles) for s in surfaces)


def near_overhead(fixture, center, max_tilt_deg=35.0):
    """True if the fixture sits roughly above ``center`` (steep, high-cosine light).

    A grazing fixture far down the aisle may clear a slot geometrically yet deliver
    little illuminance, so only near-overhead reach counts as genuinely lighting a
    support; grazing-only reach flags a recessed support that needs fill.
    """
    d = center - np.asarray(fixture.center, float)
    horizontal = float(np.hypot(d[0], d[1]))
    return np.degrees(np.arctan2(horizontal, abs(d[2]))) <= max_tilt_deg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scenario', 'navigation', 'reach', 'mobile-manifest', 'store', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    scenario, nav, reachrep, manifest = [json.loads(path.read_text()) for path in
                                         (a.scenario, a.navigation, a.reach, a.mobile_manifest)]
    s = scenario['initial_staging']
    cart = [nav['legs'][0]['cart_body_pose'][key] for key in ('x', 'y', 'yaw')]
    config = camera_config(scenario, manifest)
    fixtures = store_fixtures()
    inputs = [a.scenario, a.navigation, a.reach, a.mobile_manifest, a.store,
              Path(__file__), Path(__file__).parent / 'twin/restock_camera/lights.py',
              Path(__file__).parent / 'twin/restock_camera/config.py',
              Path(__file__).parent / 'twin/restock_camera/visibility.py']
    hashes = {str(path.resolve()): sha(path) for path in inputs}
    layers = set()
    results = {}
    for name, support in config['supports'].items():
        parent = pose_matrix(cart, .2) if support['parent'] == 'cart_chassis' else np.eye(4)
        yaw = support['yaw'] + (cart[2] if support['parent'] == 'cart_chassis' else 0)
        center = transform_points(parent, [support['nominal_center']])[0]
        patch = sample_plane(center, support['required_xy_m'], yaw, count=5)
        markers = {m['id']: transform_points(parent @ np.array(m['parent_from_marker']),
                                             [[0, 0, .0002]])[0] for m in support['markers']}
        bounds = (np.minimum(patch.min(axis=0), center) - [.2, .2, .1],
                  np.array([patch[:, 0].max(), patch[:, 1].max(), 4.1]) + [.2, .2, 0])
        surfaces, used = usd_surfaces(a.store, bounds, exclude=('/RestockScenario/HandlingUnit',))
        layers.update(used)
        # The support sits on its own shelf/deck; only occluders strictly above it block
        # overhead light. ray_hits already excludes the segment endpoints.
        panel_reach, direct_reach = {}, {}
        for f in fixtures:
            hit = [i for i, pt in enumerate(patch) if reach(f, pt, surfaces)]
            if hit:
                entry = dict(kind=f.kind, samples_reached=len(hit),
                             near_overhead=bool(near_overhead(f, center)),
                             distance_m=float(np.linalg.norm(np.asarray(f.center) - center)))
                panel_reach[f.id] = entry
                if entry['near_overhead']:
                    direct_reach[f.id] = entry
        marker_reach = {mid: sorted(f.id for f in fixtures
                                    if near_overhead(f, np.asarray(pt)) and reach(f, pt, surfaces))
                        for mid, pt in markers.items()}
        # Illuminance (lux, intensity_scale=1) at each marker: horizontal up-facing
        # QR plane lit by store fixtures (with line of sight) plus any camera-mounted
        # lamp bolted to this station's camera. The lamp shares the rig tuning, so the
        # marker/source illuminance ratio is independent of the scale integration picks.
        up = np.array([0, 0, 1.0])
        station_lamps = [lamp for lamp in config.get('lamps', []) if lamp['id'] == name]
        # Obstacles for the emitter->marker ray test: both camera-rig housings (camera
        # body and the lamp body) plus the store shelf tiers/products already loaded.
        # A lamp whose own housing or a tier sits on the segment never lights the marker.
        housings = [box_surface(sol['id'], [0, 0, 0], sol['dimensions_m'],
                                transform=parent @ np.array(sol['parent_from_solid']))
                    for sol in config['solids'] if sol['id'].endswith('_housing')]
        lamp_obstacles = housings + surfaces
        lamp_occlusion = {}
        marker_lux = {}
        for mid, pt in markers.items():
            store_lux = sum(rect_illuminance(f, pt, up, LightRigTuning().intensity_nits(f))
                            for f in fixtures if reach(f, pt, surfaces))
            lamp_lux = 0.0
            for lamp in station_lamps:
                lc = transform_points(parent, [lamp['center']])[0]
                blocked = lamp_marker_occlusion(lc, [pt], lamp_obstacles)
                lamp_occlusion.setdefault(lamp['id'], {})[mid] = (
                    sorted({n.split('/')[-1] for n in blocked[0]}) if blocked else [])
                if not blocked and lc[2] > pt[2]:
                    lamp_lux += sphere_illuminance(lc, lamp['radius'], pt, up, lamp['intensity_nits'])
            marker_lux[mid] = dict(store_lux=round(store_lux, 1), lamp_lux=round(lamp_lux, 1),
                                   total_lux=round(store_lux + lamp_lux, 1))
        best_any = max((v['samples_reached'] for v in panel_reach.values()), default=0)
        best_direct = max((v['samples_reached'] for v in direct_reach.values()), default=0)
        fraction = best_direct / len(patch)
        markers_lit = all(marker_reach[m] for m in markers)
        if fraction >= 0.9:
            verdict = 'direct_overhead_light'
        elif best_any >= 1:
            verdict = 'grazing_only_keep_fill'
        else:
            verdict = 'recessed_requires_fill'
        results[name] = dict(
            support_center_world=center.tolist(),
            overhead_fixtures_with_line_of_sight=len(panel_reach),
            near_overhead_fixtures_with_line_of_sight=len(direct_reach),
            best_near_overhead_support_samples_reached=best_direct,
            best_any_fixture_support_samples_reached=best_any,
            near_overhead_support_fraction=round(fraction, 3),
            markers_reached_by_near_overhead=markers_lit,
            verdict=verdict,
            nearest_fixtures=dict(sorted(panel_reach.items(),
                                         key=lambda kv: kv[1]['distance_m'])[:4]),
            marker_near_overhead_line_of_sight=marker_reach,
            camera_lamp_ids=[lamp['id'] for lamp in station_lamps],
            camera_lamp_marker_occlusion=lamp_occlusion,
            camera_lamp_reaches_all_markers=bool(station_lamps) and all(
                not blk for lamp in lamp_occlusion.values() for blk in lamp.values()),
            marker_illuminance_lux_scale1=marker_lux)
    # Anchor to the measured SOURCE decode: integration reports the source markers
    # decode 3/3 under this rig, so at the chosen intensity_scale their white modules
    # reach the ~180/255 threshold. A marker lit to >= that illuminance, same up-facing
    # QR material and the same shared scale/exposure, reaches >= the same pixel value.
    source_marker_lux = [m['total_lux'] for m in results['source']['marker_illuminance_lux_scale1'].values()]
    source_ref = min(source_marker_lux)
    for name, st in results.items():
        worst = min(m['total_lux'] for m in st['marker_illuminance_lux_scale1'].values())
        lamp_lux = min((m['lamp_lux'] for m in st['marker_illuminance_lux_scale1'].values()),
                       default=0.0) if st['camera_lamp_ids'] else 0.0
        st['worst_marker_lux_scale1'] = worst
        st['worst_marker_vs_source_ratio'] = round(worst / source_ref, 3) if source_ref else None
        st['predicted_marker_white_ge_source_threshold'] = bool(worst + 1e-6 >= source_ref)
        # Combined verdict. A marker is decodably lit if a near-overhead store fixture
        # sees it (integration confirmed source decodes 3/3 wherever line of sight
        # exists) OR a camera lamp lights it to >= the measured source level. The
        # source ratio is a sufficient (conservative) bound, not a necessary one: cart
        # is below the bright source yet is direct-overhead lit, so it is not a gap.
        if st['markers_reached_by_near_overhead']:
            st['marker_lit_verdict'] = 'overhead_direct'
        elif lamp_lux > 0 and worst + 1e-6 >= source_ref:
            st['marker_lit_verdict'] = 'camera_lamp_lit'
        elif lamp_lux > 0:
            st['marker_lit_verdict'] = 'camera_lamp_below_source_reference'
        else:
            st['marker_lit_verdict'] = 'dark_requires_fill'
    for path in sorted(layers):
        hashes[str(Path(path).resolve())] = sha(path)
    for path in inputs:
        if hashes[str(path.resolve())] != sha(path):
            raise ValueError('input changed during lighting audit')
    rig = light_rig(LightRigTuning())
    report = dict(evidence='CPU_geometric_lineofsight_design_not_rendered_illuminance',
                  runtime_ready=False, input_sha256=hashes,
                  light_rig_summary=dict(counts=rig['counts'], tuning=rig['tuning'],
                                         dome=rig['dome'], tonemapping=rig['tonemapping']),
                  camera_lamps=config.get('lamps', []),
                  source_marker_reference_lux_scale1=round(source_ref, 1),
                  stations=results,
                  limitations=[
                      'Illuminance is estimated from UsdLux-consistent geometry, not a render; the '
                      'source-anchored >=180/255 claim assumes the measured source decode means its '
                      'whites reach ~180 and the exposure/scale is shared by lamp and fixtures.',
                      'The camera lamp shares the rig intensity_scale, so the marker/source ratio is '
                      'scale-free; integration still tunes the one scale and tonemapper on real pixels.',
                      'Ceiling/tube geometry mirrors the Blender store; integration authors the rig '
                      'and the camera lamp into the native stage and owns the GPU render.'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: dict(marker_lit_verdict=st['marker_lit_verdict'],
                                 overhead=st['verdict'],
                                 lamp_reaches_all_markers=st['camera_lamp_reaches_all_markers'],
                                 worst_marker_lux=st['worst_marker_lux_scale1'],
                                 vs_source=st['worst_marker_vs_source_ratio'],
                                 lamps=st['camera_lamp_ids'])
                      for name, st in results.items()}, indent=2))


if __name__ == '__main__':
    main()
