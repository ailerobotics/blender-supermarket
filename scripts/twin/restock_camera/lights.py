"""Store light rig for the native restock composition.

Reproduces the Blender store's ceiling LED panels (``build_supermarket.build_lighting``)
and the backroom service tubes (``store_environment.add_service_environment``) as
UsdLux fixtures so the native composition can be lit without opening Blender. These
are physical fixture *designs*, never a rendered result or a sensing claim: the
integration worker authors them into the native stage, picks a tonemapper, and tunes
the single ``intensity_scale`` until the QR white modules read below clipping.

Geometry and optical axes: metres, world +Z up. A down-facing ceiling fixture needs
no rotation because both a Blender area light and a UsdLux RectLight emit along local
-Z, which is world -Z at identity orientation.

Sharp edge this rig exists to surface: the shelf support markers sit in a ~0.24 m
shelf slot (surface shelf at z~0.73, the next tier at z~0.97 directly above), so the
overhead ceiling panels have no line of sight to them. A dome/ambient fill or a local
task fill co-located with the shelf camera is required; ceiling panels alone leave the
recessed support dark. Use ``plan_restock_lighting.py`` to prove line of sight offline.
"""
from dataclasses import dataclass, field, asdict
import math

import numpy as np

from .geometry import look_at, usd_camera_transform

# --- Blender store constants (scripts/build_supermarket.py, scripts/store_environment.py).
# Kept in sync by the decimal values; see plan_restock_lighting.py provenance hashes.
H_CEIL = 4.0            # STORE["height"]
_W2, _D2 = 13.0, 11.0   # STORE width/2, depth/2

# Sales-floor ceiling LED panels: build_lighting() grid, AREA RECTANGLE 1.2x0.6, 45 W.
_SALES_XS = [-12 + 3 * i for i in range(9)]
_SALES_YS = [-_D2 + 1.5 + 3 * j for j in range(int((2 * _D2 - 1.5) // 3) + 1)]
_SALES = dict(width=1.2, height=0.6, power_w=45.0, color=(1.0, 0.97, 0.93),
              z=H_CEIL - 0.03)
# Backroom service tubes: add_service_environment() grid, AREA RECTANGLE 1.2x0.3, 100 W.
_BACKROOM_XS = [-9.0, -5.0, -1.0]
_BACKROOM_YS = [13.0, 16.0, 18.5]
_BACKROOM = dict(width=1.2, height=0.3, power_w=100.0, color=(1.0, 1.0, 1.0),
                 z=3.85)


@dataclass(frozen=True)
class RectFixture:
    """A down-facing (or aimed) rectangular emitter matching one Blender light."""
    id: str
    kind: str                       # 'sales_panel' | 'backroom_tube' | 'task_fill'
    center: tuple                   # world xyz of the emitter centre
    width: float                    # local X extent (m), == Blender size
    height: float                   # local Y extent (m), == Blender size_y
    power_w: float                  # Blender radiant power (W)
    color: tuple                    # linear RGB, == Blender light color
    world_from_light: tuple = None  # 4x4 row-major; None => identity (emits world -Z)

    def area_m2(self):
        return self.width * self.height

    def radiance_w(self):
        """Lambertian radiance W/(m^2.sr) that reproduces the Blender radiant power."""
        return self.power_w / (self.area_m2() * math.pi)


@dataclass(frozen=True)
class LightRigTuning:
    """The single knob integration tunes; ratios between fixtures are physical.

    UsdLux intensity is authored in Omniverse 'nits' as
    ``luminous_efficacy_lm_per_w * fixture.radiance_w() * intensity_scale`` with
    ``normalize=False`` so a fixture's emitted radiance (hence appearance) is area
    independent and the panel/tube brightness ratio equals Blender's exitance ratio.
    The efficacy is a documented approximation of a neutral-white LED; it and
    ``intensity_scale`` fold into one scalar the integration worker calibrates against
    the chosen tonemapper. The radiometric ``radiance_w()`` is published alongside so
    integration can instead convert to whatever units its render settings expect.
    """
    luminous_efficacy_lm_per_w: float = 150.0
    intensity_scale: float = 1.0
    exposure: float = 0.0
    normalize: bool = False
    dome_enabled: bool = True
    dome_intensity_nits: float = 300.0
    dome_color: tuple = (1.0, 0.98, 0.95)

    def intensity_nits(self, fixture):
        return (self.luminous_efficacy_lm_per_w * fixture.radiance_w()
                * self.intensity_scale)


def _grid_fixtures():
    out = []
    for x in _SALES_XS:
        for y in _SALES_YS:
            out.append(RectFixture(id=f'sales_panel_{x:+.0f}_{y:+.1f}', kind='sales_panel',
                                   center=(float(x), float(y), _SALES['z']),
                                   width=_SALES['width'], height=_SALES['height'],
                                   power_w=_SALES['power_w'], color=_SALES['color']))
    for x in _BACKROOM_XS:
        for y in _BACKROOM_YS:
            out.append(RectFixture(id=f'backroom_tube_{x:+.0f}_{y:+.1f}', kind='backroom_tube',
                                   center=(float(x), float(y), _BACKROOM['z']),
                                   width=_BACKROOM['width'], height=_BACKROOM['height'],
                                   power_w=_BACKROOM['power_w'], color=_BACKROOM['color']))
    return out


def store_fixtures():
    """Every Blender ceiling panel and backroom tube as down-facing RectFixtures."""
    return _grid_fixtures()


def task_fill_fixture(fixture_id, eye, target, *, width=0.18, height=0.06, power_w=12.0,
                      color=(1.0, 0.98, 0.95)):
    """A small rect fill aimed from ``eye`` toward ``target`` (emission along -Z).

    For recessed supports (the shelf slot) that the ceiling panels cannot reach. This
    is a task aid co-located with a camera mount, not a store fixture; it is only
    emitted into the rig when explicitly requested. ``usd_camera_transform`` turns the
    optical look_at (whose +Z points at the target) into a transform whose local -Z,
    the UsdLux emission axis, points at the target.
    """
    pose = usd_camera_transform(look_at(np.asarray(eye, float), np.asarray(target, float)))
    return RectFixture(id=fixture_id, kind='task_fill', center=tuple(map(float, eye)),
                       width=width, height=height, power_w=power_w, color=color,
                       world_from_light=tuple(map(tuple, pose.tolist())))


def camera_lamp(lamp_id, parent, center, *, radius=0.03, power_w=4.0,
                color=(1.0, 0.98, 0.95), tuning=None):
    """An omnidirectional sphere lamp physically mounted at a camera (a ring light).

    Co-located with the lens, it lights exactly what the camera sees — recessed
    supports (the shelf slot) that store fixtures cannot reach. A sphere radiates in
    every direction, so unlike an aimed rect fill it has no orientation failure mode.
    Intensity uses the SAME tuning as the store rig, so a single intensity_scale drives
    lamp and fixtures together and the lamp/source illuminance ratio is scale-free.
    """
    tuning = tuning or LightRigTuning()
    area = math.pi * radius ** 2            # projected (disk) area, as seen from the support
    radiance = power_w / (area * math.pi)   # Lambertian radiance matching the radiant power
    return dict(id=lamp_id, parent=parent, kind='camera_lamp', type='sphere',
                center=list(map(float, center)), radius=float(radius), power_w=float(power_w),
                color=list(color), area_m2=float(area), radiance_w=float(radiance),
                intensity_nits=float(tuning.luminous_efficacy_lm_per_w * radiance
                                     * tuning.intensity_scale))


def sphere_illuminance(center, radius, point, normal, nits):
    """Illuminance (lux) at ``point`` with surface ``normal`` from a sphere emitter.

    d>>r disk approximation: E = nits * pi r^2 * cos(surface) / d^2. UsdLux SphereLight
    intensity (normalize=False) is the surface luminance, matching this estimate.
    """
    d = np.asarray(point, float) - np.asarray(center, float)
    dist = float(np.linalg.norm(d))
    if dist <= radius:
        return float('inf')
    u = d / dist
    cos_surf = max(0.0, float(np.asarray(normal, float) @ (-u)))
    return nits * math.pi * radius ** 2 * cos_surf / dist ** 2


def lamp_marker_occlusion(lamp_center, marker_points, surfaces):
    """Cast a ray from a lamp emitter to each marker centroid against ``surfaces``.

    ``surfaces`` is a list of ``visibility.Surface`` (camera housing, lamp housing,
    shelf tiers, products). Returns {marker_index: [blocker names]} for every ray that
    is blocked; empty means the emitter physically reaches all markers. A lamp whose own
    housing or a shelf tier sits on the emitter->marker segment never lights the marker,
    whatever its intensity -- this is the check the illuminance estimate alone missed.
    """
    from .visibility import ray_hits
    eye = np.asarray(lamp_center, float)
    blocked = {}
    for i, pt in enumerate(np.asarray(marker_points, float)):
        names = [s.name for s in surfaces if ray_hits(eye, pt, s.triangles)]
        if names:
            blocked[i] = sorted(set(names))
    return blocked


def rect_illuminance(fixture, point, normal, nits):
    """Illuminance (lux) at ``point`` from a down-facing (-Z) rectangular emitter."""
    d = np.asarray(point, float) - np.asarray(fixture.center, float)
    dist = float(np.linalg.norm(d))
    if dist == 0.0:
        return float('inf')
    u = d / dist
    cos_surf = max(0.0, float(np.asarray(normal, float) @ (-u)))
    cos_src = max(0.0, float(u @ np.array([0, 0, -1.0])))   # emitter normal is -Z
    return nits * fixture.area_m2() * cos_src * cos_surf / dist ** 2


def light_rig(tuning=None, *, fills=()):
    """Serializable light rig: fixtures with computed intensities, dome, tonemapping.

    ``fills`` is an optional iterable of extra ``RectFixture`` task fills (see
    ``task_fill_fixture``) appended after the store fixtures.
    """
    tuning = tuning or LightRigTuning()
    fixtures = [*store_fixtures(), *fills]
    rig = dict(
        schema='restock-light-rig.v1',
        evidence='physical_fixture_design_not_rendered_sensor_capture',
        runtime_ready=False,
        source='blender build_supermarket.build_lighting + store_environment.add_service_environment',
        tuning=asdict(tuning),
        fixtures=[dict(**asdict(f), area_m2=f.area_m2(), radiance_w=f.radiance_w(),
                       intensity_nits=tuning.intensity_nits(f)) for f in fixtures],
        dome=dict(enabled=tuning.dome_enabled, intensity_nits=tuning.dome_intensity_nits,
                  color=list(tuning.dome_color),
                  purpose='ambient fill for recessed supports the ceiling panels cannot reach'),
        tonemapping=tonemapping_settings(tuning),
        counts=dict(sales_panel=len(_SALES_XS) * len(_SALES_YS),
                    backroom_tube=len(_BACKROOM_XS) * len(_BACKROOM_YS),
                    task_fill=len(tuple(fills))),
        limitations=[
            'Authoring lights is not a decode, exposure or physics result; tune intensity_scale '
            'against the tonemapper until QR white modules read below clipping on real pixels.',
            'Fixture positions/sizes/power mirror the Blender store; absolute nits depend on the '
            'render unit system, so calibrate the single intensity_scale rather than editing ratios.',
            'Ceiling panels have no line of sight into the shelf slot; keep the dome fill (or a '
            'local task fill) or the recessed shelf support stays dark. Prove with plan_restock_lighting.py.',
        ])
    return rig


def tonemapping_settings(tuning=None):
    """Blender's finish plus the closest Isaac RTX recommendation integration applies.

    Blender renders with the AgX view transform, 'Medium High Contrast' look and a
    -0.35 EV exposure (scripts/build_supermarket.setup_render), a modest HDRI world and
    a fog-glow bloom on the bright fixtures. The decode-critical target is contrast, not
    peak brightness: set a fixed exposure (no auto-exposure) so the white QR modules land
    ~0.7-0.85 and the black modules near 0; do not let the fixtures clip the markers flat.
    """
    tuning = tuning or LightRigTuning()
    return dict(
        blender_reference=dict(view_transform='AgX', look='AgX - Medium High Contrast',
                               exposure_ev=-0.35, world_hdri_strength=1.0,
                               bloom='compositor fog glow, threshold 1.0'),
        isaac_rtx_recommendation=dict(
            tonemapper='AgX if available, else a filmic operator (e.g. HejlDawson/Hable)',
            exposure_ev=-0.35, auto_exposure=False,
            note='fixed exposure keeps decode stable; verify QR white modules ~0.7-0.85, '
                 'not clipped to 1.0, and black modules near 0'),
        intensity_scale=tuning.intensity_scale)


def author_light_rig(stage, rig, *, parent_path='/World', tuning=None):
    """Author the rig's UsdLux fixtures (and dome) under ``parent_path``.

    Integration owns native assembly; this authors only lights, never cameras, scene
    geometry, render settings or a sensing result. Returns the authored prim paths.
    Idempotent per id. The ``tuning`` argument, if given, overrides the intensities in
    ``rig`` (e.g. to re-tune intensity_scale without rebuilding the rig dict).
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
    if not stage.GetPrimAtPath(parent_path):
        raise ValueError('light rig parent prim missing: ' + parent_path)
    paths = {}

    def safe(name):
        # USD prim names allow only [A-Za-z0-9_]; mangle fixture-id punctuation/signs.
        table = {'-': 'm', '+': 'p', '.': 'd', ':': '_'}
        return ''.join(table.get(ch, ch if (ch.isalnum() or ch == '_') else '_') for ch in name)

    def place(prim, world_from_light):
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        if world_from_light is None:
            return
        xform.AddTransformOp().Set(Gf.Matrix4d(np.asarray(world_from_light, float).T.tolist()))

    for f in rig['fixtures']:
        path = parent_path + '/RestockLight_' + safe(f['id'])
        rect = UsdLux.RectLight.Define(stage, path)
        rect.CreateWidthAttr(float(f['width']))
        rect.CreateHeightAttr(float(f['height']))
        rect.CreateColorAttr(Gf.Vec3f(*[float(c) for c in f['color']]))
        intensity = (tuning.intensity_nits(RectFixture(
            f['id'], f['kind'], tuple(f['center']), f['width'], f['height'], f['power_w'],
            tuple(f['color']))) if tuning is not None else f['intensity_nits'])
        rect.CreateIntensityAttr(float(intensity))
        rect.CreateExposureAttr(float((tuning or LightRigTuning()).exposure))
        rect.CreateNormalizeAttr(bool((tuning or LightRigTuning()).normalize))
        prim = rect.GetPrim()
        world = f.get('world_from_light')
        if world is None:
            # Down-facing: translate only; UsdLux emits local -Z == world -Z.
            UsdGeom.Xformable(prim).ClearXformOpOrder()
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*[float(c) for c in f['center']]))
        else:
            place(prim, world)
        paths[f['id']] = str(prim.GetPath())

    dome = rig.get('dome', {})
    if dome.get('enabled'):
        path = parent_path + '/RestockLight_dome_fill'
        dl = UsdLux.DomeLight.Define(stage, path)
        dl.CreateIntensityAttr(float(dome['intensity_nits']))
        dl.CreateColorAttr(Gf.Vec3f(*[float(c) for c in dome['color']]))
        paths['dome_fill'] = str(dl.GetPrim().GetPath())
    return paths


def author_camera_lamps(stage, config, *, parent_paths, tuning=None):
    """Author the camera-mounted sphere lamps in ``config['lamps']`` as UsdLux SphereLights.

    Parallel to ``config.author_camera_design``: ``parent_paths`` maps each lamp's parent
    ('map'/'cart_chassis') to an existing prim path. Omnidirectional, so no orientation.
    ``tuning`` (if given) overrides the stored per-lamp nits (e.g. to re-scale). Returns
    the authored prim paths; idempotent per id.
    """
    from pxr import Gf, UsdGeom, UsdLux
    paths = {}
    for lamp in config.get('lamps', ()):
        parent = lamp['parent']
        if parent not in parent_paths or not stage.GetPrimAtPath(parent_paths[parent]):
            raise ValueError('camera lamp parent missing: ' + parent)
        path = parent_paths[parent] + '/RestockLamp_' + lamp['id'].replace(':', '_').replace('-', 'm').replace('.', 'd')
        sphere = UsdLux.SphereLight.Define(stage, path)
        sphere.CreateRadiusAttr(float(lamp['radius']))
        sphere.CreateColorAttr(Gf.Vec3f(*[float(c) for c in lamp['color']]))
        if tuning is not None:
            nits = (tuning.luminous_efficacy_lm_per_w * lamp['radiance_w'] * tuning.intensity_scale)
        else:
            nits = lamp['intensity_nits']
        sphere.CreateIntensityAttr(float(nits))
        sphere.CreateNormalizeAttr(bool((tuning or LightRigTuning()).normalize))
        prim = sphere.GetPrim()
        UsdGeom.Xformable(prim).ClearXformOpOrder()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*[float(c) for c in lamp['center']]))
        paths[lamp['id']] = str(prim.GetPath())
    return paths
