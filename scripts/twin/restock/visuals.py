"""Config-switched, visual-only material pass for the restock mission scene.

Pure USD (``pxr``) — no Isaac, no GPU. Called once at scene composition
(``scene.build``) AFTER the robot, rollwagen and scenario are authored and BEFORE
the stage is saved. It binds ``UsdPreviewSurface`` materials so the rendered
walkthrough reads as a real UR10 (light-aluminium link tubes, UR-blue joint
caps, dark joint housings) on a neutral-grey mobile base towing a galvanised
steel roll-container, over a sealed-concrete backroom — instead of plain white.

HARD SCOPE — visual materials ONLY. This module never creates or edits geometry,
colliders, physics, mass, joints, cameras, lights, the carton QR label, the
support/shelf markers the station cameras decode, or any xform. It only:
  * defines ``UsdShade`` materials under ``/Visuals`` and
  * binds them to existing robot / cart gprims (``MaterialBindingAPI``), and
  * overrides the ``diffuseColor``/``roughness`` of a few *backroom* store
    materials (sublayered read-only store; the override lives in the output
    root layer, so the committed store file is never touched).
**Default OFF** (``visuals`` key absent ⇒ the pass is a no-op and the scene is
byte-identical to the pre-visuals look). A run opts in with ``"visuals": true`` /
``{"enabled": true}``.

Palette (sRGB hex; authored into USD as linear diffuseColor). The UR values
follow Universal Robots' own scheme — grey-aluminium housings (RAL 9007), light
aluminium link tubes, and the signature light-blue joint caps:
  UR_SILVER   #C7CACC  light aluminium link tubes (upper-arm / forearm)
  UR_BLUE     #3D7AB8  joint caps / rings (shoulder + wrist 1/2/3)
  UR_HOUSING  #53565A  dark joint housing / UR base unit (RAL 9007-ish)
  TOOL_STEEL  #6A6E72  gripper plates / suction tool (machined steel)
  BASE_GREY   #8A8D90  mobile base body (neutral grey, not white)
  WHEEL_DARK  #2E3134  drive wheels, casters, bumpers (dark grey rubber)
  HITCH_DARK  #3A3D40  drawbar / hitch steel
  CART_GALV   #9DA3A8  galvanised steel roll-container mesh + frame (metallic)
  CART_PLASTIC#1E57A0  coloured plastic base plate (supermarket roll-cage blue)
  CART_WHEEL  #26282B  roll-container swivel casters
Backroom floor/walls are re-toned in place (sealed concrete, warm off-white plaster)
via direct-linear overrides on the store's own materials — see ``_ENV_MATERIALS``.
"""
from __future__ import annotations

KEY = 'visuals'
_OFF = ('', '0', 'off', 'none', 'false')

# sRGB hex palette (see module docstring). Converted to linear at author time.
UR_SILVER = '#C7CACC'
UR_BLUE = '#3D7AB8'
UR_HOUSING = '#53565A'
TOOL_STEEL = '#6A6E72'
BASE_GREY = '#8A8D90'
WHEEL_DARK = '#2E3134'
HITCH_DARK = '#3A3D40'
CART_GALV = '#9DA3A8'
CART_PLASTIC = '#1E57A0'
CART_WHEEL = '#26282B'
# Backroom floor/wall re-tone values are direct-linear tuples in _ENV_MATERIALS below
# (to match the store's own diffuseColor convention), not sRGB hex.

# Robot link -> (material name, hex, roughness, metallic). Keys are the imported
# URDF link prim names under the robot root (see twin.mobile_assets / the Isaac
# URDF importer). Any link absent from the imported robot is silently skipped, so
# a reduced fixture robot composes fine.
_ARM = {
    # UR10 arm: silver tubes, blue joints, dark base/housing.
    'ur_base_link': (UR_HOUSING, 0.5, 0.6),
    'ur_base_link_inertia': (UR_HOUSING, 0.5, 0.6),
    'ur_base': (UR_HOUSING, 0.5, 0.6),
    'ur_shoulder_link': (UR_BLUE, 0.45, 0.0),
    'ur_upper_arm_link': (UR_SILVER, 0.4, 0.7),
    'ur_forearm_link': (UR_SILVER, 0.4, 0.7),
    'ur_wrist_1_link': (UR_BLUE, 0.45, 0.0),
    'ur_wrist_2_link': (UR_BLUE, 0.45, 0.0),
    'ur_wrist_3_link': (UR_SILVER, 0.4, 0.7),
    'ur_flange': (TOOL_STEEL, 0.4, 0.8),
    'ur_tool0': (TOOL_STEEL, 0.4, 0.8),
}
_TOOL_PREFIXES = ('surface_plate', 'suction_cups', 'big_cube', 'cube_without_surface_plates')
_BASE = {
    'base_link': (BASE_GREY, 0.55, 0.1),
    'arm_mount': (BASE_GREY, 0.55, 0.1),
    'left_wheel': (WHEEL_DARK, 0.85, 0.0),
    'right_wheel': (WHEEL_DARK, 0.85, 0.0),
    'front_caster_fork': (HITCH_DARK, 0.6, 0.3),
    'rear_caster_fork': (HITCH_DARK, 0.6, 0.3),
    'front_caster_wheel': (WHEEL_DARK, 0.85, 0.0),
    'rear_caster_wheel': (WHEEL_DARK, 0.85, 0.0),
    'hitch': (HITCH_DARK, 0.5, 0.4),
}
# Backroom store materials to re-tone -> (linear diffuseColor, roughness). Values are
# DIRECT linear (not sRGB) to match the store's own Blender-exported diffuseColor
# convention (e.g. FX_concrete ships at ~0.26 linear): a lighter, cooler sealed-concrete
# backroom floor and a warm off-white wall plaster. Only backroom surfaces; the
# sales-floor floor/walls keep their textured maps. Missing materials are skipped, and
# the override lives in the output root layer so the sublayered store stays read-only.
_ENV_MATERIALS = {
    'FX_concrete': ((0.32, 0.335, 0.355), 0.5),   # sealed/polished backroom concrete
    'FX_ceiling': ((0.82, 0.80, 0.74), 0.8),      # warm off-white wall plaster
}


def enabled(config) -> bool:
    """Whether the visuals pass runs. **Default OFF** (absent/None ⇒ disabled), so a
    run with no ``visuals`` key composes byte-identically to today's plain look. The
    integration lane opts a run in with ``"visuals": true`` / ``{"enabled": true}`` /
    ``"on"``. Mirrors the opt-in style of ``cage_collision``."""
    raw = config.get(KEY) if isinstance(config, dict) else config
    if raw is None or raw is False:
        return False
    if raw is True:
        return True
    if isinstance(raw, dict):
        return bool(raw.get('enabled', True))  # a present block is an explicit opt-in
    if str(raw).strip().lower() in _OFF:
        return False
    return True


def _linear(hex_srgb):
    """sRGB hex string -> linear RGB tuple for a USD ``diffuseColor``."""
    h = hex_srgb.lstrip('#')
    srgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)


def _material(stage, scope_path, name, hex_srgb, roughness=0.5, metallic=0.0):
    from pxr import Sdf, UsdShade, Gf
    path = f'{scope_path}/{name}'
    existing = stage.GetPrimAtPath(path)
    if existing:
        return UsdShade.Material(existing)
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_linear(hex_srgb)))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(metallic))
    mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput('surface', Sdf.ValueTypeNames.Token))
    return mat


def _bind(stage, prim_path, material):
    """Bind ``material`` at the LINK prim ``prim_path`` (visual only); the render-purpose
    binding inherits onto the link's visual meshes.

    Bind at the link, never at its ``visuals`` child. In the imported robot each
    ``<link>/visuals`` carries a REFERENCE into the URDF-importer physics layer, and
    several are placeholder/frame links (``ur_base``, ``ur_flange``, ``*_frame`` …) whose
    reference is UNRESOLVED (dangling). Authoring a material-binding override directly
    onto those prims floods Isaac with ``Unresolved reference`` errors and corrupts the
    articulation composition — the warm9 failure where the towed base stayed parked
    (533/540 ticks) the whole outbound and made no progress. The link prim itself carries
    no such reference, so authoring there never touches the physics layer; the binding
    still inherits onto the visual meshes at render, while PhysX reads wheel friction from
    the separate physics-purpose binding on the collider (configure_wheel_traction).
    Returns the bound path or None."""
    from pxr import UsdShade
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return str(prim.GetPath())


def _retone_env_material(stage, mat_path, rgb_linear, roughness):
    """Override the ``diffuseColor``/``roughness`` of an existing store material's
    ``UsdPreviewSurface`` shader in the output root layer (store stays read-only).
    ``rgb_linear`` is a direct linear triple (matches the store's convention). Skips
    a textured (connected) diffuse input and a missing material."""
    from pxr import Sdf, UsdShade, Gf
    mat = stage.GetPrimAtPath(mat_path)
    if not mat or not mat.IsValid():
        return None
    for shp in mat.GetChildren():
        shader = UsdShade.Shader(shp)
        if not shader or shader.GetIdAttr().Get() != 'UsdPreviewSurface':
            continue
        diffuse = shader.GetInput('diffuseColor')
        if diffuse and diffuse.GetConnectedSources()[0]:
            return None  # textured surface: leave its maps alone
        over = stage.OverridePrim(shp.GetPath())
        over.CreateAttribute('inputs:diffuseColor', Sdf.ValueTypeNames.Color3f, False).Set(
            Gf.Vec3f(*rgb_linear))
        over.CreateAttribute('inputs:roughness', Sdf.ValueTypeNames.Float, False).Set(float(roughness))
        return str(shp.GetPath())
    return None


def apply(stage, assets, root, *, cart_colors=True):
    """Author + bind the visual materials. ``assets`` is ``mobile_assets.author``'s
    result (robot_root + cart paths); ``root`` is the store default-prim ``Sdf.Path``.
    Returns a manifest of what was bound (counts + paths) for logging / tests.

    Idempotent per stage: a second call reuses the ``/Visuals`` materials. Every
    bind/override is guarded on prim existence, so a fixture robot/store with only
    some of the prims composes without error.
    """
    from pxr import UsdGeom
    scope = '/Visuals'
    UsdGeom.Scope.Define(stage, scope)
    bound = []

    robot_root = assets['robot_root']
    for link, (hex_srgb, rough, metal) in {**_ARM, **_BASE}.items():
        mat = _material(stage, scope, link, hex_srgb, rough, metal)
        p = _bind(stage, f'{robot_root}/{link}', mat)
        if p:
            bound.append(p)
    # Gripper plates / suction tool share one machined-steel look.
    tool_mat = _material(stage, scope, 'tool_steel', TOOL_STEEL, 0.4, 0.8)
    robot = stage.GetPrimAtPath(robot_root)
    if robot and robot.IsValid():
        for child in robot.GetChildren():
            if any(child.GetName().startswith(pfx) for pfx in _TOOL_PREFIXES):
                p = _bind(stage, child.GetPath(), tool_mat)
                if p:
                    bound.append(p)

    # Rollwagen: galvanised steel mesh/frame, blue plastic base plate, dark casters.
    cart = assets.get('cart') if isinstance(assets, dict) else None
    if cart_colors and cart:
        from pxr import Usd, UsdGeom as _UG, UsdShade
        galv = _material(stage, scope, 'cart_galv', CART_GALV, 0.45, 0.85)
        plastic = _material(stage, scope, 'cart_plastic', CART_PLASTIC, 0.35, 0.0)
        cwheel = _material(stage, scope, 'cart_wheel', CART_WHEEL, 0.8, 0.0)
        body = cart.get('body')
        body_prim = stage.GetPrimAtPath(body) if body else None
        if body_prim and body_prim.IsValid():
            for p in Usd.PrimRange(body_prim):
                if not p.IsA(_UG.Gprim):
                    continue
                name = p.GetName()
                mat = plastic if name == 'deck' else galv
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat)
                bound.append(str(p.GetPath()))
        # Swivel casters/wheels are siblings of the chassis body under the cart root.
        cart_root = cart.get('root')
        cart_prim = stage.GetPrimAtPath(cart_root) if cart_root else None
        if cart_prim and cart_prim.IsValid():
            for child in cart_prim.GetChildren():
                if child.GetName().startswith(('wheel_', 'fork_')):
                    for p in Usd.PrimRange(child):
                        if p.IsA(_UG.Gprim):
                            UsdShade.MaterialBindingAPI.Apply(p).Bind(cwheel)
                            bound.append(str(p.GetPath()))

    # Backroom store materials: re-tone diffuse/roughness (store stays read-only).
    env = []
    mats_scope = root.AppendChild('_materials')
    for name, (rgb_linear, rough) in _ENV_MATERIALS.items():
        r = _retone_env_material(stage, mats_scope.AppendChild(name), rgb_linear, rough)
        if r:
            env.append(r)

    return {'enabled': True, 'materials_scope': scope, 'bound_count': len(bound),
            'bound': bound, 'env_retoned': env}
