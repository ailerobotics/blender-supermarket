"""CPU proof of the config-switched, visual-only material pass (restock.visuals).

Pure-logic tier (no pxr): the switch defaults ON, accepts the documented falsey
forms, and the sRGB->linear conversion is correct. USD tier (pxr / usd-core, no
Isaac): the pass binds UR-scheme / grey-base / galvanised-cart materials onto the
real link prim names, re-tones the backroom concrete, and NEVER touches geometry,
colliders, physics, the carton QR label or the support/shelf the cameras decode.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import visuals as V


@pytest.mark.parametrize('config,want', [
    # DEFAULT OFF: absent/None/false => byte-identical.
    ({}, False), ({'visuals': None}, False), (None, False), ({'visuals': False}, False),
    ({'visuals': {'enabled': False}}, False), ({'visuals': 'off'}, False),
    ({'visuals': '0'}, False), ({'visuals': 'false'}, False),
    # Explicit opt-in.
    ({'visuals': True}, True), ({'visuals': {'enabled': True}}, True),
    ({'visuals': {}}, True), ({'visuals': 'on'}, True),
])
def test_switch_defaults_off_and_honours_opt_in_forms(config, want):
    assert V.enabled(config) is want


def test_srgb_to_linear_endpoints_and_monotonic():
    assert V._linear('#000000') == (0.0, 0.0, 0.0)
    assert V._linear('#FFFFFF') == pytest.approx((1.0, 1.0, 1.0))
    mid = V._linear('#808080')[0]
    assert 0.2 < mid < 0.25  # 0.5 sRGB -> ~0.214 linear, darker than 0.5


# ---- USD tier -------------------------------------------------------------- #
pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Gf  # noqa: E402


def _fixture_robot(stage, root='/MobileRobot'):
    """A stand-in with the real imported link names, each with a visuals mesh and a
    separate collisions mesh (so we can prove colliders are never re-bound)."""
    UsdGeom.Xform.Define(stage, root)
    links = (list(V._ARM) + list(V._BASE) + ['surface_plate_1', 'suction_cups', 'lidar_frame'])
    for link in links:
        UsdGeom.Xform.Define(stage, f'{root}/{link}')
        vis = UsdGeom.Mesh.Define(stage, f'{root}/{link}/visuals')
        col = UsdGeom.Mesh.Define(stage, f'{root}/{link}/collisions')
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    return root


def _fixture_cart(stage, root='/Rollwagen'):
    UsdGeom.Xform.Define(stage, root)
    body = f'{root}/chassis'
    UsdGeom.Xform.Define(stage, body)
    for name in ('deck', 'shelf', 'post_0', 'side_0_horizontal_1', 'back_vertical_1'):
        UsdGeom.Cube.Define(stage, f'{body}/{name}')
    for i in range(2):
        UsdGeom.Xform.Define(stage, f'{root}/fork_{i}')
        UsdGeom.Cube.Define(stage, f'{root}/fork_{i}/shape')
        UsdGeom.Xform.Define(stage, f'{root}/wheel_{i}')
        UsdGeom.Cylinder.Define(stage, f'{root}/wheel_{i}/shape')
    return {'root': root, 'body': body, 'dock': body + '/dock',
            'stock_supports': [body + '/deck', body + '/shelf']}


def _assets(stage):
    return {'robot_root': _fixture_robot(stage), 'cart': _fixture_cart(stage)}


def test_binds_ur_scheme_to_visuals_only_and_leaves_colliders_untouched():
    stage = Usd.Stage.CreateInMemory()
    from pxr import Sdf
    root = Sdf.Path('/Supermarket')
    UsdGeom.Xform.Define(stage, root)
    manifest = V.apply(stage, _assets(stage), root)
    # The shoulder joint cap is UR blue, the forearm tube is UR silver -- bound to visuals.
    def bound_mat(path):
        prim = stage.GetPrimAtPath(path)
        return UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    shoulder = bound_mat('/MobileRobot/ur_shoulder_link/visuals')
    forearm = bound_mat('/MobileRobot/ur_forearm_link/visuals')
    assert shoulder and forearm and shoulder.GetPath() != forearm.GetPath()
    shoulder_rgb = UsdShade.Shader(stage.GetPrimAtPath(str(shoulder.GetPath()) + '/Surface')) \
        .GetInput('diffuseColor').Get()
    assert shoulder_rgb == pytest.approx(Gf.Vec3f(*V._linear(V.UR_BLUE)))
    # The link prim itself and its collision mesh carry NO material binding.
    assert not bound_mat('/MobileRobot/ur_forearm_link')
    assert not bound_mat('/MobileRobot/ur_forearm_link/collisions')
    assert manifest['bound_count'] > 10


def test_cart_deck_is_plastic_and_mesh_is_galvanised_steel():
    stage = Usd.Stage.CreateInMemory()
    from pxr import Sdf
    UsdGeom.Xform.Define(stage, '/Supermarket')
    V.apply(stage, _assets(stage), Sdf.Path('/Supermarket'))
    def mat_name(path):
        m = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial()[0]
        return m.GetPath().name if m else None
    assert mat_name('/Rollwagen/chassis/deck') == 'cart_plastic'
    assert mat_name('/Rollwagen/chassis/post_0') == 'cart_galv'
    assert mat_name('/Rollwagen/chassis/side_0_horizontal_1') == 'cart_galv'
    assert mat_name('/Rollwagen/wheel_0/shape') == 'cart_wheel'


def test_ancestor_bind_resolves_onto_a_deep_unbound_referenced_mesh(tmp_path):
    """The real imported UR asset puts the visual mesh deep under the link's `visuals`
    Xform (e.g. .../visuals/forearm/Scene/mesh) with NO binding of its own; the pass
    binds at the `visuals` Xform. Prove that ancestor binding resolves onto the deep
    mesh even when the geometry arrives via a reference (the import's composition arc),
    so the pass is effective and not a silent no-op."""
    # Asset layer: a link subtree whose mesh carries no material binding.
    asset = tmp_path / 'arm_asset.usda'
    ast = Usd.Stage.CreateNew(str(asset))
    UsdGeom.Xform.Define(ast, '/visuals_src')
    UsdGeom.Mesh.Define(ast, '/visuals_src/forearm/Scene/mesh')
    ast.SetDefaultPrim(ast.GetPrimAtPath('/visuals_src'))
    ast.GetRootLayer().Save()
    # Main stage: robot link whose `visuals` child REFERENCES the asset subtree.
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/Supermarket')
    UsdGeom.Xform.Define(stage, '/MobileRobot')
    UsdGeom.Xform.Define(stage, '/MobileRobot/ur_forearm_link')
    vis = UsdGeom.Xform.Define(stage, '/MobileRobot/ur_forearm_link/visuals')
    vis.GetPrim().GetReferences().AddReference(str(asset))
    cart = _fixture_cart(stage)
    from pxr import Sdf
    V.apply(stage, {'robot_root': '/MobileRobot', 'cart': cart}, Sdf.Path('/Supermarket'))
    mesh = stage.GetPrimAtPath('/MobileRobot/ur_forearm_link/visuals/forearm/Scene/mesh')
    assert mesh and mesh.IsValid()
    bound = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    assert bound and bound.GetPath().name == 'ur_forearm_link'  # the silver tube material


def test_disabled_is_a_noop():
    stage = Usd.Stage.CreateInMemory()
    from pxr import Sdf
    UsdGeom.Xform.Define(stage, '/Supermarket')
    assets = _assets(stage)
    before = sorted(str(p.GetPath()) for p in stage.Traverse())
    # Caller is responsible for the switch; apply() only runs when enabled.
    if V.enabled({'visuals': False}):
        V.apply(stage, assets, Sdf.Path('/Supermarket'))
    after = sorted(str(p.GetPath()) for p in stage.Traverse())
    assert after == before  # no /Visuals scope, no bindings


def test_env_retone_skips_textured_diffuse_and_overrides_flat_concrete():
    stage = Usd.Stage.CreateInMemory()
    from pxr import Sdf
    UsdGeom.Xform.Define(stage, '/Supermarket')
    # flat-colour concrete: re-toned; textured floor: skipped.
    for name, textured in (('FX_concrete', False), ('floor_tiles_08', True)):
        mat = UsdShade.Material.Define(stage, f'/Supermarket/_materials/{name}')
        sh = UsdShade.Shader.Define(stage, f'{mat.GetPath()}/Principled_BSDF')
        sh.CreateIdAttr('UsdPreviewSurface')
        di = sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
        if textured:
            tex = UsdShade.Shader.Define(stage, f'{mat.GetPath()}/Image_Texture')
            tex.CreateIdAttr('UsdUVTexture')
            di.ConnectToSource(tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
        else:
            di.Set(Gf.Vec3f(0.26, 0.25, 0.23))
    manifest = V.apply(stage, _assets(stage), Sdf.Path('/Supermarket'))
    conc = UsdShade.Shader(stage.GetPrimAtPath('/Supermarket/_materials/FX_concrete/Principled_BSDF'))
    want = V._ENV_MATERIALS['FX_concrete'][0]
    got = conc.GetInput('diffuseColor').Get()
    assert tuple(got) == pytest.approx(want)
    assert got[0] > 0.26  # lighter than the original drab concrete
    assert any('FX_concrete' in p for p in manifest['env_retoned'])
    assert not any('floor_tiles_08' in p for p in manifest['env_retoned'])
