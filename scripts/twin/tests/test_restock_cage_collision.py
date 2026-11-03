"""CPU proof of the opt-in sim-only carton-vs-cage contact filter (restock.cage_collision).

Pure-logic tier (no pxr): the switch is off by default and byte-identical when off, a typo raises, and
mission.json + the walkthrough caption name the workaround. USD tier (pxr / usd-core, no Isaac): the
FilteredPairsAPI is authored on the handling unit against EVERY rollwagen cage collider -- never the deck
or the upper shelf the carton rests on -- and it round-trips through the composed mission.usda.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import cage_collision as cc

ON = {'cage_collision': 'filtered'}


@pytest.mark.parametrize('config', [{}, {'cage_collision': None}, {'cage_collision': False},
                                    {'cage_collision': ''}, {'cage_collision': 'off'}, None])
def test_off_by_default_and_byte_identical(config):
    assert cc.mode(config) is None
    assert cc.mission_labels(config) == {}
    summary = {'state': 'succeeded', 'note': 'n', 'walkthrough_caption': 'base'}
    before = json.dumps(summary, sort_keys=True)
    assert json.dumps(cc.stamp_summary(summary, config), sort_keys=True) == before
    assert cc.caption('base', config) == 'base'
    assert cc.caption(None, config) is None


def test_unknown_value_raises_at_boot_and_is_recorded_not_hidden():
    with pytest.raises(ValueError, match='unknown cage collision mode'):
        cc.mode({'cage_collision': 'filterd'})
    summary = cc.stamp_summary({'note': ''}, {'cage_collision': 'filterd'})
    assert summary['cage_collision_mode'] == 'invalid' and 'filterd' in summary['cage_collision_error']


def test_on_labels_mission_json_and_caption_once():
    assert cc.mode({'cage_collision': ' Filtered '}) == 'filtered'
    summary = {'note': 'Mission success alone does not verify video.', 'walkthrough_caption': 'start_at=transfer'}
    cc.stamp_summary(summary, ON)
    cc.stamp_summary(summary, ON)                     # idempotent: both summary writers may stamp
    assert summary['cage_collision_mode'] == 'filtered'
    assert summary['workarounds'] == ['cage_collision=filtered']
    assert summary['note'].count('CAGE_COLLISION=filtered') == 1
    assert summary['walkthrough_caption'] == f'start_at=transfer | {cc.CAPTION}'
    assert cc.caption(None, ON) == cc.CAPTION
    assert 'CAGE COLLISION FILTERED' in cc.CAPTION and 'workaround' in cc.CAPTION


def test_demo_cut_shows_the_label():
    """The demo cut picks the workaround up from mission.json with no cut-side change."""
    from twin.demo.run_evidence import workaround_labels

    class Run:
        mission = cc.stamp_summary({'state': 'succeeded'}, ON)
        events, caption = [], None

    labels = {(lab.key, lab.value) for lab in workaround_labels(Run)}
    assert ('cage_collision_mode', 'filtered') in labels
    assert ('workarounds', 'cage_collision=filtered') in labels


def test_registered_as_a_runtime_key_for_the_activation_preflight():
    from twin.restock_manipulation.config_preflight import KEY_CONSUMER, RUNTIME, check_activation
    assert KEY_CONSUMER['cage_collision'] == RUNTIME
    assert check_activation({}, ON, ['cage_collision']) == []
    assert 'MIS-PLACED' in check_activation(ON, {}, ['cage_collision'])[0]


# ---------------- USD tier (pxr, no Isaac) ----------------

def _cart_with_unit():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    from twin.mobile_assets import author_cart
    from twin.restock.scene import promote_stock
    stage = Usd.Stage.CreateInMemory()
    cart = author_cart(stage)
    UsdGeom.Xform.Define(stage, '/RestockScenario')
    mesh = UsdGeom.Mesh.Define(stage, '/RestockScenario/HandlingUnit')
    mesh.CreatePointsAttr([(x, y, z) for x in (-.3, .3) for y in (-.2, .2) for z in (-.08, .08)])
    promote_stock(stage, '/RestockScenario/HandlingUnit', 'juice_orange', 0.45)
    return stage, cart


def test_filter_targets_every_cage_collider_and_never_a_support():
    pytest.importorskip('pxr')          # USD tier: skip precisely (not fail) where pxr/usd-core is absent
    from pxr import UsdPhysics
    stage, cart = _cart_with_unit()
    ev = cc.author_filter(stage, '/RestockScenario/HandlingUnit', cart['body'])
    unit = stage.GetPrimAtPath('/RestockScenario/HandlingUnit')
    assert unit.HasAPI(UsdPhysics.FilteredPairsAPI)
    targets = [str(p) for p in UsdPhysics.FilteredPairsAPI(unit).GetFilteredPairsRel().GetTargets()]
    assert targets == ev['filtered']
    names = {t.rsplit('/', 1)[-1] for t in targets}
    # author_cart: 4 corner posts, 2 side walls x (9 horizontal + 7 vertical) bars, back wall 9 + 5 bars.
    assert len(targets) == 4 + 2 * (9 + 7) + 9 + 5 == 50
    assert {f'post_{i}' for i in range(4)} <= names
    assert {'side_0_horizontal_6', 'side_1_vertical_7', 'back_horizontal_9', 'back_vertical_5'} <= names
    # The carton still rests on the shelf / deck and still meets the drawbar.
    kept = {p.rsplit('/', 1)[-1] for p in ev['kept']}
    assert kept == {'deck', 'shelf', 'drawbar'} and not names & kept
    every_collider = {str(c.GetPath()) for c in stage.GetPrimAtPath(cart['body']).GetChildren()
                      if c.HasAPI(UsdPhysics.CollisionAPI)}
    assert set(targets) | set(ev['kept']) == every_collider
    assert unit.GetCustomDataByKey('restock:cage_collision') == 'filtered'


def test_filter_refuses_a_missing_unit_body_or_cage():
    pytest.importorskip('pxr')          # USD tier: skip precisely (not fail) where pxr/usd-core is absent
    from pxr import Usd, UsdGeom
    stage, cart = _cart_with_unit()
    with pytest.raises(ValueError, match='not a rigid body'):
        cc.author_filter(stage, '/RestockScenario', cart['body'])
    with pytest.raises(ValueError, match='chassis body absent'):
        cc.author_filter(stage, '/RestockScenario/HandlingUnit', '/Nope')
    bare = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(bare, '/Rollwagen/chassis')
    with pytest.raises(ValueError, match='no rollwagen cage colliders'):
        cc.cage_collider_paths(bare, '/Rollwagen/chassis')


def _fixture_inputs(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    source, robot = tmp_path / 'store.usda', tmp_path / 'fixture_robot.usda'
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, '/Supermarket')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Mesh.Define(stage, '/Supermarket/GRP_Fixtures/Case')
    stage.GetRootLayer().Save()
    arm = Usd.Stage.CreateNew(str(robot))
    root = UsdGeom.Xform.Define(arm, '/FixtureRobot')
    arm.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    arm.GetRootLayer().Save()
    return source, robot


def test_scene_build_authors_the_filter_into_mission_usda_only_when_on(tmp_path):
    """The composed scratch mission.usda carries the filter (evidence on disk); off => no FilteredPairsAPI."""
    pytest.importorskip('pxr')
    from pxr import Usd, UsdPhysics
    from twin.restock.scene import build
    source, robot = _fixture_inputs(tmp_path)
    kw = dict(robot_pose=(0, 0, 0), cart_pose=(-1.27, 0, 0), stock_path='/Supermarket/GRP_Fixtures/Case',
              sku='milk', stock_mass_kg=2.)
    off = build(source, robot, tmp_path / 'off.usda', **kw)
    assert 'cage_collision' not in off
    composed_off = Usd.Stage.Open(str(tmp_path / 'off.usda'))
    assert not any(p.HasAPI(UsdPhysics.FilteredPairsAPI) for p in composed_off.Traverse())
    on = build(source, robot, tmp_path / 'on.usda', cage_collision='filtered', **kw)
    assert on['cage_collision']['unit'] == on['stock'] and len(on['cage_collision']['filtered']) == 50
    text = (tmp_path / 'on.usda').read_text()
    assert 'PhysicsFilteredPairsAPI' in text and 'physics:filteredPairs' in text
    composed_on = Usd.Stage.Open(str(tmp_path / 'on.usda'))
    unit = composed_on.GetPrimAtPath(on['stock'])
    targets = {str(p) for p in UsdPhysics.FilteredPairsAPI(unit).GetFilteredPairsRel().GetTargets()}
    assert '/Rollwagen/chassis/post_3' in targets and '/Rollwagen/chassis/shelf' not in targets
    with pytest.raises(ValueError, match='unknown cage collision mode'):
        build(source, robot, tmp_path / 'typo.usda', cage_collision='filter', **kw)
    assert not (tmp_path / 'typo.usda').exists()      # refused before any output is created
