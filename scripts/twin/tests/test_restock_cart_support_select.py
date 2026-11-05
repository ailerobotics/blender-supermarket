"""CPU tests for sim-cart-support-select-v1 -- selectable place support + honest collider sweep.

The cold ground-truth collect sets the welded carton (0.45 x 0.30 x 0.18 m) down on the Rollwagen.
The DECK (`/Rollwagen/chassis/deck`) is UNPLACEABLE FROM ABOVE -- its own upper shelf blocks the
vertical descent (coldwin46: the carton drives -12 mm into `/Rollwagen/chassis/shelf` mid-descent).
The UPPER SHELF (`/Rollwagen/chassis/shelf`) is open from above: a vertical cw43-style descent clears
the mouth posts by >= 177 mm. This suite proves the selector, the honest collider sweep (carton
filtered vs the cage, pad not), the footprint rule, the chosen xy, and the preflight.

The sweep + cart geometry need `usd-core` (`author_cart`); the pure selector/classification tests do
not. Run:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_restock_cart_support_select.py
"""
import math

import pytest

from twin.restock_manipulation import cart_support
from twin.restock_manipulation.isaac_factory import parse_manipulation_config
from twin.restock_manipulation import config_preflight as cpf

DECK = '/Rollwagen/chassis/deck'
SHELF = '/Rollwagen/chassis/shelf'
STOCK = [DECK, SHELF]


# --------------------------------------------------------------- 1. selectable support (pure)

def test_resolve_support_prim_default_is_deck():
    # Absent selector -> stock_supports[0] (deck): byte-identical with the legacy hardcode.
    assert cart_support.resolve_support_prim(STOCK, None) == DECK


def test_resolve_support_prim_by_index():
    assert cart_support.resolve_support_prim(STOCK, 0) == DECK
    assert cart_support.resolve_support_prim(STOCK, 1) == SHELF
    assert cart_support.resolve_support_prim(STOCK, -1) == SHELF


def test_resolve_support_prim_by_name_and_path():
    assert cart_support.resolve_support_prim(STOCK, 'shelf') == SHELF
    assert cart_support.resolve_support_prim(STOCK, 'deck') == DECK
    assert cart_support.resolve_support_prim(STOCK, SHELF) == SHELF


def test_resolve_support_prim_fails_closed():
    with pytest.raises(ValueError):
        cart_support.resolve_support_prim(STOCK, 5)            # index out of range
    with pytest.raises(ValueError):
        cart_support.resolve_support_prim(STOCK, 'drawbar')   # not a support
    with pytest.raises(ValueError):
        cart_support.resolve_support_prim(STOCK, True)        # bool is not an index
    with pytest.raises(ValueError):
        cart_support.resolve_support_prim([], 'shelf')        # no supports


def _cp(**over):
    cp = {'location': 'rollwagen', 'load_center_body_xy': [0.135, 0.0],
          'support_z_world': 0.9712, 'free_dimensions': [0.65, 0.45, 0.30]}
    cp.update(over)
    return {'manipulation': {
        'profiles': [{'sku': 'juice_orange', 'dimensions': [.45, .30, .18],
                      'tcp_quaternion': [.5, .5, .5, .5], 'tcp_surface_offset': .0025,
                      'grasp_mode': 'side'}],
        'stations': {'source': {'slots': [{'id': 's0'}], 'base_from_camera': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
                     'shelf': {'slots': [{'id': 'd0'}], 'base_from_camera': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}},
        'shelfbot_root': '/home/x/shelfbot_newgripper', 'walls': {},
        'voxel_dims': [2.4, 2.4, 2.4], 'voxel_size': .02,
        'cart_placement': cp}}


def test_parse_cart_placement_absent_support_prim_is_byte_identical():
    c = parse_manipulation_config(_cp(), 'juice_orange')
    assert 'support_prim' not in c.cart_placement       # absent key -> legacy default downstream


def test_parse_cart_placement_support_prim_parsed():
    assert parse_manipulation_config(_cp(support_prim='shelf'), 'juice_orange').cart_placement['support_prim'] == 'shelf'
    assert parse_manipulation_config(_cp(support_prim=1), 'juice_orange').cart_placement['support_prim'] == 1


def test_parse_cart_placement_support_prim_malformed_rejected():
    with pytest.raises(ValueError):
        parse_manipulation_config(_cp(support_prim=1.5), 'juice_orange')
    with pytest.raises(ValueError):
        parse_manipulation_config(_cp(support_prim=True), 'juice_orange')


# --------------------------------------------------------------- 2. honest collider classification (pure)

def test_honest_collider_parts_classes():
    from twin.restock_manipulation import place_clearance as pc
    # support dropped; cage bodies -> pad only; other kept -> carton + pad.
    assert pc.honest_collider_parts('shelf', 'shelf') == ()
    assert pc.honest_collider_parts('post_2', 'shelf') == ('pad',)
    assert pc.honest_collider_parts('side_0_horizontal_6', 'shelf') == ('pad',)
    assert pc.honest_collider_parts('back_vertical_3', 'shelf') == ('pad',)
    assert pc.honest_collider_parts('deck', 'shelf') == ('carton', 'pad')
    assert pc.honest_collider_parts('drawbar', 'shelf') == ('carton', 'pad')
    # on a deck place the shelf is a NON-support kept body the carton can strike.
    assert pc.honest_collider_parts('shelf', 'deck') == ('carton', 'pad')
    assert pc.is_cage_body('post_0') and pc.is_cage_body('side_1_vertical_2')
    assert not pc.is_cage_body('deck') and not pc.is_cage_body('shelf') and not pc.is_cage_body('drawbar')


# --------------------------------------------------------------- 2/3. honest sweep + footprint (usd-core)

pxr = pytest.importorskip('pxr')  # author_cart needs usd-core

from twin.restock_manipulation import place_clearance as pc  # noqa: E402

YAW = -math.pi / 2.0
CARTON = pc.CARTON_DIMS_M
MARGIN = 0.030


@pytest.fixture(scope='module')
def bodies():
    return pc.rollwagen_collider_boxes()


@pytest.fixture(scope='module')
def geom():
    return pc.GripperCartonGeometry.shipped()


def _descent(load_xy, support_name, bodies, standoff, n=20):
    support_z = bodies[support_name][1][2]       # support top face (cart body frame)
    return pc.place_descent_poses(load_xy, support_z, YAW, carton_h=CARTON[2], standoff_m=standoff, n=n)


def test_cw43_shelf_place_reads_clear_honest(bodies, geom):
    """cw43 (legacy xy 0.18) at the SHELF: the honest sweep reads CLEAR (it physically released without
    contact). Every non-support body clears by the 30 mm margin; the mouth posts clear by >> margin."""
    poses = _descent((0.18, 0.0), 'shelf', bodies, standoff=0.12)
    _, pbm = pc.sweep_clearance(poses, geom, bodies, honest_support='shelf')
    assert 'shelf' not in pbm                                  # the support is dropped from the strike check
    assert pc.clearance_reject(pbm, support=('shelf',)) is None
    assert pbm['post_2'] > MARGIN and pbm['post_3'] > MARGIN   # mouth posts clear (pad vs post)
    assert min(pbm['post_2'], pbm['post_3']) > 0.10            # by a wide margin (>= 100 mm)


def test_cw46_deck_place_honest_strikes_the_upper_shelf(bodies, geom):
    """cw46 (xy 0.18, full 0.68 m vertical descent to the DECK) STRIKES the upper shelf: the deck is
    unplaceable from above because its own `/Rollwagen/chassis/shelf` blocks the descent. The honest
    sweep names `shelf` (a non-support KEPT collider the CARTON hits), negative gap -- NOT a post."""
    poses = _descent((0.18, 0.0), 'deck', bodies, standoff=0.68, n=40)
    _, pbm = pc.sweep_clearance(poses, geom, bodies, honest_support='deck')
    assert pbm['shelf'] < 0.0                                  # carton penetrates the shelf on the way down
    reject = pc.clearance_reject(pbm, support=('deck',))
    assert reject is not None and 'shelf' in reject
    # it is NOT a mouth-post strike: the pad clears post_2 by a wide margin at the deck target.
    assert pbm['post_2'] > MARGIN


def test_old_minus93mm_bar_is_a_noncollider_for_the_carton(bodies, geom):
    """The historical lumped sweep's -93 mm reading was the carton box vs `side_0_horizontal_6` (an
    upper SIDE bar). That is a CAGE body -- cage_collision filters it out of CARTON contacts -- so the
    honest sweep counts it against the pad only, never the carton box. Prove the carton-box reading is
    the tight one and the honest classification drops it to pad-only."""
    bar = 'side_0_horizontal_6'
    assert pc.is_cage_body(bar)
    assert pc.honest_collider_parts(bar, 'shelf') == ('pad',)   # carton excluded
    # At the legacy shelf place the lumped (carton+pad) reading vs the bar is far tighter than the
    # honest (pad-only) reading -- i.e. the old -93-class number was a phantom carton-vs-filtered-bar.
    T = pc.carton_pose_cart((0.18, 0.0), bodies['shelf'][1][2], YAW, carton_h=CARTON[2])
    R, t = __import__('numpy').asarray(T)[:3, :3], __import__('numpy').asarray(T)[:3, 3]
    lo, hi = bodies[bar]
    carton_gap = float(pc._box_aabb_sdf(geom.carton_surface_points() @ R.T + t, lo, hi).min())
    pad_gap = float(pc._box_aabb_sdf(geom.pad_surface_points() @ R.T + t, lo, hi).min())
    assert pad_gap > carton_gap                                 # the honest (pad) gap is looser than the carton box
    assert pad_gap > MARGIN                                     # and the honest collider clears the margin


def test_shelf_footprint_support_range_and_chosen_xy(bodies):
    """The shelf slab supports the 0.45 x 0.30 carton (yaw -pi/2 -> cart-x extent 0.30, cart-y 0.45)
    with >= 30 mm margin for load_x in [-0.20, 0.20], load_y in [-0.065, 0.065]. The chosen xy
    [0.135, 0] is inside that range with the LARGEST combined (y-limited 95 mm) footprint margin."""
    sl, sh = bodies['shelf']
    hx, hy = CARTON[1] / 2.0, CARTON[0] / 2.0    # cart-x half 0.15, cart-y half 0.225
    xmin, xmax = sl[0] + MARGIN + hx, sh[0] - MARGIN - hx
    ymin, ymax = sl[1] + MARGIN + hy, sh[1] - MARGIN - hy
    assert xmin == pytest.approx(-0.20, abs=1e-6) and xmax == pytest.approx(0.20, abs=1e-6)
    assert ymin == pytest.approx(-0.065, abs=1e-6) and ymax == pytest.approx(0.065, abs=1e-6)
    lx, ly = 0.135, 0.0
    assert xmin <= lx <= xmax and ymin <= ly <= ymax
    # every edge margin >= 30 mm, and the binding margin is the y-edge (95 mm), the max achievable.
    margins = [sh[0] - (lx + hx), (lx - hx) - sl[0], sh[1] - (ly + hy), (ly - hy) - sl[1]]
    assert min(margins) >= MARGIN
    assert min(margins) == pytest.approx(0.095, abs=1e-6)


def test_cw44_overhang_and_cw45_height(bodies):
    """cw44 (xy 0.33) OVERHANGS the shelf: its front face sits 100 mm past the shelf +x edge
    (unsupported -> tips off). cw45 (xy 0.18) is supported in-plane (50 mm front margin) but fell on
    HEIGHT: it was released at the legacy constant, which lands the carton BELOW the shelf top face
    rather than resting on it. Numbers only."""
    sl, sh = bodies['shelf']
    hx = CARTON[1] / 2.0
    assert (0.33 + hx) - sh[0] == pytest.approx(0.10, abs=1e-6)  # cw44: 100 mm overhang past the +x edge
    assert sh[0] - (0.18 + hx) == pytest.approx(0.05, abs=1e-6)  # cw45: 50 mm front margin -> in-plane OK
    assert sh[2] == pytest.approx(0.9325, abs=1e-3)              # shelf TOP (cart body frame)
    assert sl[2] < sh[2]                                         # a release below the top face is in/under the slab


# --------------------------------------------------------------- 5. preflight

def _rt(cp):
    # The preflight reads the NESTED block the factory reads (config['manipulation']['cart_placement']).
    return {'manipulation': {'cart_placement': cp}}


def test_preflight_support_selection_valid_and_invalid():
    assert cpf.check_support_selection(_rt({'support_prim': 'shelf'}), stock_supports=STOCK) == []
    assert cpf.check_support_selection(_rt({'support_prim': 1}), stock_supports=STOCK) == []
    assert cpf.check_support_selection({}, stock_supports=STOCK) == []            # absent -> no-op
    bad = cpf.check_support_selection(_rt({'support_prim': 'nope'}), stock_supports=STOCK)
    assert bad and 'support_prim' in bad[0]


def test_preflight_support_selection_reads_nested_not_top_level():
    # sim-preflight-nested-cart-placement-v1: a VALID top-level selector must NOT mask an INVALID nested
    # one -- the factory reads the nested block, so the gate must too (coldwin47 class).
    cfg = {'cart_placement': {'support_prim': 'shelf'},            # top-level, valid -- IGNORED
           'manipulation': {'cart_placement': {'support_prim': 'nope'}}}  # nested, invalid -- READ
    bad = cpf.check_support_selection(cfg, stock_supports=STOCK)
    assert bad and 'support_prim' in bad[0]


def test_preflight_support_height_shelf_top_consistent():
    """verify_place / descent on the SHELF top: the runtime + scenario place heights must agree with the
    resolved shelf top (0.9712 world). A stale value is refused; the shelf top passes."""
    runtime = _rt({'support_z_world': 0.9712})
    scenario = {'initial_staging': {'cart_support_z_world': 0.9712}}
    assert cpf.check_support_height(runtime, scenario, resolved_top_z=0.9712) == []
    stale = cpf.check_support_height(_rt({'support_z_world': 0.9325}), {},
                                     resolved_top_z=0.9712)
    assert len(stale) == 1 and '0.9325' in stale[0]
