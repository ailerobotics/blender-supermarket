"""CPU tests for the collect place-on-cart between-posts geometry + reach check
(sim-place-target-between-posts-v1).

The clearance sweep needs ``usd-core`` (``author_cart`` cart colliders); the reach check additionally
needs the dataset mobile URDF (``level_place_ik`` FK), so it is skipped where absent. The measured-run
evidence + the verdict logic are pure.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_target_between_posts.py
"""
import math
from pathlib import Path

import pytest

pxr = pytest.importorskip('pxr')  # noqa: F841 -- author_cart needs usd-core

from twin.restock_manipulation import place_target_between_posts as pt  # noqa: E402
from twin.restock_manipulation import place_clearance as pc  # noqa: E402

_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'


@pytest.fixture(scope='module')
def bodies():
    return pc.rollwagen_collider_boxes()


@pytest.fixture(scope='module')
def geom():
    return pt._geom()


# --------------------------------------------------------------- measured-run evidence (pure)

def test_all_three_runs_parked_at_the_same_pose():
    """The premise correction: the three runs PARKED identically -- the divergence is not a park
    difference. Every park pose is within 1 cm of the others (and of (-4.61, 14.77))."""
    parks = [e.park_base_world for e in pt.RUN_EVIDENCE]
    for a in parks:
        for b in parks:
            assert math.hypot(a[0] - b[0], a[1] - b[1]) < 0.01
        assert math.hypot(a[0] - (-4.61), a[1] - 14.77) < 0.02


def test_base_shoves_during_place_in_the_struck_runs():
    """The named cause: the struck runs' base moved ~0.33 m from the common park pose during the place
    phase, while the cleared run barely moved."""
    d = pt.measured_drift()
    assert d['base_shove_m'] >= 0.30            # cw68/cw69 shoved ~0.33-0.39 m west
    moved = {e.name: math.hypot(e.base_world[0] - e.park_base_world[0],
                                e.base_world[1] - e.park_base_world[1]) for e in pt.RUN_EVIDENCE}
    assert moved['coldwin64'] < 0.05            # cleared run: base stayed put
    assert moved['coldwin68'] >= 0.30 and moved['coldwin69'] >= 0.30


def test_measured_execution_drift_is_cart_y_dominant():
    """The drift from the commanded target is ~0.5 m in cart-y (toward post_2) -- the quantity no shelf
    spot can absorb. The cleared run landed near-nominal."""
    d = pt.measured_drift(pt.OLD_TARGET_XY)
    assert abs(d['per_run']['coldwin64'][1]) < 0.05                 # cleared ~ nominal
    assert d['per_run']['coldwin68'][1] <= -0.45                    # struck: -y toward post_2
    assert d['per_run']['coldwin69'][1] <= -0.45
    assert pt.park_variance_m() >= 0.45                             # the spread to absorb


# --------------------------------------------------------------- static clearance (needs usd)

def test_old_target_is_not_statically_marginal(bodies, geom):
    """At its NOMINAL the old (0.135, 0) spot clears post_2 comfortably (~0.185 m) -- so the strike was
    NOT a marginal static spot; it was the drift. Guards against re-blaming the geometry."""
    ic = pt.idealized_clearance(pt.OLD_TARGET_XY, bodies=bodies, geom=geom)
    assert ic['min_nonsupport'][1] >= pt.DEFAULT_MARGIN_M
    assert ic['posts']['post_2'] > 0.15


def test_shipped_centre_improves_post_clearance(bodies, geom):
    """The shipped (0, 0) centre gives MORE static post clearance than the old spot and stays clear of
    every non-support body above the 0.04 m floor."""
    old = pt.idealized_clearance(pt.OLD_TARGET_XY, bodies=bodies, geom=geom)
    new = pt.idealized_clearance(pt.SHIPPED_TARGET_XY, bodies=bodies, geom=geom)
    assert new['posts']['post_2'] > old['posts']['post_2']         # 0.258 > 0.185
    assert new['posts']['post_2'] >= 0.25
    assert new['min_nonsupport'][1] >= pt.DEFAULT_MARGIN_M
    # symmetric: post_2 == post_3 at the centre.
    assert abs(new['posts']['post_2'] - new['posts']['post_3']) < 1e-6


def test_choose_spot_returns_the_centred_nominal(bodies, geom):
    """The scan selects the symmetric shelf centre (carton visually centred in the cage), not a grid
    edge, because the brief's min non-support clearance is side-bar-limited and flat across the range."""
    best, table = pt.choose_spot(bodies=bodies, geom=geom)   # no robot => clearance-only
    assert best == pt.SHIPPED_TARGET_XY
    assert all(r['min_nonsupport'] >= pt.DEFAULT_MARGIN_M for r in table if r['xy'][0] <= 0.125)


def test_old_spot_goes_marginal_against_post_2_under_the_measured_drift(bodies, geom):
    """Applying the struck runs' MEASURED drift vectors collapses the old spot's clearance toward a
    post_2 strike (CPU geometry reads the measured flush contact with an ~84 mm calibration offset, so
    ~0.09-0.10 m here == the real ~0 gap). This is the test the brief asked for: the old (0.135, 0) is
    flagged marginal against post_2."""
    worst_ns, worst_post, per_run = pt.worst_executed_clearance(pt.OLD_TARGET_XY, bodies=bodies, geom=geom)
    assert worst_ns < 0.12                                     # collapsed from ~0.184 static
    # the tightest run is the struck one and the binding body is post_2.
    struck = min(per_run.items(), key=lambda kv: kv[1]['min_post'])
    assert struck[0] in ('coldwin68', 'coldwin69')
    assert struck[1]['min_post_body'] == 'post_2'


def test_no_shelf_spot_absorbs_the_measured_drift(bodies, geom):
    """scope item 3: no shelf spot (incl. the shipped centre) can absorb a ~0.5 m cart-y drift -- the
    required margin (0.04 + park variance) exceeds the max achievable post clearance on the shelf. The
    verdict FAILs 'absorbs_drift' and names the base shove; the nominal is necessary, not sufficient."""
    v_old = pt.evaluate(pt.OLD_TARGET_XY, bodies=bodies, geom=geom, robot=None)
    v_new = pt.evaluate(pt.SHIPPED_TARGET_XY, bodies=bodies, geom=geom, robot=None)
    assert v_old['required_m'] > 0.5
    assert not v_old['absorbs_drift'] and not v_new['absorbs_drift']
    assert 'base' in v_new['reason'].lower() and 'post_2' in v_new['reason']
    # every candidate on the scan grid fails to absorb the drift.
    _, table = pt.choose_spot(bodies=bodies, geom=geom)
    required = pt.DEFAULT_MARGIN_M + pt.park_variance_m()
    assert all(r['min_post'] < required for r in table)


# --------------------------------------------------------------- reach (needs usd + URDF)

@pytest.mark.skipif(not _URDF.exists(), reason='dataset mobile URDF not present')
def test_shipped_centre_is_reachable_and_level_from_the_park_pose(bodies, geom):
    """The shipped (0, 0) standoff is reachable AND level (<= the never-loosened 2.0 deg bound) from the
    real park pose via the existing level_place_ik path, seeded from the fixed preplace shape."""
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(str(_URDF))
    park = pt.RUN_EVIDENCE[0]
    res = pt.reach_check(pt.SHIPPED_TARGET_XY, park.park_base_world, park.cart_world, robot)
    assert res.ok, res.reason
    assert res.pos_err_m <= 1e-3
    assert res.tilt_deg <= 2.0


@pytest.mark.skipif(not _URDF.exists(), reason='dataset mobile URDF not present')
def test_old_target_also_reachable_so_reach_is_not_the_cause(bodies, geom):
    """The old target is reachable+level too -- confirming the strike was the drift, not a reach loss."""
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(str(_URDF))
    park = pt.RUN_EVIDENCE[0]
    res = pt.reach_check(pt.OLD_TARGET_XY, park.park_base_world, park.cart_world, robot)
    assert res.ok, res.reason
