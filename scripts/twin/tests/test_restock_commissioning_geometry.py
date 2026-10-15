"""Parent-aware commissioning geometry (CPU; pure, no Isaac/GPU).

The cart station camera is parented to cart_chassis, a MOVING frame; computing its commissioning
extrinsic/polygons as if world-fixed (the source-only path) silently corrupts place targets.
commissioning_geometry composes the parent world pose so the cart is correct, and reduces exactly
to the historical world-fixed formula for parent='map'. Design geometry, not sensor evidence.
"""
import numpy as np

from scripts.twin.restock_camera.config import camera_config
from scripts.twin.restock_camera.sensing import commissioning_geometry
from scripts.twin.restock_camera.geometry import (rigid, pose_matrix, transform_points, project,
                                                  Intrinsics, scale_intrinsics)


def design():
    scenario = dict(handling_unit=dict(local_wall=3, grasp_mode='side', dimensions_m=[.6,.4,.16]),
        initial_staging=dict(support_center_xy=[-5.73,17.1], unit_bottom_z=.6,
            target_surface_pose_world=[-3.05,4.826,.73,0], cart_load_center_body_xy=[.08,0],
            cart_support_z_world=.225))
    return camera_config(scenario, dict(spec=dict(arm_mount_z=.5)))


def test_map_station_reduces_to_the_world_fixed_formula():
    d = design()
    world_from_base = pose_matrix([-5.73,16.,0.], .5)
    out = commissioning_geometry(d, 'source', world_from_parent=np.eye(4), world_from_base=world_from_base)
    expected = np.linalg.inv(rigid(world_from_base)) @ rigid(d['cameras']['source']['parent_from_optical'])
    np.testing.assert_allclose(out['base_from_camera'], expected, atol=1e-12)
    # The empty support/witness quads project inside the design source view.
    k = Intrinsics(**d['cameras']['source']['intrinsics'])
    for uv in (out['support_uv'], out['witness_uv']):
        assert uv.shape == (4,2)
        assert np.all(uv >= 0) and np.all(uv < [k.width, k.height])


def test_cart_station_composes_the_moving_parent_world_pose():
    d = design()
    world_from_parent = pose_matrix([1.2,-0.4,1.7], .20)   # the cart_chassis prim's world pose
    world_from_base = pose_matrix([1.0,0.0,0.0], .5)
    out = commissioning_geometry(d, 'cart', world_from_parent=world_from_parent, world_from_base=world_from_base)
    world_from_optical = rigid(world_from_parent) @ rigid(d['cameras']['cart']['parent_from_optical'])
    expected = np.linalg.inv(rigid(world_from_base)) @ world_from_optical
    np.testing.assert_allclose(out['base_from_camera'], expected, atol=1e-12)
    # support_world is the parent-frame quad lifted into world by the cart pose (not left in parent).
    np.testing.assert_allclose(out['support_world'],
        transform_points(rigid(world_from_parent), transform_points(np.linalg.inv(rigid(world_from_parent)), out['support_world'])),
        atol=1e-9)


def test_cart_support_polygons_are_invariant_to_the_cart_world_pose():
    """The camera and the deck share the cart frame, so the camera sees the deck identically
    wherever the cart is: support_uv/witness_uv must NOT depend on world_from_parent (while the
    extrinsic does). This is why a staging-vs-transfer cart pose cannot corrupt the selected
    commissioning pixels, and the live transfer re-registration keeps the extrinsic current."""
    d = design()
    base = pose_matrix([1.0,0.0,0.0], .5)
    a = commissioning_geometry(d, 'cart', world_from_parent=pose_matrix([0.,0.,1.5], .2), world_from_base=base)
    b = commissioning_geometry(d, 'cart', world_from_parent=pose_matrix([2.3,-1.1,1.5], .2, ), world_from_base=base)
    np.testing.assert_allclose(a['support_uv'], b['support_uv'], atol=1e-9)
    np.testing.assert_allclose(a['witness_uv'], b['witness_uv'], atol=1e-9)
    assert not np.allclose(a['base_from_camera'], b['base_from_camera'])   # extrinsic tracks the pose


def test_source_matches_the_build_sources_inline_projection():
    """Regression against the historical source-only inline computation in commissioning_sources."""
    d = design()
    cam, support = d['cameras']['source'], d['supports']['source']
    k = Intrinsics(**cam['intrinsics'])
    world_from_optical = rigid(cam['parent_from_optical'])          # parent='map', /World identity
    center = np.asarray(support['nominal_center'], float)
    req = support['required_xy_m']

    def quad_world(c_xy, size, yaw, z):
        cs, sn = np.cos(yaw), np.sin(yaw)
        hx, hy = size[0]/2., size[1]/2.
        rot = np.array([[cs,-sn],[sn,cs]])
        return np.asarray([[c_xy[0]+ (rot@(lx,ly))[0], c_xy[1]+ (rot@(lx,ly))[1], z]
                           for lx,ly in ((-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy))], float)
    support_world = quad_world(center[:2], (req[0]+.04, req[1]+.04), float(support['yaw']), float(center[2]))
    wd = support['witness_design']
    witness_world = quad_world(np.asarray(wd['center'])[:2], wd['size_xy'], float(wd['yaw']), float(wd['center'][2]))
    exp_support = project(transform_points(np.linalg.inv(world_from_optical), support_world), k)[0]
    exp_witness = project(transform_points(np.linalg.inv(world_from_optical), witness_world), k)[0]

    out = commissioning_geometry(d, 'source', world_from_parent=np.eye(4), world_from_base=np.eye(4))
    np.testing.assert_allclose(out['support_uv'], exp_support, atol=1e-9)
    np.testing.assert_allclose(out['witness_uv'], exp_witness, atol=1e-9)


# --- render-scale: a `fast` commissioning projects onto the reduced grid ------------------------

def test_render_scale_default_is_byte_identical_to_the_native_projection():
    d = design()
    wfb = pose_matrix([-5.73,16.,0.], .5)
    native = commissioning_geometry(d, 'source', world_from_parent=np.eye(4), world_from_base=wfb)
    explicit = commissioning_geometry(d, 'source', world_from_parent=np.eye(4), world_from_base=wfb,
                                      render_scale=1.0)
    k = Intrinsics(**d['cameras']['source']['intrinsics'])
    assert native['intrinsics'] == k and explicit['intrinsics'] == k
    np.testing.assert_array_equal(native['support_uv'], explicit['support_uv'])
    np.testing.assert_array_equal(native['witness_uv'], explicit['witness_uv'])


def test_fast_render_scale_projects_the_polygons_onto_the_reduced_grid():
    """A 0.6 commission projects the empty support/witness quads through the SCALED intrinsics, so
    the selected pixels land on the reduced render-product grid the `fast` frame is rendered at --
    the fix for a baseline whose polygons/intrinsics were full-res while the mission rendered 0.6."""
    d = design()
    wfb = pose_matrix([-5.73,16.,0.], .5)
    k = Intrinsics(**d['cameras']['source']['intrinsics'])
    ks = scale_intrinsics(k, 0.6)
    full = commissioning_geometry(d, 'source', world_from_parent=np.eye(4), world_from_base=wfb)
    fast = commissioning_geometry(d, 'source', world_from_parent=np.eye(4), world_from_base=wfb,
                                  render_scale=0.6)
    assert fast['intrinsics'] == ks                               # the scaled calibration
    # Each quad still projects inside the (smaller) scaled image.
    for uv in (fast['support_uv'], fast['witness_uv']):
        assert np.all(uv >= 0) and np.all(uv < [ks.width, ks.height])
    # Exactly the same world polygons reprojected through the scaled intrinsics (not a coincidence).
    world_from_optical = rigid(d['cameras']['source']['parent_from_optical'])
    for key in ('support', 'witness'):
        exp = project(transform_points(np.linalg.inv(world_from_optical), fast[key + '_world']), ks)[0]
        np.testing.assert_allclose(fast[key + '_uv'], exp, atol=1e-9)
    # The world polygons themselves are scale-invariant (same geometry, only the projection changes).
    np.testing.assert_allclose(fast['support_world'], full['support_world'], atol=1e-12)


# --- the commissioning hook now resolves the cart station (no Isaac at import) -----------------

def _job():
    return {'job': {'source_bin': 'backstock_00', 'shelf_id': 'G3_back_s2'}}


def test_resolve_station_now_accepts_the_cart(monkeypatch):
    from scripts.twin.restock.commissioning_sources import _resolve_station
    monkeypatch.setenv('COMMISSIONING_STATIONS', 'rollwagen')
    assert _resolve_station(_job()) == ('cart', 'rollwagen')


def test_resolve_station_still_accepts_source_and_shelf(monkeypatch):
    from scripts.twin.restock.commissioning_sources import _resolve_station
    monkeypatch.setenv('COMMISSIONING_STATIONS', 'backstock_00')
    assert _resolve_station(_job()) == ('source', 'backstock_00')
    monkeypatch.setenv('COMMISSIONING_STATIONS', 'G3_back_s2')
    assert _resolve_station(_job()) == ('shelf', 'G3_back_s2')


def test_resolve_station_refuses_unknown_location_and_multiples(monkeypatch):
    import pytest
    from scripts.twin.restock.commissioning_sources import _resolve_station
    monkeypatch.setenv('COMMISSIONING_STATIONS', 'nowhere')
    with pytest.raises(ValueError, match='unknown commissioning location'):
        _resolve_station(_job())
    monkeypatch.setenv('COMMISSIONING_STATIONS', 'backstock_00,rollwagen')
    with pytest.raises(ValueError, match='exactly one'):
        _resolve_station(_job())
