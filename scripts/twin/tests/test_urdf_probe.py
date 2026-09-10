"""URDF probe — pure parsing / FK / package:// resolution. Needs only numpy (no pxr).

    .venv-usd/bin/python -m pytest scripts/twin/tests/test_urdf_probe.py
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from twin import urdf_probe as P  # noqa: E402


TWO_LINK = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base"/>
  <link name="arm">
    <inertial><mass value="2.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit effort="10" velocity="1" lower="-1.57" upper="1.57"/>
  </joint>
  <ros2_control name="c" type="system">
    <joint name="j1"><command_interface name="position">
      <param name="initial_value">0.5</param></command_interface></joint>
  </ros2_control>
</robot>
"""


def _write(tmp_path, text, name="robot.urdf"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_parse_structure(tmp_path):
    r = P.parse_urdf(_write(tmp_path, TWO_LINK))
    assert r.name == "two_link"
    assert set(r.links) == {"base", "arm"}
    assert len(r.joints) == 1 and r.joints[0].jtype == "revolute"
    assert P.find_roots(r) == ["base"]
    assert r.initial_positions == {"j1": 0.5}
    assert r.links["arm"].inertial.mass == 2.0
    assert P.child_of_movable_joint(r) == {"arm": r.joints[0]}


def test_rpy_to_matrix_yaw_90():
    R = P.rpy_to_matrix((0.0, 0.0, math.pi / 2))
    # yaw 90° maps +X -> +Y
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)


def test_fk_revolute_places_child():
    # child frame at joint origin (1,0,0); with q=0 it's just the origin
    robot = P.Robot(
        name="t",
        links={"base": P.Link("base"), "arm": P.Link("arm")},
        joints=[P.Joint("j1", "revolute", "base", "arm",
                        origin_xyz=(1.0, 0.0, 0.0), axis=(0, 0, 1),
                        lower=-1, upper=1, effort=1, velocity=1)],
    )
    W0 = P.link_world_transforms(robot, {"j1": 0.0})
    assert np.allclose(W0["arm"][:3, 3], [1, 0, 0])
    # rotate 90° about Z: the frame origin is unchanged (rotation about its own origin)
    W90 = P.link_world_transforms(robot, {"j1": math.pi / 2})
    assert np.allclose(W90["arm"][:3, 3], [1, 0, 0], atol=1e-9)
    # but its local +X axis now points +Y in the world
    assert np.allclose(W90["arm"][:3, :3] @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)


def test_fk_prismatic_translates_along_axis():
    robot = P.Robot(
        name="t",
        links={"base": P.Link("base"), "slide": P.Link("slide")},
        joints=[P.Joint("p", "prismatic", "base", "slide",
                        origin_xyz=(0, 0, 0), axis=(0, 1, 0),
                        lower=0, upper=0.3, effort=1, velocity=1)],
    )
    W = P.link_world_transforms(robot, {"p": 0.2})
    assert np.allclose(W["slide"][:3, 3], [0, 0.2, 0])


def test_box_world_points_transform():
    box = P.Geometry("box", box_size=(0.2, 0.2, 0.2))
    link = np.eye(4)
    link[:3, 3] = [1.0, 2.0, 3.0]
    pts = P.geometry_world_points(box, link)
    assert pts.shape == (8, 3)
    # centred at the link origin, half-size 0.1
    assert np.allclose(pts.mean(0), [1.0, 2.0, 3.0])
    assert np.isclose(pts[:, 0].max() - pts[:, 0].min(), 0.2)


def test_resolve_package_uri_entry_pkg(tmp_path):
    # <root>/pkg/rel  (root is a parent-of-packages dir, e.g. /opt/ros/.../share)
    pkgroot = tmp_path / "shareroot"
    (pkgroot / "mypkg" / "meshes").mkdir(parents=True)
    f = pkgroot / "mypkg" / "meshes" / "a.stl"
    f.write_text("solid s\nendsolid s\n")
    rp, cand = P.resolve_package_uri("package://mypkg/meshes/a.stl", [str(pkgroot)])
    assert rp and os.path.realpath(rp) == os.path.realpath(str(f))


def test_resolve_package_uri_root_declares_pkg(tmp_path):
    # <root>/rel  where the root itself declares the package in package.xml
    repo = tmp_path / "mypkg_repo"
    (repo / "meshes").mkdir(parents=True)
    (repo / "package.xml").write_text("<package><name>mypkg</name></package>")
    (repo / "meshes" / "a.obj").write_text("v 0 0 0\n")
    rp, _ = P.resolve_package_uri("package://mypkg/meshes/a.obj", [str(repo)])
    assert rp and rp.endswith("meshes/a.obj")


def test_resolve_package_uri_missing_returns_candidates(tmp_path):
    rp, cand = P.resolve_package_uri("package://nope/x.stl", [str(tmp_path)])
    assert rp is None and cand  # tried something, found nothing


def test_discover_packages_finds_by_name_and_dirname(tmp_path):
    a = tmp_path / "aaa"; a.mkdir()
    (a / "package.xml").write_text("<package><name>foo</name></package>")
    b = tmp_path / "foo"; b.mkdir()  # matches by directory name
    found = P.discover_packages("foo", globs=[str(tmp_path / "*")])
    assert os.path.realpath(str(a)) in found
    assert os.path.realpath(str(b)) in found


def test_mesh_references(tmp_path):
    r = P.parse_urdf(_write(tmp_path, TWO_LINK))
    # TWO_LINK uses boxes only -> no mesh refs
    assert P.mesh_references(r) == []
