"""Offline robot-URDF validator — exercised against synthetic good/bad URDFs.

Pure logic: needs only numpy (no pxr), so it runs in the same suite as the other twin
tests. Mirrors test_validate_usd.py's shape.

    .venv-usd/bin/python -m pytest scripts/twin/tests/test_validate_robot_urdf.py
"""
import os
import sys

import numpy as np  # noqa: F401  (asserts the dep the validator needs is present)
import pytest

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _SCRIPTS)
import validate_robot_urdf as V  # noqa: E402


def _fails(rep):
    return [name for status, name, _ in rep.checks if status == V.Report.FAIL]


def test_selftest_passes():
    assert V.run_selftest() == 0


def test_good_synthetic_passes(tmp_path):
    d = str(tmp_path / "good"); os.makedirs(d)
    urdf = V.build_synthetic_urdf(d, good=True)
    rep = V.validate(urdf, package_roots=[d], store_facts_path=None, check_wrapper=False)
    assert not rep.failed, rep.render()


def test_bad_synthetic_fails(tmp_path):
    d = str(tmp_path / "bad"); os.makedirs(d)
    urdf = V.build_synthetic_urdf(d, good=False)
    rep = V.validate(urdf, package_roots=[d], store_facts_path=None, check_wrapper=False)
    assert rep.failed
    f = _fails(rep)
    assert "every mesh reference resolves" in f      # dangling mesh
    assert "every dynamic link has positive mass" in f  # zero mass


PKG_URDF = """<?xml version="1.0"?>
<robot name="pkgbot">
  <link name="base"/>
  <link name="arm">
    <inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1"/></inertial>
    <visual><geometry>
      <mesh filename="package://mypkg/meshes/a.stl"/></geometry></visual>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="j" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit effort="10" velocity="1" lower="-1" upper="1"/>
  </joint>
</robot>
"""


def test_package_mesh_resolves_only_from_the_right_root(tmp_path, monkeypatch):
    """The load-bearing defect: a package:// mesh present in one repo, absent in another
    same-named one. Pointing at the wrong root FAILS with a diagnosis; the right one PASSES."""
    from twin import urdf_probe as P
    # Make discovery scan the temp dirs so the diagnosis can find where the mesh really is.
    monkeypatch.setattr(P, "DEFAULT_DISCOVERY_GLOBS", (str(tmp_path / "*"),))
    right = tmp_path / "right_pkg"
    (right / "meshes").mkdir(parents=True)
    (right / "package.xml").write_text("<package><name>mypkg</name></package>")
    (right / "meshes" / "a.stl").write_text("solid s\nendsolid s\n")
    wrong = tmp_path / "wrong_pkg"
    (wrong / "meshes").mkdir(parents=True)
    (wrong / "package.xml").write_text("<package><name>mypkg</name></package>")  # same name, no mesh

    urdf = tmp_path / "pkgbot.urdf"
    urdf.write_text(PKG_URDF)

    # Wrong root -> dangling mesh -> FAIL, and the diagnosis names the right dir.
    rep_bad = V.validate(str(urdf), package_roots=[str(wrong)], store_facts_path=None, check_wrapper=False)
    assert rep_bad.failed
    assert "every mesh reference resolves" in _fails(rep_bad)
    detail = next(d for s, n, d in rep_bad.checks if n == "every mesh reference resolves")
    assert os.path.realpath(str(right)) in detail  # points at where the mesh actually is

    # Right root -> resolves -> that check passes.
    rep_ok = V.validate(str(urdf), package_roots=[str(right)], store_facts_path=None, check_wrapper=False)
    assert "every mesh reference resolves" not in _fails(rep_ok)


def test_absolute_mesh_path_flagged(tmp_path):
    urdf = tmp_path / "abs.urdf"
    urdf.write_text(PKG_URDF.replace("package://mypkg/meshes/a.stl", "/build/machine/a.stl"))
    rep = V.validate(str(urdf), package_roots=[str(tmp_path)], store_facts_path=None, check_wrapper=False)
    assert "no absolute build-machine mesh paths" in _fails(rep)


def test_invalid_inertia_fails(tmp_path):
    # izz huge -> violates the triangle inequality (ixx+iyy < izz)
    urdf = tmp_path / "badinertia.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="bi">
  <link name="base"/>
  <link name="arm">
    <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="9.0"/></inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="j" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit effort="10" velocity="1" lower="-1" upper="1"/></joint>
</robot>
""")
    rep = V.validate(str(urdf), package_roots=[str(tmp_path)], store_facts_path=None, check_wrapper=False)
    assert "every dynamic link has a valid inertia tensor" in _fails(rep)


def test_joint_without_limits_fails(tmp_path):
    urdf = tmp_path / "nolimit.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="nl">
  <link name="base"/>
  <link name="arm">
    <inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1"/></inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="j" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/></joint>
</robot>
""")
    rep = V.validate(str(urdf), package_roots=[str(tmp_path)], store_facts_path=None, check_wrapper=False)
    assert "every movable joint has valid limits" in _fails(rep)


def test_dangling_joint_link_fails(tmp_path):
    urdf = tmp_path / "dangling.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="dl">
  <link name="base"/>
  <joint name="j" type="fixed"><parent link="base"/><child link="ghost"/></joint>
</robot>
""")
    rep = V.validate(str(urdf), package_roots=[str(tmp_path)], store_facts_path=None, check_wrapper=False)
    assert "every joint references existing links" in _fails(rep)


def test_pedestal_offset_mismatch_fails(tmp_path):
    # A pedestal_link mounted at the WRONG XY offset vs the composer's constant -> FAIL.
    urdf = tmp_path / "pedestal.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="ped">
  <link name="world"/>
  <link name="pedestal_link">
    <inertial><mass value="10"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
    <collision><geometry><box size="0.2 0.4 0.7"/></geometry></collision>
  </link>
  <joint name="pj" type="fixed"><parent link="world"/><child link="pedestal_link"/>
    <origin xyz="0 -0.3 0.35"/></joint>
</robot>
""")
    rep = V.validate(str(urdf), package_roots=[str(tmp_path)], store_facts_path=None, check_wrapper=False)
    assert "pedestal-mount offset matches composer" in _fails(rep)


def test_real_urdf_if_present():
    """When the real robot repo is on this box, the committed default config validates
    fully (this is the offline proof the captain relies on)."""
    if not os.path.isfile(V.DEFAULT_URDF):
        pytest.skip("real shelfbot URDF not present on this box")
    rep = V.validate(V.DEFAULT_URDF)  # default roots + wrapper agreement
    assert not rep.failed, rep.render()
