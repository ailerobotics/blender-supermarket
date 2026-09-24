"""Offline USD validator — exercised against synthetic good/bad stages.

Needs the USD (``pxr``) bindings, which the other twin tests deliberately do NOT
(they are pure logic). ``pxr`` is present in a usd-core venv or under env_isaaclab;
where it is absent (plain env_isaaclab), the whole module is skipped rather than
failing — same as ``usdchecker`` would be skipped if the wheel lacked it.

    .venv-usd/bin/python -m pytest scripts/twin/tests/test_validate_usd.py
"""
import os
import sys

import pytest

pytest.importorskip("pxr")  # skip cleanly where USD is unavailable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import validate_usd as V  # noqa: E402
from pxr import Usd  # noqa: E402


def _validate(path, repair=False):
    return V.validate(path, repair=repair)


def test_clean_stage_passes(tmp_path):
    store = V.build_synthetic_stage(str(tmp_path / "good"), good=True)
    rep = _validate(store)
    assert not rep.failed, rep.render()
    # sanity: the characterisation actually ran
    assert rep.facts["up_axis"] == "Z"
    assert rep.facts["meters_per_unit"] == 1.0
    assert rep.facts["default_prim"] == "/Supermarket"


def test_dangling_texture_fails(tmp_path):
    store = V.build_synthetic_stage(str(tmp_path / "bad"), good=False)
    rep = _validate(store)
    assert rep.failed
    # the failing check is specifically the asset-resolution one
    fails = [name for status, name, _ in rep.checks if status == V.Report.FAIL]
    assert "every asset path resolves" in fails


def test_missing_default_prim_fails(tmp_path):
    from pxr import UsdGeom
    store = V.build_synthetic_stage(str(tmp_path / "nodp"), good=True)
    stage = Usd.Stage.Open(store)
    stage.ClearDefaultPrim()
    stage.GetRootLayer().Save()
    rep = _validate(store)
    assert rep.failed
    assert any(name == "default prim" and status == V.Report.FAIL
               for status, name, _ in rep.checks)


def test_wrong_up_axis_fails(tmp_path):
    from pxr import UsdGeom
    store = V.build_synthetic_stage(str(tmp_path / "yup"), good=True)
    stage = Usd.Stage.Open(store)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.GetRootLayer().Save()
    rep = _validate(store)
    assert rep.failed
    assert any(name == "up-axis" and status == V.Report.FAIL
               for status, name, _ in rep.checks)


def test_missing_composer_group_fails(tmp_path):
    store = V.build_synthetic_stage(str(tmp_path / "nogrp"), good=True)
    stage = Usd.Stage.Open(store)
    stage.RemovePrim("/Supermarket/GRP_Cameras/Cam_Aisle")
    stage.GetRootLayer().Save()
    rep = _validate(store)
    assert rep.failed
    assert any("spawn camera" in name and status == V.Report.FAIL
               for status, name, _ in rep.checks)


def test_repair_reconciles_from_assets_textures(tmp_path, monkeypatch):
    # A bad stage references ./textures/MISSING_floor.png; seed a same-named file in
    # a fake assets/textures/ and prove --repair-textures copies it and re-passes.
    store = V.build_synthetic_stage(str(tmp_path / "bad"), good=False)
    fake_assets = tmp_path / "assets_textures"
    fake_assets.mkdir()
    (fake_assets / "MISSING_floor.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(V, "ASSETS_TEXTURES", str(fake_assets))
    rep = _validate(store, repair=True)
    assert not rep.failed, rep.render()
    assert os.path.exists(os.path.join(os.path.dirname(store), "textures", "MISSING_floor.png"))


def test_dangling_texture_is_cosmetic_not_blocking(tmp_path):
    # A dangling *texture* is a real failure but must NOT block booting: the surface
    # loads untextured, the geometry is intact. The boot gate keys off blocking_failed.
    store = V.build_synthetic_stage(str(tmp_path / "bad"), good=False)
    rep = _validate(store)
    assert rep.failed              # it is a hard failure
    assert rep.cosmetic_failed     # ... but classified cosmetic
    assert not rep.blocking_failed  # ... so it does NOT block the boot gate
    # and the render calls it out as such
    assert "[COSMETIC]" in rep.render()


def test_structural_failure_is_blocking(tmp_path):
    # A missing default prim is a structural defect (the composer's AddReference has no
    # target) — it MUST block booting.
    store = V.build_synthetic_stage(str(tmp_path / "nodp"), good=True)
    stage = Usd.Stage.Open(store)
    stage.ClearDefaultPrim()
    stage.GetRootLayer().Save()
    rep = _validate(store)
    assert rep.blocking_failed
    assert not rep.cosmetic_failed


def test_repaired_texture_leaves_no_failure(tmp_path, monkeypatch):
    # After repair the cosmetic failure is gone entirely — neither blocking nor cosmetic.
    store = V.build_synthetic_stage(str(tmp_path / "bad"), good=False)
    fake_assets = tmp_path / "assets_textures"
    fake_assets.mkdir()
    (fake_assets / "MISSING_floor.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(V, "ASSETS_TEXTURES", str(fake_assets))
    rep = _validate(store, repair=True)
    assert not rep.failed
    assert not rep.blocking_failed
    assert not rep.cosmetic_failed


def test_absolute_path_flagged(tmp_path):
    from pxr import UsdShade, Sdf
    store = V.build_synthetic_stage(str(tmp_path / "abs"), good=True)
    stage = Usd.Stage.Open(store)
    shd = UsdShade.Shader.Get(stage, "/Supermarket/_materials/floor/tex")
    # point at an absolute path that does not exist -> both absolute + unresolved
    shd.GetInput("file").Set("/build/machine/textures/floor.png")
    stage.GetRootLayer().Save()
    rep = _validate(store)
    assert rep.failed
    names = [name for status, name, _ in rep.checks if status == V.Report.FAIL]
    assert "no absolute build-machine paths" in names


# --------------------------------------------------------------------------- #
# The *real* export defect — floor + walls (the 6 Poly Haven maps)             #
# --------------------------------------------------------------------------- #
# Blender's USD exporter authors each texture reference by image *datablock name*
# but copies the *backing* file into export/textures/. For the two Poly Haven
# materials the MCP tool imports from temp files, so the datablocks are named
# `floor_tiles_08_Diffuse.jpg` … while the file on disk is a `tmp*.jpg` orphan —
# the reference dangles. These are the exact 6 names the real export drops (see
# data/twin-usd-validate-v1/report.md §2). The synthetic single-texture tests above
# cover the mechanism; these tests pin the real signature and, crucially, prove the
# versioned assets/textures/ can always repair it.
REAL_DEFECT_TEXTURES = [
    "floor_tiles_08_Diffuse.jpg",
    "floor_tiles_08_Rough.jpg",
    "floor_tiles_08_nor_gl.jpg",
    "white_plaster_02_Diffuse.jpg",
    "white_plaster_02_Rough.jpg",
    "white_plaster_02_nor_gl.jpg",
]


def _build_real_defect_stage(dirpath):
    """A store stage reproducing the real export defect: the floor and walls bind
    materials whose UsdUVTexture inputs author `./textures/<datablock name>` for the 6
    Poly Haven maps, while only orphan `tmp*.jpg` files exist on disk — exactly what
    Blender emits before `_stage_textures_for_export()` (scripts/export_usd.py) fixes
    the datablock names, or `--repair-textures` reconciles the artifact."""
    from pxr import Usd, UsdGeom, UsdShade, Sdf
    os.makedirs(os.path.join(dirpath, "textures"), exist_ok=True)
    # The exporter copies the *backing* temp file, so tmp*.jpg orphans land on disk
    # while the authored (datablock-name) references point at nothing.
    for i in range(len(REAL_DEFECT_TEXTURES)):
        with open(os.path.join(dirpath, "textures", f"tmp{i}orphan.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff")  # jpeg magic — enough to exist on disk
    path = os.path.join(dirpath, "store.usda")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Supermarket")
    stage.SetDefaultPrim(root.GetPrim())
    for grp in V.COMPOSER_GROUPS:
        UsdGeom.Xform.Define(stage, f"/Supermarket/{grp}")
    for g in V.GONDOLA_FIXTURES + V.GONDOLA_PRODUCTS:
        UsdGeom.Xform.Define(stage, f"/Supermarket/{g}")
    UsdGeom.Xform.Define(stage, "/Supermarket/GRP_Cameras/Cam_Aisle")
    floor = UsdGeom.Mesh.Define(stage, "/Supermarket/GRP_Structure/Floor")
    floor.CreateExtentAttr([(-13, -11, 0), (13, 11, 0)])
    for i, texname in enumerate(REAL_DEFECT_TEXTURES):
        shd = UsdShade.Shader.Define(stage, f"/Supermarket/_materials/m{i}/tex")
        shd.CreateIdAttr("UsdUVTexture")
        shd.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{texname}")
    stage.GetRootLayer().Save()
    return path


def test_real_defect_signature_flagged_cosmetic(tmp_path):
    # The validator must catch all 6 real dangling floor/wall maps, and classify them
    # cosmetic (untextured surface, geometry intact) — never boot-blocking.
    store = _build_real_defect_stage(str(tmp_path / "defect"))
    rep = _validate(store)
    assert rep.failed
    assert rep.cosmetic_failed and not rep.blocking_failed, rep.render()
    unresolved = set(rep.facts.get("unresolved", []))
    assert set(REAL_DEFECT_TEXTURES) <= unresolved, unresolved


def test_versioned_assets_textures_covers_defect_names():
    # Regression guard for the repair safety net: --repair-textures (and preflight's
    # in-place repair) can only close the gap if the committed assets/textures/ carries
    # a file for every one of the 6 names the export authors. If any source map is
    # renamed or deleted this fails loudly instead of silently un-repairing the store.
    missing = [n for n in REAL_DEFECT_TEXTURES
               if not os.path.exists(os.path.join(V.ASSETS_TEXTURES, n))]
    assert not missing, f"assets/textures/ is missing repair sources: {missing}"


def test_repair_textures_reconciles_real_defect_from_versioned_assets(tmp_path):
    # End-to-end against the REAL versioned assets/textures/ (no monkeypatch): the 6
    # dangling maps are copied in and the stage then validates clean — proving the
    # documented remedy (validate_usd.py --repair-textures) actually clears the defect.
    store = _build_real_defect_stage(str(tmp_path / "defect"))
    rep = _validate(store, repair=True)
    assert not rep.failed, rep.render()
    texdir = os.path.join(os.path.dirname(store), "textures")
    for name in REAL_DEFECT_TEXTURES:
        assert os.path.exists(os.path.join(texdir, name)), f"repair did not place {name}"
