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
