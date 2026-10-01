"""Validate the committed smoke fixture offline (no Blender).

These are the "tiny render's annotations" checks the task asks for: the fixture was
rendered once by ``generate.py`` and committed; CI has no Blender, so everything
here reads the committed PNG/JSON and checks internal consistency.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

import dataset as ds
from conftest import FIXTURES

SCENARIOS = ["depletion_G1R_smoke", "spills_aisle_smoke", "people_movement_smoke",
             "misplaced_G1R_b2_smoke", "multiview_G1R_b2_smoke"]


@pytest.mark.parametrize("name", SCENARIOS)
def test_dataset_validates(name):
    ok, errs = ds.validate_dataset(FIXTURES, name)
    assert ok, errs


def test_multiview_depth_is_metric():
    name = "multiview_G1R_b2_smoke"
    man = json.load(open(ds.paths(FIXTURES, name)["manifest"]))
    depth_imgs = [i for i in man["images"] if i.get("depth")]
    assert depth_imgs, "multiview should carry depth"
    dp = os.path.join(ds.paths(FIXTURES, name)["base"], depth_imgs[0]["depth"])
    arr = np.array(Image.open(dp))
    assert Image.open(dp).mode in ("I;16", "I")
    nz = arr[arr > 0]
    assert nz.size > 0
    assert 100 < int(nz.min()) and int(nz.max()) < 65000   # plausible mm range


def test_misplaced_annotations_carry_expected_vs_actual():
    name = "misplaced_G1R_b2_smoke"
    gt = json.load(open(ds.paths(FIXTURES, name)["ground_truth"]))
    assert gt["misplacements"]
    types = {m["type"] for m in gt["misplacements"]}
    assert {"wrong_slot", "chilled_on_ambient"} <= types


def test_spill_distractors_present_and_not_hazards():
    name = "spills_aisle_smoke"
    coco = json.load(open(ds.paths(FIXTURES, name)["coco"]))
    by_name = {c["name"]: c for c in coco["categories"]}
    for d in ("floor_marking", "shadow", "paper_sign"):
        if d in by_name:
            assert by_name[d]["is_distractor"] and not by_name[d]["is_hazard"]
    # at least one real hazard category exists in the export
    assert any(c["is_hazard"] for c in coco["categories"])


def test_coco_boxes_within_images():
    name = "depletion_G1R_smoke"
    coco = json.load(open(ds.paths(FIXTURES, name)["coco"]))
    sizes = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    for a in coco["annotations"]:
        w, h = sizes[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        assert bw > 0 and bh > 0
        assert -0.5 <= x and -0.5 <= y and x + bw <= w + 0.5 and y + bh <= h + 0.5


def test_camera_intrinsics_extrinsics_present():
    name = "multiview_G1R_b2_smoke"
    man = json.load(open(ds.paths(FIXTURES, name)["manifest"]))
    for cname in man["cameras"]:
        cj = json.load(open(os.path.join(ds.paths(FIXTURES, name)["cameras"], cname + ".json")))
        assert {"fx", "fy", "cx", "cy"} <= set(cj["intrinsics"])
        assert len(cj["extrinsics"]["camera_to_world"]) == 4
