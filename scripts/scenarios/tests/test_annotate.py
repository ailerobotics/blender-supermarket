"""Annotation helpers: COCO assembly, RLE round-trip, index-pass refinement."""
import numpy as np

import annotate
import scenario_defs as sd


def _decode_rle(rle):
    """Inverse of annotate.mask_to_rle (column-major, starts with zeros)."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        flat[pos:pos + c] = val
        pos += c
        val ^= 1
    return flat.reshape(w, h).T          # un-fortran


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    mask = rng.random((17, 23)) > 0.5
    rle = annotate.mask_to_rle(mask)
    back = _decode_rle(rle).astype(bool)
    assert np.array_equal(mask, back)


def test_refine_with_index_tightens_and_filters():
    # synthetic index map: object 5 occupies a 4x4 block
    idx = np.zeros((20, 20), dtype=np.int64)
    idx[5:9, 6:10] = 5
    img = dict(camera="c", frame=0, width=20, height=20, annotations=[
        dict(id="a", category="water", kind="spill", pass_index=5,
             bbox=[0, 0, 20, 20], bbox_amodal=[0, 0, 20, 20]),
        dict(id="b", category="cola_can", kind="product", pass_index=9,  # not present
             bbox=[1, 1, 3, 3], bbox_amodal=[1, 1, 3, 3]),
    ])
    out = annotate.refine_with_index(img, idx, min_px=2)
    kinds = {a["category"]: a for a in out["annotations"]}
    assert "water" in kinds                       # hazard kept
    assert "cola_can" not in kinds                # occluded product dropped
    w = kinds["water"]
    assert w["bbox"] == [6, 5, 4, 4]              # tight box from the mask
    assert w["area_px"] == 16 and w["segmentation"] is not None


def test_build_coco(plan):
    scn = sd.build_misplaced(plan)
    cam = scn.cameras[0]
    ann = annotate.image_annotation(scn, cam, 0)
    img = dict(file_name="frames/c/f0000.png", width=cam.width, height=cam.height,
               frame=0, camera=cam.name, annotations=ann["annotations"])
    coco = annotate.build_coco([img], annotate.categories_in(scn))
    assert coco["images"] and coco["annotations"] and coco["categories"]
    cat_ids = {c["id"] for c in coco["categories"]}
    assert all(a["category_id"] in cat_ids for a in coco["annotations"])
