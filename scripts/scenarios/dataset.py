"""Dataset layout: writers and an offline validator (pure Python).

Layout (one directory per scenario) is documented in ``docs/scenario-format.md``;
this module is the executable spec. ``validate_dataset`` is what the offline tests
run against the committed smoke fixture — it needs no Blender, only Pillow to probe
image headers.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

FORMAT_VERSION = "1.0"


def paths(root: str, scenario: str) -> dict:
    base = os.path.join(root, scenario)
    return dict(
        base=base,
        manifest=os.path.join(base, "manifest.json"),
        coco=os.path.join(base, "coco.json"),
        ground_truth=os.path.join(base, "ground_truth.json"),
        cameras=os.path.join(base, "cameras"),
        frames=os.path.join(base, "frames"),
        depth=os.path.join(base, "depth"),
        annotations=os.path.join(base, "annotations"),
    )


def rel_frame(camera: str, frame: int) -> str:
    return os.path.join("frames", camera, f"f{frame:04d}.png")


def rel_depth(camera: str, frame: int) -> str:
    return os.path.join("depth", camera, f"f{frame:04d}.png")


def rel_annotation(camera: str, frame: int) -> str:
    return os.path.join("annotations", camera, f"f{frame:04d}.json")


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def write_camera(root, scenario, cam_json) -> None:
    p = paths(root, scenario)
    _write_json(os.path.join(p["cameras"], cam_json["name"] + ".json"), cam_json)


def write_annotation(root, scenario, camera, frame, ann) -> None:
    p = paths(root, scenario)
    _write_json(os.path.join(p["base"], rel_annotation(camera, frame)), ann)


def write_coco(root, scenario, coco) -> None:
    _write_json(paths(root, scenario)["coco"], coco)


def write_ground_truth(root, scenario, gt) -> None:
    _write_json(paths(root, scenario)["ground_truth"], gt)


def write_manifest(root, scenario, manifest) -> None:
    manifest = dict(manifest, format_version=FORMAT_VERSION)
    _write_json(paths(root, scenario)["manifest"], manifest)


# ------------------------------------------------------------------- validation
def _png_info(path: str) -> Optional[Tuple[int, int, str]]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size[0], im.size[1], im.mode
    except Exception:
        return None


def validate_dataset(root: str, scenario: str) -> Tuple[bool, List[str]]:
    """Check a scenario dataset is internally consistent. Offline, Blender-free."""
    errs: List[str] = []
    p = paths(root, scenario)
    if not os.path.isfile(p["manifest"]):
        return False, [f"missing manifest: {p['manifest']}"]
    man = json.load(open(p["manifest"]))
    if man.get("format_version") != FORMAT_VERSION:
        errs.append(f"format_version {man.get('format_version')} != {FORMAT_VERSION}")

    # cameras
    for cname in man.get("cameras", []):
        cj = os.path.join(p["cameras"], cname + ".json")
        if not os.path.isfile(cj):
            errs.append(f"missing camera json: {cname}")
            continue
        cam = json.load(open(cj))
        K = cam.get("intrinsics", {})
        for key in ("fx", "fy", "cx", "cy", "width", "height"):
            if key not in K:
                errs.append(f"camera {cname} intrinsics missing {key}")
        ex = cam.get("extrinsics", {}).get("camera_to_world")
        if not (isinstance(ex, list) and len(ex) == 4):
            errs.append(f"camera {cname} missing 4x4 camera_to_world")

    # per-image: frame PNG, annotation JSON, boxes in bounds, depth if declared
    n_imgs = 0
    for img in man.get("images", []):
        fp = os.path.join(p["base"], img["file"])
        info = _png_info(fp)
        if info is None:
            errs.append(f"missing/unreadable frame: {img['file']}")
            continue
        n_imgs += 1
        W, H, _ = info
        if (W, H) != (img["width"], img["height"]):
            errs.append(f"{img['file']} size {W}x{H} != manifest {img['width']}x{img['height']}")
        aj = os.path.join(p["base"], img["annotation"])
        if not os.path.isfile(aj):
            errs.append(f"missing annotation: {img['annotation']}")
            continue
        ann = json.load(open(aj))
        for a in ann.get("annotations", []):
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0 or x < -0.5 or y < -0.5 or x + bw > W + 0.5 or y + bh > H + 0.5:
                errs.append(f"{img['annotation']} box out of bounds: {a['bbox']}")
        if img.get("depth"):
            dinfo = _png_info(os.path.join(p["base"], img["depth"]))
            if dinfo is None:
                errs.append(f"missing depth: {img['depth']}")
            elif dinfo[2] not in ("I;16", "I", "L"):
                errs.append(f"depth {img['depth']} mode {dinfo[2]} is not 16-bit")

    if n_imgs == 0:
        errs.append("dataset has no readable frames")

    # COCO parses and references valid categories
    if os.path.isfile(p["coco"]):
        coco = json.load(open(p["coco"]))
        cat_ids = {c["id"] for c in coco.get("categories", [])}
        for a in coco.get("annotations", []):
            if a["category_id"] not in cat_ids:
                errs.append(f"coco annotation {a['id']} unknown category {a['category_id']}")
                break
    else:
        errs.append("missing coco.json")

    return (len(errs) == 0), errs
