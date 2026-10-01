"""Per-image ground truth + COCO export, derived from a :class:`Scenario`.

Two levels of truth, both exact:
  * **projection** (no render needed): every present entity's world AABB is projected
    into the camera and clipped to the image -> an amodal 2D box. This is what the
    offline tests check, and the fallback when no mask pass is available.
  * **mask refinement** (needs the rendered IndexOB pass): the per-object pixel set
    gives an occlusion-correct tight box, pixel area, visibility fraction and a COCO
    RLE segmentation. ``refine_with_index`` adds these in the Blender process.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import camera as cam_mod
from state import Scenario, Entity, HAZARD_CLASSES, DISTRACTOR_CLASSES


def _entity_category(e: Entity) -> dict:
    hazard = e.cls in HAZARD_CLASSES
    distractor = e.cls in DISTRACTOR_CLASSES
    return dict(name=e.cls, kind=e.kind, is_hazard=hazard, is_distractor=distractor)


def image_annotation(scn: Scenario, cam: cam_mod.CameraPose, frame: int) -> dict:
    """Projection-only annotation for one (camera, frame). Always available offline."""
    c2w, K = cam.c2w(), cam.K()
    W, H = cam.width, cam.height
    anns: List[dict] = []
    for e, center, rot in scn.state_at(frame):
        if not e.annotate:
            continue
        corners = e.corners_at(frame)
        raw = cam_mod.project_box(c2w, K, corners)
        if raw is None:
            continue
        clipped = cam_mod.clip_box(raw, W, H)
        if clipped is None:
            continue
        x0, y0, x1, y1 = clipped
        pr = cam_mod.project(c2w, K, center)
        anns.append(dict(
            id=f"{cam.name}:{frame}:{e.id}",
            entity_id=e.id, category=e.cls, kind=e.kind,
            is_hazard=e.cls in HAZARD_CLASSES,
            is_distractor=e.cls in DISTRACTOR_CLASSES,
            bbox=[round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
            bbox_amodal=[round(raw[0], 2), round(raw[1], 2),
                         round(raw[2] - raw[0], 2), round(raw[3] - raw[1], 2)],
            center_px=[round(pr[0], 2), round(pr[1], 2)] if pr else None,
            depth_m=round(pr[2], 4) if pr else None,
            world_center=[round(v, 4) for v in center],
            world_size=[round(v, 4) for v in e.size],
            pass_index=e.pass_index,
            segmentation=None, area_px=None, visibility=None,
            meta=e.meta))
    return dict(camera=cam.name, frame=frame, time_s=round(frame / scn.fps, 4),
                width=W, height=H, annotations=anns)


def mask_to_rle(mask) -> dict:
    """COCO *uncompressed* RLE (column-major counts). ``mask`` is a 2D bool array."""
    import numpy as np
    h, w = mask.shape
    flat = np.asfortranarray(mask).ravel(order="F").astype(np.uint8)
    counts: List[int] = []
    # RLE always starts with a run of zeros (COCO convention).
    prev = 0
    run = 0
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = v
    counts.append(run)
    return dict(size=[h, w], counts=counts)


def refine_with_index(img_ann: dict, index_map, min_px: int = 8) -> dict:
    """Refine projection boxes with the rendered IndexOB pass (needs numpy).

    Replaces amodal boxes with occlusion-correct tight boxes, adds pixel area,
    visibility fraction and RLE segmentation, and drops entities with too few
    visible pixels (fully occluded / off screen).
    """
    import numpy as np
    idx = np.rint(np.asarray(index_map)).astype(np.int64)
    kept: List[dict] = []
    for a in img_ann["annotations"]:
        pi = a.get("pass_index")
        if pi is None or pi <= 0:
            kept.append(a)
            continue
        mask = idx == pi
        n = int(mask.sum())
        if n < min_px:
            a = dict(a, visibility=0.0, area_px=0)
            # keep hazards/distractors even if tiny so onset timing is exact;
            # drop occluded products to avoid phantom positives.
            if a["kind"] == "product":
                continue
            kept.append(a)
            continue
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        amodal = a.get("bbox_amodal") or [x0, y0, bw, bh]
        amodal_area = max(1.0, amodal[2] * amodal[3])
        a = dict(a,
                 bbox=[x0, y0, bw, bh],
                 area_px=n,
                 visibility=round(min(1.0, n / amodal_area), 4),
                 segmentation=mask_to_rle(mask))
        kept.append(a)
    return dict(img_ann, annotations=kept)


# --------------------------------------------------------------------------- COCO

def build_coco(scenarios_images: Sequence[dict], categories: Sequence[str],
               info: Optional[dict] = None) -> dict:
    """Assemble a COCO detection dict from a list of per-image annotation dicts.

    Each input dict must carry ``file_name``, ``width``, ``height`` plus the
    ``annotations`` list produced above (bbox in pixels, optional segmentation).
    """
    cat_id = {c: i + 1 for i, c in enumerate(sorted(categories))}
    images, annotations = [], []
    ann_id = 1
    for img_i, img in enumerate(scenarios_images, start=1):
        images.append(dict(id=img_i, file_name=img["file_name"],
                           width=img["width"], height=img["height"],
                           frame=img.get("frame"), camera=img.get("camera")))
        for a in img["annotations"]:
            if a["category"] not in cat_id:
                continue
            bbox = a["bbox"]
            rec = dict(id=ann_id, image_id=img_i, category_id=cat_id[a["category"]],
                       bbox=[float(b) for b in bbox],
                       area=float(a.get("area_px") or bbox[2] * bbox[3]),
                       iscrowd=0, visibility=a.get("visibility"),
                       is_hazard=a.get("is_hazard"), is_distractor=a.get("is_distractor"))
            if a.get("segmentation"):
                rec["segmentation"] = a["segmentation"]
            annotations.append(rec)
            ann_id += 1
    return dict(
        info=info or dict(description="blender-supermarket synthetic scenario"),
        images=images, annotations=annotations,
        categories=[dict(id=i, name=c,
                         is_hazard=c in HAZARD_CLASSES,
                         is_distractor=c in DISTRACTOR_CLASSES)
                    for c, i in cat_id.items()])


def categories_in(scn: Scenario) -> List[str]:
    return sorted({e.cls for e in scn.entities if e.annotate})
