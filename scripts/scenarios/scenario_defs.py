"""The five parameterised, seeded scenario builders.

Each ``build_*`` returns a fully-resolved :class:`Scenario`. Defaults are tuned for
a *small* set (a handful of frames/cameras) that still exercises a vision pipeline;
pass bigger numbers for a fuller run. Everything is deterministic given ``seed``.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import catalog_geom as geom
from camera import CameraPose
from state import Scenario, Entity
import common
import people as people_mod

RES = (640, 480)


# --------------------------------------------------------------------------- cams
def _shelf_cam(name: str, front_x: float, bay_y: float, standoff: float = 2.0,
               lens: float = 30.0, eye_z: float = 1.45, tgt_z: float = 1.0) -> CameraPose:
    return CameraPose(name=name, eye=(front_x + standoff, bay_y, eye_z),
                      target=(front_x, bay_y, tgt_z), lens_mm=lens,
                      width=RES[0], height=RES[1])


def _bay_y(plan, fixture_id, bay_index):
    return common.bay_centers_y(plan, fixture_id)[bay_index]


# --------------------------------------------------------------------- depletion
def build_depletion(plan: dict, seed: int = 1, n_frames: int = 16, fps: int = 6,
                    fixture_id: str = "G1", side: str = "R", bays=(2, 3),
                    n_remove: int = 10, n_restock: int = 2, n_shoppers: int = 1,
                    **_) -> Scenario:
    rng = random.Random(seed)
    pidx = common.PassIndex()
    front_x, facing = common.gondola_side_front(plan, fixture_id, side)
    bay_ys = [_bay_y(plan, fixture_id, b) for b in bays]
    mid_y = sum(bay_ys) / len(bay_ys)

    # products on the watched bays (annotated), rest of the gondola as backdrop
    watched = [s for s in geom.iter_slots(plan, fixtures=[fixture_id])
               if s.facing_dir == facing and s.bay_index in bays]
    prods: List[Entity] = []
    for slot in watched:
        for f in slot.facings:
            prods.append(Entity(
                id=f"{f.slot_id}_f{f.facing_index}", kind="product", cls=f.sku,
                pass_index=pidx.next(), center=f.center, size=f.size, rot_z=f.rot_z,
                meta=dict(slot_id=f.slot_id, facing_index=f.facing_index,
                          bay=slot.bay_index, level=slot.level, cat=slot.cat)))
    # No neighbour footprints: a close shelf-facing camera would sit inside the
    # opposite gondola's box. The shelf boards + back panel give enough backdrop.

    # depletion schedule: remove front facings one by one; restock a few.
    # A restock is a *second* unit (new pass_index) that appears later in the same
    # slot, so the single present-window per entity stays expressible and exact.
    removable = list(prods)
    rng.shuffle(removable)
    events = []
    restocks: List[Entity] = []
    chosen = removable[:n_remove]
    for k, e in enumerate(chosen):
        f_off = 1 + int((k + 1) / (n_remove + 1) * (n_frames - 2))
        e.present_to = f_off - 1
        events.append(dict(frame=f_off, type="remove", slot=e.meta["slot_id"],
                           sku=e.cls, entity=e.id))
    for e in chosen[:n_restock]:                        # a couple come back (restock)
        back = min(n_frames - 1, (e.present_to or 0) + rng.randint(3, 6))
        r = Entity(id=e.id + "_restock", kind="product", cls=e.cls,
                   pass_index=pidx.next(), center=e.center, size=e.size,
                   rot_z=e.rot_z, present_from=back, meta=dict(e.meta, restock=True))
        restocks.append(r)
        events.append(dict(frame=back, type="restock", slot=e.meta["slot_id"],
                           sku=e.cls, entity=r.id))
    prods += restocks

    # occluding shoppers crossing the shelf
    shoppers: List[Entity] = []
    for i in range(n_shoppers):
        y0, y1 = mid_y - 1.3, mid_y + 1.3      # cross the watched bays, stay in FOV
        appear = rng.randint(1, max(2, n_frames // 3))
        dur = rng.randint(4, 8)
        z = people_mod.PERSON_Z
        standoff = front_x + 0.7
        path = [(appear, (standoff, y0, z), math.pi / 2),
                (min(n_frames - 1, appear + dur), (standoff, y1, z), math.pi / 2)]
        shoppers.append(Entity(id=f"shopper_{i}", kind="person", cls="person",
                               pass_index=pidx.next(), center=path[0][1],
                               size=people_mod.PERSON_SIZE, rot_z=math.pi / 2,
                               present_from=appear, present_to=path[-1][0], path=path,
                               meta=dict(session_id="s_occluder%d" % i, role="occluder")))

    cams = [_shelf_cam(f"shelf_{fixture_id}{side}", front_x, mid_y, standoff=2.4, lens=24)]
    scn = Scenario(name=f"depletion_{fixture_id}{side}", kind="depletion", seed=seed,
                   fps=fps, n_frames=n_frames, cameras=cams,
                   entities=prods + shoppers,
                   params=dict(fixture_id=fixture_id, side=side, bays=list(bays),
                               n_remove=n_remove, n_restock=n_restock,
                               n_shoppers=n_shoppers),
                   ground_truth=dict(events=sorted(events, key=lambda e: e["frame"]),
                                     slots=[s.slot_id for s in watched],
                                     per_frame=depletion_counts(prods, n_frames)))
    return scn


def depletion_counts(prods, n_frames) -> List[dict]:
    """Per-frame per-slot per-SKU facing count — the depletion ground truth."""
    out = []
    for f in range(n_frames):
        slots: Dict[str, Dict[str, int]] = {}
        for e in prods:
            if e.kind != "product" or not e.present_at(f):
                continue
            slots.setdefault(e.meta["slot_id"], {})
            slots[e.meta["slot_id"]][e.cls] = slots[e.meta["slot_id"]].get(e.cls, 0) + 1
        out.append(dict(frame=f, slots=slots))
    return out


# ------------------------------------------------------------------------ spills
def build_spills(plan: dict, seed: int = 2, n_frames: int = 12, fps: int = 4,
                 aisle_x: Optional[float] = None, **_) -> Scenario:
    rng = random.Random(seed)
    pidx = common.PassIndex()
    store = plan["store"]
    if aisle_x is None:
        aisle_x = plan["gondolas"][1]["x"] + plan["bay_w"]   # a walkable aisle lane
    base_y = 0.0
    z = 0.006                       # thin puddle sitting on the floor

    def puddle(cls, dx, dy, r, onset):
        return Entity(id=f"{cls}_{len(entities)}", kind="spill", cls=cls,
                      pass_index=pidx.next(), center=(aisle_x + dx, base_y + dy, z / 2),
                      size=(r, r, z), present_from=onset,
                      meta=dict(onset_s=round(onset / fps, 3),
                                floor_xy=[round(aisle_x + dx, 3), round(base_y + dy, 3)],
                                radius_m=round(r / 2, 3)))

    entities: List[Entity] = []
    hazards = [
        puddle("water", -0.3, -1.5, 0.55, onset=2),
        puddle("milk", 0.2, 0.4, 0.5, onset=5),
        puddle("juice", -0.2, 1.8, 0.4, onset=7),
        puddle("oil", 0.35, -0.6, 0.35, onset=9),
    ]
    entities += hazards
    # debris + a fallen product
    entities.append(Entity(id="debris_0", kind="spill", cls="debris", pass_index=pidx.next(),
                           center=(aisle_x + 0.1, base_y - 2.4, 0.03), size=(0.3, 0.25, 0.06),
                           present_from=4, meta=dict(onset_s=round(4 / fps, 3),
                           floor_xy=[round(aisle_x + 0.1, 3), round(base_y - 2.4, 3)])))
    fallen_sku = "cola_15l"
    fs = geom.BY_ID[fallen_sku]
    entities.append(Entity(id="fallen_0", kind="spill", cls="fallen_product",
                           pass_index=pidx.next(),
                           center=(aisle_x - 0.3, base_y + 2.6, fs["d"] / 2),
                           size=(fs["h"], fs["w"], fs["d"]), rot_z=math.pi / 2,
                           present_from=6, meta=dict(sku=fallen_sku,
                           onset_s=round(6 / fps, 3),
                           floor_xy=[round(aisle_x - 0.3, 3), round(base_y + 2.6, 3)])))
    # distractors (must NOT trigger): painted floor line, paper sign, a dark shadow patch
    entities.append(common.floor_marking("floormark_0", "floor_marking",
                    (aisle_x, base_y, 0.001), (0.12, 6.0, 0.002), pidx,
                    note="aisle guide line"))
    entities.append(common.floor_marking("shadow_0", "shadow",
                    (aisle_x + 0.5, base_y - 3.2, 0.001), (0.8, 1.2, 0.002), pidx,
                    note="cast shadow patch"))
    entities.append(Entity(id="papersign_0", kind="distractor", cls="paper_sign",
                    pass_index=pidx.next(), center=(aisle_x - 0.4, base_y + 0.1, 0.0005),
                    size=(0.21, 0.297, 0.001),
                    meta=dict(note="dropped A4 flyer, flat on floor")))

    # two cameras: a high aisle view and an angled view
    cam_hi = CameraPose("spill_overhead", eye=(aisle_x, base_y, 3.6),
                        target=(aisle_x, base_y + 0.01, 0.0), lens_mm=18,
                        width=RES[0], height=RES[1])
    cam_ang = CameraPose("spill_angled", eye=(aisle_x + 1.6, base_y - 3.5, 1.7),
                         target=(aisle_x, base_y + 0.5, 0.0), lens_mm=24,
                         width=RES[0], height=RES[1])
    gt = dict(hazards=[dict(id=e.id, cls=e.cls, onset_frame=e.present_from,
                            onset_s=e.meta.get("onset_s"),
                            floor_xy=e.meta.get("floor_xy")) for e in entities
                       if e.kind == "spill"],
              distractors=[dict(id=e.id, cls=e.cls) for e in entities
                           if e.kind == "distractor"])
    return Scenario(name="spills_aisle", kind="spills", seed=seed, fps=fps,
                    n_frames=n_frames, cameras=[cam_hi, cam_ang], entities=entities,
                    params=dict(aisle_x=round(aisle_x, 3)), ground_truth=gt)


# ------------------------------------------------------------------------ people
def build_people(plan: dict, seed: int = 3, n_frames: int = 16, fps: int = 6,
                 n_people: int = 5, **_) -> Scenario:
    pidx = common.PassIndex()
    store = plan["store"]
    w2, d2 = store["width"] / 2, store["depth"] / 2
    ppl = people_mod.generate_people(plan, n_people, n_frames, fps, seed)
    for p in ppl:
        p.pass_index = pidx.next()
    footprints = common.footprint_entities(plan, geom.all_fixture_ids(plan), pidx)
    cams = [
        CameraPose("cam_entrance", eye=(w2 - 3.5, -d2 + 1.0, 2.6),
                   target=(2.0, 2.0, 1.0), lens_mm=16, width=RES[0], height=RES[1]),
        CameraPose("cam_aisle", eye=(plan["gondolas"][1]["x"] + plan["bay_w"], -d2 + 2.0, 2.4),
                   target=(plan["gondolas"][1]["x"] + plan["bay_w"], 6.0, 1.0),
                   lens_mm=20, width=RES[0], height=RES[1]),
        CameraPose("cam_checkout", eye=(3.0, -d2 + 4.5, 2.6), target=(1.5, -d2 + 2.0, 0.9),
                   lens_mm=20, width=RES[0], height=RES[1]),
    ]
    gt = dict(trajectories=people_mod.trajectories_gt(ppl, n_frames),
              sessions=[p.meta["session_id"] for p in ppl])
    return Scenario(name="people_movement", kind="people", seed=seed, fps=fps,
                    n_frames=n_frames, cameras=cams, entities=ppl + footprints,
                    params=dict(n_people=n_people), ground_truth=gt)


# --------------------------------------------------------------------- misplaced
def build_misplaced(plan: dict, seed: int = 4, n_frames: int = 1, fps: int = 1,
                    fixture_id: str = "G1", side: str = "R", bay: int = 2, **_) -> Scenario:
    pidx = common.PassIndex()
    front_x, facing = common.gondola_side_front(plan, fixture_id, side)
    bay_y = _bay_y(plan, fixture_id, bay)
    slots = [s for s in geom.iter_slots(plan, fixtures=[fixture_id])
             if s.facing_dir == facing and s.bay_index == bay]
    prods: List[Entity] = []
    for slot in slots:
        for f in slot.facings:
            prods.append(Entity(id=f"{f.slot_id}_f{f.facing_index}", kind="product",
                                cls=f.sku, pass_index=pidx.next(), center=f.center,
                                size=f.size, rot_z=f.rot_z,
                                meta=dict(slot_id=f.slot_id, cat=slot.cat)))
    # inject two misplacements: swap one facing's SKU, and drop a chilled SKU here
    misplacements = []
    target = next(p for p in prods if p.meta["slot_id"] == slots[1].slot_id)
    wrong_sku = "cola_can"          # a soft drink on a cereal shelf
    expected = target.cls
    shelf_z = target.center[2] - target.size[2] / 2        # recover the shelf surface
    target.cls = wrong_sku
    target.size = common.world_size(wrong_sku, target.rot_z)
    cx, cy, _ = target.center
    target.center = (cx, cy, shelf_z + target.size[2] / 2)
    target.meta.update(misplaced=True, expected_sku=expected, actual_sku=wrong_sku)
    misplacements.append(dict(slot=target.meta["slot_id"], expected_sku=expected,
                              actual_sku=wrong_sku, type="wrong_slot"))
    # a chilled dairy item left on this ambient shelf
    chilled = "milk_whole"
    cs = geom.BY_ID[chilled]
    top = slots[0]
    place = top.facings[-1]
    chill_ent = Entity(id=f"{top.slot_id}_misplaced_chilled", kind="product", cls=chilled,
                       pass_index=pidx.next(),
                       center=(place.center[0], place.center[1] + 0.14, top.shelf_z + cs["h"] / 2),
                       size=common.world_size(chilled, place.rot_z), rot_z=place.rot_z,
                       meta=dict(slot_id=top.slot_id, cat=top.cat, misplaced=True,
                                 expected_sku="(ambient %s)" % top.cat, actual_sku=chilled,
                                 reason="chilled item on ambient shelf"))
    prods.append(chill_ent)
    misplacements.append(dict(slot=top.slot_id, expected_sku="ambient:%s" % top.cat,
                              actual_sku=chilled, type="chilled_on_ambient"))
    cams = [_shelf_cam(f"shelf_{fixture_id}{side}_b{bay}", front_x, bay_y,
                       standoff=1.6, lens=35)]
    return Scenario(name=f"misplaced_{fixture_id}{side}_b{bay}", kind="misplaced",
                    seed=seed, fps=fps, n_frames=n_frames, cameras=cams, entities=prods,
                    params=dict(fixture_id=fixture_id, side=side, bay=bay),
                    ground_truth=dict(misplacements=misplacements))


# --------------------------------------------------------------------- multiview
def build_multiview(plan: dict, seed: int = 5, n_views: int = 5,
                    fixture_id: str = "G1", side: str = "R", bay: int = 2, **_) -> Scenario:
    pidx = common.PassIndex()
    front_x, facing = common.gondola_side_front(plan, fixture_id, side)
    bay_y = _bay_y(plan, fixture_id, bay)
    slots = [s for s in geom.iter_slots(plan, fixtures=[fixture_id])
             if s.facing_dir == facing and s.bay_index == bay]
    prods: List[Entity] = []
    for slot in slots:
        for f in slot.facings:
            prods.append(Entity(id=f"{f.slot_id}_f{f.facing_index}", kind="product",
                                cls=f.sku, pass_index=pidx.next(), center=f.center,
                                size=f.size, rot_z=f.rot_z,
                                meta=dict(slot_id=f.slot_id, level=slot.level)))
    # posed camera arc across the bay front (one frame, n_views poses)
    cams: List[CameraPose] = []
    for i in range(n_views):
        a = (i / max(1, n_views - 1) - 0.5)        # -0.5..0.5
        cams.append(CameraPose(
            name=f"view_{i}",
            eye=(front_x + 1.3 + 0.2 * abs(a), bay_y + a * 1.6, 1.0 + 0.4 * (i % 2)),
            target=(front_x - 0.1, bay_y, 1.0), lens_mm=28,
            width=RES[0], height=RES[1]))
    # per-slot remaining-depth-of-stock ground truth (volumetric fill)
    fill = []
    for slot in slots:
        rows = max(1, min(3, int((slot.depth - geom.FRONT_GAP) //
                                 (max(geom.BY_ID[f.sku]["d"] for f in slot.facings) + geom.ROW_STEP_PAD))
                          )) if slot.facings else 0
        stock_depth = 0.0
        if slot.facings:
            d = max(geom.BY_ID[f.sku]["d"] for f in slot.facings)
            stock_depth = rows * (d + geom.ROW_STEP_PAD)
        fill.append(dict(slot=slot.slot_id, level=slot.level, shelf_depth=round(slot.depth, 3),
                         stock_depth_m=round(stock_depth, 3),
                         fill_ratio=round(min(1.0, stock_depth / slot.depth), 3) if slot.depth else 0,
                         n_facings=len(slot.facings)))
    scn = Scenario(name=f"multiview_{fixture_id}{side}_b{bay}", kind="multiview",
                   seed=seed, fps=1, n_frames=1, cameras=cams, entities=prods,
                   with_depth=True, params=dict(fixture_id=fixture_id, side=side,
                   bay=bay, n_views=n_views),
                   ground_truth=dict(slot_fill=fill))
    return scn


REGISTRY = {
    "depletion": build_depletion,
    "spills": build_spills,
    "people": build_people,
    "misplaced": build_misplaced,
    "multiview": build_multiview,
}
