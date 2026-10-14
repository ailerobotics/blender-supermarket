"""Each scenario builds deterministically, with sane ground truth and on-image boxes."""
import annotate
import scenario_defs as sd


def _build(plan, kind):
    return sd.REGISTRY[kind](plan)


def test_all_build(plan):
    for kind in sd.REGISTRY:
        scn = _build(plan, kind)
        assert scn.kind == kind
        assert scn.n_frames >= 1 and scn.cameras and scn.entities


def test_deterministic_given_seed(plan):
    a = _build(plan, "depletion")
    b = _build(plan, "depletion")
    assert [e.id for e in a.entities] == [e.id for e in b.entities]
    assert a.ground_truth["events"] == b.ground_truth["events"]


def test_projected_boxes_in_bounds(plan):
    for kind in sd.REGISTRY:
        scn = _build(plan, kind)
        for cam in scn.cameras:
            for frame in range(min(scn.n_frames, 3)):
                ann = annotate.image_annotation(scn, cam, frame)
                for a in ann["annotations"]:
                    x, y, w, h = a["bbox"]
                    assert w > 0 and h > 0
                    assert -0.5 <= x and -0.5 <= y
                    assert x + w <= cam.width + 0.5 and y + h <= cam.height + 0.5


def test_depletion_counts_monotonic_without_restock(plan):
    scn = sd.build_depletion(plan, n_remove=6, n_restock=0, n_shoppers=0, n_frames=12)
    pf = scn.ground_truth["per_frame"]
    totals = [sum(sum(sk.values()) for sk in f["slots"].values()) for f in pf]
    assert totals == sorted(totals, reverse=True)      # only removals -> non-increasing
    assert totals[0] > totals[-1]


def test_spills_have_onset_and_distractors(plan):
    scn = sd.build_spills(plan)
    gt = scn.ground_truth
    classes = {h["cls"] for h in gt["hazards"]}
    assert {"water", "milk", "juice", "oil"} <= classes
    assert {"debris", "fallen_product"} <= classes
    assert {d["cls"] for d in gt["distractors"]} == {"floor_marking", "shadow", "paper_sign"}
    for h in gt["hazards"]:
        assert h["onset_frame"] is not None and h["floor_xy"]


def test_misplaced_gt(plan):
    scn = sd.build_misplaced(plan)
    mis = scn.ground_truth["misplacements"]
    assert any(m["type"] == "wrong_slot" for m in mis)
    assert any(m["type"] == "chilled_on_ambient" for m in mis)
    for m in mis:
        assert m["expected_sku"] != m["actual_sku"]


def test_multiview_depth_and_fill(plan):
    scn = sd.build_multiview(plan, n_views=4)
    assert scn.with_depth and len(scn.cameras) == 4
    for f in scn.ground_truth["slot_fill"]:
        assert 0 <= f["fill_ratio"] <= 1 and f["shelf_depth"] > 0
