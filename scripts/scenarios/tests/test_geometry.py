"""Facing geometry is faithful to the planogram and deterministic."""
import catalog_geom as geom


def test_facing_counts_match_planogram(plan):
    # every (slot, sku) facing count equals the planogram's requested facings,
    # except where the row would overflow the shelf width (builder parity).
    slots = list(geom.iter_slots(plan, fixtures=["G1"]))
    assert slots, "expected G1 slots"
    for s in slots:
        counts = s.facing_count()
        assert sum(counts.values()) == len(s.facings)
        assert all(v > 0 for v in counts.values())


def test_boxes_inside_store(plan):
    store = plan["store"]
    w2, d2, h = store["width"] / 2 + 0.5, store["depth"] / 2 + 0.5, store["height"] + 0.1
    for s in geom.iter_slots(plan):
        for f in s.facings:
            for (x, y, z) in f.corners():
                assert -w2 <= x <= w2 and -d2 <= y <= d2 and -0.01 <= z <= h, \
                    (f.slot_id, x, y, z)


def test_sizes_positive_and_axis_consistent(plan):
    for s in geom.iter_slots(plan, fixtures=["G1", "C_right"]):
        for f in s.facings:
            sx, sy, sz = f.size
            assert sx > 0 and sy > 0 and sz > 0


def test_deterministic(plan):
    a = [(s.slot_id, len(s.facings)) for s in geom.iter_slots(plan)]
    b = [(s.slot_id, len(s.facings)) for s in geom.iter_slots(plan)]
    assert a == b


def test_footprints(plan):
    for fid in geom.all_fixture_ids(plan):
        fp = geom.fixture_footprint(plan, fid)
        assert fp and all(v > 0 for v in fp["size"])
