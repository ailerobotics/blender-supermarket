#!/usr/bin/env python3
"""Generate planogram.json: store layout + every bay's shelf heights and facings.

Edit the LAYOUT tables below (or the JSON afterwards) and re-run the builder.
Coordinates: metres, Z up, origin at floor centre. Gondola runs are along +Y.
Side "L" = face pointing to -X, side "R" = face pointing to +X.
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from catalog import SKUS, BY_ID, CATEGORIES, by_cat

random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "planogram.json")

STORE = dict(width=26.0, depth=22.0, height=4.0)      # interior size
BAY_W, N_BAYS, RUN_Y0 = 1.25, 6, -3.0                 # gondola run: 6 bays of 1.25 m from y=-3.75
GONDOLA_PITCH, GONDOLA_X0 = 3.05, -9.15
MIN_PITCH = 0.28   # shelves never closer than this (top-to-top)
FIX = dict(base_z=0.17, base_depth=0.57, shelf_depth=0.47, shelf_t=0.03, height=2.0,
           upright_w=0.05, back_t=0.04, clearance=0.045, endcap_w=1.0)

# gondola runs: (id, L-side categories [(cat, bays)], R-side categories, front endcap sku, back endcap sku)
GONDOLAS = [
    ("G1", [("cereal", 6)],        [("coffee_tea", 6)],                   "chips_paprika", "cereal_honey"),
    ("G2", [("snacks", 6)],        [("confectionery", 6)],                "cola_can",      "candy_gummy"),
    ("G3", [("soft_drinks", 6)],   [("water_juice", 6)],                  "water_15l",     "juice_orange"),
    ("G4", [("canned", 6)],        [("pasta_rice", 6)],                   "pasta_spag",    "can_tomato"),
    ("G5", [("baking", 6)],        [("condiments", 6)],                   "oil_olive",     "ketchup"),
    ("G6", [("cleaning", 6)],      [("paper", 4), ("pet", 2)],            "toilet_paper",  "detergent"),
    ("G7", [("personal_care", 6)], [("cleaning", 3), ("personal_care", 3)], "choc_milk",   "shampoo"),
]
# single-sided wall units: (id, x, y0, facing, n_bays, categories, height, levels_max)
WALL_UNITS = [
    ("W_left",  -12.85, -3.0, "+X", 6, [("beer_wine", 6)]),
    ("W_back",  -11.5, 10.85, "-Y", 5, [("bakery", 3), ("confectionery", 2)]),
]
# chillers (glass door reach-in cabinets), single sided: (id, x, y0, facing, n_bays, categories)
CHILLERS = [
    ("C_right", 12.85, -1.0, "-X", 8, [("dairy", 8)]),
    ("C_back",  -2.5, 10.85, "-Y", 12, [("dairy", 6), ("water_juice", 3), ("soft_drinks", 3)]),
]

def level_plan(skus, max_top, base_z=FIX["base_z"], max_levels=6, chiller=False):
    """Assign SKUs to shelf levels by height (tall at the bottom) and return
    [(shelf_z, [sku ids])] bottom-up. Shelf z = top surface of the shelf."""
    skus = sorted(skus, key=lambda s: -s["h"])
    n = len(skus)
    # group heights into up to max_levels bands so that each band fits below max_top
    levels = []
    z = base_z
    i = 0
    while i < n and len(levels) < max_levels:
        band = []
        hmax = skus[i]["h"]
        # put SKUs with similar height (within 40% of the tallest) in this band
        while i < n and skus[i]["h"] >= 0.6 * hmax and len(band) < max(2, n // max(1, max_levels - len(levels)) + 1):
            band.append(skus[i]["id"]); i += 1
        levels.append((round(z, 3), band, hmax))
        z += max(hmax + FIX["clearance"] + FIX["shelf_t"], MIN_PITCH)
    # any leftovers go on the last level
    while i < n:
        levels[-1][1].append(skus[i]["id"]); i += 1
    # if we have spare height, add empty repeat levels re-using short skus
    while z + MIN_PITCH < max_top and len(levels) < max_levels:
        short = [s for s in skus if s["h"] <= max_top - z - FIX["clearance"] - FIX["shelf_t"]]
        if not short: break
        band = [s["id"] for s in random.sample(short, min(len(short), 3))]
        hmax = max(BY_ID[b]["h"] for b in band)
        levels.append((round(z, 3), band, hmax))
        z += max(hmax + FIX["clearance"] + FIX["shelf_t"], MIN_PITCH)
    return levels

def fill_shelf(width, sku_ids, level_index, rng):
    """Return [(sku, facings)] blocks that fill `width` metres."""
    items, x = [], 0.0
    gap = 0.012
    order = sku_ids[:]
    rng.shuffle(order)
    k = 0
    while order:
        sku = BY_ID[order[k % len(order)]]
        w = sku["w"] + gap
        remaining = width - x
        if remaining < w: break
        # blocks of 2..6 facings, capped by remaining width
        want = rng.randint(2, 6) if sku["w"] > 0.09 else rng.randint(4, 10)
        facings = max(1, min(want, int(remaining // w)))
        items.append([sku["id"], facings])
        x += facings * w
        k += 1
        if k > 40: break
    return items

def side_bays(cats, n_bays, side_id, rng, top_z=FIX["height"] - 0.1, depth_base=FIX["base_depth"], depth_up=FIX["shelf_depth"], chiller=False):
    bays = []
    bay_i = 0
    for cat, nb in cats:
        skus = by_cat(cat)
        levels = level_plan(skus, top_z, base_z=(0.12 if chiller else FIX["base_z"]), max_levels=6)
        for b in range(nb):
            shelves = []
            for li, (z, ids, hmax) in enumerate(levels):
                depth = depth_up if (li > 0 or chiller) else depth_base
                shelves.append(dict(z=z, depth=depth, max_h=round(hmax, 3),
                                    items=fill_shelf(BAY_W - FIX["upright_w"] - 0.02, ids, li, rng)))
            bays.append(dict(index=bay_i, cat=cat, shelves=shelves))
            bay_i += 1
    return bays

def main():
    rng = random.Random(42)
    plan = dict(store=STORE, fixture=FIX, bay_w=BAY_W, gondolas=[], wall_units=[], chillers=[], endcaps=[], aisles=[])
    for gi, (gid, L, R, ec_front, ec_back) in enumerate(GONDOLAS):
        x = GONDOLA_X0 + gi * GONDOLA_PITCH
        g = dict(id=gid, x=x, y0=RUN_Y0, n_bays=N_BAYS,
                 sides={"L": dict(cats=[c for c, _ in L], bays=side_bays(L, N_BAYS, "L", rng)),
                        "R": dict(cats=[c for c, _ in R], bays=side_bays(R, N_BAYS, "R", rng))})
        plan["gondolas"].append(g)
        run_len = N_BAYS * BAY_W
        for which, sku, y, facing in (("front", ec_front, RUN_Y0 - FIX["endcap_w"] / 2 - 0.02, "-Y"),
                                      ("back", ec_back, RUN_Y0 + run_len + FIX["endcap_w"] / 2 + 0.02, "+Y")):
            s = BY_ID[sku]
            levels = level_plan([s], FIX["height"] - 0.1, max_levels=6)
            shelves = [dict(z=z, depth=(FIX["base_depth"] if li == 0 else FIX["shelf_depth"]), max_h=round(hm, 3),
                            items=fill_shelf(FIX["endcap_w"] - FIX["upright_w"] - 0.02, ids, li, rng))
                       for li, (z, ids, hm) in enumerate(levels)]
            plan["endcaps"].append(dict(id=f"{gid}_{which}", x=x, y=y, facing=facing, w=FIX["endcap_w"], sku=sku, shelves=shelves))
    for wid, x, y0, facing, nb, cats in WALL_UNITS:
        plan["wall_units"].append(dict(id=wid, x=x, y0=y0, facing=facing, n_bays=nb, cats=[c for c, _ in cats],
                                       bays=side_bays(cats, nb, "W", rng, top_z=2.2)))
    for cid, x, y0, facing, nb, cats in CHILLERS:
        plan["chillers"].append(dict(id=cid, x=x, y0=y0, facing=facing, n_bays=nb, cats=[c for c, _ in cats],
                                     bays=side_bays(cats, nb, "C", rng, top_z=1.85, depth_base=0.55, depth_up=0.55, chiller=True)))
    # aisle signs: aisle k is the corridor left of gondola k (k=1..7) and right of G7 (k=8)
    xs = [g["x"] for g in plan["gondolas"]]
    faces = []  # (x, categories) of every face bordering the corridors, left to right
    faces.append((WALL_UNITS[0][1] + 0.3, [c for c, _ in WALL_UNITS[0][5]]))
    for g in plan["gondolas"]:
        faces.append((g["x"] - 0.6, g["sides"]["L"]["cats"]))
        faces.append((g["x"] + 0.6, g["sides"]["R"]["cats"]))
    faces.append((CHILLERS[0][1] - 0.4, [c for c, _ in CHILLERS[0][5]]))
    for k in range(0, len(faces) - 1, 2):
        (xa, ca), (xb, cb) = faces[k], faces[k + 1]
        cats = []
        for c in ca + cb:
            if c not in cats: cats.append(c)
        plan["aisles"].append(dict(no=k // 2 + 1, x=round((xa + xb) / 2, 3), y=RUN_Y0 - 1.2,
                                   left=[CATEGORIES[c] for c in ca], right=[CATEGORIES[c] for c in cb]))
    json.dump(plan, open(OUT, "w"), indent=1)
    n_items = sum(f for g in plan["gondolas"] for side in g["sides"].values() for b in side["bays"] for sh in b["shelves"] for _, f in sh["items"])
    print(f"wrote {OUT}: {len(plan['gondolas'])} gondolas, {len(plan['endcaps'])} endcaps, {len(plan['wall_units'])} wall units, {len(plan['chillers'])} chillers, {len(plan['aisles'])} aisle signs, ~{n_items} gondola facings")

if __name__ == "__main__":
    main()
