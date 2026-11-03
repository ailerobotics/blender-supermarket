# sim-robust-portal-floor-seam-v1

Establish — on CPU, before any code — what physically stops the coupled outbound tow at
**x = −3.935 m** at the `sales_backroom_portal`, and remove it. Branch
`fm/sim-robust-portal-floor-seam-v1` off `fm/twin-robot-in-store-boot-v1` @ d37341b (after
PRs 243–259).

TL;DR: **there is no collider edge, seam, or z-step at the pin.** The `Backroom_floor` is a
single flat continuous slab. The pin is a **PhysX triangle-mesh internal-edge (“ghost”)
contact** on that flat floor: the cart’s front-left caster lands on the floor’s cooked quad
**diagonal**, and the coupled base halts at `cart_x + hitch_offset = −5.205 + 1.27 = −3.935`
to the millimetre. The fix authors the flat drive floor as a **box (`boundingCube`) collider**
— no internal edge — behind the opt-in `portal_floor_box` switch (default off ⇒ byte-identical).

## 1. Step 1 — the geometry, on CPU, before any code

### 1a. Base rig at the pin (`scripts/twin/mobile_assets.py`, `MobileSpec`)

- Drive wheels are jointed at **base-local x = 0** (`{side}_wheel_joint` origin
  `(0, ±track/2, r−base_z)`), collider a **cylinder** r=0.16, width 0.08, axis Y. So the
  **drive-wheel ground contact x == the base origin x**: a base pinned at x=−3.935 has its
  drive contact at −3.935.
- Trailing casters at base-local x = ±(length/2−0.12) = ±0.38; caster wheel r=0.065 raised by
  `caster_lift_m` (0.03 in the pinned runs) ⇒ ground contact **0.03 m above** the drive plane,
  i.e. **airborne**; the two drive wheels bear the load. The trace confirms this exactly
  (`wheel_z` drive L/R 0.160 steady ≤0.4 mm; casters front 0.095 / rear 0.094 steady;
  `contact` left=True right=True, front=False rear=False).
- Coupled hitch is rigid in x: `base_x + hitch_x == cart_x + cart_hitch_x` ⇒
  **`base_x − cart_x = cart_hitch_x − hitch_x = 0.65 − (−0.62) = 1.27 m`.**

### 1b. The composed stage has NO collider at the pin

Flattened `mission.usda` (+ the `supermarket.usdc` sublayer) read with the `.venv-usd`
`usd-core`, traversing **instance proxies and prototypes**, every prim with a Collision API /
every Gprim, and **actual mesh vertices** (not just AABBs). Over the pin region
(x∈[−4.25,−3.55], y∈[14.6,15.9]) the only floor-level collision geometry is:

| prim | world AABB | note |
|---|---|---|
| `GRP_Structure/Backroom_floor` | x[−12,1] y[11,19.25] **z[−0.12, 0.0]** | single flat slab, top z=0, **8 verts, 6 quad faces — no internal vertex/seam** |
| `GRP_Structure/Ground_exterior` | x[−60,60] y[−60,60] z[−0.03,−0.01] | 1 cm underlay, no interference |
| `GRP_Structure/Backroom_ceiling` | … z[4.0,4.05] | overhead |
| `GRP_EnvironmentDecor/backroom_turning_line_*` | z[0.002,0.004] | painted floor lines, **collision_role=none (non-physical)** |

- **Point-level check:** 0 collision-prim vertices within **0.25 m** of any drive-wheel,
  front-corner, front-caster or chassis-centre probe at the pin pose.
- **All-collider x-edge sweep:** no collision prim has an x-edge within 6 cm of −3.935 /
  −3.875 / −3.775 at floor level and y∈[15.2,15.3]. (The nearest, `Rack_3` posts at
  x≈−3.95, sit at y≈17.9–19.0, ~2.7 m north of the lane.)
- **All floor/ground meshes:** exactly three (above). Sales `Floor` (y[−11,11]) abuts the
  backroom floor at y=11 — ~4 m south of the lane, never under it.

So the captain’s / validator’s floor-edge hypothesis (“backroom floor-plane end or sales-floor
boundary, a vertical rim or sub-mm seam”) is **refuted by the geometry**: there is no such
feature. `check_portal_floor_seam.py` reproduces the base-rig numbers; the scan scripts are in
the PR description’s investigation log.

### 1c. The trace discriminator (launch1b, `RESTOCK_TRACE_CASTERS=1`)

The base cruises straight into the pin then x locks at **−3.9350 (±0.1 mm) for ~27 s** while the
controller keeps commanding the launch cruise; `wheel_z` rock-steady (no climb); both casters
airborne; yaw 0.0–0.9° (aligned). A geometrically static hold on a flat floor with a horizontal
contact normal — i.e. an **in-plane edge**, not a z-step.

## 2. Root cause — a PhysX triangle-mesh internal-edge ghost contact

The `Backroom_floor` collision is authored with **`physics:approximation = "none"`** — a full
**triangle mesh**. PhysX cooks the top quad `[(-12,11),(1,11),(1,19.25),(-12,19.25)]` into two
triangles sharing one internal **diagonal edge** (fan from v0: the segment
`(-12,11)→(1,19.25)`). On a flat surface that internal edge produces a spurious in-plane
contact normal (“internal-edge” / “ghost” collision) when a rigid roller comes to low speed
across it.

The **cart’s front-left swivel caster** lands on that diagonal. Perpendicular distance
front-left caster → cooked diagonal, from each run’s nav-trace cart pose
(`scripts/twin/restock/portal_floor.py`, reproduced by `check_portal_floor_seam.py`):

| run | approach cruise | front-left caster → diagonal | result |
|---|---|---|---|
| launch1b | 0.192 | **−0.002 m** (on the edge) | PIN |
| coldwin18 | 0.181 | **−0.014 m** | PIN |
| coldwin9 | **0.222** | **+0.040 m** | PIN |
| coldwin17 | 0.206 | ~0 | PIN |
| coldwin10 | 0.214 | ~0 | PIN |
| **launch1a** | 0.208 | **+0.238 m** (clear, in the NW triangle) | **CROSS** |

- **Momentum is not the discriminator** — coldwin9 approached *faster* than the passer
  launch1a and still pinned. **Caster-on-edge is.** launch1a was ~0.2 m north of the lane
  (y≈15.45 vs the authored lane y≈15.25), so its caster had already crossed the diagonal and
  was clear.
- **The exact x comes from the hitch**, not from the base’s own wheels: the cart catches at a
  near-fixed pose (cart_x ≈ −5.205), and `base_x = cart_x + 1.27 = −3.935` — matching the whole
  7-run cluster (−3.931…−3.935) and launch1b’s `base − cart = 1.269 m` in the trace.
- The alternative quad diagonal (v1–v3) puts the front-left caster **+0.609 m** away — it does
  not fit — confirming PhysX’s fan-from-v0 triangulation is the one the caster sits on.

This fully explains every trace fact: flat floor (no z change), in-plane normal, both drive
wheels loaded / casters airborne (the caught body is a *cart* caster, transmitted through the
hitch), millimetre-exact x (hitch offset off a fixed edge), and “stochastic” only in the sense
of whether the couple seat / lane-y leaves a cart caster on vs clear of the edge.

## 3. The fix — flat drive floor as a box collider (opt-in, default off, byte-identical off)

A flat floor has no business being a triangle-mesh collider: a **box (`boundingCube`)** is the
same slab with six flat faces and **no internal edge**, so the casters roll across cleanly.

- `twin/colliders.py`: new kind `BOX` (approximation `"boundingCube"`), set
  `DRIVE_FLOOR_MESHES = {"Backroom_floor"}`, and `classify(prim, *, flat_floor_box=False)` —
  when on, the drive-floor mesh classifies as `BOX` instead of `TRIANGLE_MESH`. Every other
  structure/fixture/shelf/wall mesh is untouched; `flat_floor_box=False` (the default) is
  byte-identical to the historical `classify(prim)`.
- `twin/compose.py`: `author_static_colliders(stage, root, *, flat_floor_box=False)` threads it
  and applies the box approximation via the same `_apply_collider` path.
- `twin/restock/scene.py`: `build(..., portal_floor_box=None)` → `floor_box` → the composer;
  records `portal_floor_box` in the build result.
- `twin/restock/physics_setup.py`: passes `config.get('portal_floor_box')` and prints a
  labelled `PORTAL_FLOOR_BOX enabled=…` line next to `CAGE_COLLISION` / `VISUALS`.

The box slab is geometrically identical to the floor (top z=0, same footprint); it changes no
visual, geometry, mass, joint, camera or marker, so the sensing/decode/physics-mass suites are
unaffected. The whole bounded outbound-to-bay path (x −4.5…−2.0, y≈15.3–15.45) stays on
`Backroom_floor`, so this one mesh covers the proof.

### CPU before/after (the seam is gone)

`scripts/check_portal_floor_seam.py` (exit 0 ⇒ the switch removes the portal-band internal edge):

```
triangle-mesh floor (switch OFF): 1 internal edge in the portal caster band  <-- the defect
box floor           (switch ON ): 0 internal edge in the portal caster band  <-- seam GONE
authored round-trip: flat_floor_box=False -> approximation 'none'; =True -> 'boundingCube'  OK
```

CPU tests `scripts/twin/tests/test_portal_floor_seam.py` (14, pxr-optional): policy on/off +
byte-identical off + non-drive meshes untouched; base-rig numbers (hitch 1.27, drive contact =
base origin, caster geometry); the defect/fix edge counts; the recorded-pin caster-on-diagonal
evidence; and the authored-stage approximation round-trip. `test_colliders.py`,
`test_compose.py`, `test_restock_scene.py` still green.

## 4. Bounded GPU proof — handed to the integration lane (not run here)

`docs/examples/portal-floor-box-mission-fragment.json` — merge the one top-level key
`"portal_floor_box": true` ON TOP of the PR258 + PR259 navigation blocks (keep them on).
Recipe: `run_mobile_restock.py --start-at couple --stop-after outbound` from the **coldwin18
seat**, `RESTOCK_TRACE_CASTERS=1`, flock `/tmp/aile-gpu.lock`, `HEAVY_MB=2500` if whispr holds
VRAM, **never `--force`, never touch whispr (pid 1339)**, **repeated 3×**. PASS = all three
`check_tow_traction.py --gpu-run` → GPU_RUN PASS (reached the bay, cross x=−3.90,
`single_tick_stops=0`, `trace_has_casters=true`) with **no `NAV_BREAKAWAY enter` / no
`coupled_no_progress`**; cross-check `check_portal_floor_seam.py --run <run>` on a PASS. Then the
full capstone.

## 5. Scope

Out of scope / untouched: any reverse motion / the coupled-reverse interlock, PR207’s drive
force & `caster_lift_m`, PR241’s lead-in, PR201’s thresholds, PR236’s bound, PR250’s rock,
PR258’s heading-hold, PR259’s momentum launch, the couple fine-dock, and any cart mass / friction.
If the capstone later tows onto the **sales floor** (`GRP_Structure/Floor`, y<11 — no real run
has reached it), that slab is also a triangle mesh with its own cooked diagonal; add it to
`DRIVE_FLOOR_MESHES` then, as a follow-up.

## Files

- `scripts/twin/colliders.py` — `BOX` / `APPROX["boundingCube"]` / `DRIVE_FLOOR_MESHES` /
  `classify(flat_floor_box=…)` / `plan_colliders` box count.
- `scripts/twin/compose.py` — `author_static_colliders(flat_floor_box=…)`.
- `scripts/twin/restock/scene.py`, `physics_setup.py` — `portal_floor_box` wiring + labelled log.
- `scripts/twin/restock/portal_floor.py` — pure-geometry root-cause/analysis (no pxr).
- `scripts/check_portal_floor_seam.py` — CPU before/after check + recorded-pin evidence.
- `scripts/twin/tests/test_portal_floor_seam.py` — CPU tests.
- `docs/examples/portal-floor-box-mission-fragment.json`, `docs/restock_navigation.md`.
