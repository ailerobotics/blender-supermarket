# Blender Supermarket — a generated digital twin

A walk-through model of a railway-station supermarket ("Freshmart"), built for
robotics simulation. 26 × 22 m of shop floor, seven double-sided gondola runs on
a real planogram, 93 fictional-brand SKUs with printed labels and shelf-edge
price tags, glass-door and open chillers, four checkout lanes, a produce corner,
a bakery wall and hanging aisle signage — about 26 000 objects and 23 400 product
instances, built headless in ~4 seconds and exported to USD for Isaac Sim.

The **commands** to drive all of this live in **[RUNBOOK.md](RUNBOOK.md)** — start
and stop Blender, talk to it over MCP, regenerate the store, render, and export.
This README is the *why*: the design, what the repository chooses to keep, and the
decisions behind both.

---

## Why the store is generated, not modelled

The store is produced by four small programs from a product catalog and a
planogram. It is deliberately **not** a hand-sculpted Blender scene. The reason is
what the twin is *for*: a shelf-scanning / navigation robot needs a store that can
change — reprice a product, move a category to another aisle, widen the gondola
pitch, add a SKU — and it needs every object to carry machine-readable ground
truth (which SKU, which category, which shelf slot, what price). Both of those are
painful in a hand-built mesh and trivial when the store is a function of data:

- **Editable and reproducible.** The layout is a set of Python tables
  (`GONDOLAS`, `WALL_UNITS`, `CHILLERS` in `scripts/gen_planogram.py`) and a
  product list (`scripts/catalog.py`). Change a value, re-run, and the whole store
  rebuilds deterministically in seconds. Nothing is drawn by hand, so nothing
  drifts out of sync.
- **Labelled by construction.** Because every product is placed by code from the
  catalog, the builder writes custom properties onto each object (sku, category,
  brand, price, shelf slot, facing, row) and onto each fixture. Those survive the
  USD export as `userProperties:*` and become free training labels downstream — see
  *Consuming the USD export* below.
- **Cheap to scale.** All copies of one SKU share a single prototype (a collection
  instance), so 23 400 products cost 196 real meshes on disk and export as true USD
  instances. Modelling that by hand is not sensible; generating it is.

The result is a store that is data first and geometry second — the geometry is an
output, and the inputs (catalog, planogram tables) are the things a human edits.

---

## The four stages

The pipeline composes as **catalog → planogram → labels → scene**, each stage
consuming the previous stage's output. RUNBOOK.md §3 has the exact commands.

| # | Stage | Program | Reads | Writes |
|---|-------|---------|-------|--------|
| 1 | **Catalog** | `scripts/catalog.py` | — (it *is* the source data) | 93 SKUs in memory: real dimensions, packaging form, brand, colours, price, weight |
| 2 | **Planogram** | `scripts/gen_planogram.py` | catalog | `scripts/planogram.json` — store size, every gondola/wall/chiller, and each bay's shelf heights and facings |
| 3 | **Labels** | `scripts/gen_labels.py` | catalog + planogram | `assets/labels/*.png` (product artwork + `tag_*` price tags), `assets/signs/*.png` (aisle, category and store signs), rendered with PIL |
| 4 | **Scene** | `scripts/build_supermarket.py` | base scene + planogram + labels + catalog | the built `.blend` (see RUNBOOK §3 for the `--save` command) |

Stage 4 runs in 13 named sub-stages
(`clear structure library gondolas walls chillers front produce signage lighting
cameras hierarchy render`) either headless (`blender -b … --python
build_supermarket.py`) or live inside the running GUI through the MCP bridge, one
stage per call. From the built scene, three leaf programs branch off — stills
(`render_shots.py`), the fly-through video (`render_walkthrough.py`) and the USD
export (`export_usd.py`) — plus a self-contained HTML report (`gen_report.py`).

**Order matters for a fresh checkout:** stage 4 loads the label PNGs from
`assets/labels/`, so run stage 3 (`gen_labels.py`) before the build. The committed
`planogram.json` means you can skip stage 2 unless you have changed the layout
tables. See *Rebuilding from a fresh clone* below.

> **Path caveat.** `build_supermarket.py`, `export_usd.py`, `render_shots.py` and
> `render_walkthrough.py` hardcode the absolute project root
> `~/workspaces/blender-supermarket` (for `ROOT` and for default output paths).
> `gen_planogram.py` and `gen_labels.py` use paths relative to the script and work
> from any location. To rebuild in a clone at a different path, either check the
> clone out at `~/workspaces/blender-supermarket` or adjust those absolute paths.

---

## What the base scene carries (and the build cannot regenerate)

`scenes/supermarket_base.blend` is the starting point every headless build opens.
It is small by design and holds only the things that came from the **network** and
therefore cannot be reproduced by running the generator scripts offline:

- **Two Poly Haven PBR materials** — `floor_tiles_08` and `white_plaster_02`
  (2k), downloaded through the MCP `download_polyhaven_asset` tool. The builder
  re-uses them by name (`polyhaven_mat()`), only re-scaling their mapping and
  wiring the normal map; if they are absent it silently falls back to a flat grey
  material, so the store still builds but loses its floor and wall look.
- **The world HDRI** — `wide_street_01_2k.hdr`, the street seen through the
  shopfront glazing and the scene's ambient light.

These are **externally linked**, not packed: the base `.blend` references the JPEG
and HDR files in `assets/textures/` by relative path (verified: the image
datablocks carry no packed data). So the base scene and `assets/textures/` are a
pair — both are required, both came from Poly Haven, and both are therefore kept
in version control (see the file policy). The two Poly Haven materials and the
world are pinned with a fake user so a `.blend` save never purges them for having
no object user.

Everything else in a built scene — all fixtures, all products, all the fixture
materials (`FX_*`, `PR_*`), the lighting, cameras and signage — is generated by
`build_supermarket.py` and needs no network.

---

## The camera set

`build_cameras()` places nine cameras, saved in the scene's `Cameras` collection.
Eight are fixed vantage points; the last is an animated fly-through.

| Camera | Purpose |
|--------|---------|
| `Cam_Entrance` | The establishing shot from just inside the doors — produce, aisle signs, checkouts and chillers in one frame. The report hero image. |
| `Cam_Aisle` | Straight down aisle 4 between water/juice and canned goods — the "shopper's eye" corridor view (1.6 m). |
| `Cam_Shelf` | A close-up of the soft-drinks bay 1.4 m from the ticket rail — for judging label legibility, rims and price tags at shelf distance (the robot's working range). |
| `Cam_Checkout` | Lanes 1–3 with impulse shelves and the street HDRI through the glazing. |
| `Cam_Produce` | The produce tables and crates in front of the dairy chillers. |
| `Cam_Overview` | A high (3.7 m) corner view of the whole floor — wine wall, aisle signs, pallets, checkouts, exit. |
| `Cam_Exterior` | Outside the building — the facade and store sign. |
| `Cam_TopDown` | An orthographic plan view; `render_shots.py` hides the ceiling and lights for it, giving a clean planogram cutaway. |
| `Cam_Walk` | The animated walk-through: 17 waypoints, 30 frames apart, from the entrance through aisles 4 and 6 to a checkout lane and out the exit (510 frames at 24 fps ≈ 21 s), gaze levelled 4 m ahead. `render_walkthrough.py` renders it to `renders/walkthrough.mp4`. |

For roaming the scene live (walk navigation, eye height, gravity), see RUNBOOK §4.

---

## Consuming the USD export

`scripts/export_usd.py` writes `export/supermarket.usdc` (plus `export/textures/`)
for a simulator such as **NVIDIA Isaac Sim**. The stage is **Z-up, 1 metre per
unit**, rooted at prim `/Supermarket`, with UsdPreviewSurface materials, exported
lights and cameras. RUNBOOK §6 has the import steps; the design intent:

- **Instancing.** Products export as USD instances (~24 900 instanceable prims
  over ~196 prototypes; only a few hundred real meshes on the stage). This keeps
  the stage light and means a collider or material change applied to a *prototype*
  applies to every instance of that SKU at once — exactly what you want for tens of
  thousands of products.
- **Semantics for free.** Every product prim carries
  `userProperties:sku, category, brand, price, slot, facing, row`; every fixture
  carries `fixture, gondola, side, bay, level`. Map `sku`/`category` to semantic
  labels for Replicator ground truth; the `slot` id gives planogram-compliance
  labels (the SKU expected at each shelf position) at no extra cost.
- **Physics is added downstream.** Products export as static Xforms. In the
  simulator add colliders (convex hull on products, triangle mesh on fixtures and
  structure) and make only the products the robot handles into rigid bodies (mass
  from the catalog weight); leave the rest static.
- **Lighting note.** The world HDRI is *not* exported (world-material conversion is
  off); light the imported stage with a dome light. PET bottles use transmission —
  if RTX real-time looks noisy, swap their prototype material for an opaque tinted
  plastic in Blender and re-export.

---

## Rebuilding from a fresh clone

A clone contains all the *sources*, the *un-regenerable, network-sourced* assets,
and the *expensive-to-render evidence* (stills, walkthrough, report). The
artifacts that are cheap to rebuild — the built scene, labels/signs and the USD
export — are gitignored; regenerate them with the pipeline below (steps 1–4a).
The render steps (4b) are only needed if you have changed the store and want to
refresh the committed evidence. Full commands are in RUNBOOK.md:

```
# 1. (only if you changed the layout tables) planogram
python3 scripts/gen_planogram.py                      # -> scripts/planogram.json

# 2. labels, tags and signs  (build step below needs these on disk)
python3 scripts/gen_labels.py                         # -> assets/labels, assets/signs

# 3. the scene, from the network-sourced base file
blender -b scenes/supermarket_base.blend --python scripts/build_supermarket.py -- --all --save scenes/supermarket.blend

# 4a. cheap derived output (ignored), from the built scene
blender -b scenes/supermarket.blend --python scripts/export_usd.py            # -> export/  (~26 s)

# 4b. expensive evidence (versioned) — only to refresh after a store change
blender -b scenes/supermarket.blend --python scripts/render_shots.py -- ...   # -> renders/*.png (GPU-hours)
blender -b scenes/supermarket.blend --python scripts/render_walkthrough.py -- ...  # -> renders/walkthrough.mp4 (GPU-hours)
python3 scripts/gen_report.py                         # -> report.html
```

The only thing a clone *cannot* reproduce offline is the base scene's Poly Haven
materials and HDRI — which is exactly why they are versioned rather than
regenerated. Everything else above is reproducible from committed files with no
network (verified — see the decision log).

---

## File policy — what is versioned and why

The governing rule:

> **An artifact is ignored only when its regeneration is both _offline_ and
> _cheap_ — seconds to about a minute — and the measured cost is recorded below.**
> Anything whose reproduction needs the network, or costs more than that (a render
> farm's GPU-hours), is versioned even though a command exists to rebuild it:
> something that takes hours is irreplaceable in practice, and the renders are the
> project's visible evidence.

Possibility alone is not enough — the built scene and the USD export regenerate in
seconds, so they are ignored; the Cycles stills and the walkthrough video regenerate
in GPU-hours, so they are kept. `.gitignore` implements this; the table is its
rationale, with the regeneration cost **measured on this machine** (single RTX-class
GPU) so the rule is auditable.

| Artifact | Size | Decision | Regen cost (measured) | Why |
|----------|------|----------|-----------------------|-----|
| `scenes/supermarket_base.blend` | 32 MB | **versioned** | n/a (network) | Holds the Poly Haven materials + HDRI, which came from the network; no script rebuilds it. |
| `assets/textures/` (Poly Haven PBR + HDRI) | 26 MB | **versioned** | n/a (network) | Downloaded from Poly Haven; externally linked by the base `.blend`. Not regenerable offline. |
| `report_assets/*.jpg` | 1.4 MB | **versioned** | n/a (curated) | Hand-curated JPEG stills that `gen_report.py` inlines. No committed command produces them — inputs, not outputs. |
| `scripts/**` (incl. `catalog.py`) | — | **versioned** | — | Source. |
| `scripts/planogram.json` | 0.4 MB | **versioned** | 0.03 s | Cheap, but kept anyway: the small, diffable, deterministic interface between the planogram and label/scene stages; lets a rebuild skip stage 2. |
| `renders/*.png` (8 Cycles stills) | ~ | **versioned** | **GPU-hours** — Cycles 200 samples, 1920×1080, ×8 cameras, OptiX (measured proxy: 34.6 s for one 640×360 EEVEE 16-sample still; full quality is orders of magnitude more and was not re-run) | Not cheap to reproduce, and the visible evidence of the project. `blender -b scenes/supermarket.blend --python scripts/render_shots.py -- --samples 200 --res 1920x1080` |
| `renders/walkthrough.mp4` | ~ | **versioned** | **GPU-hours** — 510 frames, EEVEE, 1280×720 (measured proxy: 23.2 s for 4 frames at 320×180 EEVEE 8-sample; full length is ~two orders of magnitude more, not re-run) | Same: expensive, and visible evidence. `blender -b scenes/supermarket.blend --python scripts/render_walkthrough.py --` |
| `report.html` | 1.7 MB | **versioned** | 0.03 s | The exception to the cost rule: it is cheap, but it is the human-facing deliverable that presents the render evidence (inlining the `report_assets` stills), so it is kept with `renders/` as the project's visible record. `python3 scripts/gen_report.py` |
| `scenes/supermarket.blend` (built scene) | 116 MB | **ignored** | **4.0 s** | Offline and cheap. `blender -b scenes/supermarket_base.blend --python scripts/build_supermarket.py -- --all --save scenes/supermarket.blend` |
| `assets/labels/`, `assets/signs/` | 5 MB | **ignored** | **6.7 s** | Deterministic PIL output, offline and cheap: `python3 scripts/gen_labels.py` from `catalog.py` + `planogram.json`. |
| `export/` (`supermarket.usdc` + `textures/`) | 21 MB | **ignored** | **26.3 s** | Offline and cheap, from the built scene: `blender -b scenes/supermarket.blend --python scripts/export_usd.py`. |
| `*.blend1`, `logs/`, `__pycache__/`, `*.pyc` | — | **ignored** | — | Autosaves, run logs and bytecode — transient by definition. |

Every cost above was measured by regenerating the artifact headless into a scratch
path (the built scene, USD export, labels, planogram and report were also compared
to the committed copy and matched — see the decision log). The two render classes
were exercised with reduced-cost proxies to prove the pipeline runs; their
full-quality cost is GPU-hours, which is precisely why they are versioned rather
than reproduced.

---

## Repository facts

- **Blender 4.5.13 LTS**, installed as a tarball at `~/opt/blender`
  (symlink `~/.local/bin/blender`). The LTS line is chosen for a stable Python API
  and the USD exporter the Isaac pipeline relies on.
- **blender-mcp bridge** (ahujasid), run on demand with `uvx blender-mcp` and
  registered in Claude Code at user scope; the Blender-side addon "MCP for Blender"
  auto-starts a socket on `localhost:9876`. It lets an agent build and inspect the
  scene in the *live* GUI, stage by stage. Setup and re-registration in RUNBOOK §2.
  Server telemetry has been switched off (`disable_telemetry`).
- **No git remote.** This is a **local-only** project; firstmate registered it as
  such on **2026-09-09**. There is nothing to push or pull — history lives on this
  machine.

---

## Decision log

Dated record of the choices baked into the repo. Newest first.

### 2026-09-09 — File policy refined: ignore only what is offline AND cheap
The repo was committed whole (b4bfa88) to stop untracked work being lost; this
decides which large artifacts stay. The condition for *ignoring* an artifact is
two-part: its regeneration must be **offline** and **cheap** (seconds to about a
minute). Reproducibility alone is not enough — the Cycles stills and the
walkthrough video *can* be regenerated, but only at GPU-hours, so they are kept
along with `report.html`, which presents them and is the project's visible
evidence. The genuinely cheap outputs are ignored: built scene (**4.0 s**),
labels/signs (**6.7 s**), `export/` (**26.3 s**). Versioned: base scene and
`assets/textures/` (network), `report_assets/` (curated), `planogram.json` (0.03 s
but the small stage interface), `renders/` and `walkthrough.mp4` (GPU-hours),
`report.html` (0.03 s but the deliverable), all sources.

Reproducibility was verified by regenerating each into a scratch path and comparing
to the committed copy: `planogram.json` and all 223 label/sign PNGs came back
**byte-identical**; a headless rebuild produced **26 006 objects / 23 418 product
instances** (committed scene: same); the USD export came back 4 781 652 vs
4 783 096 bytes with 225 textures each (0.03 % delta); a headless EEVEE still and a
4-frame `Cam_Walk` MP4 both rendered (reduced-cost proxies, since full quality is
GPU-hours); `report.html` regenerated identically apart from its embedded date and
runtime-probed sizes. All regeneration costs in the file-policy table were measured
on this machine. See that table.

### 2026-09-09 — `planogram.json` stays versioned though it is regenerable
It is the small, diffable, deterministic hand-off between the planogram stage and
the label/scene stages. Committing it lets a rebuild skip stage 2 unless the
layout tables changed, and makes a layout change reviewable as a JSON diff. The
ignore rule permits versioning a regenerable file; it only forbids *ignoring* a
non-regenerable one.

### 2026-09-06 — Base scene holds the network assets, externally linked
The Poly Haven materials and HDRI are the only things a rebuild cannot reproduce
offline, so they live in `scenes/supermarket_base.blend` (the file every build
opens) rather than being re-downloaded per build. They are pinned with a fake user
so a save never purges the unused datablocks, and the texture files are linked
from `assets/textures/` rather than packed — hence both the base scene and that
folder are versioned together.

### 2026-09-06 — Store generated from a catalog + planogram, not hand-modelled
Chosen so the store is editable, reproducible and labelled by construction, and so
products can share one prototype mesh each and export as true USD instances. The
inputs a human edits are `catalog.py` and the layout tables in
`gen_planogram.py`; the geometry is an output.

### 2026-09-06 — Products carry semantics through to USD
Each product and fixture is tagged at build time (sku, category, brand, price,
slot, facing, row / fixture, gondola, side, bay, level) so the USD export gives a
simulator planogram-compliance and semantic-segmentation ground truth for free.

### 2026-09-06 — USD instancing + Z-up metres for Isaac Sim
Export uses linked-duplicate → USD instancing and Z-up/metre units so the stage is
light and drops into Isaac Sim with colliders and physics added on the prototypes.
