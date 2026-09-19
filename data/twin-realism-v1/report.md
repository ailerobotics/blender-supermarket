# Supermarket twin — realism pass v1

**2026-09-14 · branch `fm/twin-realism-v1` · the generated store's look**

Goal: the nine camera stills and the walkthrough should read as a real supermarket
at a glance, so the twin is both a believable render set and a plausible visual
domain for the shelf robot's perception. Every change lands in the **generator**
(`scripts/build_supermarket.py`, materials / lighting / placement), never a
hand-edited `.blend`, so a headless rebuild reproduces it.

A phone-readable, self-contained version with all crops inlined is beside this file:
**[report.html](report.html)**.

## Render path (why CPU)

The RTX 3060 (12 GB) was held by the live `ur10-cumotion` Isaac/ROS stack (~5.8 GB;
`pgrep -af 'isaac|omni.kit|kit/kit'` matched the running cumotion container), so per
the task GPU rules the evidence renders fell back to **CPU Cycles, 64 samples,
1280×720, OpenImageDenoise**. No process was touched. Before and after use identical
engine, samples and resolution. `render_shots.py --cpu` was added for exactly this.

## 1 · Gap audit (from current `main`)

Three representative cameras — `Cam_Entrance` (overview), `Cam_Aisle` (shopper
corridor), `Cam_Shelf` (robot working range) — rendered at evidence quality. Gaps
ranked by visual impact (crops in `report.html` / `before/`):

| # | Gap | Where | Why it reads as CG |
|---|-----|-------|--------------------|
| 1 | **Flat, matte floor — no reflections, no wear** | all three | Dry residential terracotta tile with zero specular response; a real store floor is buffed vinyl that mirrors the LEDs and has scuffed traffic lanes. The single strongest tell. |
| 2 | **Cloned products** | Aisle, Shelf | Every facing of a SKU is an identical copy at an identical angle and tone, so a shelf reads as one object stamped N times. |
| 3 | **Uniform plastic produce** | Entrance | Fruit is a grid of identical spheres — one size, one colour, one gloss. |
| 4 | **Clipped-flat lighting** | all three | LED panels clip to flat white with no bloom; highlights look computed, and nothing on the floor picks the lights back up. |

Everything already present and *not* a gap (the base is strong): shelf-edge price
rails with printed tags, powder-coated steel fixtures, end caps, hanging aisle
signage, a glazed shopfront, ceiling tiles + recessed LED fixtures, AgX colour
management, per-form product materials (cardboard, PET transmission, metal cans,
glass jars) with bevels and printed wrap labels.

## 2 · Before / after

Same three cameras, same engine and samples. Evidence JPEGs in `before/` and `after/`:

| Camera | Before | After |
|--------|--------|-------|
| Cam_Entrance | `before/Cam_Entrance.jpg` | `after/Cam_Entrance.jpg` |
| Cam_Aisle | `before/Cam_Aisle.jpg` | `after/Cam_Aisle.jpg` |
| Cam_Shelf | `before/Cam_Shelf.jpg` | `after/Cam_Shelf.jpg` |

The floor now softly reflects the shelf uprights and LEDs with a subtle grime/wear
variation (polished vinyl, not a mirror); produce varies in size and shade; shelves
of one SKU show per-instance tonal + angle spread; the LED fixtures bloom softly.

## 3 · What changed, per commit

1. **Floor: polished commercial vinyl (reflections + traffic wear)** — `polish_floor()`
   rebuilds the imported Poly Haven tile material: a low-roughness clear coat for soft
   streaky reflections of the LEDs, a tightened base roughness so they read, and a
   large-scale procedural traffic-wear/grime layer so the floor is scuffed and
   non-uniform. *Fixes gap 1.*
2. **Products & produce: per-instance variation to kill the cloned look** —
   `add_instance_variation()` gives a shared material a per-instance brightness/
   saturation wobble (Object Info › Random; brand hue preserved) on every branded
   label, the fruit and the bread; `place_products()` gets wider yaw jitter, an
   occasional lean and per-facing depth jitter; produce/bakery get per-instance size
   grading. *Fixes gaps 2 and 3.*
3. **Lighting: photographic fog-glow bloom + exposure trim** — a compositor fog-glow
   glare so the LED fixtures bloom softly instead of clipping flat, and exposure
   −0.3 → −0.35 for highlight headroom now that the floor reflects the lights.
   *Fixes gap 4.*
4. **Make the generator path-portable + add a CPU render path** (enabler) — `ROOT`
   derives from each script's location so a rebuild is self-contained in any checkout;
   `render_shots.py --cpu` (CPU Cycles + OpenImageDenoise) for evidence renders while
   the GPU is held.

### Render cost (per camera, CPU Cycles 64 spp / 1280×720)

| Camera | Before (s) | After (s) |
|--------|-----------:|----------:|
| Cam_Entrance | 131.5 | 142.4 |
| Cam_Aisle | 149.4 | 166.3 |
| Cam_Shelf | 153.4 | 150.3 |

After is marginally slower (added clear coat, per-instance variation, compositor
fog-glow at the same sample count). Geometry is essentially unchanged (26,006 →
25,963 objects; the delta is the new RNG draws reshuffling the empty/missing-facing
pattern). Headless build stays ~4 s.

## 4 · USD export & pre-flight

Re-exported the realism-pass scene to USD (`export_usd.py`, 26 s) and ran the offline
`scripts/preflight.py` (no GPU): **STORE PASS · ROBOT PASS · COMPOSER PASS → VERDICT
PASS (exit 0)**.

The new floor/label realism is **procedural** (clear coat, noise grime, per-instance
Object-Info variation), which UsdPreviewSurface does not serialize — so the export
carries the same base textures as before and introduces **no new dangling
references**. The 6 floor/wall Poly Haven maps remain the documented **COSMETIC**
standing defect (decision log 2026-09-10), auto-repaired by the pre-flight from the
versioned `assets/textures/`. Standalone `validate_usd.py` on the raw export reports
the same 6 as COSMETIC (stage loads, geometry intact, robot stands).

## Reproduce

```bash
python3 scripts/gen_labels.py                         # labels/signs (gitignored)
blender -b scenes/supermarket_base.blend --python scripts/build_supermarket.py -- --all --save /tmp/after.blend
blender -b /tmp/after.blend --python scripts/render_shots.py -- --cpu --samples 64 --res 1280x720 --cams Cam_Entrance,Cam_Aisle,Cam_Shelf --out /tmp/shots
blender -b /tmp/after.blend --python scripts/export_usd.py    # -> export/ (gitignored)
.venv-usd/bin/python scripts/preflight.py                     # PASS, exit 0
```
