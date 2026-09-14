#!/usr/bin/env python3
"""Assemble the realism-pass v1 report (report.html + report.md), self-contained.

Reads the before/after evidence JPEGs in this directory, makes JPEG (downscaled)
data-URIs and audit crops, and writes a phone-readable report.html plus report.md.
Run after the before/ and after/ sets exist. Needs PIL only (no Blender/GPU).
"""
import base64, io, os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BEFORE, AFTER = os.path.join(HERE, "before"), os.path.join(HERE, "after")
CAMS = ["Cam_Entrance", "Cam_Aisle", "Cam_Shelf"]
CAM_LABEL = {"Cam_Entrance": "Cam_Entrance — establishing overview",
             "Cam_Aisle": "Cam_Aisle — shopper corridor (1.6 m)",
             "Cam_Shelf": "Cam_Shelf — robot working range (1.4 m)"}


def datauri(img, quality=82, maxw=1100):
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width > maxw:
        img = img.resize((maxw, round(img.height * maxw / img.width)), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def full(path):
    return datauri(Image.open(path))


def crop(path, box, maxw=760):
    """box = (l,t,r,b) as fractions of the image."""
    im = Image.open(path); W, H = im.size
    c = im.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
    return datauri(c, maxw=maxw)


# ---- gap audit crops (from the BEFORE renders) --------------------------------
GAPS = [
    ("1", "Flat, matte floor — no reflections, no wear",
     "The floor is a dry, residential-looking terracotta tile with zero specular "
     "response. A real supermarket floor is buffed vinyl/porcelain that softly "
     "mirrors the ceiling LEDs and the products, with scuffed traffic lanes. This "
     "flatness is the single strongest 'this is CG' cue in every aisle shot.",
     ("Cam_Aisle", (0.28, 0.62, 0.72, 1.0))),
    ("2", "Cloned products — one SKU stamped N times",
     "Every facing of a SKU is an identical copy at an identical angle and tone, so "
     "a shelf reads as one object repeated rather than many real packs. Real shelves "
     "have the same SKU too, but with print/lighting spread and hand-placed angles.",
     ("Cam_Shelf", (0.0, 0.15, 0.55, 0.6))),
    ("3", "Uniform plastic produce",
     "The fruit is a grid of identical spheres — one size, one colour, one gloss — "
     "which reads as moulded plastic. Graded produce varies in size and shade.",
     ("Cam_Entrance", (0.02, 0.72, 0.5, 1.0))),
    ("4", "Clipped-flat lighting",
     "The LED panels clip to flat white with no bloom, so highlights look computed "
     "rather than photographed; nothing on the floor picks the lights back up.",
     ("Cam_Aisle", (0.30, 0.0, 0.80, 0.22))),
]

COMMITS = [
    ("Floor: polished commercial vinyl (reflections + traffic wear)",
     "Rebuilds the imported Poly Haven tile material: a low-roughness clear coat for "
     "soft streaky reflections of the LEDs, a tightened base roughness so they read, "
     "and a large-scale procedural traffic-wear / grime layer so the floor is scuffed "
     "and non-uniform. Fixes gap 1."),
    ("Products & produce: per-instance variation to kill the cloned look",
     "A per-instance brightness/saturation wobble (Object Info &gt; Random, brand hue "
     "preserved) on every branded label, the fruit and the bread; wider yaw jitter, an "
     "occasional lean and per-facing depth jitter; and per-instance size grading for "
     "produce. Fixes gaps 2 and 3."),
    ("Lighting: photographic fog-glow bloom + exposure trim",
     "A compositor fog-glow glare so the LED fixtures bloom softly instead of clipping "
     "flat, and exposure -0.3 -> -0.35 for highlight headroom now that the floor "
     "reflects the lights. Fixes gap 4."),
    ("Make the generator path-portable + add a CPU render path (enabler)",
     "ROOT derives from each script's location so a rebuild is self-contained in any "
     "checkout; render_shots.py gains --cpu (CPU Cycles + OpenImageDenoise) for evidence "
     "renders while the GPU is held by the live robot stack."),
]

# filled in by the caller via env-like globals below
RENDER_NOTE = os.environ.get("RENDER_NOTE", "")
COST_ROWS = os.environ.get("COST_ROWS", "")
VALIDATE_NOTE = os.environ.get("VALIDATE_NOTE", "")

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1a1a1a;--mut:#666;--line:#e2e2df;--accent:#14285a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:20px 16px 64px}
h1{font-size:1.7rem;margin:.2em 0}h2{font-size:1.25rem;margin:1.8em 0 .5em;border-bottom:2px solid var(--accent);padding-bottom:.2em}
h3{font-size:1.02rem;margin:1.2em 0 .3em}
.sub{color:var(--mut);margin:.2em 0 1.2em}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
img{width:100%;height:auto;border-radius:6px;display:block}
.ba{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.ba figcaption{font-size:.8rem;color:var(--mut);text-align:center;margin-top:4px}
@media(max-width:640px){.ba{grid-template-columns:1fr}}
.tag{display:inline-block;background:var(--accent);color:#fff;border-radius:5px;font-size:.72rem;
padding:2px 8px;font-weight:600;letter-spacing:.02em;vertical-align:middle}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.5em 0}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#f0f0ee}
code,.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85em}
.gapn{display:inline-block;width:1.5em;height:1.5em;line-height:1.5em;text-align:center;
background:#c0392b;color:#fff;border-radius:50%;font-weight:700;font-size:.8rem;margin-right:6px}
.ok{color:#1a7d3c;font-weight:600}
"""


def ba_block(cam):
    b = full(os.path.join(BEFORE, f"{cam}.jpg"))
    a = full(os.path.join(AFTER, f"{cam}.jpg"))
    return (f'<h3>{CAM_LABEL[cam]}</h3><div class="ba">'
            f'<figure><img src="{b}"><figcaption>BEFORE (current main)</figcaption></figure>'
            f'<figure><img src="{a}"><figcaption>AFTER (realism pass)</figcaption></figure></div>')


def gap_block(n, title, desc, crop_spec):
    cam, box = crop_spec
    c = crop(os.path.join(BEFORE, f"{cam}.jpg"), box)
    return (f'<div class="card"><h3><span class="gapn">{n}</span>{title}</h3>'
            f'<p>{desc}</p><img src="{c}">'
            f'<figcaption style="font-size:.78rem;color:var(--mut);margin-top:4px">'
            f'crop from BEFORE / {cam}</figcaption></div>')


def build_html():
    gaps = "".join(gap_block(*g) for g in GAPS)
    ba = "".join(ba_block(c) for c in CAMS)
    commits = "".join(f"<tr><td><b>{t}</b><br><span style='color:var(--mut)'>{d}</span></td></tr>"
                      for t, d in COMMITS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supermarket twin — realism pass v1</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Supermarket twin — realism pass v1</h1>
<p class="sub">2026-09-14 · generated store look · branch <code>fm/twin-realism-v1</code></p>
<div class="card"><p>Goal: the nine camera stills and the walkthrough should read as a real
supermarket at a glance, so the twin is both a believable render set and a plausible
visual domain for the shelf robot's perception. Every change lands in the
<b>generator</b> (materials, lighting, placement), never a hand-edited <code>.blend</code>,
so a headless rebuild reproduces it.</p>
<p>{RENDER_NOTE}</p></div>

<h2>1 · Gap audit</h2>
<p>Three representative cameras rendered from the current <code>main</code> at evidence
quality, gaps ranked by visual impact with a crop each.</p>
{gaps}

<h2>2 · Before / after</h2>
<p>Same three cameras, same engine and samples, current <code>main</code> vs. the realism pass.</p>
{ba}

<h2>3 · What changed, per commit</h2>
<table><tbody>{commits}</tbody></table>
<h3>Render cost (per camera, evidence quality)</h3>
{COST_ROWS}

<h2>4 · USD export &amp; pre-flight</h2>
<div class="card">{VALIDATE_NOTE}</div>

<p class="sub" style="margin-top:2em">Full write-up alongside this file: <code>report.md</code>.
Evidence PNGs: <code>data/twin-realism-v1/before</code>, <code>.../after</code>.</p>
</div></body></html>"""


if __name__ == "__main__":
    html = build_html()
    open(os.path.join(HERE, "report.html"), "w").write(html)
    print("wrote report.html", os.path.getsize(os.path.join(HERE, "report.html")) // 1024, "KB")
