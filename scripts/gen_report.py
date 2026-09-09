#!/usr/bin/env python3
"""Assemble report.html (images inlined as data URIs) from the current project state."""
import base64, json, os, sys, collections, datetime, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from catalog import SKUS, CATEGORIES
plan = json.load(open(os.path.join(HERE, "planogram.json")))

def img(name):
    p = os.path.join(ROOT, "report_assets", f"{name}.jpg")
    return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()

def runbook_blocks():
    """Pull the fenced code blocks (with their section titles) out of RUNBOOK.md."""
    out, title, buf, inblock = [], None, [], False
    for line in open(os.path.join(ROOT, "RUNBOOK.md")):
        if line.startswith("## "): title = line[3:].strip()
        elif line.startswith("```"):
            if inblock: out.append((title, "".join(buf))); buf = []
            inblock = not inblock
        elif inblock: buf.append(line)
    return out

forms = collections.Counter(s["form"] for s in SKUS)
cats = collections.Counter(s["cat"] for s in SKUS)
usd_size = os.path.getsize(os.path.join(ROOT, "export", "supermarket.usdc")) / 1e6 if os.path.exists(os.path.join(ROOT, "export", "supermarket.usdc")) else 0
mp4 = os.path.join(ROOT, "renders", "walkthrough.mp4")
mp4_ok = os.path.exists(mp4) and os.path.getsize(mp4) > 1e6
today = datetime.date.today().isoformat()

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

aisle_rows = "".join(f"<tr><td class='num'>{a['no']}</td><td>{esc(' · '.join(a['left']))}</td><td>{esc(' · '.join(a['right']))}</td><td class='mono'>x = {a['x']:+.2f} m</td></tr>" for a in plan["aisles"])
form_rows = "".join(f"<tr><td class='mono'>{f}</td><td>{n}</td><td>{d}</td></tr>" for f, n, d in [
    ("box", forms["box"], "cardboard box, label front and back, bevelled edges (cereal, tea, pasta, powder, eggs)"),
    ("bag", forms["bag"], "pillow-puffed pouch; foil finish for snacks, matte for flour/rice/coffee/bread"),
    ("can", forms["can"], "revolved tin with rims and a 360° wrap label (drinks, tomatoes, beans, pet food, deodorant)"),
    ("bottle", forms["bottle"], "revolved PET (tinted transmission) or glass; label band, coloured cap (soda, water, oils, wine, beer)"),
    ("bottle_hh", forms["bottle_hh"], "rounded household bottle with neck and cap (detergent, sprays, shampoo, ketchup)"),
    ("jar", forms["jar"], "clear glass with opaque contents inside, metal lid, label band (sauces, spreads, spices)"),
    ("carton", forms["carton"], "1 L brick with screw cap, label on all four sides (milk, juice)"),
    ("tub", forms["tub"], "tapered tub with over-sized lid (yogurt, cookie tin)"),
    ("flat", forms["flat"], "flat pack, label on top, stacked up to five high (chocolate bars, butter, cheese)"),
])
cat_rows = "".join(f"<span class='chip'>{esc(CATEGORIES[c])} <b>{n}</b></span>" for c, n in cats.items())

mcp_rows = "".join(f"<tr><td class='mono'>{t}</td><td><span class='ok'>{r}</span></td><td>{esc(n)}</td></tr>" for t, r, n in [
    ("list_tools", "28 tools", "server spawned with uvx, connected to the addon (protocol 5, addon 1.6, Blender 4.5.13)"),
    ("disable_telemetry", "OFF", "the server ships with prompt/code/screenshot collection on; switched off for this machine"),
    ("get_scene_info", "OK", "returns object list, counts, materials; needs a user_prompt argument in this release"),
    ("execute_blender_code", "OK", "ran the whole builder in 5 staged calls; each call is a fresh namespace, 180 s timeout per call"),
    ("get_viewport_screenshot", "OK", "PNG of the live viewport; used for every visual check in this session"),
    ("search_polyhaven_assets", "OK", "capped at 20 alphabetical results, so the Poly Haven API was queried directly to pick assets"),
    ("download_polyhaven_asset", "3 assets", "floor_tiles_08 and white_plaster_02 (2k PBR) as materials; wide_street_01 (2k HDRI) as the world"),
    ("set_texture / get_object_info", "not needed", "the builder assigns the imported materials itself and scales their Mapping node"),
    ("Sketchfab / Hyper3D / Hunyuan / Poly Pizza", "not exercised", "need API keys or generate meshes; candidates for real carts, people and props later"),
])

runbook_html = "".join(f"<h4>{esc(t or '')}</h4><pre><code>{esc(b)}</code></pre>" for t, b in runbook_blocks())

html = f"""<title>Freshmart Build</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100,800;110,900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#f6f4ee; --surface:#ffffff; --ink:#1b202b; --ink-2:#4b5262; --muted:#7a8191; --line:#dcd8cd;
  --navy:#14285a; --navy-ink:#ffffff; --tag:#f5b400; --tag-ink:#1b202b; --red:#c8321e; --green:#2b7a3d; --code-bg:#1d2130; --code-ink:#e6e4dc;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper:#15181e; --surface:#1d2129; --ink:#ebe8df; --ink-2:#c3c1b8; --muted:#8e94a3; --line:#2f343e;
  --navy:#1d3577; --navy-ink:#ffffff; --tag:#f5b400; --tag-ink:#1b202b; --red:#ee6a55; --green:#63b874; --code-bg:#0f1218; --code-ink:#e6e4dc;
}} }}
:root[data-theme="dark"] {{
  --paper:#15181e; --surface:#1d2129; --ink:#ebe8df; --ink-2:#c3c1b8; --muted:#8e94a3; --line:#2f343e;
  --navy:#1d3577; --navy-ink:#ffffff; --tag:#f5b400; --tag-ink:#1b202b; --red:#ee6a55; --green:#63b874; --code-bg:#0f1218; --code-ink:#e6e4dc;
}}
* {{ box-sizing:border-box }}
body {{ background:var(--paper); color:var(--ink); font-family:"IBM Plex Sans", "Segoe UI", system-ui, sans-serif; font-size:16px; line-height:1.55; margin:0; }}
main {{ max-width:1000px; margin:0 auto; padding:40px 24px 80px; }}
h1,h2,h3 {{ font-family:"Archivo", "Arial Narrow", Arial, sans-serif; font-variation-settings:"wdth" 110; text-wrap:balance; margin:0; }}
h1 {{ font-size:clamp(40px, 7vw, 68px); font-weight:900; line-height:0.95; letter-spacing:-0.01em; }}
h3 {{ font-size:19px; font-weight:800; margin:26px 0 8px; }}
h4 {{ font-family:"IBM Plex Sans"; font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:22px 0 6px; font-weight:600; }}
p {{ max-width:70ch; margin:10px 0; }}
a {{ color:var(--navy); }}
:root[data-theme="dark"] a {{ color:#9db4ff }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) a {{ color:#9db4ff }} }}
.eyebrow {{ font-family:"IBM Plex Mono", monospace; font-size:13px; color:var(--muted); letter-spacing:.04em; }}
header {{ display:grid; gap:14px; padding-bottom:26px; border-bottom:2px solid var(--line); }}
header .lede {{ font-size:18px; color:var(--ink-2); max-width:62ch; }}
.status {{ display:flex; flex-wrap:wrap; gap:8px; }}
.pill {{ font-size:13px; font-weight:600; padding:4px 10px; border-radius:999px; border:1px solid var(--line); background:var(--surface); }}
.pill.ok {{ border-color:var(--green); color:var(--green); }}
.pill.wait {{ border-color:var(--tag); color:var(--ink-2); }}
/* ticket rail: the shelf-edge price strip */
.rail {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:10px; margin:26px 0 8px; padding:12px; background:var(--surface); border:1px solid var(--line); border-radius:6px; }}
.tag {{ background:#fff; color:#1b202b; border-radius:3px; padding:8px 10px; box-shadow:0 1px 0 rgba(0,0,0,.15); border-top:4px solid var(--navy); }}
.tag .v {{ font-family:"Archivo"; font-variation-settings:"wdth" 100; font-weight:800; font-size:26px; line-height:1.05; font-variant-numeric:tabular-nums; }}
.tag .k {{ font-size:12px; color:#5a6070; margin-top:3px; }}
/* section header styled as the hanging aisle sign */
section {{ margin-top:48px; }}
.sign {{ display:grid; grid-template-columns:auto 1fr; align-items:stretch; margin-bottom:18px; border-radius:4px; overflow:hidden; box-shadow:0 2px 0 rgba(0,0,0,.18); }}
.sign .n {{ background:var(--tag); color:var(--tag-ink); font-family:"Archivo"; font-weight:900; font-variation-settings:"wdth" 100; font-size:15px; letter-spacing:.06em; padding:12px 14px; display:flex; align-items:center; }}
.sign h2 {{ background:var(--navy); color:var(--navy-ink); font-size:24px; font-weight:800; padding:10px 18px; }}
figure {{ margin:0; }}
figure img {{ width:100%; height:auto; display:block; border-radius:4px; border:1px solid var(--line); }}
figcaption {{ font-size:13px; color:var(--muted); margin-top:6px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:16px; }}
.hero {{ margin-top:26px; }}
table {{ width:100%; border-collapse:collapse; font-size:14.5px; background:var(--surface); }}
.tbl {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:600; }}
tr:last-child td {{ border-bottom:none }}
td.num {{ font-family:"Archivo"; font-weight:800; font-size:18px; color:var(--navy); width:52px; text-align:center; }}
:root[data-theme="dark"] td.num {{ color:#9db4ff }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) td.num {{ color:#9db4ff }} }}
.mono {{ font-family:"IBM Plex Mono", monospace; font-size:13.5px; }}
.ok {{ color:var(--green); font-weight:600; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }}
.chip {{ font-size:13px; padding:3px 9px; border-radius:3px; background:var(--surface); border:1px solid var(--line); }}
.chip b {{ color:var(--red); }}
pre {{ background:var(--code-bg); color:var(--code-ink); padding:12px 14px; border-radius:6px; overflow-x:auto; font-size:13px; line-height:1.5; margin:4px 0 14px; }}
code {{ font-family:"IBM Plex Mono", monospace; }}
p code, li code {{ background:var(--surface); border:1px solid var(--line); padding:0 5px; border-radius:3px; font-size:13.5px; }}
ul {{ padding-left:20px; max-width:75ch; }} li {{ margin:4px 0; }}
kbd {{ font-family:"IBM Plex Mono"; font-size:12.5px; background:var(--surface); border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:1px 6px; }}
.flow {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:12px 0 18px; }}
.flow .step {{ background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--navy); border-radius:4px; padding:8px 12px; font-size:14px; }}
.flow .step small {{ display:block; color:var(--muted); font-family:"IBM Plex Mono"; font-size:12px; }}
.flow .arrow {{ color:var(--muted); font-size:18px; }}
.note {{ border-left:4px solid var(--tag); background:var(--surface); padding:10px 14px; border-radius:0 6px 6px 0; margin:14px 0; max-width:75ch; }}
@media (prefers-reduced-motion: no-preference) {{ .sign {{ transition:transform .2s }} }}
</style>
<main>
<header>
  <div class="eyebrow">~/workspaces/blender-supermarket · {today} · Blender 4.5.13 LTS · blender-mcp addon 1.6</div>
  <h1>Freshmart Build</h1>
  <div class="lede">A railway-station supermarket you can walk through in Blender: 26 × 22 m of floor, seven gondola runs on a real planogram, {len(SKUS)} fictional-brand SKUs with printed labels and price tags, chillers, checkouts, produce and signage. Built end to end through the Blender MCP server, exported to USD for Isaac Sim.</div>
  <div class="status">
    <span class="pill ok">MCP server + addon installed</span>
    <span class="pill ok">Scene built through MCP</span>
    <span class="pill ok">8 stills rendered (Cycles / OptiX)</span>
    <span class="pill ok">USD export verified ({usd_size:.1f} MB, instanced)</span>
    <span class="pill {'ok' if mp4_ok else 'wait'}">Walkthrough video {'rendered' if mp4_ok else 'rendering in the background'}</span>
  </div>
</header>

<div class="rail">
  <div class="tag"><div class="v">26 × 22 m</div><div class="k">shop floor, 4.0 m ceiling</div></div>
  <div class="tag"><div class="v">{len(SKUS)}</div><div class="k">SKUs, 9 packaging forms</div></div>
  <div class="tag"><div class="v">23 418</div><div class="k">products on shelves</div></div>
  <div class="tag"><div class="v">7 + 8</div><div class="k">gondola runs, aisles</div></div>
  <div class="tag"><div class="v">20</div><div class="k">chiller bays, 4 checkouts</div></div>
  <div class="tag"><div class="v">3.4 s</div><div class="k">full rebuild, headless</div></div>
</div>

<figure class="hero"><img src="{img('entrance')}" alt="Render from the entrance: produce tables, aisle signs, checkouts and chillers"><figcaption>Cam_Entrance, Cycles 200 samples with OptiX denoising on the RTX 3060. The same view is live in the EEVEE viewport below.</figcaption></figure>

<section>
  <div class="sign"><div class="n">SETUP</div><h2>What was installed</h2></div>
  <div class="tbl"><table>
    <tr><th>Component</th><th>Where</th><th>Notes</th></tr>
    <tr><td>Blender 4.5.13 LTS</td><td class="mono">~/opt/blender → ~/.local/bin/blender</td><td>tarball install, no sudo needed on this box; LTS line chosen for the stable Python API and USD exporter Isaac pipelines rely on. Python 3.11 with numpy and the <code>pxr</code> USD bindings inside.</td></tr>
    <tr><td>uv 0.12.10</td><td class="mono">~/.local/bin/uv, uvx</td><td>runs the MCP server on demand: <code>uvx blender-mcp</code></td></tr>
    <tr><td>blender-mcp (ahujasid)</td><td class="mono">~/.claude.json (user scope)</td><td><code>claude mcp add -s user blender -- uvx blender-mcp</code>. The tools appear in Claude Code sessions started after this one.</td></tr>
    <tr><td>Blender addon “MCP for Blender” 1.6</td><td class="mono">~/.config/blender/4.5/scripts/addons/blender_mcp.py</td><td>installed with <code>uvx blender-mcp install-addon</code> (needed <code>BLENDERMCP_ADDONS_DIR</code> because Blender lives in a non-standard path), enabled and saved in the user preferences. It auto-starts its socket on <code>localhost:9876</code> when Blender opens, so no button clicks are required.</td></tr>
    <tr><td>Blender GUI</td><td class="mono">tmux session “blender” on DISPLAY=:0</td><td><code>scripts/launch_blender.sh</code> restarts it with the scene loaded. The KDE session on the HDMI monitor shows the live viewport.</td></tr>
    <tr><td>Telemetry</td><td>—</td><td>the MCP server collects prompts, code and screenshots by default. Switched <b>off</b> with its <code>disable_telemetry</code> tool; anonymous tool-name counts remain.</td></tr>
  </table></div>
</section>

<section>
  <div class="sign"><div class="n">MCP</div><h2>Capability test results</h2></div>
  <p>Because a running Claude session cannot hot-load a newly added MCP server, a small stdio client (<code>scripts/mcpcall.py</code>) spawns the identical <code>uvx blender-mcp</code> process and calls its tools. Everything below went through that path, i.e. the same code your next Claude Code session will use.</p>
  <div class="tbl"><table>
    <tr><th>Tool</th><th>Result</th><th>What was learned</th></tr>{mcp_rows}
  </table></div>
  <h3>Quirks worth knowing</h3>
  <ul>
    <li><b>Per-call namespace.</b> Variables and functions do not survive between <code>execute_blender_code</code> calls. Import your script as a module (<code>sys.path.insert</code> + <code>import build_supermarket</code>) so state persists; <code>exec(open(...))</code> loses it.</li>
    <li><b>180 s per command.</b> The addon and server both time out at 180 s, so long builds are split into stages. The whole store takes about 4 s, so this never bit.</li>
    <li><b>Poly Haven temp files vanish.</b> The addon loads textures from /tmp and deletes the files. Pixel data stays in memory; the builder saved it into <code>assets/textures/</code> and repointed the images.</li>
    <li><b>Zero-user datablocks are purged on save.</b> The imported floor and wall materials had no users and disappeared from the first saved file. They are pinned with a fake user in <code>scenes/supermarket_base.blend</code> now.</li>
    <li><b>Normal maps unconnected.</b> The importer links Diffuse and Roughness only; the builder wires <code>nor_gl</code> through a Normal Map node.</li>
  </ul>
</section>

<section>
  <div class="sign"><div class="n">STORE</div><h2>Layout and planogram</h2></div>
  <div class="grid2">
    <figure><img src="{img('topdown')}" alt="Top-down cutaway of the store"><figcaption>Cam_TopDown with ceiling and lights hidden. Entrance and exit are at the bottom (front wall, glazed), checkouts front-centre, produce front-right, wine wall left, glass-door dairy chillers right, open multideck chillers and bakery along the back.</figcaption></figure>
    <div>
      <div class="tbl"><table>
        <tr><th>Aisle</th><th>Left face</th><th>Right face</th><th>Centre</th></tr>{aisle_rows}
      </table></div>
      <p style="font-size:14px;color:var(--muted)">Gondola runs are 6 bays × 1.25 m along Y, pitch 3.05 m (1.9 m clear aisle). Every run has promotional end caps front and back.</p>
    </div>
  </div>
  <h3>Fixtures, to real dimensions</h3>
  <ul>
    <li>Double-sided gondola: 2.0 m tall, 0.57 m base deck at 0.17 m, upper shelves 0.47 m deep and 30 mm thick with a 42 mm ticket rail, uprights every 1.25 m, pegboard back panel, kick plate and top cap.</li>
    <li>Shelf heights are derived per category from the tallest product on each level (0.28 m minimum pitch): cereal gets four shelves, canned goods and personal care six.</li>
    <li>Wall units (single-sided, 2.2 m): beer &amp; wine on the left wall, bakery on the back wall. Reach-in chillers with glass doors, handles and LED strips (dairy); open multidecks with canopies (drinks, dairy).</li>
    <li>Front of store: four checkout lanes (belt, scanner glass, POS screen, card terminal, lane light with number sign) with impulse shelves, entry gates, basket stack, nested wire-mesh carts, chip dump bin, promo pallets stacked with shrink-wrapped cases.</li>
  </ul>
  <h3>Products</h3>
  <div class="chips">{cat_rows}</div>
  <div class="tbl"><table><tr><th>Form</th><th>SKUs</th><th>Geometry</th></tr>{form_rows}</table></div>
  <figure style="margin-top:14px"><img src="{img('labels')}" alt="Sample of generated labels"><figcaption>Labels, wrap labels and price tags are rendered with PIL from <code>catalog.py</code> (brand, name, flavour, weight, nutrition table, barcode) so every SKU is distinct and readable at shelf distance. Brands are fictional.</figcaption></figure>
  <p>Stocking is deliberately imperfect: 2–10 facings per block, up to three rows deep, some front rows sold out, the odd empty column, ±2° jitter on boxes and free rotation on cans, and occasionally a knocked-over product.</p>
</section>

<section>
  <div class="sign"><div class="n">LOOK</div><h2>Renders</h2></div>
  <div class="grid2">
    <figure><img src="{img('aisle')}" alt="Aisle 4 render"><figcaption>Cam_Aisle: aisle 4 between water &amp; juice and canned goods, floor tiles and plaster from Poly Haven.</figcaption></figure>
    <figure><img src="{img('shelf')}" alt="Shelf close-up render"><figcaption>Cam_Shelf: soft drinks bay, 1.4 m from the ticket rail. Wrap labels, rims, price tags.</figcaption></figure>
    <figure><img src="{img('checkout')}" alt="Checkouts render"><figcaption>Cam_Checkout: lanes 1–3, impulse shelves, the street HDRI through the shopfront glazing.</figcaption></figure>
    <figure><img src="{img('produce')}" alt="Produce render"><figcaption>Cam_Produce: produce tables and crates in front of the dairy chillers.</figcaption></figure>
    <figure><img src="{img('overview')}" alt="Overview render"><figcaption>Cam_Overview from the front-left corner at 3.7 m: wine wall, aisle signs, pallets, checkouts, exit.</figcaption></figure>
    <figure><img src="{img('viewport')}" alt="Live EEVEE viewport screenshot via MCP"><figcaption>The live viewport in the open Blender window (EEVEE, captured with the MCP screenshot tool). This is what you roam in.</figcaption></figure>
  </div>
</section>

<section>
  <div class="sign"><div class="n">ROAM</div><h2>Walking around in Blender</h2></div>
  <p>The scene is open in the Blender window on the desktop, viewport in Material Preview with scene lights and world. Hover the 3D view and press <kbd>Shift</kbd>+<kbd>`</kbd> for walk navigation: <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to move, mouse to look, <kbd>Tab</kbd> toggles gravity (eye height preset to 1.65 m, speed 2.5 m/s), <kbd>Shift</kbd> to run, <kbd>Esc</kbd> to stop. <kbd>Numpad 0</kbd> looks through the active camera; the fly-through <code>Cam_Walk</code> runs 510 frames from the entrance through aisles 4 and 6 to checkout lane 2 and out the exit (press <kbd>Space</kbd>).</p>
  <p>Collections mirror the store: <code>Structure</code>, <code>Fixtures/*</code>, <code>Products/*</code> and <code>PriceTags/*</code> per gondola or unit, <code>Signage</code>, <code>Lighting</code>, <code>Cameras</code>. Hide <code>PriceTags</code> or <code>Lighting</code> if the viewport gets slow. Products are collection instances of the prototypes in the excluded <code>Prototypes</code> collection: edit a prototype mesh or material and every instance of that SKU updates.</p>
</section>

<section>
  <div class="sign"><div class="n">ISAAC</div><h2>Path to Isaac Sim</h2></div>
  <div class="tbl"><table>
    <tr><th>Export fact</th><th>Value</th></tr>
    <tr><td>File</td><td class="mono">export/supermarket.usdc ({usd_size:.1f} MB) + export/textures/</td></tr>
    <tr><td>Stage</td><td>Z-up, 1 m per unit, root prim <code>/Supermarket</code>, 63 rect lights, 8 cameras, UsdPreviewSurface materials with textures</td></tr>
    <tr><td>Hierarchy</td><td class="mono">/Supermarket/GRP_Products/GRP_Products_G3/cola_can_G3L_b4_s1_f1_r1 … GRP_Fixtures/…, GRP_Structure, GRP_Signage, GRP_Lighting</td></tr>
    <tr><td>Instancing</td><td>24 927 instanceable prims over 196 prototypes; only 766 real meshes on the stage (verified with pxr inside Blender)</td></tr>
    <tr><td>Semantics</td><td>every product prim carries <code>userProperties:sku</code>, <code>category</code>, <code>brand</code>, <code>price</code>, <code>slot</code> (gondola/side/bay/shelf), <code>facing</code>, <code>row</code>; fixtures carry <code>fixture</code>, <code>gondola</code>, <code>side</code>, <code>bay</code>, <code>level</code></td></tr>
  </table></div>
  <h3>Next steps for simulation</h3>
  <ul>
    <li>Reference <code>supermarket.usdc</code> in a stage, then add colliders: convex hull on <code>GRP_Products</code>, triangle mesh on <code>GRP_Fixtures</code> and <code>GRP_Structure</code>. Colliders on instanceable prims are applied on the prototype, which is what you want for 25k products.</li>
    <li>Map <code>userProperties:sku</code> and <code>category</code> to semantic labels for Replicator ground truth; the slot ids give planogram compliance labels for free (expected SKU per shelf position).</li>
    <li>Products are static Xforms; make the ones the shelfbot handles rigid bodies (mass from the catalog weight) and leave the rest static.</li>
    <li>PET bottles use transmission; if RTX real-time looks noisy, swap their prototype material for an opaque tinted plastic in Blender and re-export.</li>
    <li>The shopfront HDRI is not exported (world material conversion off); use a dome light in Isaac.</li>
  </ul>
</section>

<section>
  <div class="sign"><div class="n">PIPELINE</div><h2>How the store is generated</h2></div>
  <div class="flow">
    <div class="step">catalog.py<small>{len(SKUS)} SKUs, dims, colours, prices</small></div><span class="arrow">→</span>
    <div class="step">gen_planogram.py<small>layout + shelves + facings → planogram.json</small></div><span class="arrow">→</span>
    <div class="step">gen_labels.py<small>PIL labels, tags, signs</small></div><span class="arrow">→</span>
    <div class="step">build_supermarket.py<small>13 stages, headless or via MCP</small></div><span class="arrow">→</span>
    <div class="step">supermarket.blend<small>25.6k objects, 196 prototypes</small></div><span class="arrow">→</span>
    <div class="step">render_shots.py · render_walkthrough.py · export_usd.py<small>stills, MP4, USD</small></div>
  </div>
  <p>Everything is regenerable: change a product in the catalog, a category assignment in the planogram tables, or a fixture dimension, and rebuild in a few seconds. The base file <code>scenes/supermarket_base.blend</code> only holds the Poly Haven materials and HDRI.</p>
  <div class="note">Limits of this first version: products are procedural stand-ins (no scanned real-brand geometry), there are no people or clutter on the floor, produce is spheres in crates, and the wire carts and baskets are simple. The MCP’s Sketchfab, Poly Pizza and Hyper3D tools are the obvious route to real carts, trolleys and scanned products when you want them.</div>
</section>

<section>
  <div class="sign"><div class="n">RUNBOOK</div><h2>Recurring commands</h2></div>
  <p>Same content as <code>RUNBOOK.md</code> in the project folder.</p>
  {runbook_html}
</section>
</main>
"""
out = os.path.join(ROOT, "report.html")
open(out, "w").write(html)
print("wrote", out, round(os.path.getsize(out) / 1e6, 2), "MB", "walkthrough:", mp4_ok)
