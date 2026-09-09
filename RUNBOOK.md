# Blender Supermarket — RUNBOOK

Project: `~/workspaces/blender-supermarket` · Blender 4.5.13 LTS at `~/opt/blender` (symlink `~/.local/bin/blender`) · MCP: `blender-mcp` (ahujasid) via `uvx`.

## 1. Start / stop Blender with the scene (MCP socket auto-starts on :9876)
```bash
~/workspaces/blender-supermarket/scripts/launch_blender.sh              # GUI on DISPLAY=:0, tmux session "blender"
~/workspaces/blender-supermarket/scripts/launch_blender.sh scenes/supermarket_base.blend   # empty base (only textures + HDRI)
tmux attach -t blender      # console output;  Ctrl-b d  to detach
tmux kill-session -t blender
ss -ltnp | grep 9876        # is the addon socket up?
```

## 2. Claude Code ↔ Blender MCP
```bash
claude mcp list                                   # "blender: uvx blender-mcp" must be listed (user scope, ~/.claude.json)
claude mcp add -s user blender -- uvx blender-mcp # re-register if missing
BLENDERMCP_ADDONS_DIR=~/.config/blender/4.5/scripts/addons uvx blender-mcp install-addon   # re-install/upgrade the addon
```
Inside a Claude Code session the tools are `mcp__blender__get_scene_info`, `execute_blender_code`, `get_viewport_screenshot`, `download_polyhaven_asset`, … (28 tools). They only load at session start, so restart Claude after `claude mcp add`.

Stand-alone MCP test client (spawns the same `uvx blender-mcp` server over stdio):
```bash
cd ~/workspaces/blender-supermarket
uv run --with mcp scripts/mcpcall.py --schema                                    # list tools + args
uv run --with mcp scripts/mcpcall.py get_scene_info
uv run --with mcp scripts/mcpcall.py execute_blender_code '{"code":"import bpy; print(len(bpy.data.objects))"}'
uv run --with mcp scripts/mcpcall.py get_viewport_screenshot '{"max_size":1400}' --save renders/viewport.png
uv run --with mcp scripts/mcpcall.py download_polyhaven_asset '{"asset_id":"floor_tiles_08","asset_type":"textures","resolution":"2k"}'
```
Telemetry of the MCP server was switched OFF (`disable_telemetry`); re-enable only via Blender Preferences ▸ Add-ons ▸ MCP for Blender.

## 3. Regenerate the store (catalog → planogram → labels → scene)
```bash
cd ~/workspaces/blender-supermarket
python3 scripts/gen_planogram.py     # layout + per-bay shelves/facings  -> scripts/planogram.json  (edit GONDOLAS/WALL_UNITS/CHILLERS here)
python3 scripts/gen_labels.py        # PIL labels, price tags, signs     -> assets/labels, assets/signs
# headless full build (≈4 s) from the base file that holds the Poly Haven materials + HDRI:
blender -b scenes/supermarket_base.blend --python scripts/build_supermarket.py -- --all --save scenes/supermarket.blend
# or rebuild inside the running GUI through MCP, stage by stage:
uv run --with mcp scripts/mcpcall.py execute_blender_code '{"code":"import sys,importlib; sys.path.insert(0,\"/home/manishsaini/workspaces/blender-supermarket/scripts\"); import build_supermarket as B; importlib.reload(B); B.build([\"clear\",\"structure\",\"library\",\"gondolas\"])"}'
uv run --with mcp scripts/mcpcall.py execute_blender_code '{"code":"import sys; sys.path.insert(0,\"/home/manishsaini/workspaces/blender-supermarket/scripts\"); import build_supermarket as B; B.build([\"walls\",\"chillers\",\"front\",\"produce\",\"signage\",\"lighting\",\"cameras\",\"render\"], save=\"/home/manishsaini/workspaces/blender-supermarket/scenes/supermarket.blend\")"}'
```
Stages: `clear structure library gondolas walls chillers front produce signage lighting cameras render`. Products = `scripts/catalog.py` (93 SKUs, fictional brands, real dimensions).

## 4. Roam around in Blender
* Viewport is set to Material Preview (EEVEE) with the store lights + HDRI.
* **Walk mode:** hover the 3D view, press **Shift+`** (or View ▸ Navigation ▸ Walk). WASD to move, mouse to look, **Tab** toggles gravity (eye height 1.65 m preset), Shift = run, Esc/Enter to exit.
* **Cameras:** `Cam_Entrance`, `Cam_Aisle`, `Cam_Shelf`, `Cam_Checkout`, `Cam_Produce`, `Cam_Overview`, `Cam_Exterior`, `Cam_TopDown`, `Cam_Walk` (animated fly-through, frames 1–510 (24 fps, ~21 s)). Numpad 0 = look through the active camera; Space plays the walk.
* Collections: `Structure`, `Fixtures/*`, `Products/*` (per gondola/unit), `PriceTags/*`, `Signage`, `Lighting`, `Cameras` — hide `PriceTags` or `Lighting` for a faster viewport.

## 5. Renders
```bash
cd ~/workspaces/blender-supermarket
blender -b scenes/supermarket.blend --python scripts/render_shots.py -- --samples 200 --res 1920x1080            # all cameras, Cycles/OptiX
blender -b scenes/supermarket.blend --python scripts/render_shots.py -- --samples 64 --res 1280x720 --cams Cam_Aisle   # quick check (~15 s)
blender -b scenes/supermarket.blend --python scripts/render_shots.py -- --eevee --samples 32 --cams Cam_Overview       # EEVEE still
blender -b scenes/supermarket.blend --python scripts/render_walkthrough.py -- --res 1280x720 --samples 24            # renders/walkthrough.mp4
scripts/pipeline_bg.sh &  tail -f logs/pipeline.log      # USD export + all stills + walkthrough in one go
```

## 6. Export for Isaac Sim (USD)
```bash
blender -b scenes/supermarket.blend --python scripts/export_usd.py            # -> export/supermarket.usdc (+ export/textures/)
```
Z-up, metres, root prim `/Supermarket`, products instanced per SKU mesh, custom properties on every product prim (`userProperties:sku`, `category`, `brand`, `price`, `slot`, `facing`, `row`) and on fixtures (`fixture`, `gondola`, `side`, `bay`, `level`).
In Isaac Sim: File ▸ Import / add reference to the `.usdc`; add colliders (Physics ▸ Colliders preset "convex hull" on `Products/*`, "triangle mesh" on `Fixtures/*`), lights are exported as USD lights.

## 7. Where things are
```
scripts/catalog.py          product catalog (SKU dims, colors, prices)      scripts/planogram.json   generated layout + shelf facings
scripts/gen_planogram.py    layout generator                                 scripts/gen_labels.py    PIL label/tag/sign renderer
scripts/build_supermarket.py Blender builder (staged)                        scripts/render_shots.py  stills
scripts/render_walkthrough.py fly-through video                              scripts/export_usd.py    Isaac USD export
scripts/mcpcall.py          stdio MCP test client                            scripts/launch_blender.sh start GUI + MCP socket
scenes/supermarket.blend    THE scene (25.7k objects, 24.9k product instances, 93 shared meshes)
scenes/supermarket_base.blend  empty base: Poly Haven floor_tiles_08, white_plaster_02, HDRI wide_street_01 (fake-user pinned)
assets/textures, assets/labels, assets/signs, renders/, export/, logs/
```
