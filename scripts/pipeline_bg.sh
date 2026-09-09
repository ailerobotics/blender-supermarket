#!/usr/bin/env bash
# Background pipeline: USD export -> Cycles stills -> EEVEE walkthrough. Logs to logs/pipeline.log
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
LOG=logs/pipeline.log; : > "$LOG"
{
  echo "== USD export $(date)"; blender -b scenes/supermarket.blend --python scripts/export_usd.py 2>&1 | grep -E "USD written|Error|Traceback|retrying|line [0-9]+"
  echo "== stills $(date)";     blender -b scenes/supermarket.blend --python scripts/render_shots.py -- --samples 200 --res 1920x1080 --cams Cam_Entrance,Cam_Aisle,Cam_Shelf,Cam_Checkout,Cam_Produce,Cam_Overview,Cam_TopDown,Cam_Exterior 2>&1 | grep -E "rendered|Error|Traceback|line [0-9]+"
  echo "== walkthrough $(date)"; blender -b scenes/supermarket.blend --python scripts/render_walkthrough.py -- --res 1280x720 --samples 24 2>&1 | grep -E "walkthrough rendered|Error|Traceback|line [0-9]+|Append frame (1|100|200|300|390)$"
  echo "== done $(date)"
} >> "$LOG" 2>&1
