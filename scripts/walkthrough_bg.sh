#!/usr/bin/env bash
cd "$(dirname "$0")/.."; export PATH="$HOME/.local/bin:$PATH"
blender -b scenes/supermarket.blend --python scripts/render_walkthrough.py -- --res "${1:-1280x720}" --samples "${2:-24}" 2>&1 | grep --line-buffered -E "^frame|walkthrough rendered|Traceback|line [0-9]+|eevee opts" | uniq > logs/walkthrough.log
