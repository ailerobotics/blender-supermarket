#!/usr/bin/env bash
# Start Blender GUI on the desktop (DISPLAY :0) inside tmux session "blender" with the supermarket scene.
# The MCP addon auto-starts its socket on localhost:9876, so Claude Code's `blender` MCP server can connect.
set -u
export PATH="$HOME/.local/bin:$PATH"
ROOT="$HOME/workspaces/blender-supermarket"
SCENE="${1:-$ROOT/scenes/supermarket.blend}"
tmux kill-session -t blender 2>/dev/null
tmux new-session -d -s blender -c "$ROOT" "DISPLAY=:0 XDG_SESSION_TYPE=x11 $HOME/opt/blender/blender '$SCENE' 2>&1 | tee -a $ROOT/logs/blender.log"
for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ':9876 ' && { echo "Blender up, MCP socket listening on 9876 (tmux attach -t blender)"; exit 0; }; sleep 1; done
echo "Blender started but port 9876 not seen yet — check: tail logs/blender.log"
