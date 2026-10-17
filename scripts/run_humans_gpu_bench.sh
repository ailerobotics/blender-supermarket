#!/usr/bin/env bash
# Queued, guarded GPU evidence for the opt-in humans layer (sim-restock-humans-v1).
#
# Blocks up to 1 h for /tmp/aile-gpu.lock (flock -w 3600); once it holds the lock it
# runs ONE bounded (<=15 min) native ground-truth human-safety bench through
# scripts/run_restock_render_smoke.py --humans-only: scripted people cross the robot
# in the full store while the ground-truth detection + reaction run on the real
# candidate-camera frames (CLEAR -> SLOW -> STOP -> resume). LOWEST GPU priority:
# the outer flock only proceeds when the lock is free (the integration lane holds it
# across its commission+mission sequence). The VRAM guard is unchanged -- never
# --force; heavy-threshold 2500 covers the always-on whispr (~2.2 GB). ROS_DOMAIN_ID
# 104 (this lane) though the bench runs no ROS node. Cleans up only this run's own
# FastDDS shared-memory segments.
#
# Usage: scripts/run_humans_gpu_bench.sh   (no args; paths are pinned below)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/manishsaini/miniforge3/envs/env_isaaclab/bin/python
FFMPEG=/home/manishsaini/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
RUNTIME=/home/manishsaini/datasets/sim-gpu-userspace-recovery-s1/root     # 580.178.04, matches the live kernel driver
DATA=/home/manishsaini/datasets/sim-restock-humans-v1
CONFIG="$DATA/humans-bench-runtime.json"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$DATA/gpu-bench-$STAMP"      # render_smoke creates this (must not pre-exist)
mkdir -p "$DATA"

# Runtime-config: the mission's own store/scenario/manifest/camera paths + an
# explicit ground-truth humans block (gt_visible_range < the crossing's far point so
# the person leaves view and the base resumes to full speed).
"$PY" - "$CONFIG" <<'PYEOF'
import json, sys
cfg = {
  "scenario": "/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/mission-scenario.json",
  "mobile_manifest": "/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile/mobile_manifest.json",
  "camera_config": "/home/manishsaini/firstmate/data/sim-restock-camera-v2/camera-config.json",
  "store_usd": "/home/manishsaini/datasets/sim-store-realism-v2/supermarket.usdc",
  "package_roots": ["/opt/ros/jazzy/share", "/home/manishsaini/workspaces/shelfbot_newgripper"],
  "humans_bench_view": "shelf",
  "humans": {
    "enabled": True, "backend": "teleport",
    "detection": {"source": "ground_truth", "cameras": ["shelf"],
                  "gt_visible_range_m": 5.5, "gt_fov_halfangle_rad": 1.2},
    "reaction": {"slow_base_scale": 0.35, "stop_base_scale": 0.0,
                 "reach_radius_m": 1.3, "arm_hold_margin_m": 0.3,
                 "resume_clear_dwell_s": 2.0, "state_timeout_s": 0.7}
  }
}
open(sys.argv[1], "w").write(json.dumps(cfg, indent=2) + "\n")
print("wrote", sys.argv[1])
PYEOF

START="$(date +%s)"
cleanup_shm() {
  # Remove ONLY segments this run created (newer than START, owned by us). Never a
  # live one, never a rig's or another lane's (fleet ROS hygiene, inbox 004).
  find /dev/shm -maxdepth 1 -name 'fastrtps_*' -newermt "@$START" -user "$(id -un)" -delete 2>/dev/null || true
}
trap cleanup_shm EXIT

echo "[humans-gpu-bench] $(date -Is) waiting for /tmp/aile-gpu.lock (up to 3600 s)..."
flock -w 3600 /tmp/aile-gpu.lock env \
  AILE_GPU_LOCK_INHERITED=1 ROS_DOMAIN_ID=104 ROS_LOCALHOST_ONLY=1 \
  PYTHONPATH="$ROOT/scripts" \
  "$PY" "$ROOT/scripts/run_restock_render_smoke.py" \
    --humans-only --view shelf --runtime-config "$CONFIG" \
    --output "$OUT" --ffmpeg "$FFMPEG" \
    --isaac-python "$PY" --nvidia-runtime "$RUNTIME" \
    --heavy-threshold-mb 2500
status=$?

if [ "$status" -eq 0 ]; then
  echo "[humans-gpu-bench] PASSED; evidence in $OUT"
else
  echo "[humans-gpu-bench] exit=$status (lock not acquired within 3600 s, or guard/bench failure); see $OUT/run.json and $OUT/isaac.log"
fi
echo "$OUT"
exit "$status"
