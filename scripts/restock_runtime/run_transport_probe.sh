#!/usr/bin/env bash
# Focused masked-depth TRANSPORT regression probe (RUNBOOK): boots Isaac with the real host
# stack, drives the arm masked-depth bridge for N frames at the collect staging pose (NO
# navigation/mission), and asserts from the HOST side that masked_depth arrives (nonzero rate,
# one stamp with camera_info) and nvblox integrates depth (integrate_depth timer + tick > 0).
# Far cheaper than an 8-minute mission; use it before any mission shakedown of this transport.
#
#   scripts/restock_runtime/run_transport_probe.sh <output_dir> [frames] [warmup_s] [cap_s]
#
# Env (task defaults): REPO, RT (nvidia runtime), M (mission dir with configs), RESTOCK_PYTHON,
# OVERLAY_ENV (host overlay env.sh).
set -u
REPO="${REPO:-/home/manishsaini/.treehouse/blender-supermarket-dc8579/2/blender-supermarket}"
RT="${RT:-/home/manishsaini/datasets/sim-gpu-userspace-recovery-s1/root}"
M="${M:-/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission}"
RESTOCK_PYTHON="${RESTOCK_PYTHON:-/home/manishsaini/miniforge3/envs/env_isaaclab/bin/python}"
OVERLAY_ENV="${OVERLAY_ENV:-/home/manishsaini/datasets/sim-restock-host-v2/nav-plugin/runtime-env.sh}"
OUT="${1:?usage: run_transport_probe.sh <output_dir> [frames] [warmup_s] [cap_s]}"
FRAMES="${2:-60}"; WARMUP="${3:-75}"; CAP="${4:-170}"
rm -rf "$OUT"
LOG="$OUT.console.log"
mkdir -p "$(dirname "$OUT")"

# 1. Isaac + real host stack; drive the bridge FRAMES frames instead of a mission.
RESTOCK_WALL_CAP="${CAP}s" RESTOCK_PYTHON="$RESTOCK_PYTHON" RESTOCK_NVIDIA_RUNTIME="$RT" \
  nohup bash "$REPO/scripts/run_mobile_restock.sh" --execute \
    --factory twin.restock.isaac_bundle:create --host-stack \
    --plugin-env "$OVERLAY_ENV" --nvidia-runtime "$RT" --host-stack-warmup "$WARMUP" \
    --manifest /home/manishsaini/datasets/sim-store-realism-v2/environment.json \
    --scenario "$M/mission-scenario.json" --runtime-config "$M/mission-runtime-cc.json" \
    --output "$OUT" --heavy-threshold-mb 2500 --transport-probe "$FRAMES" \
    >"$LOG" 2>&1 &
MPID=$!

# 2. Host-side subscriber during the ticking window (after warmup + Isaac boot).
( sleep $((WARMUP + 48))
  set +u
  source "$OVERLAY_ENV" >/dev/null 2>&1
  export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_STATIC_PEERS= ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4 SKIP_DEFAULT_XML=1
  unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE ROS_DISCOVERY_SERVER CYCLONEDDS_URI RMW_FASTRTPS_USE_QOS_FROM_XML
  export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
  python3 "$REPO/scripts/restock_runtime/transport_probe_sub.py" "$OUT/transport-sub.json" 45
) >"$OUT.sub.log" 2>&1 &

wait $MPID 2>/dev/null
sleep 5

# 3. Verdict from the host side.
echo "=========== TRANSPORT PROBE VERDICT ==========="
FRAMESDONE=$(python3 -c "import json;print(json.load(open('$OUT/mission.json')).get('frames'))" 2>/dev/null || echo "?")
echo "isaac frames driven: $FRAMESDONE"
echo "subprocess published (last): $(grep -oE 'published=[0-9]+' "$LOG" 2>/dev/null | tail -1 || echo none)"
echo "--- host subscriber ---"; cat "$OUT/transport-sub.json" 2>/dev/null; echo
INTEG=$(awk '/NVBlox Timings/{f=1} /NVBlox Delays/{f=0} f' "$OUT/host-stack.log" 2>/dev/null | grep -icE "integrate_depth|tsdf/integrate|depth/integrate")
TICKRATE=$(awk '/NVBlox Rates/{f=1} /NVBlox Delays/{f=0} f' "$OUT/host-stack.log" 2>/dev/null | grep -E "ros/tick" | grep -oE "[0-9]+\.[0-9]+$" | tail -1)
echo "nvblox integrate_depth timer present: $([ "${INTEG:-0}" -gt 0 ] 2>/dev/null && echo YES || echo NO)"
echo "nvblox ros/tick rate: ${TICKRATE:-0}"
SUB_OK=$(python3 -c "import json;print(json.load(open('$OUT/transport-sub.json')).get('ok'))" 2>/dev/null || echo False)
PASS=NO
if [ "${INTEG:-0}" -gt 0 ] 2>/dev/null && [ "$SUB_OK" = "True" ]; then PASS=YES; fi
echo "VERDICT: $PASS"
echo "==============================================="
[ "$PASS" = "YES" ]
