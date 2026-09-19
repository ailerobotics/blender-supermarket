#!/usr/bin/env bash
# Bounded GPU run — stand the robot in the store under Isaac Sim, safely.
#
# Wraps scripts/compose_robot_in_store.sh (which activates env_isaaclab and sets the
# ROS package path) with the operational guard rails this tight 12 GB box needs:
#   * a HARD wall-clock cap (default 20 min) — Isaac Sim has OOM-hung at boot here;
#   * a RAM watchdog that stops THIS run (its own process group, by PID) if available
#     system RAM drops below a floor (default 4 GB), so a runaway can't wedge the box;
#   * a background sampler that records peak GPU and RAM for the run report.
#
# It does NOT pass --force and never overrides the guard's hard Blender gate, and by
# default it does NOT touch the guard's heavy-consumer threshold — the guard default
# (1500 MiB) stays authoritative. If the long-running `whispr` service (~2.2 GB VRAM, a
# protected always-on consumer that never frees the GPU) is resident, the guard will
# correctly REFUSE the boot; to accept whispr as a known-stable consumer, set
# HEAVY_MB=2500 explicitly (an OPT-IN, not a --force, and the free-VRAM budget >= 8 GB is
# unchanged). See data/twin-robot-in-store-v1/report.md and RUNBOOK section 8.
#
# Usage:
#   bash scripts/run_robot_in_store_gpu.sh                 # default output dir + bounds
#   HEAVY_MB=2500 bash scripts/run_robot_in_store_gpu.sh   # accept whispr (~2.2 GB) as stable
#   TIMEOUT_S=900 RAM_FLOOR_GB=6 bash scripts/run_robot_in_store_gpu.sh
#   bash scripts/run_robot_in_store_gpu.sh --steps 240     # extra args pass through
#
# Refuses to launch unless the pre-boot readiness checks pass (no Blender, free GPU,
# free RAM); on any failure it prints the numbers and exits non-zero WITHOUT booting.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/twin-robot-in-store-v1}"
mkdir -p "$OUT_DIR"

STORE_USD="${STORE_USD:-$REPO_ROOT/export/supermarket.usdc}"  # this repo's export, not ~/workspaces
STAGE="${STAGE:-$OUT_DIR/store_with_robot.usd}"
SHOT="${SHOT:-$OUT_DIR/store_with_robot.png}"
LOG="${LOG:-$OUT_DIR/isaac_run.log}"
METRICS="${METRICS:-$OUT_DIR/run_metrics.txt}"
TIMEOUT_S="${TIMEOUT_S:-1200}"        # 20 min hard wall-clock cap
RAM_FLOOR_GB="${RAM_FLOOR_GB:-4}"     # stop our run if available RAM drops below this
GPU_FLOOR_MB="${GPU_FLOOR_MB:-8000}"  # pre-boot: require this much free VRAM
RAM_MIN_GB="${RAM_MIN_GB:-12}"        # pre-boot: require this much available RAM
HEAVY_MB="${HEAVY_MB:-}"              # OPT-IN heavy-consumer threshold; empty => leave the
                                      # guard default (1500 MiB) in force. Set HEAVY_MB=2500
                                      # to accept whispr (~2.2 GB) as a known-stable consumer.

log() { echo "[run] $*"; }

# --- Pre-boot readiness gate (mirrors the brief; refuse rather than risk an OOM) ---
if pgrep -af '/blender( |$)|opt/blender/blender|blenderplayer' | grep -qv "$$"; then
  echo "REFUSING: a Blender process is present — close/save Blender first." >&2
  pgrep -af 'opt/blender/blender' >&2 || true
  exit 3
fi
if ss -ltn 2>/dev/null | grep -q ':9876'; then
  echo "REFUSING: the Blender MCP socket :9876 is listening — Blender is still up." >&2
  exit 3
fi
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
RAM_AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
if [ -z "${GPU_FREE:-}" ]; then
  echo "REFUSING: could not read nvidia-smi free VRAM." >&2; exit 3
fi
if [ "$GPU_FREE" -lt "$GPU_FLOOR_MB" ]; then
  echo "REFUSING: only ${GPU_FREE} MiB VRAM free (need >= ${GPU_FLOOR_MB})." >&2; exit 3
fi
if [ "${RAM_AVAIL_GB:-0}" -lt "$RAM_MIN_GB" ]; then
  echo "REFUSING: only ${RAM_AVAIL_GB} GiB RAM available (need >= ${RAM_MIN_GB})." >&2; exit 3
fi
log "readiness OK: no Blender, ${GPU_FREE} MiB VRAM free, ${RAM_AVAIL_GB} GiB RAM available."
log "outputs -> stage=$STAGE  screenshot=$SHOT  log=$LOG  metrics=$METRICS"
log "bounds  -> wall-clock ${TIMEOUT_S}s, RAM floor ${RAM_FLOOR_GB} GiB, heavy-threshold ${HEAVY_MB:-<guard default 1500>} MiB"

: > "$LOG"
: > "$METRICS"
echo "start_epoch=$(date +%s)" >> "$METRICS"
echo "gpu_free_pre_mb=$GPU_FREE ram_avail_pre_gb=$RAM_AVAIL_GB" >> "$METRICS"

# --- Launch the real run in its OWN process group so we can tear down Isaac's whole
#     child tree by PID (setsid: $! is the group leader, PGID == PID). ---
HEAVY_ARGS=()
[ -n "$HEAVY_MB" ] && HEAVY_ARGS=(--heavy-threshold-mb "$HEAVY_MB")
setsid bash "$SCRIPT_DIR/compose_robot_in_store.sh" \
    --store "$STORE_USD" --save "$STAGE" --screenshot "$SHOT" \
    ${HEAVY_ARGS[@]+"${HEAVY_ARGS[@]}"} "$@" \
    > "$LOG" 2>&1 &
RUN_PID=$!
log "launched Isaac run (process group $RUN_PID); tailing $LOG"

stop_group() {  # TERM then KILL our own group only
  local why="$1"
  echo "[run] stopping run group $RUN_PID: $why" | tee -a "$LOG" >&2
  kill -TERM -"$RUN_PID" 2>/dev/null
  for _ in $(seq 1 15); do kill -0 "$RUN_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -"$RUN_PID" 2>/dev/null
}

# Wall-clock watcher.
( sleep "$TIMEOUT_S"; kill -0 "$RUN_PID" 2>/dev/null && stop_group "wall-clock ${TIMEOUT_S}s hit" ) &
CLOCK_PID=$!

# RAM watchdog + peak GPU/RAM sampler.
PEAK_GPU=0; PEAK_RAM=0
( while kill -0 "$RUN_PID" 2>/dev/null; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    avail_mb=$(free -m | awk '/^Mem:/{print $7}')
    used_ram_mb=$(free -m | awk '/^Mem:/{print $3}')
    [ -n "${used:-}" ] && [ "$used" -gt "$PEAK_GPU" ] && PEAK_GPU=$used
    [ -n "${used_ram_mb:-}" ] && [ "$used_ram_mb" -gt "$PEAK_RAM" ] && PEAK_RAM=$used_ram_mb
    echo "peak_gpu_used_mb=$PEAK_GPU peak_ram_used_mb=$PEAK_RAM last_ram_avail_mb=${avail_mb:-?}" > "$METRICS.live"
    if [ -n "${avail_mb:-}" ] && [ "$avail_mb" -lt $((RAM_FLOOR_GB*1024)) ]; then
      stop_group "available RAM ${avail_mb} MiB < ${RAM_FLOOR_GB} GiB"
      break
    fi
    sleep 3
  done ) &
WATCH_PID=$!

wait "$RUN_PID"; RC=$?
kill "$CLOCK_PID" "$WATCH_PID" 2>/dev/null
wait "$CLOCK_PID" "$WATCH_PID" 2>/dev/null

# Fold the live peaks into the metrics file.
[ -f "$METRICS.live" ] && cat "$METRICS.live" >> "$METRICS" && rm -f "$METRICS.live"
echo "end_epoch=$(date +%s) exit_code=$RC" >> "$METRICS"

echo "----------------------------------------------------------------------"
log "run exit code: $RC"
[ -f "$STAGE" ] && log "stage written: $STAGE ($(du -h "$STAGE" | cut -f1))" || log "NO stage written"
[ -f "$SHOT" ] && log "screenshot written: $SHOT ($(du -h "$SHOT" | cut -f1))" || log "NO screenshot written"
log "metrics:"; sed 's/^/    /' "$METRICS"
log "last 25 log lines:"; tail -25 "$LOG" | sed 's/^/    /'
exit "$RC"
