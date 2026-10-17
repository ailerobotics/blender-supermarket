#!/usr/bin/env bash
# Env launcher for the arm-carry bench (RUNBOOK §14). Mirrors run_mobile_restock.sh's --execute
# ROS-bridge env so assemble()'s native transport graph (isaacsim.ros2.bridge) loads the bundled
# Jazzy rclpy, not the system one. The bench itself holds the GPU lock + VRAM guard inside Python
# (run_arm_carry_bench.py), drives the ArmCommandGate in-process (no host stack, no trajectory_host),
# and bounds its own per-move sim budget; this wrapper only fixes the ELF-loader/DDS env that must be
# set BEFORE Python starts. RESTOCK_NVIDIA_RUNTIME (optional) is the task-local NVIDIA userspace.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESTOCK_PYTHON="${RESTOCK_PYTHON:-python3}"

unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH || true
export PYTHONPATH="$SCRIPT_DIR"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=88
export ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export SKIP_DEFAULT_XML=1
export ROS_STATIC_PEERS=
unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE ROS_DISCOVERY_SERVER \
      CYCLONEDDS_URI RMW_FASTRTPS_USE_QOS_FROM_XML || true

jazzy="$("$RESTOCK_PYTHON" - <<'PY'
import glob, os, sys
prefix = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
pat = os.path.join(prefix, 'lib', 'python*', 'site-packages', 'isaacsim',
                   'exts', 'isaacsim.ros2.bridge', 'jazzy', 'lib')
cands = sorted({os.path.realpath(p) for p in glob.glob(pat)})
cands = [c for c in cands if os.path.isfile(os.path.join(c, 'librmw_fastrtps_cpp.so'))]
print(cands[0] if len(cands) == 1 else '')
PY
)"
if [ -z "$jazzy" ] || [ ! -f "$jazzy/librmw_fastrtps_cpp.so" ]; then
  echo "run_arm_carry_bench.sh: one bundled Isaac Jazzy bridge lib dir required for $RESTOCK_PYTHON" >&2
  exit 3
fi
if [ -n "${RESTOCK_NVIDIA_RUNTIME:-}" ]; then
  export PATH="$RESTOCK_NVIDIA_RUNTIME/usr/bin:$PATH"
  export LD_LIBRARY_PATH="$RESTOCK_NVIDIA_RUNTIME/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$jazzy"

# Short bounded wall cap for the whole bench session (GPU etiquette: keep sessions under 15 min).
RESTOCK_WALL_CAP="${RESTOCK_WALL_CAP:-800s}"
exec timeout --signal=TERM --kill-after=20s "$RESTOCK_WALL_CAP" \
  "$RESTOCK_PYTHON" "$SCRIPT_DIR/run_arm_carry_bench.py" "$@"
