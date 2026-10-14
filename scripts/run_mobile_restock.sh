#!/usr/bin/env bash
# Explicit runtime remains guarded inside Python. CPU checks do not probe the GPU.
# GNU timeout owns only this launch's process group, never unrelated applications.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESTOCK_PYTHON="${RESTOCK_PYTHON:-python3}"

# The --execute path boots Isaac Sim in RESTOCK_PYTHON. Its ROS2 bridge must load
# the bundled Jazzy rclpy, not the system one: the ELF loader and rclpy resolver
# are fixed at process creation, so this env has to be set HERE, before Python
# starts (see data/mobile-restock-v2/report.md, "Isaac ROS bridge"). Two things
# are required and must not be undone inside Python:
#   1. Drop the system ROS overlay from the resolver. A login shell sources
#      /opt/ros/<distro>/setup, which puts python3.12 site-packages on PYTHONPATH
#      and /opt/ros on AMENT_PREFIX_PATH; Isaac then imports that rclpy (wrong ABI:
#      "No module named 'rclpy._rclpy_pybind11'") instead of its bundled one. We
#      set PYTHONPATH to only this scripts dir (for twin.*) and clear the ROS
#      prefix vars. The host stack child re-sources its own overlay regardless.
#   2. Put the bundled Jazzy native libs on LD_LIBRARY_PATH and select FastRTPS,
#      so the bundled rclpy's C extension resolves ("rclpy loaded").
# RESTOCK_NVIDIA_RUNTIME (optional) is the task-local matching NVIDIA userspace
# whose nvidia-smi/libs the GPU guard and Isaac need beside a mismatched host driver.
for arg in "$@"; do
  if [ "$arg" = "--execute" ]; then
    unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH || true
    export PYTHONPATH="$SCRIPT_DIR"
    export ROS_DISTRO=jazzy
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    # The GPU guard mandates this exact domain-88 loopback isolation; set it
    # unconditionally so an inherited ROS value from the parent shell cannot trip it.
    export ROS_DOMAIN_ID=88
    export ROS_LOCALHOST_ONLY=1
    export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    # Share ONE Fast DDS transport with the host stack. Isaac's bundled Fast DDS otherwise uses
    # the default SHM+UDP builtin transport + default XML profile, which the host's UDPv4-only,
    # SKIP_DEFAULT_XML Fast DDS (twin.restock_host.config / launch_restock_host_stack.sh) cannot
    # discover -- so every Isaac OmniGraph ROS2 PUBLISHER (/clock, /tf, /tf_static,
    # /restock_sim/joint_states, /manipulation/job, /station/command) is invisible to the host
    # (run-57/59/60: pub=0 from the host side; the masked-depth + control subprocesses only worked
    # because they spawn under exactly these pins). Match the host pins so the two Fast DDS
    # participants interoperate on loopback UDPv4 and all OmniGraph publishers reach the host.
    export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
    export SKIP_DEFAULT_XML=1
    export ROS_STATIC_PEERS=
    unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE ROS_DISCOVERY_SERVER \
          CYCLONEDDS_URI RMW_FASTRTPS_USE_QOS_FROM_XML || true
    # Resolve the one bundled Jazzy bridge lib dir via the interpreter itself so
    # the exact pythonX.Y is right and stray siblings/symlinks cannot double-match.
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
      echo "run_mobile_restock.sh: one bundled Isaac Jazzy bridge lib dir required for $RESTOCK_PYTHON" >&2
      exit 3
    fi
    if [ -n "${RESTOCK_NVIDIA_RUNTIME:-}" ]; then
      export PATH="$RESTOCK_NVIDIA_RUNTIME/usr/bin:$PATH"
      export LD_LIBRARY_PATH="$RESTOCK_NVIDIA_RUNTIME/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    fi
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$jazzy"
    break
  fi
done

# Wall-clock cap for the whole launch (host warmup + Isaac boot + mission). Default 40 min
# for a collect+couple stage pair; override with RESTOCK_WALL_CAP (e.g. a short bounded
# smoke/measurement run). The resource guards (GPU lock, 4 GiB RAM watchdog) are unchanged.
RESTOCK_WALL_CAP="${RESTOCK_WALL_CAP:-2400s}"
exec timeout --signal=TERM --kill-after=15s "$RESTOCK_WALL_CAP" \
  "$RESTOCK_PYTHON" "$SCRIPT_DIR/run_mobile_restock.py" "$@"
