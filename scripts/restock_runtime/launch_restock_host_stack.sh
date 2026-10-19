#!/usr/bin/env bash
# Task-isolated launch recipe for the restock host PLANNING stack + mapper.
#
# Starts, on DDS domain 88 and entirely under the TASK-LOCAL NVIDIA runtime closure
# (scripts/restock_runtime/runtime-env.sh -> the extracted /opt/ros/jazzy overlay +
# nav-plugin env + producer install), the full host-ROS manipulation graph defined by
# twin.restock_host.launch.description: the MoveIt move_group, the Isaac cuMotion
# servers (StaticPlanningSceneServer, CumotionPlanner, ObjectAttachmentNode,
# RobotSegmenter), the shelfbot object_segmenter, AND the nvblox mapper (NvbloxNode)
# co-launched so collision_time/ESDF is fresh. Nothing here touches the real rig's
# shared containers: it is one process group, loopback-only, in this task's overlay.
#
# This is the launch/plumbing recipe only. Starting the stack is NOT evidence of
# sensor/endpoint readiness; the owner authorizes GPU compute and integration gates on
# live evidence. Run under the exclusive GPU guard (verify_restock_host_stack_gpu.py).
# NB: no `set -u` -- sourcing the ROS/ament setup files references unset variables
# (AMENT_TRACE_SETUP_FILES etc.), which `set -u` would abort on.
set -eo pipefail

REPO="${RESTOCK_REPO:?set RESTOCK_REPO to the worktree root}"
PLUGIN_ENV="${RESTOCK_PLUGIN_ENV:-/home/manishsaini/datasets/sim-restock-host-v2/nav-plugin/runtime-env.sh}"
CONFIG="${RESTOCK_HOST_CONFIG:-/home/manishsaini/datasets/sim-restock-host-v2/refined/host.json}"

export RESTOCK_WITH_GPU_USERSPACE="${RESTOCK_WITH_GPU_USERSPACE:-1}"
# shellcheck source=/dev/null
source "$PLUGIN_ENV"
if [ -n "${RESTOCK_NVIDIA_RUNTIME:-}" ]; then
  export LD_LIBRARY_PATH="$RESTOCK_NVIDIA_RUNTIME/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export PATH="$RESTOCK_NVIDIA_RUNTIME/usr/bin:$PATH"
fi
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
# Task-only DDS: loopback, domain 88, no inherited discovery server / peers.
export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_STATIC_PEERS=''
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4 SKIP_DEFAULT_XML=1

# fm359: -u (and PYTHONUNBUFFERED from the launcher env) force UNBUFFERED host stdout so every line
# reaches host-stack.log immediately. When stdout is a FILE, python block-buffers it; the teardown kill
# then drops the last unflushed block -- run-283 lost the pan failure reason, truncated mid-stream at
# PLAN_DELIVERY. Unbuffered => a transient can never again leave us without the evidence line.
exec python3 -u "$REPO/scripts/run_restock_host.py" --config "$CONFIG" --launch
