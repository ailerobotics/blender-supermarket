#!/usr/bin/env bash
# Guarded GPU wrapper for the choreographed restock DEMO (captain 2026-10-20).
#
# Activates env_isaaclab (Isaac Sim 5.1) and runs scripts/twin/demo/run_demo_gpu.py, which
# holds the shared GPU flock (twin.restock.guard.gpu_slot; VRAM guard unchanged; never
# --force) for the whole capture. Unlike the autonomous mission this demo does NOT enable
# the ROS2 bridge or launch any host stack, so the heavy bridge/DDS env is not needed; only
# the ROS domain-88 isolation pins the GPU guard mandates are set. The package_roots in the
# runtime config resolve the URDF meshes (no ROS_PACKAGE_PATH shim required).
#
#   bash scripts/run_demo_gpu.sh --config <cfg.json> --out <dir> --fps 10 --encode
#   SMOKE: ... --smoke 8   (captures ~8 frames, one per stage, for a framing/boot check)
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${SCRIPT_DIR}"
# GPU-guard-mandated domain-88 loopback isolation (the guard binds the domain-88 port).
export ROS_DOMAIN_ID=88
export ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

WALL_CAP="${DEMO_WALL_CAP:-1500s}"
exec timeout --signal=TERM --kill-after=15s "${WALL_CAP}" \
  "${CONDA_PREFIX}/bin/python" "${SCRIPT_DIR}/twin/demo/run_demo_gpu.py" "$@"
