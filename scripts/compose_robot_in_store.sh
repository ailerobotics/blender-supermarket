#!/usr/bin/env bash
# Wrapper for compose_robot_in_store.py: activates env_isaaclab (Isaac Sim 5.1) and
# puts ur_description + shelfbot on the ROS package path so the URDF's package://
# meshes resolve during URDF->USD import.
#
#   bash scripts/compose_robot_in_store.sh --report-usd      # read USD, no GPU boot
#   bash scripts/compose_robot_in_store.sh --dry-run         # full plan, no GPU boot
#   bash scripts/compose_robot_in_store.sh                   # REAL run (boots Isaac Sim)
#
# The bare form boots Isaac Sim (a GPU session). Do NOT run it while the captain's
# Blender GUI holds the scene — the built-in GPU guard will refuse, but close
# Blender first regardless. See docs/isaac_robot_in_store.md.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ROS package path for package:// mesh resolution during URDF import.
#   ur_description -> /opt/ros/jazzy/share ; shelfbot -> ~/workspaces/shelfbot
if [ -d /opt/ros/jazzy ]; then
  # shellcheck source=/dev/null
  source /opt/ros/jazzy/setup.bash 2>/dev/null || true
fi
export AMENT_PREFIX_PATH="${HOME}/workspaces/shelfbot:${AMENT_PREFIX_PATH:-}"
export ROS_PACKAGE_PATH="${HOME}/workspaces/shelfbot:/opt/ros/jazzy/share:${ROS_PACKAGE_PATH:-}"

# Absolute python path (a stray venv on PATH can shadow the conda env's isaacsim).
exec "${CONDA_PREFIX}/bin/python" "${SCRIPT_DIR}/compose_robot_in_store.py" "$@"
