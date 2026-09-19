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
#   ur_description -> /opt/ros/jazzy/share
#   shelfbot       -> the repo that DECLARES ROS package `shelfbot` in its package.xml
#                     AND carries the gripper meshes: ~/workspaces/shelfbot_newgripper.
# NB: ~/workspaces/shelfbot (the older robot repo) ALSO declares name `shelfbot` but has
# NO description/gripper/meshes — pointing at it silently imports the arm with the gripper
# meshes missing (Isaac imports even when a mesh is not found). See scripts/validate_robot_urdf.py.
if [ -d /opt/ros/jazzy ]; then
  # ROS setup.bash references unbound vars; under `set -u` that aborts the whole
  # shell (and `|| true` can't catch a fatal nounset error). Relax nounset around it.
  set +u
  # shellcheck source=/dev/null
  source /opt/ros/jazzy/setup.bash 2>/dev/null || true
  set -u
fi
SHELFBOT_PKG="${SHELFBOT_PKG:-${HOME}/workspaces/shelfbot_newgripper}"
# Build an ephemeral package-path shim so `package://shelfbot/...` resolves to
# ${SHELFBOT_PKG} whatever resolution model the importer uses — ROS_PACKAGE_PATH
# entry/pkg (<shim>/shelfbot/rel), crawl-by-package.xml-name, or ament share/
# (<shim>/share/shelfbot). Symlinks only; nothing is written to the repo or committed.
TWIN_ROS_SHIM="$(mktemp -d)/ros_pkgs"
mkdir -p "${TWIN_ROS_SHIM}/share"
ln -sfn "${SHELFBOT_PKG}" "${TWIN_ROS_SHIM}/shelfbot"
ln -sfn "${SHELFBOT_PKG}" "${TWIN_ROS_SHIM}/share/shelfbot"
export AMENT_PREFIX_PATH="${TWIN_ROS_SHIM}:${AMENT_PREFIX_PATH:-}"
export ROS_PACKAGE_PATH="${TWIN_ROS_SHIM}:${TWIN_ROS_SHIM}/shelfbot:/opt/ros/jazzy/share:${ROS_PACKAGE_PATH:-}"

# Absolute python path (a stray venv on PATH can shadow the conda env's isaacsim).
exec "${CONDA_PREFIX}/bin/python" "${SCRIPT_DIR}/compose_robot_in_store.py" "$@"
