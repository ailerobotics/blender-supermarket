# Task-local nvblox RUNTIME environment for the navigation mapper + evidence producer.
# Source in a host-Python ROS shell. Starts nothing; changes no global config.
#
# Layering (first wins for ament):
#   1. complete Isaac GXF/nitros runtime closure  = full image /opt/ros/jazzy overlay
#   2. task nav-plugin env.sh  (host ROS + domain-88 nav2 overlay + image-root CUDA + nvblox_nav2 install)
#   3. built EvidenceProducerNode component       = evidence/producer-install
#
# The overlay supplies the packages the stock env lacked (gxf_isaac_gxf_helpers and
# its whole transitive GXF/nitros closure) that the nvblox node constructor requests.

HOSTV2=/home/manishsaini/datasets/sim-restock-host-v2
OVERLAY="$HOSTV2/runtime-closure/overlay/jazzy"
PRODUCER_INSTALL="$HOSTV2/nav-plugin/evidence/producer-install"
RECOVERY="/home/manishsaini/datasets/sim-gpu-userspace-recovery-s1/root"

# base task env (host ROS, nav2 domain-88 overlay, image-root CUDA, nvblox_nav2 plugin)
source "$HOSTV2/nav-plugin/env.sh"

# 1. complete runtime closure overlay (ament + libs + gxf extension manifests)
export AMENT_PREFIX_PATH="$OVERLAY:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="$OVERLAY:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$OVERLAY/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$OVERLAY/lib/python3.12/site-packages:$OVERLAY/local/lib/python3.12/dist-packages:${PYTHONPATH:-}"
export PATH="$OVERLAY/bin:$PATH"

# 3. built native evidence producer component (restock_mapper_evidence)
if [ -d "$PRODUCER_INSTALL" ]; then
  export AMENT_PREFIX_PATH="$PRODUCER_INSTALL:$AMENT_PREFIX_PATH"
  export LD_LIBRARY_PATH="$PRODUCER_INSTALL/lib:$LD_LIBRARY_PATH"
  export PYTHONPATH="$PRODUCER_INSTALL/lib/python3.12/site-packages:$PYTHONPATH"
fi

# GPU userspace: recovery driver libs (libcuda.so.1, libnvidia-ml) for host runs whose
# system driver userspace is version-mismatched. CUDA toolkit runtime (libcudart.so.13)
# already comes from image-root via nav-plugin/env.sh. Only prepend for a GPU run.
if [ "${RESTOCK_WITH_GPU_USERSPACE:-0}" = "1" ]; then
  export LD_LIBRARY_PATH="$RECOVERY/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
  export PATH="$RECOVERY/usr/bin:$PATH"
fi

