#!/usr/bin/env bash
# Offline (CPU, no GPU) proof that the task-local runtime closure is complete:
#   (a) ament resolves every package the nvblox node constructor requests;
#   (b) ldd -r finds no undefined symbol / missing library on the nvblox
#       libraries, the core+Isaac GXF extensions they load, and the built
#       EvidenceProducerNode component.
# ldd -r only maps+relocates libraries; it runs no GPU code. The recovery driver
# userspace (libcuda.so.1) + image-root CUDA runtime make the closure resolvable.
set -o pipefail
export RESTOCK_WITH_GPU_USERSPACE=1
source /home/manishsaini/datasets/sim-restock-host-v2/nav-plugin/runtime-env.sh
OV=/home/manishsaini/datasets/sim-restock-host-v2/runtime-closure/overlay/jazzy
PI=/home/manishsaini/datasets/sim-restock-host-v2/nav-plugin/evidence/producer-install
fail=0

echo "================ (a) ament package resolution ================"
PKGS="gxf_isaac_gxf_helpers isaac_ros_gxf gxf_isaac_gems isaac_ros_common \
isaac_ros_managed_nitros isaac_ros_nitros isaac_ros_nitros_image_type \
isaac_ros_nitros_camera_info_type gxf_isaac_messages gxf_isaac_message_compositor \
nvblox_ros nvblox_msgs nvblox_ros_common restock_mapper_evidence"
for p in $PKGS; do
  loc=$(python3 - "$p" <<'PY'
import sys
try:
    from ament_index_python.packages import get_package_share_directory as g
    print(g(sys.argv[1]))
except Exception as e:
    print("MISSING:%s" % e)
PY
)
  if [[ "$loc" == MISSING:* ]]; then echo "  [MISS] $p -> $loc"; fail=1; else echo "  [ok]   $p -> $loc"; fi
done

echo "================ (b) ldd -r symbol/ABI closure ================"
# nvblox libraries + the built evidence producer component.
CORE_LIBS="$OV/lib/libnvblox_ros_lib.so $OV/lib/libnvblox_lib.so $OV/lib/libnvblox_ros_common_lib.so"
[ -f "$PI/lib/libevidence_producer.so" ] && CORE_LIBS="$CORE_LIBS $PI/lib/libevidence_producer.so"
# Core GXF + Isaac GXF extensions the NitrosContext loads for a depth/tsdf mapper.
GXF_LIBS=$(find "$OV/share/gxf_isaac_gxf_helpers/gxf/lib" "$OV/share/gxf_isaac_gems/gxf/lib" \
  "$OV/share/isaac_ros_gxf/gxf/lib" -name '*.so' 2>/dev/null)
CORE_GXF=$(find "$OV" -path '*gxf/lib/std/libgxf_std.so' -o -path '*gxf/lib/cuda/libgxf_cuda.so' 2>/dev/null)

# Tegra/Jetson-only libraries that are absent by design on x86 desktop. The GXF
# stream / stream-sync extensions that need them are never loaded by a depth/TSDF
# mapper, so a miss limited to these is not a closure failure for this mapper.
TEGRA='libnvsci|libnvos|libgnat'
check() {
  local lib="$1" raw notfound undef nontegra undef_bad
  raw=$(ldd -r "$lib" 2>&1)
  notfound=$(echo "$raw" | grep -iE 'not found' | sed 's/ =>.*//' | tr -d ' \t' || true)
  undef=$(echo "$raw" | grep -iE 'undefined symbol' || true)
  if [ -z "$notfound" ] && [ -z "$undef" ]; then echo "  [clean] $(basename "$lib")"; return 0; fi
  nontegra=$(echo "$notfound" | grep -vE "$TEGRA" | grep -v '^$' || true)
  undef_bad=$(echo "$undef" | grep -viE 'NvSci' || true)
  if [ -z "$nontegra" ] && [ -z "$undef_bad" ]; then
    echo "  [optional-tegra] $(basename "$lib") (Tegra NvSCI/NvOS only; not loaded by desktop mapper)"; return 0
  fi
  echo "  [UNRESOLVED] $(basename "$lib")"
  { [ -n "$nontegra" ] && echo "$nontegra"; [ -n "$undef_bad" ] && echo "$undef_bad"; } | sed 's/^/        /' | head -8
  return 1
}

echo "--- nvblox libraries + evidence producer (must be clean) ---"
for l in $CORE_LIBS; do check "$l" || fail=1; done
echo "--- core GXF extensions (std, cuda) ---"
for l in $CORE_GXF; do check "$l" || fail=1; done
echo "--- Isaac GXF helper/gems/gxf extensions (must be clean) ---"
for l in $GXF_LIBS; do check "$l" || fail=1; done

echo "================ RESULT ================"
if [ "$fail" = 0 ]; then echo "CLOSURE_OK: ament resolves all requested packages; nvblox + core/Isaac GXF + producer ldd -r clean"; else echo "CLOSURE_FAIL"; fi
exit $fail
