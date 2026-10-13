#!/usr/bin/env bash
# Extract the complete Isaac GXF/nitros RUNTIME closure the navigation nvblox
# mapper needs into a task-local overlay. The stock task env (nav-plugin/env.sh)
# resolves CUDA but lacks the Isaac GXF ament packages (gxf_isaac_gxf_helpers and
# its transitive closure) the NvbloxNode constructor requests at runtime.
#
# Extracting the whole image /opt/ros/jazzy (~2.0 GB, 480 ament packages, 60 GXF
# extensions, nvblox libraries) guarantees the complete transitive closure with
# no missing-package iteration. The overlay is a large task artifact and is NOT
# committed; this script reproduces it. Bounded CPU container, no GPU.
set -euo pipefail
IMG='nvcr.io/nvidia/isaac/ros@sha256:0ff40ee743fbd5532a0864e4a7270c958421c5f52e863d5413a2d2c127b163a1'
DEST="${1:-/home/manishsaini/datasets/sim-restock-host-v2/runtime-closure/overlay}"
mkdir -p "$DEST"
cid="$DEST/.extract.cid"; rm -f "$cid"
docker run --pull never --runtime runc --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges --memory 2g --memory-swap 2g \
  --cpus 1 --pids-limit 128 --user "$(id -u):$(id -g)" --cidfile "$cid" \
  -e NVIDIA_VISIBLE_DEVICES=void --mount "type=bind,src=$DEST,dst=$DEST" \
  --entrypoint /bin/bash "$IMG" -c "cp -a /opt/ros/jazzy '$DEST/' && du -sh '$DEST/jazzy'"
docker rm "$(cat "$cid")" >/dev/null 2>&1 || true; rm -f "$cid"
echo "overlay ready at $DEST/jazzy"
