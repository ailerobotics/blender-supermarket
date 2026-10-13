# Navigation nvblox mapper — task-local RUNTIME closure

Recipe to run the navigation nvblox mapper (and the host
`restock_mapper_evidence::EvidenceProducerNode`) on the host GPU. These files
encode task-local absolute dataset paths; they are a reproducible recipe, not a
general library. The large overlay (~2.0 GB) and the built producer are task
dataset artifacts, not committed. Full measured evidence + honest limits:
`data/sim-restock-host-v2/runtime-closure-report.md`.

## The gap and the fix

The stock task env (`sim-restock-host-v2/nav-plugin/env.sh`) resolves CUDA
(`libcudart.so.13` from `image-root`) but lacks the Isaac GXF/nitros ament
closure the `NvbloxNode` constructor requests (`gxf_isaac_gxf_helpers` + its
transitive packages), so the node constructor fails at runtime. The fix extracts
the complete image `/opt/ros/jazzy` into a task-local overlay and prepends it.

## Steps

```bash
# 1. extract the complete runtime closure (bounded CPU container, ~10 s)
bash scripts/restock_runtime/extract_runtime_overlay.sh

# 2. prove closure offline (CPU, no GPU): ament resolution + ldd -r
bash scripts/restock_runtime/offline_closure_proof.sh      # -> CLOSURE_OK

# 3. source the runtime env (add RESTOCK_WITH_GPU_USERSPACE=1 for a GPU run)
source scripts/restock_runtime/runtime-env.sh
```

`runtime-env.sh` layers (first wins for ament): full-image `/opt/ros/jazzy`
overlay (the closure) → `nav-plugin/env.sh` (host ROS + domain-88 nav2 overlay +
image-root CUDA + nvblox_nav2 install) → `evidence/producer-install` (the built
EvidenceProducerNode) → optional recovery GPU driver userspace.

## Guarded GPU verification

`scripts/run_evidence_mapper_gpu.py` holds the exclusive GPU guard (flock
`/tmp/aile-gpu.lock`, refuse on a protected Blender process / MCP socket, ≥8 GiB
free VRAM, ≥12 GiB RAM, domain-88 loopback), launches the mapper as the host
producer via `scripts/launch_evidence_mapper.py --execute`, measures construction
(node present, the three incorporation services + `evidence_snapshot` topic
advertised, no GXF-closure/CUDA error) and tears down by exact PID group, capped
under 5 minutes. It prefers the canonical `twin.restock.guard.gpu_slot` when
importable, else an inline equivalent with the same refusals.

Publishing a first slice BOUND to an incorporation identity additionally needs
the integration sensor+handshake driver (NitrosImage depth + `camera_info` +
camera→`map` TF + `/clock`, plus EmptyGeneration → OpenAdmission →
RegisterAcquisition); that is sensor/integration-owned and not driven here.
