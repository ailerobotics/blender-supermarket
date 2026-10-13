# Mapper evidence: native producer + pure admission policy

This package binds every navigation map slice to the exact per-sensor
incorporated-acquisition ledger that produced it, so the slice carries documented
provenance the installed `nvblox_msgs/DistanceMapSlice` cannot. It contains:

- `include/restock_mapper_evidence/state.hpp` — a pure, ROS-independent
  admission/ledger **policy**. It decides identity/emptiness/ordering but cannot
  measure a map; the native adapter supplies real counts. Caller-supplied counts
  in its tests are never a certificate.
- the typed interfaces (`msg/`, `srv/`) agreed in navigation's 2026-10-13
  producer-interface review and frozen in instruction 019.
- `include/restock_mapper_evidence/producer_node.hpp` + `src/producer_node.cpp` —
  `EvidenceProducerNode`, a task-local subclass of the **installed**
  `nvblox::NvbloxNode`. It overrides the real virtual hooks `processDepthImage`
  and `processEsdf`, advertises the `EmptyGeneration`/`OpenAdmission`/
  `RegisterAcquisition` services on the base node's mutually-exclusive
  `group_processing_` callback group, and publishes `~/evidence_snapshot`.

The interfaces preserve identity, original acquisition clocks, certificate and
producer lifetime, full per-sensor ledger, layer inventory and embedded slice.
`steady_clock_domain` identifies the shared host `CLOCK_MONOTONIC` domain.
Snapshot indices strictly increase; ESDF revisions identify completed native
updates. `full_height_observation_verified` stays false: native full-height
unknown-column acceptance is a separate CLOSED gate.

## How the producer binds incorporation to the slice

- `processDepthImage` captures the per-frame integrated watermark, calls the real
  base method, and records an incorporation in the policy **only** when that
  frame's watermark advances to this exact acquisition stamp — a base `true`
  return (rate-limited / shape-clear discard) is not incorporation.
- `processEsdf` runs the real native ESDF update, synchronizes the CUDA stream,
  re-slices the static ESDF layer with the installed `EsdfSlicer` +
  `EsdfSliceConverter`, and publishes an `EvidenceSnapshot` whose embedded slice
  is frozen together with the full ledger, nonce, certificate, monotonic index
  and ESDF revision. `State::snapshot` withholds the envelope unless every
  configured sensor is incorporated and fresh in both clocks.
- `EmptyGeneration` measures `numAllocatedBlocks` on every maintained layer plus
  queue/watermark emptiness with admission CLOSED; an absent/unknown layer is not
  zero. `OpenAdmission` rechecks emptiness atomically and makes the acquisition
  cutoff the admission boundary (same-tick replay stays rejected). One-shot per
  generation; a restart is a new native instance with a new certificate.

Integration disables the base raw slice publication (remap `static_map_slice`)
so only the envelope carries certified geometry; the consumer republishes the
embedded slice. The launch binding is documented in `docs/restock_host.md`.

## Build and CPU verification

```bash
# Pure policy test only (no ROS, no nvblox):
cmake -S native/restock_mapper_evidence -B /abs/scratch/evidence-policy \
  -DEVIDENCE_BUILD_ROS_INTERFACES=OFF
cmake --build /abs/scratch/evidence-policy --parallel 1
ctest --test-dir /abs/scratch/evidence-policy --output-on-failure

# Interfaces + native producer component (needs installed nvblox_ros; run in the
# confined plugin container, CPU only). EVIDENCE_BUILD_PRODUCER links the
# installed nvblox_ros/nvblox libraries and proves signature + ABI closure:
cmake -S <staged-pkg> -B <build> -DEVIDENCE_BUILD_PRODUCER=ON -DCMAKE_INSTALL_PREFIX=<install>
cmake --build <build> --parallel 1 && cmake --install <build>
LD_LIBRARY_PATH=<install>/lib:$LD_LIBRARY_PATH ldd -r <install>/lib/libevidence_producer.so
python3 native/restock_mapper_evidence/test/consumer_roundtrip.py
```

The node is **never instantiated** by the CPU build: `NvbloxNode`'s constructor
may allocate GPU resources, so construction, spinning and all native runtime
proof belong to the integration worker under its GPU guards. Building the
component only proves the overridden virtual signatures and symbol/ABI closure
against the installed library. Measured build/verification evidence:
`data/sim-restock-host-v2/producer-build-evidence.md`.
