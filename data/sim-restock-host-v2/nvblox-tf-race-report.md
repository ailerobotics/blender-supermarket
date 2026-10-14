# nvblox depth-before-/tf relay race: does nvblox retry or drop? (host-lane diagnosis)

Bounded diagnosis for firstmate msg 046 (2026-10-14), no mission run. Integration run-73:
station handshake holds, /tf and masked depth both stamped `world.current_time` with /tf
emitted first, yet nvblox never integrates and `collision_time` stays -1. Hypothesis: /tf
and depth reach the host through two independent host-DDS relay processes, so a depth frame
can arrive at nvblox before the matching /tf for its stamp; if nvblox tries the lookup once
on receipt it drops the frame.

## Method

New host-lane diagnostic tooling (branch `fm/nvblox-tf-race-diag-v1`, NOT a PR -- see verdict):
`scripts/restock_runtime/tf_race_probe.py` publishes real-shaped masked depth (frame
`restock_camera_arm`) while a **separate** process `tf_lag_publisher.py` supplies the matching
`ur_base_link -> restock_camera_arm` transform on dynamic `/tf`, stamped behind the sim clock
by a controllable lag. `scripts/run_tf_race_diag_gpu.py` runs it under the standard GPU guard
with `RESTOCK_NVBLOX_DIAG=1`, an env-gated toggle added to `launch.py` that turns on nvblox's
`print_queue_drops_to_console` / `print_delays_to_console` (inert in production).
Output: `/home/manishsaini/datasets/sim-restock-host-v2/tf-race-diag-1/`.

## What nvblox actually does (authoritative: headers + console)

**nvblox queues depth and retries the TF lookup; it does NOT drop on first miss.** From
`nvblox_ros/nvblox_node.hpp`: `processQueue(queue, mutex, ready_check, process)` only processes
items whose `ready_check` (`isPoseAvailable`/`canTransform` at the image stamp) passes; a
not-ready depth image is left in the queue and retried every `tick_period_ms` (10 ms). The
only eviction is FIFO overflow: `pushOntoQueue` drops the **oldest** item once the queue
exceeds `maximum_input_queue_length` (default **10**). Confirmed on the live stack -- the only
drop lines are `Dropped an item from: depth_queue. Size of queue: 11` (overflow), never a
per-lookup TF-miss drop. `transformer.hpp`: for the TF-tree path (`use_tf_transforms: true`,
`use_topic_transforms: false`) the 100 ms `timestamp_tolerance_ns_` applies to transform
*topics* only -- it does NOT loosen the TF-tree lookup, which goes to tf2's exact-time lookup
(fails on future extrapolation).

Empirically, with the matching /tf delivered by a separate lagging process, nvblox still
integrated the depth in 3 of 4 trials (the ESDF `get_esdf_and_gradient` integration stamp
advanced into the trial's published-stamp window) -- direct evidence of queue-and-retry.
Caveat: the probe's sim `/clock` advances far faster than wall time, so the intended
millisecond lags (0/50/150/500 ms) are swamped and the per-lag column is not a clean
threshold; the robust result is the mechanism (retry + overflow-only drop), not a per-lag
cutoff. (The probe's ack columns are void here -- its `prepare` raced ahead of the ack node's
first `/clock`, so the ack read `now=0.000` and faulted; irrelevant to the nvblox question.)

## Governing parameters

- `maximum_input_queue_length` (default 10) -- depth frames buffered awaiting a transform.
- `tick_period_ms` (10) -- retry cadence; every queued frame is re-checked every tick.
- No per-lookup TF wait/timeout exists for the TF-tree path. The tf2 Buffer cache (~10 s)
  outlives the queue window, so a /tf that arrives late is usable when it lands.

**Tolerated depth-ahead-of-/tf skew ≈ `maximum_input_queue_length / depth_publish_rate_hz`.**
At 30 Hz that is only 10/30 ≈ 0.33 s; at the mission's render-paced masked-depth rate
(low, order ~1 Hz given ~1.85 s render ticks) it is ~10 s.

## Verdict and fix

**The fix is NOT a host parameter** -- so, per the task's gate, no host-parameter PR.

Because nvblox retries and only drops on overflow, a /tf that *eventually* becomes resolvable
within `queue_len / rate` seconds is tolerated. At the mission's render-paced depth rate the
existing queue of 10 already buffers ~seconds, so a merely-late /tf would be absorbed. The
mission's **permanent** non-integration (`collision_time` stuck at -1, no voxels ever) therefore
is not queue overflow -- it means the /tf for the depth stamps never becomes **resolvable at
lookup time**: a persistent stamp skew (tf stamp < depth stamp -> future extrapolation, tf2
refuses), a gap so the buffer has no sample bracketing the depth stamp, or a frame-id mismatch.
All three are integration-side (the relay stamping/ordering), which is exactly what integration
is already fixing (align stamps to `world.current_time`, emit /tf first).

Recommendation to route to integration: guarantee that, at every depth stamp, the tf2 buffer
holds a `ur_base_link <- restock_camera_arm` sample **at or bracketing** that stamp when nvblox
retries -- i.e. publish /tf continuously (not one-shot) so the newest /tf stamp is always ≥ the
newest depth stamp, with identical `world.current_time` stamping and the /tf relay started
before depth. Host-side `maximum_input_queue_length` is the only host knob and is worth raising
(e.g. 10 -> 30, ~3x the skew tolerance) **only** if the mission's masked-depth rate is high
enough that transient relay skew overflows 10 frames; at render-paced rates it changes nothing
and does not address a persistent stamp mismatch.
