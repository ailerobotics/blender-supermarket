# Bounded GPU validation config — hand to the integration lane `twin-robot-in-store-boot-v1`

One bounded run confirms the CPU decision (`report.md`) on real physics: the lidar yaw fix ON (now the
code default) **plus** the proposed `tow_lane_trust_static` retirement, over warm8's own couple → outbound
→ return tow lanes. The worker does **not** take the GPU; this is scheduled through the integration lane,
serialized on `/tmp/aile-gpu.lock` (standard non-blocking acquire, never `--force`), `HEAVY_MB=2500`
opt-in only (the always-on `whispr` ~2.2 GB tripping the 1500 MiB default is accepted, per AGENTS.md).

## The config (two edits to a COPY of the warm8 runtime config)

Do not edit the live `mission-runtime-couple.json` in place — make a validation copy (e.g.
`mission-runtime-couple-yawfix.json`) with exactly these two changes:

1. **Fix on** — it is now the code default, so simply **omit** `lidar_sensor_yaw_fix` (or set it `true`
   explicitly for the run record). The boot prints `LIDAR_SENSOR_TO_BASE (0.52, 0.0, 0.0)
   (lidar_sensor_yaw_fix=True)` — confirm that line.
2. **Retire the trust_static waiver** — remove `navigation.tow_lane_trust_static` (or set it `false`).
   **Keep** `navigation.tow_lane_static_tol_m` (0.12) and `navigation.tow_lane_trust_unknown` (true): the
   yaw fix does not make those unnecessary (see `report.md` §3).

Everything else (scenario, lanes, poses, camera config, stages) stays byte-identical to warm8.

## Run

Same invocation as the warm8 baseline (from its console log), writing to a fresh scratch output dir, with
the validation config:

```
scripts/run_robot_in_store_gpu.sh \
  --scenario   /home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/mission-scenario.json \
  --runtime-config <the validation copy>.json \
  --manifest   /home/manishsaini/datasets/sim-store-realism-v2/environment.json \
  --output     /home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/<warmN-yawfix> \
  --heavy-threshold-mb 2500
```

(The GPU-session runbook is RUNBOOK §8; the readiness gate + 20-min cap + RAM watchdog are built into the
wrapper.)

## PASS / FAIL criteria (couple + outbound + return)

**PASS** iff, with `tow_lane_trust_static` OFF:
- boot prints the yaw-0 wiring (`lidar_sensor_yaw_fix=True`);
- **couple** reaches the standoff/dock and latches (no `blocked_no_route` on the couple legs beyond what
  `couple_static_drive` already handles);
- **outbound** tows dock → bay and arrives, with **no** `LANE_BLOCKED_DIAG … trust_static=False`
  seed-collision on a static-only cloud and **no** forward-capsule `obstacle_stop` deadlock on the lane;
- **return** tows bay → `cart_parking` and latches `arrived_stopped`.

**FAIL / keep the waiver** if the outbound or return lane reports `lane_blocked` (seed_collision) with
`trust_static=False` on a cloud with no dynamic obstacle source — that would be the residual
unmapped-static return the CPU AABB model cannot see (`report.md` §3 caveat). In that case keep
`tow_lane_trust_static` and report the kept-return diagnostic (`LANE_BLOCKED_DIAG … kept_near_static=…`).

Report the couple/outbound/return verdicts back through firstmate; the worker folds the PASS/FAIL into its
done line.
