# Route alerts — the mission robot as a mobile store sensor

While the restocking robot tows the cart down the aisle its cameras see the floor and
shelves. **Route alerts** bridges that camera stream to the read-only
the `store-vision` suite (a separate package) detectors so the drive
produces the same findings a fixed store camera would — floor **spills**, aisle
**obstructions**, misplaced / out-of-cold-chain **strays** — publishes them as
`robot.route_alert.*` events plus optional staff tasks, and feeds the obstruction
footprints to navigation **through the moving-obstacle costmap layer the
`sim-restock-dynobs-v1` lane owns** (`twin.navigation.dynamic`). The obstruction path
doubly serves the collision-aware requirement.

Everything here is **opt-in and default off**: with no `route_alerts` config section the
mission imports nothing from this package and runs byte-identically. See
`scripts/twin/route_alerts/` and RUNBOOK §19.

## Two modes — the working default and the smart variant

`route_alerts.mode` (launcher `--route-alerts[=act]`) picks between the two delivered
behaviours. **One version with the working default is kept**; the smart behaviour is
opt-in on top:

| Mode | Flag | What the robot does | Navigation |
|---|---|---|---|
| **`findings`** (default) | `--route-alerts` | Pure **sensor**: publishes `robot.route_alert.*` events + staff tasks + the evidence log. Findings never change where the robot drives. | **byte-identical** — `latest_detections()` returns `[]`, the dynobs layer is fed nothing. |
| **`act`** (smart) | `--route-alerts=act` | Everything `findings` does, **plus** a confirmed **obstruction** footprint is exposed so it raises the route cost. | fed **only** through the dynobs costmap layer (`navigation.dynamic_obstacles`, nav_mode `sensor`); needs no new layer or gate. |

Only an `obstruction` footprint ever moves the robot (it is the one finding that is a
base passability hazard); `spill`/`strays` never feed navigation even in `act`. The
findings themselves are published in **both** modes — only the navigation seam is gated.
This is the captain's "keep one version with working defaults, then move to the smart one
if the working default is acting stupid": run `findings` for a sensor-only baseline, flip
to `act` when the costmap coupling is wanted.

## Pipeline

```
drive camera (RGB+depth, people-masked)
      │  RouteFrame                              scripts/twin/route_alerts/frames.py
      ▼
StoreVisionBridge ── store-vision analyzers ──► store-vision Event findings
      │   (spill, obstruction; strays optional)  scripts/twin/route_alerts/bridge.py
      ├─► to_robot_events ──► robot.route_alert.* events   (BT lane + mission artifact)
      ├─► build_staff_tasks ──► de-duplicated StaffTask    (store-vision tasks)
      └─► obstructions_to_detections ──► twin.navigation.dynamic.Detection
                                            │   scripts/twin/route_alerts/costmap.py
                                            ▼
            DynamicObstacleLayer (owned by sim-restock-dynobs-v1)
            → SensorNavigation.step(detections=…)  ← add-only occupancy union
```

`RouteAlertsRunner.observe(frames, now)` (`runner.py`) is the whole subsystem from the
mission's point of view; the mission attaches it via the guarded
`RouteAlertsMissionHook` (`mission.py`) only when configured.

## Per-detector GPU / CPU path

GPU-accelerated inference is preferred wherever store-vision has a GPU path (the single
RTX 3060 is shared with the running mission under `/tmp/aile-gpu.lock`); the classical
CPU path is used where no GPU path exists, **stated per detector**:

| Detector | CPU path (default) | GPU path (opt-in `gpu: true`) | Feeds nav? |
|---|---|---|---|
| **spill** | classical OpenCV floor-hazard pipeline (people-masked). | store-vision OWLv2 + SAM 2 open-vocab refine (`store-vision[spill]`, torch/transformers) — sharpens class + extent; shares the GPU with the mission. | no |
| **obstruction** | classical depth-protrusion + clean-floor change (people-masked). This path **also** produces the passability layer + footprints that feed navigation, so CPU is both sufficient and preferred (no GPU contention). | OWLv2 open-vocab object classifier — **label quality only**; geometry/footprints unchanged. | **yes, in `act` mode** (dynobs) |
| **strays / coldchain** | *none* — semantic: needs SKU-per-slot recognition. | store-vision products recognizer (OWLv2 / SigLIP, GPU) → `SlotObservation`s → `StrayDetector`. Off unless a `slot_observation_provider` + detector are wired. | no |

`chiller-door`: store-vision today has **no dedicated per-frame "open chiller door"
vision detector** (`openclose` is a *semantic checklist* aggregating other modules'
events, not a pixel detector). The runner leaves a pluggable slot; a proposal for a
store-vision chiller-door detector is in `data/sim-route-alerts-v1/report.md`.

## The one navigation seam (`costmap.py`)

The brief requires the obstruction passability layer to feed navigation **only** through
the costmap layer `sim-restock-dynobs-v1` owns — no parallel layer. That layer's contract
is `twin.navigation.dynamic.Detection(position, radius, source)`:

* `position` — footprint centre in the **map** frame (the runner maps a base-frame
  footprint into the map with the robot pose when the frame carries one; else the
  footprint is already map-frame).
* `radius` — the obstruction's own plan-view extent (half-diagonal of the footprint,
  `+ footprint_pad_m`). The robot clearance is **not** added here — the nav gate adds it.
* `source` — `route_obstruction` (configurable), for the evidence log only; it never
  relaxes a check.

`RouteAlertsRunner.latest_detections()` exposes the current map-frame detections for the
navigation runtime (dynobs-owned) to pass into `SensorNavigation.step(detections=…)`,
which **unions** their predicted occupancy into the sensed observation (add-only: a route
alert can turn a stop into motion, never the reverse). **The method is gated by
`mode`**: it returns `[]` in the working-default `findings` mode (nav byte-identical) and
the obstruction footprints only in `act` mode.

The live seam: `twin.route_alerts.nav_source.source_from_hook(hook)` turns the mission
hook into the `source(now) -> [Detection]` the dynobs layer pulls each navigate tick. In
`act` mode `isaac_bundle.create` wires it as the navigation `detection_source` — the SAME
seam the scenario / humans feed (`navigation.dynamic_obstacle_source`) uses, so the two
cannot both own it at once (composing them is a follow-up). `act` therefore requires
nav_mode `sensor` and `navigation.dynamic_obstacles` set (the dynobs layer); the bundle
refuses honestly otherwise rather than silently doing nothing.

## GDPR-by-design — people-masked, no biometrics

Findings are people-masked. `RouteFrame.agents` carries the people / cart detections to
mask out (store-vision's `agent_mask_from_detections` excludes them before any hazard is
extracted); the humans lane's (`sim-restock-humans-v1`) detection stream feeds straight
in. The live frame-builder **refuses to analyse a frame with no people-mask source**
(returns `None` every tick) rather than run an un-masked detector over people. No raw
frames or biometrics are persisted — only events, store-frame geometry and session-scoped
ids.

## Fixed-camera caveat (honest limitation)

store-vision's spill / obstruction build a clean-floor background + a persistence tracker
tuned for a **fixed** camera. On a moving robot they are soundest on the depth-protrusion
path (obstruction fits a floor plane per frame) and while the base is slowed / stopped at
a hazard (the nav gate already stops for obstacles, giving the tracker consistent frames).
Production uses the detector defaults (`persist_seconds` ≈ 6–8 s); a caller wanting faster
reaction on the drive sets `tuning: {"persist_seconds": …}` per detector. A
motion-compensated / single-shot mode is a store-vision proposal (see the report). The
camera→floor homography is treated as the rigid mount transform (config `floors`), and
mapping base-frame footprints into the map frame is done with the robot pose.

## Configuration (`route_alerts` section of the mission runtime-config)

```jsonc
{
  "route_alerts": {
    "spill":       { "enabled": true, "gpu": false },
    "obstruction": { "enabled": true, "gpu": false },
    "strays":      { "enabled": false, "gpu": true },   // needs a products recognizer
    "mode": "findings",           // "findings" (default, sensor-only) | "act" (feeds dynobs)
    "cadence": 1,                 // run every Nth offered frame
    "staff_tasks": true,
    "camera": "route",            // the drive camera id the floors calibration is keyed by
    "floors":  { "route": { "image_to_floor": [[...],[...],[...]], "zone_id": "aisle" } },
    "aisles":  { "aisle-1": { "x0": 0, "y0": 0, "x1": 2, "y1": 10, "axis": "y", "min_width_m": 0.9 } },
    "obstruction_source": "route_obstruction",
    "footprint_pad_m": 0.0
  }
}
```

Launcher: `--route-alerts` (the working default, `findings`), `--route-alerts=act` (the
smart variant), or `--route-alerts-config PATH` (a tuned section; a bare `--route-alerts=act`
beside it promotes that section to `act`). Live people-masking additionally needs
`config["agents_factory"]` wired to the humans lane's detections. For `act` the mission
must also run nav_mode `sensor` with `navigation.dynamic_obstacles` set (the dynobs layer).

## Recorded-frames bridge (replay a saved run, CPU, no simulator)

`twin.route_alerts.recorded` drives the identical runner over a **saved run's** recorded
camera frames instead of a live port: `load_frame_index` / `iter_recorded_frames` read
`<run>/mission-frames/frames.json` + the PNGs and yield `(rgb, simulation_time, seq)`
stamped with the run's **real** mission timestamps. This is the primary CPU evidence path
(no GPU session). The recorded `mission-frames` are the cinematic chase-camera RGB (no
depth, uncalibrated), so an *evidence replay* seeds the hazards — `seed_spill` composites
a blob onto the real pixels, `seed_obstruction` supplies a synthetic depth protrusion, and
`floor_calibration` a synthetic image→floor homography — while the detectors, the
event/task publishing, the mode gate and the dynobs layer are the identical production
code. A faithful mobile floor-facing RGB-D camera is the integration lane's follow-up (see
the v1 report's option A).

## Tests and evidence

* CPU tests: `scripts/twin/route_alerts/tests/` (pure-logic tests run anywhere; the
  integration tests need store-vision — run under its venv). RUNBOOK §19.
* CPU evidence (recorded frames, the primary v2 path):
  `scripts/run_route_alerts_recorded.py --run <saved run> --out <dir>` — seeds a spill +
  obstruction into a saved run's recorded frames and drives the runner in **both** modes,
  asserting detection with the run's real timestamps, that `findings` feeds navigation
  nothing (byte-identical) and that `act` feeds the obstruction into the dynobs layer and
  raises its cost. Self-asserting; run under the store-vision venv.
* CPU evidence (synthetic aisle): `scripts/run_route_alerts_evidence.py` — seeds a spill +
  obstruction (+ a moving shopper) into store-vision synthetic aisle frames, drives the
  **real** runner (detect: spill + obstruction + aisle-blocked + robot events + staff tasks
  + people masked), then drives the **real** coupled controller through the **real** dynobs
  layer so it stops for and re-routes around the obstruction the detector found.
* Live in-Isaac smoke: `scripts/run_route_alerts_smoke.py` (v1).
