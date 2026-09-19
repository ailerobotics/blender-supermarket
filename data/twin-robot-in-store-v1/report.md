# Twin robot-in-store v1 — first real Isaac Sim run

**Status: PREPARED, BOOT PENDING OWNER APPROVAL.** Everything that needs no simulator
boot is done and green; the Isaac Sim boot itself is gated on the repository owner's
explicit approval token and has **not** been performed. This report is filled with the
offline evidence now; the boot-only sections (peak GPU/RAM, boot time, screenshot, the
stepped settle verdict) are marked **PENDING** and are completed after the approved run.

Goal (RUNBOOK §8): stand the fleet's UR10 (`shelfbot_newgripper`) in the supermarket
twin under Isaac Sim 5.1 (`env_isaaclab`) — import the URDF to USD, author physics +
colliders, place the robot in an aisle, step physics, and leave evidence (composed
stage + aisle-camera screenshot + a base-height log) that it rests on the floor.

## What ran (no boot)

| Step | Result |
|---|---|
| Contract docs read | `AGENTS.md`, RUNBOOK §8, `docs/isaac_robot_in_store.md` |
| Export staleness | **Stale** — on-disk `export/supermarket.usdc` was Sep 14, predating the realism pass (`a59aa4d`, in HEAD) that changed `build_supermarket.py`. |
| Export regenerated (headless, no GPU) | `blender -b … build_supermarket.py -- --all --save scenes/supermarket.blend` (25 963 objects, 2.8 s) → `blender -b … export_usd.py` (5.0 MB USD, 23.7 s). Old export backed up to `/tmp/twin-robot-in-store-v1-backup/`. |
| Pre-flight (`scripts/preflight.py`) | **PASS (exit 0)** — store + robot + composer all green, on the fresh export. |
| Offline tests | `scripts/twin/tests/` — **77 passed** (incl. new `test_settle.py`, 10 cases). |

### Store facts (measured from the fresh export, no GPU)
- Up axis **Z**, **1.0** m/unit, default prim `/Supermarket`, floor at **Z = 0.0** (26 × 22 m).
- Colliders the composer authors: **702** triangle-mesh (static structure/fixtures/signage)
  + **197** convex-hull on shared product/tag source meshes → **196/196** product
  prototypes inherit (24 927 instances, authored once each). PhysicsScene = Z-gravity 9.81.
- Composed stage ≈ **54 068** prims (instancing preserved).
- The regenerated export no longer carries the 6 dangling floor/wall textures (the fixed
  `export_usd.py` authored resolvable paths), so the pre-flight is clean **without** the
  in-place texture repair the stale artifact needed.

### Robot facts (measured, no GPU)
- URDF parses to one kinematic tree (26 links / 25 joints, root `world`); **all 16 mesh
  references resolve** (14 UR → `/opt/ros/jazzy/share/ur_description`, 2 gripper OBJs →
  `~/workspaces/shelfbot_newgripper`).
- Spawn: pedestal at **(−1.525, 0.75, 0.0)** in the G3–G4 aisle (1.5 m wide, 0.95 m
  clearance); robot root xform translate **(−1.525, 1.95, 0.0)** (compensates the URDF
  pedestal offset (0, −1.2)). Home-pose arm envelope fits the aisle (0.198 m to the near
  shelf). Standing upstream WARN: placeholder inertias on the two gripper links (a
  fixed-joint end-effector detail; report-only, lives in another repo).

## Readiness at report time (measured)

| Check | Required | Measured | OK |
|---|---|---|---|
| No Blender process / `:9876` socket | none | none | ✅ |
| Free GPU memory (`nvidia-smi`) | ≥ 8 GB | **9603 MiB free** of 12288 | ✅ |
| Available RAM (`free -g`) | ≥ 12 GB | **27 GiB** available | ✅ |

**GPU-guard caveat (folded into the boot decision):** a protected always-on service —
`whispr` (`~/whispr/server.py`, pid 1334, **2184 MiB** VRAM) — sits above the guard's
default heavy-consumer threshold (1500 MiB), so the guard **refuses** the boot as-is
(confirmed by `--dry-run`). whispr must not be touched and will not free the GPU. The
bounded run wrapper therefore passes **`--heavy-threshold-mb 2500`** (names whispr as a
known-stable consumer) — this is **not** `--force`, and the real OOM protection (free
VRAM ≥ 8 GB) is unchanged and satisfied. The hard Blender gate is never overridden.

## The one command that reproduces the run

```bash
cd ~/workspaces/blender-supermarket      # (or this worktree)
bash scripts/run_robot_in_store_gpu.sh
```

`scripts/run_robot_in_store_gpu.sh` (new) re-checks readiness (no Blender, ≥8 GB VRAM,
≥12 GB RAM) and, only if clean, launches `scripts/compose_robot_in_store.sh` inside its
own process group with:
- a **hard 20-minute wall-clock cap** (`TIMEOUT_S`, default 1200 s);
- a **RAM watchdog** that stops *this run only* (its own group, by PID) if available RAM
  drops below **4 GB** (`RAM_FLOOR_GB`);
- a **peak GPU/RAM sampler** → `run_metrics.txt`;
- `--heavy-threshold-mb 2500` and `--store <this repo's export>` (no `--force`).

Outputs land in `data/twin-robot-in-store-v1/`: `store_with_robot.usd` (stage),
`store_with_robot.png` (aisle-camera screenshot), `store_with_robot_base_height.csv`
+ `store_with_robot_settle.json` (base-height series + verdict), `isaac_run.log`,
`run_metrics.txt`. Large/machine-local artifacts (`.usd`, logs, metrics) are gitignored;
the screenshot + settle CSV/JSON + this report are committed as portable evidence.

## What had to be fixed (for a reproducible one command)

1. **`scripts/compose_robot_in_store.sh`** aborted under `set -euo pipefail` when it
   sourced `/opt/ros/jazzy/setup.bash` (ROS setup scripts reference unbound vars; under
   `set -u` that kills the whole shell before `|| true` can catch it). Wrapped the ROS
   source in `set +u … set -u`. Without this the URDF import would have run with no ROS
   package path (gripper meshes silently missing). *Offline-verified: wrapper `--dry-run`
   now exits 0.*
2. **`scripts/compose_robot_in_store.py`** produced no base-height evidence — it stepped
   physics blindly. Added a sampling loop that records the robot's world bbox (pedestal
   foot Z, footprint centre XY, Z-extent) every N steps, writes a CSV + JSON, and runs a
   pure verdict (`twin/settle.py`): rests-on-floor / fell-through / drifted / exploded.
   The real run now exits non-zero if the robot does **not** rest on the floor. The
   screenshot now binds the viewport to the aisle camera (`Cam_Aisle`) before capture.
   *The pure verdict logic is unit-tested (`test_settle.py`); the sampling/screenshot
   themselves are simulator-only and stay UNVERIFIED until the boot.*
3. Default `--steps` raised 60 → 180 (3 s at 60 Hz) so the arm settles from its home pose.

## Boot-only results — PENDING (completed after the approved run)

- [ ] Time to boot Isaac Sim: _pending_
- [ ] Peak GPU / RAM during the run: _pending_ (from `run_metrics.txt`)
- [ ] URDF→USD import produced the expected articulation (arm + gripper visible): _pending_
- [ ] PhysX loaded the 702 + 197 colliders: _pending_
- [ ] Base-height settle verdict (rests on floor, no fall-through/drift/explode): _pending_
- [ ] Aisle-camera screenshot: _pending_ → `store_with_robot.png`
- [ ] Anything the real simulator revealed + the fix: _pending_

## Out of scope (follow-ups only)

Robot motion planning, manipulation/grasping, cuMotion, ROS bridges, Gaussian-splat
backdrops, per-product rigid bodies, and any change to the store's look. The two
gripper placeholder inertias are an upstream URDF data-quality fix (another repo).
The captain's longer-term intent (Gaussian-splat captured store + robot manipulation)
builds on this bring-up but is explicitly later work.
