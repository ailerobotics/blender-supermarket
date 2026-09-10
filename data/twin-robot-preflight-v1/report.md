# Shelf robot URDF — offline Isaac pre-flight (twin-robot-preflight-v1)

**Question:** when the captain boots Isaac Sim to stand the shelfbot UR10 in the store,
will the **robot asset** import and behave — and if not, exactly what stops it? This is
the last of the three robot-in-store links never checked without a GPU (the store was
proven in PR #2, the composer in PR #3).

**Answer:** **Yes — after one tooling fix, which this PR makes.** The robot description
parses into one clean kinematic tree, all masses/inertias/joint-limits are sane, its
collision model is correct, and it fits its aisle spawn with real margin. The **one
load-bearing defect was in this repo's tooling, not the URDF**: the compose wrapper put
package `shelfbot` on the ROS package path as `~/workspaces/shelfbot`, but the gripper
meshes the URDF references live only in `~/workspaces/shelfbot_newgripper`. Isaac's URDF
importer *imports even when a mesh is not found* (it logs an error and continues), so this
would have **silently** produced a robot with the gripper geometry missing on the captain's
first boot. Fixed here; proven offline; guarded by a new pre-flight.

Verified entirely offline (no Isaac, no GPU) with pure `xml`+`numpy` for the robot side and
`usd-core` for the composition side, safe beside the live Blender session. Robot asset:
`~/workspaces/shelfbot_newgripper/description/shelfbot_gripper_isaac.urdf` (read-only — it
lives in another repo and was never edited).

---

## The load-bearing defect (fixed in this repo)

`shelfbot_gripper_isaac.urdf` references its two gripper meshes by ROS package:

- `package://shelfbot/description/gripper/meshes/cube_without_surface_plates_visual.obj`
- `package://shelfbot/description/gripper/meshes/suction_cups_visual.obj`

Two different repos on this box **both declare ROS package name `shelfbot`** in their
`package.xml`:

| repo | `package.xml` `<name>` | has `description/gripper/meshes/*.obj`? |
|---|---|---|
| `~/workspaces/shelfbot` (older robot repo) | `shelfbot` | **no** |
| `~/workspaces/shelfbot_newgripper` | `shelfbot` | **yes** |

The compose wrapper (`scripts/compose_robot_in_store.sh`) and every hint/doc pointed package
`shelfbot` at `~/workspaces/shelfbot`. So `package://shelfbot/…gripper…` did **not** resolve
to the actual mesh files. The 14 UR arm meshes (`package://ur_description/…` →
`/opt/ros/jazzy/share/ur_description`) were always fine; only the 2 gripper OBJs dangled.

Because the importer imports anyway (NVIDIA's own changelog: *"URDF gets imported even if
meshes are not found"*), the failure mode is **silent**: the arm imports, the gripper's box
colliders survive (they are URDF primitives, not the missing OBJs), but the gripper's visual
geometry is absent — the captain's screenshot would show an arm with no gripper, and nothing
would obviously say why.

**Fix (this repo only — the URDF is correct and untouched):**
- `scripts/compose_robot_in_store.sh` now points package `shelfbot` at
  `~/workspaces/shelfbot_newgripper` via a small ephemeral symlink shim
  (`<tmp>/shelfbot` and `<tmp>/share/shelfbot`), so `package://shelfbot/…` resolves under
  every resolver model (`ROS_PACKAGE_PATH` `entry/pkg`, crawl-by-`package.xml`-name, and
  ament `share/`). Nothing is written to the repo.
- The stale hints in `scripts/twin/cli.py` (dry-run) and `scripts/compose_robot_in_store.py`
  (`_import_urdf_to_usd` docstring), plus `RUNBOOK.md` §8 and `docs/isaac_robot_in_store.md`,
  now name the right repo and the ambiguity.

Confirmed offline: with the shim, all 16 mesh references resolve; the gripper OBJs resolve
under `entry/pkg`, ament `share/`, and crawl-by-name.

---

## What else was checked (all pass)

Read from the URDF with `scripts/twin/urdf_probe.py`; asserted by `scripts/validate_robot_urdf.py`.

| Check | Result |
|---|---|
| Parses; one connected kinematic tree | **26 links, 25 joints, single root `world`** (13 fixed, 6 revolute, 6 prismatic) |
| Every mesh reference resolves | **16/16** (14 UR + 2 gripper), no absolute build-machine paths |
| Dynamic links have positive mass | **12/12** (7.1, 12.7, 4.27, 2.0, 2.0, 0.365 kg arm; 0.1 kg ×6 tines) |
| Inertias positive-definite + triangle inequality | **all valid** (2 placeholders flagged — see below) |
| No geometry link left without inertia | **pass** (the 9 massless links are all pure fixed frames) |
| Movable joints have limits + effort/velocity | **12/12** (revolute ±2π except elbow 0.1–π; tines 0–0.2 m) |
| Collider ≠ full visual mesh | **pass** — UR links use dedicated collision STLs; gripper/pedestal/tines use box primitives |
| Coordinate convention / units | **Z-up, metres** — matches the store (Z-up, 1 m/unit) |

## Store consistency — real numbers

| Fact | Value |
|---|---|
| URDF pedestal-mount offset (world→`pedestal_link` XY) | **(0, −1.2)** — matches composer `geometry.PEDESTAL_OFFSET_FROM_ROOT` exactly |
| Pedestal ground footprint | **0.2 × 0.4 m** (box, bottom on Z=0) — matches the store pre-flight's assumption |
| Home pose (ros2_control `initial_value`) | pan 0, lift −2.5, elbow 1.7, wrist1 −0.7, wrist2 −1.5708, wrist3 0 (tucked) |
| Home-pose collision envelope (robot-local) | X [−0.552, 0.100], Y [−1.400, −0.891], Z [0.000, 1.656] m |
| Arm reach **across the aisle** (X, the tight axis) | **0.552 m** from the pedestal centre |
| Arm **planar radius** from pedestal centre | **0.591 m** |
| Aisle it spawns in (G3–G4, X = −1.525) | **1.5 m wide** → 0.75 m half-width to each shelf face |
| Clearance to the shelves at the home pose | **L ≈ 0.198 m, R ≈ 0.65 m** (both positive → fits) |
| Store planar clearance at the spawn | 0.95 m (arm's 0.591 m radius is inside it) |

So the arm's tucked home pose **does** extend beyond the 0.2 × 0.4 m pedestal footprint
(0.552 m across-aisle vs 0.1 m), but it stays inside the 1.5 m aisle with ~0.20 m to spare on
the near side — it does **not** penetrate a shelf at yaw 0. Robot top is 1.656 m (well under
the store ceiling).

---

## One upstream data-quality note (not a blocker; not fixable here)

The gripper links `cube_without_surface_plates` and `suction_cups` each carry a **placeholder
inertia** `ixx=iyy=izz=0.016667` for a 0.1 kg body — a radius of gyration of ~0.41 m on a
~0.29 m box (physically implausible; the value is 1/60, a clear default). Both are rigidly
attached by **fixed** joints, so PhysX will not destabilise, but they inflate the composed
end-effector inertia. The fix is in the robot description
(`shelfbot_gripper_isaac.urdf`, the `<inertial>` of those two links, ~lines 373–429), which
lives in another repo — so it is **reported, not edited**. `validate_robot_urdf.py` emits it as
a WARN so it is not forgotten; firstmate can route the URDF change separately.

---

## How to re-run

```bash
cd ~/workspaces/blender-supermarket
.venv-usd/bin/python scripts/validate_robot_urdf.py            # robot only; exit 0 = pass
.venv-usd/bin/python scripts/preflight.py                      # store + robot + composer; exit 0 = boot
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-usd/bin/python -m pytest scripts/twin/tests/           # 64 tests
```

## What still needs the GPU boot

Only the simulator itself: the *execution* of the URDF→USD import (its inputs are now proven —
URDF parses, every mesh resolves), PhysX loading the colliders, the robot settling under gravity
without penetration once stepped, and the screenshot. Everything the simulator consumes — the
composed store stage **and** the robot description — is now proven offline.
