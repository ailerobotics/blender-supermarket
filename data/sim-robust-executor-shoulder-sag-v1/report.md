# Executor loaded-pick shoulder_lift sag: load-aware gravity feedforward

**Lane:** sim-robust-executor-shoulder-sag-v1 · branch `fm/twin-robot-in-store-boot-v1`
**Symptom (execretest, PR 214, 2026-10-23):** the moving-arm transfer with the host trajectory executor
(`RESTOCK_SCRIPTED_TRANSFER_ARM=executor`) was down to one joint. The pick is a 126-pt clear-branch
trajectory, the arm follows, and 5 of 6 joints reach the goal to ~1e-5 rad. The one miss is **joint 1,
`ur_shoulder_lift`**: it settles **0.098 rad (~5.6°) short** of the loaded pick goal with velocity ~0
after the full 30 s settle window:

```
TRAJ_HOST_GOAL_MISS pos_err=0.0981 vel=0.0069 settle=30.01
  per_joint_err=[8e-5, 0.0981, 5.7e-5, 1.8e-5, -9e-6, -3.4e-5]
  meas_q=[0.3482,-3.5491, 2.2172,-1.7122, 1.2201, 3.1418]
  goal_q=[0.3481,-3.6472, 2.2171,-1.7122, 1.2201, 3.1418]
-> TRAJ_HOST_FAULT measured goal tolerance not reached
```

Run dir: `~/datasets/twin-robot-in-store-boot-v1/mission/execretest/`
(`host-stack.log`, `execretest.console.log`, `scripted-transfer/scripted-arm-hold.json` = the failing
126-pt plan, `assembly.json`). Drive config read back from the Isaac boot: `kp=3e5`, `kd=8000`,
`max_effort=400` on all arm joints; gravity FF **on** (`arm_gravity_ff` default True).

## Step 1 — cause, established by CPU static torque balance

The `trajectory_host` executor only interpolates/streams position+velocity targets and checks the goal;
it applies **no** gravity feedforward. The FF is in `physics_setup.assemble` (`build_arm_gravity` +
`apply`), computed on CPU from the URDF tree + `_ISAAC_BODY_MASSES` and fed as `-G(q)` joint effort.
Reproduced on CPU (`scripts/twin/tests/test_restock_shoulder_sag_ff.py`, URDF
`~/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf`; the CPU model matches the sim's own
`ARM_GRAVITY_FF` log at the start pose — same code):

| scheme | shoulder_lift residual net torque | predicted steady-state offset | verdict |
|---|---|---|---|
| **FF@commanded, no carton (shipping)** | +0.994 Nm | **−0.10 rad** | reproduces the observed +0.098 |
| FF@commanded + carton | +4.51 Nm | −0.46 rad | worse |
| FF@measured, no carton | −3.34 Nm | +0.34 rad | worse |
| **FF@measured + carton (fix)** | **0.000 Nm** | **0.000 rad** | PASS |

Two independent errors combine:

1. **The FF is evaluated at the COMMANDED (goal) arm config, not the measured one.** Gravity acts where
   the arm physically *is*; during the final descent / terminal hold the commanded config leads the
   measured one, so the FF (anchored at the goal) over-compensates the actual gravity at the lagging
   measured pose by ~+4.3 Nm.
2. **The 0.45 kg welded carton is omitted from the FF mass model** (`_ISAAC_BODY_MASSES` has no payload),
   leaving ~−3.5 Nm uncompensated.

These nearly cancel to a ~+1 Nm residual the drive must hold. The decisive asymmetry: the drive supplies
that ~1 Nm at a **0.098 rad** error on shoulder_lift, while the identically-configured elbow supplies
9.5 Nm at **1e-4 rad** — an effective torque-per-rad of ~10 Nm/rad for shoulder_lift vs ~95000 for the
elbow (back-calculated from the run; the exact 0.098 reproduction confirms it). So shoulder_lift is the
only joint soft enough that a ~1 Nm residual shows as a tolerance-breaking miss.

This **rules out** the task's hypothesis set as stated: it is *not* a missing/undersized carton FF alone
(adding the carton to the shipping commanded-pose FF makes it **worse**, −0.46 rad), and it is *not* a
soft-joint measurement artifact (the joint genuinely rests there, v≈0, zero `TRAJ_HOST_TRACK_MISS` all
move). It is the **FF evaluation pose + payload omission together**, exposed on the softest joint.

## Step 2 — fix (make the arm reach; tolerance unchanged)

`gravity_ff_effort` (new pure helper in `physics_setup.py`, CPU-tested) holds the FF logic:

- when `arm_gravity_ff_loaded_carry` is on, evaluate robot gravity at the **measured** arm config, and
- add the carried carton's gravity (`build_payload_gravity`, a point mass at `gripper_pad_tcp`) **only
  while the gripper is held** (gated on the mission's canonical `NativePanelContact.read`, which honours
  the shared stage weld, so a welded-carry transfer reads held throughout).

Together these drive the terminal net torque to ~0 so the loaded pick **reaches** its goal on every joint
— the `trajectory_host` goal tolerance stays tight at **0.010 rad** (no loosening; the terminal check is
now the pure, tested `goal_reached` predicate). Each correction alone is worse, so they ship as one key.

### Backward compatibility (byte-identical when off)
- `arm_gravity_ff_loaded_carry` default **False** ⇒ `apply()` evaluates the FF at the commanded `q` with
  no payload, exactly as before. The payload held-gate holder stays `None`.
- payload mass defaults to the scenario `handling_unit.mass_kg`; `arm_gravity_ff_payload_kg` /
  `arm_gravity_ff_payload_link` are optional overrides.
- the scripted/**still** path and every non-loaded phase are unchanged.
- all three keys registered RUNTIME in `config_preflight.KEY_CONSUMER` (a mis-placed key fails the
  launch loudly, per the host/runtime split).

### CPU tests
- `test_restock_shoulder_sag_ff.py` — (a) pure `gravity_ff_effort` logic (pose selection, held-gated
  payload, threshold, sign; no Isaac/URDF); (b) the execretest equilibrium: legacy FF reproduces the
  0.098 rad miss and fails the 0.010 tol; each correction alone still fails; measured-pose + held payload
  drives net→0 and passes (skips if the machine URDF is absent).
- `test_restock_trajectory_host.py::test_goal_reached_*` — the terminal `goal_reached` predicate keeps
  the tight tolerance (the 0.098 miss still fails; a reaching arm passes; stale/moving rejected).

Full suite (`.venv-usd`): the changed-module tests all pass; the only failures are the pre-existing
`test_restock_sensed_views` wall-clock QR-decode budget flake (fails identically on the clean tree
without these changes) and nothing in the gravity/executor path.

## Step 3 — bounded GPU proof (hand to the integration lane)

The transfer1 executor staging is **unchanged** except the one new key. Fragment:
`docs/examples/shoulder-sag-ff-mission-fragment.json` — merge `"arm_gravity_ff_loaded_carry": true` into
the transfer1 executor **runtime** config (the one with `arm_stiffness_abs` etc.), keep everything else.
PASS = the executor pick reaches its goal (no `TRAJ_HOST_GOAL_MISS` on `ur_shoulder_lift`, no
`TRAJ_HOST_FAULT`) and the moving-arm transfer proceeds past the pick. The mission runtime configs live
outside the repo (integration-lane owned) and were **not** edited here.

**GPU proof status: PENDING** — handed to the integration lane via the config fragment (no GPU run taken
in this lane; the mission boot owns GPU priority + commissioning + host stack).
