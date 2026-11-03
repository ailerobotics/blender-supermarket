# Mission stages, isolation & the per-stage audit

The Isaac restock mission is a fixed in-order prefix of **seven stages**
(`twin.restock.mission.ACTIONS`):

```
collect → couple → outbound → park → transfer → verify → return
```

`collect` picks the carton from the backstock and loads it on the rollwagen; `couple` latches the
tow hitch; `outbound` tows the coupled train to the transfer bay; `park` stationarises it;
`transfer` moves the carton from the cart onto the bay; `verify` commits the stock; `return` tows
the cart to its parking spot, uncouples, and drives home empty. The cart stays coupled through
park/transfer/verify and uncouples at the **start of return** (the uncouple is a return step, never
a park step).

This doc is for driving a **single stage in isolation** on a dev bench (to find which stage
misbehaves), and pairs with the per-stage audit tool below. It is CPU-only; no GPU run is launched
from here.

## Entry / exit points

Two orthogonal dev-bench knobs bound a run (`run_mobile_restock.py` flags, also runtime-config
keys of the same name; the bench run is labelled in `mission.json` and by a burned-in walkthrough
caption so it can never pass for the full seven-stage cycle):

- **`--stop-after <stage>`** cuts `config['stages']` to the in-order prefix ending at that stage
  (`twin.restock.mission.bounded_stages`). It is supported for **every** stage — including `verify`
  and `return` — so the END of a single-stage run is never the limiting factor.
- **`--start-at <stage>`** begins the real state machine at a later stage, **synthesising** the
  skipped prefix's physical end state out of band (`twin.restock.shortcut`) and recording each
  skipped stage honestly as a `<stage>_shortcut` event (never a performed success). `start_at`
  **cannot skip `verify`** (the only stock-commit gate — `Mission` refuses any start_at *after* it,
  so inventory is never left uncredited; this safety bound is intact). Supported start points are
  `shortcut._SUPPORTED_START_AT = ('couple', 'outbound', 'park', 'transfer', 'verify')`. Every one
  except `couple` spawns the coupled, loaded train at that stage's sanctioned build-time pose via a
  **staging preset** (`--start-staging`), one preset per stage:

  | `--start-at` | `--start-staging` | base spawn pose (derived from) | carton |
  |---|---|---|---|
  | `couple`   | *(none)*            | the dock-aligned couple **seat** derived from the active outbound lane `couple_dock` (else the scenario spawn default) | resting on cart, **uncoupled** |
  | `outbound` | `dock_coupled`      | the coupled tow lane **start** (post-couple dock)      | resting level on cart, **coupled** |
  | `park`     | `bay_parked`        | the coupled tow lane **goal** (stationarise point)     | resting level on cart, **coupled** |
  | `transfer` | `recorded_bay_park` | the recorded park→transfer handover of a real tow       | resting level on cart, **coupled** |
  | `verify`   | `bay_placed`        | the transfer/verify park pose (`shelf_approach`)       | **placed level on the bay top**, arm empty, **coupled** |

  The `dock_coupled` / `bay_parked` base poses are **derived from the configured flow**
  (`navigation.tow_lanes` + `mission_poses`), so the train spawns exactly on the tow lane the live
  coupled drive follows (no off-lane spawn → no `lane_not_acquired`); the derivation follows whichever
  of the dock-align / shelf-staging flows the run is configured for. `bay_placed` places the carton on
  the configured bay target (`transfer_ground_truth.target` or the scenario `transfer_bay` top), exactly
  as a succeeded transfer leaves it.

  `couple` takes **no** staging preset, but when the active outbound lane declares a `couple_dock` its
  shortcut spawns the post-collect train at the dock-aligned couple **seat** —
  `shortcut.couple_start_poses` reuses PR 282's single source `navigation.lanes.couple_seat_from_cart`
  (the same inverse-hitch derivation the outbound lane start follows), for both the legacy EAST dock and
  the dock-align SOUTH dock — so `--start-at couple --stop-after outbound` seats the base ON the lane
  start and the real couple latch + outbound tow **acquire** the lane instead of seating ~1 m / 90° off
  (coldwin23-2 `lane_not_acquired_timeout`). The cart is left **uncoupled** (the real couple stage
  latches). A lane with no `couple_dock` (the backroom-bay bench) is **byte-identical** — the scenario
  spawn default stands.

To run **one stage alone**: `--start-at <stage> --start-staging <preset> --stop-after <stage>` (for
`collect`, the first stage, just `--stop-after collect`; `couple` needs no staging). **`return` in
isolation** is `--start-at verify --start-staging bay_placed --stop-after return`: `start_at` may not
skip the `verify` stock-commit gate, so the verify-entry (placed + coupled) state is synthesised and
the one instant ground-truth `verify` runs (committing stock honestly) before the real return tow.

## Per-stage isolation matrix

Preconditions describe the physical state each stage needs **at entry** — i.e. what its upstream
stages leave behind (seeded cart pose, carried unit, coupled state). This table is generated by
`twin.restock.shortcut.isolation_matrix()` (a test asserts it stays consistent with `ACTIONS`,
`_SUPPORTED_START_AT` and the staging presets, so it cannot silently drift):

| Stage | Can run alone? | Bench recipe | Entry precondition |
|---|---|---|---|
| `collect`  | **yes** | `--stop-after collect` | spawn defaults only: robot at the backstock pickup, cart at the dock, arm stowed, carton in the backstock (no carried unit, uncoupled) |
| `couple`   | **yes** | `--start-at couple --stop-after couple` | carton loaded on the cart (collect done); robot still at the pickup, cart at the dock, uncoupled |
| `outbound` | **yes** | `--start-at outbound --start-staging dock_coupled --stop-after outbound` | carton loaded AND the cart physically coupled behind the robot at the dock (couple done) |
| `park`     | **yes** | `--start-at park --start-staging bay_parked --stop-after park` | loaded + coupled, towed to the transfer bay (outbound done) |
| `transfer` | **yes** | `--start-at transfer --start-staging recorded_bay_park --stop-after transfer` | loaded + coupled + parked at the transfer bay, arm stowed (park done) |
| `verify`   | **yes** | `--start-at verify --start-staging bay_placed --stop-after verify` | the carton placed on the bay top, still coupled (transfer done) |
| `return`   | **yes** | `--start-at verify --start-staging bay_placed --stop-after return` | stock committed at verify, still coupled at the bay (verify done) |

### How each late-stage start is synthesised

`outbound`, `park` and `verify` now each have a start-phase shortcut, and `return` is reached via a
`verify` start without skipping the stock-commit gate:

- **`outbound` alone** / **`park` alone**: the train spawns **coupled + loaded** at the coupled tow
  lane **start** (outbound: the post-couple dock) or **goal** (park: the stationarise point). The base
  pose is derived from the configured `navigation.tow_lanes` so the spawn sits exactly on the lane the
  live coupled drive follows; `shortcut.apply_start_state` authors the real robot↔cart hitch and settles
  the carton on the cart. (If the configured flow has no tow lane, the synthesis fails closed rather than
  spawn at an invented dock pose.)
- **`verify` alone**: the carton spawns **placed level on the bay top** (the configured
  `transfer_ground_truth.target` / scenario `transfer_bay`), the arm empty and the cart still coupled —
  exactly as a succeeded transfer leaves it. The real `verify` stage then confirms the placement and
  commits stock.
- **`return` alone**: `start_at` may **not** skip the `verify` stock-commit gate (`Mission` refuses any
  start_at after it), so `return` is run as `--start-at verify … --stop-after return`: the verify-entry
  (placed + coupled) state is synthesised, the one instant ground-truth `verify` runs (committing stock
  honestly) and the real coupled return tow → uncouple → empty drive-home then follows. The safety bound
  is intact — stock is still committed exactly once, by the real `verify`.

Base poses follow whichever flow is configured (the dock-align / shelf-staging tow lanes drive the
`outbound`/`park` spawns). The **shelf-staging** sales-floor transfer/verify/return — an *uncoupled*
endcap service with the carton *carried* rather than resting on the cart — stays integration-owned per
`docs/examples/tow-lanes-mission-fragment.json`; the synthesis here targets the coupled backroom-bay
tail that `check_mission_tail` validates.

What this lane added, CPU-verified:

- `shortcut._SUPPORTED_START_AT` extended to `('couple', 'outbound', 'park', 'transfer', 'verify')`
  with a staging preset per stage (`dock_coupled` / `bay_parked` / `recorded_bay_park` / `bay_placed`),
  base poses derived from the configured tow lanes + `mission_poses`;
- `shortcut.plan_start_state` / `staged_scenario` / `apply_start_state` synthesise the carton on the
  cart (loaded starts) or on the bay (verify), coupling the cart at synthesis;
- `shortcut.stage_isolation(stage)` / `isolation_matrix()` now report **every** stage isolatable;
- tests (`test_restock_shortcut.py`, `test_restock_stage_start_synthesis.py`) asserting each
  synthesised state satisfies that stage's ACTIONS precondition, `Mission` accepts the start_at (and
  refuses one past `verify`), the spawn sits on the configured tow lane, and the preflight gates
  (`config_preflight`, `check_mission_tail`) still pass on the synthesised config.

### GPU acceptance (integration lane)

These starts are CPU-proved only; the integration lane owns the bounded GPU proof that each boots in
isolation and reaches its first action. With the backroom-bay tail config (the
`fixtures/mission_tail_bay.json` flow: `navigation.tow_lanes` + `mission_poses` + `transfer_bay` +
`verify_ground_truth`), run each on the GPU (one `/tmp/aile-gpu.lock` holder, never beside live Blender):

```
# outbound alone (reaches the first coupled tow NAV_STEP)
python3 scripts/run_mobile_restock.py --execute --runtime-config <cfg> \
    --start-at outbound --start-staging dock_coupled --stop-after outbound
# park alone (reaches the stationarise/park phase at the bay)
python3 scripts/run_mobile_restock.py --execute --runtime-config <cfg> \
    --start-at park --start-staging bay_parked --stop-after park
# verify alone (reaches the ground-truth verify confirmation + stock commit)
python3 scripts/run_mobile_restock.py --execute --runtime-config <cfg> \
    --start-at verify --start-staging bay_placed --stop-after verify
# return alone (reaches the coupled return tow → uncouple → empty home)
python3 scripts/run_mobile_restock.py --execute --runtime-config <cfg> \
    --start-at verify --start-staging bay_placed --stop-after return
```

Acceptance per stage: the shortcut prints `START_PHASE_SHORTCUT applied start_at=<stage> …
coupled=True`, the spawned train is coupled + at the expected pose, and the stage reaches its first
real action (outbound/park: first `NAV_STEP`; verify: first verifier confirmation; return: first
coupled return `NAV_STEP`) within a few sim-seconds of the first mission tick — not the ~45-min cold
cycle. Each run is labelled in `mission.json` (`start_at`, `shortcut`, the `start_staging` record) and
by the burned-in walkthrough caption, so a bench run can never pass for the full seven-stage cycle.

## The per-stage audit tool

`scripts/stage_audit.py` answers, for any finished run: **which stage failed, why, and what did it
look like.** CPU only; it only reads the run dir.

```
python3 scripts/stage_audit.py <run-dir> [<run-dir> …] --out <dir> [--fps 20] [--no-clips]
```

For every run it writes into `<dir>`:

- **`stage_audit.md`** / **`stage_audit.json`** — one row per stage: result
  (`PASS`/`FAIL`/`SKIPPED`), the **failure reason by name** from the durable record
  (`mission.json` `stage_timeline` + the inventory event log — the same record the stock commit
  uses), attempt count, the **sim-time** span (Isaac stage clock) and **wall-time** span
  (interpolated from `nav-trace.jsonl`'s `wall`), recorded frame count, and the key metrics the
  stage already logs in its evidence (e.g. `bottom_gap`, `coupled`, `arrived`, `outbound_brake_mode`).
- **`clips/<run>__<stage>.mp4`** — one short clip per stage cut from `mission-frames`, carrying the
  **same stage + sim-time overlay** as the whole-run video. The overlay + stage ladder are the
  shared `twin.demo.stage_video` module (lifted out of the integration lane's
  `make_whole_run_video.py`, now a thin wrapper over it — one drawing implementation, not a copy),
  so a clip's band matches the whole-run video frame for frame. Stages with no recorded frames
  (e.g. an instantaneous ground-truth `park`/`verify`) get a table row but no clip.
- **`index.html`** — a self-contained page (the tables + an inline `<video>` per clip) for the
  phone watch page.

Result mapping: `succeeded → PASS`; `failed` / `unavailable` (a provisioned-capability gap, which
the mission records FAILED) / `started`-but-never-finished → `FAIL`; a `start_at`-synthesised
prefix or a stage cut by `stop_after` / never reached → `SKIPPED` (the note column says which).

CPU test: `scripts/twin/demo/tests/test_stage_audit.py` (and `test_stage_video.py` for the overlay).
Run with `PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
scripts/twin/demo/tests/test_stage_audit.py scripts/twin/demo/tests/test_stage_video.py -q`.
