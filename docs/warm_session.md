# Warm-session reset (dev iteration)

A cold run of the Isaac restock mission pays a FIXED ~215 s boot (Kit process start + stage load +
host/planner/mapper launch + first ESDF/frame readiness) BEFORE a single mission frame renders —
measured 211–222 s across runs (`data/sim-warm-session-reset-v1/boot-vs-mission.txt`), ~8.5–10.5 % of a
full ~42 min cycle but ~42 % of a short fast-failing debug run, and it recurs on EVERY run. With
stage-isolated runs (`--start-at`/`--stop-after`, PR 284/297) now the main iteration tool, a real warm
reset multiplies every proof: the next mission starts in the time of a stage re-seat, not a full boot.

A **warm session** keeps the live `SimulationApp` + the loaded USD stage (the expensive ~205 s) up
between missions and re-seats the stage to its post-boot baseline, then runs the next mission on the
same session.

## Pieces

| Piece | Where | Role |
|-------|-------|------|
| Safety core | `twin/restock/warm_session.py` | Pure, CPU-only. The conservative reuse-vs-cold-boot **decision** (`decide_reset`), the fail-closed reset **sequence** runner, and the combined `plan_and_run_reset` gate. Names the reason either way. |
| **Provider** | `twin/restock/warm_reset_provider.py` | The `config['warm_reset']` `module:function` the launcher drives: `build(...)` → a `WarmResetProvider` that is callable as `reset(attempt=, now=)` (runs the ordered in-session reset) and exposes `stage_census()`. |
| Launcher loop | `scripts/run_mobile_restock.py` | `run_consecutive_warm` drives N missions on one session; `warm_run_count` routes `--warm-attempts`/`--consecutive-runs`. `bundle.warm_reset` + `bundle.warm_stage_state` carry the provider. |
| Dry-run | `scripts/check_warm_session_reset.py` | Offline proof of the decision + sequence (and, with `--provider`, the real provider orchestration) against a recorded state fixture. |

## The reset sequence (provider `__call__`)

Each step **raises** `WarmResetRefused` to refuse; a refusal is a cold-boot fallback with the step
named (the captain's rule: a reset in doubt falls back to a cold boot and says why — never a silent
half-reset).

1. **`stop_runtime`** — stop the mission runtime + host stack cleanly, WITHOUT exiting Kit (never
   `app.close()`). The live app + loaded stage survive.
2. **`reset_stage`** — prune the mission-spawned prims (anything present now that was not in the
   post-boot baseline census: the carton, welds, markers, a respawned cart), then `world.reset()` to
   re-seat every registered body to its post-boot default pose AND reset the sim clock to t=0.
3. **`reseat`** — re-establish + verify the next run's start precondition: for a `--start-at` run,
   re-run `shortcut.apply_start_state` (which re-seats robot/cart/carton and runs PR 297's sim-time
   settle + rest/heading verify, raising on a mis-seated train — `START_STATE_UNSUPPORTED` /
   `START_STATE_SETTLE_MISMATCH`); for a cold dock-aligned run, `shortcut.verify_dock_seat`; otherwise a
   no-op (`world.reset` already re-seated).
4. **`relaunch_host`** — re-launch / re-sync the host stack (planner / mapper) so it serves the fresh
   stage and leaves `bundle.adapter` ready for the next `Mission`. **Integration-lane-owned** (the host
   processes are not built in this repo), so it is delegated to a lane hook (see below). With no hook
   wired the provider refuses this step → cold boot, reason named.
5. **`verify`** — the live app is still running and an optional lane readiness hook passes. (The
   launcher runs the runtime-input `preflight_verify` as its own next step, so the brief's "existing
   preflight gates + PR 297 resting/heading verify" is covered across the provider + the loop.)

## Fallback rules (honoured; any doubt → cold boot, reason named)

`run_consecutive_warm`'s `_between` builds the gate from the live session and consults
`warm_session.plan_and_run_reset`, which cold-boots (stops the warm loop, records `warm_fallback` in
`<output>/warm-session.json`) when ANY of:

- **HOST / boot-fixed key changed** between runs (`config_preflight.KEY_CONSUMER` HOST keys +
  `warm_session.BOOT_FIXED_KEYS`: scenario, host_config, package_roots, commissioning_dir,
  render_width/height, low_vram_render) — the live host stack + Kit cannot re-apply it. An unregistered
  changed key is conservatively a cold boot too.
- **Dirty stage / leaked / missing prims** — `bundle.warm_stage_state()` (the provider's `stage_census`)
  diffs the live prim set against the post-boot baseline; a mismatch or an explicit `dirty` flag refuses.
  A census that itself fails is reported as a dirty stage, never swallowed.
- **VRAM not returned** — live used VRAM more than `vram_tol_mib` (default 400) above the post-boot
  baseline sampled once before run 1.
- **A reset step refused** — any of the provider's five steps raising (including `relaunch_host` with no
  hook wired).

## Config lines (for the integration lane)

```jsonc
{
  // the production provider (exposes stage_census -> bundle.warm_stage_state):
  "warm_reset": "twin.restock.warm_reset_provider:build",
  // integration-lane host lifecycle hooks (host stack is not built in this repo):
  "warm_reset_hooks": "<your_module>:build_warm_hooks"
}
```

`warm_reset_hooks` is a `module:function` factory called `fn(app=, assembly=, parts=, plan=, config=,
output=)` returning an object/mapping with any of these callables (each raises to refuse):

- `stop_runtime()` — tear down the host stack cleanly without exiting Kit (else the provider falls back
  to `adapter.stop()` and only asserts Kit stayed up);
- `relaunch_host(attempt=, now=)` **or** `reset_host_state(attempt=, now=)` — one of these MUST be
  supplied, or `relaunch_host` refuses → cold boot. It must leave `bundle.adapter` ready for the next
  `Mission` (planning scene cleared of `restock_carried_unit`, nvblox ESDF re-initialised, station acks
  fresh);
- `verify(now=)` — optional final readiness gate.

See `docs/examples/warm-session-mission-fragment.json`.

## Running it

```bash
# N consecutive collect-only missions on ONE session (each in <output>/run-K, fresh seeded inventory):
run_mobile_restock.py ... --start-at collect --stop-after collect --consecutive-runs 2
# equivalently, once a provider is wired:
run_mobile_restock.py ... --start-at collect --stop-after collect --warm-attempts 2
```

`--warm-attempts N` with a `warm_reset` provider wired means N consecutive independent missions (the
fix for the stop-at-first-success retry semantics the integration lane hit — its three-run collect
acceptance ran as three cold boots because there was no provider to reset the stage in place). Without a
provider, `--warm-attempts` keeps its legacy retry-a-failed-attempt meaning (stops on the first
complete run). `--consecutive-runs` is the explicit N-independent-missions flag and works on its own.
Both are default 1 ⇒ a single run, byte-identical.

## Acceptance (integration lane, GPU)

1. Two consecutive **collect-only** missions on one session (`--start-at collect --stop-after collect
   --warm-attempts 2`) with `warm_reset` + `warm_reset_hooks` wired.
2. Both runs end with **identical stage verdicts** (`run-1/mission.json` and `run-2/mission.json`
   collect verdict match).
3. The second run **starts within a stated time well under 60 s** (vs ~215 s cold) — timed from the end
   of run 1's teardown to run 2's first mission tick; `<output>/warm-session.json` records
   `reused: true` for the between-runs gate and the reset `wall_seconds`.
4. **VRAM baseline restored** — the `vram_returned` check passes between the runs (used within 400 MiB
   of the post-boot baseline logged as `WARM_SESSION ... post_boot_vram_mib=`).

If any fallback rule trips, `warm-session.json` carries `warm_fallback` with the named reason and run 2
cold-boots in a fresh process instead — that is the correct, safe outcome, not a failure of the lane.

## CPU proof

```bash
python3 scripts/check_warm_session_reset.py --state scripts/twin/tests/fixtures/warm_session_state.json
python3 scripts/check_warm_session_reset.py --state scripts/twin/tests/fixtures/warm_session_state.json --provider --inject-fault relaunch_host --expect cold
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -m pytest \
  scripts/twin/tests/test_warm_session.py \
  scripts/twin/tests/test_restock_warm_session.py \
  scripts/twin/tests/test_restock_warm_reset_provider.py -q
```

## Follow-ups (out of scope here)

- The `warm_reset_hooks` host relaunch/re-sync implementation (integration-lane-owned); until wired,
  warm reuse cold-boot-falls-back at `relaunch_host`.
- Camera/stations re-commission on reset if a baseline lifetime expires mid-session (the mission's
  `warm_reset` config['warm_reset'] lifetime guard already refuses an expired baseline).
- Measured boot-vs-warm timing evidence from the GPU acceptance run.
