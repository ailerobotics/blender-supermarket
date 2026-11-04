# Fleet dashboard — watch the Isaac mission + the store (read-only, CPU, opt-in)

A phone-first, read-only panel that merges the **live Isaac restocking mission** with
**store-vision store state** into one view over the tailnet. It is strictly additive:
the dashboard the mission never imports or launches, so a mission run is
**byte-identical** whether or not the dashboard is up. No GPU.

It comes in **two shapes**, same aggregation, same page:

- **Live server** — a standalone CPU HTTP server (`:8794`) that fetches the watch page's
  `status.json` and recomputes the view on every request. Richer (live frame proxy, the
  opt-in approval POST path), needs a running process.
- **Static panel** — a zero-process `fleet.html` + `fleet.json` pair rendered *into a run
  directory* and served with nothing more than `python3 -m http.server` over that
  directory (the same pattern the firstmate watch page uses). The page polls `./fleet.json`
  relative to itself and links all media (newest frame, `walkthrough.mp4`, `bt_view.html`)
  **relative** to the run dir, so the whole panel serves over one static mount with no
  extra routes. This is the shape the watch page (`:8787`) can carry — see
  *Serving from the watch page* below.

- Module: `scripts/twin/dashboard/` (pure aggregation in `aggregate.py`; live I/O in
  `server.py`; the static generator in `snapshot.py` + its page in `page_static.py`).
- Launcher / dry-run / static render: `scripts/run_fleet_dashboard.py` (RUNBOOK §17).
- Tests: `scripts/twin/tests/test_fleet_dashboard.py` (no network, no GPU, no simulator;
  the static tests replay a saved run directory).

## What it shows

1. **Mission pulse** — from the existing firstmate watch page `status.json`
   (`http://<DASHBOARD_HOST>:8787/`, read-only): alive/idle, run name, frame count, the live
   trace (stage, tick, sim time, coupled/parked, base x/y).
2. **Mission pipeline** — the `reserved → collect → couple → outbound → park → transfer →
   verify → return` timeline for the resolved run, read from that run's
   `mission.json` events under
   `~/datasets/twin-robot-in-store-boot-v1/mission/<run>/`. The stage the
   run is sitting on reads *active* while running and *failed* if the run failed there (a
   recorded stage-entry event means "entered", not "completed"); verify's physics-verified
   verdict is surfaced on success.
3. **Media** — latest rendered frame and the run/stage walkthrough videos, **linked back**
   to the upstream watch page (never re-served here).
4. **Behaviour tree** — lights up automatically on the first `sim-restock-bt-v1`-enabled
   run (see below); otherwise the card reads "Not published by this run."
5. **Store state** — store-vision shelf fill / depletion / alerts, overlaid from a
   documented JSON export (`--store-state <path>`); absent ⇒ "Not configured."
6. **Restock requests** — an opt-in app trigger/monitor queue (see below).

## Reuse, not replacement

The dashboard **reads** the firstmate-owned watch page (`:8787`) rather than replacing it:
it consumes that page's `status.json` for the live pulse and links its media back to it. It
**reads** the mission run directories for per-run detail. It binds to the tailnet IP
`<DASHBOARD_HOST>` on a **free** port (default `8794`; `8787/8790/8791/8793` are taken) so it
coexists with the watch page. All media links point at the upstream base.

## Serving from the watch page (static panel)

The captain watches the mission from a phone via the firstmate watch page
(`http://<DASHBOARD_HOST>:8787`, `data/isaac-watch/serve.py`), which already serves per-run
media. The static panel is built to drop straight into that world:

```
# render the pair into the newest (or --run) run directory, re-rendering every 3 s
PYTHONPATH=scripts python3 scripts/run_fleet_dashboard.py \
    --static-out ~/datasets/twin-robot-in-store-boot-v1/mission/<run> --watch 3 &
```

Then make it reachable one of two ways (both named for firstmate in this lane's report):

1. **`python3 -m http.server` over the run dir** (no watch-page change):
   `python3 -m http.server 8794 --bind <DASHBOARD_HOST> -d <run_dir>` → `…:8794/fleet.html`.
   Media resolves because the frame, `walkthrough.mp4` and `bt_view.html` are in that dir.
2. **A `/fleet` route on the watch page** (firstmate adds two lines to `serve.py`): serve
   `<newest_run>/fleet.html` at `/fleet` and `<newest_run>/fleet.json` at `/fleet.json`, and
   link it from the page. The exact route + the sidecar command are in
   `data/sim-fleet-dashboard-v2/report.md`.

`--media-base <url>` instead points the frame/report links back at the watch page
(`<url>/latest.jpg`, `<url>/report`) when the panel is *not* served from the run dir.

## Store-state export contract

store-vision is a separate, **local-only, read-only** project for this lane, so the
dashboard does **not** import it. Instead it consumes a small JSON export at
`--store-state`. All keys are optional; extra keys are tolerated:

```json
{
  "generated_at": "2026-10-17T20:40:00Z",
  "source": "storevision.availability+hq",
  "store_id": "supermarket-twin",
  "on_shelf_availability_pct": 94.2,
  "slots": [
    {"slot": "G3_back_s2", "sku": "SKU-1234", "fill_pct": 0.0, "state": "out_of_stock"},
    {"slot": "G3_back_s1", "sku": "SKU-1200", "fill_pct": 62.0, "state": "limited"}
  ],
  "alerts": [
    {"type": "spill", "severity": "warn", "count": 1, "location": "aisle-3"},
    {"type": "obstruction", "severity": "critical", "count": 0}
  ]
}
```

`state` accepts both the dashboard's own labels (`oos`/`empty`/`low`) and store-vision's
native `availability.types.AVAILABILITY_STATES` vocabulary (`in_stock` / `limited` /
`out_of_stock` / `unknown`). A slot needs restock when its state is
`oos|out_of_stock|empty|low|limited`; it counts as out-of-stock only for
`oos|out_of_stock|empty`. `on_shelf_availability_pct` is store-wide availability.

### Proposed store-vision exporter (proposal only — NOT committed here)

store-vision already reconstructs the live feed via
`storevision.availability.readmodel.availability_from_store(store)` →
`AvailabilitySnapshot` (`store_id`, `generated_at`, per-SKU `SkuAvailability` with
`state ∈ AVAILABILITY_STATES`, `on_shelf_units`, …). A ~20-line CLI
(`storevision.availability.cli export-fleet --out store-state.json`) could fold that
snapshot (plus the HQ overview's alert counts) into the shape above and write it on a
timer. That change belongs in store-vision and is out of scope for this read-only lane;
it is written up as a proposal in this lane's report and never committed to store-vision.

## Behaviour-tree panel (sim-restock-bt-v1)

The behaviour-tree lane (`sim-restock-bt-v1`, py_trees mission layer, opt-in via `"bt": true`
in the runtime-config) writes a live snapshot into each run directory every tick
(`data/sim-restock-bt-v1/report.md` §4):

- `bt_snapshot.json` — the current tree (nested `name`/`status`/`feedback`/`kind`/`children`)
  plus `tick`, `clock`, `mission` {state,reason,degraded}, `control`
  {pause,slow,replan,abort}, `active_events`, `last_event`, and an append-only `log` of
  `{t,type,data,action}`.
- `bt_view.html` — the lane's own self-contained phone page (polls `bt_snapshot.json`).
- `bt_events.jsonl` — append-only event/action timeline.

The dashboard **auto-discovers** `bt_snapshot.json` in the resolved run directory — the
same run its pipeline reads — so the panel lights up on the first bt-enabled run with **zero
config**. It renders the tree coloured by node status (success/running/failure/invalid),
the control-flag chips, the last event and a recent event/action log, and links the richer
`bt_view.html` at `<upstream>/<run_root basename>/<run>/bt_view.html` (the watch page serves
the run tree). `--bt <path>` overrides auto-discovery with an explicit snapshot file.

The event topic `/restock_bt/events` (`std_msgs/String`, JSON per event) is the lane's live
bus; the dashboard reads the per-tick `bt_snapshot.json` the lane already persists rather
than subscribing to ROS, so it stays CPU-only and needs no ROS on the phone path.

## App trigger / monitor queue (opt-in, default off)

A phone app (store-vision `missions` / `shelf-lens` / `lovable`, or any operator) can
`POST /trigger` a restock *request*. It is a **monitored queue, not an actuator**: the
request is written as a JSON file to `--requests-dir` and listed back read-only; writing
one does **not** drive the robot. A demand-driven job feed (rank-1 `sim-smart-job-feed-v1`)
or an operator can later consume the queue to build a mission job. Enqueuing is gated
behind `--enable-trigger` (which requires `--requests-dir`); without it the dashboard is
strictly read-only and `POST /trigger` returns `403`.

Request body: `sku` (required, non-empty), `quantity` (required, positive int), and
optional `source`, `shelf`, `priority` (int), `note`, `provenance`. Bad input is refused
with `400` and no file is written.

### Smart approval variant (`--approval-trigger`, default off)

The **working default** is the firstmate-decided pre-approved sim: no trigger at all (the
panel is read-only) or the plain monitored queue above (`--enable-trigger`). The **smart
variant** adds the phone approval step the captain's intent calls for, kept as a *separate
opt-in* so the working default is never replaced silently:

- With `--approval-trigger` (requires `--enable-trigger`), a `POST /trigger` request lands
  as `pending_approval` (carrying an `approval` block) and does **nothing** on its own.
- A phone `POST /approve` / `POST /reject` with `{"id": "<trigger id>"}` records the decision
  (`approval.state`, `by`, `at`). On approve the request is also written as a
  `storevision.restock.job_plan/v1` proposal under `<requests-dir>/approved/<id>.json`.
- That proposal is exactly what the **rank-1 demand feed honours**: an operator (or firstmate)
  runs `run_mobile_restock.py --job-from <approved/…json>`, and `twin.restock.job_from_plan`
  re-validates the SKU + destination slot against `scripts/planogram.json`, refusing an
  unknown slot or SKU (honest failure — the dashboard proposes, it never actuates). The
  proposal carries the request's `source`/`shelf`; pose/xyz remain store-vision's to fill.

This keeps the single source of truth honest: the approval queue is a *monitored proposal
queue*, never a robot actuator, and the mission job feed stays the only thing that turns an
approved SKU into a run.

### Owner decision surfaced

**Should the phone approval gate sit in the sim loop?** Per the firstmate-decided default
for this lane, the human-approval gate is **off by default and selectable**, and the sim
run is pre-approved. This dashboard honours that: the trigger queue is opt-in and never
blocks or drives a mission on its own. Wiring the queue into a mission approval gate is a
downstream (`sim-restock-downstream-v1`) orchestrator decision, coordinated through status
lines, not implemented here.

## HTTP routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | phone-first HTML page (self-contained, no external assets) |
| `/status.json` | GET | aggregated mission + store + triggers view (`schema: fm-fleet-dashboard.v1`) |
| `/config.json` | GET | `{trigger_enabled, approval_required}` (drives the page's form + buttons) |
| `/triggers.json` | GET | stored trigger requests, newest first |
| `/healthz` | GET | `ok` |
| `/trigger` | POST | enqueue a restock request (only when `--enable-trigger`; else `403`) |
| `/approve` | POST | approve a `pending_approval` request by `{id}` (only when `--approval-trigger`) |
| `/reject` | POST | reject a `pending_approval` request by `{id}` (only when `--approval-trigger`) |

The static panel serves no endpoints — it is the `fleet.html` + `fleet.json` pair over a
static mount, read-only (no POST), so the approval path is live-server only.

The upstream `status.json` fetch has a short timeout and **degrades gracefully**
(`upstream.ok=false`, pulse falls back to the newest run directory) when the watch page is
down; it never raises and never proxies upstream media.
