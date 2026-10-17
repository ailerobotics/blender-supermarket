# Fleet dashboard — watch the Isaac mission + the store (read-only, CPU, opt-in)

A phone-first, read-only panel that merges the **live Isaac restocking mission** with
**store-vision store state** into one view over the tailnet. It is strictly additive:
the dashboard is a standalone CPU server that the mission never imports or launches, so
a mission run is **byte-identical** whether or not the dashboard is up. No GPU.

- Module: `scripts/twin/dashboard/` (pure aggregation in `aggregate.py`, I/O in `server.py`).
- Launcher + dry-run: `scripts/run_fleet_dashboard.py` (RUNBOOK §17).
- Tests: `scripts/twin/tests/test_fleet_dashboard.py` (no network, no GPU, no simulator).

## What it shows

1. **Mission pulse** — from the existing firstmate watch page `status.json`
   (`http://100.100.47.65:8787/`, read-only): alive/idle, run name, frame count, the live
   trace (stage, tick, sim time, coupled/parked, base x/y).
2. **Mission pipeline** — the `reserved → collect → couple → outbound → park → transfer →
   verify → return` timeline for the resolved run, read from that run's
   `mission.json` events under
   `/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/<run>/`. The stage the
   run is sitting on reads *active* while running and *failed* if the run failed there (a
   recorded stage-entry event means "entered", not "completed"); verify's physics-verified
   verdict is surfaced on success.
3. **Media** — latest rendered frame and the run/stage walkthrough videos, **linked back**
   to the upstream watch page (never re-served here).
4. **Behaviour tree** — rendered when a lane publishes a BT view JSON (`--bt <path>`);
   otherwise the card reads "Not published by this run." This is the seam for
   `sim-restock-bt-v1`.
5. **Store state** — store-vision shelf fill / depletion / alerts, overlaid from a
   documented JSON export (`--store-state <path>`); absent ⇒ "Not configured."
6. **Restock requests** — an opt-in app trigger/monitor queue (see below).

## Reuse, not replacement

The dashboard **reads** the firstmate-owned watch page (`:8787`) rather than replacing it:
it consumes that page's `status.json` for the live pulse and links its media back to it. It
**reads** the mission run directories for per-run detail. It binds to the tailnet IP
`100.100.47.65` on a **free** port (default `8794`; `8787/8790/8791/8793` are taken) so it
coexists with the watch page. All media links point at the upstream base.

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
| `/config.json` | GET | `{trigger_enabled}` (the page shows the form only when true) |
| `/triggers.json` | GET | stored trigger requests, newest first |
| `/healthz` | GET | `ok` |
| `/trigger` | POST | enqueue a restock request (only when `--enable-trigger`; else `403`) |

The upstream `status.json` fetch has a short timeout and **degrades gracefully**
(`upstream.ok=false`, pulse falls back to the newest run directory) when the watch page is
down; it never raises and never proxies upstream media.
