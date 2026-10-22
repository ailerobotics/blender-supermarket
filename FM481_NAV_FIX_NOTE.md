# fm481 coupled-tow outbound robustness — diagnosis + diff (for sim-video-couple-speed-v1)

**Left uncommitted per fm473** (lane sim-video-couple-speed-v1 owns this). Diff touches
`scripts/twin/navigation/articulated.py` (the coupled `ArticulatedController.gate_velocity`) and
`scripts/twin/navigation/core.py` (adds a `coupled_creep_floor=0.03` ctor param, inherited).

## Real root cause (from run-355 nav-trace.jsonl, NOT the original "start-yaw" hypothesis)

The outbound coupled tow (`outbound_dock_to_bay`, a STRAIGHT leg -4.46→-2.0 at y=15.25, yaw 0)
drove well for the first ~0.5 m (rv≈0.23 m/s), then **stalled at x≈-3.935 for 121 s** (parked=True)
until blocked_timeout. At the stall the base under-translated: `vel.rv`/`vel.rw` were both
~0.01–0.05 m/s — **near-zero velocity NOISE**. The measured-curvature brake computes
`measured_k = odom.angular/odom.linear`; on 80 % of the ticks where `|rv|>0.02` this noise ratio
exceeded `1/R_min+0.03` → `measured_turn_radius` → coast (base commanded to stop this tick) →
next tick re-enters the same noisy near-zero band → **livelock**. The follower commanded full
0.25 m/s forward the whole time; the base simply did not achieve it (traction/caster-lift under tow).

Key point: the lane follower **already** caps commanded curvature at `1/R_min` (articulated.py
~line 999), so the original fm481 idea of "clamp the COMMANDED w to v/R_min" is **inert** on this
path — the refusal is driven by the MEASURED (odom) arc, i.e. jitter at a near-stationary crawl.

## The fix in this diff

In the measured-curvature brake (articulated.py ~line 889): when it would fire AND the base is
essentially stopped (`|odom.linear| < max(0.06, 2*creep_floor)`) with a forward drive commanded,
treat the ratio as noise — keep a forward **creep** (`v>=coupled_creep_floor`) with the arc clamped
to within `R_min`, and CONTINUE (the creep is still footprint- and braking-rollout-checked just
below) so the base accelerates OUT of the noise band instead of coasting back into it. A genuine
sub-R_min arc at real tow speed still refuses — the safety brake is unchanged above the band.
`core.py` only gains the `coupled_creep_floor` ctor param (default 0.03) used by the above.

## Suggested proof (not yet written — lane owns it)

`scripts/check_tow_lanes.py` style, driving the REAL `ArticulatedController` closed-loop over the
mission's real `outbound_dock_to_bay` lane, but with a **velocity-lag plant** on the fed-back odom
(rv relaxes toward commanded v with τ≈0.6 s; rw tracks fast) seeded at the run-355 stall state
(x≈-3.935, start-yaw 0.6°, rv≈0). Assert: OLD articulated.py livelocks (many `measured_turn_radius`,
~0 net progress, no goal); NEW reaches the goal with ~0 measured refusals; and an obstacle placed
ahead on the lane still yields `obstacle_stop` under NEW (safety preserved).

## Caveat
This is NOT byte-identical-when-off: there is no feature flag; it always changes the coupled gate in
the near-stationary noise band. If a flag is wanted, gate the creep-escape on a
`navigation.coupled_creep_escape` config bool (default on for the demo). Mission's outbound lane is
straight; verify no regression on any curvy/portal coupled lane (steady-speed following is unchanged
because `|odom.linear|≈v` there, so the brake path is untouched).
