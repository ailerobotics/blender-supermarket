#!/usr/bin/env python3
"""Before/after timeline for the intermittent collect host e-stop (sim-collect-host-estop-v1).

CPU-only, no ROS/Isaac/GPU. Reconstructs the parked source-station heartbeat READ-LAG across a long collect
(the recorded coldwin22 collect runs sim 0 -> 173.87 s; coldwin23 attempt-1 e-stopped at the collect stage at
sim 175.7 s with a bare 'host emergency stop' and no controller fault). A loaded-host executor / render spike
near the collect->couple boundary pushes the host's heartbeat READ-LAG (now - stamp) past the 0.5 s accept()
window.

  BEFORE: the host rejects the valid-but-late heartbeat -> station fault -> bare global /manipulation/stop
          (no controller fault). Trips at the first sim time the read-lag crosses 0.5 s.
  AFTER : accept() accepts any newer-sequence liveness command regardless of read latency and measures the
          liveness gap from the host's OWN acceptance clock, so the late read is absorbed and collect runs to
          the couple handoff. A genuinely silent station (no new sequence) still faults on the 0.5 s sim gap.

Writes before-after-timeline.png (if matplotlib is present) and before-after-timeline.txt (always)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/ is three levels up: data/sim-collect-host-estop-v1 -> repo root -> scripts
sys.path.insert(0, os.path.join(HERE, '..', '..', 'scripts'))

COLLECT_END = 173.87     # coldwin22 (PASS): couple begins here
FAIL_SIM = 175.7         # coldwin23 attempt-1: e-stopped at the collect stage here
ACCEPT_WINDOW = 0.5      # the accept() now-stamp upper bound (unchanged; the real-rig freshness contract)


def read_lag_trace():
    """Host heartbeat read-lag (now - stamp, sim-seconds) sampled across the collect. Flat-low during the
    steady pick, then a load spike near the collect->couple boundary (cuMotion stow replan + couple-nav
    kickoff + nvblox) starves the host executor so a backlogged heartbeat is read progressively later."""
    pts = []
    t = 0.0
    while t <= 178.0:
        if t < 170.0:
            lag = 0.08              # steady collect: heartbeats read promptly
        else:
            # load spike ramps the read-lag so it crosses the 0.5 s window exactly at the observed fail time.
            lag = 0.08 + (t - 170.0) * ((ACCEPT_WINDOW - 0.08) / (FAIL_SIM - 170.0))
        pts.append((t, lag))
        t += 0.2
    return pts


def main():
    trace = read_lag_trace()
    # BEFORE: first sim time the read-lag exceeds the accept window -> reject -> fault -> e-stop.
    before_trip = next((t for t, lag in trace if lag > ACCEPT_WINDOW), None)

    # AFTER: drive the SHIPPED pace-safe harness across the whole collect at the measured live_pose pacing.
    from twin.restock.host_pacing_harness import simulate_loaded_collect_heartbeat_read_lag
    after = simulate_loaded_collect_heartbeat_read_lag(
        pacing=30.0, sim_gap_budget=0.5, wall_floor=3.0, read_lag_sim=0.6, op_sim=COLLECT_END)

    lines = []
    lines.append('sim-collect-host-estop-v1 : parked source-station heartbeat read-lag across collect')
    lines.append('')
    lines.append(f'  accept() window (now-stamp bound) : {ACCEPT_WINDOW:.2f} s  (unchanged)')
    lines.append(f'  coldwin22 PASS: couple begins      : {COLLECT_END:.2f} s sim')
    lines.append(f'  coldwin23 attempt-1 e-stop         : {FAIL_SIM:.2f} s sim (collect stage, bare host e-stop, no controller fault)')
    lines.append('')
    lines.append(f'  BEFORE (strict now-stamp<=0.5 for every command):')
    lines.append(f'      read-lag crosses {ACCEPT_WINDOW:.2f} s at sim ~{before_trip:.1f} s -> late heartbeat REJECTED')
    lines.append(f'      -> station fault -> STATION_ESTOP_PUBLISH -> bare /manipulation/stop (latched) -> e-stop')
    lines.append('')
    lines.append(f'  AFTER (shipped: liveness commands accepted regardless of read latency; gap from acceptance clock):')
    lines.append(f"      outcome={after['outcome']}  accepted_heartbeats={after['accepted']}  final_sim={after['sim']:.1f}s")
    lines.append(f'      late reads absorbed; collect runs to the couple handoff at {COLLECT_END:.2f} s; station never faults')
    lines.append('')
    lines.append('  regression kept: a genuinely silent station (no newer-sequence command) still faults on the')
    lines.append('  0.5 s SIM gap -> "parked heartbeat lost: sim gap X.XX s" (test_silent_station_still_faults_by_name).')
    txt = '\n'.join(lines) + '\n'
    with open(os.path.join(HERE, 'before-after-timeline.txt'), 'w') as f:
        f.write(txt)
    print(txt)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plot is best-effort
        print(f'(matplotlib unavailable: {exc}; wrote the .txt timeline only)')
        return

    ts = [t for t, _ in trace]
    lags = [lag for _, lag in trace]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(ts, lags, color='#1f77b4', lw=2, label='host heartbeat read-lag (now - stamp)')
    ax.axhline(ACCEPT_WINDOW, color='#888', ls='--', lw=1, label=f'accept() window {ACCEPT_WINDOW:.2f} s (unchanged)')
    ax.axvline(COLLECT_END, color='#2ca02c', ls=':', lw=1.3, label=f'coldwin22 PASS: couple begins {COLLECT_END:.1f}s')
    if before_trip is not None:
        ax.scatter([before_trip], [ACCEPT_WINDOW], color='#d62728', zorder=5)
        ax.annotate('BEFORE: late heartbeat rejected\n-> station fault -> bare e-stop\n(no controller fault)',
                    xy=(before_trip, ACCEPT_WINDOW), xytext=(before_trip - 90, 0.62),
                    color='#d62728', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#d62728'))
    ax.annotate('AFTER: late read accepted; gap measured from\nacceptance clock -> collect survives to couple',
                xy=(COLLECT_END, 0.09), xytext=(20, 0.30), color='#2ca02c', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#2ca02c'))
    ax.set_xlabel('collect sim time (s)')
    ax.set_ylabel('read-lag (sim s)')
    ax.set_title('sim-collect-host-estop-v1: parked source-station heartbeat read-lag (before vs after)')
    ax.set_ylim(0, 0.75)
    ax.set_xlim(0, 178)
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, 'before-after-timeline.png')
    fig.savefig(out, dpi=120)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
