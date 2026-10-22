"""sim-robust-watchdogs: the mission's producer watchdogs must not trip merely because the
simulation renders slower than wall time, while a genuinely dead producer is still caught.

These replay the FALSE-TRIP signatures the sim-robust-audit named (arm-command heartbeat; the
station-ack frozen-stamp StationAckStreamStalled of runs 341/346/348/350/354/356; the no-base-
progress nav-stuck budget) and assert NO trip, then replay a genuine host death and assert the
trip. CPU-only: no Isaac, ROS or GPU -- every watchdog is driven through its pure seam with the
sim/wall clocks supplied by the test. See scripts/twin/restock/pace_watchdog.py.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import pace_watchdog, native_transport
from twin.restock.native_channel import StationClient, StationAckStreamStalled
from twin.restock.runtime import RuntimeAdapter


# ----------------------------------------------------------------------------- pace_watchdog core
def test_is_dead_either_fires_on_sim_or_wall():
    assert pace_watchdog.is_dead(sim_elapsed=9, wall_elapsed=0, sim_budget=8, wall_backstop=30, mode='either')
    assert pace_watchdog.is_dead(sim_elapsed=0, wall_elapsed=31, sim_budget=8, wall_backstop=30, mode='either')
    assert not pace_watchdog.is_dead(sim_elapsed=1, wall_elapsed=1, sim_budget=8, wall_backstop=30, mode='either')


def test_is_dead_both_needs_sim_and_wall():
    assert not pace_watchdog.is_dead(sim_elapsed=9, wall_elapsed=1, sim_budget=8, wall_backstop=30, mode='both')
    assert not pace_watchdog.is_dead(sim_elapsed=1, wall_elapsed=31, sim_budget=8, wall_backstop=30, mode='both')
    assert pace_watchdog.is_dead(sim_elapsed=9, wall_elapsed=31, sim_budget=8, wall_backstop=30, mode='both')


def test_is_dead_none_legs_and_bad_mode():
    assert not pace_watchdog.is_dead(sim_elapsed=None, wall_elapsed=99, sim_budget=8, wall_backstop=None, mode='either')
    assert not pace_watchdog.is_dead(sim_elapsed=99, wall_elapsed=None, sim_budget=None, wall_backstop=8, mode='either')
    with pytest.raises(ValueError):
        pace_watchdog.is_dead(sim_elapsed=1, wall_elapsed=1, sim_budget=1, wall_backstop=1, mode='nope')


# ----------------------------------------------------------------------------- arm-command heartbeat
def _arm_gate(monkeypatch, wall):
    """An ArmCommandGate with a stub articulation and a test-controlled wall clock."""
    monkeypatch.setattr(native_transport.time, 'monotonic', lambda: wall[0])
    robot = NS(get_dof_index=lambda name: 0,
               get_joint_positions=lambda joint_indices=None: np.zeros(6))
    applied = []
    gate = native_transport.ArmCommandGate(robot, [(-10., 10.)] * 6,
                                           lambda q, v, i: applied.append((q, v)))
    gate.applied = applied
    return gate


def test_arm_heartbeat_survives_a_slow_render_tick_while_stamps_advance(monkeypatch):
    # The audit's single-most-common terminal symptom ('arm command heartbeat lost'): the OLD pure-WALL
    # 30 s backstop charged a render-paced command gap against wall time. Here the host stream is HEALTHY
    # (stamps advancing with sim now) but ONE render tick takes 40 s wall -- far past the old 30 s wall
    # backstop. The sim leg sees a tiny gap (now - host_now ~ relay lag) and the wall leg is disarmed the
    # instant sim now advances, so no trip.
    wall = [0.0]
    gate = _arm_gate(monkeypatch, wall)
    gate.started = True
    gate._host_now = 10.0
    gate.watchdog(10.5)                 # healthy: sim now 10.5, host stamp 10.0 (lag 0.5)
    wall[0] = 40.0                      # a single 40 s-wall render tick (>> old 30 s wall backstop)...
    gate._host_now = 11.8              # ...but a fresh command arrived: stamps advanced with sim
    gate.watchdog(12.0)                # sim advanced -> wall leg reset, sim gap 0.2 s -> NO trip
    assert not gate.faulted


def test_arm_heartbeat_fires_on_true_host_death_stamps_frozen_while_sim_advances(monkeypatch):
    # cuMotion CUDA OOM takes move_group + trajectory_host down (audit rank 1): no more arm commands, so
    # the greatest-received stamp freezes while Isaac keeps stepping /clock. The SIM leg grows past the
    # budget within a few sim seconds and faults -- independent of render pace.
    wall = [0.0]
    gate = _arm_gate(monkeypatch, wall)
    gate.started = True
    gate._host_now = 11.5             # last command stamp before the host died
    wall[0] = 1.0
    gate.watchdog(13.0)              # sim gap 1.5 s < 8 s budget -> alive
    assert not gate.faulted
    wall[0] = 2.0
    with pytest.raises(RuntimeError, match='arm command heartbeat lost'):
        gate.watchdog(20.0)          # sim gap 8.5 s > 8 s budget -> dead
    assert gate.faulted


def test_arm_heartbeat_wall_backstop_catches_a_frozen_sim(monkeypatch):
    # If the sim clock itself stops advancing while the watchdog keeps being called (a paused/wedged sim
    # the sim leg cannot see -- now - host_now cannot grow), the wall-since-sim-now-advanced backstop is
    # the one remaining catch.
    wall = [0.0]
    gate = _arm_gate(monkeypatch, wall)
    gate.started = True
    gate._host_now = 11.5
    gate.watchdog(12.0)              # sim now 12.0 (arms the wall timer at wall 0)
    wall[0] = 20.0
    gate.watchdog(12.0)              # now frozen at 12.0; wall gap 20 < 30 backstop -> still alive
    assert not gate.faulted
    wall[0] = 31.0
    with pytest.raises(RuntimeError, match='arm command heartbeat lost'):
        gate.watchdog(12.0)          # now still frozen; wall gap 31 > 30 backstop -> dead
    assert gate.faulted


# ----------------------------------------------------------------------------- station-ack frozen stamp
class _Channel:
    def __init__(self):
        self.reply = None

    def publish(self, value):
        pass

    def read(self):
        return self.reply


# run-341/346/348 froze the FIRST stamp at warm start (stage ~6-8 s); run-350/354/356 froze deep in the
# mission (stage ~216-221 s). Both are the render-lagged-but-alive publisher: the reply keeps arriving
# for our epoch, only the host had not yet advanced its /clock-derived stamp.
@pytest.mark.parametrize('frozen_stamp,begin_stage,fire_stage', [
    (0.950000049, 0.9, 6.017),      # run-341
    (3.250000169, 3.2, 8.300),      # run-346 / run-348
    (216.40001128600002, 216.3, 221.417),   # run-350 / run-354
    (214.36667784600002, 214.3, 219.417),   # run-356
])
def test_station_ack_frozen_startup_lag_does_not_false_trip(frozen_stamp, begin_stage, fire_stage):
    # Replay the named false trips: the stamp sits frozen for ~5 STAGE seconds, but under the measured
    # 0.17-0.31x pacing that is only ~16-30 WALL seconds -- inside the 45 s wall backstop, so a
    # still-delivering (alive) publisher is NOT declared dead. It surfaces as the generic 'stale' the
    # warm() loop swallows, never the fatal StationAckStreamStalled.
    stage = [begin_stage]
    wall = [0.0]
    ch = _Channel()
    client = StationClient(ch, wall_clock=lambda: wall[0], stage_clock=lambda: stage[0],
                           heartbeat_s=1000., stage_unavailable_s=1000., ack_frozen_s=2.0,
                           ack_frozen_wall_s=45.0)
    client.begin('restock_camera_source', begin_stage)
    ch.reply = dict(epoch=1, stamp=frozen_stamp, state='ready', collision_time=frozen_stamp - 0.05)
    frozen_stage = fire_stage - begin_stage
    # Advance the stage clock to the point the run tripped, with wall tracking a conservative 0.25x pace.
    for frac in (0.5, 1.0):
        stage[0] = begin_stage + frozen_stage * frac
        wall[0] = (stage[0] - begin_stage) / 0.25      # ~5 stage-s -> ~20 wall-s, < 45 s backstop
        with pytest.raises(ValueError, match='stale'):  # generic stale, NOT StationAckStreamStalled
            client.tick(0., parked=True)
    assert wall[0] < 45.0        # the whole replay stayed inside the wall backstop


def test_station_ack_frozen_truly_dead_publisher_still_trips():
    # A genuinely dead publisher stays frozen FOREVER, so real wall time piles up past the backstop: the
    # dual (stage AND wall) condition is met and it fails fast with the named cause (run-335 wedge).
    stage = [0.9]
    wall = [0.0]
    ch = _Channel()
    client = StationClient(ch, wall_clock=lambda: wall[0], stage_clock=lambda: stage[0],
                           heartbeat_s=1000., stage_unavailable_s=1000., ack_frozen_s=2.0,
                           ack_frozen_wall_s=45.0)
    client.begin('restock_camera_source', 0.9)
    ch.reply = dict(epoch=1, stamp=0.9, state='ready', collision_time=0.85)
    stage[0] = 2.0
    wall[0] = 1.0                # first sighting of the frozen stamp: records frozen-since, only 'stale'
    with pytest.raises(ValueError, match='stale'):
        client.tick(0., parked=True)
    stage[0] = 10.0
    wall[0] = 50.0               # frozen past BOTH the 2 s stage budget and the 45 s wall backstop
    with pytest.raises(StationAckStreamStalled) as exc:
        client.tick(0., parked=True)
    assert 'publisher stopped' in str(exc.value)


# ----------------------------------------------------------------------------- nav no-base-progress
def _nav_adapter(*, nav_stuck_sim_s=0.0, nav_stuck_wall_s=0.0, wall):
    nav = NS(start=lambda g, c, **k: None,
             step=lambda dt, p: NS(status='running', linear=0., angular=0., reason='planning_stop'),
             cancel=lambda: None)
    mobile = NS(park=lambda: None, hold=lambda: None, release=lambda: None,
                command=lambda *a, **k: None, attach=lambda o: None,
                coupling_state=lambda o: {'healthy': True})
    manip = NS(start=lambda a, j: None, cart_location='rollwagen',
               step=lambda dt, s: NS(status='running', reason='', controls=()))
    port = NS(advance=lambda dt: None, controls=lambda c: None, stop=lambda: None)
    poses = {k: [0, 0, 0] for k in ('home', 'pickup', 'cart_dock', 'shelf_approach')}
    adapter = RuntimeAdapter(nav, mobile, manip, port, poses, fixture_mode=True,
                             wall_clock=lambda: wall[0], nav_stuck_sim_s=nav_stuck_sim_s,
                             nav_stuck_wall_s=nav_stuck_wall_s)
    adapter.action = 'outbound'
    return adapter


def test_nav_stuck_sim_budget_is_immune_to_render_pacing():
    # The audit's rank-3/5 'sim-clock-gate the no-base-progress watchdog': a 120 s WALL budget is only
    # ~20-37 s SIM at 0.17-0.31x, so render pacing shortens the real budget. With a SIM budget set the
    # base gets its full intended sim-seconds of no-progress before the stage fails -- here a frozen base
    # that has already burned 400 s WALL (the pure-wall budget would have long fired) but only 30 s SIM
    # (< the 120 s sim budget) must stay RUNNING.
    wall = [0.0]
    a = _nav_adapter(nav_stuck_sim_s=120.0, nav_stuck_wall_s=120.0, wall=wall)
    odo = NS(pose=(2.0, -1.0, 1.57))          # frozen base pose (never progresses)
    a.time = 0.0
    assert a._nav_stuck_check('planning_stop', odo) is None   # baseline
    a.time = 30.0        # 30 s SIM of no progress...
    wall[0] = 400.0      # ...but 400 s WALL (far past the 120 s wall budget, which is now disarmed)
    assert a._nav_stuck_check('planning_stop', odo) is None   # sim budget governs -> still running


def test_nav_stuck_sim_budget_fires_on_a_real_stall():
    wall = [0.0]
    a = _nav_adapter(nav_stuck_sim_s=120.0, nav_stuck_wall_s=120.0, wall=wall)
    odo = NS(pose=(2.0, -1.0, 1.57))
    a.time = 0.0
    a._nav_stuck_check('planning_stop', odo)
    a.time = 121.0       # frozen base for 121 s SIM > 120 s sim budget -> real stall
    wall[0] = 500.0
    r = a._nav_stuck_check('planning_stop', odo)
    assert r is not None and 'no base progress' in r and "last reason 'planning_stop'" in r


def test_nav_stuck_legacy_wall_only_path_is_unchanged():
    # No sim budget configured => byte-identical legacy behaviour: the WALL budget alone fails fast.
    wall = [0.0]
    a = _nav_adapter(nav_stuck_sim_s=0.0, nav_stuck_wall_s=5.0, wall=wall)
    odo = NS(pose=(2.0, -1.0, 1.57))
    a.time = 0.0
    a._nav_stuck_check('lane_blocked', odo)
    a.time = 0.5
    wall[0] = 6.0        # past the 5 s wall budget, base frozen
    r = a._nav_stuck_check('lane_blocked', odo)
    assert r is not None and 'no base progress' in r and 'lane_blocked' in r


def test_nav_stuck_resets_on_base_progress_under_either_budget():
    # A base that is actually moving resets both timers and never trips, whichever budget is set.
    wall = [0.0]
    a = _nav_adapter(nav_stuck_sim_s=10.0, nav_stuck_wall_s=10.0, wall=wall)
    a.time = 0.0
    a._nav_stuck_check('planning_stop', NS(pose=(0.0, 0.0, 0.0)))
    for i in range(1, 6):
        a.time = i * 100.0        # huge sim gaps...
        wall[0] = i * 100.0       # ...and huge wall gaps
        moved = NS(pose=(i * 0.5, 0.0, 0.0))   # but the base advanced 0.5 m each step
        assert a._nav_stuck_check('planning_stop', moved) is None
