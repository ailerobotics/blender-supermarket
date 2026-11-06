"""sim-retreat-station-heartbeat-scope-v1: CPU unit proof for the capture-silent parked-station
heartbeat suspension (restock_host.incorporation_ack.IncorporationAckState).

coldwin81 cleared the loaded place + contact release and the scoped sensor watchdog held through the
empty retreat, but the run died on the LAST watchdog still counting the capture-silent retreat window:
the parked-station SIM-gap heartbeat ('parked heartbeat lost: sim gap 60.02 s') fired during a ~98 sim-s
retreat the mission does not heartbeat. This proves the opt-in key suspends ONLY the SIM leg over the
capture-silent window (re-armed on the next capture-bearing phase), keeps the frozen-sim WALL backstop
armed, and is byte-identical when off. Pure: no ROS / Isaac / GPU (the snapshot sequencing is exercised
on the shipped state object with a fake wall clock).
"""
import pytest

from twin.restock_host.incorporation_ack import IncorporationAckState
from twin.restock.pace_watchdog import frozen_sim_wall_backstop


def _state(*, suspend, sim_gap=60.0, wall_floor=30.0, wall=None):
    """A prepared, active parked epoch with the cw82 budgets, the suspend key set as the env would."""
    wall = wall if wall is not None else [0.0]
    st = IncorporationAckState(wall_clock=lambda: wall[0])
    st.sim_gap_s = float(sim_gap)
    st.wall_gap_s = float(wall_floor)
    st.wall_backstop_s = frozen_sim_wall_backstop(sim_gap, wall_floor)
    st.hb_capture_silent_suspend = bool(suspend)
    st.accept(dict(action='prepare', epoch=1, sequence=0, stamp=0.0, parked=True,
                   camera_frame='restock_camera_collect'), 0.0)
    return st, wall


# --- note_heartbeat_phase window edges (named lines, latch) -------------------------------------------

def test_key_off_never_latches_or_logs_byte_identical():
    st, _ = _state(suspend=False)
    for phase in ('observe', 'retreat', 'stow', 'release', 'verify'):
        assert st.note_heartbeat_phase(phase) is None
        assert st.heartbeat_suspended is False


def test_entry_latches_and_logs_suspend_once_then_rearm_on_capture_bearing():
    st, _ = _state(suspend=True)
    # capture-bearing phases never suspend
    assert st.note_heartbeat_phase('observe') is None
    assert st.heartbeat_suspended is False
    # entry to a capture-silent phase latches + logs ONE suspend line
    line = st.note_heartbeat_phase('retreat')
    assert line is not None and line.startswith('STATION_HEARTBEAT_SUSPENDED phase=retreat')
    assert 'WALL backstop' in line
    assert st.heartbeat_suspended is True
    # staying capture-silent (stow, release) does not re-log
    assert st.note_heartbeat_phase('stow') is None
    assert st.note_heartbeat_phase('release') is None
    assert st.heartbeat_suspended is True
    # re-entry to a capture-bearing phase re-arms + logs ONE rearm line
    rline = st.note_heartbeat_phase('verify')
    assert rline is not None and rline.startswith('STATION_HEARTBEAT_REARMED phase=verify')
    assert st.heartbeat_suspended is False
    assert st.note_heartbeat_phase('observe') is None   # already re-armed, no duplicate


# --- snapshot behaviour over the cw81 retreat --------------------------------------------------------

def test_suspended_survives_the_98s_retreat_and_resumes_clean():
    st, wall = _state(suspend=True)
    st.note_heartbeat_phase('retreat')          # enter the capture-silent window
    # 98 sim-s pass with no heartbeat; even at 16x pacing that is ~1568 wall-s
    for sim in (10.0, 40.0, 60.02, 80.0, 98.0):
        wall[0] = sim * 16.0
        snap = st.snapshot(sim, endpoints_ready=True, process_alive=True)
        assert snap['state'] != 'failed', (sim, snap)
    # re-arm at verify, then the very next snapshot must NOT fault on the accumulated 98 s gap
    st.note_heartbeat_phase('verify')
    snap = st.snapshot(98.1, endpoints_ready=True, process_alive=True)
    assert snap['state'] != 'failed', snap
    assert st.fault is None


def test_key_off_reproduces_the_known_cw81_sim_gap_fault():
    st, wall = _state(suspend=False)
    st.note_heartbeat_phase('retreat')          # no-op when off (never latches)
    assert st.heartbeat_suspended is False
    wall[0] = 60.02 * 16.0
    snap = st.snapshot(60.02, endpoints_ready=True, process_alive=True)
    assert snap['state'] == 'failed'
    assert 'parked heartbeat lost: sim gap' in snap['reason']


def test_suspended_still_catches_a_frozen_sim_on_the_wall_backstop():
    # The SIM clock is frozen (a genuinely dead simulator) while suspended: the window never ends, so the
    # derived frozen-sim WALL backstop (60 x 50 = 3000 s) must still catch it by name.
    st, wall = _state(suspend=True)
    st.note_heartbeat_phase('retreat')
    # sim stays at 0 (frozen); wall advances well past the 3000 s backstop
    wall[0] = 3001.0
    snap = st.snapshot(0.0, endpoints_ready=True, process_alive=True)
    assert snap['state'] == 'failed'
    assert 'wall gap' in snap['reason'] and 'capture-silent' in snap['reason']


def test_suspended_does_not_mask_a_process_death():
    st, wall = _state(suspend=True)
    st.note_heartbeat_phase('retreat')
    snap = st.snapshot(5.0, endpoints_ready=True, process_alive=False)
    assert snap['state'] == 'failed'
    assert snap['reason'] == 'station process lost'


# --- single-source phase set + preflight registration ------------------------------------------------

def test_capture_phases_leaf_is_the_single_source_shared_with_isaac_sensors():
    from twin.restock_manipulation.capture_phases import sensing_render_required, CAPTURE_SILENT_PHASES
    from twin.restock_manipulation import isaac_sensors
    # isaac_sensors re-exports the SAME objects (no drift between the sensor watchdog and the station node)
    assert isaac_sensors.sensing_render_required is sensing_render_required
    assert isaac_sensors._SENSING_SUSPEND_PHASES is CAPTURE_SILENT_PHASES
    # the retreat/stow window is capture-silent; observe/verify are capture-bearing
    for p in ('retreat', 'stow', 'release', 'preplace', 'place', 'couple'):
        assert sensing_render_required(p) is False, p
    for p in ('observe', 'verify', 'idle', 'unknown'):
        assert sensing_render_required(p) is True, p


def test_key_registered_host_in_config_preflight():
    from twin.restock_manipulation.config_preflight import KEY_CONSUMER, HOST
    assert KEY_CONSUMER.get('capture_silent_station_heartbeat_suspend') == HOST


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
