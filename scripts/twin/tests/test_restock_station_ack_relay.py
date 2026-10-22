"""sim-robust-station-ack: the live_pose transfer no longer dies at the station-ack freeze.

Root cause (armcam / transfer2 / bench1, all froze at ~sim 31 s with
``StationAckStreamStalled: stamp frozen at 31.05s ... (publisher stopped)``): a PER-OP
manipulation failure (the host controller replied ``failed`` -- in armcam a wrist-branch
trajectory tracking fault, a SEPARATE IK issue) ran ``ros_manipulation.stop(close_sensing=False)``
-> ``IsaacRosGraph.close()`` -> ``host_control.close()``, terminating the SHARED out-of-Isaac
control relay (``control_ipc.HostControlPublisher``). That relay forwards the host->Isaac
return topics, including ``/restock_sim/station/status``. With it gone, the StationClient's
reader thread hit EOF and kept reading the LAST ack stamp forever while the mission stage clock
advanced -- so the frozen-stamp watchdog raised ``StationAckStreamStalled``, MASKING the true
stage reason (``collision planning or execution failed``).

The relay is a STATION-level resource, exactly like the masked-depth feed fm474 already keeps
alive across a per-op failure. The fix keeps it alive too (``graph.close(close_control=False)``),
tearing it down only at the genuine mission teardown. These tests reproduce the frozen stream on
CPU with a faithful host producer (the REAL ``IncorporationAckState``) + a fake relay that models
the reader-thread EOF, and prove that a kept-alive relay keeps the ack stream fresh.

Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= to dodge the box's ROS pytest plugins.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.native_channel import StationClient, StationAckStreamStalled  # noqa: E402
from twin.restock_host.incorporation_ack import IncorporationAckState  # noqa: E402

# Small, deterministic frozen-stamp budgets for the CPU test (the production defaults are 2.0 s
# sim / 45 s wall). Passed to the StationClient ctor; the env is cleared so it cannot shadow them.
ACK_FROZEN_S = 1.0
ACK_FROZEN_WALL_S = 5.0


class FakeStationRelay:
    """CPU model of control_ipc.HostControlPublisher's station path + the host ack producer.

    - ``publish(value)`` is the Isaac->host station/command seam: it feeds the REAL
      ``IncorporationAckState`` (prepare/heartbeat/stop), as the real relay subprocess does over DDS.
    - ``pump()`` is the host's sim-time timer tick: it reads a fresh snapshot (stamp = the sim clock)
      and -- WHILE the relay is alive -- caches it as the latest forwarded ``/restock_sim/station/status``.
    - ``read()`` hands the StationClient the latest cached snapshot (a NEW dict each call, matching
      ``json.loads``), or None.
    - ``close()`` models ``host_control.close()`` / the reader-thread EOF: ``pump`` stops updating, so
      ``read`` returns the FROZEN last snapshot forever -- the exact relay-teardown bug.
    """

    def __init__(self, host, clock):
        self.host, self.clock = host, clock
        self._latest = None
        self.alive = True
        self.closed = False

    # --- StationClient channel seam (JsonChannel.read/publish operate on dicts) ---
    def publish(self, value):
        if self.alive:
            self.host.accept(dict(value), self.clock())

    def read(self):
        return dict(self._latest) if self._latest is not None else None

    # --- host sim-time timer: produce + forward a fresh snapshot while connected ---
    def pump(self):
        snap = self.host.snapshot(self.clock(), endpoints_ready=True, process_alive=True)
        if self.alive:
            self._latest = snap

    # --- control_ipc.HostControlPublisher.close seam (what IsaacRosGraph.close calls) ---
    def close(self):
        self.alive = False
        self.closed = True


def _harness():
    """A StationClient reading a parked-mapper ack through the fake relay, on one sim + wall clock."""
    sim = [0.0]
    wall = [0.0]
    host = IncorporationAckState(wall_clock=lambda: wall[0])
    relay = FakeStationRelay(host, clock=lambda: sim[0])
    client = StationClient(relay, wall_clock=lambda: wall[0], stage_clock=lambda: sim[0],
                           ack_frozen_s=ACK_FROZEN_S, ack_frozen_wall_s=ACK_FROZEN_WALL_S)
    return sim, wall, host, relay, client


def _drive_to_fresh(sim, wall, relay, client, ticks=5):
    """Open the epoch and drive a few healthy ticks so the client has accepted a fresh,
    advancing ack (host stays 'warming'; stamp advances every tick)."""
    client.begin('restock_camera_arm', sim[0])     # sends the first prepare -> host opens epoch 1
    for _ in range(ticks):
        sim[0] += 0.1
        wall[0] += 0.5
        relay.pump()                                # host publishes a fresh snapshot at this sim stamp
        client.tick(sim[0], parked=True)
    assert client.latest is not None and client.latest['epoch'] == 1
    assert client.latest['stamp'] == pytest.approx(sim[0])   # the ack stamp tracks the sim clock


def _clear_env(monkeypatch):
    for key in ('RESTOCK_STATION_ACK_FROZEN_S', 'RESTOCK_STATION_ACK_FROZEN_WALL_S',
                'RESTOCK_STAGE_UNAVAILABLE_S', 'RESTOCK_HEARTBEAT_UNAVAILABLE_S'):
        monkeypatch.delenv(key, raising=False)


def test_closed_relay_freezes_the_station_ack_and_trips_the_watchdog(monkeypatch):
    # REPRODUCTION: tearing the relay down mid-stage (the bug) freezes the forwarded ack; the client
    # reads the last stamp forever while the stage clock advances -> StationAckStreamStalled.
    _clear_env(monkeypatch)
    sim, wall, host, relay, client = _harness()
    _drive_to_fresh(sim, wall, relay, client)

    client.stop()        # ros_manipulation.stop() nulls self.latest (the retry churn re-enters here)
    relay.close()        # graph.close() -> host_control.close(): reader-thread EOF, read() now frozen

    with pytest.raises(StationAckStreamStalled):
        for _ in range(400):
            sim[0] += 0.1
            wall[0] += 1.0
            client.latest = None             # the retry re-enters ros_manipulation.stop() each failed
            relay.pump()                     # step -> station.stop() nulls latest; relay stays frozen
            client.tick(sim[0], parked=True)
    assert relay.closed                      # the stall is a consequence of the torn-down relay


def test_kept_alive_relay_keeps_the_ack_fresh_after_a_per_op_failure(monkeypatch):
    # THE FIX: keep the relay alive across the per-op failure. The host keeps publishing fresh snapshots
    # (advancing stamp), the client keeps reading them, and the frozen-stamp watchdog NEVER trips -- so
    # the mission can surface the TRUE stage reason instead of a downstream StationAckStreamStalled.
    _clear_env(monkeypatch)
    sim, wall, host, relay, client = _harness()
    _drive_to_fresh(sim, wall, relay, client)

    client.stop()        # same per-op client stop; but the relay is NOT closed (the fix)
    assert relay.alive

    last_stamp = None
    for _ in range(400):                     # far beyond both frozen budgets (1 s sim / 5 s wall)
        sim[0] += 0.1
        wall[0] += 1.0
        client.latest = None                 # same retry churn as the repro (re-null each failed step)
        relay.pump()                         # relay alive -> keeps forwarding advancing-stamp snapshots
        client.tick(sim[0], parked=True)     # must not raise StationAckStreamStalled
        now_stamp = relay.read()['stamp']
        assert last_stamp is None or now_stamp > last_stamp   # the ack stamp keeps advancing
        last_stamp = now_stamp


def test_genuine_host_death_still_trips_the_watchdog(monkeypatch):
    # The fix must NOT weaken the watchdog for a GENUINE producer death (run-335: a sibling host IPC
    # teardown froze the still-delivering publisher). With the relay alive but the host STOPPED
    # advancing its snapshot (its sim-time timer wedged), the stamp freezes and the watchdog still fires.
    _clear_env(monkeypatch)
    sim, wall, host, relay, client = _harness()
    _drive_to_fresh(sim, wall, relay, client)

    client.stop()
    relay.pump = lambda: None                # host wedged: no fresh snapshot, but the relay is alive

    with pytest.raises(StationAckStreamStalled):
        for _ in range(400):
            sim[0] += 0.1
            wall[0] += 1.0
            client.latest = None             # retry churn re-nulls latest (as above)
            relay.pump()
            client.tick(sim[0], parked=True)
    assert relay.alive                       # not our teardown -- a real frozen producer
