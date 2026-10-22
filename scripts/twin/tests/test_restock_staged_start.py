"""CPU tests for the staged-start settle hold + host clock sync (twin.restock.staged_start).

Includes a replay of the start_at=transfer bench (bench1) clock timings through the REAL host checks
(RosTransferBridge._sensors packet freshness + TransferController collision freshness): the legacy packet
reproduces both bench1 failures byte-for-byte, the synced packet passes both at the default bounds, and the
offset is still 1.55 s after a 20 s hold -- so the hold alone could never have cleared them.
No Isaac/ROS/GPU imports.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.inventory import Job
from twin.restock.ros_manipulation import RosManipulationRuntime
from twin.restock.runtime import ManipulationBridge, Packet, RuntimeAdapter
from twin.restock.shortcut import shortcut_summary, walkthrough_caption
from twin.restock.staged_start import (SettleHold, is_staged_start, staged_clock_sync,
                                       staged_settle_seconds, summary_fields, sync_packet_clock)
from twin.restock_manipulation import (Observation, ProductProfile, Sensors, Target,
                                       TransferController)
from twin.restock_manipulation.ros_adapter import RosTransferBridge

FIXTURE = Path(__file__).parent / 'fixtures' / 'bench1_clock_skew.log'
DT = 1 / 60
PROFILE = ProductProfile('sku-box', (.10, .10, .10), (1., 0., 0., 0.))
JOB = dict(job_id='job-1', sku='sku-box', quantity=1, source_bin='backroom', shelf_id='shelf-a')


# --- config gating: a cold run is never touched -------------------------------------------------

def test_cold_and_collect_starts_are_never_staged_whatever_the_config_says():
    for cfg in ({}, {'start_at': None}, {'start_at': 'collect'},
                {'staged_start_settle_s': 30, 'staged_start_clock_sync': True}):
        assert not is_staged_start(cfg)
        assert staged_settle_seconds(cfg) == 0.0
        assert staged_clock_sync(cfg) is False
        assert summary_fields(cfg) == {}


def test_staged_start_defaults_and_overrides():
    for start_at in ('couple', 'transfer'):
        cfg = {'start_at': start_at}
        assert is_staged_start(cfg)
        assert staged_settle_seconds(cfg) == 20.0
        assert staged_clock_sync(cfg) is True
    cfg = {'start_at': 'transfer', 'staged_start_settle_s': 0, 'staged_start_clock_sync': False}
    assert staged_settle_seconds(cfg) == 0.0 and staged_clock_sync(cfg) is False
    assert summary_fields(cfg)['staged_start_mode'] == 'off'
    assert summary_fields({'start_at': 'transfer'}) == {
        'staged_start_settle_s': 20.0, 'staged_start_clock_sync': True,
        'staged_start_mode': 'settle hold 20s; host clock sync'}
    for bad in (-1, float('nan'), 601, 'x', True):
        with pytest.raises(ValueError):
            staged_settle_seconds({'start_at': 'transfer', 'staged_start_settle_s': bad})
    with pytest.raises(ValueError):
        staged_clock_sync({'start_at': 'transfer', 'staged_start_clock_sync': 'yes'})


def test_mission_json_and_caption_name_the_switches():
    cfg = {'start_at': 'transfer', 'start_staging': 'recorded_bay_park'}
    out = shortcut_summary({'start_at': 'transfer', 'skipped': ['collect']}, cfg)
    assert out['staged_start_mode'] == 'settle hold 20s; host clock sync'
    assert out['walkthrough_caption'].endswith('| settle hold 20s; host clock sync')
    assert walkthrough_caption({}) is None                      # cold run: no caption at all


# --- packet clock sync --------------------------------------------------------------------------

def _sensors(stamp, collision, **kw):
    obs = Observation(f'o-{stamp}', 'sensor', stamp, 'sku-box', 'rollwagen', (.4, 0., .2, 0., 0., 0., 1.), .99)
    return Sensors(stamp, (obs,), None, True, False, stamp, collision, (0.,) * 6, ('rollwagen',), stamp, **kw)


def test_sync_packet_clock_records_offset_and_moves_collision_into_packet_clock():
    synced = sync_packet_clock(_sensors(0.6167, 1.95), 1.55)
    assert synced.clock_offset == 1.55
    assert synced.collision_time == pytest.approx(0.40)
    assert synced.simulation_time == 0.6167 and synced.feedback_time == 0.6167   # packet clock untouched
    assert sync_packet_clock(_sensors(0.6, -1.0), 1.55).collision_time == -1.0   # 'no map yet' sentinel kept
    with pytest.raises(ValueError):
        sync_packet_clock(_sensors(0.6, 1.0), float('inf'))
    assert _sensors(0.6, 1.0).clock_offset == 0.0                                  # default: legacy packet


# --- settle hold --------------------------------------------------------------------------------

def _run_hold(hold, ready_from, *, until=200.0, flap_at=()):
    t = 0.0
    while t < until:
        t = round(t + DT, 6)
        ready = t >= ready_from and not any(a <= t < b for a, b in flap_at)
        if hold.update(t, ready=ready, parked=True):
            return t
    return None


def test_hold_waits_for_settle_time_and_steady_streams():
    hold = SettleHold(20.0)
    released = _run_hold(hold, ready_from=0.55)                  # bench1: station ready ~0.55 s
    assert hold.reason == 'settled' and 20.0 <= released < 20.0 + 2 * DT
    assert 'STAGED_SETTLE_HOLD released reason=settled' in hold.release_line
    assert hold.update(released + 1, ready=False, parked=False) is True       # latched


def test_hold_requires_a_continuous_steady_window_after_a_drop():
    hold = SettleHold(20.0)
    released = _run_hold(hold, ready_from=0.55, flap_at=((19.5, 19.6),))   # ack/collision went stale
    assert hold.reason == 'settled' and released >= 19.6 + 2.0 - DT        # waited a fresh 2 s window


def test_hold_is_bounded_and_releases_on_cap_when_streams_never_steady():
    hold = SettleHold(20.0)
    released = _run_hold(hold, ready_from=1e9)
    assert hold.reason == 'cap' and released == pytest.approx(80.0, abs=2 * DT)
    assert 'reason=cap' in hold.release_line


def test_hold_short_settle_clamps_the_steady_window():
    hold = SettleHold(1.0)
    assert hold.steady_s == 1.0
    with pytest.raises(ValueError):
        SettleHold(0)


# --- RuntimeAdapter warm phase ------------------------------------------------------------------

class _Owned:
    """Sensor-owning manip runtime: warm() reports the station ready from ``ready_from`` (mission clock)."""
    cart_location = 'rollwagen'

    def __init__(self, ready_from):
        self.ready_from, self.started, self.sync = ready_from, [], None

    def warm(self, t, parked=None):
        return t >= self.ready_from

    def start(self, op, job):
        self.started.append(op)

    def step(self, dt, t, parked=None):
        return NS(status='running', reason='', phase='observe', controls=(), evidence=())

    def stop(self):
        pass


def _adapter(staged_settle_s, ready_from=0.55):
    tick = [0]

    def advance(dt):
        tick[0] += 1
        return Packet(f'physics:{tick[0]}', 'test_fixture', tick[0] * DT, None, None, None, None,
                      True, True, 0., True)
    port = NS(advance=advance, controls=lambda c: None, stop=lambda: None)
    nav = NS(start=lambda *a, **k: None, step=None, cancel=lambda: None)
    mobile = NS(park=lambda: None, hold=lambda: None)
    owned = _Owned(ready_from)
    manip = ManipulationBridge(owned, owns_sensor_io=True)
    poses = {k: [0, 0, 0] for k in ('home', 'pickup', 'cart_dock', 'shelf_approach')}
    adapter = RuntimeAdapter(nav, mobile, manip, port, poses, fixture_mode=True,
                             staged_settle_s=staged_settle_s)
    return adapter, owned


def _dispatch_time(adapter, owned, *, max_s=200.0):
    adapter.start('transfer', Job('j', 'milk', 1, 'back', 'shelf'))
    while adapter.time < max_s:
        r = adapter.step(DT)
        assert r.status != 'failed', r.reason
        if owned.started:
            return adapter.time, r.reason
    return None, None


def test_staged_transfer_dispatch_waits_for_the_settle_hold(capsys):
    adapter, owned = _adapter(20.0)
    t, reason = _dispatch_time(adapter, owned)
    assert 20.0 <= t < 20.0 + 2 * DT and reason == 'collision map warm; dispatching grasp'
    assert owned.started == ['transfer']
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith('STAGED_SETTLE_HOLD')]
    assert len(lines) == 1 and 'reason=settled' in lines[0] and 'action=transfer' in lines[0]
    # the hold is latched: a re-begun (retried) transfer dispatches on the first ready tick again
    owned.started.clear()
    t0 = adapter.time
    t2, _ = _dispatch_time(adapter, owned, max_s=t0 + 5)
    assert t2 == pytest.approx(t0 + DT)


def test_no_hold_is_byte_identical_dispatch_on_first_ready():
    adapter, owned = _adapter(0.0)
    assert adapter.settle_hold is None
    t, reason = _dispatch_time(adapter, owned)
    assert t == pytest.approx(0.55, abs=DT) and reason == 'collision map warm; dispatching grasp'


def test_hold_cap_dispatches_like_the_warm_cap():
    adapter, owned = _adapter(20.0, ready_from=1e9)
    t, reason = _dispatch_time(adapter, owned)
    assert t == pytest.approx(80.0, abs=2 * DT) and reason == 'warm-up tick cap reached; dispatching grasp'


# --- Isaac runtime stamps the packet; the wrapper chain delegates -------------------------------

def _ros_runtime(offset, collision_world):
    published = []
    sim = {'t': 0.0}
    station = NS(stage_clock=lambda: sim['t'] + offset, begin=lambda *a: None, tick=lambda *a, **k: None,
                 ready=lambda now: True, collision_time=collision_world, latest=None,
                 heartbeat=lambda: None, stop=lambda: None)
    feedback = NS(collision_time=-1., parked=lambda: True, set_parked=lambda v: None)

    def capture(t, *a):
        return _sensors(t, feedback.collision_time)
    graph = NS(start=lambda *a: None, publish_sensors=published.append, poll=lambda *a: None,
               close=lambda: None)
    rt = RosManipulationRuntime(graph, capture, prepare_station=lambda *a: None,
                                station=station, feedback=feedback)
    rt.start('transfer', dict(job_id='j', source_bin='back', shelf_id='shelf'))
    return rt, published, sim


def test_runtime_publishes_raw_packet_unless_clock_sync_enabled(capsys):
    rt, published, sim = _ros_runtime(1.55, 1.95)
    sim['t'] = 0.6167
    rt.step(DT, 0.6167)
    assert published[-1].clock_offset == 0.0 and published[-1].collision_time == 1.95   # legacy, raw
    rt.set_clock_sync(True)
    sim['t'] = 0.6334
    rt.step(DT, 0.6334)
    assert published[-1].clock_offset == pytest.approx(1.55)
    assert published[-1].collision_time == pytest.approx(0.40)
    assert 'STAGED_CLOCK_SYNC op=transfer clock_offset=1.5500' in capsys.readouterr().out


def test_clock_sync_reaches_the_inner_runtime_through_the_mission_wrapper_chain():
    # run-310 family: a setter only on an outer wrapper would be a silent no-op.
    from twin.restock_manipulation.isaac_factory import HostManipulationRuntime
    rt, _, _ = _ros_runtime(1.55, 1.95)
    host = HostManipulationRuntime(runtime=rt, sensors=NS(), motion=NS(), profiles={})
    bridge = ManipulationBridge(host, owns_sensor_io=True, prepare_station=lambda *a: None)
    assert bridge.set_clock_sync(True) is True and rt.clock_sync is True
    assert ManipulationBridge(NS(cart_location='rollwagen'), owns_sensor_io=True).set_clock_sync(True) is False


# --- bench1 replay through the real host checks -------------------------------------------------

def _clock_offsets():
    """world_current_time - stage_time per section of the evidence fixture (every CLOCK_DIAG line)."""
    offsets, section = {}, None
    for line in FIXTURE.read_text().splitlines():
        if line.startswith('## '):
            section = line[3:].split('.console.log')[0]
        m = re.match(r'CLOCK_DIAG tick=\d+ stage_time=([\d.]+) world_current_time=([\d.]+)', line)
        if m:
            offsets.setdefault(section, []).append(round(float(m[2]) - float(m[1]), 4))
    return offsets


def _bench1():
    text = FIXTURE.read_text()
    stamp = float(re.search(r'PUBLISH_SENSORS contact=False stamp=([\d.]+)', text)[1])
    collision = float(re.search(r"'state': 'ready'.*?'collision_time': ([\d.]+)", text)[1])
    failed = re.search(r"MANIP_HOST_FAILED op=transfer phase='failed' reason='(.*)'", text)[1]
    return stamp, collision, failed


class _Motion:
    def begin(self, *a, **k): pass
    def begin_joint_goal(self, *a, **k): pass
    def poll(self): return 'running'
    def stop(self): pass
    def update_payload(self, held, profile): return True
    def last_fault_was_sim_deadline(self): return False


class _NodeClock:
    def __init__(self, seconds):
        self.ns = int(round(seconds * 1e9))
    def get_clock(self):
        return self
    def now(self):
        return NS(nanoseconds=self.ns)


def _host_step(sensors, node_now, *, max_age, render_lead):
    """One /restock_sim/sensors message through the REAL host path: RosTransferBridge._sensors (packet
    freshness) -> TransferController.step (collision/feedback invariant). Returns the emitted Result."""
    c = TransferController(_Motion(), {'sku-box': PROFILE}, fixture_mode=True)
    c.max_age, c.render_lead = max_age, render_lead
    c.start('transfer', JOB)
    obs = Observation(f'source-{sensors.simulation_time}', 'test_fixture', sensors.simulation_time,
                      'sku-box', c.source, (.4, 0., .2, 0., 0., 0., 1.), .99)
    tgt = Target(f'target-{sensors.simulation_time}', 'test_fixture', sensors.simulation_time,
                 c.destination, (.5, .2, .1, 0., 0., 0., 1.), True, (.3, .3, .4))
    sensors = replace(sensors, observations=(obs,), target=tgt,
                      scanned_locations=(c.source, c.destination),
                      valid_until_steady=time.monotonic() + 100.)
    b = RosTransferBridge.__new__(RosTransferBridge)
    b.node, b.controller, b.last_sim = _NodeClock(node_now), c, None
    emitted = []
    b._emit = emitted.append
    b._sensors(NS(data=json.dumps(asdict(sensors), allow_nan=False)))
    return emitted[-1]


def test_clock_offset_is_the_shortcut_settle_time_and_constant_for_the_run():
    offsets = _clock_offsets()
    assert offsets['run-330'] == [0.05]                                # cold: boot steps only
    assert offsets['run-356'] == [0.55]                                # start_at=couple: +30 settle steps
    assert set(offsets['bench1']) == {1.55} and len(offsets['bench1']) >= 4   # transfer: +90, constant


def test_bench1_replay_legacy_reproduces_both_failures_and_sync_clears_them():
    offset = _clock_offsets()['bench1'][0]
    stamp, collision_world, recorded = _bench1()
    host_now = stamp + offset                       # /clock of the same tick the packet was captured
    legacy = _sensors(stamp, collision_world)
    # attempt 1 (RESTOCK_SENSOR_MAX_AGE_S=1.5): the host packet check sees the offset as packet age
    r = _host_step(legacy, host_now, max_age=1.5, render_lead=0.5)
    assert r.status == 'failed' and 'packet simulation timestamp stale or future (now-stamp=1.550' in r.reason
    # attempt 2 (bench-only 3.0): the packet passes, the collision check trips with bench1's exact reason
    r = _host_step(legacy, host_now, max_age=3.0, render_lead=0.5)
    assert r.status == 'failed' and r.reason == recorded
    assert 'collision_dt=-1.3333' in recorded
    # staged-start clock sync: both checks measure true age and pass at the 1.5 s bound -- and even at
    # the controller's own default lead (0.2 s), since the true collision age is 0.217 s
    synced = sync_packet_clock(legacy, offset)
    for lead in (0.5, 0.2):
        r = _host_step(synced, host_now, max_age=1.5, render_lead=lead)
        assert r.status == 'running', r.reason


def test_bench1_replay_hold_alone_cannot_clear_the_offset():
    offset = _clock_offsets()['bench1'][0]
    stamp, collision_world, _ = _bench1()
    age = stamp + offset - collision_world          # 0.217 s: the ESDF was fresh at dispatch
    hold = SettleHold(20.0)
    released = _run_hold(hold, ready_from=stamp)
    assert hold.reason == 'settled'
    # after the 20 s hold the clocks still differ by the same constant offset, so the first packet at
    # release (equally fresh collision world) fails the legacy checks exactly as before ...
    legacy = _sensors(released, released + offset - age)
    r = _host_step(legacy, released + offset, max_age=1.5, render_lead=0.5)
    assert r.status == 'failed' and 'now-stamp=1.550' in r.reason
    r = _host_step(legacy, released + offset, max_age=3.0, render_lead=0.5)
    assert r.status == 'failed' and 'collision_dt=-1.3333' in r.reason
    # ... and passes once synced
    r = _host_step(sync_packet_clock(legacy, offset), released + offset, max_age=1.5, render_lead=0.2)
    assert r.status == 'running', r.reason


def _payload_gate(sensors, node_now, max_age, capsys):
    """The host payload-attach precondition gate (RosPayloadSync.__call__) with the ROS message modules
    stubbed; returns the PAYLOAD_SYNC_DIAG gate name it stopped at (scene service left not-ready)."""
    import types
    from unittest.mock import patch
    from twin.restock_manipulation.ros_payload import RosPayloadSync
    mods = {}
    for name, attrs in (('rclpy.action', ('ActionClient',)), ('rclpy.time', ('Time',)),
                        ('moveit_msgs.srv', ('ApplyPlanningScene',)),
                        ('moveit_msgs.msg', ('AttachedCollisionObject', 'CollisionObject')),
                        ('shape_msgs.msg', ('SolidPrimitive',)), ('geometry_msgs.msg', ('Pose',)),
                        ('isaac_ros_cumotion_interfaces.action', ('AttachObject',))):
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, object)
        mods[name] = mod
    for parent in ('rclpy', 'moveit_msgs', 'shape_msgs', 'geometry_msgs', 'isaac_ros_cumotion_interfaces'):
        mods.setdefault(parent, types.ModuleType(parent))
    sync = RosPayloadSync.__new__(RosPayloadSync)
    sync.node, sync.sensors, sync.attach = _NodeClock(node_now), sensors, None
    sync.controller = NS(max_age=max_age)
    sync.scene = NS(service_is_ready=lambda: False)
    capsys.readouterr()
    with patch.dict(sys.modules, mods):
        assert sync(False, PROFILE) is False
    out = capsys.readouterr().out
    return re.search(r'PAYLOAD_SYNC_DIAG gate=(\S+)', out)[1]


def test_bench1_replay_payload_attach_gate_measures_true_age_when_synced(capsys):
    # The third cross-clock host check: the payload node compares its /clock with the packet
    # feedback_time, so on the bench (offset 1.55 s) the grasp-time attach would refuse at the 1.5 s bound.
    offset = _clock_offsets()['bench1'][0]
    stamp, collision_world, _ = _bench1()
    legacy = _sensors(stamp, collision_world)
    assert _payload_gate(legacy, stamp + offset, 1.5, capsys) == 'precondition'
    synced = sync_packet_clock(legacy, offset)
    assert _payload_gate(synced, stamp + offset, 1.5, capsys) == 'scene_service'   # past the clock gate
