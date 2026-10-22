"""sim-video-speedup-v1: the opt-in simulation speed profile (twin.restock.sim_speed) and its wiring
into IsaacPort / the BT snapshot writer / mission.json. CPU only (fake Isaac world), no GPU."""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec
from twin.navigation.sensors import Scan
from twin.restock import sim_speed as S
from twin.restock.isaac_port import IsaacPort, SENSING_ACTIONS
from twin.restock.isaac_bundle import idle_camera_set
from twin.restock.ground_truth_verify import mission_modes


# ---------------------------------------------------------------- config resolution
def test_absent_config_is_legacy():
    for config in ({}, {'sim_speed': None}, {'sim_speed': False}, {'sim_speed': {'profile': 'off'}}):
        speed = S.from_config(config)
        assert speed == S.SimSpeed()
        assert not speed.active
        assert S.pacing_legacy(speed)
        assert S.summary_fields(config) == {}


def test_max_profile_turns_every_lever_on():
    speed = S.from_config({'sim_speed': {'profile': 'max'}})
    assert speed.active and speed.profile == 'max'
    assert speed.pause_idle_cameras is True
    assert speed.pacing == 'host_coupled' and speed.host_stage_max_ratio == 1.0
    assert speed.walkthrough_hz == 3.0 and speed.trace_probe is False
    assert speed.bt_snapshot_hz == 5.0 and speed.tick_profile_s == 10.0
    assert S.from_config({'sim_speed': 'max'}) == speed           # shorthand string form


def test_overrides_on_top_of_profile():
    speed = S.from_config({'sim_speed': {'profile': 'max', 'walkthrough_hz': 2, 'pacing': 'off',
                                         'host_stage_max_ratio': 2.5}})
    assert speed.walkthrough_hz == 2.0 and speed.pacing == 'off' and speed.host_stage_max_ratio == 2.5
    assert speed.pause_idle_cameras is True                        # rest of the profile kept
    # A single lever without a profile is a 'custom' run, labelled as such.
    custom = S.from_config({'sim_speed': {'tick_profile_s': 5}})
    assert custom.active and custom.profile == 'off' and custom.tick_profile_s == 5.0
    assert S.summary_fields({'sim_speed': {'tick_profile_s': 5}})['sim_speed_mode'] == 'custom'


@pytest.mark.parametrize('block, match', [
    ({'profile': 'turbo'}, 'profile'),
    ({'profile': 'max', 'pacng': 'off'}, 'unknown keys'),
    ({'pacing': 'fast'}, 'pacing'),
    ({'walkthrough_hz': 0}, 'walkthrough_hz'),
    ({'walkthrough_hz': True}, 'walkthrough_hz'),
    ({'host_stage_max_ratio': -1}, 'host_stage_max_ratio'),
    ({'host_stage_max_ratio': float('nan')}, 'host_stage_max_ratio'),
    ({'pause_idle_cameras': 'yes'}, 'pause_idle_cameras'),
    ({'tick_profile_s': -3}, 'tick_profile_s'),
])
def test_invalid_config_fails_closed(block, match):
    with pytest.raises(ValueError, match=match):
        S.from_config({'sim_speed': block})
    with pytest.raises(ValueError):
        S.from_config({'sim_speed': ['max']})


def test_summary_fields_and_mission_modes_label_the_profile():
    fields = S.summary_fields({'sim_speed': {'profile': 'max'}})
    assert fields['sim_speed_mode'] == 'max'
    assert fields['sim_speed']['pacing'] == 'host_coupled'
    json.dumps(fields, allow_nan=False)                            # mission.json-serialisable
    assert mission_modes({'sim_speed': {'profile': 'max'}})['sim_speed_mode'] == 'max'
    assert 'sim_speed_mode' not in mission_modes({})               # byte-identical when off


def test_walkthrough_stride_only_widens():
    dt = 1 / 60
    off = S.SimSpeed()
    assert S.widen_record_stride(6, off, dt) == 6                  # no lever => unchanged
    hz3 = S.from_config({'sim_speed': {'walkthrough_hz': 3}})
    assert S.widen_record_stride(6, hz3, dt) == 20                 # 3 Hz sim = every 20th 60 Hz tick
    assert S.widen_record_stride(30, hz3, dt) == 30                # never records MORE often
    hz60 = S.from_config({'sim_speed': {'walkthrough_hz': 120}})
    assert S.widen_record_stride(1, hz60, dt) == 1


# ---------------------------------------------------------------- pacing
def test_host_pacer_never_delays_in_process_stages():
    pacer = S.HostStagePacer(1.0)
    for action in ('couple', 'outbound', 'park', 'return', None):
        assert all(pacer.delay(action, 1 / 60, 100.0) == 0.0 for _ in range(50))


def test_host_pacer_caps_host_coupled_stages_at_ratio():
    dt = 1 / 60
    for ratio in (1.0, 2.0):
        pacer = S.HostStagePacer(ratio)
        wall = 0.0
        for _ in range(600):                                       # 10 s of sim, zero-cost ticks
            wall += pacer.delay('transfer', dt, wall)
        # Ticks cost nothing, so the pacer alone sets the wall: 10 s sim / ratio (+ first tick free).
        assert wall == pytest.approx(10.0 / ratio - dt / ratio, abs=1e-6)


def test_host_pacer_burst_credit_is_bounded_after_a_slow_stretch():
    dt = 1 / 60
    pacer = S.HostStagePacer(1.0, burst_s=0.25)
    pacer.delay('collect', dt, 0.0)
    # A 30 s wall stall (a slow render) banks at most burst_s of credit ...
    wall = 30.0
    free = 0
    while pacer.delay('collect', dt, wall) == 0.0:
        free += 1
    # ... so only ~burst_s of sim runs unpaced before the real-time cap bites again.
    assert free * dt == pytest.approx(0.25 + dt, abs=dt)


def test_host_pacer_drops_credit_when_leaving_the_stage():
    dt = 1 / 60
    pacer = S.HostStagePacer(1.0)
    for _ in range(10):
        pacer.delay('transfer', dt, 0.0)
    assert pacer.delay('outbound', dt, 0.0) == 0.0
    assert pacer.delay('transfer', dt, 0.0) == 0.0                 # fresh stage: first tick free


def test_host_pacer_wait_sleeps_the_delay():
    slept = []
    clock = iter([0.0, 0.0, 0.0])
    pacer = S.HostStagePacer(1.0, clock=lambda: next(clock), sleep=slept.append)
    pacer.wait('verify', 0.5)
    pacer.wait('verify', 0.5)
    assert slept == [pytest.approx(0.5)] and pacer.slept_s == pytest.approx(0.5)


# ---------------------------------------------------------------- tick profiler
def test_tick_profiler_reports_sections_and_ratio():
    t = [0.0]
    lines = []
    prof = S.TickProfiler(1.0, clock=lambda: t[0], emit=lines.append)
    sim = 0.0
    for i in range(120):                                           # 2 s of sim at 60 Hz
        prof.begin()
        t[0] += 0.010
        prof.mark('step')
        t[0] += 0.005
        prof.mark('lidar')
        sim += 1 / 60
        prof.end(sim, rendered=(i % 20 == 0), action='couple')
        t[0] += 0.002                                              # mission logic between ticks
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith('TICK_PROFILE action=couple')
    assert 'step=10.0' in line and 'lidar=5.0' in line and 'mission=2.0' in line
    ratio = float(line.split('ratio=')[1].split()[0])
    assert ratio == pytest.approx((1 / 60) / 0.017, rel=0.02)


# ---------------------------------------------------------------- idle cameras
def test_idle_camera_pauser_pauses_nav_stages_and_restores_sensing():
    calls, logs = [], []
    pauser = S.IdleCameraPauser({'station:source': 'S', 'arm': 'A', 'gone': None},
                                toggle=lambda cam, on: calls.append((cam, on)) or True,
                                log=logs.append)
    assert pauser.begin_action('couple') == ['station:source', 'arm']
    assert calls == [('S', False), ('A', False)]
    assert pauser.begin_action('outbound') is None                 # idempotent: no re-toggle
    assert pauser.begin_action('transfer') == ['station:source', 'arm']
    assert calls[-2:] == [('S', True), ('A', True)]
    assert pauser.begin_action('verify') is None
    assert any('PAUSED for couple' in m for m in logs) and any('ENABLED for transfer' in m for m in logs)


def test_idle_camera_toggle_failure_is_logged_not_raised():
    logs = []

    def toggle(cam, on):
        if cam == 'broken':
            raise RuntimeError('render product destroyed')
        return cam != 'torn_down'
    applied = S.set_cameras_enabled({'a': 'ok', 'b': 'broken', 'c': 'torn_down'}, False,
                                    toggle=toggle, log=logs.append)
    assert applied == ['a']
    assert any('toggle skipped b' in m for m in logs)


def test_idle_camera_set_collects_station_arm_and_mast_never_the_chase():
    chase = object()
    sources = {'cameras': {'source': 'SRC', 'cart': 'CART'}, 'recording_camera': chase,
               'mast_depth': NS(camera='MAST')}
    manip = NS(controller=NS(_collect_bridge=NS(arm_camera='ARM'), _sensors=NS(camera='ARM2')))
    cams = idle_camera_set(sources, manip)
    assert cams == {'station:cart': 'CART', 'station:source': 'SRC', 'arm': 'ARM', 'mast': 'MAST'}
    assert chase not in cams.values()
    # Fallback arm handle; absent mast / manipulation => simply absent.
    assert idle_camera_set({'cameras': {}}, NS(controller=NS(_sensors=NS(camera='ARM2')))) == {'arm': 'ARM2'}
    assert idle_camera_set({}, None) == {}


# ---------------------------------------------------------------- BT snapshot throttle
def test_bt_snapshot_writer_throttles_on_sim_time_and_always_writes_terminal(tmp_path, monkeypatch):
    pytest.importorskip('py_trees')
    from twin.restock.bt import view as V
    writes = []
    monkeypatch.setattr(V, 'snapshot_dict', lambda ctx, root, tick=None: writes.append(tick) or {'tick': tick})
    mission = NS(terminal=False)
    ctx = NS(now=0.0, mission=mission, bb=NS(get=lambda key: []))
    writer = V.SnapshotWriter(tmp_path, min_interval_s=0.2)
    for i in range(60):                                            # 1 s of sim at 60 Hz
        ctx.now = round((i + 1) / 60, 6)
        writer.write(ctx, None, tick=i)
    assert len(writes) == 5                                        # every ~0.2 s sim, not every tick
    mission.terminal = True
    writer.write(ctx, None, tick=60)                               # terminal: always written
    assert writes[-1] == 60
    legacy = V.SnapshotWriter(tmp_path / 'legacy')
    writes.clear()
    for i in range(10):
        legacy.write(ctx, None, tick=i)
    assert len(writes) == 10                                       # default: every tick (unchanged)


# ---------------------------------------------------------------- IsaacPort end-to-end on a fake world
class _Body:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def get_world_pose(self):
        return np.array([self.x, self.y, 0.2]), np.array([1.0, 0.0, 0.0, 0.0])

    def get_linear_velocity(self):
        return np.zeros(3)

    def get_angular_velocity(self):
        return np.zeros(3)


class _World:
    def __init__(self):
        self.current_time = 0.5
        self.renders = []

    def get_physics_dt(self):
        return 1 / 60

    def step(self, render):
        self.renders.append(bool(render))
        self.current_time += 1 / 60


class _Mobile:
    spec = MobileSpec()
    _braked = False

    def coupling_state(self, docking):
        return {'healthy': False}

    def __getattr__(self, name):                                    # brake engage/release no-ops
        return lambda *a, **k: None


class _Range:
    coupled = workspace_self = carry_active = False

    def read(self, now):
        beams = 72
        return Scan(now, tuple([10.0] * beams), -math.pi, 2 * math.pi / beams, 10.0)


class _Recorder:
    def __init__(self, root):
        self.directory = root / 'mission-frames'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames = 0

    def capture(self, frame):
        self.frames += 1


def _fake_port(tmp_path, *, sim_speed=None, idle_cameras=None, realtime=False):
    names = ('left_wheel_joint', 'right_wheel_joint')
    contact = NS(sim_attach_contact_ready=lambda: False, sim_attach_weld=lambda: None, read=lambda: False)
    recorder = _Recorder(tmp_path)
    port = IsaacPort(world=_World(), robot=NS(dof_names=names, get_joint_positions=lambda: np.zeros(2)),
                     robot_body=_Body(-5.0, 16.0), cart_body=_Body(-6.2, 16.0), mobile=_Mobile(),
                     range_sensor=_Range(), transport=NS(contains=lambda positions: True),
                     contact=contact, hold_arm=lambda: None, verification=lambda now: None,
                     recording_camera=NS(get_current_frame=lambda: {}), recorder=recorder,
                     record_stride=4, realtime=realtime, sim_speed=sim_speed, idle_cameras=idle_cameras)
    return port, recorder


def _run(port, action, ticks):
    port.begin_action(action, {})
    return [port.advance(1 / 60) for _ in range(ticks)]


def _trace(tmp_path):
    rows = [json.loads(line) for line in (tmp_path / 'nav-trace.jsonl').read_text().splitlines()]
    for row in rows:
        row.pop('wall')                                             # the only nondeterministic field
    return rows


def test_port_legacy_default_is_unchanged(tmp_path):
    port, recorder = _fake_port(tmp_path)
    assert port._host_pacer is None and port._idle_cameras is None and port._profiler is None
    packets = _run(port, 'couple', 12)
    assert [p.simulation_time for p in packets] == pytest.approx([i / 60 for i in range(1, 13)])
    assert port.world.renders == [i % 4 == 0 for i in range(1, 13)]
    rows = _trace(tmp_path)
    assert len(rows) == 12 and all('probe' in r for r in rows)      # the legacy probe still written
    assert recorder.frames == 3


def test_port_max_profile_same_packets_and_renders_trace_without_probe(tmp_path):
    legacy_dir, max_dir = tmp_path / 'legacy', tmp_path / 'max'
    legacy, _ = _fake_port(legacy_dir)
    fast, _ = _fake_port(max_dir, sim_speed=S.from_config({'sim_speed': {'profile': 'max', 'tick_profile_s': None}}))
    a, b = _run(legacy, 'outbound', 30), _run(fast, 'outbound', 30)
    # Same physics stepping, same render decisions, same packets: the levers change no decision.
    assert legacy.world.renders == fast.world.renders
    assert [(p.simulation_time, p.parked, p.coupled, p.transport_ready) for p in a] == \
           [(p.simulation_time, p.parked, p.coupled, p.transport_ready) for p in b]
    old, new = _trace(legacy_dir), _trace(max_dir)
    assert [{k: v for k, v in r.items() if k != 'probe'} for r in old] == new   # only the probe dropped


def test_port_max_profile_disables_legacy_pacer_and_pauses_idle_cameras(tmp_path):
    toggles = []

    class Cam:
        def __init__(self, name):
            self.name = name

        def set_render_products_enabled(self, enabled):            # unused: the toggle is native_cameras'
            raise AssertionError

    speed = S.from_config({'sim_speed': {'profile': 'max'}})
    cams = {'station:source': Cam('source'), 'arm': Cam('arm')}
    port, _ = _fake_port(tmp_path, sim_speed=speed, idle_cameras=cams, realtime=True)
    assert port.realtime is False                                   # legacy absolute pacer removed
    assert isinstance(port._host_pacer, S.HostStagePacer)
    port._idle_cameras._toggle = lambda cam, on: toggles.append((cam.name, on)) or True
    port.begin_action('couple', {})
    assert toggles == [('source', False), ('arm', False)]
    port.begin_action('transfer', {})
    assert toggles[-2:] == [('source', True), ('arm', True)]
    assert set(SENSING_ACTIONS) == set(S.HOST_COUPLED_ACTIONS)


def test_port_tick_profile_line(tmp_path, capsys):
    speed = S.from_config({'sim_speed': {'tick_profile_s': 0.25}})
    port, _ = _fake_port(tmp_path, sim_speed=speed)
    _run(port, 'park', 40)
    out = capsys.readouterr().out
    assert 'TICK_PROFILE action=park' in out
    for section in ('step=', 'lidar=', 'trace=', 'ratio='):
        assert section in out


def test_sim_speed_is_a_registered_runtime_key_for_the_activation_preflight():
    # The orchestrator can `--require sim_speed` so a host/runtime config mis-placement fails the boot.
    from twin.restock_manipulation.config_preflight import KEY_CONSUMER, RUNTIME, check_activation
    assert KEY_CONSUMER['sim_speed'] == RUNTIME
    assert check_activation({}, {'sim_speed': {'profile': 'max'}}, ['sim_speed']) == []
    assert 'MIS-PLACED' in check_activation({'sim_speed': {'profile': 'max'}}, {}, ['sim_speed'])[0]
