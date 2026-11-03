"""sim-robust-vram-headroom-v1: the opt-in VRAM headroom monitor (twin.restock.vram_headroom) -- its
config resolution, the shed-order ratchet + floor + ceiling policy on a recorded VRAM timeline
(coldwin13's included), stage-safety, and mission.json provenance. CPU only, no GPU/Isaac."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import vram_headroom as V
from twin.restock.ground_truth_verify import mission_modes


# ---------------------------------------------------------------- config resolution
def test_absent_config_is_disabled_byte_identical():
    for config in ({}, {'vram_headroom': None}, {'vram_headroom': False}):
        cfg = V.from_config(config)
        assert cfg == V.VramHeadroom()
        assert cfg.active is False
        assert V.summary_fields(config) == {}


def test_enabled_shapes():
    assert V.from_config({'vram_headroom': True}).active is True
    cfg = V.from_config({'vram_headroom': {'enabled': True, 'floor_mib': 1000}})
    assert cfg.active and cfg.floor_mib == 1000.0
    cfg2 = V.from_config({'vram_headroom': {'enabled': False, 'floor_mib': 1000}})
    assert cfg2.active is False


def test_from_config_fails_closed_on_typo():
    with pytest.raises(ValueError):
        V.from_config({'vram_headroom': {'enabledd': True}})          # unknown key
    with pytest.raises(ValueError):
        V.from_config({'vram_headroom': {'floor_mib': -5}})           # invalid value
    with pytest.raises(ValueError):
        V.from_config({'vram_headroom': {'shed_order': ['bogus']}})   # unknown shed
    with pytest.raises(ValueError):
        V.from_config({'vram_headroom': {'shed_order': ['capture_rate', 'capture_rate']}})  # dup


def test_summary_fields_provenance_only_when_on():
    on = V.summary_fields({'vram_headroom': {'enabled': True}})
    assert on['vram_headroom_mode'] == 'on'
    assert on['vram_headroom']['floor_mib'] == V.FLOOR_MIB_DEFAULT
    # mission_modes merges it: off => no key; on => the mode key present.
    assert 'vram_headroom_mode' not in mission_modes({})
    assert mission_modes({'vram_headroom': {'enabled': True}})['vram_headroom_mode'] == 'on'


# ---------------------------------------------------------------- shed policy (pure ratchet)
def test_above_floor_never_sheds():
    p = V.ShedPolicy(800.0, V.DEFAULT_SHED_ORDER)
    for free in (5000, 1200, 801, 800):   # >= floor
        assert p.observe(free, set(V.DEFAULT_SHED_ORDER)) == V.Decision()
    assert p.fired == []


def test_fires_in_fixed_order_one_per_sample():
    order = ('capture_rate', 'arm_feed_economy', 'secondary_cameras')
    p = V.ShedPolicy(800.0, order)
    avail = set(order)
    assert p.observe(700, avail).fire == 'capture_rate'
    assert p.observe(600, avail).fire == 'arm_feed_economy'
    assert p.observe(500, avail).fire == 'secondary_cameras'
    # Everything fired + still below floor => ceiling named once, then low.
    assert p.observe(400, avail) == V.Decision(ceiling=True)
    assert p.observe(300, avail) == V.Decision(low=True)
    assert p.fired == list(order)


def test_stage_blocked_shed_is_skipped_then_fired_later():
    order = ('capture_rate', 'arm_feed_economy', 'secondary_cameras')
    p = V.ShedPolicy(800.0, order)
    ever = set(order)
    # arm_feed_economy blocked (e.g. transfer): capture_rate then secondary_cameras fire; arm stays
    # unfired so the ceiling is NOT named (it may fire in a later stage) -> low.
    transfer_avail = {'capture_rate', 'secondary_cameras'}
    assert p.observe(700, transfer_avail, ever).fire == 'capture_rate'
    assert p.observe(700, transfer_avail, ever).fire == 'secondary_cameras'
    assert p.observe(700, transfer_avail, ever) == V.Decision(low=True)
    # Later stage unblocks it:
    assert p.observe(700, set(order), ever).fire == 'arm_feed_economy'
    assert p.observe(700, set(order), ever) == V.Decision(ceiling=True)


def test_unregistered_shed_does_not_hold_ceiling_open():
    # Only two sheds have actuators (fireable_ever); arm_feed_economy is never registered.
    order = ('capture_rate', 'arm_feed_economy', 'secondary_cameras')
    p = V.ShedPolicy(800.0, order)
    ever = {'capture_rate', 'secondary_cameras'}
    avail = {'capture_rate', 'secondary_cameras'}
    assert p.observe(700, avail, ever).fire == 'capture_rate'
    assert p.observe(700, avail, ever).fire == 'secondary_cameras'
    # Both ever-fireable sheds fired => ceiling named even though arm_feed_economy never fired.
    assert p.observe(700, avail, ever) == V.Decision(ceiling=True)


def test_stage_safety_predicate():
    for stage in ('collect', 'transfer', 'verify'):
        assert V.shed_is_stage_safe('arm_feed_economy', stage) is False
        assert V.shed_is_stage_safe('capture_rate', stage) is True
        assert V.shed_is_stage_safe('secondary_cameras', stage) is True
    for stage in ('couple', 'outbound', 'park', 'return', None):
        assert V.shed_is_stage_safe('arm_feed_economy', stage) is True


# ---------------------------------------------------------------- monitor on a recorded timeline
def _monitor(tmp_path, actuators, floor=800.0, period=0.2):
    cfg = V.from_config({'vram_headroom': {'enabled': True, 'floor_mib': floor,
                                           'sample_period_s': period}})
    readings = {'q': []}

    def sampler():
        return readings['q'].pop(0) if readings['q'] else None
    mon = V.VramHeadroomMonitor(cfg, sampler=sampler, actuators=actuators, run_dir=tmp_path,
                                clock=lambda: 0.0, log=lambda line: None)
    return mon, readings


def test_monitor_sheds_in_order_and_writes_durable(tmp_path):
    calls = []
    actuators = {
        'capture_rate': lambda: calls.append('capture_rate') or 'record_stride 2->6',
        'secondary_cameras': lambda: calls.append('secondary_cameras') or 'teardown ok',
    }
    mon, readings = _monitor(tmp_path, actuators, floor=800.0, period=0.2)
    mon.begin_stage('transfer')
    # A coldwin13-shaped descent toward the ceiling (used, free). free drops below 800 -> sheds fire.
    timeline = [(6471, 5437), (9991, 1917), (10359, 1549), (10549, 900),
                (10545, 700), (10549, 650), (11000, 400), (11200, 300)]
    readings['q'] = [(u, f) for (u, f) in timeline]
    for i, _ in enumerate(timeline):
        mon.tick(float(i))      # 1 sim-s apart (>> the 0.2 s period) so every reading is sampled
    # capture_rate fires at the first below-floor sample (free=700), secondary_cameras at the next.
    assert calls == ['capture_rate', 'secondary_cameras']
    # The ceiling is named (both registered sheds fired, still below floor). arm not registered.
    assert mon.ceiling_reached is True
    assert mon.peak_used == 11200.0
    # Durable last reading written for the crash record.
    rec = json.loads((tmp_path / 'vram-headroom.json').read_text())
    assert rec['free_mib'] == 300.0 and rec['headroom_mib'] == 300.0
    assert rec['peak_used_mib'] == 11200.0
    assert rec['ceiling_reached'] is True
    assert rec['fired_sheds'] == ['capture_rate', 'secondary_cameras']


def test_monitor_respects_sample_period(tmp_path):
    calls = []
    actuators = {'capture_rate': lambda: calls.append('x') or 'ok'}
    mon, readings = _monitor(tmp_path, actuators, floor=800.0, period=1.0)
    mon.begin_stage('park')
    readings['q'] = [(10000, 500), (10000, 500), (10000, 500)]
    mon.tick(0.0)     # first sample -> below floor -> fire
    mon.tick(0.5)     # within the 1.0 s period -> NOT sampled
    mon.tick(1.0)     # period elapsed -> sampled, but shed already fired (ratchet, no re-fire)
    assert calls == ['x']
    assert len(readings['q']) == 1   # exactly two samples consumed


def test_monitor_above_floor_is_quiet(tmp_path):
    calls = []
    mon, readings = _monitor(tmp_path, {'capture_rate': lambda: calls.append('x') or 'ok'})
    mon.begin_stage('transfer')
    readings['q'] = [(6000, 6000), (7000, 5000)]
    mon.tick(0.0)
    mon.tick(0.2)
    assert calls == []
    assert mon.policy.fired == []
