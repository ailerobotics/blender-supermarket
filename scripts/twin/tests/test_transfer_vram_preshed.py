"""CPU tests for the transfer-entry VRAM pre-shed + planner/render spike guard (sim-transfer-vram-preshed-v1).

No Isaac / ROS / GPU: the config resolution, the pre-shed sequencing + >= 1.5 GB gate, and the spike-guard
pause/recover/timeout policy are all pure (injected actuators, VRAM sampler, sleep and clock).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_transfer_vram_preshed.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock import spike_guard as sg
from twin.restock import transfer_preshed as tp


# --------------------------------------------------------------------------- pre-shed config
def test_preshed_absent_and_false_are_disabled():
    assert not tp.from_config({}).active
    assert not tp.from_config({'transfer_preshed': False}).active
    assert not tp.from_config({'transfer_preshed': {'enabled': False}}).active
    assert tp.summary_fields({}) == {}


def test_preshed_true_and_block_enable():
    assert tp.from_config({'transfer_preshed': True}).active
    cfg = tp.from_config({'transfer_preshed': {'enabled': True, 'min_free_mib': 2000}})
    assert cfg.active and cfg.min_free_mib == 2000
    fields = tp.summary_fields({'transfer_preshed': True})
    assert fields['transfer_preshed_mode'] == 'on'
    assert fields['transfer_preshed']['enabled'] is True


def test_preshed_rejects_unknown_key_and_bad_value_and_bad_shed():
    for block in ({'nope': 1}, {'min_free_mib': -1}, {'min_free_mib': 'x'},
                  {'order': ['not_a_shed']}, {'order': ['mast_feed', 'mast_feed']}, {'order': 5}):
        with pytest.raises(ValueError):
            tp.from_config({'transfer_preshed': block})


# --------------------------------------------------------------------------- pre-shed sequencing + gate
def _sampler(free_values):
    seq = iter(free_values)

    def sample():
        return (0.0, next(seq))
    return sample


def test_run_preshed_fires_actuators_in_order_and_skips_missing():
    cfg = tp.from_config({'transfer_preshed': {'enabled': True, 'min_free_mib': 1000}})
    fired = []
    actuators = {
        'mast_feed': lambda: fired.append('mast') or 'mast dropped',
        'decode_economy': lambda: fired.append('decode') or 'decode economy',
        # collect_capture / collect_esdf_bounds have no actuator this run -> skipped
    }
    # before 117 (cw93 steady), after 1600 (the shed opened headroom)
    res = tp.run_preshed(cfg, actuators, sampler=_sampler([117.0, 1600.0]), log=lambda *a: None)
    assert fired == ['mast', 'decode']                      # PRESHED_ORDER (mast before decode)
    assert [n for n, _ in res.fired] == ['mast_feed', 'decode_economy']
    assert set(res.skipped) == {'collect_capture', 'collect_esdf_bounds'}
    assert res.before_free_mib == 117.0 and res.after_free_mib == 1600.0 and res.ok


def test_run_preshed_raises_by_name_when_free_stays_below_floor():
    """The cw93 failure: nothing freed enough headroom -> refuse the transfer render BY NAME up front
    instead of booting it into a near-full card."""
    cfg = tp.from_config({'transfer_preshed': {'enabled': True, 'min_free_mib': 1536}})
    actuators = {'mast_feed': lambda: 'mast dropped'}
    with pytest.raises(tp.TransferPreshedInsufficientFree, match='refusing to boot the transfer render'):
        tp.run_preshed(cfg, actuators, sampler=_sampler([117.0, 300.0]), log=lambda *a: None)


def test_run_preshed_require_min_free_false_does_not_raise():
    cfg = tp.from_config({'transfer_preshed': {'enabled': True, 'min_free_mib': 1536,
                                               'require_min_free': False}})
    res = tp.run_preshed(cfg, {'mast_feed': lambda: 'x'}, sampler=_sampler([117.0, 300.0]),
                         log=lambda *a: None)
    assert res.ok and res.after_free_mib == 300.0            # below floor but not required -> proceeds


def test_run_preshed_without_sampler_cannot_gate_and_proceeds():
    cfg = tp.from_config({'transfer_preshed': {'enabled': True}})
    res = tp.run_preshed(cfg, {'mast_feed': lambda: 'x'}, sampler=None, log=lambda *a: None)
    assert res.ok and res.before_free_mib is None and res.after_free_mib is None


def test_run_preshed_actuator_error_is_logged_not_raised():
    cfg = tp.from_config({'transfer_preshed': {'enabled': True, 'min_free_mib': 100}})

    def boom():
        raise RuntimeError('seam gone')
    res = tp.run_preshed(cfg, {'mast_feed': boom}, sampler=_sampler([50.0, 200.0]), log=lambda *a: None)
    assert res.ok                                           # free cleared the floor despite the failed shed
    assert res.fired[0][0] == 'mast_feed' and 'FAILED' in res.fired[0][1]


# --------------------------------------------------------------------------- spike guard config
def test_spike_guard_absent_and_false_disabled():
    assert not sg.from_config({}).active
    assert not sg.from_config({'transfer_spike_guard': False}).active
    assert sg.summary_fields({}) == {}


def test_spike_guard_from_env():
    assert not sg.from_env({}).active
    g = sg.from_env({'RESTOCK_TRANSFER_SPIKE_GUARD': '1', 'RESTOCK_SPIKE_GUARD_FLOOR_MIB': '512'})
    assert g.active and g.floor_mib == 512


def test_spike_guard_rejects_unknown_key():
    with pytest.raises(ValueError):
        sg.from_config({'transfer_spike_guard': {'floor': 1}})


# --------------------------------------------------------------------------- spike guard pause/recover/timeout
class _Clock:
    """A fake monotonic clock advanced by the fake sleep, so the bounded wait is deterministic."""
    def __init__(self):
        self.t = 0.0

    def sleep(self, dt):
        self.t += dt

    def now(self):
        return self.t


def test_spike_guard_proceeds_immediately_when_clear():
    g = sg._resolve({'enabled': True, 'floor_mib': 400})
    clk = _Clock()
    d = g.wait_for_headroom(lambda: (0.0, 1200.0), sleep=clk.sleep, clock=clk.now, log=lambda *a: None)
    assert d.proceed and d.waited_s == 0.0 and d.reason == 'clear'


def test_spike_guard_pauses_then_resumes_when_free_recovers():
    g = sg._resolve({'enabled': True, 'floor_mib': 400, 'poll_period_s': 0.5, 'max_wait_s': 20})
    frees = iter([100.0, 150.0, 900.0])        # below floor, below floor, recovered
    clk = _Clock()
    d = g.wait_for_headroom(lambda: (0.0, next(frees)), sleep=clk.sleep, clock=clk.now,
                            log=lambda *a: None)
    assert d.proceed and d.reason == 'recovered' and d.waited_s == pytest.approx(1.0)


def test_spike_guard_fails_by_name_when_free_never_recovers__cw93_replay():
    """Replay the cw93 spike trace: free sits at the measured 117 MiB steady and the planner retries drive
    it DOWN (toward the 13.14 GB crash), never clearing the 400 MiB floor. The guard must PAUSE and then
    FAIL BY NAME at the bounded wait -- the caller aborts the plan instead of letting the allocator spike."""
    g = sg._resolve({'enabled': True, 'floor_mib': 400, 'poll_period_s': 0.5, 'max_wait_s': 3.0})
    # measured anchors: steady free 117 (vram-headroom.json), then the retry spike eats the rest.
    trace = iter([117.0] + [100.0, 60.0, 30.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

    def sampler():
        return (0.0, next(trace))
    clk = _Clock()
    d = g.wait_for_headroom(sampler, what='cuMotion plan retry', sleep=clk.sleep, clock=clk.now,
                            log=lambda *a: None)
    assert not d.proceed
    assert 'vram spike guard' in d.reason and 'stayed below' in d.reason
    assert d.waited_s >= 3.0                                 # the bounded wait elapsed


def test_spike_guard_inactive_proceeds_without_reading_vram():
    g = sg.SpikeGuard()                                     # disabled sentinel

    def sampler():
        raise AssertionError('must not read VRAM when disabled')
    d = g.wait_for_headroom(sampler, log=lambda *a: None)
    assert d.proceed and d.reason == 'inactive'


def test_spike_guard_unmeasured_proceeds():
    g = sg._resolve({'enabled': True})
    d = g.wait_for_headroom(lambda: None, log=lambda *a: None)
    assert d.proceed and d.reason == 'unmeasured'
