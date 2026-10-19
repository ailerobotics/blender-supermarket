"""CPU tests for the fm378 release-jump probe pure core (no Isaac)."""
import subprocess

import pytest

from twin.tools import release_jump_probe as probe


def test_vram_used_mib_parses_nvidia_smi(monkeypatch):
    class R:
        stdout = '8192\n'
    monkeypatch.setattr(probe.subprocess, 'run', lambda *a, **k: R())
    assert probe.vram_used_mib() == 8192


def test_vram_used_mib_none_on_failure(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError('no nvidia-smi')
    monkeypatch.setattr(probe.subprocess, 'run', boom)
    assert probe.vram_used_mib() is None


def test_trace_rejects_unknown_phase():
    t = probe.ReleaseTrace()
    with pytest.raises(ValueError):
        t.add('bogus', 1.0, 100)


def test_trace_used_filters_phase_and_none():
    t = probe.ReleaseTrace()
    t.add('welded', 1.0, 5000)
    t.add('welded', 2.0, None)        # dropped (no reading)
    t.add('post_detach', 3.0, 9000)
    assert t.used('welded') == [5000]
    assert t.used('post_detach') == [9000]


def test_verdict_detects_jump_over_threshold():
    t = probe.ReleaseTrace()
    for i in range(3):
        t.add('welded', i, 8000)
    t.add('detach', 4, 8000)
    t.add('post_detach', 5, 9200)
    t.add('post_detach', 6, 10700)    # +2.7 GB peak after detach
    v = probe.verdict(t, threshold_mib=1000)
    assert v['detected'] is True
    assert v['baseline_mib'] == 8000 and v['baseline_phase'] == 'welded'
    assert v['peak_after_detach_mib'] == 10700
    assert v['delta_mib'] == 2700


def test_verdict_no_jump_under_threshold():
    t = probe.ReleaseTrace()
    t.add('welded', 0, 8000)
    t.add('post_detach', 1, 8200)     # +200 MiB only
    v = probe.verdict(t, threshold_mib=1000)
    assert v['detected'] is False and v['delta_mib'] == 200


def test_verdict_falls_back_to_preweld_baseline_when_no_welded():
    t = probe.ReleaseTrace()
    t.add('pre_weld', 0, 5000)
    t.add('post_detach', 1, 7500)
    v = probe.verdict(t, threshold_mib=1000)
    assert v['baseline_phase'] == 'pre_weld' and v['baseline_mib'] == 5000
    assert v['detected'] is True and v['delta_mib'] == 2500


def test_verdict_none_when_no_post_samples():
    t = probe.ReleaseTrace()
    t.add('welded', 0, 8000)
    v = probe.verdict(t)
    assert v['peak_after_detach_mib'] is None and v['delta_mib'] is None and v['detected'] is False


def test_format_verdict_mentions_detection_and_delta():
    v = probe.verdict_from = {'detected': True, 'baseline_mib': 8000, 'baseline_phase': 'welded',
                              'peak_after_detach_mib': 10700, 'delta_mib': 2700, 'threshold_mib': 1000}
    s = probe.format_verdict(v)
    assert 'RELEASE_JUMP detected=True' in s and 'delta=2700' in s
