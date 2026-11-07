"""CPU tests for the advisory VRAM budget gate (twin.restock.gpu_budget)."""
import json
import subprocess
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import gpu_budget as g


def test_single_arm_feed_fits_the_card_with_whispr_resident():
    # The transfer2-equivalent config: one full-res arm masked-depth feed, mast off, whispr resident.
    v = g.check_budget([(1280, 720)])
    assert v.fits
    # Estimate anchored to the measured peak (11882 MiB used on the 12 GB card).
    assert v.estimated_mib == pytest.approx(11882.0, abs=1.0)
    assert v.detail['num_feeds'] == 1


def test_two_full_res_feeds_do_not_fit():
    # Enabling a second full-res feed (the opt-in mast camera_1) blows the card -> gate refuses.
    v = g.check_budget([(1280, 720), (1280, 720)])
    assert not v.fits
    assert v.free_after_mib < v.margin_mib


def test_lowering_arm_resolution_cuts_the_estimate():
    full = g.check_budget([(1280, 720)]).estimated_mib
    half = g.check_budget([(640, 360)]).estimated_mib
    assert half < full                      # the config lever (smaller feed) reduces pressure
    assert g.feed_mib(640, 360) < g.feed_mib(1280, 720)


def test_resident_services_consume_headroom():
    # Reserving whispr (default) leaves less than treating the card as empty of residents.
    with_whispr = g.check_budget([(1280, 720)]).free_after_mib
    without = g.check_budget([(1280, 720)], resident_mib=0).free_after_mib
    assert without - with_whispr == pytest.approx(g.WHISPR_MIB, abs=1.0)


def test_feeds_from_config_reads_enabled_blocks_and_resolution():
    cfg = {'masked_depth': {'enabled': True},
           'mast_depth': {'enabled': True, 'resolution': [640, 480]}}
    feeds = g.feeds_from_config(cfg)
    assert feeds == [(1280, 720), (640, 480)]                 # arm default + mast override


def test_feeds_from_config_skips_disabled_and_absent():
    assert g.feeds_from_config({'masked_depth': {'enabled': False}}) == []
    assert g.feeds_from_config({}) == []
    # mast enabled but arm disabled -> only the mast feed is counted
    assert g.feeds_from_config({'mast_depth': {'enabled': True}}) == [(1280, 720)]


def test_feeds_from_config_applies_economy_resolution_scale():
    # An economy resolution_scale shrinks the counted feed resolution (self-consistent with the camera).
    cfg = {'masked_depth': {'enabled': True, 'economy': {'resolution_scale': 0.5}}}
    assert g.feeds_from_config(cfg) == [(640, 360)]
    # publish_interval_s alone (no scale) leaves the resolution at the default.
    cfg2 = {'masked_depth': {'enabled': True, 'economy': {'publish_interval_s': 0.1}}}
    assert g.feeds_from_config(cfg2) == [(1280, 720)]


def test_feed_requires_positive_resolution():
    with pytest.raises(ValueError):
        g.feed_mib(0, 720)


def test_cli_verdict_exit_code(capsys):
    assert g._main(['--feed', '1280x720']) == 0                # fits
    assert g._main(['--feed', '1280x720', '--feed', '1280x720']) == 1   # over budget
    out = capsys.readouterr().out
    assert 'GPU_BUDGET fits=True' in out and 'GPU_BUDGET fits=False' in out


_SCRIPT = Path(__file__).resolve().parents[1] / 'restock' / 'gpu_budget.py'


def test_runs_by_file_path_with_a_config(tmp_path):
    # Invoking the module BY PATH (not ``-m``) must still work: feeds_from_config pulls in the
    # masked_depth_economy sibling, which has no parent package in this invocation, so a naive
    # relative import would ImportError. Run it from an unrelated cwd with a clean PYTHONPATH so
    # the script's own sys.path repair is the only thing that can satisfy the import.
    cfg = tmp_path / 'mission.json'
    cfg.write_text(json.dumps({'masked_depth': {'enabled': True}}))
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), str(cfg)],
        cwd=str(tmp_path), env={'PATH': '/usr/bin:/bin'},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert 'GPU_BUDGET fits=True' in proc.stdout
    assert 'num_feeds' in proc.stdout and "'num_feeds': 1" in proc.stdout


# --------------------------------------------------------------------------- transfer model (sim-transfer-vram-preshed-v1)
# The cw93 config: live_pose transfer, arm masked-depth feed at economy resolution_scale 0.5, mast off.
_CW93_CONFIG = {
    'masked_depth': {'enabled': True, 'economy': {'resolution_scale': 0.5, 'publish_interval_s': 0.1}},
    'mast_depth': {'enabled': False},
}


def test_transfer_lean_base_decomposition_preserves_the_legacy_anchor():
    # lean base + decode cameras + collect ESDF bounds == the measured transfer mission base.
    assert (g.TRANSFER_LEAN_BASE_MIB + g.DECODE_CAMERAS_MIB
            + g.ESDF_COLLECT_BOUNDS_MIB) == pytest.approx(g.BASE_MISSION_MIB, abs=1e-6)


def test_cw93_transfer_is_refused_without_the_preshed():
    """The cw93 transfer modelled as it ran: decode cameras + collect ESDF bounds resident AND the full
    planner-retry working set -> predicted peak over the card (reproduces the 13.1 GB SIGSEGV)."""
    v = g.check_transfer_budget(_CW93_CONFIG, preshed=False, spike_guard=False)
    assert not v.fits
    assert v.free_at_peak_mib < 0                               # over the whole card
    # The modelled peak matches the measured crash (gpuVRAM_0 = 13142 MiB) within the model tolerance.
    assert v.predicted_peak_mib == pytest.approx(13142.0, abs=120.0)


def test_cw93_transfer_is_accepted_with_the_preshed_and_spike_guard():
    v = g.check_transfer_budget(_CW93_CONFIG, preshed=True, spike_guard=True)
    assert v.fits
    assert v.free_at_peak_mib >= v.margin_mib                   # clears card - 1 GB
    # the post-shed STEADY headroom also clears the >= 1.5 GB the transfer render must boot into
    assert v.detail['preshed_free_ok'] is True
    assert v.free_at_steady_mib >= g.PRESHED_MIN_FREE_MIB


def test_preshed_alone_without_spike_guard_still_refused():
    # The pre-shed opens steady headroom but the UNGUARDED planner spike still breaches card - 1 GB:
    # both mitigations are required (so the ON fragment enables both).
    v = g.check_transfer_budget(_CW93_CONFIG, preshed=True, spike_guard=False)
    assert not v.fits
    assert v.detail['preshed_free_ok'] is True                  # steady is fine; the spike is the problem


def test_transfer_margin_is_card_minus_1gb():
    assert g.TRANSFER_PEAK_MARGIN_MIB == pytest.approx(1024.0)
    v = g.check_transfer_budget(_CW93_CONFIG, preshed=True, spike_guard=True)
    # fits iff predicted peak <= card - margin
    assert (v.predicted_peak_mib <= v.detail['card_mib'] - v.margin_mib) == v.fits


def test_preshed_drops_the_mast_feed_from_the_transfer_peak():
    cfg = dict(_CW93_CONFIG, mast_depth={'enabled': True})
    with_mast = g.check_transfer_budget(cfg, preshed=False, spike_guard=False)
    preshed = g.check_transfer_budget(cfg, preshed=True, spike_guard=True)
    assert with_mast.detail['mast_feed_mib'] > 0
    assert preshed.detail['mast_feed_mib'] == 0                 # the secondary feed is shed


def test_transfer_cli_exit_codes(capsys, tmp_path):
    cfg = tmp_path / 'cw93.json'
    cfg.write_text(json.dumps(_CW93_CONFIG))
    assert g._main(['--transfer', str(cfg)]) == 1                               # refused (no mitigation)
    assert g._main(['--transfer', '--preshed', '--spike-guard', str(cfg)]) == 0  # accepted
    out = capsys.readouterr().out
    assert 'GPU_TRANSFER_BUDGET fits=False' in out and 'GPU_TRANSFER_BUDGET fits=True' in out


def test_by_path_and_by_module_agree(tmp_path):
    # The by-path repair must not change the verdict: -m and the file-path form print the same line.
    cfg = tmp_path / 'mission.json'
    cfg.write_text(json.dumps({'masked_depth': {'enabled': True},
                               'mast_depth': {'enabled': True}}))
    scripts_root = str(_SCRIPT.resolve().parents[2])
    by_path = subprocess.run(
        [sys.executable, str(_SCRIPT), str(cfg)],
        cwd=str(tmp_path), env={'PATH': '/usr/bin:/bin'},
        capture_output=True, text=True)
    by_module = subprocess.run(
        [sys.executable, '-m', 'twin.restock.gpu_budget', str(cfg)],
        cwd=str(tmp_path), env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': scripts_root},
        capture_output=True, text=True)
    assert by_path.returncode == by_module.returncode == 1, (by_path.stderr, by_module.stderr)
    assert by_path.stdout == by_module.stdout
