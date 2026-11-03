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
