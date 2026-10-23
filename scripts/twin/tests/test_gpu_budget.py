"""CPU tests for the advisory VRAM budget gate (twin.restock.gpu_budget)."""
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
