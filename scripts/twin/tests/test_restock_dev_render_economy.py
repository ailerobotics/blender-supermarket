"""DEV render-economy stride resolution (CPU; no Isaac/ROS/GPU).

`isaac_bundle.resolve_render_strides` is the pure contract that decides the walkthrough
(`record_stride`) and sensing (`sensing_stride`) render cadence a dev run uses. The
devloop opt-in (`SUPERMARKET_DEV_RENDER_ECONOMY`) widens both to cut the dominant
whole-scene render cost; with the knob unset a production run must be byte-identical.
These are analytic fixtures over the resolver, not a renderer/physics demonstration.
"""
from scripts.twin.restock.isaac_bundle import (
    resolve_render_strides,
    _DEV_ECONOMY_RECORD,
    _DEV_ECONOMY_SENSING,
    _DEV_ECONOMY_SENSING_CAP,
)


def test_default_is_unchanged_when_the_knob_is_absent():
    cfg = {'record_stride': 2, 'sensing_stride': 3}
    assert resolve_render_strides(cfg, {}) == (2, 3, None)


def test_defaults_apply_when_config_omits_the_strides():
    # The production defaults (record 2, sensing 1) with no opt-in.
    assert resolve_render_strides({}, {}) == (2, 1, None)


def test_falsey_values_are_treated_as_off():
    cfg = {'record_stride': 2, 'sensing_stride': 3}
    for val in ('', '0', 'off', 'false', 'no', '  ', None):
        rec, sen, note = resolve_render_strides(cfg, {'SUPERMARKET_DEV_RENDER_ECONOMY': val})
        assert (rec, sen, note) == (2, 3, None), val


def test_on_applies_dev_defaults_and_a_marker_note():
    cfg = {'record_stride': 2, 'sensing_stride': 3}
    for on in ('1', 'on', 'true', 'yes', 'On', 'TRUE'):
        rec, sen, note = resolve_render_strides(cfg, {'SUPERMARKET_DEV_RENDER_ECONOMY': on})
        assert rec == _DEV_ECONOMY_RECORD
        assert sen == _DEV_ECONOMY_SENSING
        assert note is not None and 'DEV_RENDER_ECONOMY' in note
        assert f'record_stride={rec}' in note and f'sensing_stride={sen}' in note


def test_strides_only_ever_widen_never_narrow():
    # A config that already renders coarser than the dev default is left as-is (max wins):
    cfg = {'record_stride': 40, 'sensing_stride': 12}
    rec, sen, note = resolve_render_strides(cfg, {'SUPERMARKET_DEV_RENDER_ECONOMY': '1'})
    assert rec == 40                       # base coarser than dev record 20 -> kept
    assert sen == 12                       # base coarser than dev sensing 8 -> kept
    assert note is not None


def test_explicit_per_knob_spec():
    cfg = {'record_stride': 2, 'sensing_stride': 1}
    rec, sen, note = resolve_render_strides(
        cfg, {'SUPERMARKET_DEV_RENDER_ECONOMY': 'record=30,sensing=6'})
    assert (rec, sen) == (30, 6)
    assert note is not None


def test_sensing_stride_is_capped_at_the_freshness_bound():
    # Even an aggressive explicit request stays under the freshness window cap.
    cfg = {'record_stride': 2, 'sensing_stride': 1}
    rec, sen, _ = resolve_render_strides(
        cfg, {'SUPERMARKET_DEV_RENDER_ECONOMY': f'sensing={_DEV_ECONOMY_SENSING_CAP + 25}'})
    assert sen == _DEV_ECONOMY_SENSING_CAP


def test_malformed_spec_fails_closed_to_dev_defaults_not_production():
    # A garbled spec must never silently run at the slow production rate; it falls to the
    # dev defaults (the safe, faster intent of opting in at all).
    cfg = {'record_stride': 2, 'sensing_stride': 1}
    rec, sen, note = resolve_render_strides(
        cfg, {'SUPERMARKET_DEV_RENDER_ECONOMY': 'record=abc,garbage'})
    assert rec == _DEV_ECONOMY_RECORD
    assert sen == _DEV_ECONOMY_SENSING
    assert note is not None


def test_negative_and_zero_values_are_ignored():
    cfg = {'record_stride': 2, 'sensing_stride': 1}
    rec, sen, _ = resolve_render_strides(
        cfg, {'SUPERMARKET_DEV_RENDER_ECONOMY': 'record=0,sensing=-4'})
    # invalid numbers are skipped, so the dev defaults stand.
    assert rec == _DEV_ECONOMY_RECORD
    assert sen == _DEV_ECONOMY_SENSING
