"""sim-realtime-factor-v1: the opt-out physics-step lever (twin.restock.physics_step) and its
threading into mission.json provenance + preflight. CPU only (pure resolver + check), no GPU."""
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import physics_step as P
from twin.restock.ground_truth_verify import mission_modes


# ---------------------------------------------------------------- resolver (default byte-identical)
def test_absent_is_exactly_one_sixtieth():
    for config in ({}, {'physics_dt': None}, {'other': 1}):
        dt = P.resolve_dt(config)
        assert dt == 1 / 60 and P.is_default(dt)
    assert P.summary_fields({}) == {}                     # default run: byte-identical summary
    assert mission_modes({}) == {} or 'physics_dt' not in mission_modes({})


def test_resolve_number_hz_and_dt_forms_agree():
    assert P.resolve_dt({'physics_dt': 0.025}) == 0.025
    assert P.resolve_dt({'physics_dt': {'dt': 0.025}}) == 0.025
    assert P.resolve_dt({'physics_dt': {'hz': 40}}) == pytest.approx(0.025)
    assert not P.is_default(P.resolve_dt({'physics_dt': {'hz': 40}}))


@pytest.mark.parametrize('block, match', [
    (0, 'positive'),
    (-0.01, 'positive'),
    (True, 'number of seconds'),                          # a bool is not a step
    ('x', 'number of seconds'),
    (1 / 10, 'outside'),                                  # coarser than MAX_DT (1/15)
    (1 / 500, 'outside'),                                 # finer than MIN_DT (1/240)
    ({'hz': 5}, 'outside'),                               # 0.2 s step is out of range
    ({'hz': 0}, 'positive'),
    ({'dt': 0.025, 'hz': 40}, 'exactly one'),
    ({}, 'exactly one'),
    ({'foo': 1}, 'unknown keys'),
])
def test_malformed_config_fails_closed(block, match):
    with pytest.raises(ValueError, match=match):
        P.resolve_dt({'physics_dt': block})


def test_summary_and_modes_label_only_a_raised_step():
    cfg = {'physics_dt': {'hz': 36}}
    s = P.summary_fields(cfg)
    assert s['physics_dt_mode'] == 'raised'
    assert s['physics_hz'] == pytest.approx(36) and s['physics_dt'] == pytest.approx(1 / 36)
    assert mission_modes(cfg)['physics_dt_mode'] == 'raised'      # recorded in mission.json
    assert P.summary_fields({'physics_dt': 1 / 60}) == {}         # explicit default still byte-identical


# ---------------------------------------------------------------- contact-tolerance check
def test_weld_window_is_the_binding_constraint():
    # The grasp weld descent (3 mm window) binds, not the crawling dock; the horizontal tow speed is
    # only an informational stability indicator, never a hard bound.
    dt_max, detail = P.max_safe_physics_dt(v_tow_mps=0.556)
    assert detail['binding'] == 'weld'
    assert dt_max == pytest.approx(0.5 * 0.003 / 0.05)           # 1/33.3 s
    assert detail['tow_step_mm'] == pytest.approx(1e3 * 0.556 * dt_max)
    # A chosen 1/36 step keeps the weld sample inside the half-window and the tow step under 20 mm.
    for hz, weld_mm in ((40, 1.25), (36, 1.39), (30, 1.67)):
        assert 1e3 * 0.05 * (1 / hz) == pytest.approx(weld_mm, abs=0.02)
    assert 1 / 36 <= dt_max + 1e-9 or 1 / 38 <= dt_max + 1e-9     # the 2x step sits within/at the bound


def test_check_validates_inputs():
    for bad in ('v_grasp_descent_mps', 'weld_window_m', 'safety'):
        with pytest.raises(ValueError, match=bad):
            P.max_safe_physics_dt(**{bad: 0})
    with pytest.raises(ValueError, match='v_tow_mps'):
        P.max_safe_physics_dt(v_tow_mps=-1)


def test_describe_names_a_raised_step():
    assert 'legacy' in P.describe(1 / 60)
    assert 'RAISED' in P.describe(1 / 36)
