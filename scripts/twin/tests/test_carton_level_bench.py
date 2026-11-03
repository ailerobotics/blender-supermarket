"""CPU proof test: the carton-level bench keeps every CARRIED segment under the 2 deg bound while
the unleveled fold tilts the carton well off level. No Isaac/ROS/GPU.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest scripts/twin/tests/test_carton_level_bench.py
"""
import os

import pytest

from twin.restock_manipulation import carton_level as cl

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin16/mobile-import.urdf'
pytestmark = pytest.mark.skipif(not os.path.exists(URDF), reason='mission URDF not present')


def test_bench_before_tilts_after_is_level_and_clear():
    from twin.tools import carton_level_bench as b
    u, res = b.run(URDF, bound_deg=cl.DEFAULT_TILT_BOUND_DEG)
    assert {r.name for r in res} == {'lift', 'tuck', 'pan', 'preplace', 'place'}
    # the unleveled fold really does tilt the carton off level somewhere in the carry
    assert max(r.before_max_tilt for r in res) > 10.0
    # the level constraint holds every carried segment under the bound AND in the clear wrist region
    for r in res:
        assert r.after_max_roll <= cl.DEFAULT_TILT_BOUND_DEG, r.name
        assert r.after_max_pitch <= cl.DEFAULT_TILT_BOUND_DEG, r.name
        assert r.after_in_clear_wrist, f'{r.name} leveled out of the clear wrist region'
    assert b.all_pass(res)


def test_bench_tighter_bound_still_holds():
    from twin.tools import carton_level_bench as b
    _, res = b.run(URDF, bound_deg=1.0, n=12)
    assert all(r.after_max_roll <= 1.0 and r.after_max_pitch <= 1.0 for r in res)
