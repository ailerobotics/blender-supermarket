"""CPU tests for the cuRobo MPC bench pure logic: the representative store world,
the scenario target/person paths, the clearance maths, world rebasing, the mission
joint-window builder, and -- using the REAL injected gate controller -- the gate
behaviour trial (STOP holds, resume to freshest, out-of-window refusal). No torch,
cuRobo or Isaac: the solver loop (arm_mpc_run.run_scenario) is exercised only on GPU.
"""
import math

import pytest

from twin.tools import arm_mpc_bench as bench
from twin.tools.arm_mpc_bench import (
    ARM_JOINTS, PERSON, StoreAisle, store_world, store_world_with_person,
    make_target_fn, make_person_fn, min_sphere_clearance, person_only_world,
    shift_world, nominal_reach, mission_windows, run_gate_trial,
    MpcScenarioMetrics, metrics_table, behaviour_note, SCENARIOS)


# --------------------------------------------------------------------------- gate fixture
@pytest.fixture(scope='module')
def gate():
    try:
        return bench.load_gate()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'prototype gate not importable: {exc}')


@pytest.fixture
def windows(gate):
    return mission_windows([(-2 * math.pi, 2 * math.pi)] * 6, gate)


# --------------------------------------------------------------------------- world
def test_store_world_has_shelf_and_floor():
    w = store_world()
    assert set(w['cuboid']) == {'ground', 'target_plate', 'target_back', 'opposite_shelf'}
    # floor top is below z=0 (base works above it)
    g = w['cuboid']['ground']
    assert g['pose'][2] + g['dims'][2] / 2.0 < 0.0


def test_store_world_with_person_adds_named_cuboid():
    w = store_world_with_person()
    assert PERSON in w['cuboid']
    assert w['cuboid'][PERSON]['dims'] == bench.PERSON_DIMS
    # a plan-view radius consistent with the Detection contract (half box diagonal)
    assert bench.PERSON_PLANVIEW_RADIUS == pytest.approx(math.hypot(0.5, 0.5) / 2.0, abs=1e-3)


def test_store_aisle_validate_rejects_bad_geometry():
    with pytest.raises(ValueError):
        StoreAisle(aisle_width=-1.0).validate()
    with pytest.raises(ValueError):
        StoreAisle(target_shelf_x=0.0).validate()
    StoreAisle().validate()  # defaults are valid


# --------------------------------------------------------------------------- scenarios
def test_scenarios_enumerated():
    assert SCENARIOS == ('follow', 'person', 'blocked', 'unreachable')
    for sc in SCENARIOS:
        assert behaviour_note(sc)


def test_follow_target_sweeps_in_y_bounded():
    c = (0.85, 0.0, 0.9)
    f = make_target_fn('follow', c)
    ys = [f(t)[1] for t in [i * 0.5 for i in range(32)]]
    assert max(ys) <= c[1] + 0.25 + 1e-6 and min(ys) >= c[1] - 0.25 - 1e-6
    assert max(ys) > c[1] + 0.1 and min(ys) < c[1] - 0.1  # genuinely sweeps
    # x stays fixed at the reach centre
    assert all(abs(f(t)[0] - c[0]) < 1e-9 for t in [0.0, 1.0, 3.0])


def test_unreachable_target_far_from_centre():
    c = (0.85, 0.0, 0.9)
    tx = make_target_fn('unreachable', c)(0.0)
    assert math.dist(tx, c) > 1.5


def test_person_walks_monotonically_down_aisle():
    c = (0.85, 0.0, 0.9)
    p = make_person_fn('person', c)
    ys = [p(t)[1] for t in range(0, 13)]
    assert ys == sorted(ys)          # monotonic non-decreasing
    assert ys[-1] - ys[0] > 2.0      # crosses the aisle
    # blocked parks between base and target; follow/unreachable have no person
    assert make_person_fn('blocked', c)(0.0) == make_person_fn('blocked', c)(5.0)
    assert make_person_fn('follow', c) is None
    assert make_person_fn('unreachable', c) is None


def test_scenario_needs_person():
    assert bench.scenario_needs_person('person') and bench.scenario_needs_person('blocked')
    assert not bench.scenario_needs_person('follow')
    assert not bench.scenario_needs_person('unreachable')


# --------------------------------------------------------------------------- clearance
def test_min_sphere_clearance_sign():
    world = {'cuboid': {'box': {'dims': [1.0, 1.0, 1.0], 'pose': [0, 0, 0, 1, 0, 0, 0]}}}
    # sphere centre 1.0 away along x, radius 0.2 -> gap = 1.0 - 0.5 - 0.2 = 0.3
    assert min_sphere_clearance([(1.0, 0, 0, 0.2)], world) == pytest.approx(0.3, abs=1e-6)
    # sphere centre inside the box -> negative
    assert min_sphere_clearance([(0.0, 0, 0, 0.2)], world) < 0.0


def test_person_only_world_filters():
    w = store_world_with_person()
    po = person_only_world(w)
    assert set(po['cuboid']) == {PERSON}
    assert person_only_world(store_world())['cuboid'] == {}


# --------------------------------------------------------------------------- rebasing
def test_shift_world_translates_all_cuboids():
    w = store_world()
    s = shift_world(w, (0.1, -0.2, 0.3))
    for name, c in s['cuboid'].items():
        o = w['cuboid'][name]['pose']
        assert c['pose'][0] == pytest.approx(o[0] + 0.1)
        assert c['pose'][1] == pytest.approx(o[1] - 0.2)
        assert c['pose'][2] == pytest.approx(o[2] + 0.3)
    # original unchanged (copy, not mutation)
    assert w['cuboid']['ground']['pose'][0] == 0.0


def test_nominal_reach_matches_design():
    a = StoreAisle()
    assert nominal_reach(a) == (a.target_shelf_x, 0.0, a.plate_z)


# --------------------------------------------------------------------------- windows
def test_mission_windows_keyed_by_mission_joints(gate):
    w = mission_windows([(-1.0, 1.0)] * 6, gate)
    assert set(w.windows) == set(ARM_JOINTS)
    assert w.within({j: 0.0 for j in ARM_JOINTS})
    assert not w.within({ARM_JOINTS[0]: 5.0})


def test_mission_windows_rejects_bad_input(gate):
    with pytest.raises(ValueError):
        mission_windows([(-1.0, 1.0)] * 5, gate)        # wrong count
    with pytest.raises(ValueError):
        mission_windows([(1.0, -1.0)] * 6, gate)         # hi <= lo


# --------------------------------------------------------------------------- gate behaviour
def test_gate_stop_holds_and_resumes(gate, windows):
    tl = [(gate.RUN, 1.0)] * 4 + [(gate.STOP, 0.0)] * 4 + [(gate.RUN, 1.0)] * 4
    cmds = [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]] * 12
    r = run_gate_trial(gate, windows, timeline=tl, commands=cmds, start_position=[0.0] * 6)
    assert r.travel_while_not_run == pytest.approx(0.0, abs=1e-12)  # HARD rule
    assert r.ok
    assert r.travel_while_run > 0.0
    assert r.dropped_during_stop == 4
    assert r.resumed_to_fresh


def test_gate_refuses_out_of_window_command(gate):
    w = mission_windows([(-1.0, 1.0)] * 6, gate)
    tl = [(gate.RUN, 1.0)] * 3
    cmds = [[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3        # shoulder_pan way outside [-1,1]
    r = run_gate_trial(gate, w, timeline=tl, commands=cmds, start_position=[0.0] * 6)
    assert r.window_refusals == 3
    assert r.travel_while_run == pytest.approx(0.0, abs=1e-12)  # refused -> no travel


def test_gate_starved_run_holds(gate, windows):
    tl = [(gate.RUN, 1.0)] * 3
    r = run_gate_trial(gate, windows, timeline=tl, commands=[None, None, None],
                       start_position=[0.0] * 6)
    assert r.held == 3 and r.forwarded == 0
    assert r.travel_while_run == pytest.approx(0.0, abs=1e-12)


def test_gate_trial_length_mismatch(gate, windows):
    with pytest.raises(ValueError):
        run_gate_trial(gate, windows, timeline=[(gate.RUN, 1.0)], commands=[],
                       start_position=[0.0] * 6)


# --------------------------------------------------------------------------- metrics
def test_metrics_as_dict_and_table():
    m = MpcScenarioMetrics(scenario='follow', steps=300, control_rate_hz=58.0,
                           solve_ms_mean=17.0, solve_ms_p95=18.0, solve_ms_max=21.0,
                           tracking_steady_mean_m=0.038, min_clearance_m=0.068,
                           est_process_mib=215.0)
    d = m.as_dict()
    assert d['control_rate_hz'] == 58.0
    assert d['solve_ms']['mean'] == 17.0
    assert d['min_clearance_m'] == 0.068
    table = metrics_table([m])
    assert 'follow' in table and 'rate_hz' in table


def test_load_gate_missing_dir_raises(monkeypatch):
    monkeypatch.setenv('MPC_CUROBO_DIR', '/nonexistent/path/xyz')
    monkeypatch.setattr(bench.os.path, 'isdir', lambda p: False)
    with pytest.raises(RuntimeError):
        bench.load_gate('/also/nonexistent')
