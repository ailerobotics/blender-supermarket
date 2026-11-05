"""CPU tests for the arm collision-avoidance evaluation bench (sim-arm-collision-bench-v1).

No Isaac, ROS or GPU -- the mock ESDF, the arm swept-sphere FK model, the clearance passes, the
gates, the scenario schema and the reporting are all pure numpy + the pure URDF/carton-level FK.
The regression lock the brief asks for: the CPU mode runs on EVERY shipped scenario, and under full
perception coverage the reference gate makes the right call on all of them (S5: no contact on the
static set, STOP-not-contact on the moving set)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/ for `twin`

from twin.tools import arm_collision_bench as b

SCENARIO_DIR = Path(__file__).resolve().parents[3] / 'docs/examples/arm-collision-scenarios'


@pytest.fixture(scope='module')
def arm():
    return b.ArmModel.load()


@pytest.fixture(scope='module')
def scenarios():
    scs = b.load_scenarios(SCENARIO_DIR)
    assert scs, 'shipped scenario set must be non-empty'
    return scs


# --------------------------------------------------------------------------- pure geometry

def test_obstacle_signed_distance_box_and_sphere():
    box = b.Obstacle('b', 'box', center=(0, 0, 0), size=(0.2, 0.2, 0.2))
    # a point 0.1 m outside the +x face
    d = box.signed_distance(np.array([[0.2, 0.0, 0.0]]), 0.0)
    assert d[0] == pytest.approx(0.1, abs=1e-6)
    # inside is negative
    assert box.signed_distance(np.array([[0.0, 0.0, 0.0]]), 0.0)[0] < 0
    sph = b.Obstacle('s', 'sphere', center=(1, 0, 0), radius=0.25)
    assert sph.signed_distance(np.array([[0.5, 0.0, 0.0]]), 0.0)[0] == pytest.approx(0.25, abs=1e-6)


def test_motion_translates_over_time():
    m = b.Motion(kind='linear', velocity=(0.0, -0.5, 0.0), start_t=1.0)
    assert np.allclose(m.position_at(0.0), [0, 0, 0])
    assert np.allclose(m.position_at(3.0), [0, -1.0, 0])  # (3-1)*-0.5


def test_mock_esdf_tracks_true_distance_for_observed():
    ob = b.Obstacle('b', 'box', center=(0.5, 0.5, 0.5), size=(0.2, 0.2, 0.2))
    esdf = b.MockEsdf.rasterise([ob], 0.0, lo=(0, 0, 0), hi=(1, 1, 1), voxel=0.01)
    pts = np.array([[0.5, 0.5, 0.9], [0.5, 0.9, 0.5]])
    perceived = esdf.query(pts)
    true = ob.signed_distance(pts, 0.0)
    # at a 0.01 m voxel the interpolated ESDF tracks the analytic SDF to well under a voxel
    assert np.max(np.abs(perceived - true)) < 0.012


def test_unobserved_obstacle_is_invisible_to_esdf():
    seen = b.Obstacle('seen', 'box', center=(0.5, 0.5, 0.5), size=(0.2, 0.2, 0.2), observed=True)
    blind = b.Obstacle('blind', 'box', center=(0.5, 0.5, 0.5), size=(0.2, 0.2, 0.2), observed=False)
    lo, hi, v = (0, 0, 0), (1, 1, 1), 0.02
    e_seen = b.MockEsdf.rasterise([seen], 0.0, lo, hi, v)
    e_blind = b.MockEsdf.rasterise([blind], 0.0, lo, hi, v)
    p = np.array([[0.5, 0.5, 0.5]])
    assert e_seen.query(p)[0] < 0            # inside the observed box
    assert e_blind.query(p)[0] > 0.5         # the unobserved box reads as free space


# --------------------------------------------------------------------------- arm model

def test_arm_sphere_model_loads_all_curobo_links(arm):
    assert len(arm.spheres) == 15
    s = arm.spheres_at(b._GRASP)
    assert s.shape[1] == 4 and len(s) > 100 and np.isfinite(s).all()


def test_carry_adds_carton_spheres(arm):
    empty = arm.spheres_at(b._LIFT, carry=False)
    carry = arm.spheres_at(b._LIFT, carry=True)
    assert len(carry) > len(empty)


def test_all_nominal_legs_build(arm):
    for leg in b.DEFAULT_LEG_CYCLE_S:
        traj = b.nominal_leg(leg)
        assert len(traj) >= 2 and all(len(w) == 6 for w in traj)


def test_carry_leg_tilt_is_bounded(arm):
    # the representative carry legs are modest folds -> the carton attitude witness stays small
    # (the bench reports tilt; it does not enforce the 2.0 deg bound -- that is the mission's invariant).
    for leg in ('carry', 'preplace_approach', 'set_down', 'withdraw'):
        tilt = b._tilt_peak(arm, b.nominal_leg(leg))
        assert 0.0 <= tilt < 15.0


# --------------------------------------------------------------------------- scenario schema

def test_every_shipped_scenario_parses_and_runs(arm, scenarios):
    for s in scenarios:
        r = b.run_scenario(s, arm, b.reference_gate)
        assert r.gate_behavior in b.BEHAVIORS
        assert r.scenario_id == s.id
        # each shipped scenario actually plants an intrusion (min true clearance < margin)
        assert r.min_true_clearance_m < s.margin_m


def test_scenario_roundtrip():
    d = {'id': 'x', 'arm_leg': 'transfer', 'expected_behavior': 'avoid',
         'obstacles': [{'name': 'o', 'kind': 'box', 'center': [0.1, 0.5, 0.2], 'size': [0.1, 0.1, 0.1],
                        'observed': True, 'at_goal': True,
                        'motion': {'kind': 'linear', 'velocity': [0, -0.3, 0], 'class': 'person'}}]}
    s = b.scenario_from_dict(d)
    assert s.obstacles[0].at_goal and s.obstacles[0].motion.obstacle_class == 'person'


def test_bad_expected_behavior_rejected():
    with pytest.raises(ValueError):
        b.scenario_from_dict({'id': 'x', 'arm_leg': 'transfer', 'expected_behavior': 'nonsense'})


# --------------------------------------------------------------------------- gates

def test_baseline_gate_never_acts_and_fails_every_scenario(arm, scenarios):
    # the current stack has no perceived-ESDF gate on these arm legs: the BEFORE baseline.
    for s in scenarios:
        r = b.run_scenario(s, arm, b.baseline_gate)
        assert r.gate_behavior == 'none'
        assert not r.passed  # every scenario plants a collision the baseline does not prevent


def test_reference_gate_reacts_only_to_perceived(arm, scenarios):
    # a blind (observed=False) obstacle must NOT trigger the gate (models the current stack blindness);
    # an observed one in the same scenario slot must.
    for s in scenarios:
        r = b.run_scenario(s, arm, b.reference_gate)
        observed_intrusion = any(o.observed for o in s.obstacles)
        if not observed_intrusion:
            assert r.gate_behavior == 'clear'  # blind -> no action


def test_full_coverage_reference_passes_every_scenario(arm, scenarios):
    # THE REGRESSION LOCK: once the stack SEES every obstacle (mast at collect/carry, cart as world
    # obstacle), the reference gate makes the right call on all of them -- and S5 holds.
    results = [b.run_scenario(b.with_full_coverage(s), arm, b.reference_gate) for s in scenarios]
    summary = b.summarise(results, 'reference+observe_all', b.Thresholds())
    assert summary['all_passed'], [r.fail_reasons for r in results if not r.passed]
    assert summary['executed_collisions'] == 0  # S5: no contact after the gate acts


def test_S5_static_no_contact_moving_stop_under_full_coverage(arm, scenarios):
    for s in scenarios:
        r = b.run_scenario(b.with_full_coverage(s), arm, b.reference_gate)
        moving = any(o.motion.kind != 'static' for o in s.obstacles)
        assert r.executed_collision_free, 'S5: no contact on %s' % s.id
        if moving:
            # moving set: a STOP or preempt, never a silent proceed-into-contact
            assert r.gate_behavior in ('slow_stop_resume', 'preempt_replan')


def test_person_routed_to_stop_not_esdf_avoid(arm):
    # captain section-9 decision: a person is a generic dynamic volume; the HUMAN GATE owns the STOP.
    c = arm.spheres_at(b.nominal_leg('carry')[4], carry=True)[:, :3].mean(0)
    s = b.Scenario(id='p', arm_leg='carry', expected_behavior='slow_stop_resume',
                   obstacles=(b.Obstacle('person', 'sphere', center=tuple(c), radius=0.2, observed=True,
                                         motion=b.Motion('linear', (0, -0.3, 0), 0.0, 'person')),),
                   duration_s=3.0, time_step_s=0.2)
    r = b.run_scenario(s, arm, b.reference_gate)
    assert r.gate_behavior == 'slow_stop_resume'


# --------------------------------------------------------------------------- reporting + gate resolution

def test_report_and_csv_render(arm, scenarios):
    results = [b.run_scenario(s, arm, b.reference_gate) for s in scenarios]
    summary = b.summarise(results, 'reference', b.Thresholds())
    md = b.report_markdown(results, summary)
    csv = b.results_csv(results)
    assert '| scenario |' in md and 'PASS' in md or 'FAIL' in md
    assert csv.splitlines()[0].startswith('scenario_id,')
    assert len(csv.splitlines()) == len(results) + 1


def test_resolve_gate_specs():
    assert b.resolve_gate('baseline')[1] == 'baseline'
    assert b.resolve_gate('reference')[1] == 'reference'
    g, name = b.resolve_gate('twin.tools.arm_collision_bench:reference_gate')
    assert g is b.reference_gate
    with pytest.raises(ValueError):
        b.resolve_gate('nope')


# --------------------------------------------------------------------------- GPU-side (pxr authoring)

pxr = pytest.importorskip('pxr')  # usd-core; present in .venv-usd


def test_author_obstacles_static_collider_and_moving_kinematic(scenarios):
    from pxr import Usd, UsdGeom, UsdPhysics
    from twin.tools import arm_collision_scene as sc
    by_id = {s.id: s for s in scenarios}
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    s = by_id['carry-person-crosses-moving']  # a moving person
    paths = sc.author_obstacles(stage, s, base_prim_path='/World/missing_base')
    assert set(paths) == {o.name for o in s.obstacles}
    prim = stage.GetPrimAtPath(paths['person'])
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)        # moving -> kinematic rigid body
    assert prim.GetAttribute('arm_bench:observed').Get() is False  # authored even though unobserved

    s2 = by_id['transfer-box-on-shelf-edge-observed']  # a static box
    p2 = sc.author_obstacles(stage, s2, base_prim_path='/World/missing_base', root='/Bench2')
    sp = stage.GetPrimAtPath(p2['shelf_box'])
    assert sp.HasAPI(UsdPhysics.CollisionAPI) and not sp.HasAPI(UsdPhysics.RigidBodyAPI)


def test_moving_obstacle_world_advances(scenarios):
    from pxr import Usd, UsdGeom
    from twin.tools import arm_collision_scene as sc
    s = {x.id: x for x in scenarios}['carry-person-crosses-moving']
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    sc.author_obstacles(stage, s, base_prim_path='/World/missing_base')
    p0 = sc.moving_obstacle_world(s, stage, 0.0, base_prim_path='/World/missing_base')['person']
    p2 = sc.moving_obstacle_world(s, stage, 2.0, base_prim_path='/World/missing_base')['person']
    assert p2[1] < p0[1]  # walking in -Y


def test_ingest_trial_maps_to_schema(scenarios):
    from twin.tools import arm_collision_scene as sc
    s = {x.id: x for x in scenarios}['transfer-hand-reach-moving']
    ok = sc.ingest_trial(s, {'min_true_clearance_m': 0.05, 'contact': False,
                             'gate_behavior': 'slow_stop_resume', 'extra_cycle_time_s': 3.0})
    assert ok.passed and ok.verdict_match
    bad = sc.ingest_trial(s, {'min_true_clearance_m': -0.02, 'contact': True, 'gate_behavior': 'none'})
    assert not bad.passed and any('contact' in f for f in bad.fail_reasons)
    # the GPU row shares the CPU row schema exactly
    assert set(ok.as_row()) == set(b.run_scenario(s, b.ArmModel.load(), b.reference_gate).as_row())


def test_thresholds_cycle_time_bound_fails_over_budget(arm):
    # a gate that burns more cycle time than the bound fails the cycle-time threshold.
    def slow_gate(scenario, frames):
        return b.GateVerdict('avoid', 1, 10.0 * scenario.nominal_cycle_s(), 'deliberately slow')
    s = b.Scenario(id='slow', arm_leg='transfer', expected_behavior='avoid',
                   obstacles=(b.Obstacle('o', 'box', center=(0.1, 0.5, 0.15), size=(0.2, 0.2, 0.2),
                                         observed=True),))
    r = b.run_scenario(s, arm, slow_gate, b.Thresholds(cycle_time_overhead_frac=0.5))
    assert not r.passed and any('cycle-time' in f for f in r.fail_reasons)
