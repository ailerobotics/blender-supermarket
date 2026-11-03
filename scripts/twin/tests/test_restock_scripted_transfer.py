"""RESTOCK_TRANSFER_MODE=scripted -- CPU proof of the scripted transfer fallback (no Isaac/ROS/GPU).

Geometry = the run-356 transfer park (dataset run-356/, read-only; same inputs as
test_restock_transfer_live_pose): arm base ur_base_link at (-2.079, 15.254, 0.50) yaw -0.5 deg, the warm-start
carton on the rollwagen UPPER shelf (cart body +0.18 m -> world (-3.169, 15.265, 1.0125), yaw -90 deg), the
ground-truth bay target (-2.0, 16.424, 0.73). The arm chain below is the run-356 mobile-import.urdf
ur_base_link -> gripper_pad_tcp chain verbatim (checked against the dataset file when it is present).
"""
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock import scripted_arm_client as client  # noqa: E402
from twin.restock import scripted_transfer as st  # noqa: E402
from twin.restock.inventory import Inventory, Job  # noqa: E402
from twin.restock.mission import ACTIONS, Mission  # noqa: E402
from twin.restock.runtime import Packet, RuntimeAdapter  # noqa: E402
from twin.restock.trajectory_host import Plan  # noqa: E402
from twin.restock_manipulation import transfer_mode as tm  # noqa: E402
from twin.restock_manipulation.ros_adapter import validate_trajectory  # noqa: E402
from twin.restock_manipulation.transfer_carry import (DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN,  # noqa: E402
                                                      in_clear_region)

RUN356_URDF = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/run-356/mobile-import.urdf')
DATASET = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1')
MOBILE_XRDF = DATASET / 'host-cad' / 'mobile.xrdf'
WARM5_HOSTLOG = DATASET / 'mission' / 'warm5' / 'host-stack.log'
STOW = np.array([0., -2.5, 1.7, -0.7, -1.5708, 0.])
BASE = st.pose_T((-2.079, 15.254, 0.50), st.yaw_quat(math.radians(-0.5)))
CARTON = st.pose_T((-3.169, 15.265, 1.0125), st.yaw_quat(-math.pi / 2))
TARGET = (-2.0, 16.424, 0.73)
# execretest (2026-10-22): the SAME bay park, but the live carton sits ~3.8 cm closer on the cart upper shelf
# (nav-trace carton = world (-3.1313, 15.2513, 1.0124), yaw -90.14 deg). That small reach change makes the
# scripted ArmIK unwrap the pick's ur_wrist_1 onto a far +2*pi branch (raw ~+4.1) that in_clear_region's wrap
# reads as "clear" -- the executor-pick jam this module's clear-branch fix addresses. Same URDF as run-356.
EXECRETEST_CARTON = st.pose_T((-3.1313, 15.2513, 1.0124), st.yaw_quat(math.radians(-90.14)))
DIMS = (0.6, 0.4, 0.16)
SCRIPTED_ENV = {'RESTOCK_TRANSFER_MODE': 'scripted'}

# run-356 mobile-import.urdf, ur_base_link -> gripper_pad_tcp (name, type, parent, child, xyz, rpy, lower, upper)
_CHAIN = [
    ('ur_base_link-base_link_inertia', 'fixed', 'ur_base_link', 'ur_base_link_inertia', (0, 0, 0), (0, 0, math.pi), None, None),
    ('ur_shoulder_pan_joint', 'revolute', 'ur_base_link_inertia', 'ur_shoulder_link', (0, 0, 0.1273), (0, 0, 0), -2 * math.pi, 2 * math.pi),
    ('ur_shoulder_lift_joint', 'revolute', 'ur_shoulder_link', 'ur_upper_arm_link', (0, 0, 0), (1.570796327, 0, 0), -2 * math.pi, 2 * math.pi),
    ('ur_elbow_joint', 'revolute', 'ur_upper_arm_link', 'ur_forearm_link', (-0.612, 0, 0), (0, 0, 0), 0.1, math.pi),
    ('ur_wrist_1_joint', 'revolute', 'ur_forearm_link', 'ur_wrist_1_link', (-0.5723, 0, 0.163941), (0, 0, 0), -2 * math.pi, 2 * math.pi),
    ('ur_wrist_2_joint', 'revolute', 'ur_wrist_1_link', 'ur_wrist_2_link', (0, -0.1157, 0), (1.570796327, 0, 0), -2 * math.pi, 2 * math.pi),
    ('ur_wrist_3_joint', 'revolute', 'ur_wrist_2_link', 'ur_wrist_3_link', (0, 0.0922, 0), (1.570796326589793, math.pi, math.pi), -2 * math.pi, 2 * math.pi),
    ('ur_wrist_3-flange', 'fixed', 'ur_wrist_3_link', 'ur_flange', (0, 0, 0), (0, -math.pi / 2, -math.pi / 2), None, None),
    ('ur_flange-tool0', 'fixed', 'ur_flange', 'ur_tool0', (0, 0, 0), (math.pi / 2, 0, math.pi / 2), None, None),
    ('gripper_mount_joint', 'fixed', 'ur_tool0', 'big_cube', (-0.000368, -0.000549, 0.130241), (1.5707963, 0, 3.1415927), None, None),
    ('suction_cups_joint', 'fixed', 'big_cube', 'suction_cups', (0.120804, 0.117792, 0.04302), (-1.535325, 0, 0), None, None),
    ('gripper_pad_tcp_joint', 'fixed', 'suction_cups', 'gripper_pad_tcp', (-0.120313, 0.044122, 0.02245), (0.035471327, 0, 3.141592654), None, None),
]


def _urdf_text():
    links = sorted({c[2] for c in _CHAIN} | {c[3] for c in _CHAIN})
    out = ['<robot name="run356_arm_chain">'] + [f'  <link name="{name}"/>' for name in links]
    for name, kind, parent, child, xyz, rpy, lo, hi in _CHAIN:
        out.append(f'  <joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
                   f'<origin xyz="{" ".join(map(repr, map(float, xyz)))}" rpy="{" ".join(map(repr, map(float, rpy)))}"/>'
                   + ('<axis xyz="0 0 1"/>' if kind == 'revolute' else '')
                   + (f'<limit lower="{lo!r}" upper="{hi!r}" effort="330" velocity="2.09"/>' if kind == 'revolute' else '')
                   + '</joint>')
    return '\n'.join(out + ['</robot>'])


@pytest.fixture(scope='module')
def urdf(tmp_path_factory):
    path = tmp_path_factory.mktemp('urdf') / 'arm.urdf'
    path.write_text(_urdf_text())
    return path


@pytest.fixture(scope='module')
def ik(urdf):
    return st.ArmIK(urdf)


class FakeDriver:
    """Measured state + carton actuation for the CPU runs. Never exposes an arm command."""

    def __init__(self, ik, q=STOW):
        self.ik, self.q, self.c = ik, np.array(q, float), CARTON.copy()
        self.calls, self.carton_trace = [], []

    def base_world_T(self):
        return BASE

    def carton_world_T(self):
        return self.c

    def tool_world_T(self):
        return BASE @ self.ik.fk(self.q)

    def arm_joint_positions(self):
        return self.q.copy()

    def attach_carton(self):
        self.calls.append('attach')

    def set_carton_world_T(self, T):
        self.c = np.asarray(T, float)
        self.carton_trace.append(self.c.copy())

    def place_carton(self, T):
        self.calls.append('place')
        self.c = np.asarray(T, float)


class FakeExecutor:
    """Stands in for the host trajectory executor: checks each goal exactly like trajectory_host (start vs
    measured, Plan acceptance), then plays it on the fake driver's joints on the mission clock."""

    def __init__(self, driver, ik, outcomes=None, dt=1 / 30):
        self.driver, self.ik, self.dt = driver, ik, dt
        self.outcomes = dict(outcomes or {})
        self.sent, self.running, self.holds = [], None, []

    def hold(self, measured):
        self.holds.append(np.asarray(measured, float))      # keep-alive: current joints held, not silence

    def send(self, name, points):
        limits = list(zip(self.ik.lower, self.ik.upper))
        traj = NS(joint_names=list(st.ARM_JOINTS),
                  points=[NS(positions=list(q), velocities=list(v), effort=[],
                             time_from_start=NS(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))))
                          for t, q, v in points])
        validate_trajectory(traj, self.driver.q, limits)
        plan = Plan(traj, self.driver.q, limits, max_speed=0.3, max_acceleration=0.3 * (.5 / .15))
        self.sent.append(name)
        self.running = dict(name=name, plan=plan, t=0.0)

    def poll(self):
        run = self.running
        outcome = self.outcomes.get(run['name'], 'succeeded')
        if outcome == 'rejected':
            return dict(status='rejected', detail='goal not accepted by the executor')
        run['t'] += self.dt
        self.driver.q = run['plan'].sample(run['t'])[0]
        if outcome == 'aborted' and run['t'] > 1.0:
            return dict(status='aborted', detail='trajectory tracking error')
        if run['t'] >= run['plan'].times[-1]:
            return dict(status='succeeded', detail='')
        return None


def _packet(i, source='sensor'):
    return Packet(f'physics:{i}', source, i / 30, None, None, None, None, True, True, 0., True)


def _run(runner, ticks=20000):
    runner.begin()
    r = None
    for i in range(ticks):
        r = runner.step(1 / 30, _packet(i + 1))
        if r.status != 'running':
            return r
    raise AssertionError(f'scripted transfer did not finish: {r}')


def _runner(ik, driver, arm=None, **kw):
    return st.ScriptedTransfer(driver, ik=ik, target_xyz_world=TARGET, dims=DIMS, location='G3_back_s2',
                               sku='milk', quantity=1, arm=arm, **kw)


# ---------------------------------------------------------------- switch + labels

def test_scripted_mode_is_one_env_switch_with_its_own_labels():
    assert tm.transfer_mode(SCRIPTED_ENV) == tm.SCRIPTED
    assert tm.transfer_mode({}) is None
    caption = tm.transfer_caption('BENCH RUN start_at=transfer', SCRIPTED_ENV)
    assert caption.startswith('BENCH RUN start_at=transfer | SCRIPTED TRANSFER (workaround)')
    assert tm.transfer_caption(None, SCRIPTED_ENV) == tm.CAPTIONS[tm.SCRIPTED]
    assert tm.transfer_caption('BENCH', {}) == 'BENCH'                      # off => unchanged
    summary = tm.stamp_summary({'note': 'run'}, SCRIPTED_ENV)
    assert summary['transfer_mode'] == 'scripted'
    assert summary['workarounds'] == ['transfer_mode=scripted']
    assert 'TRANSFER_MODE=scripted' in summary['note']
    assert summary['walkthrough_caption'] == tm.CAPTIONS[tm.SCRIPTED]
    assert tm.stamp_summary({'note': 'run'}, {}) == {'note': 'run'}         # off => byte-identical


def test_arm_mode_defaults_to_still_and_fails_loud_on_typos():
    assert st.scripted_arm_mode({}, {}) == st.ARM_STILL
    assert st.scripted_arm_mode({'scripted_transfer': {'arm': 'executor'}}, {}) == st.ARM_EXECUTOR
    assert st.scripted_arm_mode({'scripted_transfer': {'arm': 'executor'}},
                                {st.ARM_ENV: 'still'}) == st.ARM_STILL        # env wins
    assert st.scripted_arm_mode({'scripted_transfer': None}, {}) == st.ARM_STILL
    with pytest.raises(ValueError, match='unknown scripted-transfer arm mode'):
        st.scripted_arm_mode({}, {st.ARM_ENV: 'teleport'})


def test_build_is_none_when_off_and_refuses_a_missing_target(urdf):
    assert st.build_scripted_transfer(assembly=None, config={}, scenario={}, environ={}) is None
    with pytest.raises(ValueError, match='transfer_ground_truth'):
        st.build_scripted_transfer(assembly=None, config={}, scenario={}, environ=SCRIPTED_ENV)
    with pytest.raises(ValueError, match="'enabled' is what"):           # target without the GT opt-in
        st.build_scripted_transfer(assembly=None, scenario={}, environ=SCRIPTED_ENV,
                                   config={'transfer_ground_truth': {'target': list(TARGET)}})
    assembly = NS(stage=None, command_gate=None, evidence={'composition': {'stock': '/World/Carton'}},
                  paths={'ur_base_link': '/R/ur_base_link', 'gripper_pad_tcp': '/R/gripper_pad_tcp'})
    config = {'transfer_ground_truth': {'enabled': True, 'location': 'G3_back_s2', 'target': list(TARGET)},
              'scripted_transfer': {'urdf': str(urdf)}, 'masked_depth': {'runtime_env': '/env.sh', 'domain': None}}
    scenario = {'handling_unit': {'dimensions_m': list(DIMS)}, 'job': {'sku': 'milk', 'shelf_id': 'G3_back_s2'}}
    still = st.build_scripted_transfer(assembly=assembly, config=config, scenario=scenario, environ=SCRIPTED_ENV)
    assert still.arm is None and still.arm_mode == st.ARM_STILL
    assert still.driver.carton_path == '/World/Carton'                   # the composed stock prim
    with tempfile.TemporaryDirectory() as out:
        executor = st.build_scripted_transfer(assembly=assembly, config=config, scenario=scenario, output=out,
                                              environ=dict(SCRIPTED_ENV, **{st.ARM_ENV: 'executor'}))
        assert isinstance(executor.arm, st.ExecutorArm) and executor.arm.domain == 88
    with pytest.raises(ValueError, match='host overlay env'):
        st.build_scripted_transfer(assembly=assembly, config=dict(config, masked_depth={}), scenario=scenario,
                                   environ=dict(SCRIPTED_ENV, **{st.ARM_ENV: 'executor'}))


# ---------------------------------------------------------------- geometry + timing

def test_fixture_chain_matches_the_run356_import_when_available(ik):
    if not RUN356_URDF.is_file():
        pytest.skip('run-356 dataset not present')
    real = st.ArmIK(RUN356_URDF)
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.uniform(-2.5, 2.5, 6)
        q[2] = rng.uniform(0.2, 3.0)
        assert np.allclose(real.fk(q), ik.fk(q), atol=1e-9)


def test_side_grip_puts_the_pad_on_the_near_face():
    contact, approach, out = st.side_grip((0., 0., 0.5), 0.0, dims=DIMS, wall=3, contact_margin=0.00275,
                                          standoff=0.15)
    # wall 3 => grip angle 3pi/2: the pad sits on the -y... side at half the 0.4 m depth + margin
    assert np.allclose(contact[:3, 3], (0., 0.20275, 0.5), atol=1e-9)
    assert np.allclose(approach[:3, 3], (0., 0.35275, 0.5), atol=1e-9)
    assert np.allclose(out, (0., 1.), atol=1e-9)


def test_timing_obeys_the_executors_own_plan_checks():
    rng = np.random.default_rng(7)
    q0 = np.array([0.3, -2.0, 1.5, -0.5, -1.2, 0.1])
    path = [q0 + np.cumsum(rng.uniform(-0.04, 0.04, (25, 6)), axis=0)[i] for i in range(25)] + [q0 + 1.2]
    samples = st.time_keyframe(q0, path, minimum=1.0)
    assert st.hermite_violation(samples) is None
    limits = [(-2 * math.pi, 2 * math.pi)] * 6
    traj = NS(joint_names=list(st.ARM_JOINTS),
              points=[NS(positions=list(q), velocities=list(v), effort=[],
                         time_from_start=NS(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))))
                      for t, q, v in samples])
    Plan(traj, q0, limits, max_speed=0.3, max_acceleration=1.0)          # the executor accepts it
    fast = [(t * 0.3, q, v / 0.3) for t, q, v in samples]                # 3.3x faster: both must refuse
    assert st.hermite_violation(fast) is not None
    traj.points = [NS(positions=list(q), velocities=list(v), effort=[],
                      time_from_start=NS(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))) for t, q, v in fast]
    with pytest.raises(ValueError):
        Plan(traj, q0, limits, max_speed=0.3, max_acceleration=1.0)
    assert st.time_keyframe(q0, [q0.copy()]) == []                       # no motion => no samples


def test_run356_plan_solves_every_keyframe_inside_reach(ik):
    base_inv = np.linalg.inv(BASE)
    goal = (base_inv @ np.array([TARGET[0], TARGET[1], TARGET[2] + DIMS[2] / 2 + 0.002, 1.0]))[:3]
    plan = st.plan_transfer(ik, STOW, base_inv @ CARTON, goal, math.radians(0.5), dims=DIMS)
    assert plan.ik_complete
    assert [k.name for k in plan.keyframes] == ['reach_high', 'approach', 'contact', 'lift', 'extract',
                                                'carry_high', 'above_place', 'place', 'retreat',
                                                'retreat_high', 'stow']
    assert plan.choice['target_flipped'] and not plan.choice['source_flipped']   # near face, like live_pose
    assert plan.choice['worst_station_reach'] < 1.05
    # the cage-mouth finding: the arm runs out of reach ~0.34 m into the 0.46 m extraction
    assert 0.30 < plan.choice['extract_m'] < plan.choice['extract_wanted_m'] == pytest.approx(0.46)
    for k in plan.keyframes:
        assert max(k.ik_error) < 0.005 and k.samples
    for name, names in st.SEGMENTS:
        assert st.hermite_violation(st.segment_points(plan.keyframes, names), (ik.lower, ik.upper)) is None


def _plan(ik, wrist_guard):
    base_inv = np.linalg.inv(BASE)
    goal = (base_inv @ np.array([TARGET[0], TARGET[1], TARGET[2] + DIMS[2] / 2 + 0.002, 1.0]))[:3]
    return st.plan_transfer(ik, STOW, base_inv @ CARTON, goal, math.radians(0.5), dims=DIMS,
                            wrist_guard=wrist_guard)


def test_wrist_guard_puts_the_pick_on_the_clear_branch_without_moving_the_tcp(ik):
    # warm5 (2026-10-22): the scripted executor pick followed the ur_gripper-vs-ur_forearm self-colliding
    # wrist branch (wrist_1 ~1.37, wrist_2 ~-1.27), the host executor faulted on the >1.3 rad tracking error,
    # the e-stop then rejected the retry and the arm-command heartbeat starved. The default-on wrist guard
    # keeps every MOVING keyframe in the clear carry wrist branch (wrist_1 <= cap, wrist_2 >= min) -- while
    # the unguarded plan does not -- and the TCP poses are identical, so the still-mode carton path is
    # byte-for-byte unchanged (only the arm's joint branch moves).
    guarded, unguarded = _plan(ik, True), _plan(ik, False)
    assert guarded.ik_complete and unguarded.ik_complete
    moving = [k for k in guarded.keyframes if k.name != 'stow']       # 'stow' == the parked start/return pose
    assert all(in_clear_region(k.q, DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN) for k in moving)
    pick_half = {'reach_high', 'approach', 'contact', 'lift'}
    unguarded_pick = [k for k in unguarded.keyframes if k.name in pick_half]
    assert not all(in_clear_region(k.q, DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN) for k in unguarded_pick)
    # identical tool path => identical carton path in still mode
    by_name = {k.name: k for k in unguarded.keyframes}
    for k in guarded.keyframes:
        assert np.allclose(k.tcp, by_name[k.name].tcp, atol=1e-6)
    assert guarded.choice == unguarded.choice                        # same near-face / reach / extract


def _execretest_plan(ik, wrist_guard=True):
    base_inv = np.linalg.inv(BASE)
    goal = (base_inv @ np.array([TARGET[0], TARGET[1], TARGET[2] + DIMS[2] / 2 + 0.002, 1.0]))[:3]
    return st.plan_transfer(ik, STOW, base_inv @ EXECRETEST_CARTON, goal, math.radians(0.5), dims=DIMS,
                            wrist_guard=wrist_guard)


def test_clear_branch_reexpresses_the_far_wrist1_branch_that_wrap_reads_as_clear():
    # The wrap blind spot, in isolation: a config whose RAW ur_wrist_1 is +4.12 (a far +2*pi branch) wraps to
    # -2.16, so in_clear_region reads it as clear even though a joint below the ~0.69 stop must sweep UP
    # through the stop to reach +4.12. clear_branch re-expresses it to the equivalent in-limit branch at or
    # below the cap (identical orientation, no sweep through the stop), and returns None when the cap is
    # unreachable within the joint limits.
    from twin.restock_manipulation.transfer_carry import clear_branch
    cap, lower, upper = DEFAULT_WRIST1_CAP, [-2 * math.pi] * 6, [2 * math.pi] * 6
    far = [0.29, -2.91, 2.01, 4.122, 1.05, 1.76]
    assert in_clear_region(far, cap, DEFAULT_WRIST2_MIN)             # wrap says "clear" -- the blind spot
    reexpressed = clear_branch(far, cap, lower, upper)
    assert reexpressed is not None and reexpressed[3] <= cap + 1e-9   # commanded raw wrist_1 now below the cap
    assert math.isclose(reexpressed[3], 4.122 - 2 * math.pi, abs_tol=1e-9)
    # an already-clear raw angle is returned unchanged; wrist_2/3 are never shifted
    assert clear_branch([0, -2.5, 1.7, -0.70, -1.5708, 0.0], cap, lower, upper)[3] == -0.70
    # no in-limit turn can satisfy an impossibly low cap => None (let the host report a named failure)
    assert clear_branch(far, -6.0, [-2 * math.pi] * 6, [2 * math.pi] * 6) is None


def test_execretest_pick_commands_wrist1_below_the_self_collision_stop(ik):
    # The brief's repro: plan the execretest pick from the transport (STOW) pose and assert every COMMANDED
    # waypoint keeps ur_wrist_1 at or below the clear cap. Before the clear-branch fix the guarded pick
    # endpoints passed in_clear_region (wrap) yet the host streamed raw wrist_1 ramping to ~+4.57, jamming the
    # arm on the ur_wrist_1-vs-ur_forearm stop at ~0.69 (TRAJ_HOST_FAULT 1.31 rad). The raw-angle check below
    # is what the wrapped predicate misses.
    plan = _execretest_plan(ik, wrist_guard=True)
    assert plan.ik_complete and plan.wrist_guard == (DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN)
    assert plan.wrist1_within_cap()
    for _name, names in st.SEGMENTS:
        pts = st.segment_points(plan.keyframes, names)
        for _t, q, _v in pts:
            assert float(q[3]) <= DEFAULT_WRIST1_CAP + 1e-6, f'commanded wrist_1 {q[3]:.3f} sweeps the stop'
    # the fix only re-expresses the arm's wrist branch: the carried tool path (and thus the still-mode carton
    # path) is identical to the unguarded plan, so no still/scripted run regresses.
    unguarded = {k.name: k for k in _execretest_plan(ik, wrist_guard=False).keyframes}
    for k in plan.keyframes:
        assert np.allclose(k.tcp, unguarded[k.name].tcp, atol=1e-3)
    assert plan.choice == _execretest_plan(ik, wrist_guard=False).choice


def test_executor_fails_fast_with_a_named_reason_when_no_clear_branch_exists(ik, monkeypatch):
    # When no clear-branch solution exists for a moving keyframe (here forced by stubbing clear_branch so the
    # pick keeps its far colliding branch), the executor must NOT stream the self-colliding trajectory: it
    # fails fast with a named reason and holds the arm (keep-alive), never the silent still degrade that
    # starves the arm-command heartbeat.
    monkeypatch.setattr(st, 'clear_branch', lambda *a, **k: None)
    driver = FakeDriver(ik)
    driver.c = EXECRETEST_CARTON.copy()                       # the geometry that unwraps wrist_1 onto the far branch
    arm = FakeExecutor(driver, ik)
    runner = _runner(ik, driver, arm=arm)
    runner.begin()
    assert not runner.plan.wrist1_within_cap()                # the forced plan is self-colliding
    r = runner.step(1 / 30, _packet(1))
    assert r.status == 'failed'
    assert 'clear-branch' in r.reason and 'ur_wrist_1' in r.reason
    assert runner.arm_mode == st.ARM_EXECUTOR and arm.holds   # arm held, not degraded to still
    assert not arm.sent                                       # never streamed the self-colliding goal


def _selfcollision_min_gap_fn():
    """min signed self-collision gap (m) over the arm's own non-ignored link-sphere pairs at a config, from
    the asset URDF FK + XRDF spheres (dataset-gated; the integration lane's validated sphere geometry)."""
    import yaml
    from twin.restock_host.world_obstacles import arm_spheres_world
    xrdf = yaml.safe_load(MOBILE_XRDF.read_text())
    ignore = set()
    for a, bs in (xrdf.get('self_collision', {}).get('ignore', {}) or {}).items():
        for b in bs:
            ignore.add(frozenset((a, b)))
    arm_order = list(st.ARM_JOINTS)

    def gap(q):
        centres, radii, owners = arm_spheres_world(str(RUN356_URDF), str(MOBILE_XRDF),
                                                   dict(zip(arm_order, [float(x) for x in q])))
        owners = np.asarray(owners)
        links = sorted(set(owners.tolist()))
        worst = math.inf
        for i in range(len(links)):
            for j in range(i + 1, len(links)):
                a, b = links[i], links[j]
                if frozenset((a, b)) in ignore:
                    continue
                ca, ra = centres[owners == a], radii[owners == a]
                cb, rb = centres[owners == b], radii[owners == b]
                d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
                worst = min(worst, float((d - ra[:, None] - rb[None, :]).min()))
        return worst
    return gap


@pytest.mark.skipif(not (RUN356_URDF.is_file() and MOBILE_XRDF.is_file()),
                    reason='needs the dataset URDF + XRDF sphere model')
def test_guard_is_self_collision_clear_where_the_unguarded_pick_penetrates():
    # Real geometry (asset URDF FK + XRDF collision spheres), not just the clear-region predicate: the
    # unguarded scripted pick interpenetrates the arm (gripper vs forearm), the guarded one stays clear.
    gap = _selfcollision_min_gap_fn()
    ik = st.ArmIK(RUN356_URDF)
    unguarded = min(gap(k.q) for k in _plan(ik, False).keyframes if k.q is not None)
    guarded = min(gap(k.q) for k in _plan(ik, True).keyframes if k.q is not None)
    assert unguarded < 0.0, f'expected the unguarded pick to self-collide, min gap {unguarded:.4f}'
    assert guarded > 0.0, f'guarded plan must stay self-collision-clear, min gap {guarded:.4f}'


@pytest.mark.skipif(not (RUN356_URDF.is_file() and MOBILE_XRDF.is_file() and WARM5_HOSTLOG.is_file()),
                    reason='needs warm5 host-stack.log + the dataset URDF/XRDF')
def test_warm5_recorded_pick_was_host_accepted_and_physically_self_collided():
    # Replay warm5's recorded executor pick through the HOST validation path: the host ACCEPTED it (the
    # TRAJ_HOST_PLAN line is logged only on GoalResponse.ACCEPT) and it passes validate_trajectory, so the
    # warm5 failure was a physical tracking fault, not a validation rejection -- yet the sphere model shows
    # it self-collides, exactly the branch the guard replaces.
    import ast
    import re
    lines = WARM5_HOSTLOG.read_text().splitlines()
    plan_line = next(l for l in lines if 'TRAJ_HOST_PLAN' in l and 'points=' in l)   # logged only on ACCEPT
    names = ast.literal_eval(re.search(r'names=(\[.*?\])', plan_line).group(1))
    pts = ast.literal_eval(re.search(r'points=(\[.*\])\s*$', plan_line).group(1))
    order = [names.index(j) for j in st.ARM_JOINTS]
    samples = [(float(t), np.array([float(q[k]) for k in order])) for t, q in pts]
    start = samples[0][1]
    limits = list(zip(st.ArmIK(RUN356_URDF).lower, st.ArmIK(RUN356_URDF).upper))
    traj = NS(joint_names=list(st.ARM_JOINTS),
              points=[NS(positions=list(q), velocities=([0.] * 6 if i == len(samples) - 1 else []),
                         effort=[], time_from_start=NS(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))))
                      for i, (t, q) in enumerate(samples)])
    validate_trajectory(traj, start, limits)                         # host accepts it -> no exception
    gap = _selfcollision_min_gap_fn()
    assert min(gap(q) for _t, q in samples) < 0.0                    # yet it physically self-collides


# ---------------------------------------------------------------- stage runs

def test_still_mode_lands_the_carton_on_the_ground_truth_target_without_arm_commands(ik):
    driver = FakeDriver(ik)
    r = _run(_runner(ik, driver))
    assert r.status == 'succeeded'
    ev = r.evidence
    assert ev['released'] and ev['transfer_mode'] == 'scripted' and ev['verify_mode'] == 'ground_truth'
    assert ev['arm_mode'] == 'still' and ev['arm_follow'] is False and ev['arm_fallback_reason'] is None
    # checked against the sim ground truth => labelled like vr1's ground-truth transfer, never 'sensor'
    assert ev['source'] == 'ground_truth' and ev['ground_truth'] is True
    assert ev['observation_id'].startswith('ground_truth:transfer:physics:')
    assert ev['xy_offset'] < 1e-9 and ev['bottom_gap'] == pytest.approx(0.002)
    assert np.allclose(driver.c[:3, 3], (TARGET[0], TARGET[1], TARGET[2] + 0.082))
    assert driver.calls == ['attach', 'place']
    assert np.allclose(driver.q, STOW)                                    # the arm never moved
    # the carried carton clears the bay top until the final place descent, and stays level
    trace = driver.carton_trace
    assert len(trace) > 300
    assert min(T[2, 3] - DIMS[2] / 2 for T in trace) > TARGET[2]
    assert all(abs(T[2, 2]) > 0.999 for T in trace)


def test_executor_mode_moves_the_arm_through_the_executor_and_the_carton_rides_the_tool(ik):
    driver = FakeDriver(ik)
    arm = FakeExecutor(driver, ik)
    runner = _runner(ik, driver, arm=arm)
    r = _run(runner)
    assert r.status == 'succeeded', r.reason
    assert arm.sent == ['pick', 'carry', 'retreat']                      # three goals, each Plan-accepted
    ev = r.evidence
    assert ev['arm_mode'] == 'executor' and ev['arm_follow'] is True
    assert ev['xy_offset'] < 1e-9 and ev['bottom_gap'] == pytest.approx(0.002)
    assert np.max(np.abs(driver.q - STOW)) < 0.011                       # back at the start pose


def test_ik_plan_latency_report_and_budget_parse():
    # The sender reports the begin()->send IK-plan latency and flags it OVER the budget (loud, not fatal:
    # the executor re-anchors the trajectory origin to the first feedback stamp, so a slow IK no longer
    # races the execution clock). Budget <=0 disables the verdict but still reports the number.
    over, line = st.ik_latency_report(95.1, 120.0)
    assert not over and 'ik_plan_latency=95.1s' in line and 'within budget' in line
    over, line = st.ik_latency_report(130.0, 120.0)
    assert over and 'OVER budget 120s' in line
    over, line = st.ik_latency_report(9999.0, 0.0)
    assert not over and 'budget disabled' in line and 'ik_plan_latency=9999.0s' in line
    # env parse: default, explicit, disable, and a loud failure on garbage
    assert st.ik_plan_budget_seconds({}) == st.IK_PLAN_BUDGET_DEFAULT_S
    assert st.ik_plan_budget_seconds({'RESTOCK_SCRIPTED_TRANSFER_IK_BUDGET_S': '45'}) == 45.0
    assert st.ik_plan_budget_seconds({'RESTOCK_SCRIPTED_TRANSFER_IK_BUDGET_S': '0'}) == 0.0
    with pytest.raises(ValueError, match='not a number'):
        st.ik_plan_budget_seconds({'RESTOCK_SCRIPTED_TRANSFER_IK_BUDGET_S': 'soon'})


def test_begin_records_the_ik_plan_latency(ik):
    # begin() measures and stores the IK-plan latency, and it rides the transfer evidence so a creeping
    # latency is visible in mission.json without re-parsing host-stack.log.
    driver = FakeDriver(ik)
    runner = _runner(ik, driver, arm=FakeExecutor(driver, ik))
    r = _run(runner)
    assert r.status == 'succeeded', r.reason
    assert runner.ik_plan_latency_s is not None and runner.ik_plan_latency_s >= 0.0
    assert r.evidence['ik_plan_latency_s'] == round(runner.ik_plan_latency_s, 3)


def test_executor_carton_is_rigid_on_the_measured_tool(ik):
    driver = FakeDriver(ik)
    arm = FakeExecutor(driver, ik)
    runner = _runner(ik, driver, arm=arm)
    runner.begin()
    rels = []
    for i in range(20000):
        r = runner.step(1 / 30, _packet(i + 1))
        if runner.carried and arm.running and arm.running['name'] == 'carry':
            rels.append(np.linalg.inv(driver.tool_world_T()) @ driver.c)
        if r.status != 'running':
            break
    assert r.status == 'succeeded'
    assert len(rels) > 100
    assert max(np.max(np.abs(R - rels[0])) for R in rels) < 0.02          # one-tick lag at most


def test_a_rejected_goal_fails_fast_and_keeps_the_arm_held(ik):
    # The warm5 trap: a host rejection used to degrade silently to still mode, which -- once the host
    # executor has e-stopped -- sends no arm command and the Isaac arm-command heartbeat starves into the
    # misleading 'arm command heartbeat lost' crash. In executor mode a rejection now fails FAST with a named
    # reason and holds the current joints (keep-alive), never the silent still fallback.
    driver = FakeDriver(ik)
    arm = FakeExecutor(driver, ik, outcomes={'pick': 'rejected'})
    r = _run(_runner(ik, driver, arm=arm))
    assert r.status == 'failed'
    assert 'rejected the pick trajectory' in r.reason and 'no silent still fallback' in r.reason
    assert arm.holds                                            # keep-alive hold issued (heartbeat alive)
    assert driver.calls == []                                   # nothing grasped/placed: no carton teleport


def test_an_aborted_goal_fails_the_stage_honestly_and_holds(ik):
    driver = FakeDriver(ik)
    arm = FakeExecutor(driver, ik, outcomes={'carry': 'aborted'})
    r = _run(_runner(ik, driver, arm=arm))
    assert r.status == 'failed' and 'aborted the carry trajectory' in r.reason
    assert arm.holds                                            # keep-alive on abort too


def test_an_infeasible_segment_fails_fast_not_silent_still(ik):
    # A trajectory the host would refuse (pre-send Hermite check) must also fail fast with a named reason,
    # never the silent still fallback.
    driver = FakeDriver(ik)
    arm = FakeExecutor(driver, ik)
    runner = _runner(ik, driver, arm=arm)
    runner.begin()
    # corrupt the first pick keyframe's samples so the re-seeded Hermite check refuses the segment: start at
    # the measured pose (STOW, so the start-match gate passes) then a 5 rad jump in 10 ms (accel violation)
    bad = runner.plan.keyframes[0]
    bad.samples = [(0.0, STOW.copy(), np.zeros(6)), (0.01, STOW + 5.0, np.zeros(6))]
    r = None
    for i in range(20000):
        r = runner.step(1 / 30, _packet(i + 1))
        if r.status != 'running':
            break
    assert r.status == 'failed' and 'infeasible' in r.reason and 'no silent still fallback' in r.reason
    assert arm.holds and arm.sent == []                         # never sent the bad goal; held instead


# ---------------------------------------------------------------- mission integration

class _Port:
    def __init__(self):
        self.i = 0

    def advance(self, dt):
        self.i += 1
        return _packet(self.i)

    def controls(self, controls):
        pass

    def stop(self):
        pass


def _adapter(scripted=None, manip_starts=None):
    starts = [] if manip_starts is None else manip_starts
    nav = NS(start=lambda *a, **k: None, step=lambda dt, p: NS(status='running', reason=''), cancel=lambda: None)
    mobile = NS(park=lambda: None, hold=lambda: None)
    manip = NS(start=lambda a, j: starts.append(a), cart_location='rollwagen', stop=lambda: None,
               warm=lambda dt, p: False, step=lambda dt, p: NS(status='running', reason='', controls=()))
    poses = {k: [0, 0, 0] for k in ('home', 'pickup', 'cart_dock', 'shelf_approach')}
    adapter = RuntimeAdapter(nav, mobile, manip, _Port(), poses, fixture_mode=False)
    adapter.scripted_transfer = scripted
    return adapter


def _mission(ik, tmp_path, starts, **kw):
    inv = Inventory(str(tmp_path / 'stock.sqlite'))
    inv.seed('rollwagen', 'milk', 1)
    job = Job('j1', 'milk', 1, 'rollwagen', 'G3_back_s2')
    adapter = _adapter(_runner(ik, FakeDriver(ik)), starts)
    mission = Mission(inv, job, adapter, timeout_s=300., stages=ACTIONS[:5], start_at='transfer', **kw)
    mission.start()
    for _ in range(20000):
        if mission.terminal:
            break
        mission.tick(1 / 30)
    return mission, inv


def test_mission_transfer_stage_completes_on_the_scripted_runner(ik, tmp_path):
    starts = []
    # transfer_ground_truth.enabled => run_mobile_restock passes mission_ground_truth_sources = ('ground_truth',)
    mission, inv = _mission(ik, tmp_path, starts, ground_truth_sources=('ground_truth',))
    assert mission.state == 'complete', mission.reason
    assert starts == []                                                 # no host manipulation job
    events = [json.loads(row['evidence']) for row in
              inv.db.execute("SELECT * FROM events WHERE state='transfer_succeeded'")]
    assert events and events[0]['transfer_mode'] == 'scripted' and events[0]['released'] is True
    assert events[0]['source'] == 'ground_truth'
    inv.close()


def test_mission_refuses_the_scripted_evidence_without_the_ground_truth_opt_in(ik, tmp_path):
    mission, inv = _mission(ik, tmp_path, [])
    assert mission.state == 'failed' and 'requires sensor evidence' in mission.reason
    inv.close()


def test_default_off_keeps_the_host_transfer_path():
    adapter = _adapter()
    adapter.start('transfer', Job('j', 'milk', 1, 'rollwagen', 'G3_back_s2'))
    assert adapter.phase == 'warm'                                      # the existing warm -> host job path


# ---------------------------------------------------------------- host client plumbing (no ROS)

def test_executor_arm_writes_the_goal_and_reads_the_one_result_line(tmp_path):
    procs = []

    class Proc:
        returncode = 0

        def __init__(self, argv, env, stdout, stderr):
            self.argv, self.env = argv, env
            procs.append(self)

        def poll(self):
            return 0

    arm = st.ExecutorArm(runtime_env='/opt/env.sh', workdir=tmp_path, _popen=Proc)
    arm.send('pick', [(0.0, np.zeros(6), np.zeros(6)), (1.0, np.ones(6) * .1, np.zeros(6))])
    goal = json.loads((tmp_path / 'scripted-arm-pick.json').read_text())
    assert goal['action'] == '/restock_sim/arm_controller/follow_joint_trajectory'
    assert goal['names'] == list(st.ARM_JOINTS) and len(goal['points']) == 2
    assert 'ROS_DOMAIN_ID=88' in procs[0].argv[2] and 'source /opt/env.sh' in procs[0].argv[2]
    assert 'AMENT_PREFIX_PATH' not in procs[0].env                      # clean env, like control_ipc
    assert arm.poll()['status'] == 'rejected'                          # exited without a result line
    (tmp_path / 'scripted-arm-pick.log').write_text('noise\nSCRIPTED_ARM_RESULT {"status": "succeeded", "detail": ""}\n')
    assert arm.poll() == {'status': 'succeeded', 'detail': ''}


def test_client_builds_an_immediate_goal_the_executor_accepts():
    class Goal:
        def __init__(self):
            self.trajectory = NS(joint_names=[], points=[], header=NS(stamp=NS(sec=0, nanosec=0)))

    spec = {'names': list(st.ARM_JOINTS),
            'points': [{'t': 0.0, 'q': [0.] * 6, 'v': [0.] * 6}, {'t': 2.25, 'q': [.1] * 6, 'v': [0.] * 6}]}
    goal = client.build_goal(spec, Goal, lambda: NS(positions=[], velocities=[], effort=[], time_from_start=None),
                             lambda sec, nanosec: NS(sec=sec, nanosec=nanosec))
    assert goal.trajectory.joint_names == list(st.ARM_JOINTS)
    assert goal.trajectory.points[1].time_from_start.sec == 2
    assert goal.trajectory.points[1].time_from_start.nanosec == 250000000
    assert goal.trajectory.points[1].effort == []


# ---------------------------------------------------------------- Isaac driver on a real USD stage (usd-core)

def test_isaac_driver_drives_the_carton_kinematically_through_usd():
    pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, '/RestockScenario')
    parent.AddTranslateOp().Set(Gf.Vec3d(0.5, -0.25, 0.0))
    mesh = UsdGeom.Mesh.Define(stage, '/RestockScenario/HandlingUnit')        # authored like scene.py
    mesh.AddTranslateOp().Set(Gf.Vec3d(-3.669, 15.515, 1.0125))
    mesh.AddRotateZOp().Set(-90.0)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
    for path, xyz in (('/Robot/ur_base_link', (-2.079, 15.254, 0.5)), ('/Robot/gripper_pad_tcp', (-2.9, 15.25, 1.0))):
        UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(Gf.Vec3d(*xyz))
    gate = NS(measured=lambda: np.array(STOW))
    driver = st.IsaacScriptedDriver(stage=stage, gate=gate, base_path='/Robot/ur_base_link',
                                    tool_path='/Robot/gripper_pad_tcp', carton_path='/RestockScenario/HandlingUnit')
    before = driver.carton_world_T()
    assert np.allclose(before, CARTON, atol=1e-6)                           # parent offset + rotateZ honoured
    driver.attach_carton()
    prim = stage.GetPrimAtPath('/RestockScenario/HandlingUnit')
    assert [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()] == \
        ['xformOp:translate', 'xformOp:orient']
    assert np.allclose(driver.carton_world_T(), before, atol=1e-6)           # no jump at attach
    assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True
    assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
    carried = st.pose_T((-2.6, 15.6, 1.2), st.yaw_quat(0.7))
    driver.set_carton_world_T(carried)
    assert np.allclose(driver.carton_world_T(), carried, atol=1e-6)
    goal = st.pose_T((TARGET[0], TARGET[1], TARGET[2] + 0.082), st.yaw_quat(math.pi))
    driver.place_carton(goal)
    assert np.allclose(driver.carton_world_T(), goal, atol=1e-6)
    assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
    assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True   # stays kinematic (fm379)
    assert np.allclose(driver.tool_world_T()[:3, 3], (-2.9, 15.25, 1.0))
    assert np.allclose(driver.arm_joint_positions(), STOW)


# ---------------------------------------------------------------- place approach (front insert, opt-in)
# The sales-floor endcap slot G3_back_s2 has its s3 board 0.25 m above the slot surface (0.03 m thick, front
# ticket rail +6 mm); the backroom bay is the same slot geometry with nothing above it. Model that board over
# the bay TARGET in the bay frame (the robot sits on the -y side, so the slot's open front faces -y).
UPPER_BOARD = ((TARGET[0] - 0.47, TARGET[1] - 0.22, TARGET[2] + 0.25),
               (TARGET[0] + 0.47, TARGET[1] + 0.256, TARGET[2] + 0.53))      # s3 board + its products
SLOT_RAIL = ((TARGET[0] - 0.47, TARGET[1] - 0.226, TARGET[2] - 0.036),
             (TARGET[0] + 0.47, TARGET[1] - 0.22, TARGET[2] + 0.006))        # s2 front ticket rail


def _box_hits(trace, slab, tol=1e-3):
    hx, hy, hz = (v / 2 for v in DIMS)
    pts = np.array([[sx * hx, sy * hy, sz * hz, 1.] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    hits = 0
    for T in trace:
        c = (T @ pts.T).T[:, :3]
        lo, hi = c.min(0), c.max(0)
        hits += all(lo[i] < slab[1][i] - tol and hi[i] > slab[0][i] + tol for i in range(3))
    return hits


def test_place_approach_switch_defaults_off_and_fails_loud():
    assert st.scripted_place_approach({}) == st.PLACE_ABOVE
    assert st.scripted_place_approach({st.PLACE_ENV: ' Front '}) == st.PLACE_FRONT
    with pytest.raises(ValueError, match='place approach'):
        st.scripted_place_approach({st.PLACE_ENV: 'sideways'})
    with pytest.raises(ValueError, match='place_approach'):
        _plan_place(None, 'sideways')


def _plan_place(ik, place_approach):
    base_inv = np.linalg.inv(BASE)
    goal = (base_inv @ np.array([TARGET[0], TARGET[1], TARGET[2] + DIMS[2] / 2 + 0.002, 1.0]))[:3]
    return st.plan_transfer(ik, STOW, base_inv @ CARTON, goal, math.radians(0.5), dims=DIMS,
                            place_approach=place_approach)


def test_default_place_is_byte_identical(ik):
    default, above = _plan(ik, True), _plan_place(ik, st.PLACE_ABOVE)
    assert 'place_approach' not in default.choice and default.choice == above.choice
    for a, b in zip(default.keyframes, above.keyframes):
        assert a.name == b.name and a.line == b.line and np.array_equal(a.tcp, b.tcp)


def test_front_place_slides_level_into_the_slot_under_the_upper_board(ik):
    above, front = FakeDriver(ik), FakeDriver(ik)
    assert _run(_runner(ik, above)).status == 'succeeded'
    r = _run(_runner(ik, front, place_approach=st.PLACE_FRONT))
    assert r.status == 'succeeded' and r.evidence['near_face']['place_approach'] == st.PLACE_FRONT
    # the default top-down descent drives the carton through the board; the front insert never touches it
    assert _box_hits(above.carton_trace, UPPER_BOARD) > 0
    assert _box_hits(front.carton_trace, UPPER_BOARD) == 0
    assert _box_hits(front.carton_trace, SLOT_RAIL) == 0
    assert all(abs(T[2, 2]) > 0.999 for T in front.carton_trace)                 # level throughout
    assert np.allclose(front.c[:3, 3], (TARGET[0], TARGET[1], TARGET[2] + 0.082))  # exact ground-truth place
    plan = _plan_place(ik, st.PLACE_FRONT)
    pre, place = (next(k for k in plan.keyframes if k.name == n) for n in ('above_place', 'place'))
    assert pre.line and place.line                                       # straight TCP lines, no joint swing
    assert pre.tcp[2, 3] == pytest.approx(place.tcp[2, 3])               # the slide-in is level
    assert not plan.ik_complete                                          # task-space plan: still arm only


def test_front_place_is_labelled_only_when_scripted_and_on():
    on = {**SCRIPTED_ENV, st.PLACE_ENV: 'front'}
    s = st.stamp_place_summary({'note': 'n', 'walkthrough_caption': 'cap', 'workarounds': ['transfer_mode=scripted']},
                               on)
    assert s['transfer_place_mode'] == 'front_insert'
    assert s['workarounds'] == ['transfer_mode=scripted', st.PLACE_LABEL]
    assert s['walkthrough_caption'] == f'cap | {st.PLACE_CAPTION}' and 'TRANSFER_PLACE=front_insert' in s['note']
    assert st.stamp_place_summary(dict(s), on) == s                       # idempotent
    for env in ({}, SCRIPTED_ENV, {st.PLACE_ENV: 'front'}, {**SCRIPTED_ENV, st.PLACE_ENV: 'bogus'}):
        assert st.stamp_place_summary({'note': 'n'}, env) == {'note': 'n'}
        assert st.place_caption('cap', env) == 'cap'
