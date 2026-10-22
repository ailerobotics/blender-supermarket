#!/usr/bin/env python3
"""cuRobo MPC reactive follow-and-avoid run for the MISSION arm (SIM/EVAL ONLY).

The GPU/cuRobo-dependent half of the bench. Imported machinery (world model,
scenarios, clearance, metrics, the gate trial) lives in the PURE, CPU-tested
``arm_mpc_bench`` module; here we add only the cuRobo solver loop. Runs INSIDE the
reused ``fm-plan-mpc-curobo-v1`` container (cuRobo V2, Apache-2.0 -- never a v0.7.x
NVIDIA-non-commercial tag), driven by ``scripts/run_arm_mpc_bench.py`` under the
GPU lock. No Isaac Sim is needed: cuRobo MPC is pure cuRobo + torch.

Measures per scenario: control rate, warm solve time, est. process VRAM, tracking
error, obstacle + person clearance, joint-window violations, and -- against the
LIVE MPC command stream -- the merged human-safety gate's behaviour (RUN forwards,
STOP holds with zero travel, resume heads to the freshest command).

Reuses the mission arm's own cuRobo config ``shelfbot_curobo.yml`` (same joint
windows / collision spheres the mission cuMotion path uses). The soft collision
cost is NOT a safety function (see arm_mpc_bench docstring).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time

# Make the pure bench core importable whether run from the repo or a container mount.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../scripts/twin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # .../scripts
try:
    from twin.tools import arm_mpc_bench as bench
except ImportError:
    from tools import arm_mpc_bench as bench  # type: ignore


def _gpu_used_mib():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return float('nan')


BASELINE_GPU_MIB = _gpu_used_mib()


class GpuSampler(threading.Thread):
    """Sample global GPU memory.used at ~5 Hz; keep the peak."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = BASELINE_GPU_MIB
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _gpu_used_mib())
            time.sleep(0.2)

    def stop(self):
        self._stop.set()


def robot_spheres(mpc, state):
    """Robot collision spheres (x,y,z,r) at this state from cuRobo FK, or []."""
    res = mpc.compute_kinematics(state)
    for attr in ('robot_spheres', 'link_spheres_tensor', 'robot_spheres_tensor'):
        t = getattr(res, attr, None)
        if t is not None:
            arr = t.detach().reshape(-1, 4).cpu().tolist()
            return [(a[0], a[1], a[2], a[3]) for a in arr if a[3] > 0]
    return []


def _mk_pose(x, y, z):
    from curobo.types import Pose
    for maker in (
        lambda: Pose.from_list([x, y, z, 1.0, 0.0, 0.0, 0.0]),
        lambda: Pose(position=[x, y, z], quaternion=[1.0, 0.0, 0.0, 0.0]),
    ):
        try:
            return maker()
        except Exception:  # noqa: BLE001
            continue
    from curobo.types import Pose as P
    return P.from_numpy([x, y, z], [1.0, 0.0, 0.0, 0.0])


def advance(mpc, state, JointState):
    res = mpc.optimize_action_sequence(state)
    if res.action_sequence is None or res.action_sequence.position.shape[1] == 0:
        return state, res
    nxt = res.action_sequence.position[:, -1, :]
    ns = JointState.from_position(nxt.clone(), joint_names=mpc.joint_names)
    ns.velocity = res.action_sequence.velocity[:, -1, :]
    ns.acceleration = res.action_sequence.acceleration[:, -1, :]
    return ns, res


def _gate_timeline(scenario, steps, gate):
    """Scripted gate-state timeline driven alongside the live MPC stream: RUN for the
    first and last thirds, STOP in the middle third. Proves -- against REAL MPC
    commands -- that STOP holds (zero travel) and RUN resumes toward the freshest."""
    a, b = steps // 3, 2 * steps // 3
    tl = []
    for i in range(steps):
        if i < a or i >= b:
            tl.append((gate.RUN, 1.0))
        else:
            tl.append((gate.STOP, 0.0))
    return tl


def run_scenario(args, gate, windows):
    import torch
    from curobo.model_predictive_control import (
        ModelPredictiveControl, ModelPredictiveControlCfg)
    from curobo.types import GoalToolPose, JointState

    aisle = bench.StoreAisle()
    world = (bench.store_world_with_person(aisle)
             if bench.scenario_needs_person(args.scenario) else bench.store_world(aisle))

    sampler = GpuSampler()
    sampler.start()

    cfg = ModelPredictiveControlCfg.create(
        robot=args.robot, scene_model=world, use_cuda_graph=True,
        optimization_dt=args.optimization_dt,
        interpolation_steps=args.interpolation_steps,
        optimizer_collision_activation_distance=args.collision_activation)
    mpc = ModelPredictiveControl(cfg)

    start = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names)
    start.velocity = torch.zeros_like(start.position)
    start.acceleration = torch.zeros_like(start.position)
    mpc.setup(start)

    link = mpc.tool_frames[0]
    anchor_pose = mpc.compute_kinematics(start).tool_poses.to_dict()[link]
    center = anchor_pose.position.squeeze().detach().cpu().tolist()
    start_quat = anchor_pose.quaternion.squeeze().detach().cpu().tolist()

    # Rebase the representative obstacles onto the arm's measured FK reach centre so
    # the shelf sits just in front of wherever the mission arm actually reaches.
    delta = [center[i] - bench.nominal_reach(aisle)[i] for i in range(3)]
    world = bench.shift_world(world, delta)
    # Re-apply the rebased world to the optimiser's scene (cuboids moved in bulk).
    for name, c in world['cuboid'].items():
        if name in mpc.scene_collision_checker.get_obstacle_names():
            mpc.scene_collision_checker.update_obstacle_pose(name, _mk_pose(*c['pose'][:3]))

    target_fn = bench.make_target_fn(args.scenario, center)
    person_fn = bench.make_person_fn(args.scenario, center)

    def set_goal(xyz, run_ik):
        poses = mpc.compute_kinematics(state).tool_poses.to_dict()
        poses[link].position[..., 0] = xyz[0]
        poses[link].position[..., 1] = xyz[1]
        poses[link].position[..., 2] = xyz[2]
        poses[link].quaternion[..., 0] = start_quat[0]
        poses[link].quaternion[..., 1] = start_quat[1]
        poses[link].quaternion[..., 2] = start_quat[2]
        poses[link].quaternion[..., 3] = start_quat[3]
        mpc.update_goal_tool_poses(
            GoalToolPose.from_poses(poses, ordered_tool_frames=mpc.tool_frames,
                                    num_goalset=1),
            run_ik=run_ik)

    state = start
    set_goal(target_fn(0.0), run_ik=True)
    for _ in range(8):
        state, _ = advance(mpc, state, JointState)
    torch.cuda.synchronize()

    if person_fn is not None:
        names = mpc.scene_collision_checker.get_obstacle_names()
        assert bench.PERSON in names, f'person obstacle not updatable; scene has {names}'

    # Live gate alongside the MPC stream (RUN/STOP/RUN) -- proves gate behaviour on GPU.
    tl = _gate_timeline(args.scenario, args.steps, gate)
    gctrl = gate.GatedStreamController(
        windows=windows, joint_names=list(bench.ARM_JOINTS), max_step_per_tick=0.01)
    gctrl.seed(state.position.squeeze().detach().cpu().tolist())
    gate_travel_run = gate_travel_stop = 0.0
    gate_fwd = gate_held = gate_refusals = gate_dropped_stop = 0
    gate_resumed = False
    saw_stop = False
    gate_prev = state.position.squeeze().detach().cpu().tolist()

    dt = args.optimization_dt
    solve_ms, track_err = [], []
    min_clear = min_person = float('inf')
    window_viol, max_excess = 0, 0.0
    for i in range(args.steps):
        t = i * dt
        set_goal(target_fn(t), run_ik=(i % 40 == 0))
        if person_fn is not None:
            px, py, _ = person_fn(t)
            pose = bench.person_pose(px + delta[0], py + delta[1])
            mpc.scene_collision_checker.update_obstacle_pose(bench.PERSON, _mk_pose(*pose[:3]))
            world['cuboid'][bench.PERSON]['pose'] = pose

        tic = time.time()
        state, res = advance(mpc, state, JointState)
        torch.cuda.synchronize()
        solve_ms.append(1e3 * (time.time() - tic))

        cur = mpc.compute_kinematics(state).tool_poses.to_dict()[link]
        cpos = cur.position.squeeze().detach().cpu().tolist()
        track_err.append(math.dist(cpos, target_fn(t)))

        qpos = state.position.squeeze().detach().cpu().tolist()
        qd = {j: float(v) for j, v in zip(mpc.joint_names, qpos)}
        excess = max((max(lo - qd[j], qd[j] - hi, 0.0)
                      for j, (lo, hi) in windows.windows.items() if j in qd), default=0.0)
        max_excess = max(max_excess, excess)
        if not windows.within(qd):
            window_viol += 1

        sph = robot_spheres(mpc, state)
        if sph:
            min_clear = min(min_clear, bench.min_sphere_clearance(sph, world))
            if person_fn is not None:
                min_person = min(min_person,
                                 bench.min_sphere_clearance(sph, bench.person_only_world(world)))

        # drive the gate with the freshest MPC command this tick
        action, frac = tl[i]
        sa = gctrl.step(action, frac, qpos, measured_position=gate_prev)
        step_mag = sum(abs(x - y) for x, y in zip(sa.target_position, gate_prev))
        if action == gate.RUN:
            gate_travel_run += step_mag
            if saw_stop and step_mag > 1e-9 and sa.kind == gate.FORWARD:
                gate_resumed = True
        else:
            gate_travel_stop += step_mag
            saw_stop = True
            gate_dropped_stop += 1
        gate_fwd += int(sa.kind == gate.FORWARD)
        gate_held += int(sa.kind == gate.HOLD)
        gate_refusals += int(not sa.window_ok)
        gate_prev = list(sa.target_position)

    sampler.stop()
    half = len(track_err) // 2
    m = bench.MpcScenarioMetrics(
        scenario=args.scenario, steps=args.steps,
        control_rate_hz=1.0 / statistics.mean([s / 1e3 for s in solve_ms]),
        solve_ms_mean=statistics.mean(solve_ms),
        solve_ms_p95=sorted(solve_ms)[int(0.95 * len(solve_ms)) - 1],
        solve_ms_max=max(solve_ms),
        tracking_steady_mean_m=statistics.mean(track_err[half:]),
        tracking_steady_max_m=max(track_err[half:]),
        min_clearance_m=(min_clear if min_clear != float('inf') else None),
        min_person_clearance_m=(min_person if min_person != float('inf') else None),
        joint_window_violations=window_viol,
        max_window_excess_rad=max_excess,
        est_process_mib=(sampler.peak - BASELINE_GPU_MIB),
        gate={'travel_while_run_rad': round(gate_travel_run, 6),
              'travel_while_stop_rad': round(gate_travel_stop, 6),
              'forwarded': gate_fwd, 'held': gate_held,
              'dropped_during_stop': gate_dropped_stop,
              'window_refusals': gate_refusals,
              'resumed_to_fresh': gate_resumed,
              'stop_holds': gate_travel_stop <= 1e-9},
        note=bench.behaviour_note(args.scenario))
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', choices=list(bench.SCENARIOS) + ['all'], default='all')
    ap.add_argument('--robot', default=os.environ.get(
        'SHELFBOT_CUROBO_YML', '/shelfbot/description/shelfbot_curobo.yml'))
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--optimization-dt', type=float, default=0.03)
    ap.add_argument('--interpolation-steps', type=int, default=4)
    ap.add_argument('--collision-activation', type=float, default=0.05)
    ap.add_argument('--mpc-dir', default=None, help='plan-mpc-curobo-v1 mpc_curobo dir (gate reuse)')
    ap.add_argument('--limits', default=os.environ.get('SHELFBOT_URDF', None),
                    help='mission robot URDF for the measured joint windows')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    gate = bench.load_gate(args.mpc_dir)
    windows = _load_windows(args, gate)

    scenarios = list(bench.SCENARIOS) if args.scenario == 'all' else [args.scenario]
    results = []
    for sc in scenarios:
        args.scenario = sc
        m = run_scenario(args, gate, windows)
        results.append(m)
        print(json.dumps(m.as_dict(), indent=2))

    print('\n' + bench.metrics_table(results))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump({'curobo': '0.8.0 (Apache-2.0)', 'robot': os.path.basename(args.robot),
                       'scenarios': [m.as_dict() for m in results]}, fh, indent=2)
        print('wrote', args.out)
    return 0


def _load_windows(args, gate):
    """Measured joint windows for the mission arm, read from the arm's own URDF
    ``<limit>`` tags (the envelope the trajectory host enforces). Uses the repo's
    general ``twin.urdf_probe`` parser -- NOT ``cad_tool.inspect_tool``, which is
    specialised for the CAD panel/tine tool and rejects the arm URDF. Falls back to
    UR10 nominal +-2pi if the URDF is absent, so the independent guard still runs."""
    if args.limits and os.path.exists(args.limits):
        try:
            from twin.urdf_probe import parse_urdf  # noqa: E402
        except ImportError:
            from urdf_probe import parse_urdf  # type: ignore
        jb = parse_urdf(args.limits).joints_by_name
        arm_limits = []
        for j in bench.ARM_JOINTS:
            jj = jb.get(j)
            if jj is None or jj.lower is None or jj.upper is None:
                arm_limits = None
                break
            arm_limits.append((jj.lower, jj.upper))
        if arm_limits is not None:
            return bench.mission_windows(arm_limits, gate)
    wide = [(-2 * math.pi, 2 * math.pi)] * len(bench.ARM_JOINTS)
    return bench.mission_windows(wide, gate)


if __name__ == '__main__':
    raise SystemExit(main())
