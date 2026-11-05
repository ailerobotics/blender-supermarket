"""CPU proof for the perceived-clearance collision gate (no GPU/ROS).

Covers the pure ESDF geometry + query (esdf_grid), the per-leg decision + logging
(collision_gate), the CPU ESDF providers (esdf_source), and the motion-port hook
(gated_motion): a planted obstacle in a leg's swept corridor is refused by name, a clear leg is
delegated byte-for-byte, an absent/stale/unobserved map holds, and the wrapper is byte-identical
when not installed.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from twin.restock_manipulation import esdf_grid, collision_gate, esdf_source, gated_motion
from twin.restock_manipulation.cad_tool import ARM_JOINTS

PAD_MIN = (-0.145, -0.067, -0.272)
PAD_MAX = (0.146, 0.059, 0.002)


# ------------------------------------------------------------------------------- esdf_grid --
def test_snapshot_rejects_bad_grid():
    with pytest.raises(ValueError):
        esdf_grid.EsdfSnapshot(np.zeros((2, 2)), (0, 0, 0), 0.1, 0.0)        # 2-D
    with pytest.raises(ValueError):
        esdf_grid.EsdfSnapshot(np.zeros((2, 2, 2)), (0, 0, 0), 0.0, 0.0)     # voxel 0


def test_clearance_over_box_and_unobserved():
    grid = np.full((10, 10, 10), 0.5, np.float32)
    grid[5, 5, 5] = 0.01                          # a near-surface voxel
    grid[0, 0, 0] = -1000.0                        # unobserved sentinel
    snap = esdf_grid.EsdfSnapshot(grid, (0, 0, 0), 0.1, 1.0)
    q = snap.clearance_over_box((0.5, 0.5, 0.5), (0.6, 0.6, 0.6))
    assert q.min_observed == pytest.approx(0.01, abs=1e-6)
    # A box over the unobserved voxel counts it as unobserved, not free.
    q2 = snap.clearance_over_box((0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
    assert q2.unobserved >= 1
    # A box entirely outside the grid is fully unobserved (not free).
    q3 = snap.clearance_over_box((5.0, 5.0, 5.0), (5.1, 5.1, 5.1))
    assert q3.fully_unobserved and q3.unobserved_fraction == 1.0


def test_pose_aabb_matches_translation_sweep():
    low, high = esdf_grid.pose_aabb((1.0, 0.0, 0.0, 0, 0, 0, 1), PAD_MIN, PAD_MAX)
    assert low[0] == pytest.approx(1.0 + PAD_MIN[0])
    assert high[0] == pytest.approx(1.0 + PAD_MAX[0])


def test_combined_envelope_folds_carton():
    lo, hi = esdf_grid.combined_envelope(PAD_MIN, PAD_MAX, (0.45, 0.30, 0.18))
    # The carton extends the envelope down along +Z (grasp axis) by its height.
    assert hi[2] >= 0.18 - 1e-9
    # And widens x/y to at least the carton half-extents.
    assert lo[0] <= -0.45 / 2 + 1e-9 and hi[1] >= 0.30 / 2 - 1e-9


# ----------------------------------------------------------------------------- esdf_source --
def _post_source(x0=0.40, voxel=0.03):
    """A single cart-post-like box obstacle in the base frame, fully observed."""
    return esdf_source.AnalyticBoxEsdfSource(
        [((x0, -0.10, -0.20), (x0 + 0.10, 0.10, 1.0))], voxel_size=voxel, labels=['cart_post'])


def test_analytic_source_signed_distance():
    snap = _post_source().snapshot(10.0)
    # A point well clear of the post reads positive; inside reads negative.
    assert snap.clearance_over_box((-0.2, 0.0, 0.3), (-0.15, 0.05, 0.35)).min_observed > 0.1
    assert snap.clearance_over_box((0.44, 0.0, 0.3), (0.46, 0.05, 0.35)).min_observed < 0.0


def test_moving_obstacle_source_is_time_varying():
    boxes = lambda now: [((now, -0.1, 0.0), (now + 0.1, 0.1, 1.0))]
    src = esdf_source.AnalyticBoxEsdfSource(boxes, voxel_size=0.05)
    a = src.snapshot(0.0).clearance_over_box((0.0, 0.0, 0.5), (0.05, 0.05, 0.55)).min_observed
    b = src.snapshot(0.5).clearance_over_box((0.0, 0.0, 0.5), (0.05, 0.05, 0.55)).min_observed
    assert b > a                                   # the post moved away in +x


# ---------------------------------------------------------------------------- collision_gate --
def test_gate_refuses_planted_obstacle_accepts_clear():
    logs = []
    gate = collision_gate.PerceptionCollisionGate(margin_m=0.03, mode='refuse',
                                                  logger=lambda *a, **k: logs.append(a[0]))
    snap = _post_source().snapshot(10.0)
    clear = gate.evaluate(phase='approach', start_pose=(-0.3, 0, 0.4, 0, 0, 0, 1),
                          goal_pose=(-0.2, 0, 0.4, 0, 0, 0, 1), pad_min=PAD_MIN, pad_max=PAD_MAX,
                          snapshot=snap, now=10.1)
    assert clear.action == 'proceed' and clear.ok and clear.min_clearance > 0.03
    hit = gate.evaluate(phase='approach', start_pose=(0.1, 0, 0.4, 0, 0, 0, 1),
                        goal_pose=(0.45, 0, 0.4, 0, 0, 0, 1), pad_min=PAD_MIN, pad_max=PAD_MAX,
                        snapshot=snap, now=10.2)
    assert hit.action == 'refuse' and hit.blocked and hit.nearest_body == 'cart_post'
    assert any('PERCEIVED_CLEARANCE' in m for m in logs)


def test_gate_holds_on_absent_and_stale_snapshot():
    gate = collision_gate.PerceptionCollisionGate(mode='refuse', unknown_action='hold',
                                                  max_snapshot_age_s=0.5, logger=None)
    v_none = gate.evaluate(phase='approach', start_pose=(0, 0, 0.3, 0, 0, 0, 1),
                           goal_pose=(0.1, 0, 0.3, 0, 0, 0, 1), pad_min=PAD_MIN, pad_max=PAD_MAX,
                           snapshot=None, now=1.0)
    assert v_none.action == 'hold'
    snap = _post_source().snapshot(1.0)
    v_stale = gate.evaluate(phase='approach', start_pose=(-0.3, 0, 0.3, 0, 0, 0, 1),
                            goal_pose=(-0.2, 0, 0.3, 0, 0, 0, 1), pad_min=PAD_MIN, pad_max=PAD_MAX,
                            snapshot=snap, now=5.0)                 # age 4 s > 0.5 s
    assert v_stale.action == 'hold'


def test_gate_holds_on_unobserved_corridor():
    # A small fully-observed grid; a leg whose swept volume leaves the grid is mostly unobserved.
    src = esdf_source.AnalyticBoxEsdfSource([((2.0, 2.0, 2.0), (2.1, 2.1, 2.1))],
                                            origin=(0.0, 0.0, 0.0), size=(0.3, 0.3, 0.3),
                                            voxel_size=0.05)
    snap = src.snapshot(1.0)
    gate = collision_gate.PerceptionCollisionGate(mode='refuse', unknown_action='hold',
                                                  unobserved_tol=0.2, logger=None)
    v = gate.evaluate(phase='approach', start_pose=(1.0, 1.0, 1.0, 0, 0, 0, 1),
                      goal_pose=(1.2, 1.0, 1.0, 0, 0, 0, 1), pad_min=PAD_MIN, pad_max=PAD_MAX,
                      snapshot=snap, now=1.1)
    assert v.action == 'hold' and v.unobserved_fraction > 0.2


def test_gate_inactive_phase_always_proceeds():
    gate = collision_gate.PerceptionCollisionGate(mode='refuse', phases={'approach'}, logger=None)
    v = gate.evaluate(phase='observe', start_pose=(0, 0, 0, 0, 0, 0, 1),
                      goal_pose=(1, 0, 0, 0, 0, 0, 1), pad_min=PAD_MIN, pad_max=PAD_MAX,
                      snapshot=None, now=1.0)
    assert v.action == 'proceed'


def test_warn_mode_measures_but_proceeds():
    gate = collision_gate.PerceptionCollisionGate(margin_m=0.03, mode='warn', logger=None)
    snap = _post_source().snapshot(10.0)
    v = gate.evaluate(phase='approach', start_pose=(0.1, 0, 0.4, 0, 0, 0, 1),
                      goal_pose=(0.45, 0, 0.4, 0, 0, 0, 1), pad_min=PAD_MIN, pad_max=PAD_MAX,
                      snapshot=snap, now=10.2)
    assert v.action == 'warn' and v.ok and v.min_clearance < 0.03   # measured, but proceeds


# ------------------------------------------------------------------------------ gated_motion --
class _MockPort:
    """Records the begin* calls that are actually dispatched (i.e. passed the gate)."""
    def __init__(self):
        self.calls = []

    def begin(self, pose, joints, *a, **k):
        self.calls.append(('begin', tuple(pose)))

    def begin_joint_goal(self, goal, measured, *a, **k):
        self.calls.append(('begin_joint_goal', tuple(goal)))

    def begin_joint_path(self, waypoints, measured, *a, **k):
        self.calls.append(('begin_joint_path', len(waypoints)))

    def poll(self):
        return 'running'

    def stop(self):
        self.calls.append(('stop',))

    def reset_crash_budget(self):          # a delegated-through method
        self.calls.append(('reset_crash_budget',))


_URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'


def _robot():
    from twin.urdf_probe import parse_urdf
    return parse_urdf(_URDF)


import os  # noqa: E402

skip_no_urdf = pytest.mark.skipif(not os.path.isfile(_URDF),
                                  reason='dataset arm URDF not on this box')


def _gated(inner, source, *, phase='approach', held=False, carton=None, mode='refuse'):
    gate = collision_gate.PerceptionCollisionGate(margin_m=0.03, mode=mode, logger=None)
    ctrl = SimpleNamespace(phase=phase, held=held,
                           profile=SimpleNamespace(dimensions=carton) if carton else None)
    return gated_motion.GatedMotionPort(
        inner, gate=gate, esdf_source=source, robot=_robot(),
        envelope_pad_min=PAD_MIN, envelope_pad_max=PAD_MAX,
        get_phase=lambda: ctrl.phase, get_held=lambda: ctrl.held,
        get_carton_dims=lambda: (tuple(ctrl.profile.dimensions) if ctrl.profile else None),
        steady_clock=lambda: 10.0), ctrl


@skip_no_urdf
def test_gated_port_delegates_clear_leg_and_passes_through():
    inner = _MockPort()
    # An empty ESDF (no obstacles) -> the leg is clear -> dispatched.
    port, _ = _gated(inner, esdf_source.AnalyticBoxEsdfSource([((5, 5, 5), (5.1, 5.1, 5.1))]))
    port.begin_joint_goal((0, -1.2, 1.2, 0, 0, 0), (0, -1.2, 1.2, 0, 0, 0))
    assert inner.calls == [('begin_joint_goal', (0, -1.2, 1.2, 0, 0, 0))]
    # __getattr__ delegation: a non-begin method reaches the inner port.
    port.reset_crash_budget()
    assert ('reset_crash_budget',) in inner.calls


@skip_no_urdf
def test_gated_port_refuses_leg_into_planted_obstacle():
    from twin.urdf_probe import link_world_transforms
    inner = _MockPort()
    robot = _robot()
    # FK the arm's zero-config TCP, then plant a box AT that TCP so any leg there collides.
    tcp = link_world_transforms(robot, {}, root='ur_base_link')['gripper_pad_tcp'][:3, 3]
    box = ((tcp[0] - 0.05, tcp[1] - 0.05, tcp[2] - 0.05), (tcp[0] + 0.05, tcp[1] + 0.05, tcp[2] + 0.05))
    # A synthetic grid that covers the TCP region, so the planted box is OBSERVED (not unknown).
    src = esdf_source.AnalyticBoxEsdfSource(
        [box], origin=(tcp[0] - 0.6, tcp[1] - 0.6, tcp[2] - 0.6), size=(1.2, 1.2, 1.2),
        voxel_size=0.03, labels=['planted'])
    port, _ = _gated(inner, src)
    with pytest.raises(gated_motion.GateRefusal) as exc:
        port.begin_joint_goal((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0))
    assert 'PERCEIVED_CLEARANCE_REFUSE' in str(exc.value)
    assert inner.calls == []                       # never dispatched


@skip_no_urdf
def test_gated_port_inactive_phase_is_byte_identical():
    inner = _MockPort()
    # Plant an obstacle AT the TCP, but the phase is not gated -> dispatched unchanged.
    from twin.urdf_probe import link_world_transforms
    tcp = link_world_transforms(_robot(), {}, root='ur_base_link')['gripper_pad_tcp'][:3, 3]
    box = ((tcp[0] - 0.1, tcp[1] - 0.1, tcp[2] - 0.1), (tcp[0] + 0.1, tcp[1] + 0.1, tcp[2] + 0.1))
    port, _ = _gated(inner, esdf_source.AnalyticBoxEsdfSource([box]), phase='observe')
    port.begin_joint_goal((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0))
    assert len(inner.calls) == 1                   # dispatched despite the obstacle


def test_build_gated_motion_port_off_is_identity():
    inner = _MockPort()
    out = gated_motion.build_gated_motion_port(
        inner, config={}, tool=SimpleNamespace(envelope_pad_min=PAD_MIN, envelope_pad_max=PAD_MAX,
                                               tcp='gripper_pad_tcp'),
        urdf_path=_URDF, get_phase=lambda: 'approach', get_held=lambda: False,
        get_carton_dims=lambda: None, steady_clock=lambda: 0.0, node=None)
    assert out is inner                            # disabled -> the raw port (byte-identical)
