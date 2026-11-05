"""Perceived-clearance gate at the motion-port seam (the additive hook point).

:class:`GatedMotionPort` wraps any :class:`twin.restock_manipulation.types.MotionPort` and,
before it dispatches an arm leg, consults a :class:`collision_gate.PerceptionCollisionGate`
against the live ESDF (:mod:`esdf_source`) for the swept gripper + welded-carton volume. A leg
the gate blocks is never dispatched: the wrapper raises :class:`GateRefusal` BEFORE delegating,
and the controller's ``step`` catch-all / its ``_issue_*`` wrappers turn that into a named
``_fail`` and stop the arm honestly (the mission's StageRetry then re-begins against a fresher
map -- a bounded hold-and-retry). Every leg the gate passes is delegated byte-for-byte.

This is the ONLY integration touch in the motion chain, and it is **opt-in**: the wrapper is
installed (in :mod:`ros_host`) only when the host key ``perception_collision_gate.enabled`` is
set. With the key absent the raw port is used and the whole perception layer is inert -- the
complete-cycle runs are byte-identical.

Why a port wrapper and not a controller edit: it gates EVERY leg (pick approach, lift, tuck,
pre-place, place descent, withdraw, every transfer leg -- they all flow through ``begin`` /
``begin_joint_goal`` / ``begin_joint_path``) with zero edits to the 3272-line controller, so it
cannot disturb the pre-place planner (``_issue_gated_preplace_approach``,
``carton_level.plan_level_world_approach``, ``select_reaction_natural_standoff``) that a sibling
lane is actively changing, and it is trivially byte-identical when off.

Hook points for the sibling lanes (sim-arm-reactive-replan-v1, sim-arm-collision-bench-v1):

* the wrapper exposes ``gate`` and ``esdf_source`` attributes (the reactive lane subscribes a
  re-plan/preempt callback; the bench swaps the source for scripted static + moving obstacles);
* :meth:`GatedMotionPort.leg_corridor` returns the swept AABB corridor for a (start, goal) leg
  -- the reactive lane watches it between dispatch and completion for newly-entered obstacles.
"""
from __future__ import annotations

import math

import numpy as np

from . import collision_gate, esdf_grid, esdf_source
from .cad_tool import ARM_JOINTS
from ..urdf_probe import link_world_transforms, parse_urdf


class GateRefusal(RuntimeError):
    """Raised at the port seam when the perceived-clearance gate blocks a leg.

    The controller's ``step``/``_issue_*`` paths catch it and fail the phase by name, so the
    reason string (``PERCEIVED_CLEARANCE_<ACTION> ...``) reaches the mission log and the run
    record exactly like any other named arm fault.
    """


class GatedMotionPort:
    """Motion-port wrapper that gates each leg on perceived clearance; delegates the rest."""

    def __init__(self, inner, *, gate, esdf_source, robot, tcp_link='gripper_pad_tcp',
                 arm_joints=ARM_JOINTS, root='ur_base_link',
                 envelope_pad_min, envelope_pad_max, get_phase, get_held, get_carton_dims,
                 steady_clock, grasp_axis=2, carton_face_offset=0.0, logger=print):
        if tcp_link not in robot.links:
            raise ValueError('TCP link %r is not in the gate FK URDF' % tcp_link)
        self._inner = inner
        self.gate = gate
        self.esdf_source = esdf_source
        self._robot = robot
        self._tcp_link = tcp_link
        self._arm_joints = tuple(arm_joints)
        self._root = root
        self._pad_min = tuple(float(x) for x in envelope_pad_min)
        self._pad_max = tuple(float(x) for x in envelope_pad_max)
        self._get_phase = get_phase
        self._get_held = get_held
        self._get_carton_dims = get_carton_dims
        self._steady_clock = steady_clock
        self._grasp_axis = int(grasp_axis)
        self._carton_face_offset = float(carton_face_offset)
        self._logger = logger

    # -- MotionPort surface: intercept the three dispatch methods, delegate everything else --
    def begin(self, pose, joints, *args, **kwargs):
        self._gate_leg(start_joints=joints, goal_pose=tuple(pose))
        return self._inner.begin(pose, joints, *args, **kwargs)

    def begin_joint_goal(self, goal_joints, measured_joints, *args, **kwargs):
        self._gate_leg(start_joints=measured_joints, goal_joints=goal_joints)
        return self._inner.begin_joint_goal(goal_joints, measured_joints, *args, **kwargs)

    def begin_joint_path(self, waypoints, measured_joints, *args, **kwargs):
        self._gate_path([measured_joints, *list(waypoints)])
        return self._inner.begin_joint_path(waypoints, measured_joints, *args, **kwargs)

    def __getattr__(self, name):
        # Everything the controller/host calls beyond the three begin* methods (poll, stop,
        # update_payload, reset_crash_budget, the fault probes, ...) delegates to the inner port.
        return getattr(self._inner, name)

    # ----------------------------------------------------------------------------- internals --
    def _tcp(self, joints) -> tuple:
        q = np.asarray(joints, float)
        if q.shape != (len(self._arm_joints),) or not np.isfinite(q).all():
            raise ValueError('finite arm joint vector required for gate FK')
        config = dict(zip(self._arm_joints, (float(v) for v in q)))
        world = link_world_transforms(self._robot, config, root=self._root)
        transform = world[self._tcp_link]
        pos = transform[:3, 3]
        quat = esdf_grid.rotation_to_quat(transform[:3, :3])
        return (float(pos[0]), float(pos[1]), float(pos[2]), *(float(v) for v in quat))

    def _envelope(self, held) -> tuple:
        carton = self._get_carton_dims() if held else None
        return esdf_grid.combined_envelope(self._pad_min, self._pad_max, carton,
                                           grasp_axis=self._grasp_axis,
                                           face_offset=self._carton_face_offset)

    def leg_corridor(self, start_pose, goal_pose, *, held=False):
        """Swept gripper(+carton) AABB corridor for a (start, goal) leg -- a hook for the
        reactive-replan lane to watch and the bench to plant against."""
        pad_min, pad_max = self._envelope(held)
        return collision_gate.swept_corridor_aabbs(start_pose, goal_pose, pad_min, pad_max,
                                                   samples=self.gate.samples)

    def _snapshot(self, now):
        try:
            return self.esdf_source.snapshot(now)
        except Exception as exc:                       # noqa: BLE001 -- never let a source error
            # pass as free space; a source failure is reported and the gate holds (unknown).
            if self._logger is not None:
                self._logger('PERCEIVED_CLEARANCE_SOURCE_ERROR %s: %s'
                             % (type(exc).__name__, exc), flush=True)
            return None

    def _gate_leg(self, *, start_joints, goal_pose=None, goal_joints=None):
        phase = self._get_phase()
        if not self.gate.active(phase):
            return
        held = bool(self._get_held())
        start = self._tcp(start_joints)
        goal = tuple(goal_pose) if goal_pose is not None else self._tcp(goal_joints)
        pad_min, pad_max = self._envelope(held)
        now = float(self._steady_clock())
        snapshot = self._snapshot(now)
        verdict = self.gate.evaluate(phase=phase, start_pose=start, goal_pose=goal,
                                     pad_min=pad_min, pad_max=pad_max, snapshot=snapshot,
                                     now=now, held=held)
        self._enforce(verdict)

    def _gate_path(self, configs):
        phase = self._get_phase()
        if not self.gate.active(phase):
            return
        held = bool(self._get_held())
        pad_min, pad_max = self._envelope(held)
        now = float(self._steady_clock())
        snapshot = self._snapshot(now)
        boxes = []
        for a, b in zip(configs, configs[1:]):
            boxes.extend(collision_gate.swept_corridor_aabbs(
                self._tcp(a), self._tcp(b), pad_min, pad_max, samples=self.gate.samples))
        verdict = self.gate.evaluate_corridor(phase=phase, boxes=boxes, snapshot=snapshot,
                                              now=now, held=held)
        self._enforce(verdict)

    def _enforce(self, verdict):
        if verdict.blocked:
            raise GateRefusal('PERCEIVED_CLEARANCE_%s phase=%s %s'
                              % (verdict.action.upper(), verdict.phase, verdict.reason))


# ------------------------------------------------------------------------------- builder ----
def _gate_from_config(block, *, logger=print):
    return collision_gate.PerceptionCollisionGate(
        margin_m=float(block.get('margin_m', 0.03)),
        mode=str(block.get('mode', 'refuse')),
        unknown_action=str(block.get('unknown_action', 'hold')),
        phases=(frozenset(block['phases']) if block.get('phases') is not None else None),
        samples=int(block.get('samples', 6)),
        unobserved_tol=float(block.get('unobserved_tol', 0.2)),
        max_snapshot_age_s=float(block.get('max_snapshot_age_s', 0.5)),
        logger=logger)


def _source_from_config(block, *, node=None, logger=print):
    kind = str(block.get('source', 'nvblox'))
    origin = tuple(block.get('workspace_origin_m', esdf_source.DEFAULT_WORKSPACE_ORIGIN_M))
    size = tuple(block.get('workspace_size_m', esdf_source.DEFAULT_WORKSPACE_SIZE_M))
    if kind == 'nvblox':
        if node is None:
            raise ValueError('perception_collision_gate source=nvblox requires a host ROS node')
        return esdf_source.NvbloxEsdfSource(
            node, origin=origin, size=size,
            spin_budget_s=float(block.get('esdf_spin_budget_s', 0.2)))
    if kind == 'replay':
        return esdf_source.ReplayEsdfSource(block['replay_npz'])
    raise ValueError('unknown perception_collision_gate source %r (nvblox|replay; the analytic '
                     'bench source is constructed by the bench lane, not from host config)' % kind)


def build_gated_motion_port(inner, *, config, tool, urdf_path, get_phase, get_held,
                            get_carton_dims, steady_clock, node=None, logger=print):
    """Install the gate around ``inner`` when ``config['perception_collision_gate'].enabled``.

    Returns the raw ``inner`` port unchanged when the key is absent/disabled (byte-identical).
    ``tool`` is the :class:`cad_tool.CadTool` (its ``envelope_pad_*`` and ``tcp`` define the
    swept gripper box); ``get_phase``/``get_held``/``get_carton_dims`` are read live from the
    controller at each dispatch (see :mod:`ros_host` wiring).
    """
    block = config.get('perception_collision_gate')
    if not (isinstance(block, dict) and block.get('enabled')):
        return inner
    robot = parse_urdf(str(urdf_path))
    gate = _gate_from_config(block, logger=logger)
    source = _source_from_config(block, node=node, logger=logger)
    if logger is not None:
        logger('PERCEPTION_GATE_INSTALLED margin=%.3f mode=%s unknown=%s source=%s phases=%s'
               % (gate.margin_m, gate.mode, gate.unknown_action,
                  str(block.get('source', 'nvblox')), sorted(gate.phases)), flush=True)
    return GatedMotionPort(
        inner, gate=gate, esdf_source=source, robot=robot, tcp_link=getattr(tool, 'tcp',
        'gripper_pad_tcp'), envelope_pad_min=tool.envelope_pad_min,
        envelope_pad_max=tool.envelope_pad_max, get_phase=get_phase, get_held=get_held,
        get_carton_dims=get_carton_dims, steady_clock=steady_clock,
        grasp_axis=int(block.get('carton_grasp_axis', 2)),
        carton_face_offset=float(block.get('carton_face_offset_m', 0.0)), logger=logger)
