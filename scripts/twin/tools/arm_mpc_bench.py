"""Bench-first cuRobo MPC reactive follow-and-avoid for the MISSION arm (SIM/EVAL ONLY).

This is the store-twin analogue of ``ur10/mpc_curobo/mpc_prototype.py`` (lane
``plan-mpc-curobo-v1``): drive the mission's UR10 with cuRobo V2 (Apache-2.0)
Model-Predictive-Control against the composed store colliders plus a moving-person
proxy, and MEASURE solve rate, clearance and gate behaviour -- mirroring the
``arm_carry_bench`` pattern (seconds per trial, not a 15-minute mission run).

WHAT THIS IS, STATED ONCE AND HONESTLY
    * MPC's collision avoidance is a **soft optimiser cost**, NOT a certified safety
      function. The plan-mpc-curobo-v1 evidence showed a fast person transiently
      intruded by ~3 cm. The single motion authority remains the merged human-safety
      gate (RUN/SLOW/STOP/BACKSTOP, fail-toward-stop) and the measured joint-window
      guard -- both reused VERBATIM here from the prototype's ``ros_bridge`` (see
      ``load_gate``), never re-implemented.
    * EVALUATION-GRADE and simulation-only by construction. The owner-held cuRobo
      eval-vs-ship decision (``plan-mpc-curobo-v1/decision.md``) is not pre-empted;
      this lane only produces the measured evidence that decision needs.
    * The store world here is a REPRESENTATIVE restock-aisle model (ground + the
      target shelf + the opposite aisle shelf + a moving-person proxy), exactly as
      ``cell_world`` is a representative shelf for the cell. Feeding the real composed
      USD / nvblox ESDF to the optimiser is a separate (option-3-style) follow-up.

LAYERING
    * This module is PURE (stdlib only) so the world model, the scenario paths, the
      clearance maths and the metrics are CPU unit-testable with no torch / cuRobo /
      Isaac. The gate + window guard are injected (``load_gate``) and are themselves
      pure stdlib, so the gate-behaviour trial is exercised offline too.
    * The only GPU/cuRobo-dependent part is the thin solver loop in
      ``scripts/twin/tools/arm_mpc_run.py``, driven by ``scripts/run_arm_mpc_bench.py``
      inside the reused ``fm-plan-mpc-curobo-v1`` container. This module is imported by
      both so the world, scenarios and metrics are identical offline and on-GPU.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Mission arm joint order (``ur_`` prefixed), matching
# scripts/twin/restock_manipulation/cad_tool.py ARM_JOINTS and the cuRobo cspace in
# shelfbot_curobo.yml. The gate/window guard are joint-name agnostic (they key off the
# dict), so we feed them these names explicitly.
ARM_JOINTS: Tuple[str, ...] = tuple(
    'ur_' + n for n in (
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'))

PERSON = 'person'  # name of the moving-person proxy obstacle (addressable at runtime)


# --------------------------------------------------------------------------- #
# Reuse the prototype's merged human-safety gate + measured joint-window guard. #
# --------------------------------------------------------------------------- #
def load_gate(mpc_curobo_dir: Optional[str] = None):
    """Import the REAL ``GatedStreamController`` + ``JointWindows`` from the
    plan-mpc-curobo-v1 prototype's ``ros_bridge`` (both pure stdlib). Reused, never
    copied -- the brief's "keep the merged human-safety gate as the single motion
    authority and the measured joint-window guard (both already in the prototype's
    ros_bridge)".

    ``mpc_curobo_dir`` points at the prototype's ``mpc_curobo`` package dir; defaults
    to ``$MPC_CUROBO_DIR`` then ``~/ur10/mpc_curobo`` (host) then ``/mpc`` (the mount
    path the container runner uses). Returns a module-like namespace with
    ``GatedStreamController``, ``JointWindows``, ``wrap_near`` and the gate-action
    constants ``RUN/SLOW/STOP/BACKSTOP``.
    """
    candidates = [mpc_curobo_dir, os.environ.get('MPC_CUROBO_DIR'),
                  os.path.expanduser('~/ur10/mpc_curobo'), '/mpc']
    chosen = next((c for c in candidates if c and os.path.isdir(c)), None)
    if chosen is None:
        raise RuntimeError(
            'cannot locate the plan-mpc-curobo-v1 mpc_curobo dir (set MPC_CUROBO_DIR '
            'or mount it at /mpc); the merged gate + window guard live there')
    if chosen not in sys.path:
        sys.path.insert(0, chosen)
    from ros_bridge.stream_gate import (  # noqa: E402
        GatedStreamController, FORWARD, HOLD, RUN, SLOW, STOP, BACKSTOP)
    from robot.joint_windows import JointWindows, wrap_near  # noqa: E402

    class _Gate:
        pass
    g = _Gate()
    g.GatedStreamController = GatedStreamController
    g.JointWindows = JointWindows
    g.wrap_near = wrap_near
    g.FORWARD, g.HOLD = FORWARD, HOLD
    g.RUN, g.SLOW, g.STOP, g.BACKSTOP = RUN, SLOW, STOP, BACKSTOP
    g.dir = chosen
    return g


def mission_windows(arm_limits: Sequence[Tuple[float, float]], gate):
    """Build a ``JointWindows`` (keyed by the mission ``ur_`` joint names) from the
    URDF arm limits the mission itself uses (``cad_tool.inspect_tool(urdf).arm_limits``,
    one ``(lower, upper)`` per ARM_JOINTS). The window guard is thus the SAME measured
    envelope the trajectory host enforces, just re-expressed under the mission names."""
    lims = [tuple(float(v) for v in lh) for lh in arm_limits]
    if len(lims) != len(ARM_JOINTS):
        raise ValueError(f'need {len(ARM_JOINTS)} arm limits, got {len(lims)}')
    for lo, hi in lims:
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            raise ValueError(f'invalid arm limit [{lo}, {hi}]')
    return gate.JointWindows(windows={j: (lo, hi) for j, (lo, hi) in zip(ARM_JOINTS, lims)})


# --------------------------------------------------------------------------- #
# Representative store restock-aisle world model (cuRobo scene_model dicts).    #
# --------------------------------------------------------------------------- #
# Geometry in the robot base (world) frame, metres, wxyz quaternion -- the same
# layout cuRobo consumes as scene_model["cuboid"]. The mission arm is braked at the
# aisle centre and reaches +x to the target shelf; a second shelf at -x forms the
# aisle; the person proxy walks the aisle along y, crossing the reach envelope.

@dataclass(frozen=True)
class StoreAisle:
    """Representative restock-aisle dimensions (metres). Defaults model the G3-G4
    aisle the composer spawns into: ~1.6 m aisle, ~2.0 m shelves, reach to the
    +x shelf at ~0.85 m. Every field is overridable so the bench can be re-anchored
    to measured geometry without code change."""
    aisle_width: float = 1.6           # centre-to-opposite-shelf-face * 2 (m)
    shelf_height: float = 2.0          # shelf fixture height (m)
    shelf_depth: float = 0.5           # shelf front-to-back depth (m)
    target_shelf_x: float = 0.85       # front face of the shelf being restocked (m, +x)
    plate_z: float = 0.9               # restock slot height (m) -- G3_back_s2 band
    floor_top_z: float = -0.05         # floor top surface (m); base works above it

    def validate(self) -> 'StoreAisle':
        if not (self.aisle_width > 0 and self.shelf_height > 0 and self.shelf_depth > 0
                and self.target_shelf_x > 0 and self.plate_z > 0):
            raise ValueError('invalid store-aisle geometry')
        return self


def store_world(aisle: StoreAisle = StoreAisle()) -> Dict:
    """Ground + target shelf (restock plate + back wall) + the opposite aisle shelf.
    Mirrors ``cell_world.static_world`` but at supermarket aisle scale."""
    aisle.validate()
    half = aisle.aisle_width / 2.0
    back_x = aisle.target_shelf_x + aisle.shelf_depth  # back wall of the target shelf
    return {
        'cuboid': {
            # floor slab: top at floor_top_z, kept below the base so it is not the
            # binding clearance during normal reach.
            'ground': {'dims': [8.0, 8.0, 0.10],
                       'pose': [0.0, 0.0, aisle.floor_top_z - 0.05, 1, 0, 0, 0]},
            # the restock slot plate the handling unit lands on.
            'target_plate': {'dims': [aisle.shelf_depth, 1.4, 0.04],
                             'pose': [aisle.target_shelf_x + aisle.shelf_depth / 2.0,
                                      0.0, aisle.plate_z, 1, 0, 0, 0]},
            # the target shelf's back wall (far side of the pick/place region).
            'target_back': {'dims': [0.05, 1.4, aisle.shelf_height],
                            'pose': [back_x, 0.0, aisle.shelf_height / 2.0, 1, 0, 0, 0]},
            # the opposite aisle shelf face (behind the robot, -x) -- the other aisle wall.
            'opposite_shelf': {'dims': [0.05, 1.4, aisle.shelf_height],
                               'pose': [-half, 0.0, aisle.shelf_height / 2.0, 1, 0, 0, 0]},
        },
    }


# Person stand-in: a 0.5 x 0.5 m footprint, 1.8 m tall box -- a CUBOID (not a capsule)
# because cuRobo's SceneCollision only exposes named, runtime-updatable obstacles as
# cuboids (get_obstacle_names / update_obstacle_pose); a capsule is baked and immovable.
# This matches the Detection(position, radius, source) contract shared with
# sim-restock-humans-v1 / sim-restock-dynobs-v1: a plan-view radius ~0.35 m (half the
# box diagonal) and the moving centroid are exactly what that layer emits.
PERSON_DIMS = [0.5, 0.5, 1.8]
PERSON_Z = 0.9
PERSON_PLANVIEW_RADIUS = round((0.5 ** 2 + 0.5 ** 2) ** 0.5 / 2.0, 4)  # ~0.3536 m


def person_pose(px: float, py: float) -> List[float]:
    return [px, py, PERSON_Z, 1.0, 0.0, 0.0, 0.0]


def store_world_with_person(aisle: StoreAisle = StoreAisle(),
                            person_start: Tuple[float, float] = (0.35, -1.2)) -> Dict:
    """Store world plus a moving-person proxy box, parked off to one side at start;
    the runner walks it down the aisle with ``update_obstacle_pose`` (a named cuboid)."""
    w = store_world(aisle)
    px, py = person_start
    w['cuboid'][PERSON] = {'dims': list(PERSON_DIMS), 'pose': person_pose(px, py)}
    return w


def nominal_reach(aisle: StoreAisle = StoreAisle()) -> Tuple[float, float, float]:
    """The TCP reach centre the absolute ``store_world`` is designed around -- the
    front of the restock slot. The GPU run FKs the in-window anchor to get the arm's
    ACTUAL reach centre and shifts the world by (actual - nominal) so the
    representative shelf sits just in front of wherever the mission arm reaches,
    regardless of the exact URDF mounting."""
    return (aisle.target_shelf_x, 0.0, aisle.plate_z)


def shift_world(world: Dict, delta: Sequence[float]) -> Dict:
    """Return a copy of ``world`` with every cuboid translated by ``delta`` (dx, dy, dz).
    Pure; used to rebase the representative obstacles onto the arm's measured FK reach."""
    dx, dy, dz = (float(v) for v in delta)
    out = {'cuboid': {}}
    for name, c in world.get('cuboid', {}).items():
        p = list(c['pose'])
        p[0] += dx
        p[1] += dy
        p[2] += dz
        out['cuboid'][name] = {'dims': list(c['dims']), 'pose': p}
    return out


# --------------------------------------------------------------------------- #
# Scenarios: moving Cartesian target + optional moving person.                  #
# --------------------------------------------------------------------------- #
SCENARIOS = ('follow', 'person', 'blocked', 'unreachable')


def make_target_fn(scenario: str, center: Sequence[float]) -> Callable[[float], Tuple[float, float, float]]:
    """target(t) -> xyz. ``center`` is the reachable restock-anchor TCP position
    (FK of the in-window anchor config at GPU time)."""
    cx, cy, cz = center
    if scenario == 'unreachable':
        # far outside the UR10 reach + windows (~2.4 m out across the aisle).
        return lambda t: (cx + 1.6, cy, cz + 0.6)
    if scenario == 'blocked':
        # a fixed reachable restock target offset in +y, on the far side of the person.
        return lambda t: (cx, cy + 0.30, cz)
    # follow / person: sweep the restock target side to side along y (the shelf run),
    # small amplitude so every pose stays reachable and in-window.
    def target(t):
        y = cy + 0.25 * math.sin(2 * math.pi * t / 8.0)   # 8 s period, +-0.25 m
        z = cz + 0.08 * math.sin(2 * math.pi * t / 5.0)
        return (cx, y, z)
    return target


def make_person_fn(scenario: str, center: Sequence[float]
                   ) -> Optional[Callable[[float], Tuple[float, float, float]]]:
    """person(t) -> (x, y, z) origin, or None if no person moves. The proxy crosses
    the space between the robot base and the restock shelf."""
    cx, cy, cz = center
    if scenario == 'person':
        # walk down the aisle in front of the restock region, -1.2 -> +1.2 m in y over 12 s.
        def person(t):
            y = -1.2 + 2.4 * min(max(t / 12.0, 0.0), 1.0)
            return (cx - 0.30, y, 0.0)
        return person
    if scenario == 'blocked':
        # park the person directly between the base and the restock target, and hold.
        return lambda t: (cx - 0.15, cy + 0.15, 0.0)
    return None


def scenario_needs_person(scenario: str) -> bool:
    return scenario in ('person', 'blocked')


# --------------------------------------------------------------------------- #
# Analytic clearance (optimiser-independent collision-margin metric).           #
# --------------------------------------------------------------------------- #
def _point_box_distance(p, center, dims) -> float:
    dx = max(abs(p[0] - center[0]) - dims[0] / 2.0, 0.0)
    dy = max(abs(p[1] - center[1]) - dims[1] / 2.0, 0.0)
    dz = max(abs(p[2] - center[2]) - dims[2] / 2.0, 0.0)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def min_sphere_clearance(sphere_xyzr: Sequence[Sequence[float]], world: Dict) -> float:
    """Minimum signed clearance (m) between any robot collision sphere and any obstacle
    cuboid. ``sphere_xyzr``: iterable of (x, y, z, r). Negative == penetration. Computed
    from FK spheres so the reported margin is independent of the optimiser's own cost."""
    worst = float('inf')
    for (x, y, z, r) in sphere_xyzr:
        p = (x, y, z)
        for c in world.get('cuboid', {}).values():
            cx, cy, cz = c['pose'][0], c['pose'][1], c['pose'][2]
            worst = min(worst, _point_box_distance(p, (cx, cy, cz), c['dims']) - r)
    return worst


def person_only_world(world: Dict) -> Dict:
    """A view of ``world`` containing only the person proxy, for the person-clearance
    metric. Empty cuboid dict if there is no person."""
    if PERSON in world.get('cuboid', {}):
        return {'cuboid': {PERSON: world['cuboid'][PERSON]}}
    return {'cuboid': {}}


# --------------------------------------------------------------------------- #
# Gate-behaviour trial (pure; drives the REAL injected gate controller).        #
# --------------------------------------------------------------------------- #
@dataclass
class GateTrialResult:
    """Outcome of replaying a gate-state timeline against a stream of MPC commands
    through the real ``GatedStreamController``. Proves the brief's hard rules by
    construction: nothing moves unless RUN; a command arriving during STOP is dropped;
    an out-of-window command is refused; resume heads for the freshest command."""
    travel_while_run: float = 0.0        # total |step| summed over RUN ticks (rad)
    travel_while_not_run: float = 0.0    # total |step| over non-RUN ticks (must be 0)
    forwarded: int = 0                   # ticks the shim FORWARDED
    held: int = 0                        # ticks the shim HELD
    dropped_during_stop: int = 0         # commands presented during STOP/BACKSTOP
    window_refusals: int = 0             # out-of-window commands refused (window_ok False)
    resumed_to_fresh: bool = False       # after a STOP episode, motion resumed toward the freshest cmd
    final_position: List[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """The hard rule: zero travel while not RUN."""
        return self.travel_while_not_run <= 1e-9


def run_gate_trial(gate, windows, *, timeline, commands, start_position,
                   max_step_per_tick: float = 0.01) -> GateTrialResult:
    """Replay a (gate_action, speed_fraction) ``timeline`` against ``commands`` (one
    MPC-proposed joint vector per tick, ``None`` for a starved tick) through the real
    ``GatedStreamController``. ``start_position`` seeds the hold. Both lists are the
    same length (one entry per control tick).

    This is the store-arm analogue of the prototype's mock gating proof, run as pure
    logic so the hard rules are verified offline and replay byte-identically."""
    if len(timeline) != len(commands):
        raise ValueError('timeline and commands must be the same length')
    ctrl = gate.GatedStreamController(
        windows=windows, joint_names=list(ARM_JOINTS), max_step_per_tick=max_step_per_tick)
    ctrl.seed(start_position)
    res = GateTrialResult()
    prev = [float(v) for v in start_position]
    saw_stop = False
    for (action, frac), cmd in zip(timeline, commands):
        sa = ctrl.step(action, frac, cmd, measured_position=prev)
        step_mag = sum(abs(a - b) for a, b in zip(sa.target_position, prev))
        if action == gate.RUN:
            res.travel_while_run += step_mag
        else:
            res.travel_while_not_run += step_mag
        if action in (gate.STOP, gate.BACKSTOP):
            if cmd is not None:
                res.dropped_during_stop += 1
            saw_stop = True
        if sa.kind == gate.FORWARD:
            res.forwarded += 1
            if saw_stop and action == gate.RUN and step_mag > 1e-9:
                res.resumed_to_fresh = True
        else:
            res.held += 1
        if not sa.window_ok:
            res.window_refusals += 1
        prev = list(sa.target_position)
    res.final_position = list(prev)
    return res


# --------------------------------------------------------------------------- #
# MPC run metrics (what the GPU loop records per scenario).                      #
# --------------------------------------------------------------------------- #
@dataclass
class MpcScenarioMetrics:
    """Per-scenario MPC metrics (brief: solve rate, clearance, gate behaviour)."""
    scenario: str
    steps: int = 0
    control_rate_hz: float = 0.0
    solve_ms_mean: float = 0.0
    solve_ms_p95: float = 0.0
    solve_ms_max: float = 0.0
    tracking_steady_mean_m: float = 0.0
    tracking_steady_max_m: float = 0.0
    min_clearance_m: Optional[float] = None
    min_person_clearance_m: Optional[float] = None
    joint_window_violations: int = 0
    max_window_excess_rad: float = 0.0
    est_process_mib: Optional[float] = None
    gate: Optional[dict] = None          # the gate-trial summary for this scenario
    note: str = ''

    def as_dict(self) -> dict:
        return {
            'scenario': self.scenario, 'steps': self.steps,
            'control_rate_hz': round(self.control_rate_hz, 2),
            'solve_ms': {'mean': round(self.solve_ms_mean, 3),
                         'p95': round(self.solve_ms_p95, 3),
                         'max': round(self.solve_ms_max, 3)},
            'tracking_steady_m': {'mean': round(self.tracking_steady_mean_m, 4),
                                  'max': round(self.tracking_steady_max_m, 4)},
            'min_clearance_m': (round(self.min_clearance_m, 4)
                                if self.min_clearance_m is not None else None),
            'min_person_clearance_m': (round(self.min_person_clearance_m, 4)
                                       if self.min_person_clearance_m is not None else None),
            'joint_window_violations': self.joint_window_violations,
            'max_window_excess_rad': round(self.max_window_excess_rad, 5),
            'est_process_mib': (round(self.est_process_mib, 1)
                                if self.est_process_mib is not None else None),
            'gate': self.gate,
            'note': self.note,
        }


def behaviour_note(scenario: str) -> str:
    if scenario == 'follow':
        return ('follows the sweeping restock target with the aisle shelves present; '
                'report control rate + tracking error.')
    if scenario == 'person':
        return ('person proxy walks down the aisle; min_person_clearance should stay >0 '
                '(MPC routes around) while tracking error rises as the person occludes. '
                'SOFT cost: a fast person can be transiently intruded -- not a safety function.')
    if scenario == 'blocked':
        return ('person parked on the straight-line path; min_person_clearance should stay '
                '>0 (MPC does not drive through) and tracking stays elevated.')
    if scenario == 'unreachable':
        return ('target ~1.6 m across the aisle (outside reach + windows); tracking error '
                'stays large, joint_window_violations == 0, the arm stays bounded, no thrash.')
    return ''


def metrics_table(results: Sequence[MpcScenarioMetrics]) -> str:
    """Compact text table over scenarios (report evidence)."""
    rows = [('scenario', 'rate_hz', 'solve_ms', 'track_mm', 'clear_mm', 'person_mm',
             'win_viol', 'proc_mib')]
    for m in results:
        rows.append((
            m.scenario, f'{m.control_rate_hz:.1f}', f'{m.solve_ms_mean:.1f}',
            f'{m.tracking_steady_mean_m * 1000:.0f}',
            (f'{m.min_clearance_m * 1000:.0f}' if m.min_clearance_m is not None else '-'),
            (f'{m.min_person_clearance_m * 1000:.0f}'
             if m.min_person_clearance_m is not None else '-'),
            str(m.joint_window_violations),
            (f'{m.est_process_mib:.0f}' if m.est_process_mib is not None else '-')))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return '\n'.join('  '.join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in rows)
