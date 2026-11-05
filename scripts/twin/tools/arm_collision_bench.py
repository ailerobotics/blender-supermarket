"""Arm collision-avoidance EVALUATION bench (sim-arm-collision-bench-v1).

Measurable, not asserted: this is the harness the captain asked for -- it answers
"how often does the arm avoid an unexpected obstacle, with what clearance, at what
cost in time, across the mission's arm legs" for a declarative set of planted-obstacle
scenarios.

Scope and honesty
------------------
This module is the **pure-CPU core** (numpy only; the URDF FK via ``twin.urdf_probe``
and the carton-attitude FK via ``twin.restock_manipulation.carton_level`` are both pure).
No Isaac, no ROS, no GPU, no cuMotion. It is the twin of the GPU-side runner
``run_arm_collision_bench.py`` (which boots one bounded mission segment with the same
obstacles spawned and reads the live contact reports / ESDF); both emit the SAME
per-scenario metric schema so a CPU prediction and a GPU measurement are directly
comparable.

What it measures, per scenario (CPU mode)
-----------------------------------------
Given an arm leg's **nominal swept trajectory** (the arm's collision spheres from the
curobo config, forward-kinematic at each waypoint, plus the welded carton on carry legs)
and a set of **planted obstacles** (static, and moving along a scripted path):

  * ``collision_free``       -- does the NOMINAL swept path stay clear of every obstacle
                                (min true clearance >= 0)?  (the geometric ground truth)
  * ``min_true_clearance_m`` -- analytic min signed gap, arm swept volume vs obstacles
  * ``min_perceived_clearance_m`` -- the same, read through a MOCK ESDF (obstacles
                                rasterised into a voxel grid at the mission voxel size,
                                queried by trilinear interpolation) -- the quantity a
                                perceived-clearance gate actually sees
  * ``esdf_error_m``         -- perceived - true (voxelisation + finite-grid error)
  * ``n_replans`` / ``extra_cycle_time_s`` -- what the GATE did (re-plan / stop dwell)
  * ``tilt_peak_deg``        -- carried-carton attitude peak over the leg (the standing
                                "box parallel to the ground" 2.0 deg constraint), carry
                                legs only; 0 on empty-arm legs
  * ``gate_behavior`` vs ``expected_behavior`` -- did the gate's VERDICT match?

The GATE is pluggable (:class:`Gate`).  Two are shipped:

  * :func:`baseline_gate` -- the CURRENT stack: no perceived-ESDF gate on most arm legs
    (collect drives ground-truth targets; contact/insert legs are collision-OFF
    straight lines) -- it NEVER avoids/refuses.  Run the bench with this to get the
    **BEFORE numbers** (the measurable gap: the nominal path hits the planted obstacle
    and nothing stops it).
  * :func:`reference_gate` -- a transparent collision-aware reference instrument modelling
    what the sibling lanes (sim-rgbd-arm-collision-aware-v1 /
    sim-arm-reactive-replan-v1) are building toward: refuse-before-motion when the
    perceived path is blocked at the goal, re-plan a detour when an obstacle is beside a
    free-space leg, slow/stop/resume for a person/hand, preempt+re-plan for a sliding
    obstacle.  This is the bench's reference, NOT the lanes' code.

When a sibling lane ships its real gate, plug it in with ``--gate module:function``
(any callable with the :class:`Gate` signature) and the same scenarios + schema measure
it identically -- that is the AFTER column.

Pass/fail (``--thresholds``): no contact ever; min true clearance >= margin on the
EXECUTED path (the gate's action is credited -- a refuse/stop/replan that prevents the
strike counts as collision-free); cycle-time overhead under a configurable bound; and
the gate verdict matched the scenario's expected behaviour.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from twin.restock_manipulation.cad_tool import ARM_JOINTS

# --- the mission's MEASURED nvblox arm-planning ESDF grid (scout sim-perception-architecture-s1,
# world_obstacles.py:8-10): shape [409,217,257] voxels, origin [-2.00,-0.08,-0.48] m, voxel 0.01 m.
# The bench defaults to the SAME voxel size so the perceived-clearance voxelisation error it reports
# matches the real planner's; the grid BOUNDS default to the scenario workspace (auto-fit) to stay cheap.
MISSION_VOXEL_M = 0.01
MISSION_ESDF_ORIGIN = (-2.00, -0.08, -0.48)
MISSION_ESDF_DIMS = (409, 217, 257)

# Behaviour vocabulary a scenario may declare and a gate may return.
BEHAVIORS = ('clear', 'avoid', 'preempt_replan', 'slow_stop_resume', 'refuse', 'none')

# Default per-leg nominal cycle time (s) -- the denominator for the cycle-time-overhead metric.
# Representative mission arm-leg durations; a scenario overrides with ``cycle_time_nominal_s``.
DEFAULT_LEG_CYCLE_S = {
    'collect_pick': 12.0, 'carry': 10.0, 'preplace_approach': 9.0,
    'set_down': 8.0, 'withdraw': 6.0, 'transfer': 12.0,
}

# Carried handling unit (the scenario single source of truth handling_unit.dimensions_m).
CARTON_DIMS_M = (0.45, 0.30, 0.18)
# Carton centre relative to ur_tool0 at the welded grasp (wall=3 side grip, place_clearance
# WALL3_GRIP_TCP_IN_CARTON inverted onto the tool): the carton hangs off the tool mouth-ward.
CARTON_CENTER_IN_TOOL = (0.0, 0.0, 0.14)


# --------------------------------------------------------------------------- scenario model

@dataclass(frozen=True)
class Motion:
    """A scripted obstacle path.  ``kind='static'`` never moves; ``kind='linear'`` translates at
    ``velocity`` m/s from ``start_t`` s.  ``obstacle_class`` drives the gate's reaction choice
    (a person/hand -> slow/stop; a carton/box -> preempt+replan)."""
    kind: str = 'static'
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_t: float = 0.0
    obstacle_class: str = 'static'  # static | person | hand | carton | box | tool | cage

    def position_at(self, t: float) -> np.ndarray:
        if self.kind == 'static':
            return np.zeros(3)
        dt = max(0.0, float(t) - self.start_t)
        return np.asarray(self.velocity, float) * dt


@dataclass(frozen=True)
class Obstacle:
    """A planted obstacle in the arm base frame (``ur_base_link``).  A ``box`` is an axis-aligned
    (in its own rest frame) centre+size; a ``sphere`` is centre+``radius``.  ``motion`` scripts it.

    ``observed`` models the scout's central ESDF finding (sim-perception-architecture-s1 §4):
    *unobserved == free*.  An ``observed=False`` obstacle (a thin cage bar behind the carton, a body
    outside the single-camera frustum, the cart outside the grid) is present in the TRUE clearance but
    ABSENT from the mock ESDF -- so a perceived-clearance gate is blind to it.  This is how the bench
    measures the current stack's blindness honestly (perceived says clear, true says collision).
    ``at_goal`` marks an obstacle sitting ON the leg's reach target (so the reference gate splits
    refuse from avoid)."""
    name: str
    kind: str  # 'box' | 'sphere'
    center: Tuple[float, float, float]
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # box full extents
    radius: float = 0.0                                  # sphere
    motion: Motion = field(default_factory=Motion)
    observed: bool = True
    at_goal: bool = False

    def center_at(self, t: float) -> np.ndarray:
        return np.asarray(self.center, float) + self.motion.position_at(t)

    def signed_distance(self, points: np.ndarray, t: float) -> np.ndarray:
        """Signed distance (m, negative inside) from each point (N,3) to this obstacle at time ``t``."""
        c = self.center_at(t)
        if self.kind == 'sphere':
            return np.linalg.norm(np.asarray(points, float) - c, axis=1) - float(self.radius)
        half = np.asarray(self.size, float) / 2.0
        q = np.abs(np.asarray(points, float) - c) - half
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)

    def aabb_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        c = self.center_at(t)
        if self.kind == 'sphere':
            r = float(self.radius)
            return c - r, c + r
        half = np.asarray(self.size, float) / 2.0
        return c - half, c + half


@dataclass(frozen=True)
class Thresholds:
    """Pass/fail bounds.  ``no_contact``: the executed path must never collide.
    ``min_true_clearance_m``: executed min true clearance must be >= this (defaults to the
    scenario margin).  ``cycle_time_overhead_frac``: extra cycle time / nominal must be <= this.
    ``require_verdict_match``: the gate verdict must equal the scenario's expected behaviour."""
    no_contact: bool = True
    min_true_clearance_m: Optional[float] = None
    cycle_time_overhead_frac: float = 0.5
    require_verdict_match: bool = True

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> 'Thresholds':
        d = dict(d or {})
        return cls(bool(d.get('no_contact', True)),
                   d.get('min_true_clearance_m'),
                   float(d.get('cycle_time_overhead_frac', 0.5)),
                   bool(d.get('require_verdict_match', True)))


@dataclass(frozen=True)
class Scenario:
    id: str
    arm_leg: str
    expected_behavior: str
    obstacles: Tuple[Obstacle, ...]
    description: str = ''
    margin_m: float = 0.030
    voxel_m: float = MISSION_VOXEL_M
    trajectory: Optional[Tuple[Tuple[float, ...], ...]] = None  # explicit ARM-order waypoints
    cycle_time_nominal_s: Optional[float] = None
    duration_s: float = 6.0        # scripted-motion horizon
    time_step_s: float = 0.25      # motion sampling

    def nominal_cycle_s(self) -> float:
        if self.cycle_time_nominal_s is not None:
            return float(self.cycle_time_nominal_s)
        return DEFAULT_LEG_CYCLE_S.get(self.arm_leg, 10.0)


def _obstacle_from_dict(d: dict) -> Obstacle:
    m = d.get('motion') or {}
    motion = Motion(kind=str(m.get('kind', 'static')),
                    velocity=tuple(float(v) for v in m.get('velocity', (0.0, 0.0, 0.0))),
                    start_t=float(m.get('start_t', 0.0)),
                    obstacle_class=str(m.get('class', m.get('obstacle_class', 'static'))))
    return Obstacle(name=str(d['name']), kind=str(d['kind']),
                    center=tuple(float(v) for v in d['center']),
                    size=tuple(float(v) for v in d.get('size', (0.0, 0.0, 0.0))),
                    radius=float(d.get('radius', 0.0)), motion=motion,
                    observed=bool(d.get('observed', True)), at_goal=bool(d.get('at_goal', False)))


def scenario_from_dict(d: dict) -> Scenario:
    if d.get('expected_behavior') not in BEHAVIORS:
        raise ValueError('scenario %r expected_behavior must be one of %s' % (d.get('id'), BEHAVIORS))
    traj = d.get('trajectory')
    return Scenario(
        id=str(d['id']), arm_leg=str(d['arm_leg']),
        expected_behavior=str(d['expected_behavior']),
        obstacles=tuple(_obstacle_from_dict(o) for o in d.get('obstacles', [])),
        description=str(d.get('description', '')),
        margin_m=float(d.get('margin_m', 0.030)),
        voxel_m=float(d.get('voxel_m', MISSION_VOXEL_M)),
        trajectory=tuple(tuple(float(v) for v in w) for w in traj) if traj else None,
        cycle_time_nominal_s=d.get('cycle_time_nominal_s'),
        duration_s=float(d.get('duration_s', 6.0)),
        time_step_s=float(d.get('time_step_s', 0.25)))


def load_scenario(path) -> Scenario:
    return scenario_from_dict(json.loads(Path(path).read_text()))


def load_scenarios(directory) -> List[Scenario]:
    paths = sorted(Path(directory).glob('*.json'))
    return [load_scenario(p) for p in paths]


def with_full_coverage(scenario: Scenario) -> Scenario:
    """A copy of ``scenario`` with every obstacle marked observed -- models the perception coverage the
    sibling features add (mast at collect/carry, cart as a world obstacle).  Use for the AFTER-coverage
    column when the sibling gate is not yet importable: it answers "once the stack SEES these obstacles,
    does the collision-aware gate make the right call?" without fabricating a planner."""
    return replace(scenario, obstacles=tuple(replace(o, observed=True) for o in scenario.obstacles))


# --------------------------------------------------------------------------- arm swept-volume model

# The mission robot URDF + the curobo collision-sphere model.  Package ``shelfbot`` =
# ~/workspaces/shelfbot_newgripper (AGENTS.md: declares the ROS name AND holds the gripper meshes).
DEFAULT_URDF_CANDIDATES = (
    Path.home() / 'datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf',
    Path.home() / 'workspaces/shelfbot_newgripper/description/shelfbot_gripper_isaac.urdf',
)
DEFAULT_CUROBO_YML = Path.home() / 'workspaces/shelfbot_newgripper/description/shelfbot_gripper_curobo.yml'


@dataclass(frozen=True)
class ArmModel:
    """The arm's collision-sphere swept-volume model: per-link curobo spheres forward-kinematic at a
    joint config, plus the welded carried carton for carry legs.  Pure/CPU (URDF FK + numpy)."""
    robot: object
    spheres: Dict[str, np.ndarray]  # link -> (M,4) [cx,cy,cz,r] in the link frame
    root: str = 'ur_base_link'

    @classmethod
    def load(cls, urdf_path=None, curobo_yml=None, root='ur_base_link') -> 'ArmModel':
        import yaml
        from twin.urdf_probe import parse_urdf
        if urdf_path is None:
            urdf_path = next((p for p in DEFAULT_URDF_CANDIDATES if Path(p).is_file()), None)
            if urdf_path is None:
                raise FileNotFoundError('no mission URDF found; pass --urdf')
        cfg = yaml.safe_load(Path(curobo_yml or DEFAULT_CUROBO_YML).read_text())
        cs = cfg['robot_cfg']['kinematics']['collision_spheres']
        if isinstance(cs, str):
            raise ValueError('curobo collision_spheres is a file ref; pass the inlined yml')
        spheres = {}
        for link, lst in cs.items():
            arr = np.array([[*s['center'], float(s['radius'])] for s in lst], float)
            if arr.size:
                spheres[link] = arr
        robot = parse_urdf(str(urdf_path))
        return cls(robot=robot, spheres=spheres, root=root)

    def _fk(self, config) -> Dict[str, np.ndarray]:
        from twin.urdf_probe import link_world_transforms
        cfgmap = ({k: float(v) for k, v in config.items()} if isinstance(config, dict)
                  else {j: float(v) for j, v in zip(ARM_JOINTS, config)})
        return link_world_transforms(self.robot, cfgmap, root=self.root)

    def spheres_at(self, config, carry: bool = False) -> np.ndarray:
        """(N,4) arm (and, if ``carry``, carton) collision spheres in the base frame at ``config``."""
        wl = self._fk(config)
        out = []
        for link, arr in self.spheres.items():
            T = wl.get(link)
            if T is None:
                continue
            c = arr[:, :3] @ np.asarray(T, float)[:3, :3].T + np.asarray(T, float)[:3, 3]
            out.append(np.column_stack([c, arr[:, 3]]))
        world = np.vstack(out) if out else np.zeros((0, 4))
        if carry:
            world = np.vstack([world, self._carton_spheres(wl)])
        return world

    def _carton_spheres(self, wl, step=0.06, radius=0.035) -> np.ndarray:
        """Fill the welded carton box with spheres, posed by the ur_tool0 FK (so the carton rides the
        arm through the leg -- used for both swept clearance and as the attitude witness)."""
        T = np.asarray(wl.get('ur_tool0', np.eye(4)), float)
        half = np.asarray(CARTON_DIMS_M, float) / 2.0
        grids = [np.linspace(-h, h, max(2, int(round(2 * h / step)) + 1)) for h in half]
        X, Y, Z = np.meshgrid(*grids, indexing='ij')
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]) + np.asarray(CARTON_CENTER_IN_TOOL, float)
        c = pts @ T[:3, :3].T + T[:3, 3]
        return np.column_stack([c, np.full(len(c), radius)])

    def tool_rotation(self, config) -> np.ndarray:
        return np.asarray(self._fk(config)['ur_tool0'], float)[:3, :3]


def build_trajectory(scenario: Scenario) -> List[List[float]]:
    """The arm leg's nominal swept trajectory (list of ARM-order 6-vectors).  Explicit scenario
    waypoints win; otherwise the documented representative leg from :func:`nominal_leg`."""
    if scenario.trajectory:
        return [list(w) for w in scenario.trajectory]
    return nominal_leg(scenario.arm_leg)


# Representative nominal arm legs (ARM order [pan,lift,elbow,wrist_1,wrist_2,wrist_3]).  The COLLECT
# anchors (grasp/lift/preplace) are REAL recorded mission configs (armbench-preplace-r279.json);
# the intermediate standoffs/descents are documented synthetic waypoints around them so the bench
# geometry is reproducible.  A scenario inlines exact recorded waypoints to supersede these.
_GRASP = [1.269, -1.342, 2.184, -0.840, 1.269, 0.0]
_LIFT = [1.262, -1.403, 2.186, -0.782, 1.261, 0.0]
_PREPLACE = [-3.702, -3.486, 1.425, -1.081, 0.561, 0.0]
_STOW = [0.0, -1.95, 1.90, -1.55, -1.57, 0.0]


def _lerp(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return [list(a + (b - a) * (i / (n - 1))) for i in range(n)]


def nominal_leg(leg: str) -> List[List[float]]:
    """A documented, reproducible short sweep per leg (ARM order).  Deltas are kept MODEST so the
    swept-volume geometry is a plausible single arm leg and -- on the carry legs -- the carton
    attitude witness stays near level (the leg does not itself re-orient the tool; the mission's
    leveling controller does, out of this bench's scope).  A scenario inlines exact recorded
    waypoints (``trajectory``) to supersede these."""
    approach = (np.array(_GRASP) + np.array([0.0, 0.15, -0.20, 0.0, 0.0, 0.0])).tolist()  # TCP standoff above grasp
    if leg == 'collect_pick':
        return _lerp(approach, _GRASP, 6)
    if leg == 'carry':
        # a modest tuck-fold off the lift pose (joints 2-3 only) -- orientation ~held, carton ~level
        folded = (np.array(_LIFT) + np.array([0.0, -0.15, 0.20, 0.0, 0.0, 0.0])).tolist()
        return _lerp(_LIFT, folded, 8)
    if leg == 'preplace_approach':
        near = (np.array(_PREPLACE) + np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0])).tolist()  # pan in to the preplace
        return _lerp(near, _PREPLACE, 8)
    if leg == 'set_down':
        place = (np.array(_PREPLACE) + np.array([0.0, -0.12, 0.0, 0.0, 0.0, 0.0])).tolist()  # descend
        return _lerp(_PREPLACE, place, 6)
    if leg == 'withdraw':
        up = (np.array(_PREPLACE) + np.array([0.0, 0.20, -0.10, 0.0, 0.0, 0.0])).tolist()
        return _lerp(_PREPLACE, up, 6)
    if leg == 'transfer':
        shelf = (np.array(_GRASP) + np.array([-0.20, 0.0, 0.08, 0.0, 0.0, 0.0])).tolist()
        return _lerp(approach, shelf, 8)
    raise ValueError('unknown arm leg %r (known: %s)' % (leg, tuple(DEFAULT_LEG_CYCLE_S)))


CARRY_LEGS = frozenset({'carry', 'preplace_approach', 'set_down', 'withdraw'})


# --------------------------------------------------------------------------- mock ESDF

@dataclass
class MockEsdf:
    """Obstacles rasterised into a voxel signed-distance field (the perceived-clearance instrument).
    Voxels inside an obstacle are negative, free voxels carry the distance to the nearest obstacle
    surface; a query point OUTSIDE the grid reads as occupied (distance 0) -- the nvblox
    unknown->occupied convention (isaac_motion.update_esdf).  Trilinear interpolation on query."""
    origin: np.ndarray      # (3,) world coord of voxel-centre [0,0,0]
    voxel: float
    sdf: np.ndarray         # (nx,ny,nz) float32 signed distance (m)

    @classmethod
    def rasterise(cls, obstacles: Sequence[Obstacle], t: float, lo, hi, voxel: float) -> 'MockEsdf':
        """Rasterise only the OBSERVED obstacles (``observed=False`` ones are invisible to the ESDF --
        the scout's *unobserved == free* finding)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        dims = np.maximum(2, np.ceil((hi - lo) / voxel).astype(int) + 1)
        axes = [lo[a] + voxel * np.arange(dims[a]) for a in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing='ij')
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        sdf = np.full(len(pts), np.inf, np.float32)
        for ob in obstacles:
            if not ob.observed:
                continue
            sdf = np.minimum(sdf, ob.signed_distance(pts, t).astype(np.float32))
        # No observed obstacle in a cell -> free; cap the "nearest observed surface" at the grid
        # diagonal (a sane finite "clear" reading, not a 1e3 m sentinel).
        diag = float(np.linalg.norm(hi - lo))
        sdf = np.where(np.isfinite(sdf), sdf, np.float32(diag))
        return cls(origin=lo.copy(), voxel=float(voxel), sdf=sdf.reshape(tuple(dims)))

    def query(self, points: np.ndarray) -> np.ndarray:
        """Trilinearly-interpolated signed distance at each point (N,3); outside the grid -> 0 (occupied)."""
        p = (np.asarray(points, float) - self.origin) / self.voxel
        dims = np.asarray(self.sdf.shape)
        inside = np.all((p >= 0) & (p <= dims - 1), axis=1)
        out = np.zeros(len(p), float)
        if inside.any():
            q = p[inside]
            i0 = np.floor(q).astype(int)
            i0 = np.clip(i0, 0, dims - 2)
            f = q - i0
            vals = np.zeros(len(q), float)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        w = (np.where(dx, f[:, 0], 1 - f[:, 0]) *
                             np.where(dy, f[:, 1], 1 - f[:, 1]) *
                             np.where(dz, f[:, 2], 1 - f[:, 2]))
                        vals += w * self.sdf[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
            out[inside] = vals
        return out


def _workspace_bounds(sphere_frames: Sequence[np.ndarray], obstacles: Sequence[Obstacle],
                      times: Sequence[float], voxel: float, pad_voxels: int = 6):
    pts = [s[:, :3] for s in sphere_frames if len(s)]
    lo = np.min([p.min(0) for p in pts], axis=0) if pts else np.zeros(3)
    hi = np.max([p.max(0) for p in pts], axis=0) if pts else np.zeros(3)
    for ob in obstacles:
        for t in times:
            olo, ohi = ob.aabb_at(t)
            lo, hi = np.minimum(lo, olo), np.maximum(hi, ohi)
    pad = voxel * pad_voxels
    return lo - pad, hi + pad


# --------------------------------------------------------------------------- clearance passes

def min_true_clearance(spheres: np.ndarray, obstacles: Sequence[Obstacle], t: float):
    """(min signed gap, worst obstacle name) of the swept spheres vs every obstacle at time ``t``."""
    if not len(spheres) or not obstacles:
        return math.inf, None
    best, who = math.inf, None
    for ob in obstacles:
        g = float((ob.signed_distance(spheres[:, :3], t) - spheres[:, 3]).min())
        if g < best:
            best, who = g, ob.name
    return best, who


def min_perceived_clearance(spheres: np.ndarray, esdf: MockEsdf):
    if not len(spheres):
        return math.inf
    return float((esdf.query(spheres[:, :3]) - spheres[:, 3]).min())


# --------------------------------------------------------------------------- the gate protocol

@dataclass(frozen=True)
class FrameClearance:
    t: float
    min_true: float
    worst_obstacle: Optional[str]
    min_perceived: float
    intruder: Optional[Obstacle]  # the worst OBSERVED obstacle perceived within margin at this frame


@dataclass(frozen=True)
class GateVerdict:
    behavior: str
    n_replans: int = 0
    extra_cycle_time_s: float = 0.0
    reason: str = ''


# A Gate maps (scenario, per-frame clearances) -> a verdict.
Gate = Callable[[Scenario, List[FrameClearance]], GateVerdict]


def baseline_gate(scenario: Scenario, frames: List[FrameClearance]) -> GateVerdict:
    """The CURRENT stack: no perceived-ESDF collision gate on these arm legs (collect runs
    ground-truth targets + a cage pair-filter; contact/insert legs are collision-OFF straight lines).
    It never avoids, re-plans, stops or refuses -- the nominal trajectory executes blind.  Use this
    for the BEFORE numbers: the gap is exactly the scenarios where the nominal path collides and
    nothing acts."""
    return GateVerdict('none', 0, 0.0, 'current stack: no perceived-clearance gate on this leg')


def reference_gate(scenario: Scenario, frames: List[FrameClearance]) -> GateVerdict:
    """A transparent collision-aware REFERENCE gate (the bench's instrument, not the sibling lanes'
    code) modelling what sim-rgbd-arm-collision-aware-v1 / sim-arm-reactive-replan-v1 target, and the
    captain's section-9 decisions (people as generic dynamic volumes with the human gate owning the
    STOP; the cart as a known world obstacle).  It reacts ONLY to what it PERCEIVES -- an obstacle the
    ESDF cannot see (``observed=False``) never triggers it, which is exactly how the bench measures the
    current stack's blindness.  Over the whole planned+executed sweep:

      1. A PERSON/HAND perceived within margin -> ``slow_stop_resume`` (+a stop dwell).  Per the scout,
         a person is a routing aid, NEVER the stop mechanism -- the human gate owns the STOP.
      2. A MOVING obstacle (carton sliding, cart) perceived within margin at execution -> ``preempt_replan``.
      3. A STATIC obstacle perceived within margin AT the leg goal -> ``refuse`` before motion.
      4. A STATIC obstacle perceived within margin beside a free-space leg -> ``avoid`` (re-plan a detour).
      5. Nothing perceived within margin -> ``clear``.

    The margin is the scenario's ``margin_m`` -- never widened."""
    margin = scenario.margin_m
    nominal = scenario.nominal_cycle_s()
    intruders = [f.intruder for f in frames if f.intruder is not None]
    is_person = lambda o: o.motion.obstacle_class in ('person', 'hand')
    is_moving = lambda o: o.motion.kind != 'static'
    if any(is_person(o) for o in intruders):
        return GateVerdict('slow_stop_resume', 0, 0.35 * nominal,
                           'person/hand perceived within %.0f mm -> human gate slow/stop, resume on clear' % (margin * 1e3))
    if any(is_moving(o) for o in intruders):
        return GateVerdict('preempt_replan', 1, 0.30 * nominal,
                           'moving obstacle perceived within %.0f mm at execution -> preempt + re-plan' % (margin * 1e3))
    if intruders:  # static, observed
        if any(o.at_goal for o in intruders):
            return GateVerdict('refuse', 0, 0.0,
                               'perceived static obstacle within %.0f mm AT the leg goal -> refuse before motion' % (margin * 1e3))
        return GateVerdict('avoid', 1, 0.20 * nominal,
                           'perceived static obstacle within %.0f mm beside the leg -> re-plan a detour keeping margin' % (margin * 1e3))
    return GateVerdict('clear', 0, 0.0, 'perceived path clear by >= %.0f mm' % (margin * 1e3))


# --------------------------------------------------------------------------- run one scenario

@dataclass
class ScenarioResult:
    scenario_id: str
    arm_leg: str
    expected_behavior: str
    gate_behavior: str
    verdict_match: bool
    collision_free_nominal: bool
    min_true_clearance_m: float
    min_perceived_clearance_m: float
    esdf_error_m: float
    worst_obstacle: Optional[str]
    n_replans: int
    extra_cycle_time_s: float
    cycle_time_overhead_frac: float
    tilt_peak_deg: float
    executed_collision_free: bool
    executed_min_true_clearance_m: float
    passed: bool
    fail_reasons: Tuple[str, ...]
    reason: str

    def as_row(self) -> dict:
        return {k: getattr(self, k) for k in (
            'scenario_id', 'arm_leg', 'expected_behavior', 'gate_behavior', 'verdict_match',
            'collision_free_nominal', 'min_true_clearance_m', 'min_perceived_clearance_m',
            'esdf_error_m', 'worst_obstacle', 'n_replans', 'extra_cycle_time_s',
            'cycle_time_overhead_frac', 'tilt_peak_deg', 'executed_collision_free',
            'executed_min_true_clearance_m', 'passed')}


# A gate verdict that ACTS to prevent a strike -- the executed path never reaches the obstacle,
# so a nominal-path collision is credited as prevented (collision-free executed).
_ACTING = frozenset({'avoid', 'preempt_replan', 'slow_stop_resume', 'refuse'})


def run_scenario(scenario: Scenario, arm: ArmModel, gate: Gate,
                 thresholds: Thresholds = Thresholds()) -> ScenarioResult:
    traj = build_trajectory(scenario)
    carry = scenario.arm_leg in CARRY_LEGS
    # Spatial swept volume per waypoint (config-dependent, time-independent).
    sphere_frames = [arm.spheres_at(cfg, carry=carry) for cfg in traj]
    # Time sampling for scripted-motion obstacles; map each time to a swept waypoint (sweep over the leg).
    times = list(np.arange(0.0, scenario.duration_s + 1e-9, scenario.time_step_s))
    if len(times) < 2:
        times = [0.0, scenario.duration_s]

    # Build a per-time ESDF only when an obstacle actually moves; else one static ESDF reused.
    moving = any(ob.motion.kind != 'static' for ob in scenario.obstacles)
    lo, hi = _workspace_bounds(sphere_frames, scenario.obstacles, times, scenario.voxel_m)
    static_esdf = None
    if not moving:
        static_esdf = MockEsdf.rasterise(scenario.obstacles, 0.0, lo, hi, scenario.voxel_m)

    frames: List[FrameClearance] = []
    glob_true, glob_perc, glob_err, glob_worst = math.inf, math.inf, 0.0, None
    for k, t in enumerate(times):
        # Which swept waypoint is "live" at this time (evenly map the leg across the horizon).
        wi = min(len(sphere_frames) - 1, int(k * len(sphere_frames) / max(1, len(times))))
        sph = sphere_frames[wi]
        true_c, worst = min_true_clearance(sph, scenario.obstacles, t)
        esdf = static_esdf if static_esdf is not None else MockEsdf.rasterise(
            scenario.obstacles, t, lo, hi, scenario.voxel_m)
        perc_c = min_perceived_clearance(sph, esdf)
        # The gate reacts only to what it PERCEIVES: the worst OBSERVED obstacle (unobserved ones are
        # invisible to it).  Its class picks the reaction (person/hand -> stop; object -> re-plan).
        obs_c, obs_worst = min_true_clearance(
            sph, [o for o in scenario.obstacles if o.observed], t)
        intruder = None
        if obs_c < scenario.margin_m:
            by_name = {ob.name: ob for ob in scenario.obstacles}
            intruder = by_name.get(obs_worst)
        frames.append(FrameClearance(float(t), float(true_c), worst, float(perc_c), intruder))
        if true_c < glob_true:
            glob_true, glob_worst = true_c, worst
        glob_perc = min(glob_perc, perc_c)
        glob_err = perc_c - true_c if abs(perc_c - true_c) > abs(glob_err) else glob_err

    verdict = gate(scenario, frames)
    collision_free_nominal = glob_true >= 0.0
    # Executed path: a gate that ACTS prevents the strike (stops short / detours / refuses) ->
    # collision-free, clearance held at the margin target; a non-acting gate executes the nominal path.
    if verdict.behavior in _ACTING:
        executed_free = True
        executed_min = max(scenario.margin_m, glob_true) if glob_true >= 0 else scenario.margin_m
    else:
        executed_free = collision_free_nominal
        executed_min = glob_true

    tilt_peak = _tilt_peak(arm, traj) if carry else 0.0
    overhead = verdict.extra_cycle_time_s / scenario.nominal_cycle_s()

    verdict_match = verdict.behavior == scenario.expected_behavior
    min_true_req = (thresholds.min_true_clearance_m if thresholds.min_true_clearance_m is not None
                    else scenario.margin_m)
    fails = []
    if thresholds.no_contact and not executed_free:
        fails.append('contact: executed path collided (min true %.1f mm)' % (executed_min * 1e3))
    if executed_free and executed_min < min_true_req:
        fails.append('clearance: executed min true %.1f mm < required %.1f mm'
                     % (executed_min * 1e3, min_true_req * 1e3))
    if overhead > thresholds.cycle_time_overhead_frac:
        fails.append('cycle-time: overhead %.0f%% > %.0f%% bound'
                     % (overhead * 100, thresholds.cycle_time_overhead_frac * 100))
    if thresholds.require_verdict_match and not verdict_match:
        fails.append('verdict: gate %r != expected %r' % (verdict.behavior, scenario.expected_behavior))

    return ScenarioResult(
        scenario_id=scenario.id, arm_leg=scenario.arm_leg,
        expected_behavior=scenario.expected_behavior, gate_behavior=verdict.behavior,
        verdict_match=verdict_match, collision_free_nominal=collision_free_nominal,
        min_true_clearance_m=round(glob_true, 5), min_perceived_clearance_m=round(glob_perc, 5),
        esdf_error_m=round(glob_err, 5), worst_obstacle=glob_worst,
        n_replans=verdict.n_replans, extra_cycle_time_s=round(verdict.extra_cycle_time_s, 3),
        cycle_time_overhead_frac=round(overhead, 4), tilt_peak_deg=round(tilt_peak, 4),
        executed_collision_free=executed_free, executed_min_true_clearance_m=round(executed_min, 5),
        passed=not fails, fail_reasons=tuple(fails), reason=verdict.reason)


def _tilt_peak(arm: ArmModel, traj: Sequence[Sequence[float]]) -> float:
    """Carried-carton attitude peak (deg) over the leg -- the standing 'box parallel to the ground'
    witness.  Welds the carton level at the first waypoint, replays its world up-axis along the path."""
    from twin.restock_manipulation import carton_level as cl
    tool_R0 = arm.tool_rotation(traj[0])
    up_in_tool = cl.carton_up_in_tool(tool_R0)
    _, _, tilt = cl.predict_path_tilt(arm.robot, [list(w) for w in traj], up_in_tool,
                                      tool_link='ur_tool0', root=arm.root)
    return float(tilt)


# --------------------------------------------------------------------------- reporting

def summarise(results: Sequence[ScenarioResult], gate_name: str, thresholds: Thresholds) -> dict:
    n = len(results)
    npass = sum(r.passed for r in results)
    return {
        'gate': gate_name,
        'n_scenarios': n,
        'n_passed': npass,
        'n_failed': n - npass,
        'all_passed': npass == n,
        'verdict_match_rate': round(sum(r.verdict_match for r in results) / n, 3) if n else 0.0,
        'nominal_collisions': sum(not r.collision_free_nominal for r in results),
        'executed_collisions': sum(not r.executed_collision_free for r in results),
        'thresholds': {
            'no_contact': thresholds.no_contact,
            'min_true_clearance_m': thresholds.min_true_clearance_m,
            'cycle_time_overhead_frac': thresholds.cycle_time_overhead_frac,
            'require_verdict_match': thresholds.require_verdict_match,
        },
    }


def results_csv(results: Sequence[ScenarioResult]) -> str:
    import csv
    import io
    cols = list(results[0].as_row().keys()) if results else []
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in results:
        w.writerow(r.as_row())
    return buf.getvalue()


def report_markdown(results: Sequence[ScenarioResult], summary: dict, *, title='Arm collision bench') -> str:
    L = ['# %s' % title, '',
         '**Gate:** `%s`  ·  **Scenarios:** %d  ·  **Passed:** %d / %d  ·  **Verdict-match:** %.0f%%'
         % (summary['gate'], summary['n_scenarios'], summary['n_passed'], summary['n_scenarios'],
            summary['verdict_match_rate'] * 100), '',
         '**Nominal-path collisions:** %d  ·  **Executed collisions (gate credited):** %d'
         % (summary['nominal_collisions'], summary['executed_collisions']), '',
         'Thresholds: no contact=%s · min true clearance >= %s · cycle-time overhead <= %.0f%% · verdict match=%s'
         % (summary['thresholds']['no_contact'],
            ('margin' if summary['thresholds']['min_true_clearance_m'] is None
             else '%.0f mm' % (summary['thresholds']['min_true_clearance_m'] * 1e3)),
            summary['thresholds']['cycle_time_overhead_frac'] * 100,
            summary['thresholds']['require_verdict_match']), '',
         '| scenario | leg | expect | gate | match | min true (mm) | min perceived (mm) | ESDF err (mm) | replans | +time (s) | tilt (deg) | PASS |',
         '|---|---|---|---|:-:|--:|--:|--:|--:|--:|--:|:-:|']
    for r in results:
        L.append('| %s | %s | %s | %s | %s | %.1f | %.1f | %.1f | %d | %.1f | %.2f | %s |' % (
            r.scenario_id, r.arm_leg, r.expected_behavior, r.gate_behavior,
            '✓' if r.verdict_match else '✗', r.min_true_clearance_m * 1e3,
            r.min_perceived_clearance_m * 1e3, r.esdf_error_m * 1e3, r.n_replans,
            r.extra_cycle_time_s, r.tilt_peak_deg, 'PASS' if r.passed else 'FAIL'))
    L += ['', '### Fail detail']
    any_fail = False
    for r in results:
        if not r.passed:
            any_fail = True
            L.append('- **%s**: %s' % (r.scenario_id, '; '.join(r.fail_reasons)))
    if not any_fail:
        L.append('- none')
    L += ['', '_Reasons (gate verdicts):_']
    for r in results:
        L.append('- `%s` -> %s: %s' % (r.scenario_id, r.gate_behavior, r.reason))
    return '\n'.join(L) + '\n'


# --------------------------------------------------------------------------- gate resolution

def resolve_gate(spec: str) -> Tuple[Gate, str]:
    """``'baseline'`` / ``'reference'`` / ``'module:function'`` -> (gate, name)."""
    if spec in ('baseline', 'before', 'current'):
        return baseline_gate, 'baseline'
    if spec in ('reference', 'ref', 'after'):
        return reference_gate, 'reference'
    if ':' in spec:
        import importlib
        mod, fn = spec.split(':', 1)
        g = getattr(importlib.import_module(mod), fn)
        return g, spec
    raise ValueError('unknown gate %r (baseline|reference|module:function)' % spec)


def run_suite(scenarios: Sequence[Scenario], gate_spec: str, *, urdf=None, curobo_yml=None,
              thresholds: Thresholds = Thresholds()):
    arm = ArmModel.load(urdf, curobo_yml)
    gate, gate_name = resolve_gate(gate_spec)
    results = [run_scenario(s, arm, gate, thresholds) for s in scenarios]
    return results, summarise(results, gate_name, thresholds)
