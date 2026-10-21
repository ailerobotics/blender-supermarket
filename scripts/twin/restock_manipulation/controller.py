"""Bounded transfer state machine; all runtime evidence must originate in sensors."""
from dataclasses import asdict
import math
import os
import time
from ..restock_camera.freshness import require_fresh

from .fit import axis_fits
from .types import MotionPort, ProductProfile, Result, Sensors

# Controller ticks to HOLD a settled contact step before re-checking the latch, so the arm link
# velocity decays to ~0 (the native SurfaceGripper seals a STATIONARY pad but not a moving one --
# stations verdict, firstmate run-203). Each tick is a sensor frame = many physics steps under pacing.
CONTACT_FREEZE_TICKS = 5
# fm358 cheapest guard: a staged cart-swing joint move (tuck/pan/retreat_tuck) failed ONCE in run-283
# at the plan->execute handoff -- its plan SUCCEEDED normally (~52 s, like runs 280/282) yet the move
# failed, a run-to-run transient not pinnable from the truncated host telemetry (start-tol/duration/
# tracking all ruled out). Retry a failed staged move this many times (fresh plan + deadline) before
# failing the stage, mirroring the planner's plan-reject REPLAN but at the staged-move level.
STAGED_MOVE_RETRY_MAX = 2
# fm367: a general-path collect move (the straight-line insert/retreat, or any free-space reach) that
# returns 'failed' on a SIM-CLOCK move/settle deadline miss -- a render-pacing transient, NOT a
# collision/plan-rejection (run-294: the straight insert/retreat settled 10 ms past its await_settle
# sim deadline at RELEASE and ended the mission) -- is re-issued with a fresh deadline up to this many
# times before failing the phase. Mirrors STAGED_MOVE_RETRY_MAX but for the general move path, which
# had NO retry. A collision/plan-rejection (port raised -> flag False) is NOT retried (fail-fast kept).
MOVE_DEADLINE_RETRY_MAX = 2
# fm387: which collect phases issue an arm MOVE (and so can suffer a goal-tolerance/track-miss the
# host-stack WATCHDOG surfaces -- ros_adapter failed_result_pending / drain -- OUTSIDE poll(), where
# fm372's in-poll retry never saw it). Staged moves (begin_joint_goal) use the staged budget; the rest
# of the move phases use the deadline budget. Non-move phases (observe/grasp/release/verify/idle) issue
# no begin, so a watchdog motion fault there is not re-issuable -> hard fail.
_STAGED_MOVE_PHASES = frozenset({'tuck', 'pan', 'retreat_tuck', 'stow'})
_GENERAL_MOVE_PHASES = frozenset({'approach', 'contact', 'lift', 'preplace', 'place', 'retreat'})
# fm390: the place-side poses are re-registered in the place_approach base frame after the base drives
# in front of the cart mouth, so they are NOT reach-gated in the pickup frame during observe.
_PLACE_SIDE_PHASES = frozenset({'preplace', 'place', 'retreat'})


def _retry_max_env(var, default):
    """fm371: SIM-ONLY env override for a bounded per-move retry budget. Env unset => the module
    default (byte-identical real-rig); set => the env value clamped to [0, 8]. A malformed value
    falls back to the default rather than raising at mission boot."""
    try:
        return max(0, min(8, int(os.environ.get(var, default))))
    except (TypeError, ValueError):
        return int(default)


def _float_env(var, default):
    """fm389: SIM-ONLY positive-float env override (e.g. a place-verify tolerance). Env unset/malformed
    => the module default; set => the parsed value clamped to (0, 5] m."""
    try:
        return max(1e-3, min(5.0, float(os.environ.get(var, default))))
    except (TypeError, ValueError):
        return float(default)


def _int_env(var, default, lo, hi):
    """fm395: SIM-ONLY positive-int env override (grasp confirm/wait tick counts). Env unset/malformed
    => the module default; set => the parsed value clamped to [lo, hi]."""
    try:
        return max(lo, min(hi, int(os.environ.get(var, default))))
    except (TypeError, ValueError):
        return int(default)


def _float_env_bounded(var, default, lo, hi):
    """fm419: SIM-ONLY float env override clamped to [lo, hi]. Unlike _float_env, lo may be 0 so the
    default-off (0 s) case stays EXACTLY 0 (byte-identical). Env unset/malformed => the module default."""
    try:
        return max(lo, min(hi, float(os.environ.get(var, default))))
    except (TypeError, ValueError):
        return float(default)


class TransferController:
    def __init__(self, motion: MotionPort, profiles: dict[str, ProductProfile], *,
                 cart_location: str = 'rollwagen', max_age: float = 0.5,
                 phase_timeout: float = 30.0, fixture_mode: bool = False,
                 frame: str = 'ur_base_link', pose_policy=None, transport_joints=None,
                 steady_clock=time.monotonic, contact_straight_line: bool = False,
                 contact_overpress_step_m: float = 0.0, contact_overpress_cap_m: float = 0.0,
                 staged_cart_swing: bool = False, cart_pan_joint0: float = 0.0,
                 cart_tuck_shape=None, carry_wrist1_cap=None,
                 carry_lift_shape=None, carry_wrist2_min=None,
                 cart_cage_place: bool = False, cage_axis=(1.0, 0.0, 0.0),
                 cage_mouth_standoff: float = 0.55, cage_insert_gap: float = 0.03,
                 cage_preplace_shape=None, reposition_before_place: bool = False,
                 grasp_track_m=None):
        self.steady_clock=steady_clock
        # fm390: drive the BASE in front of the rollwagen mouth between the carry fold and the place,
        # instead of reaching the place around from the collect station pose. When set (with the staged
        # cart swing), the carry fold goes lift -> tuck -> TRANSIT (hold, emit one reposition_base control)
        # and the arm waits -- the adapter pauses stepping us, drives the base to place_approach, then sets
        # base_repositioned so TRANSIT -> preplace proceeds at the new pose. Default off => byte-identical.
        self.reposition_before_place = bool(reposition_before_place)
        self.base_repositioned = False
        self._source_obs = None   # fm390: the matched source observation, for re-running the place policy
        # fm395 (A): require a STABLE confirmed attachment (s.contact = native grippedObjects OR the sim
        # fallback weld) for grasp_confirm_ticks consecutive ticks before lifting -- never advance to lift
        # on a single transient contact (run-306: a one-tick weld, released by a status flicker, let the
        # arm lift air and the carton-less lift cuMotion-collision-failed). If no stable hold forms within
        # grasp_wait_ticks, bounded-RETRY the grasp (re-seat: re-press via 'contact', re-fire the gripper)
        # up to grasp_retry_max, then FAIL collect honestly -- never lift without a held carton. Defaults
        # (1 confirm tick, 0 retries) reproduce the old single-tick advance => byte-identical when off.
        self.grasp_confirm_ticks = _int_env('RESTOCK_GRASP_CONFIRM_TICKS', 1, 1, 60)
        self.grasp_retry_max = _retry_max_env('RESTOCK_GRASP_RETRY_MAX', 0)
        self.grasp_wait_ticks = _int_env('RESTOCK_GRASP_WAIT_TICKS', 40, 2, 600)
        self._grasp_confirm = 0
        self._grasp_waits = 0
        self._grasp_retries = 0
        # fm419 (run-320): TIME-bounded tolerance (a MAX COLLISION AGE, not a tick count) for a
        # COLLISION-only staleness while a carton is HELD -- the weld/attach ESDF re-incorporation lags the
        # host collision_time (run-320: collision_time froze ~0.6 s at the grasp->lift weld, outlasting the
        # earlier fm415 5-tick cap). While the collision stamp's age (now - collision_time) is within
        # collision_stale_tol_s the arm is PAUSED -- pose held at zero velocity, the held move NOT
        # issued/advanced (never lift into a stale collision world) -- and RESUMES (re-plans from the held
        # pose) on the first fresh collision stamp; a frozen stamp's age keeps growing and trips honestly
        # once it passes the bound (no livelock). Scoped to held (post-attach) + collision-sole so it can
        # never mask a base-moving / feedback / pre-grasp fault; pre-grasp stays zero-tolerance. The
        # capture's hold-last-good (isaac_factory) already blocks the -1 mirror gap; this covers a genuine
        # brief re-incorporation lag. 0 s => off => byte-identical.
        self.collision_stale_tol_s = _float_env_bounded('RESTOCK_COLLISION_STALE_TOL_S', 0.8, 0.0, 5.0)
        # fm327 belt-and-braces wrist_1 cap for the task-space LIFT carry move (None = off).
        self.carry_wrist1_cap = None if carry_wrist1_cap is None else float(carry_wrist1_cap)
        # fm403: route the loaded LIFT to a FIXED joint-space clear-region carry shape via
        # begin_joint_goal (like cage_preplace_shape), not a task-space IK -- the task-space lift IK can
        # land in the wrist_2-vs-forearm self-collision branch (w2 swung negative; runs 300d/305/311
        # jammed even with wrist_1 capped, because wrist_2 still crossed the stop; bench-proven). The lift
        # only needs to raise the carton clear of the shelf before the carry fold, so a joint goal is the
        # right primitive. None => the legacy task-space lift (with carry_wrist1_cap). 6 finite joints.
        self.carry_lift_shape = (tuple(float(v) for v in carry_lift_shape)
                                 if carry_lift_shape is not None else None)
        if self.carry_lift_shape is not None and (len(self.carry_lift_shape) != 6
                or not all(map(math.isfinite, self.carry_lift_shape))):
            raise ValueError('carry_lift_shape must be 6 finite joint values')
        # fm403: wrist_2 LOWER bound (rad) guarding any remaining task-space carry move (the lift
        # fallback when carry_lift_shape is unset) so cuMotion cannot pick the w2-negative self-colliding
        # branch. None = off => byte-identical.
        self.carry_wrist2_min = None if carry_wrist2_min is None else float(carry_wrist2_min)
        # fm410 (run-316): PHYSICS grasp confirm. The grasp phase confirms an ATTACHMENT (s.contact =
        # weld/native grip, fm395 semantics, unchanged) WITHOUT consulting the source camera -- the arm
        # occluding the fixed source cam during the grasp/hold is the normal case, never a fault. The LIFT
        # then serves as the short physics PROBE: before advancing to carry (tuck/preplace) we require the
        # measured carton centre (s.placed_pose, the fm389 ground-truth read) to have RISEN by >= grasp_track_m
        # from its grasp-confirm baseline -- i.e. the carton physically tracked the tool up. A carton that
        # did NOT track (placed_pose flat) means the attach signal was spurious / the body slipped, so FAIL
        # honestly rather than carry a phantom load. None/<=0 (or no placed_pose from the driver -- real rig)
        # => the gate is inert and the advance is the legacy fm395 one (byte-identical).
        self.grasp_track_m = None if grasp_track_m is None else float(grasp_track_m)
        if self.grasp_track_m is not None and (not math.isfinite(self.grasp_track_m) or self.grasp_track_m < 0):
            raise ValueError('grasp_track_m must be a finite, non-negative metre threshold')
        self._carton_z_at_grasp = None   # fm410: measured carton z latched at grasp-confirm, for the lift probe
        # fm425 (run-322 cold): WALL-clock contact fail-fast. Cold run-322 reached the contact descent and
        # crept CONTACT_STEP to 12 mm with contact=False (the native suction never latched and the
        # SIM_ATTACH weld never formed), then the arm-command stream lapsed into the 90 s host transport
        # heartbeat watchdog -- an opaque 'arm command heartbeat lost' with no named cause. The contact
        # phase's own phase_timeout is SIM-clock (self.elapsed += dt), so under render pacing (1.7-3.5 s
        # WALL per capture) the WALL heartbeat trips long before the sim phase_timeout. This is a WALL
        # budget on time-without-contact-progress in the contact phase: if it elapses the step FAILS FAST
        # with a named reason BEFORE the 90 s watchdog, so a stuck contact ends the attempt legibly instead
        # of burning the rest of the GPU wall cap. Reset on entering contact and on each CONTACT_STEP
        # advance (progress). 0 s (default) => off => byte-identical; bounded below the 90 s backstop.
        self.contact_stuck_wall_s = _float_env_bounded('RESTOCK_CONTACT_STUCK_WALL_S', 0.0, 0.0, 85.0)
        self._contact_progress_wall = None   # wall time of the last contact-phase progress (see step)
        # Execute the contact descent as a seeded straight-line move through the trajectory host instead
        # of a cuMotion plan (the motion port's begin(straight_line=True) path). Default False keeps the
        # real-rig collision-planned descent; the sim mission opts in via config -- a one-flag revert.
        self.contact_straight_line = bool(contact_straight_line)
        # SIM contact-feedback over-press (m). The native SurfaceGripper seals only on physical ~0 mm
        # seal-to-carton contact, not merely within maxGripDistance (proven: isolated side_flat latched at
        # 0 mm; mission failed at 0.45-3.25 mm). The compliant stroke cannot close that gap (it only
        # retracts) and a side grasp has no gravity assist, and the nominal residual VARIES run-to-run
        # (~4-7.5 mm sensing/pose variance) so a FIXED over-press undershoots or overshoots. Instead the
        # contact move steps forward by contact_overpress_step_m (toward the carton, along approach->
        # contact) with suction on until the seal latches (s.contact) or the capped travel
        # contact_overpress_cap_m is exhausted (then a mission fault). The settle at each step gives the
        # plugin a stationary pad to latch; the cap bounds the travel so a miss cannot drive deep into the
        # carton. SIM only: the panel collider is disabled so the pad can over-press; 0 keeps the real rig
        # (its real suction cup has the physical compliance this lacks) on a single collision-planned move.
        for v in (contact_overpress_step_m, contact_overpress_cap_m):
            if not math.isfinite(v) or v < 0:
                raise ValueError('contact over-press step/cap must be finite and non-negative')
        self.contact_overpress_step_m = float(contact_overpress_step_m)
        self.contact_overpress_cap_m = float(contact_overpress_cap_m)
        if max_age <= 0 or phase_timeout <= 0:
            raise ValueError('timeouts must be positive')
        self.motion, self.profiles = motion, profiles
        self.frame = frame
        self.pose_policy = pose_policy
        self.transport_joints = tuple(transport_joints) if transport_joints is not None else None
        if self.transport_joints is not None and (len(self.transport_joints) != 6 or
                not all(map(math.isfinite,self.transport_joints)) or
                not callable(getattr(motion,'begin_joint_goal',None))):
            raise ValueError('transport stow requires six finite joints and collision-planned joint goals')
        # STAGED cart-swing (firstmate 292): the direct lift->preplace swing sweeps the welded carton
        # through the robot's own chassis/arm-mount (cuMotion has no self-collision spheres for them),
        # stalling joint0 (run-236). Route it as joint-space via points instead: lift -> TUCK (fold to
        # a high CARRY shape that raises the carton well ABOVE the body at the current shoulder-pan)
        # -> PAN (rotate the shoulder to the cart side at that high radius) -> extend to the mirrored
        # preplace; mirror on the way back (retreat -> retreat_tuck -> stow). TUCK/retreat_tuck fold at
        # the MEASURED pan; PAN rotates to ``cart_pan_joint0``. The carry shape (``cart_tuck_shape``,
        # arm joints 1-5) is CPU KDL/FK-verified (staged_swing_verify.py): it lifts the carton to
        # tcp_z~1.1 m (vs the body at z<0.26) so the fold goes UP and away from the chassis/arm_mount
        # -- run-240's STOW tuck folded the carton DOWN into the body (tcp_z 0.79, path dipped to
        # 0.092 m) and jammed shoulder_lift/wrist2. Falls back to the transport stow shape if no carry
        # shape is configured. gated ON only in the sim cart-place scenario; off => direct lift->preplace.
        self.cart_tuck_shape = (tuple(float(v) for v in cart_tuck_shape) if cart_tuck_shape is not None
                                else (self.transport_joints[1:] if self.transport_joints is not None else None))
        self.cart_pan_joint0 = float(cart_pan_joint0)
        self.staged_cart_swing = bool(staged_cart_swing) and self.cart_tuck_shape is not None \
            and self.transport_joints is not None
        if self.staged_cart_swing and (not math.isfinite(self.cart_pan_joint0)
                or len(self.cart_tuck_shape) != 5 or not all(map(math.isfinite, self.cart_tuck_shape))):
            raise ValueError('staged cart swing requires a finite cart_pan_joint0 and 5-joint cart_tuck_shape')
        # CART-CAGE open-face place (firstmate 350). The place-onto-cart is a two-step move THROUGH the
        # cart's single OPEN loading face (the composed /Rollwagen cage is modelled as world_obstacles:
        # deck + back + two side walls, open +x toward the robot -- CPU-derived from the composed stage,
        # ur_base_link frame). cuMotion plans the PREPLACE (collision-aware joint-space, seeded near the
        # pan branch) around those walls to a standoff JUST OUTSIDE the +x mouth; the PLACE then INSERTS
        # straight along the cage axis (begin(straight_line=True): collision-off seeded 2-point move)
        # -(axis) through the open face onto the deck target, stopping ``cage_insert_gap`` above the deck
        # so the carton settles the last cm on release. retreat backs straight out the same face.
        # ``cage_axis`` is the open-face direction in ur_base_link (default +x, the authoritative cart
        # pose: 1.27 m directly behind the base, mouth toward the robot). Gated on cart_cage_place; off
        # => the existing pose-IK preplace/place (byte-identical).
        self.cart_cage_place = bool(cart_cage_place)
        self.cage_axis = tuple(float(v) for v in cage_axis)
        self.cage_mouth_standoff = float(cage_mouth_standoff)
        self.cage_insert_gap = float(cage_insert_gap)
        if self.cart_cage_place:
            n = math.sqrt(sum(v * v for v in self.cage_axis))
            if (len(self.cage_axis) != 3 or not all(map(math.isfinite, self.cage_axis)) or n < 1e-6
                    or not math.isfinite(self.cage_mouth_standoff) or self.cage_mouth_standoff <= 0
                    or not math.isfinite(self.cage_insert_gap) or self.cage_insert_gap < 0):
                raise ValueError('cart cage place requires a non-zero finite cage_axis, a positive '
                                 'mouth standoff and a non-negative insert gap')
            self.cage_axis = tuple(v / n for v in self.cage_axis)  # unit open-face direction
        # Fixed-shape PREPLACE (run-279): the pose-IK preplace flips wrist_1 above the ur_wrist_2-vs-
        # ur_forearm self-collision stop (~0.70) and jams at execution (run-278: wrist_1 stuck 0.69 vs
        # cmd 1.99). When set, preplace is driven as a fixed collision-planned begin_joint_goal to this
        # CPU-IK-solved 6-joint config (wrist_1 below the stop; FK TCP = the sensed mouth standoff), and
        # the straight-line insert (place) still tracks the SENSED place target. None => pose-IK preplace
        # (byte-identical). Only meaningful with cart_cage_place.
        self.cage_preplace_shape = (tuple(float(v) for v in cage_preplace_shape)
                                    if cage_preplace_shape is not None else None)
        if self.cage_preplace_shape is not None and (len(self.cage_preplace_shape) != 6
                or not all(map(math.isfinite, self.cage_preplace_shape))):
            raise ValueError('cage_preplace_shape must be 6 finite joint values')
        self.cart_location, self.max_age = cart_location, max_age
        # fm423 (b) / run-321 cold collect: the relayed host stamps (collision_time from
        # station/status, feedback_time from /joint_states) ride a DIFFERENT clock relay than the
        # controller's node `now`, so under render pacing a freshly-incorporated stamp can arrive
        # marginally AHEAD of `now` (a one-to-few physics-step delivery-order lead). The freshness
        # lower bound `0 <= now-stamp` then wrongly flags that fresh-but-future stamp as not-fresh,
        # tripping the top-of-step invariant; with a carton HELD the fm419(a) COLLISION_STALE_PAUSE
        # then holds the lift every tick (run-321: collision_dt=-0.0167s, phase=lift) -- an indefinite
        # spurious pause that stalled the arm and starved the 90 s arm-command heartbeat ('arm command
        # heartbeat lost'). Tolerate that sub-step lead exactly as restock_camera.freshness (_RENDER_LEAD)
        # and trajectory_host.feedback_fresh (FEEDBACK_RENDER_LEAD_S) already do: a stamp up to
        # render_lead seconds in the future is fresh (strictly MORE current data -- safe); a genuinely
        # STALE stamp (age > max_age) is still rejected, and a wildly-future stamp (> render_lead) still
        # trips. Default 0.2 s matches trajectory_host's relayed-stamp-vs-node-clock race.
        self.render_lead = _float_env_bounded('RESTOCK_FRESH_RENDER_LEAD_S', 0.2, 0.0, 0.5)
        self.phase_timeout, self.fixture_mode = phase_timeout, fixture_mode
        self.phase = 'idle'
        self.held = False
        self._staged_retries = 0   # fm358: bounded retry of a failed staged cart-swing move
        self._move_deadline_retries = 0   # fm367: bounded retry of a sim-deadline fault on the general move path
        # fm371: the two bounded per-move retry budgets, SIM-ONLY env-tunable (default = the module
        # constant => byte-identical real-rig). The sim raises them so a stochastic cuMotion seed
        # streak across the long collect cart-swing re-rolls instead of ending the mission.
        self._staged_move_retry_max = _retry_max_env('RESTOCK_STAGED_MOVE_RETRY_MAX', STAGED_MOVE_RETRY_MAX)
        self._move_deadline_retry_max = _retry_max_env('RESTOCK_MOVE_DEADLINE_RETRY_MAX', MOVE_DEADLINE_RETRY_MAX)
        # fm389 direct PLACE verification tolerances (SIM-ONLY env-tunable). When the driver threads a
        # measured carton pose (s.placed_pose, RESTOCK_VERIFY_PLACE), the collect `verify` phase refuses an
        # empty-gripper success unless the carton rests on the support: centre within xy_tol of the sensed
        # support xy AND the carton bottom within z_tol of the support top. A carton that fell BELOW the
        # support (floor) is a fast hard-fail. Unset => defaults => the camera 2-frame verify is unchanged.
        self._place_verify_xy_tol = _float_env('RESTOCK_PLACE_VERIFY_XY_TOL', 0.25)
        self._place_verify_z_tol = _float_env('RESTOCK_PLACE_VERIFY_Z_TOL', 0.12)
        self._place_offdeck_frames = 0
        self.evidence = []
        self.reason = ''
        self.last_time = None

    def _overpressed_contact(self, offset):
        """Contact pose pushed ``offset`` m toward the carton, along the approach->contact descent axis,
        so the straight-line contact move can press the pad into ~0 mm seal contact. Orientation is
        unchanged. Returns the plain contact pose when offset is 0 or the approach pose is unavailable.
        See __init__ for why the SIM side grasp steps this over-press under contact feedback."""
        contact = self.poses['contact']
        approach = self.poses.get('approach')
        if offset <= 0 or approach is None:
            return contact
        d = [contact[i] - approach[i] for i in range(3)]
        n = math.sqrt(sum(v * v for v in d))
        if n < 1e-9:
            return contact
        step = offset / n
        return (contact[0] + d[0] * step, contact[1] + d[1] * step, contact[2] + d[2] * step,
                contact[3], contact[4], contact[5], contact[6])

    def _reregister_place(self, s, now):
        """fm390 PERCEIVE step. After the base has driven to place_approach in front of the rollwagen
        mouth, re-read the live deck target in the NEW arm-base frame and rebuild the place-side poses
        (preplace/place/retreat) + target_surface/target_pose from it. The place poses were registered
        in the PICKUP base frame during observe; the base has since moved, so placing from them would
        drive the arm to stale coordinates. Confirms the deck is still clear and the fresh target is
        reachable from here (the deck IS reachable from place_approach -- reach proof
        scripts/check_place_approach_reach.py). Returns a _fail Result on refusal, else None with the
        place-side poses rebuilt in place -- an empty-gripper 'placed off the deck' can never follow."""
        t = s.target
        if not t or t.location != self.destination or not self._provenance(t, now):
            return self._fail('reposition: deck target not reacquired at place_approach')
        if not t.clear:
            return self._fail('reposition: deck not clear at place_approach')
        if not self._pose_ok(t.pose) or not axis_fits(t.free_dimensions, self.profile.dimensions):
            return self._fail('reposition: deck target unreachable or undersized from place_approach')
        p = self.profile
        def pose(xyz, dz):
            return (*xyz[:2], xyz[2] + dz + p.tcp_surface_offset, *p.tcp_quaternion)
        if self.pose_policy is not None:
            # The place-side poses (preplace/place/retreat) depend only on the TARGET; the policy's
            # pick-side poses are recomputed against the stale source and discarded. cart_cage_place is
            # OFF in reposition mode -- the base nav replaces the arm-only +x mouth insert, so the place
            # is the policy's own reachable outward standoff + straight lower onto the deck interior.
            try:
                rebuilt = self.pose_policy.build(self._source_obs, t, p)
            except ValueError as exc:
                return self._fail('reposition: place policy refused fresh deck target: ' + str(exc))
            place_side = {k: tuple(rebuilt[k]) for k in _PLACE_SIDE_PHASES}
        else:
            place_side = dict(preplace=pose(t.pose, p.dimensions[2] + p.approach),
                              place=pose(t.pose, p.dimensions[2]),
                              retreat=pose(t.pose, p.dimensions[2] + p.approach))
        for k, v in place_side.items():
            if not self._pose_ok(v):
                print('REPOSITION_REACH_FAIL phase=%s mag=%.3f max_reach=%s deck=%s'
                      % (k, math.dist(v[:3], (0, 0, 0)), p.max_reach,
                         tuple(round(c, 3) for c in t.pose[:3])), flush=True)
                return self._fail('reposition: rebuilt place pose outside reach envelope')
        self.poses.update(place_side)
        self.target_pose = (*t.pose[:2], t.pose[2] + p.dimensions[2], *t.pose[3:])
        self.target_surface = t.pose
        print('REPOSITION_REREGISTER deck=%s place=%s preplace=%s'
              % (tuple(round(c, 3) for c in t.pose[:3]),
                 tuple(round(c, 3) for c in place_side['place'][:3]),
                 tuple(round(c, 3) for c in place_side['preplace'][:3])), flush=True)
        return None

    def start(self, operation: str, job: dict) -> None:
        if self.phase not in ('idle', 'succeeded', 'failed') or self.held:
            raise ValueError('transfer active or payload requires recovery')
        if operation not in ('collect', 'transfer'):
            raise ValueError('operation must be collect or transfer')
        if type(job.get('quantity')) is not int or job['quantity'] != 1:
            raise ValueError('only one sealed handling unit per operation supported')
        if job.get('sku') not in self.profiles:
            raise ValueError('unsupported SKU: configure a measured grasp profile')
        if not all(isinstance(job.get(k), str) and job[k] for k in
                   ('job_id', 'sku', 'source_bin', 'shelf_id')):
            raise ValueError('missing job identifiers')
        self.profile = self.profiles[job['sku']]
        p = self.profile
        if p.grasp_mode != 'top' and self.pose_policy is None:
            raise ValueError('side grasp requires the actual CAD wall pose policy')
        if (len(p.dimensions) != 3 or not all(math.isfinite(x) and x > 0 for x in p.dimensions)
                or len(p.tcp_quaternion) != 4
                or not math.isclose(sum(x*x for x in p.tcp_quaternion), 1, abs_tol=1e-4)
                or not all(math.isfinite(x) for x in p.tcp_quaternion)
                or not math.isfinite(p.tcp_surface_offset)
                or not math.isfinite(p.approach) or p.approach <= 0
                or not math.isfinite(p.max_reach) or p.max_reach <= 0):
            raise ValueError('invalid product geometry/calibration')
        self.job = dict(job)
        self.source = job['source_bin'] if operation == 'collect' else self.cart_location
        self.destination = self.cart_location if operation == 'collect' else job['shelf_id']
        if self.source == self.destination:
            raise ValueError('source and destination must differ')
        self.phase, self.elapsed = 'observe', 0.0
        self.last_time = None
        self.evidence, self.reason = [], ''
        self.issued = False
        self._contact_suction = False   # suction enabled once during the contact descent (grasp latch)
        self._contact_waits = 0         # bounded grace ticks awaiting a fresh acquisition deadline
        self._contact_step = 0.0        # cumulative contact-feedback over-press travel (m); see __init__
        self._contact_freeze = 0        # freeze-settle hold ticks at a contact step (see CONTACT_FREEZE_TICKS)
        self._grasp_confirm = 0         # fm395: consecutive confirmed-attachment ticks in grasp
        self._grasp_waits = 0           # fm395: ticks awaiting a confirmed attachment before a grasp retry
        self._grasp_retries = 0         # fm395: bounded grasp re-seat retries used this operation
        self._carton_z_at_grasp = None  # fm410: latched at grasp-confirm; the lift-probe tracking baseline
        self.verify_ids = set()
        self.release_time = None

    def _fresh(self, stamp, now):
        return math.isfinite(stamp) and -self.render_lead <= now - stamp <= self.max_age

    def _provenance(self, item, now):
        allowed = ('sensor', 'test_fixture') if self.fixture_mode else ('sensor',)
        return (item.source in allowed and bool(item.observation_id)
                and self._fresh(item.simulation_time, now) and item.frame == self.frame)

    def _pose_ok(self, pose):
        return (len(pose) == 7 and all(math.isfinite(x) for x in pose)
                and math.isclose(sum(x*x for x in pose[3:]), 1, abs_tol=1e-4)
                and math.dist(pose[:3], (0, 0, 0)) <= self.profile.max_reach)

    def _transition(self, phase):
        self.phase, self.elapsed, self.issued = phase, 0.0, False
        self._staged_retries = 0   # fm358: fresh retry budget per phase
        self._move_deadline_retries = 0   # fm367: fresh deadline-retry budget per phase
        if phase == 'contact':
            self._contact_progress_wall = self.steady_clock()   # fm425: start the WALL no-progress budget

    def _result(self, controls=()):
        status = self.phase if self.phase in ('failed', 'succeeded') else 'running'
        return Result(status, self.phase, self.reason, tuple(controls), tuple(self.evidence),
                      None if status=='failed' else getattr(self,'valid_until_steady',None))

    def _fail(self, reason):
        self.reason, self.phase = reason, 'failed'
        try:
            self.motion.stop()
        except Exception as exc:
            self.reason += f'; stop adapter failed: {exc}'
        # Never drop a held object on failure. Recovery is an explicit new decision.
        return self._result(({'kind': 'stop'}, {'kind': 'gripper', 'enabled': self.held}))

    def _raised_move_exc(self, exc, *, staged):
        """fm371/fm372: a collect move's poll() RAISED. Close the whole "a move raise kills the mission"
        family: route EVERY non-collision, non-joint-limit raise -- plan-exhaust (RuntimeError), a
        bad-seed trajectory that fails validate (ValueError), a sim-deadline/goal-tolerance/track-miss/
        feedback-lost surfaced as a raise, or any other class -- through the SAME bounded fresh-seed
        retry (a fresh move = a fresh plan + seed), since all are run-to-run cuMotion variance, not an
        unreachable pose. A JOINT-LIMIT violation is a hard geometric fault (the goal/plan is out of
        range) and is NEVER retried -> fail fast. (A COLLISION is a returned ``outcome=='failed'``, not
        a raise, so it never reaches here and keeps its own fail-fast branch.) Staged cart-swing moves
        use the staged budget; the general path uses the deadline budget. One MOVE_RETRY line per retry
        names the exception class + phase. Bounded: a persistent raise exhausts the (env-tunable) max
        -> _fail (no infinite loop)."""
        if 'outside joint limits' in str(exc):
            return self._fail('%s joint-limit violation: %s' % (self.phase, exc))
        if staged:
            if self._staged_retries < self._staged_move_retry_max:
                self._staged_retries += 1
                attempt, cap = self._staged_retries, self._staged_move_retry_max
            else:
                return self._fail('staged cart-swing %s move failed: %s' % (self.phase, exc))
        else:
            if self._move_deadline_retries < self._move_deadline_retry_max:
                self._move_deadline_retries += 1
                attempt, cap = self._move_deadline_retries, self._move_deadline_retry_max
            else:
                return self._fail('general %s move failed: %s' % (self.phase, exc))
        print('MOVE_RETRY phase=%s cls=%s attempt=%d/%d' %
              (self.phase, type(exc).__name__, attempt, cap), flush=True)
        self.issued = False   # re-plan + re-execute this move next tick
        return self._result()

    def retry_or_fail_move(self, reason):
        """fm387: a motion fault surfaced by the host-stack WATCHDOG (goal-tolerance / track-miss /
        watchdog-drain -- ros_adapter failed_result_pending()/drain()) fires on its 0.1 s timer OUTSIDE
        poll(), so fm372's in-poll bounded retry never saw it (run-300d: a stochastic move_group crash
        disrupted the LIFT execute -> goal-miss -> the watchdog _fail'd the mission with no MOVE_RETRY).
        Route it through the SAME bounded move retry: a goal-miss is transient (a fresh move = fresh plan
        + execute), so re-issue up to the bounded budget; a persistent miss exhausts -> _fail. A non-move
        phase (no begin to re-issue) -> _fail. Clears the port's terminal/broken state FIRST so the 0.1 s
        watchdog does not re-fire the same stale result before the controller re-issues next tick."""
        staged = self.phase in _STAGED_MOVE_PHASES
        if not (staged or self.phase in _GENERAL_MOVE_PHASES):
            return self._fail(reason)
        ack = getattr(self.motion, 'ack_failed_result', None)
        if callable(ack):
            ack()
        return self._raised_move_exc(RuntimeError(reason), staged=staged)

    def step(self, dt: float, sensors: Sensors) -> Result:
        if self.phase == 'idle':
            raise RuntimeError('call start before step')
        if self.phase in ('failed', 'succeeded'):
            return self._result()
        now = sensors.simulation_time
        self.valid_until_steady=sensors.valid_until_steady
        if (not math.isfinite(dt) or dt <= 0 or not math.isfinite(now)
                or (self.last_time is not None and now < self.last_time)):
            return self._fail('invalid or regressing simulation clock')
        self.last_time = now
        self.elapsed += dt
        if self.elapsed > self.phase_timeout:
            return self._fail('phase timeout: ' + self.phase)
        collision_fresh = self._fresh(sensors.collision_time, now)
        if (sensors.parked is not True or type(sensors.contact) is not bool
                or not self._fresh(sensors.feedback_time, now)
                or not collision_fresh
                or len(sensors.joint_positions) != 6
                or not all(math.isfinite(x) for x in sensors.joint_positions)):
            # fm412 (run-317/318): this invariant is a 5-condition OR and its single reason string left the
            # ACTUAL tripping field ambiguous (two runs died here at the grasp->weld boundary with no way to
            # tell parked vs a collision/feedback staleness). Name the exact failed condition(s) + values so
            # the next run is self-diagnosing instead of guessed-at.
            why = []
            if sensors.parked is not True: why.append(f'parked={sensors.parked!r}')
            if type(sensors.contact) is not bool: why.append(f'contact_type={type(sensors.contact).__name__}')
            if not self._fresh(sensors.feedback_time, now):
                why.append(f'feedback_dt={now - sensors.feedback_time:.4f}(band [{-self.render_lead},{self.max_age}])')
            if not collision_fresh:
                why.append(f'collision_dt={now - sensors.collision_time:.4f}(band [{-self.render_lead},{self.max_age}])')
            if len(sensors.joint_positions) != 6:
                why.append(f'njoints={len(sensors.joint_positions)}')
            elif not all(math.isfinite(x) for x in sensors.joint_positions):
                why.append('joints_nonfinite')
            # fm419 (run-320): a COLLISION-only staleness while the carton is HELD is the weld/attach ESDF
            # re-incorporation lag (run-320: collision_time froze ~0.6 s at the grasp->lift weld). Tolerate
            # it as a MAX COLLISION AGE: while the stamp age (now - collision_time) is within
            # collision_stale_tol_s, PAUSE the arm -- hold the current pose at zero velocity by NOT
            # issuing/advancing the held move (never lift into a stale collision world) and clear `issued` so
            # the move RE-PLANS from the held pose on the first fresh stamp. A frozen stamp's age keeps
            # growing and trips honestly once it passes the bound (no livelock). Scoped to held (post-attach)
            # + collision-sole so it can never mask a base-moving / feedback / pre-grasp fault; pre-grasp
            # stays zero-tolerance. 0 s => off => byte-identical. fm423 (b): the pause is for a genuinely
            # STALE stamp only -- the age must be on the stale side of the fresh band (> max_age). A
            # fresh-but-marginally-FUTURE stamp (render-pacing clock lead) is already fresh (render_lead),
            # so it never reaches here; a wildly-FUTURE stamp (age < -render_lead) is a real anomaly that
            # must TRIP honestly, NOT pause (a persistent future lead must never hold the arm forever --
            # the run-321 cold-collect failure mode).
            if (len(why) == 1 and why[0].startswith('collision_dt') and self.held
                    and self.collision_stale_tol_s > 0 and math.isfinite(sensors.collision_time)
                    and self.max_age < now - sensors.collision_time <= self.collision_stale_tol_s):
                self.issued = False   # resume = re-plan the held move from the current (held) pose
                print('COLLISION_STALE_PAUSE phase=%s now=%.3f %s (max_age %.3f)' % (
                    self.phase, now, why[0], self.collision_stale_tol_s), flush=True)
                return self._result()   # running, no new command -> arm holds pose (zero velocity)
            print('CONTROLLER_INVARIANT_TRIP phase=%s now=%.3f %s' % (self.phase, now, ' '.join(why)),
                  flush=True)
            return self._fail('base moving or stale/missing feedback/collision world [' + ','.join(why) + ']')
        if self.held and self.phase != 'release' and not sensors.contact:
            return self._fail('lost grasp')
        # fm425 (run-322 cold): WALL-clock contact fail-fast. While pressing for contact (phase 'contact',
        # not yet latched), if no progress (enter-contact / CONTACT_STEP advance) for longer than
        # contact_stuck_wall_s WALL, FAIL FAST with a named reason -- before the 90 s host arm-command
        # heartbeat watchdog can trip opaquely. Off (0) or outside the contact phase => inert.
        if (self.contact_stuck_wall_s > 0 and self.phase == 'contact' and not sensors.contact
                and self._contact_progress_wall is not None):
            stalled = self.steady_clock() - self._contact_progress_wall
            if stalled > self.contact_stuck_wall_s:
                return self._fail('contact stalled: no contact progress for %.1fs wall '
                                  '(> %.0fs budget; native suction did not latch and no SIM_ATTACH weld '
                                  'formed) -- failing fast before the arm-command heartbeat watchdog'
                                  % (stalled, self.contact_stuck_wall_s))
        try:
            if sensors.valid_until_steady is not None:
                require_fresh(sensors.valid_until_steady,self.steady_clock)
            return self._advance(sensors)
        except Exception as exc:
            return self._fail(f'adapter failure: {type(exc).__name__}: {exc}')

    def _advance(self, s):
        now = s.simulation_time
        if self.phase == 'observe':
            matches = [o for o in s.observations if o.location == self.source
                       and o.sku == self.job['sku'] and o.family == 'sealed_box'
                       and type(o.quantity) is int and o.quantity == 1 and math.isfinite(o.confidence)
                       and .9 <= o.confidence <= 1 and self._provenance(o, now)
                       and self._pose_ok(o.pose)]
            if len(matches) != 1:
                # DIAG (throttled): why no unique source match -- distinguishes a reach/provenance/
                # confidence drop from an empty observation set, since all observe early-returns look
                # identical ('observe','') in the status stream.
                self._obsn = getattr(self, '_obsn', 0) + 1
                if self._obsn % 20 == 1:
                    cand = [(o.location, o.sku, round(math.dist(o.pose[:3], (0, 0, 0)), 3),
                             o.source, round(o.confidence, 2)) for o in s.observations]
                    # per-criterion booleans for the first candidate so the real miss is unambiguous
                    crit = ''
                    if s.observations:
                        o = s.observations[0]
                        crit = (f' first[loc={o.location==self.source},sku={o.sku==self.job["sku"]},'
                                f'fam={o.family=="sealed_box"},qty={o.quantity==1},conf={.9<=o.confidence<=1},'
                                f'prov={self._provenance(o,now)},pose_ok={self._pose_ok(o.pose)},'
                                f'dt={round(now-o.simulation_time,4)},frame={o.frame==self.frame}]')
                    print(f'OBSERVE_GATE matches={len(matches)} src={self.source!r} sku={self.job["sku"]!r} '
                          f'max_reach={self.profile.max_reach} observations={cand}{crit}', flush=True)
                return self._result()  # bounded retry with fresh sensor frames
            target = s.target
            if not target or target.location != self.destination or not self._provenance(target, now):
                self._tgtn = getattr(self, '_tgtn', 0) + 1
                if self._tgtn % 20 == 1:
                    print(f'OBSERVE_GATE target_gate_fail target={"None" if not target else target.location!r} '
                          f'want_destination={self.destination!r}', flush=True)
                return self._result()
            if not target.clear:
                return self._fail('occupied target')
            if (not self._pose_ok(target.pose)
                    or not axis_fits(target.free_dimensions, self.profile.dimensions)):
                return self._fail('unreachable or undersized target')
            if s.contact:
                return self._fail('unexpected pre-existing gripper contact')
            source = matches[0]
            p = self.profile
            def pose(xyz, dz):
                return (*xyz[:2], xyz[2] + dz + p.tcp_surface_offset, *p.tcp_quaternion)
            contact = pose(source.pose, 0)
            place = pose(target.pose, p.dimensions[2])
            self.poses = dict(approach=pose(source.pose, p.approach), contact=contact,
                              lift=pose(source.pose, p.approach),
                              preplace=pose(target.pose, p.dimensions[2] + p.approach),
                              place=place, retreat=pose(target.pose, p.dimensions[2] + p.approach))
            if self.pose_policy is not None:
                self.poses = self.pose_policy.build(source, target, p)
            if self.cart_cage_place:
                # Re-chain the place for the open-face insert, keeping the policy's reachable PLACE
                # orientation + sensed deck position. PREPLACE = the deck place pushed out the +x mouth
                # by the mouth standoff (so cuMotion routes the carton clear of the walls to a pose fully
                # OUTSIDE the cage); the insert (place phase, straight-line) then drives -(axis) back in.
                # The place target itself is lifted cage_insert_gap above the deck so the carton drops
                # the last cm on release instead of being driven into the deck collider. retreat = the
                # same mouth standoff (back straight out, empty-handed). Orientation is unchanged.
                base = self.poses['place']
                a = self.cage_axis
                place = (base[0], base[1], base[2] + self.cage_insert_gap, *base[3:])
                mouth = (place[0] + a[0] * self.cage_mouth_standoff,
                         place[1] + a[1] * self.cage_mouth_standoff,
                         place[2] + a[2] * self.cage_mouth_standoff, *base[3:])
                self.poses['place'], self.poses['preplace'], self.poses['retreat'] = place, mouth, mouth
            # fm390: when the base repositions before placing, the PLACE-side poses (preplace/place/
            # retreat) are built in the PICKUP base frame here but the arm never reaches them from this
            # pose -- the deck is only reachable once the base has driven to place_approach (reach proof
            # scripts/check_place_approach_reach.py). Gate only the PICK-side poses now; the place-side
            # poses are re-registered in the place_approach frame and reach-gated there (_reregister_place).
            gate_poses = (self.poses if not self.reposition_before_place
                          else {k: v for k, v in self.poses.items() if k not in _PLACE_SIDE_PHASES})
            if not all(self._pose_ok(v) for v in gate_poses.values()):
                import math as _m
                bad = {k: round(_m.dist(v[:3], (0, 0, 0)), 3) for k, v in gate_poses.items()
                       if not self._pose_ok(v)}
                print('REACH_ENVELOPE_FAIL max_reach=%s bad=%s source=%s target=%s'
                      % (self.profile.max_reach, bad, tuple(round(c, 3) for c in source.pose[:3]),
                         tuple(round(c, 3) for c in target.pose[:3])), flush=True)
                return self._fail('approach outside reach envelope')
            self.target_pose = (*target.pose[:2], target.pose[2] + p.dimensions[2], *target.pose[3:])
            self.target_surface = target.pose
            self.source_pose = source.pose
            self._source_obs = source   # fm390: re-run the place policy against a fresh target after repositioning
            self.evidence.append(asdict(source))
            self._transition('approach')
        if self.phase in self.poses:
            # OBSERVE-THEN-EXECUTE lock (firstmate 2026-10-15, key approach-sensing-occlusion): once
            # observe has validated the source/target registration and cuMotion has planned the reach,
            # the moving arm occludes/degrades the FIXED source camera's view of the support it reaches
            # toward, so a per-tick fixed-camera re-validation cannot hold through approach/contact/lift
            # (it made the capture fail -> /sensors stall -> wall-backstop e-stop, report run-98). Lock
            # the validated registration for those phases and treat an absent/stale/conflicting
            # fixed-camera observation as EXPECTED occlusion, not a fault: progress is gated on measured
            # contact + joint feedback (the per-tick freshness/joint/contact/collision checks above) and
            # the executor's joint-tracking and heartbeat e-stops, which still stop a genuine fault.
            # Re-validate the placement target from the fixed camera again only once lift has cleared
            # the view -- at preplace, before the first placement move.
            if self.phase == 'preplace' and not self.issued:
                t = s.target
                if (not t or t.location != self.destination or not t.clear
                        or not self._provenance(t, now)
                        or math.dist(t.pose[:3], self.target_surface[:3]) > .02):
                    return self._fail('target clearance lost')
            if not self.issued:
                if s.valid_until_steady is not None:
                    require_fresh(s.valid_until_steady,self.steady_clock)
                # run-279: wrist_1-safe fixed-shape PREPLACE. The pose-IK preplace flips wrist_1 above the
                # ur_wrist_2-vs-ur_forearm self-collision stop (~0.70) and jams at execution (run-278:
                # wrist_1 stuck 0.69 vs cmd 1.99). When a CPU-IK-solved cage_preplace_shape is configured,
                # drive preplace as a fixed collision-planned begin_joint_goal to it (wrist_1 below the
                # stop; FK TCP = the sensed mouth); the straight-line insert (place) below still tracks the
                # SENSED place target. Carton-loaded -> slow fold. Absent => the pose-IK preplace, and the
                # poll/transition (preplace->place) + target re-validation above are shared unchanged.
                if (self.phase == 'preplace' and self.cart_cage_place
                        and self.cage_preplace_shape is not None):
                    # run-279: the ~1.88 rad shoulder_pan swing into the fixed shape settles ~0.176 rad
                    # short on joint0 ONLY (acceleration-mode drive on the high-inertia extended-arm yaw;
                    # NOT a contact -- armbench reached the shape, no mid-track-miss). Relax THIS move's
                    # terminal goal gate to the executor's 0.2 clamp ceiling so an otherwise-arrived shape
                    # (every other joint exact) is not rejected; the straight-line insert (place) then
                    # re-solves to the SENSED place and corrects the small standoff offset. Tuck/pan keep
                    # the tight gate (goal_tol=None); place keeps its own 0.12.
                    self.motion.begin_joint_goal(self.cage_preplace_shape, s.joint_positions,
                                                 slow=True, goal_tol=0.2)
                    self.issued = True
                    return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))
                # fm403: wrist_2-safe fixed-shape LIFT. The task-space lift IK can flip wrist_2 to the
                # negative self-colliding branch (w2 -> -1.25 crosses the ur_wrist_2-vs-ur_forearm stop;
                # runs 300d/305/311 jammed even with wrist_1 capped -- the bench showed w1 cleared but w2
                # stuck 0.38 short). Drive the lift as a fixed collision-planned begin_joint_goal to the
                # clear-region carry shape (w1<=0.65, w2>=0, bench-proven 10/10), exactly as
                # cage_preplace_shape fixed the preplace. Carton-loaded -> slow; the big shoulder_pan swing
                # into the shape settles a little short on joint0 only, so relax THIS move's terminal gate
                # (the subsequent tuck/transit re-anchor). Absent carry_lift_shape => legacy task-space lift.
                if self.phase == 'lift' and self.carry_lift_shape is not None:
                    self.motion.begin_joint_goal(self.carry_lift_shape, s.joint_positions,
                                                 slow=True, goal_tol=0.2)
                    self.issued = True
                    return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))
                straight_contact = self.phase == 'contact' and self.contact_straight_line
                # Cart-cage open-face place: the INSERT (place) and empty-handed RETREAT are straight-line
                # moves THROUGH the single open +x face (begin(straight_line=True): collision-off seeded
                # 2-point). The cage walls are published only so cuMotion routes the collision-aware
                # PREPLACE around them to the mouth standoff; the insert then deliberately drives into the
                # open mouth and back out, never crossing a wall (CPU-proven by the fold->insert sweep).
                straight = straight_contact or (self.cart_cage_place and self.phase in ('place', 'retreat'))
                target = self._overpressed_contact(self._contact_step) if straight_contact else self.poses[self.phase]
                # Per-move goal type (firstmate msg 284/285): the cart PLACE poses have no cuMotion
                # task-space IK solution (the solver is systemically broken here); route them through
                # the reliable seeded-KDL -> joint-space path. The pick (approach/lift) keeps its proven
                # task-space goal -- a global switch oscillated the pick's joint-space trajectory (run-226).
                joint_space = self.phase in ('preplace', 'place', 'retreat')
                # fm327 belt-and-braces: cap wrist_1 on the LIFT (the one task-space CARRY pose goal) so
                # cuMotion IKs it below the ur_wrist_2-vs-ur_forearm self-collision stop (~0.70); the
                # tuck/pan joint goals already use cart_tuck_shape wrist_1=0.65. None elsewhere (the pick
                # approach may need wrist_1 above the stop to reach the backstock).
                carry_cap = self.carry_wrist1_cap if self.phase == 'lift' else None
                # fm403: belt-and-braces wrist_2 lower bound on the task-space LIFT fallback (carry_lift_shape
                # unset) so the IK cannot pick the w2-negative self-colliding branch. None elsewhere.
                carry_w2min = self.carry_wrist2_min if self.phase == 'lift' else None
                # fm345 smaller-problems-later: the carton-loaded PLACE extend (preplace/place) is too
                # fast for the soft arm drive to track the cuMotion min-jerk timing (run-270 sagged
                # 0.567 rad short at rest). Slow those two moves to ~half and widen their terminal
                # goal-gate tolerance so a converging lag does not abort mid-place; every other move
                # (pick approach/contact/grasp/lift, and the empty-handed retreat) keeps full speed and
                # the tight gate. Retreat is post-release and empty, so it is NOT relaxed.
                place_move = self.phase in ('preplace', 'place')
                # fm347 (3): the PLACE moves seed the IK from the measured PAN config and branch to the
                # nearest config to it, so shoulder_pan stays on the pan branch (cart_pan_joint0) instead
                # of flipping to a far stow-nominal branch (run-273: ~3.9 rad base rotation stalled j0).
                self.motion.begin(target, s.joint_positions, straight_line=straight,
                                  joint_space=joint_space, wrist1_cap=carry_cap, wrist2_min=carry_w2min,
                                  slow=place_move, goal_tol=(0.12 if place_move else None),
                                  near_start=place_move)
                self.issued = True
                return self._result(({'kind': 'move', 'pose': self.poses[self.phase]},))
            if self.phase == 'contact' and not self._contact_suction:
                # Suction-grasp choreography: s.contact (NativePanelContact.read) is the surface gripper's
                # grip-LATCH status (Closed + one object), NOT a touch/force signal -- only ever True with
                # suction ON. Enable suction DURING the contact descent so the gripper LATCHES the instant
                # its seals reach the carton; that latch is the measured contact that terminates the move
                # (the seam below) and the grasp. The gripper command carries an acquisition deadline, so
                # wait for a fresh valid_until_steady (under the sensing gate the first ticks may be a bare
                # held frame with none yet; run-137 emitted it a tick too early) and require_fresh it,
                # exactly like the grasp phase -- then emit it ONCE.
                if self._contact_waits < 3 and s.valid_until_steady is None:
                    # Brief grace for a fresh acquisition deadline (held-capture under the gate); do NOT
                    # wait indefinitely (run-140 stalled 218 ticks with 0 suction when it never arrived).
                    self._contact_waits += 1
                    print(f'CONTACT_SUCTION_WAIT n={self._contact_waits} valid_until=None contact={s.contact}',
                          flush=True)
                    return self._result()
                if s.valid_until_steady is not None:
                    require_fresh(s.valid_until_steady, self.steady_clock)
                self._contact_suction = True
                print(f'CONTACT_SUCTION enable valid_until={s.valid_until_steady!r} '
                      f'waits={self._contact_waits} contact={s.contact}', flush=True)
                return self._result(({'kind': 'gripper', 'enabled': True},))
            try:
                outcome = self.motion.poll()
            except Exception as exc:  # noqa: BLE001 -- fm372: route EVERY non-collision raise to retry
                # fm371/fm372: the general move path had NO guard around poll() -- a stochastic cuMotion
                # seed that exhausts the plan budget (RuntimeError), yields a trajectory that fails validate
                # (ValueError, run-298 tuck's sibling), or surfaces a sim-deadline/goal-tolerance/track-miss/
                # feedback-lost/other raise all propagated UNCAUGHT and ended the mission. All are run-to-run
                # variance, so route through the SAME bounded fresh-seed retry (joint-limit hard-fails inside
                # the helper). A genuine collision returns 'failed' (handled below) -- NOT a raise -- and
                # still fails fast.
                return self._raised_move_exc(exc, staged=False)
            if outcome == 'failed':
                # fm367: a SIM-clock move/settle deadline miss (the port flags it; a collision/plan
                # rejection raises -> flag False) is a render-pacing transient, so re-issue the SAME
                # move with a fresh deadline up to the deadline-retry max before failing the phase.
                # This is the general move path (straight insert/retreat, free-space reach) that had no
                # retry; a genuine collision/plan-rejection still fails immediately (fail-fast preserved).
                if (getattr(self.motion, 'last_fault_was_sim_deadline', lambda: False)()
                        and self._move_deadline_retries < self._move_deadline_retry_max):
                    self._move_deadline_retries += 1
                    print('MOVE_DEADLINE_RETRY phase=%s attempt=%d/%d' %
                          (self.phase, self._move_deadline_retries, self._move_deadline_retry_max), flush=True)
                    self.issued = False   # re-issue this move (fresh deadline) next tick
                    return self._result()
                return self._fail('collision planning or execution failed')
            if outcome not in ('running', 'succeeded'):
                return self._fail('invalid motion outcome')
            if outcome == 'succeeded':
                # Contact-feedback descent (firstmate run-202): the step settled. If the seal has NOT
                # latched and the capped travel is not exhausted, step the contact pose a bit further
                # toward the carton and re-issue -- this drives to actual contact regardless of the
                # run-to-run residual variance a fixed over-press cannot absorb, and the per-step settle
                # gives the plugin a stationary pad to latch (suction stays on). Hitting the cap without
                # contact is a mission fault (the latch never formed); SEAL_DIAG has logged the gap.
                if (self.phase == 'contact' and self.contact_straight_line
                        and self.contact_overpress_step_m > 0 and not s.contact):
                    # Freeze-settle (firstmate run-203, stations verdict): the native plugin seals a
                    # STATIONARY pad every time, but the trajectory settle tolerance lets the link finish
                    # with non-zero velocity -- which does NOT seal. So when a step reaches its pose
                    # without a latch, HOLD (do not re-issue) for CONTACT_FREEZE_TICKS so the drive settles
                    # the link to ~0 velocity at the measured joints, giving the plugin (or the gated
                    # SIM_ATTACH_FALLBACK) its stationary settle window to latch. Only if it still has not
                    # latched after the settle do we step further toward the carton.
                    if self._contact_freeze < CONTACT_FREEZE_TICKS:
                        self._contact_freeze += 1
                        return self._result()
                    self._contact_freeze = 0
                    if self._contact_step + self.contact_overpress_step_m <= self.contact_overpress_cap_m + 1e-9:
                        self._contact_step += self.contact_overpress_step_m
                        self.issued = False
                        self._contact_progress_wall = self.steady_clock()   # fm425: a step advance IS progress
                        print(f'CONTACT_STEP advance step={self._contact_step*1000:.1f}mm '
                              f'cap={self.contact_overpress_cap_m*1000:.0f}mm contact={s.contact}', flush=True)
                        return self._result()
                    return self._fail('contact not reached within capped over-press travel')
                next_phase = dict(approach='contact', contact='grasp', lift='preplace',
                                  preplace='place', place='release',
                                  retreat='stow' if self.transport_joints is not None else 'verify')
                if self.staged_cart_swing:
                    # Insert the body-clear via points: lift -> tuck -> pan -> preplace, and route the
                    # empty-handed return retreat -> retreat_tuck -> stow (see __init__).
                    next_phase['lift'] = 'tuck'
                    next_phase['retreat'] = 'retreat_tuck'
                # fm410: the LIFT just completed -- it is the short physics probe. Before advancing to carry,
                # require the measured carton to have TRACKED the tool up by >= grasp_track_m from the
                # grasp-confirm baseline. A carton that stayed put (placed_pose flat) was never truly held, so
                # FAIL rather than fold a phantom load through the cart. Inert when grasp_track_m is unset or
                # no baseline/placed_pose was captured (real rig) -> legacy fm395 advance (byte-identical).
                if (self.phase == 'lift' and self.grasp_track_m and self._carton_z_at_grasp is not None):
                    if s.placed_pose is None:
                        return self._fail('grasp probe: carton pose unavailable to confirm the lift tracked')
                    risen = float(s.placed_pose[2]) - self._carton_z_at_grasp
                    print('GRASP_PROBE risen=%.3f need=%.3f carton_z0=%.3f carton_z=%.3f'
                          % (risen, self.grasp_track_m, self._carton_z_at_grasp, float(s.placed_pose[2])),
                          flush=True)
                    if risen < self.grasp_track_m:
                        return self._fail('grasp not confirmed: carton did not track the tool through the lift')
                self._transition(next_phase[self.phase])
                return self._result()
            # Contact-terminated grasp descent (firstmate key contact-move-termination): while the
            # contact move runs, raise the contact signal from MEASURED contact so trajectory_host
            # succeeds the goal holding the touch pose -- the gripper is physically against the carton
            # and cannot reach the commanded free-space pose (run-132 GOAL_MISS, stalled ~3.8deg short).
            # Every free-space move (approach/lift/preplace/place/retreat) keeps its exact pose-settle:
            # only 'contact' emits this, and only once measured contact is present.
            if self.phase == 'contact' and s.contact:
                return self._result(({'kind': 'contact'},))
            return self._result()
        if self.staged_cart_swing and self.phase in ('tuck', 'pan', 'retreat_tuck'):
            # Joint-space body-clear via points (firstmate 292). TUCK/retreat_tuck fold to the compact
            # high STOW shape at the arm's CURRENT shoulder-pan (minimal joint0 motion); PAN rotates
            # joint0 to the cart side (cart_pan_joint0) at that tucked radius. Each is a collision-planned
            # joint goal, mirroring 'stow'. The per-tick held/contact/parked/freshness gates above still
            # apply (tuck/pan carry the held carton; retreat_tuck is empty-handed).
            goal = ((self.cart_pan_joint0,) + tuple(self.cart_tuck_shape) if self.phase == 'pan'
                    else (s.joint_positions[0],) + tuple(self.cart_tuck_shape))
            if not self.issued:
                if s.valid_until_steady is not None:
                    require_fresh(s.valid_until_steady, self.steady_clock)
                # tuck/pan carry the welded carton -> plan the fold gently so the carton's inertia does
                # not displace the wrist off the body-clear plan (run-246); retreat_tuck is empty-handed.
                self.motion.begin_joint_goal(goal, s.joint_positions,
                                             slow=self.phase in ('tuck', 'pan'))
                self.issued = True
                return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))
            try:
                outcome = self.motion.poll()
            except Exception as exc:  # noqa: BLE001 -- fm372: route EVERY non-collision raise to retry
                # run-290: a staged move's cuMotion planning can EXHAUST its replan/seed budget on a bad
                # stochastic streak and RAISE ('planning failed (exhausted within wall cap)'); fm371: a
                # stochastic seed can ALSO yield a trajectory that fails validate (ValueError -- run-298's
                # tuck died here, UNCAUGHT); fm372: likewise a sim-deadline/goal-tolerance/track-miss/
                # feedback-lost/other raise. All are run-to-run variance, not an unreachable pose. Route
                # every such raise through the SAME bounded fresh-seed retry (joint-limit hard-fails inside
                # the helper); a returned 'failed' (plan->execute handoff transient, run-283) still retries
                # below on the same budget; a genuine collision/invalid outcome still fails fast.
                return self._raised_move_exc(exc, staged=True)
            if outcome == 'failed':
                # fm358 guard: a staged move failed once at the plan->execute handoff (run-283 pan) after
                # a normal plan -- a transient. Re-issue (fresh plan + deadline + budget) up to the
                # staged-move retry max before failing the stage. Does NOT loosen any tolerance.
                if self._staged_retries < self._staged_move_retry_max:
                    self._staged_retries += 1
                    print('STAGED_MOVE_RETRY phase=%s attempt=%d/%d' %
                          (self.phase, self._staged_retries, self._staged_move_retry_max), flush=True)
                    self.issued = False   # re-plan + re-execute this move next tick
                    return self._result()
                return self._fail('staged cart-swing %s move failed' % self.phase)
            if outcome == 'succeeded':
                nxt = {'tuck': 'pan', 'pan': 'preplace', 'retreat_tuck': 'stow'}[self.phase]
                if self.phase == 'tuck' and self.reposition_before_place:
                    nxt = 'transit'   # fm390: hold the carry fold; the base drives to place_approach next
                self._transition(nxt)
            elif outcome != 'running':
                return self._fail('invalid staged cart-swing motion outcome')
            return self._result()
        if self.phase == 'transit':
            # fm390: carton held in the carry (tuck) pose; the BASE must drive to place_approach in front
            # of the rollwagen mouth before the arm lowers onto the deck. HOLD (no arm move) -- the adapter
            # sees phase=='transit' and PAUSES stepping us while it drives the base, then sets
            # base_repositioned so TRANSIT -> preplace proceeds at the new pose. Because we are only ticked
            # while parked (at the pick pose, then at place_approach), the per-tick parked gate and the
            # per-phase elapsed timeout never see the base motion.
            if self.base_repositioned or s.base_repositioned:
                # fm400: the host controller runs in the HOST process, so the Isaac adapter cannot set our
                # in-process self.base_repositioned -- it signals arrival on the sensor packet
                # (s.base_repositioned) instead. The in-process fixture path (adapter steps us directly)
                # still uses self.base_repositioned. Either resumes TRANSIT -> preplace at the new pose.
                # fm390 PERCEIVE: re-read the live deck target in the NEW (place_approach) base frame and
                # rebuild the place-side poses from it before placing. The poses registered during observe
                # are in the stale PICKUP frame; placing from them would drive the arm to the wrong point.
                res = self._reregister_place(s, now)
                if res is not None:
                    return res
                self._transition('preplace')
            return self._result()
        if self.phase == 'grasp':
            if not self.issued:
                if s.valid_until_steady is not None:
                    require_fresh(s.valid_until_steady,self.steady_clock)
                self.issued = True
                self._grasp_confirm = 0
                self._grasp_waits = 0
                return self._result(({'kind': 'gripper', 'enabled': True},))
            if s.contact:
                # fm395 (A): count CONSECUTIVE confirmed-attachment ticks; advance only once the hold is
                # stable for grasp_confirm_ticks (never on a single transient weld/grip).
                self._grasp_confirm += 1
                if self._grasp_confirm >= self.grasp_confirm_ticks:
                    self.held = True
                    if not self.motion.update_payload(True, self.profile):
                        return self._fail('payload collision attachment not acknowledged')
                    # fm410: latch the measured carton z NOW (attachment confirmed, arm not yet lifted) as
                    # the lift-probe baseline; the lift->carry gate requires the carton to rise from here.
                    # None when the driver provides no placed_pose (real rig) -> the probe gate is inert.
                    self._carton_z_at_grasp = (float(s.placed_pose[2])
                                               if self.grasp_track_m and s.placed_pose is not None
                                               else None)
                    self._transition('lift')
                    return self._result()
            else:
                # A seal that formed then dropped before confirming is a failed attempt -> reset the
                # streak (never carry a transient toward the confirm threshold).
                self._grasp_confirm = 0
            # fm399: the confirm-wait budget is MONOTONIC in ticks since grasp-entry / last re-seat,
            # incremented EVERY tick regardless of contact. The old code reset it on every contact
            # tick, so a FLICKERING contact (True/False/True...) kept the consecutive _grasp_confirm
            # streak below the threshold AND kept resetting the wait -> the grasp could neither confirm
            # nor retry/fail and live-locked until the coarse phase_timeout (run-309; CPU-proven,
            # grasp-confirm-wait-livelock). Counting every tick makes a flicker retry or fail honestly
            # within the budget. Off by default (grasp_retry_max=0) => byte-identical; grasp_wait_ticks
            # (default 40) is >> grasp_confirm_ticks so a genuine stable hold still confirms first.
            if self.grasp_retry_max > 0:          # fm395 bounded grasp retry (off by default => byte-identical)
                self._grasp_waits += 1
                if self._grasp_waits >= self.grasp_wait_ticks:
                    if self._grasp_retries < self.grasp_retry_max:
                        self._grasp_retries += 1
                        print('GRASP_RETRY attempt=%d/%d (no confirmed carton attachment in %d ticks) '
                              '-- re-seating via contact' % (self._grasp_retries, self.grasp_retry_max,
                                                             self.grasp_wait_ticks), flush=True)
                        self._grasp_waits = 0
                        self._transition('contact')   # re-press / re-seat the pad; grasp re-fires on return
                        return self._result()
                    return self._fail('grasp did not confirm a held carton within the retry budget')
            return self._result()
        if self.phase == 'release':
            if not self.issued:
                if s.valid_until_steady is not None:
                    require_fresh(s.valid_until_steady,self.steady_clock)
                self.issued = True
                self.release_time = now
                return self._result(({'kind': 'gripper', 'enabled': False},))
            if not s.contact:
                self.held = False
                if not self.motion.update_payload(False, self.profile):
                    return self._fail('payload collision detachment not acknowledged')
                self._transition('retreat')
            return self._result()
        if self.phase == 'stow':
            if s.contact or self.held:
                return self._fail('cannot stow with unexpected held payload')
            if not self.issued:
                if s.valid_until_steady is not None:
                    require_fresh(s.valid_until_steady,self.steady_clock)
                self.motion.begin_joint_goal(self.transport_joints,s.joint_positions)
                self.issued=True
                return self._result()
            try:
                outcome=self.motion.poll()
            except Exception as exc:  # noqa: BLE001 -- fm372: route EVERY non-collision raise to retry
                # fm372: stow is the last collect move (begin_joint_goal) -- its poll() was unguarded, so a
                # stochastic plan-exhaust/validate/other raise here also killed the mission. Route the raise
                # through the SAME bounded staged retry (joint-limit hard-fails inside the helper); a
                # returned 'failed' below still fails fast as a collision.
                return self._raised_move_exc(exc, staged=True)
            if outcome == 'failed':
                # fm389/run-300e: a returned 'failed' at STOW was hard-failed here, UNLIKE the other staged
                # moves (tuck/pan/retreat_tuck) which bounded-retry a returned 'failed' (fm358). run-300e
                # reached stow (deepest ever -- lift/place/release/retreat all cleared) and the stochastic
                # move_group double-free crashed in its action-result handler RIGHT AFTER the stow plan
                # SUCCEEDED (0 collision/IK rejects, arm hit_bodies=[]), surfacing as this returned 'failed'
                # -> hard-fail, no retry. Route it through the SAME bounded staged retry so the crash-induced
                # transient re-plans on the respawned move_group; a genuine persistent stow fail still
                # exhausts -> _fail. Consistent with the tuck/pan returned-'failed' branch.
                if self._staged_retries < self._staged_move_retry_max:
                    self._staged_retries += 1
                    print('STAGED_MOVE_RETRY phase=%s attempt=%d/%d' %
                          (self.phase, self._staged_retries, self._staged_move_retry_max), flush=True)
                    self.issued = False   # re-plan + re-execute the stow next tick
                    return self._result()
                return self._fail('collision-planned transport stow failed')
            if outcome == 'succeeded':
                if len(s.joint_positions) != 6 or max(abs(a-b) for a,b in
                        zip(s.joint_positions,self.transport_joints)) > .01:
                    return self._fail('transport joint feedback disagrees with execution')
                self._transition('verify')
            elif outcome != 'running':
                return self._fail('invalid stow motion outcome')
            return self._result()
        if self.phase == 'verify':
            if s.contact:
                return self._fail('release contact returned')
            # fm389 DIRECT place verification (sim-only, active only when the driver threads a measured
            # carton pose). The collect destination is the robot's own staged cart, which no fixed camera
            # frames post-release (the 2-frame sensor verify below never gets a cart observation), so the
            # carton-on-deck physics is proven from the controlled-asset ground-truth carton pose instead:
            # the carton must rest on the SENSED support (centre within xy_tol, bottom within z_tol of the
            # support top) for two distinct frames. A carton that fell BELOW the support (run-300e: onto the
            # floor east of the cart) is a fast hard-fail so an empty-gripper 'success' cannot propagate.
            if s.placed_pose is not None:
                c = s.placed_pose
                half_h = self.profile.dimensions[2] / 2.0
                xy = math.dist(c[:2], self.target_surface[:2])
                bottom_gap = (c[2] - half_h) - self.target_surface[2]
                on_deck = xy <= self._place_verify_xy_tol and abs(bottom_gap) <= self._place_verify_z_tol
                # A settled carton clearly OFF the deck -- on the floor (bottom_gap << 0), on the upper
                # shelf (bottom_gap >> 0), or beside the cart (xy far) -- is a fast hard-fail so an
                # empty-gripper 'success' cannot propagate. The band between on_deck and far relies on the
                # phase timeout. Thresholds are 2x the accept tolerances (sim-only env-tunable).
                far = (xy > 2 * self._place_verify_xy_tol
                       or abs(bottom_gap) > 2 * self._place_verify_z_tol)
                if far:
                    self._place_offdeck_frames += 1
                    if self._place_offdeck_frames >= 3:
                        return self._fail('place verification failed: carton not resting on deck '
                                          '(xy=%.3f bottom_gap=%.3f)' % (xy, bottom_gap))
                else:
                    self._place_offdeck_frames = 0
                if on_deck:
                    self.verify_ids.add(('place_gt', round(now, 3)))
                    if len({t for _, t in self.verify_ids}) >= 2:
                        self.evidence.append({'source': 'ground_truth_place_check', 'location': self.destination,
                                              'simulation_time': float(now), 'carton_pose_station': tuple(c),
                                              'support_pose_station': tuple(self.target_surface),
                                              'xy_offset': float(xy), 'bottom_gap': float(bottom_gap)})
                        self._transition('succeeded')
                else:
                    self.verify_ids = {x for x in self.verify_ids if x[0] != 'place_gt'}
                return self._result()
            matches = [o for o in s.observations if o.location == self.destination
                       and o.sku == self.job['sku'] and type(o.quantity) is int and o.quantity == 1
                       and o.family == 'sealed_box' and math.isfinite(o.confidence)
                       and .9 <= o.confidence <= 1
                       and self._provenance(o, now) and self._pose_ok(o.pose)
                       and o.simulation_time > self.release_time
                       and math.dist(o.pose[:3], self.target_pose[:3]) <= .04]
            # Two distinct post-release frames, with source observed absent by
            # the adapter's complete ROI scan (documented wire contract).
            if (len(matches) == 1 and self.source in s.scanned_locations
                    and self.destination in s.scanned_locations
                    and self._fresh(s.scan_time, now) and s.scan_time > self.release_time
                    and not any(o.location == self.source for o in s.observations)):
                o = matches[0]
                if (o.observation_id not in {x[0] for x in self.verify_ids}
                        and o.simulation_time not in {x[1] for x in self.verify_ids}):
                    self.verify_ids.add((o.observation_id, o.simulation_time))
                    self.evidence.append(asdict(o))
                if len(self.verify_ids) >= 2:
                    self._transition('succeeded')
            else:
                self.verify_ids.clear()
            return self._result()
        return self._result()
