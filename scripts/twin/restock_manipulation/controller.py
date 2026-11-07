"""Bounded transfer state machine; all runtime evidence must originate in sensors."""
from dataclasses import asdict
import math
import os
import time
from ..restock_camera.freshness import require_fresh
from ..restock.physics_step import (COLLISION_RELAY_LEAD_S, collision_band_floor, dt_from_env,
                                     freshness_scale_from_env)

from . import esdf_boot_gate
from . import ground_truth_collect as gtc
from . import reactive_arm
from .arm_backstop import ArmBackstopParams
from .arm_human_gate import ArmGateConfig
from .fit import axis_fits
from .transfer_carry import (DEFAULT_WRIST1_CAP, DEFAULT_WRIST2_MIN, GUARDED_PHASES, WELD, WRIST_GUARD,
                             carry_flags, resolve_reach_guard)
from .transfer_mode import LIVE_POSE, near_face_poses, transfer_mode
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
# sim-robust-phase-timeout-during-recovery-v1: default bound (SIM seconds) on the phase-clock credit a
# named move_group crash recovery may accrue in one phase. Generous on purpose -- it only has to cover a
# real multi-death recovery (respawn2b's two deaths credited ~119 sim-s; the lifetime respawn budget is a
# few deaths), and the adapter's own recovery bounds fail fast by name long before it in a genuine hang.
# It is the belt-and-braces terminator for a pathological probe-stuck-True, so a hung recovery still ends
# by name here instead of pausing forever. Env RESTOCK_RECOVERY_PAUSE_CAP_S overrides; 0 disables.
RECOVERY_PAUSE_CAP_S = 300.0
# sim-collect-retreat-collider-v3: the collect deck-place empty-retreat lift-off is driven with
# ur_shoulder_pan LOCKED at the seed, so a straight-up raise never swings the gripper into the cart
# cage. This tolerance only guards against a future liftoff_goal change silently returning a
# pan-swinging branch (the lock itself holds pan to the seed); a few mrad of DLS residual is fine.
COLLECT_RETREAT_LIFTOFF_PAN_TOL = 0.02   # rad
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
                 collect_place_preplace_shape=None, collect_place_wrist_guard: bool = False,
                 collect_place_straight_line: bool = False,
                 collect_retreat_liftoff: bool = True, collect_retreat_liftoff_dz: float = 0.15,
                 grasp_track_m=None, require_collect_source_evidence: bool = False,
                 transfer_ground_truth: bool = False, ground_truth_pick: bool = False,
                 collect_ground_truth: bool = False, grasp_press_timeout_s: float = 0.0,
                 release_detach_timeout_s: float = 0.0, transfer_preplace_reach_guard=None,
                 transfer_pick_wrist_guard: bool = False,
                 transfer_service_carry: bool = False,
                 esdf_boot_ack_grace_s: float = 0.0,
                 preplace_convergence_stall_s: float = 0.0, preplace_wrist_first: bool = False,
                 preplace_pan_first: bool = False,
                 preplace_drift_recapture_rad: float = 0.0, preplace_drift_recapture_max: int = 0,
                 carton_level=None, fk_urdf=None, place_tcp_link='gripper_pad_tcp',
                 station_pitch_hold: bool = False, station_hold_roll_deg: float = 0.0,
                 station_hold_pitch_deg: float = 0.0,
                 reactive_arm_human: bool = False, reactive_arm_config=None,
                 reactive_arm_replan: bool = False, reactive_arm_replan_config=None,
                 contact_reissue_suction_each_tick: bool = False,
                 contact_require_suction_enabled: bool = False,
                 contact_reissue_close_once: bool = False,
                 contact_accept_seal_diag: bool = False,
                 contact_seal_diag_dwell_ticks: int = 5,
                 contact_seal_diag_max_gap_mm: float = 0.5,
                 contact_seal_diag_max_angle_deg: float = 1.0,
                 contact_status_max_age_s: float = 0.0):
        self.steady_clock=steady_clock
        # sim-carton-level-place-on-cart-v1: the FK link the collect place-on-cart reach targets -- the
        # motion port's IK tcp ('gripper_pad_tcp'), NOT ur_tool0 (~0.27 m apart). The level place IK must
        # target this frame for POSITION so the leveled standoff/set-down reach the sensed deck pose; its
        # rotation equals ur_tool0's (rigid translation), so the carton attitude (leveling) is identical.
        self._place_tcp_link = str(place_tcp_link)
        # collect-gt (demo, opt-in, default off): GROUND-TRUTH COLLECT. Accept the collect SOURCE observation
        # labelled 'ground_truth' (the controlled carton asset's simulated pose, served Isaac-side when the
        # source camera refuses every frame) and let it stand in for the fm449 clean-camera gate, recording
        # collect_mode='ground_truth' in the evidence + run log. Enabled by this arg OR the shared env
        # RESTOCK_COLLECT_GROUND_TRUTH (the host process inherits the orchestrate run env). Off => unchanged.
        self.collect_ground_truth = bool(collect_ground_truth) or gtc.enabled()
        # fm390: drive the BASE in front of the rollwagen mouth between the carry fold and the place,
        # instead of reaching the place around from the collect station pose. When set (with the staged
        # cart swing), the carry fold goes lift -> tuck -> TRANSIT (hold, emit one reposition_base control)
        # and the arm waits -- the adapter pauses stepping us, drives the base to place_approach, then sets
        # base_repositioned so TRANSIT -> preplace proceeds at the new pose. Default off => byte-identical.
        self.reposition_before_place = bool(reposition_before_place)
        # sim-service-leg-carry-v1 (opt-in, default off => byte-identical): the shelf-staging TRANSFER
        # service leg picks the carton off the parked cart at the staging pose and must CARRY it to the
        # endcap before placing. That needs the TRANSIT carry-hold handshake (lift -> tuck -> TRANSIT, the
        # arm holds the level fold while the runtime drives the base, then re-registers + places at the
        # endcap) -- the SAME fm390 mechanic the collect place-on-cart uses, but for a live_pose transfer
        # (which otherwise goes tuck -> preplace with no carry-hold). When set, _repositions_before_place()
        # is True for a live_pose transfer so the carry-hold is produced; the runtime drives the carry.
        self.transfer_service_carry = bool(transfer_service_carry)
        self.base_repositioned = False
        self._source_obs = None   # fm390: the matched source observation, for re-running the place policy
        # Opt-in SIM transfer workaround (env RESTOCK_TRANSFER_MODE, twin.restock_manipulation.transfer_mode,
        # read in BOTH processes). 'live_pose': near-face pick/place yaw, every station reach-gated at observe,
        # lift -> tuck -> preplace (no TRANSIT hold, no collect-cart pan). Unset => None => byte-identical.
        self.transfer_mode = transfer_mode()
        self.near_face = None     # live_pose: which yaw representatives the transfer observe chose
        # Opt-in SIM transfer CARRY flags (env RESTOCK_TRANSFER_CARRY, twin.restock_manipulation.transfer_carry,
        # read in BOTH processes). 'wrist_guard': the transfer place moves keep the KDL IK goal in the clear
        # wrist branch (warm2 preplace jam). Unset => empty => byte-identical.
        self.transfer_carry = carry_flags()
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
        # sim-physics-step-collision-band-v1: per-FRAME sensor-freshness windows (the collision/ESDF
        # age max_age, the one-to-few-step render_lead, and this held-collision pause tolerance) encode a
        # FRAME-COUNT budget -- "tolerate up to N un-refreshed frames of ESDF/feedback staleness" -- and
        # the sim-seconds value is N*dt. nvblox integrates rendered frames and rendering_dt tracks
        # physics_dt, so one frame = dt sim-seconds; at a RAISED step the same frame-count staleness
        # spans more sim time (rtfcollect: a ~55-frame warming gap = collision_dt 1.5278 s at 1/36, which
        # is 0.92 s and inside the 1.5 s band at 1/60). Rebase each window onto the running step so the
        # FRAME budget is held constant (NOT a widening -- identical frame margin, re-expressed on the
        # running clock; the collect/transfer scene is parked/static so an older packet is strictly
        # consistent, see max_age fm476). Factor is exactly 1.0 at 1/60 => byte-identical. The step is
        # read from RESTOCK_PHYSICS_DT (exported by the launcher; see twin.restock.physics_step).
        self._freshness_scale = freshness_scale_from_env()
        self.collision_stale_tol_s = (_float_env_bounded('RESTOCK_COLLISION_STALE_TOL_S', 0.8, 0.0, 5.0)
                                      * self._freshness_scale)
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
        # fm449 (run-332 cold collect): GATE the observe->approach reach on a CLEAN, committed collect-source
        # capture AT PARK. Cold collect failed at the collect capture because the grasp/reach began before the
        # fixed source camera had a clean, synchronized evidence frame for the calibrated source slot -- the
        # descending gripper then occluded the carton top, so the undecoded-label hold ('unknown or undecoded
        # label in calibrated slot') and the sync gates had no prior. matches==1 in observe proves a DECODED
        # label but can be served from a HELD/seeded registration (fm443 seed, or an occluded footprint), so
        # the reach must NOT begin on that prior. When this flag is on, observe will not advance to 'approach'
        # until an injected collect-source evidence provider reports -- this tick, at park -- BOTH a clean
        # SYNCHRONIZED source frame AND a decoded label for the calibrated slot, keyed on the SAME clean/
        # occluded signal the camera lane computes (a non-suspended, non-occluded SourceTarget; see
        # restock_camera.source_target.build_collect_source_evidence). No clean capture within
        # collect_source_evidence_wait_ticks fails fast with the named reason 'collect_source_evidence_unavailable'
        # rather than reaching blind. Default OFF (flag False, provider never set) => the gate never arms =>
        # byte-identical. Scoped to collect (the source is a fixed-camera station; transfer's source is the
        # controlled-asset cart, not camera-sensed).
        self.require_collect_source_evidence = bool(require_collect_source_evidence)
        # fm461 (demo, opt-in): accept a GROUND-TRUTH transfer destination target (the authored bay pose,
        # source 'ground_truth') when the shelf station is NOT camera-commissioned, and verify the placement
        # from the sim-measured carton pose vs that bay target (the existing ground_truth_place_check path),
        # labelled ground_truth in the evidence + run log. Off (default) => only sensor-sourced transfer
        # targets + the camera 2-frame verify, byte-identical.
        self.transfer_ground_truth = bool(transfer_ground_truth)
        # fm462 (demo, opt-in): LABEL the transfer PICK as ground-truth. NOTE: the transfer cart-source
        # observation is ALREADY a controlled-asset read (isaac_factory._cart_pose_station -- the cart's
        # live measured deck pose, NOT camera-sensed stock), so this does not change WHERE the arm picks;
        # it records that the pick target is ground-truth (source 'ground_truth_pick') in the evidence +
        # run log so the demo is labeled honestly. It does NOT bypass the station-ACK/masked-depth pump
        # (StationAckStreamStalled) -- that is a publisher process-death, fixed on the sensing side, not here.
        self.ground_truth_pick = bool(ground_truth_pick)
        self.collect_source_evidence_wait_ticks = _int_env('RESTOCK_COLLECT_SOURCE_EVIDENCE_WAIT_TICKS', 40, 1, 600)
        self._collect_source_evidence = None   # injected callable(now) -> bool: clean source capture this tick
        self._collect_source_evidence_waits = 0
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
        # SIM-CLOCK pre-weld press budget (grasp_press_timeout_s, SIM seconds; default 0 => off =>
        # byte-identical). The honest collect presses for a seal across the contact descent + the grasp
        # confirm/retry loop; when the native suction never latches and no weld forms, the grasp phase
        # returns bare controls every tick (grasp_retry_max=0) and the only backstop is the COARSE sim
        # phase_timeout or -- under render pacing, where wall >> sim -- the opaque 90 s WALL arm-command
        # heartbeat watchdog (runs 321/327 died 'arm command heartbeat lost' at collect). This is a bound
        # on the WHOLE unconfirmed press, measured on the SIM clock (render-pacing-safe, unlike the WALL
        # contact_stuck_wall_s above; see 'Freshness on sim clock, not wall'): it FAILS FAST with a named
        # reason before that watchdog, and because retries never re-arm it, a press cannot loop forever.
        # Clamped to (0, phase_timeout] so it is a TIGHTER bound than the coarse per-phase timeout.
        if not (isinstance(grasp_press_timeout_s, (int, float))
                and math.isfinite(grasp_press_timeout_s) and grasp_press_timeout_s >= 0):
            raise ValueError('grasp_press_timeout_s must be a finite, non-negative number of sim seconds')
        self.grasp_press_timeout_s = min(float(grasp_press_timeout_s), float(phase_timeout))
        self._press_sim_start = None   # sim time the current unconfirmed pre-weld press began (see step)
        # sim-robust-cart-place-descent-v1: SIM-CLOCK release-detach budget (release_detach_timeout_s, SIM
        # seconds; default 0 => off => rely on the coarse phase_timeout, byte-identical). After the place
        # descent the RELEASE emits a gripper-open and waits for the measured contact to clear (the
        # SIM_ATTACH weld / suction to let go) before advancing to retreat/stow. If the weld fails to
        # detach, the carton would be carried back UP on retreat/stow -- the opposite of 'carton left on the
        # deck'. Without a tight bound the release just loops 'running' to the coarse phase_timeout; this
        # FAILS FAST with a NAMED reason (SIM-clock, render-pacing-safe; see 'Freshness on sim clock, not
        # wall'). Clamped to (0, phase_timeout] so it is a tighter bound than the per-phase timeout.
        if not (isinstance(release_detach_timeout_s, (int, float))
                and math.isfinite(release_detach_timeout_s) and release_detach_timeout_s >= 0):
            raise ValueError('release_detach_timeout_s must be a finite, non-negative number of sim seconds')
        self.release_detach_timeout_s = (min(float(release_detach_timeout_s), float(phase_timeout))
                                         if release_detach_timeout_s > 0 else 0.0)
        # sim-robust-esdf-boot-gate-v1: opt-in HOST key esdf_boot_ack_grace_s (SIM seconds; default 0 =>
        # off => byte-identical). Names a never-acknowledged nvblox ESDF incorporation at boot, on the sim
        # clock, BEFORE the first arm motion streams -- rather than letting the collision_dt invariant
        # discover it ~40 s in with an opaque reason (meshuris1: collision_time stuck at the -1 not-ready
        # sentinel from tick 1, invariant tripped collision_dt=40 at the first observe step now=39). The
        # gate HOLDs the op (no motion) until the first valid incorporation ack (collision_time leaves -1)
        # or the grace elapses, then fails fast by name. Once a valid ack has arrived it disarms
        # permanently and the collision_dt invariant below is the sole owner (its band untouched). See
        # esdf_boot_gate.EsdfBootGate.
        if not (isinstance(esdf_boot_ack_grace_s, (int, float)) and math.isfinite(esdf_boot_ack_grace_s)
                and esdf_boot_ack_grace_s >= 0):
            raise ValueError('esdf_boot_ack_grace_s must be a finite, non-negative number of sim seconds')
        self.esdf_boot_ack_grace_s = float(esdf_boot_ack_grace_s)
        self._esdf_boot_gate = esdf_boot_gate.EsdfBootGate(self.esdf_boot_ack_grace_s)
        # sim-arm-reactive-replan-v1 (P1): opt-in human-aware RUN/SLOW/STOP/BACKSTOP authority for the
        # free-space arm legs. The ported humangate decision core (arm_human_gate) is fed by the vendored
        # scripts/twin/humans evaluator (an injected detection provider) and drives the e-stop-free preempt
        # (motion.cancel) + a reversible ArmCommandGate pause (an injected sink). Off (reactive_arm_human
        # False / not constructed) => the supervisor is None and step() skips it entirely => byte-identical.
        self._reactive_arm_slow = False          # SLOW: OR'd into the in-scope issue sites' slow= flag
        self._arm_pause_sink = None              # injected: ArmCommandGate.set_paused (reversible)
        self._arm_detection_provider = None      # injected: now_wall -> vendored SafetyState | None
        self._reactive_arm = None
        if reactive_arm_human or reactive_arm_replan:
            cfg = dict(reactive_arm_config or {})
            gate_cfg = ArmGateConfig(**{k: cfg[k] for k in ArmGateConfig.__dataclass_fields__
                                        if k in cfg}).validate()
            bs_cfg = cfg.get('backstop') if isinstance(cfg.get('backstop'), dict) else {}
            bs_params = ArmBackstopParams(**{k: bs_cfg[k] for k in ArmBackstopParams.__dataclass_fields__
                                             if k in bs_cfg}).validate()
            phases = cfg.get('replannable_phases')
            # PR-B: execution-time corridor-watch re-plan. Build the pure ReactiveReplanController +
            # a PerceptionCollisionGate (reusing the sibling perceived-clearance logic) from config;
            # the ESDF source + corridor builder are injected Isaac-side at boot (set_arm_clearance_source),
            # so replan stays inert until they are wired.
            replan_ctrl = collision_gate = None
            if reactive_arm_replan:
                from . import reactive_replan as _rr
                from .collision_gate import PerceptionCollisionGate as _PCG
                rcfg = dict(reactive_arm_replan_config or {})
                replan_ctrl = _rr.ReactiveReplanController(
                    min_replan_interval_s=float(rcfg.get('min_replan_interval_s', 0.5)),
                    max_replans=int(rcfg.get('max_replans', 3)))
                collision_gate = _PCG(margin_m=float(rcfg.get('margin_m', 0.05)),
                                      mode='refuse', unknown_action='hold',
                                      phases=(frozenset(phases) if phases
                                              else reactive_arm.DEFAULT_REPLANNABLE_PHASES),
                                      samples=int(rcfg.get('samples', 6)),
                                      max_snapshot_age_s=float(rcfg.get('max_snapshot_age_s', 0.5)))
            self._reactive_arm = reactive_arm.ReactiveArmSupervisor(
                gate_config=gate_cfg, backstop_params=bs_params,
                replannable_phases=(frozenset(phases) if phases else None),
                enable_backstop=bool(cfg.get('enable_backstop', True)),
                enable_human=bool(reactive_arm_human),
                collision_gate=collision_gate, replan_controller=replan_ctrl)
        # sim-carton-level-carry-v1: keep the welded carton PARALLEL TO THE GROUND through every carried
        # move (captain's rule; coldwin16 swung it ~150 deg off its weld attitude during the fold). A
        # resolved carton_level.LevelConfig (or None => off => byte-identical) plus the FK URDF path lets
        # the carried fold goals be LEVELED (level_goals) and a carried carton that leaves level by more
        # than the bound be NAMED (CARTON_TILT) and FAILED (trip) rather than silently shown. The carton
        # rests level on the shelf at the weld, so its up-axis is captured from the first carried pose.
        self._carton_level = carton_level           # carton_level.LevelConfig | None
        self._fk_urdf = fk_urdf                      # URDF path for the Isaac-free FK, or None
        self._fk_robot = None                        # lazily parsed Robot (first carried step)
        self._carton_up_in_tool = None               # the level carton up-axis in the tool frame, captured at the weld
        self._carton_up_in_tool_world = None         # sim-preplace-leg2-level-v1: world-consistent up-axis (hold on)
        # sim-carton-tilt-transient-debounce-v1: a SUSTAINED-tilt debounce on the CARTON_TILT trip. The
        # measured carton tilt can spike over the bound for one-to-few ticks on a tracking/impulse
        # transient (collectacc1: the LIFT re-plan restart after a move_group respawn) while the planned
        # path is level -- that spike is reported (CARTON_TILT_TRANSIENT) but must NOT abort; only a tilt
        # that stays over the bound for the whole grace window trips. Built lazily on the first carried
        # tick (needs the resolved bound + the physics-step-scaled grace). None when carton_level is off.
        self._tilt_debounce = None
        # sim-carton-level-preplace-v2: per-carried-segment PLANNED-vs-EXECUTED max carton tilt, so a
        # bench-vs-GPU gap (collectlevel3: bench predicted ~0 at the place set-down, GPU executed 2.22)
        # is NAMED in the log next run instead of inferred. ``_carton_seg_exec[phase]`` accumulates the
        # measured (executed) max |roll|,|pitch| each carried tick; ``_carton_seg_planned[phase]`` is the
        # planned peak stashed when the move is issued (the leveled-path predict, or the straight-line
        # place-descent commanded-path replay). Logged as CARTON_SEGMENT_TILT at each carried phase exit.
        self._carton_seg_exec = {}
        self._carton_seg_planned = {}
        # sim-preplace-wrist-continuity-v1: the max per-waypoint wrist step of the planned preplace approach
        # (keyed by phase, for the CARTON_SEGMENT_TILT report), and the reaction-natural branch report.
        self._carton_seg_wrist_step = {}
        self._preplace_branch_report = None
        # sim-preplace-select-live-cost-v1: per-operation memo of the reaction-branch standoff select
        # (key -> (sol, report)). The select is a pure function of its (static) inputs, so a preplace
        # StageRetry / re-begin reuses the first result instead of re-running the ~1.5 s CPU search.
        self._preplace_branch_cache = None
        # sim-preplace-branch-select-bounded-v1: set to a reason string by _preplace_above_place_goal when
        # the reaction-branch standoff search exceeds its CPU time-box, so the preplace step REFUSES before
        # motion (instead of starving the host /manipulation/status heartbeat past the Isaac watchdog).
        self._preplace_branch_refuse = None
        # sim-preplace-sweep-tilt-v1: per-tick CARTON_TICK diagnostic for the carried phases (the welded
        # carton attitude the CARTON_TILT grader reads + the EXECUTED joint positions), throttled to
        # ~10 Hz of SIM time. coldwin50/51/52 tripped CARTON_TILT at the preplace sweep with NO per-tick
        # carton-attitude / executed-joint feedback logged, so the dynamic-swing mechanism had to be
        # inferred; this makes the next GPU run (coldwin53) diagnosable directly. None => never logged yet.
        self._carton_tick_log_t = None
        # fm427 (A1) (run-323 cold lift): the fm419(a) COLLISION_STALE_PAUSE holds the post-attach lift on
        # a genuinely-stale collision stamp (run-323: collision_dt 0.52->0.65 s, the benign weld/ESDF
        # re-incorporation lag) by returning EMPTY controls -- a SILENT hold. Under render pacing that
        # silent hold starved the 90 s host arm-command heartbeat watchdog. Two fixes: (1) during the
        # pause re-command a HOLD at the CURRENT measured joints each tick (stationary -- never advances
        # the held move into the stale collision world) so the arm-command stream stays alive and the
        # heartbeat is fed; (2) bound the pause in WALL time -- a pause that persists beyond
        # collision_pause_wall_s fails fast with a named reason instead of silently starving the watchdog.
        # 0 s (default) => the legacy empty-control pause with no wall bound (byte-identical).
        self.collision_pause_wall_s = _float_env_bounded('RESTOCK_COLLISION_PAUSE_WALL_S', 0.0, 0.0, 85.0)
        self._pause_wall_start = None        # wall time the current collision-stale pause began
        # fm430 (2a) fallback (run-323/324 cold lift): the post-attach LIFT drives to carry_lift_shape, a
        # FIXED, bench-proven clear-region joint goal that does NOT depend on the live collision world. If
        # the ESDF re-incorporation lags after the SIM_ATTACH weld (collision stamp briefly stale), the
        # fm419(a) pause HOLDS the lift until the stamp freshens -- but if it never freshens (run-323:
        # collision_dt grew 0.52->0.65) the lift stalls. This OPT-IN grace lets ONLY that specific
        # pre-validated lift RIDE THROUGH a stale collision world (proceed, not pause) for a bounded WALL
        # window, then FAIL FAST with a named reason. Scoped tightly: held + phase 'lift' + carry_lift_shape
        # set + collision-sole staleness. 0 s (default) => off => the fm419(a) pause (byte-identical). This
        # is the controller-side fallback to the host ESDF re-integration kick (restock_host).
        self.lift_collision_grace_wall_s = _float_env_bounded('RESTOCK_LIFT_COLLISION_GRACE_WALL_S', 0.0, 0.0, 85.0)
        self._lift_grace_wall_start = None   # wall time the current held-lift collision grace began
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
        # sim-pick-contact-seal-fallback-v1 (coldwin63 pick never sealed): the opt-in contact guards. All
        # default off => byte-identical. (1) reissue_suction_each_tick: re-publish the gripper-enable control
        # on EVERY contact-press tick until the seal forms, so the one-shot enable can't be lost to the
        # last-value /manipulation/status relay overwrite that dropped cw63's close command (control_ipc
        # keeps only the latest status per topic; a later heartbeat overwrote the control-bearing one before
        # IsaacRosGraph.poll read it). (2) require_suction_enabled: fail fast CONTACT_SUCTION_NOT_ENABLED at
        # the first over-press step if the plugin never reported enabled (or, with reissue_close_once, re-send
        # the close once then fail on a second miss) -- instead of pressing the whole 20 mm cap with no chance
        # of a latch. (3) accept_seal_diag: when the geometric SEAL_DIAG shows the pad flush on a body under
        # the pad for a dwell AND the plugin is enabled, treat it as contact (drives the same contact-
        # termination + gated SIM_ATTACH_FALLBACK path the plugin latch drives). See docs/examples/
        # pick-contact-seal-fallback-fragment.json.
        self.contact_reissue_suction_each_tick = bool(contact_reissue_suction_each_tick)
        self.contact_require_suction_enabled = bool(contact_require_suction_enabled)
        self.contact_reissue_close_once = bool(contact_reissue_close_once)
        self.contact_accept_seal_diag = bool(contact_accept_seal_diag)
        self.contact_seal_diag_dwell_ticks = max(1, int(contact_seal_diag_dwell_ticks))
        self.contact_seal_diag_max_gap_mm = float(contact_seal_diag_max_gap_mm)
        self.contact_seal_diag_max_angle_deg = float(contact_seal_diag_max_angle_deg)
        # sim-gripper-status-relay-v1: an OPTIONAL extra staleness bound (sim seconds) on the gripper status
        # the contact_require_suction_enabled guard reads. <=0 => presence-only (a packet that carries a
        # gripper_status_time is authoritative; one that does not is 'no status'). >0 also treats a present
        # but older-than-bound status as stale. The guard NEVER aborts on staleness alone (it logs
        # CONTACT_STATUS_STALE and continues the bounded over-press); it aborts only on a FRESH 'not enabled'.
        self.contact_status_max_age_s = float(contact_status_max_age_s)
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
        # sim-robust-cart-place-goal-v1: the COLLECT place-on-cart counterpart of cage_preplace_shape,
        # for the NON-cage reposition place (reposition_before_place + cart_cage_place False -- the base
        # drives in front of the rollwagen mouth, so there is no cage insert). The deck place/preplace
        # poses have TWO collision-free IK families: an UPRIGHT one (shoulder_lift ~ -1.9) and a DEEP fold
        # (~ -3.57). The host KDL/cuMotion nearest-branch seeds from the folded carry config and lands
        # EITHER in the deep fold -- which the soft arm cannot HOLD under the carton load (collect1:
        # shoulder_lift sagged 0.39 rad short of -3.596 at steady state -> PATH_TOLERANCE_VIOLATED ->
        # arm-command heartbeat lost) -- OR in a wrist self-colliding branch (collectsrc: 'Invalid c-space
        # position: self collision detected', the 12-replan storm). Driving the preplace to this FIXED,
        # CPU-validated UPRIGHT clear-region 6-joint config (begin_joint_goal, like cage_preplace_shape)
        # fixes the GOAL branch: FK TCP ~ the sensed preplace standoff, in the low-droop upright branch,
        # so the subsequent place (pose-IK, near_start-seeded from here, collect_place_wrist_guard on)
        # stays upright + clear and tracks the SENSED deck target. None => pose-IK preplace (byte-identical).
        # Validated clear-region + reach margin by tests/test_restock_cart_place_goal.py and recorded in
        # data/sim-robust-cart-place-goal-v1/station-contract.md.
        self.collect_place_preplace_shape = (tuple(float(v) for v in collect_place_preplace_shape)
                                             if collect_place_preplace_shape is not None else None)
        if self.collect_place_preplace_shape is not None and (len(self.collect_place_preplace_shape) != 6
                or not all(map(math.isfinite, self.collect_place_preplace_shape))):
            raise ValueError('collect_place_preplace_shape must be 6 finite joint values')
        # sim-robust-cart-place-goal-v1: extend the transfer wrist-guard clear-branch selection
        # (in_clear_region / wrist_flip_seed, transfer_carry.py) to the COLLECT place moves
        # (place/retreat pose-IK; the preplace is the fixed shape above). Off (default) => the collect
        # place pose-IK is unguarded => byte-identical. See _place_wrist_guard.
        self.collect_place_wrist_guard = bool(collect_place_wrist_guard)
        # sim-robust-collect-place-straightline-v1 (collect4, 2026-10-23): route the NON-cage COLLECT
        # deck PLACE descent + empty RETREAT as the collision-OFF straight-line seeded insert
        # (begin(straight_line=True)) -- the SAME mechanism the cart-cage place and the live_pose transfer
        # place (PR #215) already use. The collect place lowers the HELD carton ONTO the cart deck; the
        # carton is a cuMotion attached_object (ros_host attaches a 34-sphere cuboid at gripper_tcp), and
        # the attached_object is NOT self-collision-ignored against the forearm / wrist / upper-arm, so the
        # set-down fold brings the carton spheres within the attached-object self-collision margin and
        # cuMotion rejects a collision-AWARE plan to the place config ('Invalid c-space position: self
        # collision detected' -> trajopt UNKNOWN_STATUS -> MANIP_HOST_FAILED op=collect), EVEN THOUGH the
        # bare arm is self-clear there and PR #209's wrist-guard accepted the config (collect4 host-stack.log:
        # 6 self-collisions, 0 world collisions; bare-arm worst self-pair +0.0339 m at BOTH the upright
        # preplace and the place). The cart deck is not a cuMotion WORLD obstacle for this run
        # (world_obstacles=[], base_collision_object=False), so this is purely the attached-carton self-check
        # -- a conservative sphere rejection of the intended set-down, exactly like the cart-cage insert and
        # the live_pose place. The PREPLACE standoff stays collision-aware (the fixed upright
        # collect_place_preplace_shape joint goal). OFF (default) => byte-identical; collect4 staging + this
        # key = the fix. Scoped to the non-cage collect place, so transfer / live_pose / scripted-still are
        # untouched. See _collect_deck_place and data/sim-robust-collect-place-straightline-v1/report.md.
        self.collect_place_straight_line = bool(collect_place_straight_line)
        # sim-collect-retreat-abort-v2 (physcon1-fail1 / rtfcollect2-fail1, 2026-11-03): the NON-cage
        # COLLECT deck-place EMPTY retreat is a single collision-OFF straight-line joint smoothstep from
        # the folded place config to the clear collect_place_preplace_shape. Its first increment is almost
        # pure HORIZONTAL retract (gripper TCP x ~1.07 -> ~0.71 m, z only +0.07), which drags the still-
        # ENABLED wrist links (ur_wrist_2/3 -- the carton's seat links, NOT on the deferred tool-side
        # collider set) sideways into the just-RE-ENABLED carton (SIM_ATTACH re-enables the carton at
        # release) and the rollwagen cage (arm-vs-cage is NOT collision-filtered). The arm PINS: shoulder_
        # lift ~0.6 rad / elbow ~0.5 rad short for the whole settle window while the distal wrists run at
        # velocity saturation (the classify_contact_pin fingerprint), and the settle gate aborts the
        # trajectory (moveit_err=-4). It is NOT drive authority (gravity ~103 N*m << the 400 N*m release cap
        # and the smoothstep is already slow, ~0.03 rad/s^2 -> ~1 N*m inertial), NOT a wrist self-collision
        # (the wrists sit in the clear carry region) -- only a ~300 N*m external CONTACT can hold the
        # saturated drive pinned. Fix: lift the empty gripper straight UP first (separating the enabled
        # wrist links from the carton/shelf/cage), THEN fold to the clear preplace shape. The lift-off is a
        # DLS IK (collect_retreat.liftoff_goal) that raises the TCP +dz along world +Z preserving the tool
        # orientation, seeded from the measured release config (stays on the self-clear branch, leans on
        # elbow+wrist_1 so it executes even while the shoulder is pinned against a downward contact). ON by
        # default for this retreat (owner-authorised; scoped to _collect_deck_place so cart-cage / live_pose
        # transfer / scripted are byte-identical); env RESTOCK_COLLECT_RETREAT_LIFTOFF=0 disables it. See
        # collect_retreat.py and data/sim-collect-retreat-abort-v2/report.md.
        _liftoff_env = os.environ.get('RESTOCK_COLLECT_RETREAT_LIFTOFF', '').strip().lower()
        if _liftoff_env in ('0', 'false', 'off', 'no'):
            self.collect_retreat_liftoff = False
        elif _liftoff_env in ('1', 'true', 'on', 'yes'):
            self.collect_retreat_liftoff = True
        else:
            self.collect_retreat_liftoff = bool(collect_retreat_liftoff)
        self.collect_retreat_liftoff_dz = _float_env_bounded(
            'RESTOCK_COLLECT_RETREAT_LIFTOFF_DZ', collect_retreat_liftoff_dz, 0.03, 0.40)
        # sim-robust-transfer-preplace-reach-v1 (coldwin6, 2026-10-24): steer the live_pose TRANSFER
        # PREPLACE's wrist-guarded seeded-KDL IK to the HOLDABLE upright clear branch and reject a
        # clear-but-DEEP-fold goal. The preplace standoff above the bay has (at least) two wrist-clear IK
        # branches: a holdable upright one (ur_shoulder_lift ~ -1.9) and a deep fold (~ -4.0) the arm
        # physically cannot reach. The full-chain park (unlike the staged recorded_bay_park, which accepts
        # the upright branch at seed 0) lands KDL's measured seed in the deep branch, then the wrist-flip
        # re-seed that clears the wrist converges to the deep-fold clear twin -- which the plain wrist-guard
        # ACCEPTED (coldwin6 commanded ur_shoulder_lift -4.003; the arm froze 0.35 rad short at -3.65 ->
        # PATH_TOLERANCE_VIOLATED -> HOST_FAULT_ESTOP phase=preplace). A reachable UPRIGHT clear branch
        # (ur_shoulder_lift -1.95) provably EXISTS at that same park pose (CPU: tests/
        # test_restock_transfer_preplace_reach). (max_droop, upright_seed) or None (off => the plain
        # wrist-guard, byte-identical). Validated in data/sim-robust-transfer-preplace-reach-v1/report.md.
        self.transfer_preplace_reach_guard = resolve_reach_guard(transfer_preplace_reach_guard)
        # sim-robust-transfer-settle-oscillation-v1 (coldwin12, 2026-10-24): extend the clear-wrist-region
        # guard to the live_pose TRANSFER PICK (approach + contact), which the carry wrist_guard deliberately
        # does NOT cover (transfer_carry.GUARDED_PHASES = preplace/place/retreat only). The pick's approach is
        # a cuMotion task-space pose goal and the contact is a straight-line seeded-KDL IK seeded from the
        # approach branch; both were unguarded, so the approach IK intermittently landed on the ur_wrist_2-vs-
        # ur_forearm self-colliding family (coldwin12: wrist_1 1.171 above the ~0.69 stop, wrist_2 -1.033 below
        # 0) and the contact inherited it and limit-cycled 0.169 rad / ~1.6 rad/s for the whole 30 s settle
        # window (PR239's envelope correctly failed it), where coldwin11 took the clear wrist_2-positive family
        # (wrist_1 -0.18, wrist_2 +0.93) and settled instantly. On => the approach pose goal carries the
        # clear-region joint constraints (ur_wrist_1 <= carry_wrist1_cap, ur_wrist_2 >= carry_wrist2_min, the
        # proven fm327/fm403 bounds) and the contact straight-line IK is wrist-guarded (clear-branch re-seed),
        # so the pick stays on the clear family. Off (default) => byte-identical. See _pick_wrist_guard and
        # data/sim-robust-transfer-settle-oscillation-v1/report.md.
        self.transfer_pick_wrist_guard = bool(transfer_pick_wrist_guard)
        # fm476: SIM-ONLY env override for the sensor-packet freshness bound. The fm465 light rig makes
        # the TRANSFER capture's full-res render slow, so by the time the packet reaches the host the
        # sim /clock has advanced ~0.55 s past the packet stamp -- STALE beyond the 0.5 s default, so the
        # host rejected the first transfer packet ('timestamp stale or future (now-stamp=0.550, ...)')
        # -> 'failed' -> arm feed death -> stall (run-350, fm474/475 diagnosis). Raising the bound is
        # SAFE here: the transfer (and collect) scene is PARKED/static, so a slightly older packet is
        # strictly consistent. Unset => the 0.5 s default (byte-identical real-rig).
        self.cart_location = cart_location
        # The env value is the window AT 1/60; _freshness_scale rebases it onto the running step
        # (sim-physics-step-collision-band-v1; factor 1.0 at 1/60 => byte-identical).
        self.max_age = _float_env_bounded('RESTOCK_SENSOR_MAX_AGE_S', max_age, 0.1, 5.0) * self._freshness_scale
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
        # Rebased onto the running step too: the render lead is a one-to-few-PHYSICS-STEP delivery-order
        # lead, so at a coarser step one step spans more sim time and a legitimately one-step-ahead stamp
        # would otherwise be mis-flagged future (sim-physics-step-collision-band-v1; 1.0 at 1/60).
        # sim-collision-dt-render-lead-floor-v1: the band FLOOR (-render_lead -- how far a FUTURE-leading
        # stamp may be and still be fresh) is DERIVED, not a hand-set env constant. The mirrored collision
        # stamp (collision_time) is nvblox's ESDF integration time, which rides the Isaac RENDER clock;
        # under heavy live_pose render pacing that clock runs AHEAD of the host packet clock read as `now`,
        # so a FRESH collision stamp legitimately leads `now`. coldwin30 tripped collect preplace with
        # collision_dt=-0.5333 against a -0.5 floor (the collision world led `now` by 0.53 s -- FRESHER,
        # not stale), and the integration lane had to bump RESTOCK_FRESH_RENDER_LEAD_S to 0.7 by hand to
        # clear it (coldwin31). The floor is now (delivery lead + ESDF incorporation window), step-scaled
        # = 0.7 s at the default step, so the env override is unnecessary. The delivery term keeps its env
        # (default 0.2); the +COLLISION_RELAY_LEAD_S (0.5) ESDF term is what generalises PR 288's step-aware
        # rule to the default step (where the pure scaling was a no-op). The positive (stale) CEILING
        # (max_age) and the trip name are UNCHANGED -- only the future side gains headroom (strictly MORE
        # current data -- safe; see _fresh / the fm423 render-lead rationale above). Logged once at host
        # start via freshness_band_summary() so every run dir records the band in force (never silent).
        self._render_lead_delivery = _float_env_bounded('RESTOCK_FRESH_RENDER_LEAD_S', 0.2, 0.0, 0.5)
        self.render_lead = collision_band_floor(self._render_lead_delivery, dt_from_env())
        self.phase_timeout, self.fixture_mode = phase_timeout, fixture_mode
        self.phase = 'idle'
        # sim-preplace-per-joint-gains-v1: carried-preplace convergence guard + wrist-first sequencing.
        # The guard names a frozen joint (goal error not shrinking for N sim-s) as PHASE_CONVERGENCE instead
        # of the opaque 120 s phase timeout (coldwin53: shoulder_pan froze 0.56 rad short). The wrist-first
        # sequence is the fallback that sweeps the wrists at the low-reaction tucked pose before the hold
        # joints move. Both opt-in (stall 0 / flag False => inert => byte-identical).
        self.preplace_wrist_first = bool(preplace_wrist_first)
        self._preplace_wristed = False   # wrist-first sub-stage flag (mirrors _retreat_lifted)
        # sim-preplace-sweep-cage-clearance-v1: the PAN-FIRST preplace reorder (opt-in, default off =>
        # byte-identical). coldwin70: the wrist-first order EXTENDS the arm before the pan sweep, so leg-2
        # rotates the extended gripper housing on an arc that rakes /Rollwagen/chassis/post_2 + the upper
        # side-wall bar at shoulder_pan ~ 0.20 (the -48 mm strike that froze the run at a 1632.7 N*m spike).
        # When this is on AND wrist-first is on, the first leg sweeps shoulder_pan to the standoff while the
        # arm stays TUCKED at the carry shape (CPU: cage clearance +252 mm vs -48 mm), then the second leg
        # extends. Requires preplace_wrist_first (it reuses that two-leg machinery, just the inverse hold).
        self.preplace_pan_first = bool(preplace_pan_first)
        # Injected cage-clearance gate (sim-preplace-sweep-cage-clearance-v1): callable(waypoints)->reject|None
        # built by the boot/integration layer from the live coupled cart pose + the host XRDF
        # (place_clearance.preplace_cage_clearance_provider). None (default) => the gate is inert (byte-
        # identical); when set, _issue_gated_preplace_approach refuses an intersecting leg-2 sweep BY NAME
        # (PREPLACE_SWEEP_CLEARANCE prim=... min_clearance=...) before motion, instead of the blind jam.
        self._preplace_cage_clearance = None
        # sim-preplace-leg2-level-v1: the held chassis attitude during the station place. The collect place
        # pins the base at a non-level parking-brake attitude (chassis_station_pitch_hold_pitch_deg /
        # _roll_deg), so ur_base_link +Z is off world +Z and the shoulder_pan YAW axis is off vertical -- a
        # pan sweep then carries the welded carton on a cone and the WORLD tilt is not pan-invariant
        # (coldwin56 leg 2: held-wrist carton 1.64 -> 2.02). When the hold is active the carton-level FK
        # measures / levels the preplace in the WORLD frame via this rotation, so "parallel to the ground"
        # is the captain's world rule. Off => None => the base frame (byte-identical).
        self.station_pitch_hold = bool(station_pitch_hold)
        self.station_hold_roll_deg = float(station_hold_roll_deg)
        self.station_hold_pitch_deg = float(station_hold_pitch_deg)
        self._carton_base_R = None
        if self.station_pitch_hold:
            from twin.restock_manipulation import carton_level as _clbase
            self._carton_base_R = _clbase.base_rotation_rp(self.station_hold_roll_deg,
                                                           self.station_hold_pitch_deg)
        from twin.restock_manipulation.convergence_guard import ConvergenceGuard
        from twin.restock_manipulation.cad_tool import ARM_JOINTS as _ARM_JOINTS
        # sim-preplace-wrist2-hold-v1: opt-in DRIFT RE-CAPTURE on the convergence guard. When a driven
        # preplace joint drifts off its closest approach by more than preplace_drift_recapture_rad (and
        # budget remains), the guard emits a non-terminal DriftRecapture: the controller re-issues the
        # armed preplace goal (re-driving the back-driven joint) and logs PREPLACE_DRIFT_RECAPTURE once,
        # instead of waiting out the 30 s stall and faulting (cw66: the leg-2 shoulder_pan sweep back-drove
        # the arrived-but-soft wrist_2 to a -0.32 limit cycle; the static hold is trivial, so a re-issue
        # after the pan settles holds). Both <= 0 => inert => byte-identical.
        self.preplace_drift_recapture_rad = float(preplace_drift_recapture_rad)
        self.preplace_drift_recapture_max = int(preplace_drift_recapture_max)
        self._preplace_convergence = ConvergenceGuard(
            _ARM_JOINTS, stall_s=float(preplace_convergence_stall_s),
            recapture_rad=self.preplace_drift_recapture_rad,
            recapture_max=self.preplace_drift_recapture_max)
        self.preplace_convergence_stall_s = float(preplace_convergence_stall_s)
        self._preplace_recapture_logged = False   # PREPLACE_DRIFT_RECAPTURE is logged once per preplace
        self._retreat_lifted = False   # sim-collect-retreat-abort-v2: lift-off-then-fold sub-stage flag
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
        # sim-place-descent-residual-v3: the live carton-centre z (ur_base_link) at which the
        # contact-triggered weld release fires during the collect set-down, latched when the place descent is
        # issued (place_contact_release on + a measured placed_pose). None => the gate is inert (off / no
        # measured carton pose) and the legacy place->level-gate->release owns the set-down.
        self._place_contact_release_z = None
        # sim-place-release-on-contact-v1: the contact-GATED place-descent release decider + its per-descent
        # bounded-lateral-re-approach budget. Constructed only when carton_level.place_release_on_contact is
        # on; None => the legacy fixed-z place_contact_release gate above owns the set-down => byte-identical.
        # The gate is reset (and its z fallback armed) when the place descent is issued (_arm block below).
        self._place_release_gate = None
        self._place_reapproach_count = 0
        self._place_release_log_sim = None     # last PLACE_RELEASE_TICK heartbeat time (throttle)
        self._place_seat_center_z = None       # sim-place-seat-geometry-v1: armed seat (carton-centre z) per set-down
        if (self._carton_level is not None
                and getattr(self._carton_level, 'place_release_on_contact', False)):
            from twin.restock_manipulation.place_release import PlaceReleaseGate
            cl = self._carton_level
            self._place_release_gate = PlaceReleaseGate(
                effort_margin_nm=cl.place_release_effort_margin_nm,
                effort_consecutive=cl.place_release_effort_consecutive,
                stall_eps_m=cl.place_release_stall_m, stall_ticks=cl.place_release_stall_ticks,
                contact_gap_mm=(None if cl.place_release_contact_gap_mm is None
                                else float(cl.place_release_contact_gap_mm)),
                support_tokens=cl.place_support_tokens,
                obstruction_gap_mm=cl.place_obstruction_gap_mm,
                stall_height_margin_m=(
                    None if getattr(cl, 'place_release_stall_height_margin_m', None) is None
                    else float(cl.place_release_stall_height_margin_m)),
                # sim-place-seat-geometry-v1: the honest seat gate's tolerance; seat_center_z is armed
                # per-set-down at the descent-arm block (reset) only when place_seat_gate is on.
                seat_tol_m=float(getattr(cl, 'place_seat_tol_m', 0.02)))
        # sim-place-preplace-above-support-v1: which above-place lane re-targeted the collect preplace
        # standoff ('support' | 'place') and its carton-bottom clearance above the resolved support top,
        # set when _preplace_above_place_goal engages (for the engaged-standoff log).
        self._preplace_above_mode = None
        self._preplace_above_clearance = None
        # sim-preplace-level-gate-v1: the predicted EXECUTED (droop) attitude of the solved standoff, set by
        # _preplace_above_place_goal when the gate is on (for the approach-gate logging).
        self._preplace_standoff_exec_rpy = None
        # sim-place-release-serialization-v1: the contact-triggered weld release must be a proper PHASE
        # transition, not a scene mutation issued mid place-execute (coldwin43: the detach ran while the
        # place ExecuteTrajectory was still in flight -> 'planning-scene mutation issued while a plan/execute
        # is in flight (state=execute_result)' -> HOST_FAULT_ESTOP at phase=release). When the gate fires it
        # CANCELS the in-flight place move (the cancel is the stop -- the arm never keeps pressing the welded
        # carton into the deck, which is what produced the 2.0 deg set-down tilt in coldwin39-42), the release
        # phase WAITS for the cancel's terminal (settled, not-in-flight) result, and only THEN detaches. These
        # flags drive that sub-state; inert (never set) unless the contact-release gate fires => byte-identical.
        self._release_cancel_pending = False   # a contact-release cancel+settle is owed before the detach
        self._release_cancel_issued = False    # the cancel() request has been sent to the motion port
        self._release_cancel_sim_start = None  # sim time the cancel was requested (bounds the settle wait)
        # sim-place-release-cancel-handshake-v1: stopped-arm release fallback state (handshake path only).
        # Inert unless release_cancel_handshake is on AND the terminal does not arrive => byte-identical.
        self._release_stopped_ticks = 0        # consecutive ticks the measured arm has read stopped
        self._release_prev_q = None            # previous tick's joint_positions (for the finite-diff speed)
        self._release_prev_q_sim = None        # sim time of the previous joint_positions sample
        # The bounded SIM-clock budget for the place-cancel to reach a terminal result before the release
        # detach (report + fail BY NAME, never detach blindly into an in-flight planning scene). Internal
        # (env-tunable), clamped under the per-phase timeout; NO HOST/RUNTIME config key (config_preflight
        # unchanged). A cancel settles in ~1 controller tick (coldwin43 release cadence ~0.017 s/tick), so
        # this is a safety backstop, not a primary terminator.
        self.release_cancel_timeout_s = _float_env_bounded(
            'RESTOCK_RELEASE_CANCEL_SETTLE_S', min(2.0, float(phase_timeout)), 0.1, float(phase_timeout))
        # sim-release-open-reissue-v1: the opt-in release-open RE-ISSUE gate (coldwin84 lost the one-shot
        # gripper-open on the last-value relay -> the weld never released). Built only when
        # carton_level.release_reissue is on => off => None => byte-identical. The confirm window is clamped
        # under the per-phase timeout so it stays a tighter bound.
        self._release_reissue = None
        if self._carton_level is not None and getattr(self._carton_level, 'release_reissue', False):
            from . import release_reissue as _release_reissue_mod
            self._release_reissue = _release_reissue_mod.ReleaseReissueGate(
                confirm_timeout_s=min(float(self._carton_level.release_confirm_timeout_s),
                                      float(phase_timeout)),
                log_every=int(self._carton_level.release_reissue_log_every))
        # sim-release-threshold-carton-frame-v1: the opt-in post-detach FREE-AND-RESTING + clearance-lift gate
        # (coldwin85/86 fell to the floor when the retreat swept the barely-rested carton off the shelf).
        # Built only when carton_level.release_free_check is on => off => None => the detach transitions
        # straight to retreat => byte-identical. These flags drive the post-detach sub-state in the release
        # phase; inert (never set) when the gate is off.
        self._release_free = None
        if self._carton_level is not None and getattr(self._carton_level, 'release_free_check', False):
            from . import release_free as _release_free_mod
            cl = self._carton_level
            self._release_free = _release_free_mod.ReleaseFreeGate(
                settle_ticks=cl.release_free_settle_ticks, stable_eps_m=cl.release_free_stable_eps_m,
                rest_gap_m=cl.release_free_rest_gap_m,
                settle_window_ticks=cl.release_free_settle_window_ticks,
                lift_dz_m=cl.release_free_lift_dz_m, follow_frac=cl.release_free_follow_frac)
        self._released_detached = False   # a weld detach is confirmed; the free-check sub-state owns retreat
        self._rf_stage = 'settle'         # release_free sub-stage: 'settle' (hold) -> 'lift' -> (proceed/fail)
        # sim-robust-phase-timeout-during-recovery-v1: credit the per-phase timeout clock for the sim time
        # spent while the motion adapter is inside a NAMED move_group crash recovery (respawn hold /
        # post-respawn re-attach / an observed move_group death still being recovered -- motion.
        # crash_recovery_active()). respawn2b: two deaths inside the collect lift consumed ~119 of the 120 s
        # lift budget (orphaned-goal waits + re-sends), tripping 'phase timeout: lift' on a crash that had
        # otherwise recovered. The phase is NOT stuck during that window -- it is waiting out a bounded
        # recovery nothing in the phase caused -- so the credited sim time is subtracted from the timeout
        # comparison (elapsed - credit > phase_timeout). The credit is BOUNDED by _recovery_pause_cap_s so a
        # recovery that never completes still ends BY NAME here rather than pausing forever; the adapter's
        # own bounds (RESPAWN_WAIT_CAP_S, the re-attach budget, the PR 234/242 fast-fail) normally end it
        # first. No recovery => the probe is False, credit stays 0.0, the comparison is the old
        # `elapsed > phase_timeout` => byte-identical. Cap env (SIM seconds, >0 enables; 0 disables the
        # feature entirely => byte-identical); default is a generous belt-and-braces bound above a real
        # multi-death recovery, not a primary terminator.
        self._recovery_pause_cap_s = _float_env_bounded('RESTOCK_RECOVERY_PAUSE_CAP_S',
                                                        RECOVERY_PAUSE_CAP_S, 0.0, 3600.0)
        self._recovery_credit_s = 0.0        # sim time credited to crash recovery in the current phase
        self._recovery_pause_logged = False  # the once-per-phase PHASE_CLOCK_RECOVERY_PAUSE line fired
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

    def _seal_diag_tick_ok(self, s):
        """sim-pick-contact-seal-fallback-v1 guard 3b (per-tick predicate): the geometric SEAL_DIAG shows
        the pad flush on a body under the pad AND the plugin is enabled this tick. Pure; the dwell counter
        is owned by the caller. False whenever the key is off or any field is unmeasured (no evidence)."""
        if not self.contact_accept_seal_diag:
            return False
        if not s.gripper_enabled or not s.seal_hit_target:
            return False
        gap, ang = s.seal_gap_mm, s.seal_pad_angle_deg
        if gap is None or ang is None:
            return False
        return (gap <= self.contact_seal_diag_max_gap_mm
                and ang <= self.contact_seal_diag_max_angle_deg)

    def _seal_diag_contact(self, s):
        """Guard 3b accept decision: advance/track the dwell counter and return True once the seal has held
        for ``contact_seal_diag_dwell_ticks`` consecutive ticks. Logs CONTACT_SOURCE_DISAGREE once per op the
        first time the seal diagnostic accepts while the plugin latch (``s.contact``) still reads False."""
        if self._seal_diag_tick_ok(s):
            self._contact_seal_dwell += 1
        else:
            self._contact_seal_dwell = 0
        accept = self._contact_seal_dwell >= self.contact_seal_diag_dwell_ticks
        if accept and not s.contact and not self._contact_seal_disagree_logged:
            self._contact_seal_disagree_logged = True
            print('CONTACT_SOURCE_DISAGREE plugin=False seal_diag=True gap_mm=%s pad_deg=%s '
                  'hit_target=%s enabled=%s dwell=%d/%d -- accepting the geometric seal as contact '
                  '(gated SIM_ATTACH_FALLBACK forms the weld Isaac-side)'
                  % (s.seal_gap_mm, s.seal_pad_angle_deg, s.seal_hit_target, s.gripper_enabled,
                     self._contact_seal_dwell, self.contact_seal_diag_dwell_ticks), flush=True)
        return accept

    def _contact_reissue_controls(self, s):
        """sim-pick-contact-seal-fallback-v1 (item 2): the gripper-enable control to re-publish THIS contact
        tick so the one-shot enable cannot be lost to the last-value /manipulation/status relay. Returns the
        re-emit control tuple while the key is on, suction has been enabled, and neither the plugin latch nor
        the seal-diag accept has formed yet; else ``()`` (byte-identical when the key is off)."""
        if (self.contact_reissue_suction_each_tick and self.phase == 'contact'
                and self._contact_suction and not s.contact):
            return ({'kind': 'gripper', 'enabled': True},)
        return ()

    def _gripper_status_fresh(self, s):
        """sim-gripper-status-relay-v1: is there an AUTHORITATIVE, fresh Isaac gripper status this tick?

        Returns ``(present, age)`` (pure). ``present`` is True when the packet carries a
        ``gripper_status_time`` (the opt-in feedback relay populated it) AND -- when the optional
        ``contact_status_max_age_s`` bound is set -- it is not older than the bound. ``age`` is the sim
        seconds since the last authoritative status the controller saw (None when it has never seen one),
        for the CONTACT_STATUS_STALE log. A packet with no ``gripper_status_time`` means the host received
        NO gripper status (the coldwin65 starve), NOT a disabled plugin -- the guard never aborts on it."""
        t = getattr(s, 'gripper_status_time', None)
        if t is None:
            age = (None if self._gripper_status_last_time is None
                   else max(0.0, float(s.simulation_time) - float(self._gripper_status_last_time)))
            return False, age
        self._gripper_status_last_time = float(t)
        age = max(0.0, float(s.simulation_time) - float(t))
        if self.contact_status_max_age_s > 0.0 and age > self.contact_status_max_age_s:
            return False, age
        return True, age

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

    def set_collect_source_evidence(self, provider):
        """fm449: inject the collect-source clean-capture signal the observe gate polls before the reach.

        ``provider(now) -> bool`` returns True only when the fixed source camera delivered, AT PARK, a
        CLEAN synchronized frame WITH a decoded label for the calibrated source slot this tick -- NOT a
        held/seeded republish. The camera lane builds it from the same clean-frame/occluded signal the
        source provider already computes (``restock_camera.source_target.build_collect_source_evidence``:
        a non-None SourceTarget whose ``clean`` flag is set). Inert unless ``require_collect_source_evidence``
        is set; left unset (default) the gate never arms => byte-identical."""
        self._collect_source_evidence = provider

    def set_arm_pause_sink(self, sink):
        """sim-arm-reactive-replan-v1 (P1): inject the reversible arm-command-gate pause.

        ``sink(paused: bool)`` drives ``ArmCommandGate.set_paused`` (the mission wires it at boot). The
        reactive human authority calls it edge-triggered on STOP/BACKSTOP (True) and release (False).
        Inert unless the reactive layer is enabled (``reactive_arm_human``); no sink => the preempt/cancel
        hold is the sole stop mechanism (still safe)."""
        self._arm_pause_sink = sink
        if self._reactive_arm is not None:
            self._reactive_arm.set_pause_sink(sink)

    def set_human_detection_provider(self, provider):
        """sim-arm-reactive-replan-v1 (P1): inject the vendored human-safety evaluator.

        ``provider(now_wall) -> humans.perception.messages.SafetyState | None`` returns the freshest
        people judgement (CLEAR/SLOW/STOP + nearest distance), from the ground-truth or detector-fed
        ``scripts/twin/humans`` path. None (no provider, or a None return) fails the gate toward STOP.
        Inert unless the reactive layer is enabled => byte-identical."""
        self._arm_detection_provider = provider
        if self._reactive_arm is not None:
            self._reactive_arm.set_detection_provider(provider)

    def set_arm_clearance_source(self, *, esdf_source, corridor_fn):
        """sim-arm-reactive-replan-v1 (PR-B): inject the perceived-clearance seams for the execution-
        time corridor-watch re-plan (sibling sim-rgbd-arm-collision-aware-v1).

        ``esdf_source`` is an ``esdf_source.EsdfSource`` (``snapshot(now)`` -> live nvblox ESDF cropped
        to ur_base_link); ``corridor_fn(measured_joints, goal_pose, held) -> [AABB]`` builds the
        REMAINING swept gripper(+carton) corridor (current-TCP FK -> goal). The mission builds both from
        the GatedMotionPort at boot (``reactive_arm.build_corridor_fn``). Inert (replan stays off) unless
        ``reactive_arm_replan`` is enabled => byte-identical."""
        if self._reactive_arm is not None:
            self._reactive_arm.set_clearance_source(esdf_source=esdf_source, corridor_fn=corridor_fn)

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
        self.operation = operation   # fm449: scope the collect-source evidence gate to collect
        self.source = job['source_bin'] if operation == 'collect' else self.cart_location
        self.destination = self.cart_location if operation == 'collect' else job['shelf_id']
        if self.source == self.destination:
            raise ValueError('source and destination must differ')
        self.phase, self.elapsed = 'observe', 0.0
        self.last_time = None
        self.evidence, self.reason = [], ''
        self.issued = False
        # sim-robust-movegroup-second-death-v1: clear the port's TOTAL per-op move_group crash-retry budget
        # so a fresh operation gets the full cap (the per-move budget already resets in begin/begin_joint_goal).
        # Guarded for a motion stub without the method (byte-identical when absent).
        _reset_crash_budget = getattr(self.motion, 'reset_crash_budget', None)
        if callable(_reset_crash_budget):
            _reset_crash_budget()
        self._contact_suction = False   # suction enabled once during the contact descent (grasp latch)
        self._contact_waits = 0         # bounded grace ticks awaiting a fresh acquisition deadline
        self._contact_step = 0.0        # cumulative contact-feedback over-press travel (m); see __init__
        self._contact_freeze = 0        # freeze-settle hold ticks at a contact step (see CONTACT_FREEZE_TICKS)
        # sim-pick-contact-seal-fallback-v1 (coldwin63 pick): per-op state for the opt-in contact guards.
        self._contact_seal_dwell = 0    # consecutive ticks the geometric seal diagnostic held (guard 3b)
        self._contact_suction_miss = 0  # over-press-step boundaries reached with the plugin not enabled (guard 3a)
        self._contact_seal_disagree_logged = False  # CONTACT_SOURCE_DISAGREE printed once per op
        # sim-gripper-status-relay-v1 (coldwin65): per-op freshness tracking for the require_suction_enabled
        # guard. _gripper_status_last_time is the sim time of the last packet that carried an authoritative
        # gripper status; _contact_status_stale_logged throttles the CONTACT_STATUS_STALE line to once per op.
        self._gripper_status_last_time = None
        self._contact_status_stale_logged = False
        self._contact_progress_wall = None  # fm425: wall time of the last contact-phase progress (per op)
        self._press_sim_start = None    # sim time the current unconfirmed pre-weld press began (see __init__)
        self._grasp_confirm = 0         # fm395: consecutive confirmed-attachment ticks in grasp
        self._grasp_waits = 0           # fm395: ticks awaiting a confirmed attachment before a grasp retry
        self._grasp_retries = 0         # fm395: bounded grasp re-seat retries used this operation
        self._carton_z_at_grasp = None  # fm410: latched at grasp-confirm; the lift-probe tracking baseline
        self._collect_source_evidence_waits = 0  # fm449: ticks awaiting a clean source capture before the reach
        self._recovery_credit_s = 0.0           # sim-robust-phase-timeout-during-recovery-v1: fresh per op
        self._recovery_pause_logged = False
        self._preplace_branch_cache = None  # sim-preplace-select-live-cost-v1: fresh per op
        self._esdf_boot_gate.reset()   # sim-robust-esdf-boot-gate-v1: re-arm the boot ESDF-ack gate per op
        self._reactive_arm_slow = False   # sim-arm-reactive-replan-v1: fresh SLOW flag per op
        if self._reactive_arm is not None:   # re-arm the human-aware reactive authority per op
            self._reactive_arm.reset(0.0)
        self._retreat_lifted = False   # sim-collect-retreat-abort-v2: lift-off-then-fold sub-stage flag
        # sim-release-threshold-carton-frame-v1: fresh post-detach free-check sub-state per op.
        self._released_detached = False
        self._rf_stage = 'settle'
        if self._release_free is not None:
            self._release_free.reset()
        self.verify_ids = set()
        self.release_time = None
        # sim-place-release-serialization-v1: fresh contact-release cancel sub-state per op (see __init__).
        self._release_cancel_pending = False
        self._release_cancel_issued = False
        self._release_cancel_sim_start = None
        # sim-place-release-cancel-handshake-v1: fresh stopped-arm fallback sub-state per op.
        self._release_stopped_ticks = 0
        self._release_prev_q = None
        self._release_prev_q_sim = None

    def _fresh(self, stamp, now):
        return math.isfinite(stamp) and -self.render_lead <= now - stamp <= self.max_age

    def freshness_band_summary(self):
        """One greppable line describing the collision/feedback freshness band in force this run, logged
        once at host start (sim-collision-dt-render-lead-floor-v1) so every run dir records the DERIVED
        band and its widening is never silent. The floor is the derived future-lead headroom
        (delivery render lead + the step-scaled ESDF incorporation window); the ceiling is the unchanged
        stale max_age."""
        scale = self._freshness_scale
        return ('COLLISION_FRESHNESS_BAND floor=%.4f ceiling=%.4f '
                '(render_lead=%.4f = delivery %.4f + esdf_window %.4f, freshness_scale=%.3f '
                'physics_dt=%.6g) -- sim-collision-dt-render-lead-floor-v1 (derived; env '
                'RESTOCK_FRESH_RENDER_LEAD_S override unnecessary)'
                % (-self.render_lead, self.max_age, self.render_lead,
                   self._render_lead_delivery * scale, COLLISION_RELAY_LEAD_S * scale,
                   scale, dt_from_env()))

    def _provenance(self, item, now):
        allowed = ('sensor', 'test_fixture') if self.fixture_mode else ('sensor',)
        if self.transfer_ground_truth:
            allowed = allowed + ('ground_truth',)   # fm461: the authored bay transfer target (demo fallback)
        elif (self.collect_ground_truth and getattr(self, 'operation', None) == 'collect'
              and getattr(item, 'location', None) == getattr(self, 'source', None)):
            allowed = allowed + (gtc.SOURCE,)        # collect-gt: the ground-truth collect pick (source only)
        return (item.source in allowed and bool(item.observation_id)
                and self._fresh(item.simulation_time, now) and item.frame == self.frame)

    def _pose_ok(self, pose):
        return (len(pose) == 7 and all(math.isfinite(x) for x in pose)
                and math.isclose(sum(x*x for x in pose[3:]), 1, abs_tol=1e-4)
                and math.dist(pose[:3], (0, 0, 0)) <= self.profile.max_reach)

    def _live_pose_transfer(self):
        """RESTOCK_TRANSFER_MODE=live_pose AND this run is the transfer (collect is never affected)."""
        return self.transfer_mode == LIVE_POSE and getattr(self, 'operation', None) == 'transfer'

    def _collect_deck_place(self):
        """collect_place_straight_line AND this run is the NON-cage COLLECT deck place. Its place descent
        lowers the HELD carton onto the cart deck, and the carton is a cuMotion attached_object not
        self-collision-ignored against the forearm/wrist/upper-arm, so the set-down fold is rejected as a
        (conservative) attached-object SELF-collision even though the bare arm + wrist-guard are clear
        there (collect4). The place + empty retreat therefore route collision-OFF straight-line like the
        cart-cage place and the live_pose transfer place; the PREPLACE standoff stays collision-aware.
        Off (default) or a cage/transfer run => False => byte-identical."""
        return (self.collect_place_straight_line
                and getattr(self, 'operation', None) == 'collect'
                and not self.cart_cage_place)

    def _place_obstructed(self, now, decision):
        """sim-place-release-on-contact-v1: the descending carton met a NON-support body (a cage post, not
        the shelf/deck). NEVER press (would jam / drop the carton against the post) and never release there.
        Do up to ``place_reapproach_max`` BOUNDED lateral re-approaches -- shift the set-down target
        ``place_reapproach_shift_m`` laterally (base +x, along the cart width, away from the side post) and
        re-descend via the pre-place standoff -- then FAIL BY NAME. Returns a terminal/ hold Result."""
        cl = self._carton_level
        prim = decision.hit_prim or 'unknown'
        self._place_reapproach_count += 1
        shift = float(getattr(cl, 'place_reapproach_shift_m', 0.0) or 0.0)
        can_shift = ('place' in self.poses and self.poses['place'] is not None
                     and len(self.poses['place']) >= 3 and shift > 0)
        if self._place_reapproach_count <= int(getattr(cl, 'place_reapproach_max', 0)) and can_shift:
            pose = list(self.poses['place'])
            # Alternate the lateral nudge direction (+x, -x, ...) across re-approaches so a second attempt
            # does not re-seat on the same post.
            direction = 1.0 if self._place_reapproach_count % 2 == 1 else -1.0
            pose[0] = float(pose[0]) + direction * shift
            self.poses['place'] = tuple(pose)
            print('PLACE_OBSTRUCTED prim=%s phase=place now=%.3f: %s -- bounded lateral re-approach '
                  '%d/%d (shift base_x %+0.3f m, re-descend via the pre-place standoff)'
                  % (prim, now, decision.reason, self._place_reapproach_count,
                     int(getattr(cl, 'place_reapproach_max', 0)), direction * shift), flush=True)
            self.evidence.append({'place_obstructed': {'prim': prim, 'reapproach': self._place_reapproach_count,
                                                       'shift_x': direction * shift, 'sim_time': float(now)}})
            if self._place_release_gate is not None:
                self._place_release_gate.reset()
                self._place_release_log_sim = None
            self._place_contact_release_z = None
            self._transition('preplace')
            return self._result()
        return self._fail(
            'PLACE_OBSTRUCTED prim=%s: the descending carton met a non-support body (not %s) -- not pressing '
            '(would jam/drop against it) and the lateral re-approach budget (%d) is exhausted; %s'
            % (prim, list(getattr(cl, 'place_support_tokens', ('shelf', 'deck'))),
               int(getattr(cl, 'place_reapproach_max', 0)), decision.reason))

    def in_loaded_place_descent(self):
        """sim-place-descent-watchdog-suspend-v1: True while the loaded place descent is IN FLIGHT -- the
        carton is WELDED/held and the controller is lowering it in the ``place`` phase (the collect
        set-down onto the cart deck OR the live_pose transfer place onto the bay). This is the one window
        where the station/source decode ``/sensors`` stream legitimately stalls under the full-six-axis
        base lock + deep render pacing (sensing_stride 8 + a 2nd full-res live_pose render), so the SIM
        clock advances many seconds between frames while the sim is healthy (coldwin64: HOST_FAULT_ESTOP
        phase=place 'sensor stream watchdog expired' at sim gap >> 1 s). The per-frame ``/sensors`` SIM
        watchdog may be suspended over exactly this window (the WALL backstop stays armed). False
        everywhere else, so the suspension is never active outside the loaded descent -- re-armed on
        release (``held`` goes False) or on leaving the ``place`` phase."""
        return self.phase == 'place' and bool(self.held)

    def in_capture_silent_window(self):
        """sim-retreat-sensor-watchdog-scope-v1: True while the collect capture is SILENT BY DESIGN for the
        current phase -- i.e. the decode-camera render is suspended and the held registration republished
        (``sensing_render_required(phase)`` is False). This is the authoritative, single-source definition of
        the capture-silent phases (``isaac_sensors._SENSING_SUSPEND_PHASES``: the validated-reach EXECUTION
        phases approach/contact/grasp/lift/preplace/place/retreat/release/stow/couple and the carry swing-outs
        tuck/pan/retreat_tuck), so the suspension window matches EXACTLY the phases where ``/sensors`` emits
        nothing. ``in_loaded_place_descent`` (place + held, PR 357) is a subset of this; the post-place
        ``retreat`` and ``stow`` -- where coldwin76/77/78 each died on a different watchdog counting a
        legitimately quiet period -- are the phases this adds. True only inside that set, so the broad
        suspension is never active for the capture-bearing phases (observe / verify re-validation); re-armed
        the instant the phase steps back into a capture-bearing one. The WALL backstop stays armed throughout."""
        from .isaac_sensors import sensing_render_required
        return not sensing_render_required(self.phase)

    def _release_arm_stopped(self, now, sensors):
        """sim-place-release-cancel-handshake-v1: fold one release-phase tick of measured joint motion and
        report whether the arm is PROVABLY stopped -- the measured max joint speed (finite difference of
        ``sensors.joint_positions`` over the sim-time delta) has stayed at/below
        ``release_cancel_stopped_vel`` for ``release_cancel_stopped_ticks`` consecutive ticks. Returns the
        measured max joint speed (rad/s) once the stopped-tick count is met, else None (still moving, or not
        enough samples / a non-positive sim-time delta). NEVER reports stopped off a single sample, so the
        caller never detaches into a moving arm. Pure book-keeping on the controller's own release state."""
        cl = self._carton_level
        vel_thresh = float(getattr(cl, 'release_cancel_stopped_vel', 0.02))
        need_ticks = int(getattr(cl, 'release_cancel_stopped_ticks', 3))
        q = tuple(float(v) for v in sensors.joint_positions)
        prev_q, prev_t = self._release_prev_q, self._release_prev_q_sim
        self._release_prev_q, self._release_prev_q_sim = q, now
        if prev_q is None or prev_t is None or len(prev_q) != len(q):
            return None                          # first sample this release: establish the baseline, no verdict
        dt = now - prev_t
        if not (dt > 0):
            return None                          # need a positive sim-time delta to measure a speed
        max_speed = max(abs(a - b) for a, b in zip(q, prev_q)) / dt
        if max_speed <= vel_thresh:
            self._release_stopped_ticks += 1
        else:
            self._release_stopped_ticks = 0
        if self._release_stopped_ticks >= need_ticks:
            return max_speed
        return None

    def _place_wrist_guard(self):
        """The (wrist1_cap, wrist2_min) clear carry region a place-side KDL goal must satisfy, else the
        adapter re-seeds from the wrist-flipped twin (same TCP, clear branch). Active for:
          * a TRANSFER with RESTOCK_TRANSFER_CARRY=wrist_guard (the warm2 preplace jam, unchanged), or
          * a COLLECT place-on-cart with collect_place_wrist_guard set (sim-robust-cart-place-goal-v1:
            the deck place pose-IK otherwise lands in the ur_wrist_2-vs-ur_forearm self-colliding branch,
            collectsrc's 12-replan 'Invalid c-space position' storm).
        The region is the configured carry guards, else the bench-proven 0.65 / 0.0 (data/sim-restock-
        armbench-v1). None => off => byte-identical begin()."""
        op = getattr(self, 'operation', None)
        on = ((op == 'transfer' and WRIST_GUARD in self.transfer_carry)
              or (op == 'collect' and self.collect_place_wrist_guard))
        if not on:
            return None
        cap = DEFAULT_WRIST1_CAP if self.carry_wrist1_cap is None else self.carry_wrist1_cap
        w2min = DEFAULT_WRIST2_MIN if self.carry_wrist2_min is None else self.carry_wrist2_min
        return (cap, w2min)

    def _pick_wrist_guard(self):
        """sim-robust-transfer-settle-oscillation-v1: the ``(wrist1_cap, wrist2_min)`` clear carry region the
        live_pose TRANSFER PICK (approach + contact) IK must satisfy, or None (off/not applicable =>
        byte-identical). The carry wrist_guard covers only the place half (GUARDED_PHASES), so the pick was
        free to IK the ur_wrist_2-vs-ur_forearm self-colliding family (coldwin12: approach landed wrist_2
        -0.598, the straight-line contact descended to goal wrist_1 1.171 / wrist_2 -1.033 and limit-cycled
        0.169 rad short); coldwin11 took the clear wrist_2-positive family (wrist_1 -0.18, wrist_2 +0.93) and
        settled instantly. Scoped to a live_pose transfer pick approach/contact, so collect / non-live_pose /
        every other phase / the unset key are byte-identical. The region is the configured carry guards, else
        the bench-proven 0.65 / 0.0 (the same bounds _place_wrist_guard uses)."""
        if not (self.transfer_pick_wrist_guard and self._live_pose_transfer()
                and self.phase in ('approach', 'contact')):
            return None
        cap = DEFAULT_WRIST1_CAP if self.carry_wrist1_cap is None else self.carry_wrist1_cap
        w2min = DEFAULT_WRIST2_MIN if self.carry_wrist2_min is None else self.carry_wrist2_min
        return (cap, w2min)

    def _preplace_reach_guard(self):
        """sim-robust-transfer-preplace-reach-v1: the ``(max_droop, upright_seed)`` reach guard for the
        live_pose TRANSFER PREPLACE's wrist-guarded KDL IK, or None when off/not applicable. Scoped to the
        PREPLACE of a live_pose transfer with the wrist_guard active -- the one seeded-KDL pose-IC move that
        hit the deep-fold trap (place/retreat are collision-off straight-line, not pose-IK). None =>
        byte-identical. Only consulted when ``_place_wrist_guard`` is already active for this move."""
        if self.transfer_preplace_reach_guard is None:
            return None
        if not (self._live_pose_transfer() and WRIST_GUARD in self.transfer_carry
                and self.phase == 'preplace'):
            return None
        return self.transfer_preplace_reach_guard

    def _repositions_before_place(self):
        """fm390's TRANSIT base-reposition handshake applies to this move. True when:
          * a non-live_pose op has reposition_before_place set (the collect place-on-cart, unchanged), OR
          * sim-service-leg-carry-v1: a live_pose TRANSFER with transfer_service_carry set -- the shelf-
            staging service leg carries the picked carton to the endcap before placing, so the arm MUST hold
            the carry fold in TRANSIT while the runtime drives the base (which drives the carry for transfer
            only in shelf-staging mode). Without the flag a live_pose transfer stays tuck -> preplace (no
            carry-hold) so it would hold in TRANSIT to the phase timeout -- hence the opt-in gate."""
        if self.transfer_service_carry and self._live_pose_transfer():
            return True
        return self.reposition_before_place and not self._live_pose_transfer()

    def _note_recovery_pause(self):
        """sim-robust-phase-timeout-during-recovery-v1: log the sim time credited to a named move_group
        crash recovery in the current phase, ONCE per phase. Called at each phase exit (success transition
        and the phase-timeout fail path). No credit => no line => byte-identical console on a crash-free
        run. The recovery-bound fail path names the amount in its own reason instead."""
        if self._recovery_credit_s > 0 and not self._recovery_pause_logged:
            print('PHASE_CLOCK_RECOVERY_PAUSE phase=%s paused=%.2fs sim (a named move_group crash recovery '
                  'held the arm; the per-phase timeout budget was credited that time so a recovered crash '
                  'is not charged as a phase timeout)' % (self.phase, self._recovery_credit_s), flush=True)
            self._recovery_pause_logged = True

    def _transition(self, phase):
        self._note_recovery_pause()   # sim-robust-phase-timeout-during-recovery-v1: report before reset
        leaving = self.phase   # sim-carton-level-southdock-v1: the phase we are LEAVING (before reassignment)
        self.phase, self.elapsed, self.issued = phase, 0.0, False
        self._staged_retries = 0   # fm358: fresh retry budget per phase
        self._move_deadline_retries = 0   # fm367: fresh deadline-retry budget per phase
        # sim-preplace-per-joint-gains-v1: disarm the preplace convergence guard + reset the wrist-first
        # sub-stage when leaving preplace, so a stall window is never charged across a phase boundary.
        if phase != 'preplace':
            self._preplace_convergence.disarm()
            self._preplace_wristed = False
            self._preplace_recapture_logged = False   # sim-preplace-wrist2-hold-v1: fresh log per preplace
        self._recovery_credit_s = 0.0          # sim-robust-phase-timeout-during-recovery-v1: fresh per phase
        self._recovery_pause_logged = False
        # sim-carton-tilt-transient-debounce-v1: a phase boundary is a settled, level moment (the move
        # completed) -- start a fresh continuous-exceedance window so a transient in one carried phase is
        # never charged against the next. No-op when the debounce was never built (carton_level off).
        if self._tilt_debounce is not None:
            self._tilt_debounce.reset()
        # sim-carton-level-preplace-v2: report the PLANNED-vs-EXECUTED max carton tilt for the carried
        # phase we are LEAVING, so a bench-vs-GPU gap is visible by name in the log (the gap that cost the
        # collectlevel3 diagnosis). Only when carton_level is on and we measured an executed sample this
        # phase; the planned peak defaults to (unknown) for a phase whose move stashed none.
        # sim-carton-level-southdock-v1: this keyed off ``self.phase`` AFTER it was reassigned to the NEW
        # phase above, so the exec sample stashed under the DEPARTING phase never matched -> the line never
        # emitted (absent from every CARTON run incl. coldwin31). Key it off ``leaving`` and ALSO surface it
        # to the orchestrator console (evidence) so the next GPU run records planned-vs-executed per segment.
        if self._carton_level is not None and leaving in self._carton_seg_exec:
            er, ep = self._carton_seg_exec.pop(leaving)
            pr, pp = self._carton_seg_planned.pop(leaving, (None, None))
            # sim-carton-level-setdown-v1: lead with the compact planned=<deg> executed=<deg> worst-|roll|/
            # |pitch| the integration worker asked for (so the carried descent -- set-down and release
            # included -- is log-tunable without a trip), then the roll/pitch breakdown + bound. 'planned'
            # is the pre-dispatch predicted peak; 'executed' the measured max this phase.
            planned_deg = ('%.2f' % max(pr, pp)) if pr is not None else 'unknown'
            planned = ('planned_max roll=%.2f pitch=%.2f' % (pr, pp)) if pr is not None else 'planned_max=unknown'
            # sim-preplace-wrist-continuity-v1: surface the planned max per-waypoint wrist step for the
            # preplace approach (the branch-continuity signal), so a near-jump is visible without a trip.
            wstep = self._carton_seg_wrist_step.pop(leaving, None)
            wstep_s = ('' if wstep is None
                       else ' | max_wrist_step=%.3f rad (continuity <= %.3f)'
                       % (wstep, self._carton_level.preplace_max_wrist_step_rad))
            line = ('CARTON_SEGMENT_TILT phase=%s planned=%s executed=%.2f | executed_max roll=%.2f pitch=%.2f'
                    ' | %s deg (bound %.2f)%s'
                    % (leaving, planned_deg, max(er, ep), er, ep, planned, self._carton_level.bound_deg,
                       wstep_s))
            print(line, flush=True)
            self.evidence.append({'carton_segment_tilt': line})
        if phase == 'contact':
            self._contact_progress_wall = self.steady_clock()   # fm425: start the WALL no-progress budget

    def _result(self, controls=()):
        status = self.phase if self.phase in ('failed', 'succeeded') else 'running'
        return Result(status, self.phase, self.reason, tuple(controls), tuple(self.evidence),
                      None if status=='failed' else getattr(self,'valid_until_steady',None))

    def _fail(self, reason):
        # sim-robust-host-timers-audit-fix-v1 (named trigger): motion.stop() publishes the global
        # emergency stop, which the Isaac side reports to the mission as a BARE 'host emergency stop'
        # (the stop wire carries no reason). Log the NAMED host-side cause here so host-stack.log can
        # ALWAYS be traced to a distinct trigger -- which watchdog / deadline / freshness / gate fired --
        # before the stop (coldlive sample-4 had no such line, only the PLAN_WALL_DIAG hint). Cheap, one
        # line per fault, at the single controller fault chokepoint. Logs the ACTIVE phase (pre-fail).
        print('HOST_FAULT_ESTOP phase=%s reason=%r' % (self.phase, reason), flush=True)
        self.reason, self.phase = reason, 'failed'
        try:
            self.motion.stop()
        except Exception as exc:
            self.reason += f'; stop adapter failed: {exc}'
        # Never drop a held object on failure. Recovery is an explicit new decision.
        return self._result(({'kind': 'stop'}, {'kind': 'gripper', 'enabled': self.held}))

    # ------------------------------------------------- sim-carton-level-carry-v1
    # The carried phases where a WELDED carton must stay parallel to the ground. 'retreat'/'release'
    # are post-set-down (gripper opening / empty), so they are NOT level-constrained.
    _CARRIED_PHASES = ('lift', 'tuck', 'pan', 'transit', 'preplace', 'place')

    def _fk_model(self):
        """The Isaac-free FK Robot for the carton-level constraint (parsed once). None when no URDF
        was provided (=> the constraint is inert)."""
        if self._fk_robot is None and self._fk_urdf:
            from twin.urdf_probe import parse_urdf
            self._fk_robot = parse_urdf(self._fk_urdf)
        return self._fk_robot

    def _capture_level_up(self, measured):
        """At the first carried pose (carton just welded, still at the level shelf attitude), capture
        the carton's up-axis in the tool frame so later poses can be leveled / checked against it.
        Idempotent; a no-op without a config, FK model, or a 6-vector measured pose."""
        if (self._carton_level is None or self._carton_up_in_tool is not None
                or len(measured) != 6 or not all(math.isfinite(x) for x in measured)):
            return
        robot = self._fk_model()
        if robot is None:
            return
        from twin.restock_manipulation import carton_level as cl
        R = cl.tool_rotation(robot, list(measured), self._carton_level.tool_link)
        self._carton_up_in_tool = cl.carton_up_in_tool(R)
        # sim-preplace-leg2-level-v1: a SEPARATE world-consistent up-axis for the PREPLACE world-frame
        # leveling ONLY (every other lane keeps the base-frame up above, byte-identical). The carton rested
        # LEVEL IN WORLD at the weld while the chassis is held pitched, so the physically-correct carton
        # up-axis in the tool frame is (base_R @ tool_R)^T @ world_up. Measuring/leveling the preplace with
        # THIS up + base_R makes the welded carton parallel to the GROUND. None when the hold is off.
        self._carton_up_in_tool_world = None
        if self._carton_base_R is not None:
            import numpy as _np
            self._carton_up_in_tool_world = cl.carton_up_in_tool(
                _np.asarray(self._carton_base_R, float) @ R)

    def _level_carried_goal(self, goal, measured):
        """A carried joint goal with its wrist cluster re-solved so the welded carton stays LEVEL (a
        level-orientation waypoint), bounded to the clear carry wrist region (carry_wrist1_cap /
        carry_wrist2_min) so the existing self-collision shapes are respected. Off / no FK / no
        captured up-axis => ``goal`` unchanged (byte-identical). A pose with no level solution in the
        clear region raises ValueError -> the caller fails honestly rather than carrying a tilted
        carton."""
        if (self._carton_level is None or not self._carton_level.level_goals
                or len(goal) != 6):
            return goal
        robot = self._fk_model()
        if robot is None:
            return goal
        self._capture_level_up(measured)
        if self._carton_up_in_tool is None:
            return goal
        from twin.restock_manipulation import carton_level as cl
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        big = 2 * math.pi
        limits = {}
        if self.carry_wrist1_cap is not None:
            limits['ur_wrist_1_joint'] = (-big, float(self.carry_wrist1_cap))
        if self.carry_wrist2_min is not None:
            limits['ur_wrist_2_joint'] = (float(self.carry_wrist2_min), big)
        try:
            leveled = cl.level_config(robot, list(goal), self._carton_up_in_tool,
                                      joints=self._carton_level.joints,
                                      tool_link=self._carton_level.tool_link,
                                      limits=limits or None,
                                      accept_deg=self._carton_level.bound_deg)
        except ValueError as exc:
            # No level solution for this pose within the clear wrist region. Do NOT crash and do NOT
            # silently carry it tilted: proceed with the original goal; the CARTON_TILT trip then names
            # and fails the move on the next step (keep trip enabled with level_goals).
            print('CARTON_LEVEL_UNSOLVED phase=%s: %s (proceeding unleveled; CARTON_TILT trip owns it)'
                  % (self.phase, exc), flush=True)
            return goal
        return tuple(float(leveled[j]) for j in ARM_JOINTS)

    def _level_path_plan(self, goal_shape, measured):
        """sim-carry-level-lift-v1: plan a carried fixed-shape move (``goal_shape``) from the measured
        start as a LEVELED WAYPOINT PATH so the welded carton stays parallel to the ground THROUGHOUT the
        move -- not only at its endpoints. Returns a ``carton_level.LevelPathPlan`` (whose predicted
        executed tilt the caller checks against the bound before dispatch), or ``None`` to fall back to
        the single leveled-endpoint goal: when the constraint is off, ``level_path`` is off, there is no
        FK model, the up-axis is not captured yet, the goal is not a 6-vector, or the GOAL has no in-clear-
        region level solution (ValueError from the planner). Bounded to the clear carry wrist region
        (carry_wrist1_cap / carry_wrist2_min) exactly like ``_level_carried_goal``."""
        if (self._carton_level is None or not self._carton_level.level_goals
                or not self._carton_level.level_path or len(goal_shape) != 6):
            return None
        robot = self._fk_model()
        if robot is None:
            return None
        self._capture_level_up(measured)
        if self._carton_up_in_tool is None:
            return None
        from twin.restock_manipulation import carton_level as cl
        big = 2 * math.pi
        limits = {}
        if self.carry_wrist1_cap is not None:
            limits['ur_wrist_1_joint'] = (-big, float(self.carry_wrist1_cap))
        if self.carry_wrist2_min is not None:
            limits['ur_wrist_2_joint'] = (float(self.carry_wrist2_min), big)
        try:
            return cl.plan_level_path(robot, list(measured), list(goal_shape), self._carton_up_in_tool,
                                      bound_deg=self._carton_level.bound_deg,
                                      joints=self._carton_level.joints,
                                      tool_link=self._carton_level.tool_link,
                                      limits=limits or None,
                                      waypoints=self._carton_level.path_waypoints)
        except ValueError as exc:
            # The GOAL itself has no in-clear-region level solution. Fall back to the single-goal path
            # (which also levels the endpoint via _level_carried_goal, or proceeds unleveled under the
            # CARTON_TILT trip) -- never crash, never silently carry a tilt.
            print('CARTON_LEVEL_UNSOLVED phase=%s: %s (no leveled waypoint path; falling back to the '
                  'single leveled-endpoint goal)' % (self.phase, exc), flush=True)
            return None

    def _issue_level_carried_move(self, goal_shape, measured, *, slow, goal_tol):
        """Dispatch a carried fixed-shape move keeping the carton level. With ``level_path`` on and a
        feasible leveled waypoint path, replay the PREDICTED executed tilt and either
        CARTON_LEVEL_PLAN_REJECTED it (fail by name BEFORE any motion) or drive it as a multi-waypoint
        ``begin_joint_path``; otherwise fall back to the single leveled-endpoint ``begin_joint_goal``.
        Returns a controller result (the dispatched move, or the reject failure)."""
        plan = self._level_path_plan(goal_shape, measured)
        if plan is not None:
            reject = plan.reject(self._carton_level.bound_deg)
            if reject is not None:
                print(reject, flush=True)
                return self._fail(reject)
            print('CARTON_LEVEL_PATH phase=%s n=%d predicted_max roll=%.2f pitch=%.2f tilt=%.2f deg '
                  '(<= bound %.2f) -- leveled waypoint carry'
                  % (self.phase, plan.waypoints_n, plan.max_roll, plan.max_pitch, plan.max_tilt,
                     self._carton_level.bound_deg), flush=True)
            self._carton_seg_planned[self.phase] = (plan.max_roll, plan.max_pitch)  # for CARTON_SEGMENT_TILT
            self.motion.begin_joint_path(list(plan.waypoints), measured, slow=slow, goal_tol=goal_tol)
            self.issued = True
            return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))
        goal = self._level_carried_goal(goal_shape, measured)
        self.motion.begin_joint_goal(goal, measured, slow=slow, goal_tol=goal_tol)
        self.issued = True
        return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))

    def _level_approach_path_plan(self, leveled_goal, measured):
        """sim-carton-level-preplace-v1: plan the carried APPROACH from the measured carry/tuck pose to an
        ALREADY-LEVELED goal (the PR 290 full-DOF leveled place-on-cart standoff) as a LEVELED WAYPOINT
        PATH, so the welded carton stays parallel to the ground THROUGHOUT the approach -- not only at its
        endpoints. Unlike ``_level_path_plan`` it does NOT re-level the goal (that would move the PR 290
        standoff off its reach-validated TCP/branch); the goal is pinned verbatim as the final waypoint and
        only the interior wrists re-solve for level. Returns a ``carton_level.LevelPathPlan`` or ``None`` to
        fall back to the single ``begin_joint_goal`` (constraint off, ``level_path`` off, no FK model, the
        up-axis not captured, a non-6-vector goal, or an interior waypoint with no clear-region level
        solution). Bounded to the clear carry wrist region exactly like ``_level_path_plan``."""
        if (self._carton_level is None or not self._carton_level.level_goals
                or not self._carton_level.level_path or len(leveled_goal) != 6):
            return None
        robot = self._fk_model()
        if robot is None:
            return None
        self._capture_level_up(measured)
        if self._carton_up_in_tool is None:
            return None
        from twin.restock_manipulation import carton_level as cl
        big = 2 * math.pi
        limits = {}
        if self.carry_wrist1_cap is not None:
            limits['ur_wrist_1_joint'] = (-big, float(self.carry_wrist1_cap))
        if self.carry_wrist2_min is not None:
            limits['ur_wrist_2_joint'] = (float(self.carry_wrist2_min), big)
        try:
            return cl.plan_level_path_to_goal(robot, list(measured), list(leveled_goal),
                                              self._carton_up_in_tool,
                                              bound_deg=self._carton_level.bound_deg,
                                              joints=self._carton_level.joints,
                                              tool_link=self._carton_level.tool_link,
                                              limits=limits or None,
                                              waypoints=self._carton_level.path_waypoints)
        except ValueError as exc:
            # An interior waypoint has no in-clear-region level solution. Fall back to the single
            # leveled-endpoint goal -- never crash, never silently carry a tilt (the CARTON_TILT trip owns
            # what the single-goal path then does).
            print('CARTON_LEVEL_UNSOLVED phase=%s: %s (no leveled approach path; falling back to the '
                  'single leveled-endpoint goal)' % (self.phase, exc), flush=True)
            return None

    def _issue_level_approach_move(self, leveled_goal, measured, *, slow, goal_tol):
        """Dispatch the carried APPROACH to an ALREADY-LEVELED goal (the PR 290 place-on-cart standoff)
        keeping the carton level THROUGHOUT the approach. With ``level_path`` on and a feasible leveled
        waypoint path, replay the PREDICTED executed tilt and either CARTON_LEVEL_PLAN_REJECTED it (fail by
        name BEFORE any motion) or drive it as a multi-waypoint ``begin_joint_path`` to the SAME standoff
        goal; otherwise fall back to the single ``begin_joint_goal`` (byte-identical when the constraint is
        off). Returns a controller result."""
        plan = self._level_approach_path_plan(leveled_goal, measured)
        if plan is not None:
            reject = plan.reject(self._carton_level.bound_deg)
            if reject is not None:
                print(reject, flush=True)
                return self._fail(reject)
            print('CARTON_LEVEL_PATH phase=%s n=%d predicted_max roll=%.2f pitch=%.2f tilt=%.2f deg '
                  '(<= bound %.2f) -- leveled approach to standoff'
                  % (self.phase, plan.waypoints_n, plan.max_roll, plan.max_pitch, plan.max_tilt,
                     self._carton_level.bound_deg), flush=True)
            self._carton_seg_planned[self.phase] = (plan.max_roll, plan.max_pitch)  # for CARTON_SEGMENT_TILT
            self.motion.begin_joint_path(list(plan.waypoints), measured, slow=slow, goal_tol=goal_tol)
            self.issued = True
            return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))
        self.motion.begin_joint_goal(leveled_goal, measured, slow=slow, goal_tol=goal_tol)
        self.issued = True
        return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))

    def _issue_gated_preplace_approach(self, leveled_goal, measured):
        """sim-preplace-level-gate-v1: dispatch the carried pre-place APPROACH to the executed-level standoff
        ``leveled_goal`` as a DROOP-COMPENSATED, DENSIFIED, speed/accel-capped leveled ``begin_joint_path``,
        gated on the predicted EXECUTED (droop) tilt and, when ``closed_loop_level`` is on, backed by the
        bounded sensor servo. The coldwin50 fix: ``_issue_level_approach_move`` levels the interior wrists only
        GEOMETRICALLY (so the EXECUTED interior still drooped ~1.8 deg) and replays a GEOMETRIC gate (droop-
        blind), and the move was a coarse FAST sweep (run-to-run dynamic deflection -> +7.64 deg). Here the
        approach interior is wrist-droop-cancelled per waypoint (EXECUTED-level throughout), densified so each
        segment's step is small (less dynamic deflection), and time-capped at the carried speed/accel. Refuses
        before motion (CARTON_LEVEL_PLAN_REJECTED) only when the compensated executed path would still exceed
        the TRUE ``bound_deg``; a sub-bound residual over the tighter ``preplace_gate_bound_deg`` target is
        logged (the dynamics cap + closed loop own it) rather than blocking a healthy run. Falls back to
        ``_issue_level_approach_move`` when no FK model / up-axis / compensated solution (never a silent tilt)."""
        from twin.restock_manipulation import carton_level as cl
        cl_cfg = self._carton_level
        robot = self._fk_model()
        if robot is None or self._carton_up_in_tool is None:
            return self._issue_level_approach_move(leveled_goal, measured, slow=True, goal_tol=0.2)
        gc = self._place_gravity_compensation()
        limits = self._place_level_limits() or None
        tcp_link = self._place_fk_link(robot, list(leveled_goal))
        base_R = self._carton_base_R
        geometric_world = base_R is not None and self._carton_up_in_tool_world is not None
        up_level = self._carton_up_in_tool_world if geometric_world else self._carton_up_in_tool
        try:
            if geometric_world:
                # sim-preplace-leg2-level-v1: GEOMETRIC, WORLD-level, densified approach -- re-solve the wrist
                # cluster for level-IN-WORLD (held chassis attitude) at EVERY densified leg-2 waypoint, with NO
                # droop pre-tilt (the coldwin56 phantom). This is the leg-2 fix: the wrists track a
                # continuously world-level orientation as shoulder_pan sweeps on its tilted cone, instead of
                # holding the leg-1 goal (which rose 1.64 -> 2.02 across the pan descent).
                # sim-preplace-approach-seeded-v1: the branch-seeded interior solve (coldwin62 fix) when
                # opted in, else the PR 348 endpoint-anchored 'auto' (byte-identical default).
                seeded = getattr(cl_cfg, 'preplace_seeded_interior', False)
                plan = cl.plan_level_world_approach(
                    robot, list(measured), list(leveled_goal), up_level,
                    tool_link=tcp_link, limits=limits, bound_deg=cl_cfg.bound_deg,
                    max_segment_rad=cl_cfg.preplace_max_segment_rad, level_joints=cl_cfg.joints,
                    base_R=base_R, max_wrist_step_rad=cl_cfg.preplace_max_wrist_step_rad,
                    anchor='seeded' if seeded else 'auto', fast=True,
                    level_accept_deg=cl_cfg.preplace_gate_bound_deg if seeded else None)
            else:
                plan = cl.plan_compensated_approach(
                    robot, list(measured), list(leveled_goal), self._carton_up_in_tool, gc,
                    tool_link=tcp_link, limits=limits, bound_deg=cl_cfg.bound_deg,
                    max_segment_rad=cl_cfg.preplace_max_segment_rad, level_joints=cl_cfg.joints)
        except ValueError as exc:
            print('CARTON_LEVEL_UNSOLVED phase=preplace: %s (no %s approach; falling back to '
                  'the single leveled-endpoint approach)'
                  % (exc, 'world-level' if geometric_world else 'droop-compensated'), flush=True)
            return self._issue_level_approach_move(leveled_goal, measured, slow=True, goal_tol=0.2)
        waypoints = list(plan.waypoints)
        cl_peak = None
        # Extend the bounded closed-loop leveller to the approach (opt-in via closed_loop_level). The approach
        # interior is ALREADY droop-compensated (plan_compensated_approach), so use the MODEL closed loop
        # (CartonLevelServo.correct_path), NOT the sensor-anchored variant: the sensor anchor biases on the
        # MODEL droop at the start, which a pre-compensated path has already cancelled, so anchoring would
        # DOUBLE-COUNT it and push the level path back off. This is a bounded residual backstop only (it barely
        # moves an already-level path); the compensation + dynamics cap are the primary mechanism.
        # The closed-loop residual backstop is a DROOP-MODEL servo; skip it on the geometric world-level lane
        # (there is no droop pre-tilt to residually correct -- the world-level re-solve already zeros it).
        clc = None if geometric_world else cl.ClosedLoopConfig.from_level(cl_cfg)
        if clc is not None and self._carton_up_in_tool is not None:
            clc = cl.ClosedLoopConfig(authority_deg=clc.authority_deg, rate_deg_per_s=clc.rate_deg_per_s,
                                      gain=clc.gain, joints=clc.joints, tool_link=tcp_link)
            servo = cl.CartonLevelServo(robot, self._carton_up_in_tool, clc)
            waypoints = servo.correct_path([list(w) for w in waypoints], gc)
            waypoints[0] = list(measured)            # the executor starts exactly where the arm is
            waypoints[-1] = [float(v) for v in leveled_goal]   # keep the solved executed-level standoff
            cl_peak = math.degrees(max((abs(v) for v in servo.accumulated.values()), default=0.0))
            mr, mp, mt = cl.predict_path_tilt_executed(robot, waypoints, self._carton_up_in_tool,
                                                       tool_link=tcp_link, gravity_comp=gc)
            plan = cl.LevelPathPlan(waypoints=tuple(tuple(v) for v in waypoints), max_roll=mr,
                                    max_pitch=mp, max_tilt=mt, waypoints_n=plan.waypoints_n)
        reject = plan.reject(cl_cfg.bound_deg)
        if reject is not None:
            print(reject, flush=True)
            return self._fail(reject)
        # sim-preplace-wrist-continuity-v1: refuse a planned per-waypoint wrist BRANCH JUMP (the soft wrist
        # cannot track one -- coldwin57's class) before motion rather than stream it into a TRAJ_HOST
        # tracking fault. The branch-continuous re-solve already avoids a jump, so this is cheap safety.
        step_reject = plan.reject_wrist_step(cl_cfg.preplace_max_wrist_step_rad, level_joints=cl_cfg.joints)
        if step_reject is not None:
            print(step_reject, flush=True)
            return self._fail(step_reject)
        # sim-preplace-sweep-cage-clearance-v1: refuse a carried pan sweep whose waypoints would drive an arm
        # link into a Rollwagen cage body (post / side-wall bar) BY NAME before motion, instead of the blind
        # ~1632 N*m jam + opaque PHASE_CONVERGENCE timeout (coldwin70: the gripper housing raked
        # /Rollwagen/chassis/post_2 at shoulder_pan ~ 0.20). Inert unless the boot/integration layer injected
        # the provider (self._preplace_cage_clearance, built by place_clearance.preplace_cage_clearance_provider
        # from the live coupled cart pose + host XRDF) => byte-identical off.
        if self._preplace_cage_clearance is not None:
            try:
                cage_reject = self._preplace_cage_clearance(list(waypoints))
            except Exception as _cage_exc:        # a provider error must never crash the op
                cage_reject = None
                print('PREPLACE_SWEEP_CLEARANCE skipped (provider error: %s)' % _cage_exc, flush=True)
            if cage_reject is not None:
                print(cage_reject, flush=True)
                return self._fail(cage_reject)
        pk_v, pk_a = cl.path_peak_joint_speed(waypoints, speed=cl_cfg.preplace_approach_speed,
                                              accel=cl_cfg.preplace_approach_accel)
        worst = max(plan.max_roll, plan.max_pitch)
        gate = cl_cfg.preplace_gate_bound_deg
        self._carton_seg_planned['preplace'] = (plan.max_roll, plan.max_pitch)  # for CARTON_SEGMENT_TILT
        self._carton_seg_wrist_step['preplace'] = plan.max_wrist_step        # for CARTON_SEGMENT_TILT
        # sim-preplace-sweep-tilt-v1: the gate predicts tilt from the COMMANDED path + the quasi-static
        # droop model only. It is BLIND to the arm's closed-loop DRIVE DYNAMICS -- the stiff position
        # profile (kp 1e6 / eff 2000) overshoots the welded load through the 1.64 rad wrist_2 sweep and
        # swung the carton to 3-8 deg in coldwin50/51/52 while this line predicted <1 deg. The fix is the
        # SOFT carried-preplace profile (phase_drive.carried_soft_phases) + the wrist-sweep reduction, NOT
        # a tighter prediction; the honest caveat is stated here and the per-tick CARTON_TICK log + the
        # CARTON_SEGMENT_TILT phase=preplace exit line carry the EXECUTED truth.
        frame_desc = ('geometric WORLD-level (base pitch=%.2f roll=%.2f deg) densified carried approach -- '
                      'per-waypoint wrist re-solve for level-in-world, NO droop pre-tilt'
                      % (self.station_hold_pitch_deg, self.station_hold_roll_deg)) if geometric_world else (
                      'droop-compensated densified carried approach to the executed-level standoff; '
                      'COMMANDED+droop only, BLIND to drive-dynamics overshoot')
        print('CARTON_LEVEL_PREPLACE_GATE n=%d predicted_executed max roll=%.2f pitch=%.2f tilt=%.2f deg '
              '(target %.2f %s, bound %.2f) peak_joint_speed=%.3f rad/s peak_accel=%.3f rad/s^2 '
              'max_wrist_step=%.3f rad (continuity <= %.3f)%s -- %s '
              '(see CARTON_TICK / CARTON_SEGMENT_TILT)'
              % (plan.waypoints_n, plan.max_roll, plan.max_pitch, plan.max_tilt, gate,
                 'within target' if worst <= gate else 'over target (sub-bound, dynamics cap + closed loop own it)',
                 cl_cfg.bound_deg, pk_v, pk_a, plan.max_wrist_step, cl_cfg.preplace_max_wrist_step_rad,
                 '' if cl_peak is None else ' closed_loop_peak=%.2f deg' % cl_peak, frame_desc), flush=True)
        self.motion.begin_joint_path(list(waypoints), list(measured), slow=True, goal_tol=0.2,
                                     speed=cl_cfg.preplace_approach_speed,
                                     accel=cl_cfg.preplace_approach_accel)
        self.issued = True
        return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))

    def _place_level_limits(self):
        """Joint limits for the level place IK: the clear carry wrist region (carry_wrist1_cap /
        carry_wrist2_min, so the level pose stays off the ur_wrist_2-vs-forearm stop) plus the arm's
        elbow range, so a level re-plan can never drive a joint out of its envelope."""
        big = 2 * math.pi
        limits = {'ur_elbow_joint': (0.1, math.pi)}
        if self.carry_wrist1_cap is not None:
            limits['ur_wrist_1_joint'] = (-big, float(self.carry_wrist1_cap))
        if self.carry_wrist2_min is not None:
            limits['ur_wrist_2_joint'] = (float(self.carry_wrist2_min), big)
        return limits

    def _level_place_goal(self, goal, measured, target_pos=None):
        """sim-carton-level-place-on-cart-v1: a LEVEL place config that keeps the carton parallel to the
        ground AND preserves the deck reach -- the fix PR 287 flagged. The place-on-cart standoff
        (collect_place_preplace_shape) holds the TOP-grasped carton ~180 deg off level; wrist-ONLY
        leveling would hold the attitude but shift the TCP and lose the deck. This re-plans the FULL arm
        (a near-free wrist_3 flange roll turns the inverted hold upright without moving the origin), so
        the standoff keeps its reach-validated TCP and the carton is level. ``target_pos`` None =>
        preserve the goal's own TCP (the fixed standoff shape); a 3-vector => reach that sensed set-down
        position (the deck) while level. Off / no FK / no captured up-axis / non-6 goal => ``goal``
        unchanged (byte-identical). No level solution in the clear region => CARTON_LEVEL_UNSOLVED, return
        ``goal`` (the CARTON_TILT trip then names and fails it -- never a silent tilt)."""
        if (self._carton_level is None or not self._carton_level.level_goals
                or not self._carton_level.level_place or len(goal) != 6):
            return goal
        robot = self._fk_model()
        if robot is None:
            return goal
        self._capture_level_up(measured)
        if self._carton_up_in_tool is None:
            return goal
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        try:
            leveled = self._solve_level_place(robot, list(goal), target_pos)
        except ValueError as exc:
            print('CARTON_LEVEL_UNSOLVED phase=%s: %s (proceeding unleveled; CARTON_TILT trip owns it)'
                  % (self.phase, exc), flush=True)
            return goal
        return tuple(float(leveled[j]) for j in ARM_JOINTS)

    def _preplace_descent_predictor(self, robot, measured, tcp_link, limits):
        """sim-preplace-branch-descent-feasible-v1 (coldwin79): a callable the carry-connected standoff
        selector uses to evaluate each CANDIDATE standoff's PLACE DESCENT level feasibility -- the SAME
        ``plan_compensated_place_descent`` the place-descent gate (CARTON_LEVEL_PLACE_DESCENT) runs at the
        place phase, just started from the candidate standoff instead of the measured arrival. Returns the
        predicted peak executed tilt (deg) ``max(max_roll, max_pitch)`` for a candidate standoff joint map,
        raising ValueError when the descent has no in-limits level solution (the selector treats that as
        descent-infeasible). Returns ``None`` (the selector then skips the descent filter => byte-identical)
        when the feature is off, there is no resolved place pose, or the carton up-axis was not captured.

        The arguments mirror the place-phase gate exactly (``_collect_deck_place`` / ``_level_place_goal``):
        the resolved support TCP ``poses['place']`` as the target, the leveled fixed pre-place shape as the
        IK seed, the base-frame carton up-axis, the default loaded gravity compensation, and the same
        from-above / clearance / depth-bias / waypoint-count place-descent options -- so a candidate the
        selector accepts here is one the place gate also accepts, and one it discards is one the gate would
        have refused before motion."""
        cl_cfg = self._carton_level
        if (cl_cfg is None or not getattr(cl_cfg, 'preplace_descent_feasible', False)
                or 'place' not in self.poses or self._carton_up_in_tool is None):
            return None
        from twin.restock_manipulation import carton_level as cl
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        place_tcp = tuple(float(c) for c in self.poses['place'][:3])
        gc = self._place_gravity_compensation()
        # the descent IK seed = the leveled fixed pre-place shape (the exact seed the place gate passes),
        # computed once per select (not per candidate).
        seed = list(self._level_place_goal(self.collect_place_preplace_shape, measured))
        up_in_tool = self._carton_up_in_tool

        def _descent_tilt(std_map):
            std_vec = [float(std_map[j]) for j in ARM_JOINTS]
            plan = cl.plan_compensated_place_descent(
                robot, std_vec, place_tcp, list(seed), up_in_tool, gc,
                bound_deg=cl_cfg.bound_deg, tool_link=tcp_link, limits=limits,
                waypoints=cl_cfg.path_waypoints,
                approach_from_above=cl_cfg.place_from_above,
                above_clearance_m=cl_cfg.place_above_clearance_m,
                depth_bias_m=cl_cfg.place_depth_bias_m)
            return max(plan.max_roll, plan.max_pitch)

        return _descent_tilt

    def _preplace_branch_select_key(self, standoff, measured, tcp_link, limits):
        """An exact-match signature of every input the reaction-branch standoff select depends on, so the
        per-operation memo (``_preplace_branch_cache``) is reused only when the inputs are identical --
        a mismatch simply recomputes (never a stale standoff). sim-preplace-select-live-cost-v1."""
        import numpy as _np

        def _r(x):
            return tuple(round(float(v), 9) for v in _np.asarray(x, float).ravel())

        cl_cfg = self._carton_level
        lim = tuple(sorted((j, round(float(lo), 9), round(float(hi), 9))
                           for j, (lo, hi) in (limits or {}).items()))
        return (
            _r(standoff), _r(measured), tcp_link, lim,
            _r(self._carton_up_in_tool_world) if self._carton_up_in_tool_world is not None else None,
            _r(self._carton_base_R) if self._carton_base_R is not None else None,
            _r(list(self.collect_place_preplace_shape)),
            tuple(cl_cfg.joints), round(float(cl_cfg.preplace_max_segment_rad), 9),
            round(float(cl_cfg.bound_deg), 9), round(float(cl_cfg.preplace_soft_effort_nm), 9),
        )

    def _preplace_above_place_goal(self, measured):
        """The collect preplace standoff re-targeted DIRECTLY ABOVE the place load-centre, so the
        from-above vertical descent onto the support is POSITIVE and the welded gripper pad clears the
        obstacles the fixed ``collect_place_preplace_shape`` does not. Two independent lanes share this
        one mechanism (prefers the support-aware lane when both are on):

          * sim-place-preplace-above-support-v1 (``preplace_above_support``): raise the standoff so the
            carried carton BOTTOM sits at the resolved support top + ``preplace_support_clearance()``
            (default ``place_above_clearance_m``). coldwin48 selected the UPPER shelf but the DECK-tuned
            fixed shape parked the carton 34 mm BELOW the shelf top -- ``place_descent_drop_m`` read 0.0,
            so the contact-release fired in mid-air and the carton fell. Raising the standoff by the
            clearance makes the descent length the clearance and fires the release at the true support top.
          * sim-place-descent-residual-v1 (``preplace_above_place``, the DECK v1 remedy the preflight now
            refuses): raise the standoff by ``preplace_standoff_m`` to clear the cage mouth posts
            (coldwin39's post_2 CONTACT pivot).

        A full-DOF reach+level IK (``level_place_ik``) to ``place TCP + (0, 0, clearance)`` -- the base is
        gravity-level so arm +Z is world/cart +Z -- seeded from ``collect_place_preplace_shape`` to stay on
        the validated clear branch, with the loaded droop-cancel. Returns an ARM-order 6-tuple, or ``None``
        to fall back to the fixed-shape preplace (off / no FK / no captured up-axis / no place pose / IK
        refuses -- byte-identical when both flags are off)."""
        cl_cfg = self._carton_level
        # sim-preplace-branch-select-bounded-v1: clear the time-box refusal flag for this phase entry; it
        # is set below only if the reaction-branch search blows its budget (the caller then _fails).
        self._preplace_branch_refuse = None
        if (cl_cfg is None or 'place' not in self.poses
                or self.collect_place_preplace_shape is None):
            return None
        above_support = bool(getattr(cl_cfg, 'preplace_above_support', False))
        above_place = bool(cl_cfg.preplace_above_place)
        if not (above_support or above_place):
            return None
        robot = self._fk_model()
        if robot is None:
            return None
        self._capture_level_up(measured)
        if self._carton_up_in_tool is None:
            return None
        from twin.restock_manipulation import carton_level as cl
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        place = self.poses['place']
        # Prefer the support-aware clearance (the SHELF-correct standoff); else the legacy deck raise.
        clearance = (cl_cfg.preplace_support_clearance() if above_support
                     else float(cl_cfg.preplace_standoff_m))
        standoff = (float(place[0]), float(place[1]), float(place[2]) + float(clearance))
        self._preplace_standoff_exec_rpy = None
        try:
            gc = self._place_gravity_compensation()
            limits = self._place_level_limits() or None
            tcp_link = self._place_fk_link(robot, list(self.collect_place_preplace_shape))
            if (getattr(cl_cfg, 'preplace_level_gate', False) and self._carton_base_R is not None
                    and self._carton_up_in_tool_world is not None):
                # sim-preplace-leg2-level-v1: GEOMETRIC WORLD-level standoff under the held chassis attitude.
                # coldwin56 proved the K=2850 droop feed-forward is a phantom at the preplace (the executed
                # joints matched the commanded within 0.003 rad), so solve_level_standoff(gc) pre-tilted the
                # standoff wrist_1 ~+0.031 rad and left the carton ~1.8-2.0 deg off level. Solve the standoff
                # GEOMETRICALLY (gravity_comp=None) in the WORLD frame (base_R + the world up-axis) so the
                # welded carton is parallel to the GROUND, not to the tilted base. Predicted world rpy ~0.
                self._preplace_branch_report = None
                if getattr(cl_cfg, 'preplace_reaction_branch', False):
                    # sim-preplace-wrist-continuity-v1: pick the REACTION-NATURAL standoff branch (minimum
                    # |wrist_2| == the branch the soft wrist_2 drive settles toward), so leg 1 drives to it
                    # and leg 2 holds it with no executed divergence (coldwin57's wrist_2 1.63->0.33 fault).
                    try:
                        # sim-preplace-branch-select-bounded-v1 / sim-preplace-select-live-cost-v1: the
                        # search runs the CHEAP (default) path through the cached tool-chain FK (~1.5 s on
                        # the real coldwin60 inputs, down from ~25 s) and is TIME-BOXED on the CPU clock --
                        # a budget overrun raises BranchSelectBudgetExceeded so the preplace REFUSES before
                        # motion (caught at the caller) rather than blocking the /manipulation/status
                        # heartbeat. The chosen branch is identical (within 1e-4 rad of the slow FK). The
                        # result is memoised per operation so a StageRetry / re-begin is instant.
                        _ck = self._preplace_branch_select_key(standoff, measured, tcp_link, limits)
                        _cached = self._preplace_branch_cache
                        if _cached is not None and _cached[0] == _ck:
                            sol, self._preplace_branch_report = _cached[1], _cached[2]
                            print('PREPLACE_REACTION_BRANCH cached (re-begin): chosen wrist_2=%.4f'
                                  % self._preplace_branch_report.get('chosen_wrist_2', float('nan')),
                                  flush=True)
                        elif getattr(self._carton_level, 'preplace_branch_prefer_carry_connected', False):
                            # sim-preplace-pan-first-seeded-leg2-v1: pick the standoff on the branch CONNECTED
                            # to the carry/tucked shape (wrist_2 ~ the carry wrist_2) so the pan-first leg 2 is
                            # a monotonic extension the continuity gate accepts -- NOT the min-|wrist_2| 0.668
                            # branch, which sits on a different level basin the leg-2 wrist_2 cannot reach in
                            # <= 0.15 rad steps at the standoff pan (coldwin72). The gate stays the authority.
                            # sim-preplace-branch-descent-feasible-v1 (coldwin79): also hand the selector a
                            # descent_predictor so it discards any carry-connected branch whose PLACE DESCENT
                            # would tilt the carton past the bound -- not just those the leg-2 continuity gate
                            # rejects -- and chooses deterministically among the descent-feasible ones. Off =>
                            # None => byte-identical (the greedy first-leg-2-accepted choice).
                            descent_pred = self._preplace_descent_predictor(robot, measured, tcp_link, limits)
                            sol, self._preplace_branch_report = cl.select_carry_connected_standoff(
                                robot, standoff, self._carton_up_in_tool_world,
                                list(self.collect_place_preplace_shape), list(measured), tool_link=tcp_link,
                                limits=limits, base_R=self._carton_base_R,
                                level_joints=self._carton_level.joints,
                                max_segment_rad=self._carton_level.preplace_max_segment_rad,
                                bound_deg=self._carton_level.bound_deg,
                                max_wrist_step_rad=self._carton_level.preplace_max_wrist_step_rad,
                                seeded_interior=getattr(self._carton_level, 'preplace_seeded_interior', False),
                                level_accept_deg=self._carton_level.preplace_gate_bound_deg,
                                effort_limit=self._carton_level.preplace_soft_effort_nm,
                                descent_predictor=descent_pred,
                                budget_s=self._carton_level.preplace_branch_select_budget_s)
                            sol = {j: float(sol[j]) for j in ARM_JOINTS}
                            self._preplace_branch_cache = (_ck, sol, self._preplace_branch_report)
                        else:
                            sol, self._preplace_branch_report = cl.select_reaction_natural_standoff(
                                robot, standoff, self._carton_up_in_tool_world,
                                list(self.collect_place_preplace_shape), list(measured), tool_link=tcp_link,
                                limits=limits, base_R=self._carton_base_R,
                                level_joints=self._carton_level.joints,
                                max_segment_rad=self._carton_level.preplace_max_segment_rad,
                                bound_deg=self._carton_level.bound_deg,
                                effort_limit=self._carton_level.preplace_soft_effort_nm,
                                budget_s=self._carton_level.preplace_branch_select_budget_s)
                            sol = {j: float(sol[j]) for j in ARM_JOINTS}
                            self._preplace_branch_cache = (_ck, sol, self._preplace_branch_report)
                    except cl.BranchSelectBudgetExceeded as _bud:
                        # REFUSE before motion (do NOT silently degrade to level_place_ik): a search that
                        # cannot finish in its budget would, if run inline, starve the heartbeat.
                        self._preplace_branch_refuse = str(_bud)
                        return None
                    except ValueError as _bexc:
                        print('PREPLACE_REACTION_BRANCH unsolved: %s (using the single world-level standoff)'
                              % _bexc, flush=True)
                        sol = cl.level_place_ik(robot, standoff, self._carton_up_in_tool_world,
                                                list(self.collect_place_preplace_shape), tool_link=tcp_link,
                                                limits=limits, gravity_comp=None, base_R=self._carton_base_R)
                else:
                    sol = cl.level_place_ik(robot, standoff, self._carton_up_in_tool_world,
                                            list(self.collect_place_preplace_shape), tool_link=tcp_link,
                                            limits=limits, gravity_comp=None, base_R=self._carton_base_R)
                self._preplace_standoff_exec_rpy = cl.carried_rpy(
                    robot, [float(sol[j]) for j in ARM_JOINTS], self._carton_up_in_tool_world,
                    tool_link=tcp_link, base_R=self._carton_base_R)
                print('PREPLACE_LEVEL_GATE geometric world-level standoff (base pitch=%.2f roll=%.2f deg; '
                      'droop feed-forward DISABLED at preplace -- coldwin56 phantom): predicted world '
                      'roll=%.2f pitch=%.2f tilt=%.2f deg'
                      % (self.station_hold_pitch_deg, self.station_hold_roll_deg,
                         self._preplace_standoff_exec_rpy[0], self._preplace_standoff_exec_rpy[1],
                         self._preplace_standoff_exec_rpy[2]), flush=True)
                rep = self._preplace_branch_report
                if rep is not None and rep.get('reason') == 'carry-connected':
                    # sim-preplace-pan-first-seeded-leg2-v1: the carry-connected branch was chosen (the
                    # pan-first leg-2 fix). Log the chosen wrist_2, the carry wrist travel, the leg-2 gate
                    # metrics (step + tilt) that the gate ACCEPTED, and the ranked branches.
                    print('PREPLACE_REACTION_BRANCH chosen wrist_2=%.3f rad (reason=carry-connected -- '
                          'pan-first leg 2 is a monotonic extension the continuity gate accepts); '
                          'select=%.3f s (budget %.1f s, off the host hot path -- coldwin59); carry wrist '
                          'travel=%.3f rad; leg-2 max wrist step=%.3f rad (<= %.3f) predicted tilt=%.2f deg '
                          '(<= %.2f); peak soft-wrist holding torque=%.3f N*m on %s -> margin %.1f N*m to '
                          'the %.0f N*m limit. Branches (w2, carry_travel, leg2_step, accepted): %s'
                          % (rep['chosen_wrist_2'], rep.get('select_seconds', float('nan')),
                             self._carton_level.preplace_branch_select_budget_s,
                             rep['carry_wrist_travel'], rep['leg2_max_wrist_step'],
                             self._carton_level.preplace_max_wrist_step_rad, rep['leg2_max_tilt_deg'],
                             self._carton_level.bound_deg, rep['chosen_peak_torque_nm'],
                             rep['chosen_peak_joint'], rep['chosen_margin_nm'], rep['effort_limit_nm'],
                             ', '.join('w2=%.2f(trav=%.2f,step=%s,%s)%s'
                                       % (b['wrist_2'], b['carry_wrist_travel'],
                                          ('%.3f' % b['leg2_max_wrist_step']) if b['leg2_max_wrist_step']
                                          is not None else 'unsolved',
                                          'OK' if b['accepted'] else 'rej',
                                          '*' if b['chosen'] else '') for b in rep['branches'])),
                      flush=True)
                    # sim-preplace-branch-descent-feasible-v1 (coldwin79): when the descent-feasibility filter
                    # ran, log the ranked candidates with BOTH scores (leg-2 step AND predicted place-descent
                    # tilt) and the chosen branch, so a console grep proves the selector discarded any
                    # descent-infeasible branch (e.g. coldwin79's wrist_2=0.052, descent 3.43 deg) and chose
                    # deterministically among the descent-feasible ones.
                    if rep.get('descent_feasible_gate'):
                        def _dts(b):
                            dt = b.get('descent_tilt_deg')
                            if not b.get('accepted'):
                                return 'leg2_rej'
                            if dt is None:
                                return 'descent_infeasible'
                            return ('%.2f' % dt) + ('ok' if b.get('descent_ok') else '>bound')
                        print('PREPLACE_BRANCH_CANDIDATES n=%d chosen=w2=%.3f descent_tilt=%s leg2_step=%.3f '
                              '(bound %.2f deg) -- candidates (w2: leg2_step / descent_tilt): %s'
                              % (len(rep['branches']), rep['chosen_wrist_2'],
                                 ('%.2f' % rep['chosen_descent_tilt_deg'])
                                 if rep.get('chosen_descent_tilt_deg') is not None else 'n/a',
                                 rep['leg2_max_wrist_step'], self._carton_level.bound_deg,
                                 ', '.join('w2=%.2f(%s/%s)%s'
                                           % (b['wrist_2'],
                                              ('%.3f' % b['leg2_max_wrist_step'])
                                              if b['leg2_max_wrist_step'] is not None else 'unsolved',
                                              _dts(b), '*' if b['chosen'] else '')
                                           for b in rep['branches'])), flush=True)
                elif rep is not None:
                    # sim-preplace-branch-select-bounded-v1: log the chosen branch AND the measured CPU
                    # search time (once per preplace entry) so a console grep proves it stayed off the hot
                    # path (select_seconds << the host 5 s status watchdog bound).
                    print('PREPLACE_REACTION_BRANCH chosen wrist_2=%.3f rad (settling branch, min |wrist_2|); '
                          'select=%.3f s (budget %.1f s, %s path, off the host hot path -- coldwin59); '
                          'peak soft-wrist holding torque=%.3f N*m on %s -> margin %.1f N*m to the %.0f N*m '
                          'soft-wrist effort limit (NOT saturation -- torque is reported, not the selector; '
                          'min-torque would pick wrist_2=%.3f, the uphill branch). Branches: %s'
                          % (rep['chosen_wrist_2'], rep.get('select_seconds', float('nan')),
                             self._carton_level.preplace_branch_select_budget_s,
                             'full' if rep.get('full_plan') else 'cheap',
                             rep['chosen_peak_torque_nm'], rep['chosen_peak_joint'],
                             rep['chosen_margin_nm'], rep['effort_limit_nm'], rep['min_torque_wrist_2'],
                             ', '.join('w2=%.2f(tau=%.2f,sweep=%.2f)%s'
                                       % (b['wrist_2'], b['peak_torque_nm'], b['leg1_wrist_sweep_rad'],
                                          '*' if b['chosen'] else '') for b in rep['branches'])),
                      flush=True)
            elif getattr(cl_cfg, 'preplace_level_gate', False):
                # sim-preplace-level-gate-v1 (level chassis / hold off): PREFER the standoff branch with the
                # smallest predicted EXECUTED (droop) tilt + most clearance (byte-identical to the prior path).
                sol, self._preplace_standoff_exec_rpy = cl.solve_level_standoff(
                    robot, standoff, self._carton_up_in_tool,
                    list(self.collect_place_preplace_shape), gc, tool_link=tcp_link, limits=limits)
            else:
                sol = cl.level_place_ik(robot, standoff, self._carton_up_in_tool,
                                        list(self.collect_place_preplace_shape), tool_link=tcp_link,
                                        limits=limits, gravity_comp=gc)
        except ValueError as exc:
            print('PREPLACE_ABOVE_%s_UNSOLVED: %s (falling back to the fixed preplace shape)'
                  % ('SUPPORT' if above_support else 'PLACE', exc), flush=True)
            return None
        self._preplace_above_mode = 'support' if above_support else 'place'
        self._preplace_above_clearance = float(clearance)
        return tuple(float(sol[j]) for j in ARM_JOINTS)

    def _place_fk_link(self, robot, goal):
        """The FK link the place reach targets: the motion port's IK tcp (gripper_pad_tcp) when present in
        the FK URDF, else the carton_level attitude tool_link. Its rotation equals ur_tool0's (rigid
        translation), so leveling the up-axis on it is identical."""
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        from twin.urdf_probe import link_world_transforms
        try:
            links = link_world_transforms(robot, dict(zip(ARM_JOINTS, [float(v) for v in goal])),
                                          root='ur_base_link')
            if self._place_tcp_link in links:
                return self._place_tcp_link
        except Exception:  # noqa: BLE001 -- FK probe is advisory; fall back to the attitude link
            pass
        return self._carton_level.tool_link

    def _place_gravity_compensation(self):
        """sim-carton-level-place-descent-v1: the loaded gravity-droop FEEDFORWARD the collect-deck level
        place DESCENT + RELEASE use. Returns ``carton_level.gravity_compensation`` when the host config set
        one; otherwise a DEFAULT ``GravityCompensation`` so the far-extended loaded set-down gets the
        extension-following wrist droop-cancel even under a plain ``carton_level: true`` host config (NO host
        change needed -- coldwin34 ran ``carton_level: true`` => gravity_compensation None => the geometric
        set-down drooped +2.06 deg). Scoped to the place descent: every OTHER carried phase reads gc only
        through ``_solve_level_place`` (``getattr(..., None)``) and stays byte-identical."""
        from twin.restock_manipulation import carton_level as cl
        gc = getattr(self._carton_level, 'gravity_compensation', None)
        if gc is None:
            gc = cl.GravityCompensation(tool_link=self._carton_level.tool_link).validate()
        return gc

    def _closed_loop_correct_descent(self, robot, waypoints, gc, tcp_link, measured_start=None):
        """sim-place-descent-residual-v2 (brief part 3): the opt-in, bounded, sim-only CLOSED-LOOP
        carton-level BACKSTOP applied to the planned collect place-descent waypoints. When
        ``carton_level.closed_loop_level`` is on, a ``CartonLevelServo`` reads the PREDICTED executed
        attitude (the loaded droop on each commanded waypoint) and adds a bounded wrist correction
        (+-``closed_loop_authority_deg``, rate-limited) so the executed carton stays level even if the droop
        model leaves a residual -- it never substitutes for the geometric clearance (the least-extended
        approach keeps the pad off the posts). Returns the corrected waypoint list (ARM-order 6-vectors), or
        ``waypoints`` unchanged when the backstop is off (byte-identical). Pure/CPU."""
        from twin.restock_manipulation import carton_level as cl
        clc = cl.ClosedLoopConfig.from_level(self._carton_level)
        if clc is None or self._carton_up_in_tool is None:
            return waypoints, None
        clc = cl.ClosedLoopConfig(authority_deg=clc.authority_deg, rate_deg_per_s=clc.rate_deg_per_s,
                                  gain=clc.gain, joints=clc.joints, tool_link=tcp_link)
        servo = cl.CartonLevelServo(robot, self._carton_up_in_tool, clc)
        wps = [list(w) for w in waypoints]
        # sim-place-descent-residual-v3 (part 3): when closed_loop_sensor is on and a live measured start
        # config is available, anchor the correction on the MEASURED carton attitude (the quantity the
        # grader reads) so the servo accounts for a real start offset the droop model misses; else the v2
        # model-only correction (byte-identical).
        use_sensor = (getattr(self._carton_level, 'closed_loop_sensor', False)
                      and measured_start is not None and len(measured_start) == 6
                      and all(math.isfinite(x) for x in measured_start))
        if use_sensor:
            corrected = servo.correct_path_sensor_anchored(wps, gc, list(measured_start))
            print('CARTON_LEVEL_CLOSED_LOOP_SENSOR phase=place engaged: correction anchored on the measured '
                  'carton attitude (FK of the live measured joints), authority %.2f deg'
                  % clc.authority_deg, flush=True)
        else:
            corrected = servo.correct_path(wps, gc)
        peak = max((abs(v) for v in servo.accumulated.values()), default=0.0)
        return corrected, math.degrees(peak)

    def _solve_level_place(self, robot, goal, target_pos):
        """The level-place IK in the port's reach frame. ``target_pos`` None => preserve the goal's own TCP
        (the standoff); a 3-vector => reach that sensed set-down position while level. Raises ValueError
        when no clear level solution exists (the caller decides: standoff -> fail honestly, set-down ->
        fall through to the unleveled place)."""
        from twin.restock_manipulation import carton_level as cl
        limits = self._place_level_limits() or None
        tcp_link = self._place_fk_link(robot, goal)
        # sim-carton-level-southdock-v1: loaded gravity-droop feedforward. When carton_level.gravity_compensation
        # is set, the level place/preplace IK solves for the COMMANDED config whose EXECUTED (drooped) pose is
        # level, so the physical welded carton lands parallel to the ground at the far-extended south-dock place
        # (coldwin31). Applied to BOTH the preplace standoff (``target_pos`` None, preserve its EXECUTED TCP)
        # AND the set-down (``target_pos`` given) so the whole carried descent -- compensated standoff to
        # compensated set-down -- executes level throughout, covering pre-place + place + release. None => the
        # geometric level solve (byte-identical).
        gc = getattr(self._carton_level, 'gravity_compensation', None)
        if target_pos is None:
            return cl.level_preserve_tcp(robot, goal, self._carton_up_in_tool,
                                         tool_link=tcp_link, limits=limits, gravity_comp=gc)
        return cl.level_place_ik(robot, target_pos, self._carton_up_in_tool, goal,
                                 tool_link=tcp_link, limits=limits, gravity_comp=gc)

    def _collect_retreat_liftoff_goal(self, measured):
        """sim-collect-retreat-abort-v2 / sim-collect-retreat-collider-v3: the collect deck-place
        empty-retreat LIFT-OFF config -- raise the TCP straight up by ``collect_retreat_liftoff_dz``
        (world +Z) preserving the tool orientation, seeded from the measured release config so it stays
        on the current self-clear IK branch. Bounded to the clear carry wrist region so the lift-off
        cannot drive a wrist into the ur_wrist_2-vs-forearm stop. Returns a 6-tuple (ARM order) or None
        when no FK model / no straight-up solution exists (the caller then folds directly, the prior
        behaviour -- never a regression).

        sim-collect-retreat-collider-v3 (collectlevel4-fail1): the lift-off is driven with
        ur_shoulder_pan LOCKED at the measured seed. The full-DOF DLS raise is ill-conditioned at the
        folded place pose (near a wrist singularity), so a seed perturbation below the host log's 4-dp
        rounding let it converge onto a redundant branch that ALSO raised the TCP ~0.15 m but did it by
        swinging shoulder_pan ~+0.164 rad (and tilting the tool ~0.27 rad, past liftoff_goal's own
        ori_tol -- i.e. a branch the CURRENT free solve rejects for the logged seed yet the run reached).
        That base-yaw swing rakes the whole wrist/gripper cluster ~+0.12-0.19 m sideways in base +Y,
        driving the gripper suction-panel collider into the rollwagen front-LEFT cage post (arm/gripper-
        vs-cage is NOT collision-filtered -- only carton-vs-cage is), which pins the saturated pan drive
        ~0.204 rad short while the distal wrists velocity-saturate -- the recorded contact-pin fingerprint.
        A straight-up lift needs NO base-yaw rotation; locking pan removes that redundant DOF so the raise
        is the deterministic elbow+wrist branch (FK-proven: pan-locked still reaches the full dz straight
        up with ~1 mm horizontal drift and keeps the gripper >=~0.09 m from the post). shoulder_lift is
        deliberately NOT locked (locking it too makes the raise unreachable at any dz)."""
        robot = self._fk_model()
        if robot is None or len(measured) != 6 or not all(math.isfinite(x) for x in measured):
            return None
        from twin.restock_manipulation.collect_retreat import liftoff_goal
        from twin.restock_manipulation.cad_tool import ARM_JOINTS
        big = 2 * math.pi
        seed = dict(zip(ARM_JOINTS, [float(v) for v in measured]))
        # Lock the base yaw at the seed so the straight-up lift-off cannot swing the cluster into the
        # cart cage (sim-collect-retreat-collider-v3). A straight +Z raise needs no shoulder_pan motion.
        pan_seed = seed['ur_shoulder_pan_joint']
        limits = {'ur_shoulder_pan_joint': (pan_seed, pan_seed)}
        if self.carry_wrist1_cap is not None:
            limits['ur_wrist_1_joint'] = (-big, float(self.carry_wrist1_cap))
        if self.carry_wrist2_min is not None:
            limits['ur_wrist_2_joint'] = (float(self.carry_wrist2_min), big)
        try:
            sol = liftoff_goal(robot, seed, dz=self.collect_retreat_liftoff_dz,
                               tcp_link=self._place_tcp_link, limits=limits or None)
        except ValueError as exc:
            print('COLLECT_RETREAT_LIFTOFF_UNSOLVED: %s (folding straight to the preplace shape)' % exc,
                  flush=True)
            return None
        # Belt-and-suspenders: never issue a lift-off that swings the base yaw into the cage, even if a
        # future liftoff_goal change were to relax the pan lock. A straight-up lift keeps pan at the seed.
        if abs(float(sol['ur_shoulder_pan_joint']) - pan_seed) > COLLECT_RETREAT_LIFTOFF_PAN_TOL:
            print('COLLECT_RETREAT_LIFTOFF_UNSOLVED: lift-off would swing ur_shoulder_pan %.4f rad off '
                  'the seed (> %.4f) -- rejecting the cage-raking branch, folding straight to the '
                  'preplace shape' % (float(sol['ur_shoulder_pan_joint']) - pan_seed,
                                      COLLECT_RETREAT_LIFTOFF_PAN_TOL), flush=True)
            return None
        return tuple(float(sol[j]) for j in ARM_JOINTS)

    def _begin_retreat_after_detach(self, now):
        """sim-release-threshold-carton-frame-v1: entry point after the weld detach is confirmed. When the
        opt-in free-and-resting gate is on, HOLD in the release phase and let the free-check sub-state own the
        transition to retreat (prove the carton is free+resting, clearance-lift, then retreat / RELEASE_NOT_FREE).
        Off => transition straight to retreat, byte-identical with the prior ``self._transition('retreat')``."""
        if self._release_free is not None:
            self._release_free.reset()
            self._rf_stage = 'settle'
            self._released_detached = True
            print('RELEASE_FREE_BEGIN phase=release now=%.3f: weld detached -- holding to prove the carton is '
                  'free of the gripper and resting before the retreat' % now, flush=True)
            return
        self._transition('retreat')

    def _release_free_step(self, now, s):
        """sim-release-threshold-carton-frame-v1: the post-detach free-and-resting + clearance-lift sub-state
        (only reached when ``release_free_check`` is on and the detach has been confirmed). Stages:
          * 'settle': HOLD (no retreat motion) until the measured carton CENTRE z is stable AND the carton
            BOTTOM is resting on the support for ``settle_ticks`` ticks, then command a pure vertical
            clearance lift; refuse ``RELEASE_NOT_FREE`` if the carton never comes to rest.
          * 'lift': poll the clearance lift; on completion verify the carton did NOT follow the gripper up,
            then transition to retreat (the lateral fold) or refuse ``RELEASE_NOT_FREE``.
        A missing measured ``placed_pose`` (real rig / verify_place off) skips the check -> straight retreat."""
        gate = self._release_free
        if s.placed_pose is None or len(s.placed_pose) < 3 or not math.isfinite(float(s.placed_pose[2])):
            print('RELEASE_FREE_SKIP phase=release now=%.3f: no measured carton pose -- retreating without '
                  'the free-and-resting check' % now, flush=True)
            self._released_detached = False
            self._transition('retreat')
            return self._result()
        carton_z = float(s.placed_pose[2])
        half_h = self.profile.dimensions[2] / 2.0
        bottom_gap = (carton_z - half_h) - float(self.target_surface[2])
        if self._rf_stage == 'settle':
            dec = gate.settle(carton_z=carton_z, bottom_gap=bottom_gap)
            if dec.action == 'refuse':
                return self._fail('RELEASE_NOT_FREE: ' + dec.reason)
            if dec.action == 'lift':
                liftoff = self._collect_retreat_liftoff_goal(s.joint_positions)
                if liftoff is None:
                    # No straight-up IK solution from here: skip the clearance lift (no regression vs the
                    # prior straight-to-retreat behaviour) and retreat. The settle check still ran.
                    print('RELEASE_FREE_LIFT_UNSOLVED phase=release now=%.3f: %s -- retreating without the '
                          'clearance lift' % (now, dec.reason), flush=True)
                    self.evidence.append({'release_free_lift_unsolved': {'sim_time': float(now)}})
                    self._released_detached = False
                    self._transition('retreat')
                    return self._result()
                print('RELEASE_FREE_LIFT phase=release now=%.3f carton_z=%.4f bottom_gap=%.4f: %s'
                      % (now, carton_z, bottom_gap, dec.reason), flush=True)
                self.evidence.append({'release_free_lift': {
                    'sim_time': float(now), 'carton_z': carton_z, 'bottom_gap': float(bottom_gap)}})
                self.motion.begin_joint_goal(liftoff, s.joint_positions, straight_line=True)
                self._rf_stage = 'lift'
                self._retreat_lifted = True   # the retreat skips its own lift-off (the clearance lift did it)
                return self._result(({'kind': 'move', 'pose': None, 'phase': 'release'},))
            # 'settle': keep holding (no motion issued; the arm keeps its detached pose)
            return self._result()
        # self._rf_stage == 'lift'
        try:
            outcome = self.motion.poll()
        except Exception as exc:  # noqa: BLE001 -- route a raise through the staged retry like other moves
            return self._raised_move_exc(exc, staged=True)
        if outcome == 'failed':
            return self._fail('RELEASE_NOT_FREE: the post-detach vertical clearance lift failed to plan/execute')
        if outcome == 'succeeded':
            dec = gate.follow(carton_z=carton_z)
            if dec.action == 'refuse':
                return self._fail('RELEASE_NOT_FREE: ' + dec.reason)
            print('RELEASE_FREE_CLEAR phase=release now=%.3f carton_z=%.4f: %s -> retreat'
                  % (now, carton_z, dec.reason), flush=True)
            self.evidence.append({'release_free_clear': {
                'sim_time': float(now), 'carton_z': carton_z,
                'rise': (None if dec.rise is None else float(dec.rise))}})
            self._released_detached = False
            self._transition('retreat')
            return self._result()
        if outcome != 'running':
            return self._fail('invalid release clearance-lift motion outcome')
        return self._result()

    def _carton_level_frame(self, phase):
        """sim-preplace-leg2-level-v1: the ``(up_in_tool, base_R)`` the carton-level measurement uses for
        ``phase``. The collect PREPLACE under the held chassis attitude is levelled IN WORLD, so its trip /
        tick / segment must MEASURE in world too (world up-axis + ``base_R``) -- otherwise the base-frame
        metric reads the world-level carton ~2.3 deg off (the base pitch twice) and false-trips. Every other
        phase (and the hold-off case) keeps the base frame, byte-identical. Returns the base-frame up and
        ``None`` when the world up-axis is unavailable (never a silent frame mismatch)."""
        if (phase == 'preplace' and self._carton_base_R is not None
                and self._carton_up_in_tool_world is not None):
            return self._carton_up_in_tool_world, self._carton_base_R
        return self._carton_up_in_tool, None

    def _carton_tilt_trip(self, measured):
        """CARTON_TILT trip message when the welded carton (this carried pose) leaves level by more
        than the bound, else None. No-op when off / trip disabled / no FK / up-axis not yet captured."""
        if self._carton_level is None or not self._carton_level.trip:
            return None
        robot = self._fk_model()
        if robot is None or self._carton_up_in_tool is None:
            return None
        from twin.restock_manipulation import carton_level as cl
        up, base_R = self._carton_level_frame(self.phase)
        roll, pitch, _ = cl.carried_rpy(robot, list(measured), up,
                                        self._carton_level.tool_link, base_R=base_R)
        out = cl.tilt_trip(roll, pitch, self._carton_level.bound_deg)
        return out[0] if out else None

    def _handle_preplace_convergence(self, now, sensors):
        """sim-preplace-wrist2-hold-v1 / sim-preplace-per-joint-gains-v1: observe the carried-preplace
        convergence guard and act on its verdict. Returns ``(handled, result)``: ``handled`` True means
        ``step`` returns ``result`` (a drift re-capture re-issue, or a terminal PHASE_CONVERGENCE fail);
        False means continue. A :class:`DriftRecapture` re-issues the armed preplace goal to re-drive a
        back-driven joint (logged ONCE per preplace via ``_preplace_recapture_logged``) instead of faulting;
        a :class:`ConvergenceStall` fails by name as before. Pre-conditions (phase/finite joints/enabled)
        are checked by the caller."""
        from twin.restock_manipulation.convergence_guard import DriftRecapture
        verdict = self._preplace_convergence.observe(now, sensors.joint_positions)
        if isinstance(verdict, DriftRecapture):
            goal = self._preplace_convergence.goal
            if goal is None:
                return False, None
            if not self._preplace_recapture_logged:
                self._preplace_recapture_logged = True
                print(verdict.message, flush=True)
            self.evidence.append({'phase_drift_recapture': {
                'joint': verdict.joint, 'from': verdict.from_pos, 'to': verdict.to_pos,
                'error': verdict.error, 'best_error': verdict.best_error,
                'count': verdict.count, 'limit': verdict.limit, 'sim_time': float(now)}})
            self.motion.begin_joint_goal(tuple(goal), sensors.joint_positions, slow=True, goal_tol=0.2)
            self.issued = True
            return True, self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))
        if verdict is not None:
            print(verdict.message, flush=True)
            self.evidence.append({'phase_convergence': {'joint': verdict.joint, 'error': verdict.error,
                                                        'best_error': verdict.best_error,
                                                        'stalled_s': verdict.stalled_s,
                                                        'sim_time': float(now)}})
            return True, self._fail(verdict.message)
        return False, None

    def _carton_tilt_observe(self, now, measured):
        """sim-carton-tilt-transient-debounce-v1: the DEBOUNCED carried-tilt decision for this tick.

        Measures the welded carton's roll/pitch (FK from the measured joints) and advances the sustained-
        tilt debounce, returning a ``carton_level.TiltVerdict`` (``.trip`` = the CARTON_TILT fault message
        once the tilt has stayed over the bound for the whole grace window; ``.transient`` = the
        CARTON_TILT_TRANSIENT log line for a sub-grace spike that just ended). Returns ``None`` when the
        constraint/trip is off, there is no FK model, or the up-axis has not been captured yet (same gate
        as ``_carton_tilt_trip``). The grace is rebased onto the running physics step so the window is a
        constant frame-count budget (never widened); grace 0 reproduces the immediate trip."""
        if self._carton_level is None or not self._carton_level.trip:
            return None
        robot = self._fk_model()
        if robot is None or self._carton_up_in_tool is None:
            return None
        from twin.restock_manipulation import carton_level as cl
        if self._tilt_debounce is None:
            self._tilt_debounce = cl.TiltDebounce(
                bound_deg=self._carton_level.bound_deg,
                grace_s=self._carton_level.tilt_grace_s * self._freshness_scale)
        up, base_R = self._carton_level_frame(self.phase)
        roll, pitch, tilt = cl.carried_rpy(robot, list(measured), up,
                                           self._carton_level.tool_link, base_R=base_R)
        # sim-carton-level-preplace-v2: accumulate the EXECUTED max |roll|,|pitch| for this carried
        # phase (reported vs the planned peak at the phase exit). Cheap (reuses this tick's FK read).
        er, ep = self._carton_seg_exec.get(self.phase, (0.0, 0.0))
        self._carton_seg_exec[self.phase] = (max(er, abs(roll)), max(ep, abs(pitch)))
        # sim-preplace-sweep-tilt-v1: per-tick carried-phase diagnostic at ~10 Hz SIM time. Logs the
        # welded carton attitude (the grader quantity) AND the executed joint positions, so a coldwin53
        # trip is diagnosable from the console directly (the dynamic-swing characterisation had to
        # reconstruct this from the commanded trajectory because it was never logged).
        if self._carton_tick_log_t is None or (now - self._carton_tick_log_t) >= 0.1:
            self._carton_tick_log_t = now
            joints = ' '.join('%.4f' % float(v) for v in measured)
            print('CARTON_TICK phase=%s now=%.3f roll=%.2f pitch=%.2f tilt=%.2f deg | q=[%s]'
                  % (self.phase, now, roll, pitch, tilt, joints), flush=True)
        return self._tilt_debounce.observe(now, roll, pitch)

    def _carton_seg_sample(self, phase, measured):
        """sim-carton-level-setdown-v1: stash-only executed-tilt sample for a NON-trip carried phase
        (release): measure the welded carton's |roll|,|pitch| from the FK of ``measured`` and accumulate the
        max under ``phase`` for the CARTON_SEGMENT_TILT report at the phase exit. Never trips / fails the
        move (the gripper is opening). Inert (no-op) when carton_level, the FK model, or the up-axis is off."""
        if self._carton_level is None:
            return
        robot = self._fk_model()
        if robot is None or self._carton_up_in_tool is None:
            return
        from twin.restock_manipulation import carton_level as cl
        up, base_R = self._carton_level_frame(phase)
        roll, pitch, _ = cl.carried_rpy(robot, list(measured), up,
                                        self._carton_level.tool_link, base_R=base_R)
        er, ep = self._carton_seg_exec.get(phase, (0.0, 0.0))
        self._carton_seg_exec[phase] = (max(er, abs(roll)), max(ep, abs(pitch)))

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

    def _planner_crash_failfast_or_none(self):
        """sim-robust-collect-stow-stall-v1: when the motion port reports a terminal move_group planner
        crash/hang (last_fault_was_planner_unrecoverable -- a sim-clock PLANNING deadline whose bounded
        crash re-sends were exhausted because the planner did not respawn), FAIL the op FAST with a NAMED
        cause instead of looping our own move retries (MOVE_DEADLINE_RETRY / STAGED_MOVE_RETRY) to the
        per-phase timeout. coldliveapproach: the post-release collect STOW consumed the full 120-sim-s
        phase timeout re-sending plans to a crashed-and-not-respawned move_group, reported only as the
        opaque 'phase timeout: stow'. Returns a terminal _fail Result, or None when the port has no such
        fault (then the caller keeps its normal retry path -- byte-identical). Checked before the sim-
        deadline / staged retries so the terminal planner fault always wins."""
        probe = getattr(self.motion, 'last_fault_was_planner_unrecoverable', None)
        if callable(probe) and probe():
            return self._fail('%s aborted: the move_group planner crashed or hung (a plan goal was never '
                              'accepted / no plan returned within the sim planning budget) and did not '
                              'respawn -- failing fast' % self.phase)
        # sim-robust-movegroup-double-death-v1 (gap 1, coldwin4): the carried carton could not be re-attached
        # to the respawned move_group's planning scene within the bounded sim budget. The carton is NOT in
        # the scene, so re-planning would plan straight through it -- fail the op BY NAME rather than resend
        # carton-less (the coldwin4 recovery gap). Checked beside the planner-crash fast-fail; no such fault
        # on a healthy or no-carry recovery -> None (byte-identical).
        reattach = getattr(self.motion, 'last_fault_was_reattach_failure', None)
        if callable(reattach) and reattach():
            return self._fail('%s aborted: the carried carton could not be re-attached to the respawned '
                              'move_group planning scene within the re-attach budget -- failing fast rather '
                              'than planning without the carton' % self.phase)
        return None

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

    def _crash_recovery_active(self):
        """sim-robust-phase-timeout-during-recovery-v1: True iff the motion adapter reports it is inside a
        NAMED move_group crash recovery right now (respawn hold / post-respawn re-attach / an observed
        move_group death still being recovered). Read through getattr so a motion stub without the probe
        (and the real rig) => False => byte-identical phase-timeout accounting."""
        probe = getattr(self.motion, 'crash_recovery_active', None)
        return bool(probe()) if callable(probe) else False

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
        # sim-robust-phase-timeout-during-recovery-v1: while the motion adapter is inside a NAMED move_group
        # crash recovery, credit this dt out of the phase-timeout budget (elapsed - credit vs phase_timeout)
        # so a recovered crash is not charged as 'phase timeout' (respawn2b: two deaths ate ~119 of 120 s).
        # Bounded by _recovery_pause_cap_s: a recovery that outruns the bound FAILS FAST by name here (never
        # pauses forever). Disabled (cap 0) or no recovery => credit stays 0 => the comparison is exactly the
        # legacy `elapsed > phase_timeout` => byte-identical.
        if self._recovery_pause_cap_s > 0 and self._crash_recovery_active():
            if self._recovery_credit_s >= self._recovery_pause_cap_s:
                return self._fail('crash recovery exceeded its %.0fs sim pause budget in phase %s: a named '
                                  'move_group crash recovery did not complete within the bound -- failing '
                                  'fast by name instead of pausing the phase clock indefinitely'
                                  % (self._recovery_pause_cap_s, self.phase))
            self._recovery_credit_s = min(self._recovery_pause_cap_s, self._recovery_credit_s + dt)
        if self.elapsed - self._recovery_credit_s > self.phase_timeout:
            self._note_recovery_pause()   # name any credited recovery time before the (honest) timeout
            return self._fail('phase timeout: ' + self.phase)
        # sim-robust-esdf-boot-gate-v1: boot ESDF-incorporation gate (opt-in HOST key, 0 => off =>
        # byte-identical). Checked BEFORE the collision_dt invariant and before any motion streams.
        # HOLD (no motion) until the first valid incorporation ack, FAIL FAST by name if the sim-clock
        # grace elapses with collision_time still at the -1 sentinel, PROCEED once a valid ack has
        # arrived (then the collision_dt invariant below owns freshness).
        _boot = self._esdf_boot_gate.observe(now, sensors.collision_time)
        if _boot is esdf_boot_gate.HOLD:
            print('ESDF_BOOT_GATE_HOLD phase=%s now=%.3f collision_time=%.3f (grace %.1fs)'
                  % (self.phase, now, sensors.collision_time, self._esdf_boot_gate.grace_s), flush=True)
            return self._result()
        if _boot is not esdf_boot_gate.PROCEED:
            print('ESDF_BOOT_GATE_TRIP phase=%s now=%.3f %s' % (self.phase, now, _boot), flush=True)
            return self._fail(_boot)
        # sim-robust-esdf-boot-gate-v2: log the collision_time sentinel -> first valid stamp transition
        # ONCE per op (cheap one-shot), so the gate's PROCEED decision is never again invisible -- the
        # value, the sim now, and the source are recorded exactly where the gate hands freshness to the
        # collision_dt invariant below. Fires whether the first step already carried a valid ack (no HOLD,
        # the healthy case) or a HOLD preceded it (the one-tick boot race). Only when the gate is enabled,
        # so a gate-off run stays byte-identical on the console too.
        if self._esdf_boot_gate.consume_just_satisfied():
            print('ESDF_BOOT_GATE_READY phase=%s now=%.3f collision_time=%.3f (first valid nvblox '
                  'incorporation ack: the mirrored station collision_time left the -1 not-ready sentinel; '
                  'source=sensors.collision_time packet mirror; collision_dt invariant now owns freshness)'
                  % (self.phase, now, sensors.collision_time), flush=True)
        collision_fresh = self._fresh(sensors.collision_time, now)
        # sim-place-release-on-contact-v1: the CONTACT-GATED place-descent weld release (supersedes the fixed
        # carton_z gate below when on). coldwin68's carton jammed on cage post /Rollwagen/chassis/post_2 at the
        # descent START (carton_z frozen 0.075 m above the fixed z threshold), so the z gate never fired and the
        # stiff place drive pressed to 757 N*m past the 600 N*m guard. The gate releases the weld the instant
        # the support is ACTUALLY touched (effort over the pre-contact hold baseline | descent stall | the z
        # threshold, whichever first), GATED on the touched body being the support: a flush contact on a
        # NON-support body (a cage post) is an OBSTRUCTION -- do not press/release; log PLACE_OBSTRUCTED and
        # either do a bounded lateral re-approach or fail by name. Reuses the PR 334 cancel->wait-terminal->
        # detach release serialization (shared with the legacy gate below). Scoped to the collect deck place.
        _prg = self._place_release_gate
        if (_prg is not None and self.held and self.phase == 'place' and self._collect_deck_place()):
            _pz = (float(sensors.placed_pose[2])
                   if (sensors.placed_pose is not None and len(sensors.placed_pose) >= 3
                       and math.isfinite(float(sensors.placed_pose[2]))) else None)
            _dec = _prg.observe(now=now, carton_z=_pz, commanded_down=True,
                                arm_effort_max=sensors.arm_effort_max, seal_gap_mm=sensors.seal_gap_mm,
                                seal_hit_prims=sensors.seal_hit_prims)
            # Per-tick place carton_z + contact-prim heartbeat (throttled ~0.5 sim-s; every non-press tick).
            if (_dec.action != 'press' or self._place_release_log_sim is None
                    or now - self._place_release_log_sim >= 0.5):
                self._place_release_log_sim = now
                print('PLACE_RELEASE_TICK phase=place now=%.3f carton_z=%s effort=%s baseline=%s stall=%d '
                      'gap_mm=%s hit=%s action=%s trigger=%s'
                      % (now, 'n/a' if _pz is None else '%.4f' % _pz,
                         'n/a' if _dec.effort is None else '%.1f' % _dec.effort,
                         'n/a' if _dec.baseline is None else '%.1f' % _dec.baseline, _dec.stall_count,
                         'n/a' if sensors.seal_gap_mm is None else '%.2f' % float(sensors.seal_gap_mm),
                         ','.join(sensors.seal_hit_prims) if sensors.seal_hit_prims else '-',
                         _dec.action, _dec.trigger or '-'), flush=True)
            if _dec.action == 'obstructed':
                _res = self._place_obstructed(now, _dec)
                if _res is not None:
                    return _res
            elif _dec.action == 'fault':
                # sim-place-seat-geometry-v1: the descent reached/passed the modelled seat with no measured
                # contact -- the support is not where modelled. Fail by name rather than releasing into a fall.
                return self._fail('PLACE_SEAT_FAULT: ' + _dec.reason)
            elif _dec.action == 'release':
                from twin.restock_manipulation import carton_level as cl
                print('%s phase=place now=%.3f trigger=%s carton_z=%s: the descending welded carton touched '
                      'the SUPPORT -- releasing the weld at contact (cancel the in-flight place execute, then '
                      'gripper-open + detach from the settled pose) instead of pressing it; %s'
                      % (cl.PLACE_CONTACT_RELEASE, now, _dec.trigger,
                         'n/a' if _pz is None else '%.4f' % _pz, _dec.reason), flush=True)
                self.evidence.append({'place_contact_release': {
                    'trigger': _dec.trigger, 'carton_z': _pz, 'effort': _dec.effort,
                    'baseline': _dec.baseline, 'release_z': self._place_contact_release_z,
                    'sim_time': float(now)}})
                self._place_contact_release_z = None
                self._release_cancel_pending = True
                self._release_cancel_issued = False
                self._release_cancel_sim_start = now
                self._transition('release')
                return self._result()
            # action == 'press': keep descending (fall through -- the legacy z gate below is disabled when
            # place_release_on_contact is on, so no double-fire).
        # sim-place-descent-residual-v3: CONTACT-TRIGGERED weld release. The reach-independent ~2.0 deg
        # collect set-down residual is the arm PRESSING the leveled welded carton onto the cart deck -- the
        # CARTON_TILT grader reads FK of the EXECUTED joints, which deflect as the position-controlled arm
        # drives into the rigid deck reaction (no FK model sees the contact; that is why it is reach-
        # independent and model-blind). The fix is to NOT press: watch the welded carton descend (its measured
        # centre z) and, the instant it reaches within place_contact_gap_m of the planned set-down height
        # (deck contact), RELEASE the weld (transition to the release phase: gripper-open + detach) instead of
        # continuing the press. Fires BEFORE the carried-phase CARTON_TILT trip below, so the executed-joint
        # deflection never builds and the trip never sustains; the carton settles flat on the deck under
        # gravity. Scoped to the collect deck place; inert (threshold None) when off / no measured carton pose.
        # sim-place-release-serialization-v1: the place EXECUTE is STILL IN FLIGHT when this gate fires mid-
        # descent (coldwin43). The detach is a planning-scene mutation and must NEVER be issued while the
        # adapter is in a plan/execute state (the _PLAN_IN_FLIGHT_STATES invariant -- coldwin43 tripped it:
        # the release detach ran at state=execute_result -> HOST_FAULT_ESTOP). So make the release a PROPER
        # phase transition: CANCEL the in-flight place move now (the cancel is the stop -- the arm must not
        # keep pressing the carton into the deck), then the release phase waits for the cancel's terminal
        # result and detaches only from the settled, not-in-flight state. The cancel fires NO global emergency
        # stop (which would latch permanently in the Isaac host); it is an intended phase transition.
        if (self._carton_level is not None and self._carton_level.place_contact_release
                and not getattr(self._carton_level, 'place_release_on_contact', False)
                and self.held
                and self.phase == 'place' and self._place_contact_release_z is not None
                and self._collect_deck_place()
                and sensors.placed_pose is not None and len(sensors.placed_pose) >= 3
                and math.isfinite(float(sensors.placed_pose[2]))
                and float(sensors.placed_pose[2]) <= self._place_contact_release_z):
            from twin.restock_manipulation import carton_level as cl
            print('%s phase=place now=%.3f carton_z=%.4f (<= release_z %.4f): the welded carton reached the '
                  'cart deck -- releasing the weld at contact (cancel the in-flight place execute, then '
                  'gripper-open + detach from the settled pose) instead of pressing it into the deck; the '
                  'carton settles flat under gravity and the CARTON_TILT trip, which only grades the carried '
                  'carton, is pre-empted the instant we leave the place phase'
                  % (cl.PLACE_CONTACT_RELEASE, now, float(sensors.placed_pose[2]),
                     self._place_contact_release_z), flush=True)
            self.evidence.append({'place_contact_release': {'carton_z': float(sensors.placed_pose[2]),
                                                            'release_z': float(self._place_contact_release_z),
                                                            'sim_time': float(now)}})
            self._place_contact_release_z = None   # one-shot
            # Owe a cancel+settle before the release detach (the release phase drives it). Record the sim
            # time to bound the settle wait. _transition resets issued/elapsed; the release handler sees
            # _release_cancel_pending and serializes cancel -> wait-terminal -> detach.
            self._release_cancel_pending = True
            self._release_cancel_issued = False
            self._release_cancel_sim_start = now
            self._transition('release')
            return self._result()
        # sim-carton-level-carry-v1: CARTON_TILT trip. While the carton is WELDED and in a carried
        # phase, capture its level up-axis once (the grasp pose rests level on the shelf) and fail
        # the move by name if the measured carton leaves level by more than the bound -- never carry
        # a visibly tilted carton silently. Off (carton_level None) / no FK / non-6 pose => inert =>
        # byte-identical. Checked before motion streams so a tilt stops the arm, not the video.
        # sim-carton-tilt-transient-debounce-v1: the trip is DEBOUNCED over a sustained sim-time grace
        # window. A one-to-few-tick tracking/impulse spike over the bound (collectacc1: the lift re-plan
        # restart after a move_group respawn -- the planned path is level, CPU-proven) is logged as
        # CARTON_TILT_TRANSIENT and does NOT abort; only a tilt held over the bound for the whole window
        # fails. The bound is unchanged; grace 0 => legacy immediate trip => byte-identical.
        if (self._carton_level is not None and self.held and self.phase in self._CARRIED_PHASES
                and len(sensors.joint_positions) == 6
                and all(math.isfinite(x) for x in sensors.joint_positions)):
            self._capture_level_up(sensors.joint_positions)
            _verdict = self._carton_tilt_observe(now, sensors.joint_positions)
            if _verdict is not None:
                if _verdict.transient is not None:
                    print('%s phase=%s now=%.3f' % (_verdict.transient, self.phase, now), flush=True)
                if _verdict.trip is not None:
                    print(_verdict.trip, flush=True)
                    return self._fail(_verdict.trip)
        # sim-preplace-per-joint-gains-v1 (fixed sim-preplace-convergence-guard-fix-v1): frozen-joint
        # convergence guard. During preplace, watch per-joint convergence of the issued standoff goal; if a
        # DRIVEN joint (one that has moved from the armed start -- not an idle, not-yet-swept joint) stops
        # converging (jams short or RETREATS) for the sim-second stall window while still un-converged, NAME
        # it (joint + commanded/executed) as PHASE_CONVERGENCE and fail fast -- instead of running the opaque
        # 120 s phase timeout (coldwin53: shoulder_pan was driven back 0.56 rad by the wrist_2 sweep). Inert
        # when stall_s<=0 / no goal armed / the move is converging / a joint is idle => byte-identical.
        if (self.phase == 'preplace' and self._preplace_convergence.enabled
                and len(sensors.joint_positions) == 6
                and all(math.isfinite(x) for x in sensors.joint_positions)):
            _handled, _res = self._handle_preplace_convergence(now, sensors)
            if _handled:
                return _res
        # sim-carton-level-setdown-v1: RELEASE is post-set-down (gripper opening before the weld detaches)
        # so it is NOT a trip-constrained carried phase, but the welded carton is still held at the set-down
        # pose, so SAMPLE its executed tilt (no trip) and stash it under 'release' -- _transition then emits
        # the CARTON_SEGMENT_TILT phase=release line, giving the integration worker the release attitude the
        # brief asked for (the full carried descent -- set-down AND release -- reported without a trip).
        # Stash-only (never fails the move); inert when carton_level / the FK model / the up-axis is off.
        elif (self._carton_level is not None and self.held and self.phase == 'release'
              and len(sensors.joint_positions) == 6
              and all(math.isfinite(x) for x in sensors.joint_positions)):
            self._carton_seg_sample('release', sensors.joint_positions)
        # fm430 (2a) fallback: ride the post-attach LIFT (to the fixed clear-region carry_lift_shape, which
        # does NOT depend on the live collision world) THROUGH a brief ESDF-re-incorporation staleness for
        # a bounded WALL grace -- proceed (treat collision as fresh for this pre-validated move), then FAIL
        # FAST with a named reason if the collision world stays stale past the grace. Scoped to held +
        # phase 'lift' + carry_lift_shape set + collision-SOLE staleness (every other invariant field must
        # be good, so it can never mask a base-moving / feedback / pre-grasp fault). 0 s => off => the
        # fm419(a) pause path (byte-identical). Resets when collision is fresh again (below).
        if (not collision_fresh and self.lift_collision_grace_wall_s > 0 and self.held
                and self.phase == 'lift' and self.carry_lift_shape is not None
                and sensors.parked is True and type(sensors.contact) is bool
                and self._fresh(sensors.feedback_time, now) and len(sensors.joint_positions) == 6
                and all(math.isfinite(x) for x in sensors.joint_positions)
                and math.isfinite(sensors.collision_time) and now - sensors.collision_time > self.max_age):
            wall = self.steady_clock()
            if self._lift_grace_wall_start is None:
                self._lift_grace_wall_start = wall
            if wall - self._lift_grace_wall_start > self.lift_collision_grace_wall_s:
                return self._fail('held-lift collision grace exceeded %.1fs wall: the collision world '
                                  'stayed stale (collision_dt=%.4f) through the post-attach lift to the '
                                  'fixed clear-region goal -- failing fast'
                                  % (wall - self._lift_grace_wall_start, now - sensors.collision_time))
            print('LIFT_COLLISION_GRACE phase=lift now=%.3f collision_dt=%.4f (grace %.1fs) -- riding the '
                  'pre-validated clear-region lift through the post-attach ESDF lag'
                  % (now, now - sensors.collision_time, self.lift_collision_grace_wall_s), flush=True)
            collision_fresh = True   # proceed with THIS pre-validated lift despite the stale collision world
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
                # fm427 (A1): config-driven (collision_pause_wall_s > 0; default 0 => the legacy
                # empty-control pause, byte-identical). When enabled: (1) WALL-bound the pause so it
                # fails fast with a named reason instead of silently starving the 90 s arm-command
                # heartbeat (run-323 cold lift), and (2) keep the stream ALIVE -- re-command a HOLD at
                # the CURRENT measured joints each tick (stationary: the goal IS the measured pose, so
                # the arm never advances the held move into the stale collision world) so the host keeps
                # publishing an arm command and the heartbeat is fed. The keep-alive needs the motion
                # port's begin_joint_goal (the mission path has it); without it the wall bound is the
                # sole backstop.
                if self.collision_pause_wall_s > 0:
                    wall = self.steady_clock()
                    if self._pause_wall_start is None:
                        self._pause_wall_start = wall
                    if wall - self._pause_wall_start > self.collision_pause_wall_s:
                        return self._fail('collision-stale pause exceeded %.1fs wall at phase=%s (%s); '
                                          'the collision world stayed stale -- failing fast before the '
                                          'arm-command heartbeat watchdog'
                                          % (wall - self._pause_wall_start, self.phase, why[0]))
                    # BEST-EFFORT keep-alive: only meaningful when the motion port is IDLE (no active
                    # trajectory). If a move is still active the port raises ('motion already active',
                    # ros_adapter.py) -- and that active trajectory is ALREADY feeding the arm-command
                    # stream, so the hold is unnecessary; swallow the raise (never crash the pause --
                    # run-324 cold: an uncaught RuntimeError here killed the step -> heartbeat lost). The
                    # WALL budget above is the guaranteed backstop either way.
                    if (len(sensors.joint_positions) == 6
                            and callable(getattr(self.motion, 'begin_joint_goal', None))):
                        try:
                            self.motion.begin_joint_goal(tuple(sensors.joint_positions),
                                                         tuple(sensors.joint_positions))
                        except Exception as exc:   # noqa: BLE001 -- active move already feeds the stream
                            print('COLLISION_STALE_PAUSE hold skipped (%s: %s)'
                                  % (type(exc).__name__, exc), flush=True)
                print('COLLISION_STALE_PAUSE phase=%s now=%.3f %s (max_age %.3f)' % (
                    self.phase, now, why[0], self.collision_stale_tol_s), flush=True)
                return self._result()   # running; a hold (if idle) + the wall budget feed/bound the pause
            print('CONTROLLER_INVARIANT_TRIP phase=%s now=%.3f %s' % (self.phase, now, ' '.join(why)),
                  flush=True)
            return self._fail('base moving or stale/missing feedback/collision world [' + ','.join(why) + ']')
        self._pause_wall_start = None   # fm427 (A1): invariant passed (collision fresh) -> pause ended
        if collision_fresh and (not math.isfinite(sensors.collision_time)
                                or now - sensors.collision_time <= self.max_age):
            self._lift_grace_wall_start = None   # fm430 (2a): collision genuinely fresh -> grace resets
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
        # Sim-clock pre-weld press budget (grasp_press_timeout_s; default 0 => off => byte-identical).
        # Bounds the WHOLE unconfirmed press across the contact descent and the grasp confirm/retry loop:
        # the clock latches when the press first begins and clears only once the press is OVER (a CONFIRMED
        # held carton, or the phase has advanced past the press) -- so a never-sealing press or a flickering
        # contact that never confirms fails fast with a NAMED reason instead of looping until the opaque
        # wall arm-command heartbeat watchdog. See __init__.
        if self.grasp_press_timeout_s > 0:
            if self.phase in ('contact', 'grasp') and not self.held:
                if self._press_sim_start is None:
                    self._press_sim_start = now
                elif now - self._press_sim_start > self.grasp_press_timeout_s:
                    return self._fail('grasp press exceeded %.1fs sim without a confirmed held carton '
                                      '(native suction did not latch and no weld formed across the contact '
                                      'descent and the grasp confirm/retry loop) -- failing fast before the '
                                      'arm-command heartbeat watchdog' % self.grasp_press_timeout_s)
            else:
                self._press_sim_start = None
        # sim-arm-reactive-replan-v1 (P1): human-aware RUN/SLOW/STOP/BACKSTOP authority over the free-space
        # arm legs. Inert (None) unless reactive_arm_human is enabled => byte-identical. The supervisor keeps
        # the ported humangate warm every tick, acts only in its re-plannable phases, and realises SLOW/STOP
        # through the e-stop-free motion.cancel()/cancel_outcome() preempt + a reversible ArmCommandGate
        # pause (an injected sink) -- never the global emergency stop. People are the gate's job; the nvblox
        # map is never the stop (scout sim-perception-architecture-s1 §4). Both clocks are the SIM clock
        # (freshness-sim-clock-not-wall): the safety frame's evidence age is bounded in sim seconds, which
        # render pacing only makes more generous in wall time.
        if self._reactive_arm is not None:
            # PR-B corridor watch reads the remaining swept corridor from the measured joints to the
            # current phase's goal TCP pose (self.poses[phase]); None goal => the human layer still runs,
            # the corridor watch is inert this tick.
            _goal_pose = self.poses.get(self.phase) if getattr(self, 'poses', None) else None
            d = self._reactive_arm.supervise(
                now=now, now_wall=now, phase=self.phase, motion=self.motion,
                leg_running=self.issued, measured=sensors.joint_positions,
                goal_pose=_goal_pose, held=self.held)
            self._reactive_arm_slow = bool(d.slow)
            if d.reissue:
                self.issued = False
                # record the preempt+re-plan (human SLOW or PR-B corridor REPLAN) once per transition
                if d.action != getattr(self, '_react_last_action', None):
                    self._react_last_action = d.action
                    self.evidence.append({'reactive_arm': {'action': d.action, 'reason': d.reason,
                                                           'sim_time': float(now), 'reissue': True}})
                    print('REACTIVE_ARM_PREEMPT phase=%s action=%s now=%.3f %s'
                          % (self.phase, d.action, now, d.reason), flush=True)
            if d.fail is not None:
                return self._fail('reactive arm: ' + d.fail)
            if d.block:
                # Record one evidence entry per action transition (never per tick) + a throttled log.
                if d.action != getattr(self, '_react_last_action', None):
                    self._react_last_action = d.action
                    self.evidence.append({'reactive_arm': {'action': d.action, 'reason': d.reason,
                                                           'sim_time': float(now),
                                                           'paused': self._reactive_arm.paused}})
                self._react_hold_n = getattr(self, '_react_hold_n', 0) + 1
                if self._react_hold_n % 20 == 1:
                    print('REACTIVE_ARM_HOLD phase=%s action=%s now=%.3f %s'
                          % (self.phase, d.action, now, d.reason), flush=True)
                return self._result()
            elif d.action != getattr(self, '_react_last_action', None):
                self._react_last_action = d.action   # e.g. RUN resume / SLOW re-issue
        try:
            if sensors.valid_until_steady is not None:
                require_fresh(sensors.valid_until_steady,self.steady_clock)
            return self._advance(sensors)
        except Exception as exc:
            return self._fail(f'adapter failure: {type(exc).__name__}: {exc}')

    def _advance(self, s):
        now = s.simulation_time
        if self.phase == 'observe':
            # RESTOCK_TRANSFER_MODE=live_pose runs the HONEST place: its preplace/place/retreat are
            # seeded-KDL joint-space moves whose IK can land in the ur_wrist_2-vs-ur_forearm self-collision
            # branch and jam (audit transfer-preplace-wrist-branch-jam; warm2 preplace w1 2.08/w2 -1.02 ->
            # 1.3 rad track fault -> host e-stop). The ONLY thing that keeps those goals in the clear wrist
            # branch is the wrist_guard (RESTOCK_TRANSFER_CARRY=wrist_guard). live_pose fixes the PICK reach
            # but does nothing for the PLACE branch, so live_pose WITHOUT the guard silently runs the
            # jam-prone planner and dies as a generic 'arm command heartbeat lost' e-stop (rank 5 masking).
            # Fail fast with a named reason at the first transfer step instead of falling back to it.
            if self._live_pose_transfer() and WRIST_GUARD not in self.transfer_carry:
                return self._fail('live_pose transfer requires RESTOCK_TRANSFER_CARRY=wrist_guard: '
                                  'the preplace/place KDL IK jams the ur_wrist_2-vs-forearm self-collision '
                                  'branch without it (audit transfer-preplace-wrist-branch-jam)')
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
            # The size-fit (axis_fits) is frame-independent, so gate it now regardless of flow.
            if not axis_fits(target.free_dimensions, self.profile.dimensions):
                return self._fail('unreachable or undersized target')
            # sim-service-carry-drive-to-shelf-v1: when the base repositions BEFORE placing (the fm390
            # collect place-on-cart AND the shelf-staging live_pose TRANSFER service leg), the PLACE
            # target is reached only AFTER the base drives to its service pose (place_approach /
            # shelf_approach) -- it is NOT reachable from the current PICK/park pose. Gating its reach
            # here would _fail('unreachable or undersized target') -> motion.stop() -> emergency stop
            # BEFORE the pick, so the arm never folds to TRANSIT and the runtime carry-drive leg (which
            # fires on the observed TRANSIT, restock/runtime._step transfer carry leg) never runs --
            # transferobs1: the endcap place sits 2.5 m from the side-bay park, > the 1.3 m UR10 reach.
            # The place reach IS validated at PLAN TIME (resolve_transfer_place_target, from the service
            # pose, before any arm command) and again post-drive in _reregister_place, so defer it here;
            # place-side poses are likewise excluded from gate_poses below (the pick-side still gates now).
            if not self._repositions_before_place() and not self._pose_ok(target.pose):
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
                if self._live_pose_transfer():
                    # live_pose: carton + bay are symmetric under a 180 deg yaw -- pick the representatives
                    # that keep the side-grip wall facing the arm (min worst-station reach).
                    self.poses, self.near_face = near_face_poses(self.pose_policy, source, target, p)
                else:
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
            gate_poses = (self.poses if not self._repositions_before_place()
                          else {k: v for k, v in self.poses.items() if k not in _PLACE_SIDE_PHASES})
            if not all(self._pose_ok(v) for v in gate_poses.values()):
                import math as _m
                bad = {k: round(_m.dist(v[:3], (0, 0, 0)), 3) for k, v in gate_poses.items()
                       if not self._pose_ok(v)}
                print('REACH_ENVELOPE_FAIL max_reach=%s bad=%s source=%s target=%s'
                      % (self.profile.max_reach, bad, tuple(round(c, 3) for c in source.pose[:3]),
                         tuple(round(c, 3) for c in target.pose[:3])), flush=True)
                return self._fail('approach outside reach envelope')
            # fm449 (run-332 cold collect): do not leave observe for the reach until the fixed source
            # camera has delivered -- this tick, at park -- a CLEAN synchronized frame + a DECODED label
            # for the calibrated source slot. matches==1 above already proves a decoded label, but it can
            # be served from a HELD/seeded registration (the carton top read while the support is occluded
            # or off a fm443 seed); beginning the reach on that prior is exactly the cold-collect failure.
            # The injected provider reports the clean-this-tick verdict keyed on the camera lane's
            # occluded/clean signal (a non-None, non-suspended, non-occluded SourceTarget). Bounded: no
            # clean capture within collect_source_evidence_wait_ticks fails fast with the named reason.
            # Default off (flag False) => the branch never arms => byte-identical. Placed before any commit
            # side-effect (evidence append / transition) so a wait neither advances nor double-records.
            # collect-gt: a ground-truth collect pick has, by design, no camera frame to be clean -- the camera
            # refusal is exactly what it works around -- so the gate stands aside for it (labelled below).
            gt_pick = (self.collect_ground_truth and getattr(self, 'operation', None) == 'collect'
                       and source.source == gtc.SOURCE)
            if (self.require_collect_source_evidence and getattr(self, 'operation', None) == 'collect'
                    and not gt_pick):
                clean = False
                provider = self._collect_source_evidence
                if provider is not None:
                    try:
                        clean = bool(provider(now))
                    except Exception as exc:   # a swallowed source-capture failure counts as not-clean
                        print('COLLECT_SOURCE_EVIDENCE provider error (%s: %s)'
                              % (type(exc).__name__, exc), flush=True)
                        clean = False
                else:
                    # fm449 host path: the collect controller runs in the host process fed Sensors over ROS,
                    # so the Isaac-side clean verdict rides the packet (Sensors.collect_source_clean, set from
                    # the source provider's SourceTarget.clean) -- the same asdict->JSON->Sensors crossing as
                    # base_repositioned. Absent/False => not clean (byte-identical when the driver never sets it).
                    clean = bool(getattr(s, 'collect_source_clean', False))
                if not clean:
                    self._collect_source_evidence_waits += 1
                    if self._collect_source_evidence_waits > self.collect_source_evidence_wait_ticks:
                        return self._fail('collect_source_evidence_unavailable')
                    if self._collect_source_evidence_waits % 20 == 1:
                        print('COLLECT_SOURCE_EVIDENCE waiting for a clean source frame+label (%d/%d) '
                              'provider=%s' % (self._collect_source_evidence_waits,
                              self.collect_source_evidence_wait_ticks, provider is not None), flush=True)
                    return self._result()   # hold in observe; the reach has not begun
            self.target_pose = (*target.pose[:2], target.pose[2] + p.dimensions[2], *target.pose[3:])
            self.target_surface = target.pose
            self.source_pose = source.pose
            self._source_obs = source   # fm390: re-run the place policy against a fresh target after repositioning
            self.evidence.append(asdict(source))
            if gt_pick:
                self.evidence.append({'source': 'ground_truth_collect', 'collect_mode': gtc.MODE,
                                      'location': self.source,
                                      'pick_pose': tuple(round(v, 4) for v in source.pose)})
                print('GROUND_TRUTH_COLLECT source=%s pick=%s (simulated carton asset pose, not camera; '
                      'collect_mode=ground_truth)' % (self.source, tuple(round(v, 3) for v in source.pose[:3])),
                      flush=True)
            if self.ground_truth_pick and self.operation == 'transfer':
                # fm462: label the (already controlled-asset) transfer pick ground-truth, honestly.
                self.evidence.append({'source': 'ground_truth_pick', 'location': self.source,
                                      'pick_pose': tuple(round(v, 4) for v in source.pose)})
                print('GROUND_TRUTH_PICK transfer source=%s pick=%s (ground-truth controlled-asset cart '
                      'deck, not camera)' % (self.source, tuple(round(v, 3) for v in source.pose[:3])),
                      flush=True)
            if self._live_pose_transfer():
                # Label the workaround honestly in the evidence + console (mission.json records the mode).
                self.evidence.append({'source': 'transfer_mode', 'transfer_mode': LIVE_POSE,
                                      'location': self.source,
                                      'pick_top_pose': tuple(round(v, 4) for v in source.pose),
                                      'place_surface_pose': tuple(round(v, 4) for v in target.pose),
                                      **(self.near_face or {})})
                print('TRANSFER_MODE live_pose observe: pick_top=%s place_surface=%s near_face=%s reach=%s'
                      % (tuple(round(v, 3) for v in source.pose[:3]), tuple(round(v, 3) for v in target.pose[:3]),
                         self.near_face, {k: round(math.dist(v[:3], (0, 0, 0)), 3)
                                          for k, v in self.poses.items()}), flush=True)
            if self.transfer_carry and getattr(self, 'operation', None) == 'transfer':
                # Label the carry workarounds honestly in the evidence + console (mission.json records them).
                label = {'source': 'transfer_carry', 'transfer_carry': sorted(self.transfer_carry)}
                guard = self._place_wrist_guard()
                if guard is not None:
                    label['wrist_guard'] = {'wrist1_cap': guard[0], 'wrist2_min': guard[1],
                                            'phases': list(GUARDED_PHASES)}
                    print('TRANSFER_CARRY wrist_guard active: %s KDL goals must keep wrist_1<=%.2f, wrist_2>=%.2f '
                          '(wrist-flipped re-seed)' % ('/'.join(GUARDED_PHASES), guard[0], guard[1]), flush=True)
                    # sim-robust-transfer-preplace-reach-v1: name the preplace reach guard honestly.
                    if self._live_pose_transfer() and self.transfer_preplace_reach_guard is not None:
                        md, seed = self.transfer_preplace_reach_guard
                        label['preplace_reach_guard'] = {'max_droop': md, 'upright_seed': list(seed)
                                                         if seed is not None else None}
                        print('TRANSFER_CARRY preplace_reach_guard active: preplace KDL goal must also keep the '
                              'shoulder fold holdable (fold_droop<=%.2f); upright-seeded, deep-fold clear '
                              'branch rejected' % md, flush=True)
                if WELD in self.transfer_carry:
                    label['transfer_attach'] = 'weld'
                    print('TRANSFER_CARRY weld: the carton is held by the SIM fixed-joint weld from the grasp '
                          'contact to the release at the place pose (transfer_attach=weld)', flush=True)
                self.evidence.append(label)
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
                # sim-robust-cart-place-goal-v1: droop-safe + self-collision-safe fixed-shape PREPLACE for
                # the NON-cage COLLECT place-on-cart. The reposition place drives the base to the rollwagen
                # mouth, then the pose-IK preplace's nearest-branch seed (from the folded carry config) lands
                # in a branch the arm cannot execute: either the ur_wrist_2-vs-ur_forearm self-collision
                # (collectsrc's 'Invalid c-space position' replan storm) or the DEEP shoulder_lift fold
                # (~ -3.57) the soft drive cannot HOLD under the carton load (collect1 sagged 0.39 rad short
                # -> PATH_TOLERANCE_VIOLATED -> heartbeat lost). This CPU-validated UPRIGHT clear-region shape
                # (shoulder_lift ~ -1.9; FK TCP ~ the sensed preplace standoff) is driven as a fixed
                # collision-planned begin_joint_goal exactly like cage_preplace_shape; the place (pose-IK
                # below, near_start-seeded from here with collect_place_wrist_guard on) then tracks the SENSED
                # deck target, staying on the upright clear branch. Scoped to collect preplace, non-cage.
                # Absent => the pose-IK preplace (byte-identical).
                if (self.phase == 'preplace' and not self.cart_cage_place
                        and getattr(self, 'operation', None) == 'collect'
                        and self.collect_place_preplace_shape is not None):
                    # sim-carton-level-place-on-cart-v1: level the place-on-cart standoff. The
                    # reach-authored collect_place_preplace_shape holds the TOP-grasped carton ~180 deg off
                    # level (the follow-up PR 287 flagged). PR 287's note that wrist-only leveling loses the
                    # deck reach (TCP shift ~0.47 m) still holds -- so this re-plans the FULL arm while
                    # PRESERVING the TCP: a near-free wrist_3 flange roll turns the inverted hold upright
                    # without moving the origin, keeping the shape's reach-validated TCP and self-collision
                    # branch. Off / level_place off => the shape is unchanged (byte-identical); if no clear
                    # level solution exists it proceeds unleveled and the CARTON_TILT trip owns it.
                    # sim-place-preplace-above-support-v1 / sim-place-descent-residual-v1: re-target the
                    # preplace standoff DIRECTLY ABOVE the place load-centre. With preplace_above_support
                    # (the shelf-correct lane) the standoff carton bottom sits at the resolved support top +
                    # the clearance, so the from-above descent onto the SELECTED support is POSITIVE
                    # (coldwin48's mid-air release otherwise); with the legacy preplace_above_place the raise
                    # clears the deck cage mouth posts. Off / no solution => the fixed shape (byte-identical).
                    preplace_shape = self.collect_place_preplace_shape
                    above = self._preplace_above_place_goal(s.joint_positions)
                    # sim-preplace-branch-select-bounded-v1: a reaction-branch search that blew its CPU
                    # time-box REFUSES the preplace before motion with a named reason (never block the
                    # host status heartbeat past the Isaac watchdog -- coldwin59).
                    if self._preplace_branch_refuse is not None:
                        reason = ('preplace reaction-branch selection refused: '
                                  + self._preplace_branch_refuse)
                        print('PREPLACE_REACTION_BRANCH_REFUSED %s' % reason, flush=True)
                        return self._fail(reason)
                    if above is not None:
                        preplace_shape = above
                        mode = getattr(self, '_preplace_above_mode', 'place')
                        clr = getattr(self, '_preplace_above_clearance',
                                      self._carton_level.preplace_standoff_m)
                        print('PREPLACE_ABOVE_%s engaged: collect preplace standoff re-targeted directly '
                              'above the place load-centre +%.3f m (carton bottom at the resolved support '
                              'top + clearance) so the from-above descent onto the selected support is '
                              'positive and the contact release fires at the true support top '
                              '(sim-place-%s)'
                              % (mode.upper(), clr,
                                 'preplace-above-support-v1' if mode == 'support'
                                 else 'descent-residual-v1'), flush=True)
                    # sim-preplace-per-joint-gains-v1 (fallback, deliverable 3): WRIST-FIRST sequencing. When
                    # on, the FIRST preplace leg drives every joint EXCEPT shoulder_pan to the standoff (the
                    # 1.64 rad wrist_2 sweep + leveling moves happen while pan is HELD at its carry value),
                    # then the SECOND leg moves pan to the standoff with the wrists already set -- so the
                    # wrist-sweep reaction and the pan travel are decoupled in time (remove the fight rather
                    # than out-author it). The carton stays level via the SAME leveled approach below; only
                    # the GOAL of this leg changes. Off => the single standoff approach (byte-identical).
                    if self.preplace_wrist_first and not self._preplace_wristed:
                        from twin.restock_manipulation.preplace_sequence import (
                            wrist_first_subgoals, pan_first_subgoals)
                        # sim-preplace-sweep-cage-clearance-v1: pan-FIRST reorder (opt-in). Sweep shoulder_pan
                        # to the standoff while the arm stays tucked at the carry shape (leg 1 holds the wrists
                        # + lift + elbow at their carry value), then extend -- so the arc does not rake the
                        # cage mouth. Default off => wrist-first (byte-identical).
                        if self.preplace_pan_first:
                            preplace_shape, _ = pan_first_subgoals(s.joint_positions, preplace_shape)
                        else:
                            preplace_shape, _ = wrist_first_subgoals(s.joint_positions, preplace_shape)
                        # sim-preplace-leg2-level-v1: under the held chassis attitude the standoff wrists are
                        # world-level only AT the standoff pan; held at the CARRY pan (leg 1) they sit off the
                        # cone, so re-solve the leg-1 goal's wrist cluster for level-IN-WORLD at the carry pan
                        # (lift/elbow stay at the standoff). Then leg 1 ends world-level and leg 2's pan sweep
                        # stays world-level throughout. base_R None (hold off) => the standoff wrists verbatim
                        # (byte-identical to the prior wrist-first leg 1). Skipped for the pan-FIRST reorder:
                        # its leg 1 deliberately HOLDS the wrists at the compact carry shape (already the
                        # transport-level pose) so the arm stays tucked through the pan sweep -- re-solving
                        # the wrist cluster here would extend/change them and defeat the clearance (the carton
                        # is re-leveled in leg 2's extend).
                        if (not self.preplace_pan_first
                                and self._carton_base_R is not None and self._carton_up_in_tool_world is not None):
                            from twin.restock_manipulation import carton_level as _cl1
                            from twin.restock_manipulation.cad_tool import ARM_JOINTS as _AJ1
                            _robot = self._fk_model()
                            if _robot is not None:
                                try:
                                    _tcp = self._place_fk_link(_robot, list(preplace_shape))
                                    _lev = _cl1.level_config(
                                        _robot, list(preplace_shape), self._carton_up_in_tool_world,
                                        joints=self._carton_level.joints, tool_link=_tcp,
                                        limits=self._place_level_limits() or None,
                                        accept_deg=self._carton_level.bound_deg, base_R=self._carton_base_R)
                                    preplace_shape = tuple(float(_lev[j]) for j in _AJ1)
                                except ValueError as _exc:
                                    print('PREPLACE_WRIST_FIRST leg-1 world-level unsolved: %s '
                                          '(using the standoff wrists; CARTON_TILT owns it)' % _exc, flush=True)
                        if self.preplace_pan_first:
                            print('PREPLACE_PAN_FIRST leg 1: sweeping shoulder_pan to the standoff while the '
                                  'arm holds the tucked carry shape (wrists+lift+elbow), so the arc clears '
                                  'the cage mouth (sim-preplace-sweep-cage-clearance-v1)', flush=True)
                        else:
                            print('PREPLACE_WRIST_FIRST leg 1: driving the wrists + leveling cluster to the '
                                  'standoff while shoulder_pan holds at carry (sim-preplace-per-joint-gains-v1)',
                                  flush=True)
                    # sim-preplace-per-joint-gains-v1: ARM the frozen-joint convergence guard against THIS
                    # leg's goal so a non-converging preplace (coldwin53: shoulder_pan froze 0.56 rad short) is
                    # named PHASE_CONVERGENCE in seconds instead of the opaque 120 s timeout. The small wrist
                    # leveling applied below is well under the guard's tol, so arming on the leg's shape is
                    # exact enough to catch a half-radian freeze. Inert when stall_s<=0.
                    if self._preplace_convergence.enabled:
                        self._preplace_convergence.arm(preplace_shape)
                    # sim-preplace-level-gate-v1: when the preplace gate is on AND the standoff was re-targeted
                    # above the support (``above``), the standoff is ALREADY executed-level (solve_level_standoff
                    # min-executed branch); drive the carried APPROACH to it as a DROOP-COMPENSATED, DENSIFIED,
                    # speed/accel-capped leveled path (the coldwin50 fix: the geometric-interior approach drooped
                    # ~1.8 deg executed and the coarse fast sweep deflected run-to-run). Off / no above-standoff
                    # => the single leveled-endpoint approach below (byte-identical).
                    if (getattr(self._carton_level, 'preplace_level_gate', False) and above is not None):
                        return self._issue_gated_preplace_approach(preplace_shape, s.joint_positions)
                    preplace_goal = self._level_place_goal(preplace_shape, s.joint_positions)
                    # sim-carton-level-preplace-v1: drive the carry/tuck -> standoff APPROACH as a LEVELED
                    # WAYPOINT PATH, not a single begin_joint_goal. PR 290 leveled the standoff ENDPOINT and
                    # PR 296 the lift/carry, but the joint-space interpolation INTO the standoff still passed
                    # through a pitched config (collectlevel1: both ends level, FK-reconstructed ~13 deg
                    # mid-path; pitch peak 2.25 deg tripped CARTON_TILT at the pre-place transition). With
                    # level_path on, interpolate to the SAME PR 290 standoff goal (pinned verbatim) with the
                    # interior wrists re-solved level, and CARTON_LEVEL_PLAN_REJECTED before motion if the
                    # predicted path still exceeds the bound; off => the single leveled-endpoint goal
                    # (byte-identical). The standoff endpoint and the level descent (below) are unchanged.
                    return self._issue_level_approach_move(preplace_goal, s.joint_positions,
                                                           slow=True, goal_tol=0.2)
                # fm403: wrist_2-safe fixed-shape LIFT. The task-space lift IK can flip wrist_2 to the
                # negative self-colliding branch (w2 -> -1.25 crosses the ur_wrist_2-vs-ur_forearm stop;
                # runs 300d/305/311 jammed even with wrist_1 capped -- the bench showed w1 cleared but w2
                # stuck 0.38 short). Drive the lift as a fixed collision-planned begin_joint_goal to the
                # clear-region carry shape (w1<=0.65, w2>=0, bench-proven 10/10), exactly as
                # cage_preplace_shape fixed the preplace. Carton-loaded -> slow; the big shoulder_pan swing
                # into the shape settles a little short on joint0 only, so relax THIS move's terminal gate
                # (the subsequent tuck/transit re-anchor). Absent carry_lift_shape => legacy task-space lift.
                if self.phase == 'lift' and self.carry_lift_shape is not None:
                    # sim-carton-level-carry-v1 / sim-carry-level-lift-v1: raise the carton parallel to
                    # the ground. level_goals alone levels only the ENDPOINT -- the joint-space path from
                    # the grasp pose to the leveled carry_lift_shape still passes through a tilted config
                    # (coldwin25: CPU-reconstructed ~11 deg mid-path). With level_path on, drive the lift
                    # as a LEVELED WAYPOINT PATH (carton level throughout) and CARTON_LEVEL_PLAN_REJECTED
                    # any plan whose predicted executed tilt exceeds the bound before motion streams; off
                    # => the single leveled-endpoint goal (byte-identical). Carton-loaded -> slow; relax
                    # THIS move's terminal gate (the subsequent tuck/transit re-anchor).
                    return self._issue_level_carried_move(self.carry_lift_shape, s.joint_positions,
                                                          slow=True, goal_tol=0.2)
                # sim-robust-collect-stow-branch-v1 (collect5, 2026-10-23): clear-branch RETREAT for the
                # NON-cage COLLECT deck place. The place + retreat run collision-OFF straight-line
                # (_collect_deck_place) because the set-down fold is a conservative attached-carton
                # self-collision; but with collision-avoidance OFF the empty-retreat pose IK branches into a
                # DROOPED config where the forearm sits ~18 mm inside the gripper-cube (cube_without_surface_
                # plates) sphere margin (collect5 retreat endpoint [0.214,-1.755,2.075,-0.320,1.975,0], FK TCP
                # ~10 cm below the standoff). The NEXT move is the collision-AWARE staged retreat_tuck, and
                # cuMotion rejects that self-colliding START ('Invalid c-space position: self collision
                # detected' -> INVALID_INITIAL_CSPACE_POSITION -> MANIP_HOST_FAILED op=collect -> arm
                # heartbeat lost). Fix: return the empty gripper to the FIXED clear UPRIGHT standoff
                # (collect_place_preplace_shape -- the same CPU-validated clear-region shape the preplace
                # uses, FK TCP = the retreat standoff pose) via a collision-OFF straight-line JOINT smoothstep
                # (begin_joint_goal(straight_line=True)): the smoothstep monotonically un-folds the forearm
                # out of the cube margin to a self-clear config, so retreat_tuck/stow then plan from a clean
                # start. No clear escape shape configured => fail by name (never drive the collision-aware
                # tuck into a self-colliding start). Scoped to _collect_deck_place retreat, so cart-cage /
                # live_pose / scripted-still stay byte-identical.
                if self.phase == 'retreat' and self._collect_deck_place():
                    if self.collect_place_preplace_shape is None:
                        return self._fail('collect deck retreat has no clear-branch escape shape: '
                                          'collect_place_straight_line needs collect_place_preplace_shape '
                                          'to back the empty gripper off the self-colliding place fold')
                    # sim-collect-retreat-abort-v2: lift-off-THEN-fold. FIRST raise the empty gripper
                    # straight UP (+dz along world +Z, orientation preserved) to separate the still-ENABLED
                    # wrist links from the re-enabled carton / shelf / cage BEFORE the horizontal fold --
                    # the straight-to-preplace fold's first increment is almost pure horizontal retract and
                    # dragged those links into a contact that pinned the drive (physcon1/rtfcollect2). THEN
                    # (once lifted) fold to the clear preplace shape (the existing move). The lift-off is a
                    # collision-OFF straight_line joint smoothstep like the fold, so the two legs share the
                    # mechanism. If the lift-off IK cannot reach straight up from here, fall through to the
                    # direct fold (no regression vs the prior behaviour). Off / non-collect-deck => unchanged.
                    if self.collect_retreat_liftoff and not self._retreat_lifted:
                        liftoff = self._collect_retreat_liftoff_goal(s.joint_positions)
                        if liftoff is not None:
                            self.motion.begin_joint_goal(liftoff, s.joint_positions, straight_line=True)
                            self.issued = True
                            return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))
                        # IK refused: skip the lift-off, go straight to the fold (prior behaviour).
                        self._retreat_lifted = True
                    self.motion.begin_joint_goal(self.collect_place_preplace_shape, s.joint_positions,
                                                 straight_line=True)
                    self.issued = True
                    return self._result(({'kind': 'move', 'pose': None, 'phase': self.phase},))
                # sim-carton-level-place-on-cart-v1: LEVEL place descent for the collect deck place. Lower
                # the carton onto the SENSED deck slot while keeping it parallel to the ground -- a single
                # straight-line (PR 268) joint descent from the level standoff to a FULL-arm level IK
                # solution at the sensed place TCP (reach + level), seeded off the leveled standoff so the
                # descent stays on the reach-validated branch. The preplace already leveled the standoff;
                # this holds level through set-down (the whole move stays < 0.1 deg off level, CPU-proven).
                # If no clear level solution reaches the sensed deck, fall through to the existing (unleveled)
                # straight-line place and let the CARTON_TILT trip own it -- never set down in the air.
                if (self.phase == 'place' and self._collect_deck_place()
                        and self._carton_level is not None and self._carton_level.level_goals
                        and self._carton_level.level_place and self._carton_up_in_tool is not None
                        and self.collect_place_preplace_shape is not None and 'place' in self.poses):
                    robot = self._fk_model()
                    if robot is not None:
                        seed = self._level_place_goal(self.collect_place_preplace_shape, s.joint_positions)
                        target_pos = tuple(float(c) for c in self.poses['place'][:3])
                        try:
                            # sim-carton-level-place-descent-v1: PER-POINT wrist-compensated level set-down.
                            # coldwin34 ran carton_level:true => gravity_compensation None, so the place
                            # descent solved a GEOMETRIC-level set-down with NO droop pre-tilt and the welded
                            # far-extended arm drooped +2.06 deg at the set-down dwell (CARTON_TILT phase=place,
                            # sustained 0.150 s). Even with a gravity_compensation configured, the prior fix
                            # routed the wrist droop-cancel only to the single ENDPOINT of a 2-point straight
                            # line, so it did not FOLLOW THE EXTENSION through the insert. Here the descent is
                            # a per-point leveled waypoint path: the TCP target is interpolated standoff->deck
                            # and EVERY waypoint is solved reach+level+wrist-droop-cancel against its own loaded
                            # droop, driven as ONE collision-off begin_joint_path (the deck set-down fold is a
                            # conservative attached-carton self-collision, collect4). gc defaults ON for this
                            # segment (no host change), and is time-parameterised slower
                            # (place_descent_speed/accel) so the loaded execution stays level. The predicted
                            # EXECUTED (droop-aware) peak is stashed for the per-segment CARTON_SEGMENT_TILT
                            # report (place AND the post-set-down release, which holds this pose) and refuses
                            # by name before any motion if it would exceed the bound. Bound + grace UNCHANGED.
                            from twin.restock_manipulation import carton_level as cl
                            gc = self._place_gravity_compensation()
                            limits = self._place_level_limits() or None
                            tcp_link = self._place_fk_link(robot, list(seed))
                            plan = cl.plan_compensated_place_descent(
                                robot, list(s.joint_positions), target_pos, list(seed),
                                self._carton_up_in_tool, gc, bound_deg=self._carton_level.bound_deg,
                                tool_link=tcp_link, limits=limits,
                                waypoints=self._carton_level.path_waypoints,
                                approach_from_above=self._carton_level.place_from_above,
                                above_clearance_m=self._carton_level.place_above_clearance_m,
                                depth_bias_m=self._carton_level.place_depth_bias_m)
                            self._carton_seg_planned['place'] = (plan.max_roll, plan.max_pitch)
                            self._carton_seg_planned['release'] = (plan.max_roll, plan.max_pitch)
                            reject = plan.reject(self._carton_level.bound_deg)
                            if reject is not None:
                                print(reject, flush=True)
                                return self._fail(reject)
                            # sim-place-descent-residual-v2 (part 3): opt-in, bounded, sim-only CLOSED-LOOP
                            # carton-level backstop on the planned descent. Off => waypoints unchanged
                            # (byte-identical). It only bends the wrist within the authority to hold the
                            # EXECUTED carton level under a residual droop-model error; the least-extended
                            # approach (part 2) still owns the geometric pad-vs-post clearance.
                            waypoints = list(plan.waypoints)
                            waypoints, cl_peak = self._closed_loop_correct_descent(
                                robot, waypoints, gc, tcp_link, measured_start=s.joint_positions)
                            # sim-place-descent-residual-v3: latch the contact-triggered weld-release
                            # threshold for this set-down. The welded carton descends from the standoff by the
                            # planned TCP drop; when its live measured centre z (placed_pose) reaches within
                            # place_contact_gap_m of the set-down height it has met the deck -> release the weld
                            # (step() gate below) instead of pressing the leveled carton into the cart deck
                            # (the reach-independent 2.0 deg residual). Inert (None) when the gate is off or no
                            # measured carton pose is available -> the legacy place->level-gate->release path.
                            # sim-place-release-on-contact-v1: the z fallback is armed for EITHER the legacy
                            # fixed-z gate (place_contact_release) OR the contact-gated gate
                            # (place_release_on_contact, which keeps carton_z as its fallback trigger).
                            self._place_contact_release_z = None
                            self._place_reapproach_count = 0
                            _want_release_z = (self._carton_level.place_contact_release
                                               or getattr(self._carton_level, 'place_release_on_contact', False))
                            if (_want_release_z and s.placed_pose is not None and len(s.placed_pose) >= 3
                                    and math.isfinite(float(s.placed_pose[2]))):
                                try:
                                    from twin.restock_manipulation import carton_level as _cl3
                                    drop = _cl3.place_descent_drop_m(robot, waypoints, tool_link=tcp_link)
                                    # sim-release-threshold-carton-frame-v1: placed_pose[2] is the carton
                                    # CENTRE (types.py), so the geometric release threshold is expressed in
                                    # the carton-centre frame -- add the carton half-height so the trigger
                                    # fires when the carton BOTTOM reaches support_top + gap, not half-height
                                    # below (coldwin87: the legacy bottom-frame 0.5635 sat 0.078 m below the
                                    # measured rest 0.6422, so with the startup-stall suppressed the arm
                                    # pressed the welded carton into the shelf to HOST_FAULT_ESTOP).
                                    carton_height = float(self.profile.dimensions[2])
                                    self._place_contact_release_z = _cl3.place_contact_release_z_centre(
                                        float(s.placed_pose[2]), drop, self._carton_level.place_contact_gap_m,
                                        carton_height)
                                    print('PLACE_CONTACT_RELEASE_ARM phase=place start_carton_z=%.4f '
                                          'descent_drop=%.4f gap=%.4f carton_h=%.4f -> release at '
                                          'carton_z(centre)<=%.4f (fire at true contact: carton BOTTOM at '
                                          'support_top + gap, carton CENTRE + half-height)'
                                          % (float(s.placed_pose[2]), drop,
                                             self._carton_level.place_contact_gap_m, carton_height,
                                             self._place_contact_release_z), flush=True)
                                except (ValueError, KeyError) as exc:
                                    print('PLACE_CONTACT_RELEASE_ARM disabled: %s (legacy level-gate release '
                                          'owns the set-down)' % exc, flush=True)
                                    self._place_contact_release_z = None
                            # sim-place-seat-geometry-v1: when the HONEST seat gate is on, OVERRIDE the PR 374
                            # relative threshold with the ABSOLUTE corrected seat (support TOP + carton half, in
                            # the carton-centre frame == the measured placed_pose frame). The support top is the
                            # live cart-tracked collect place target z (self.target_surface[2], ur_base_link).
                            # ASSERT the seat sits BELOW the descent START carton_z: a threshold at/above the
                            # start can never be reached by the descent, so a release there fires at the standoff
                            # (exactly PR 374's bug) -- refuse the set-down BY NAME at arm time instead. Default
                            # off => this block is skipped => the PR 374 relative threshold stands (byte-identical).
                            self._place_seat_center_z = None
                            if (getattr(self._carton_level, 'place_seat_gate', False)
                                    and s.placed_pose is not None and len(s.placed_pose) >= 3
                                    and math.isfinite(float(s.placed_pose[2]))
                                    and self.target_surface is not None and len(self.target_surface) >= 3
                                    and math.isfinite(float(self.target_surface[2]))):
                                carton_half = float(self.profile.dimensions[2]) / 2.0
                                support_top_base = float(self.target_surface[2])
                                seat_center_z = support_top_base + carton_half
                                start_carton_z = float(s.placed_pose[2])
                                if seat_center_z >= start_carton_z:
                                    return self._fail(
                                        'PLACE_SEAT_THRESHOLD_ABOVE_START: the modelled seat carton-centre z '
                                        '%.4f (support top %.4f + carton half %.4f) sits at/above the descent '
                                        'START carton_z %.4f -- the descent can never reach the seat and a '
                                        'release there would fire at the standoff (PR 374). Refusing the '
                                        'set-down by name (check the resolved support top vs the standoff).'
                                        % (seat_center_z, support_top_base, carton_half, start_carton_z))
                                self._place_seat_center_z = seat_center_z
                                self._place_contact_release_z = seat_center_z   # absolute seat; bypasses PR 374
                                print('PLACE_SEAT_GATE_ARM phase=place support_top=%.4f carton_half=%.4f -> '
                                      'seat_center_z=%.4f (start_carton_z=%.4f seat_tol=%.3f): release only '
                                      'within tol of the seat; a measured contact ABOVE is PLACE_OBSTRUCTED, a '
                                      'descent past the seat with no contact is a named fault'
                                      % (support_top_base, carton_half, seat_center_z, start_carton_z,
                                         float(getattr(self._carton_level, 'place_seat_tol_m', 0.02))),
                                      flush=True)
                            # Arm the contact-gated gate with the freshly computed z fallback (which may be
                            # None -- the descent-independent triggers still fire) and the honest seat (None
                            # unless place_seat_gate armed it above => byte-identical).
                            if self._place_release_gate is not None:
                                self._place_release_gate.reset(release_z=self._place_contact_release_z,
                                                               seat_center_z=self._place_seat_center_z)
                                self._place_release_log_sim = None
                            if cl_peak is not None:
                                mr, mp, _ = cl.predict_path_tilt_executed(
                                    robot, waypoints, self._carton_up_in_tool, tool_link=tcp_link,
                                    gravity_comp=gc)
                                self._carton_seg_planned['place'] = (mr, mp)
                                self._carton_seg_planned['release'] = (mr, mp)
                                print('CARTON_LEVEL_CLOSED_LOOP phase=place engaged (sim-only backstop): '
                                      'wrist correction peak=%.2f deg (authority %.2f), corrected predicted '
                                      'executed max roll=%.2f pitch=%.2f deg (<= bound %.2f)'
                                      % (cl_peak, self._carton_level.closed_loop_authority_deg, mr, mp,
                                         self._carton_level.bound_deg), flush=True)
                            print('CARTON_LEVEL_PLACE_DESCENT phase=place n=%d predicted_max roll=%.2f '
                                  'pitch=%.2f deg (<= bound %.2f) -- per-point wrist-compensated level '
                                  'set-down at speed=%.3f accel=%.3f rad/s (extension-following droop-cancel '
                                  'so the loaded-carton execution stays level throughout the insert); '
                                  'approach=%s (above_clearance=%.3f m, depth_bias=%.3f m)'
                                  % (plan.waypoints_n, plan.max_roll, plan.max_pitch,
                                     self._carton_level.bound_deg, self._carton_level.place_descent_speed,
                                     self._carton_level.place_descent_accel,
                                     'from_above' if self._carton_level.place_from_above else 'straight_line',
                                     self._carton_level.place_above_clearance_m,
                                     self._carton_level.place_depth_bias_m), flush=True)
                            self.motion.begin_joint_path(list(waypoints), list(s.joint_positions),
                                                         slow=True, goal_tol=0.12,
                                                         speed=self._carton_level.place_descent_speed,
                                                         accel=self._carton_level.place_descent_accel)
                            self.issued = True
                            return self._result(({'kind': 'move', 'pose': self.poses['place']},))
                        except ValueError as exc:
                            print('CARTON_LEVEL_UNSOLVED phase=place: %s (unleveled straight-line place; '
                                  'CARTON_TILT trip owns it)' % exc, flush=True)
                straight_contact = self.phase == 'contact' and self.contact_straight_line
                # Cart-cage open-face place: the INSERT (place) and empty-handed RETREAT are straight-line
                # moves THROUGH the single open +x face (begin(straight_line=True): collision-off seeded
                # 2-point). The cage walls are published only so cuMotion routes the collision-aware
                # PREPLACE around them to the mouth standoff; the insert then deliberately drives into the
                # open mouth and back out, never crossing a wall (CPU-proven by the fold->insert sweep).
                #
                # live_pose transfer (armcam2, 2026-10-22): the place lowers the carton ONTO the bay
                # surface, so the place config necessarily brings the gripper tines to within ~13 mm of
                # the TransferBay top (CPU-measured: surface_plate_6 at +0.0126 m, vs the preplace standoff
                # at +0.143 m -- data/.../livepose-pick-collision), inside cuMotion's ESDF collision margin.
                # A collision-AWARE joint-space plan to that config is therefore rejected
                # ('Invalid c-space position: world collision detected' -> trajopt UNKNOWN_STATUS ->
                # MANIP_HOST_FAILED op=transfer), exactly as the preplace plans cleanly. Route the final
                # place descent and the empty-handed retreat as the SAME straight-line collision-off seeded
                # insert the cart-cage place uses (the preplace standoff stays collision-aware): touching the
                # surface is the intent, like the contact descent at the pick. Scoped to live_pose so the
                # scripted/still path and every non-live_pose run are byte-identical.
                #
                # collect deck place (collect4, 2026-10-23): the NON-cage COLLECT place lowers the HELD
                # carton onto the cart deck. The carton is a cuMotion attached_object (ros_host attaches a
                # 34-sphere cuboid at gripper_tcp) NOT self-collision-ignored against the forearm/wrist/
                # upper-arm, so the set-down fold is rejected as an attached-object SELF-collision ('Invalid
                # c-space position: self collision detected' -> MANIP_HOST_FAILED op=collect) EVEN THOUGH the
                # bare arm is self-clear and PR #209's wrist-guard accepted the config. Route it the same
                # way (place + empty retreat straight-line; preplace collision-aware) -- the deck is not a
                # cuMotion world object here, so this is purely the conservative attached-carton self-check
                # of the intended set-down. Gated on _collect_deck_place() (collect_place_straight_line,
                # non-cage collect), so transfer / live_pose / scripted-still stay byte-identical.
                straight = (straight_contact
                            or (self.cart_cage_place and self.phase in ('place', 'retreat'))
                            or (self._live_pose_transfer() and self.phase in ('place', 'retreat'))
                            or (self._collect_deck_place() and self.phase in ('place', 'retreat')))
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
                # sim-robust-transfer-settle-oscillation-v1 (coldwin12): guard the live_pose transfer PICK
                # (approach + contact) to the clear wrist region too. The approach is a cuMotion task-space
                # pose goal -- carry the clear-region joint constraints (ur_wrist_1 <= cap, ur_wrist_2 >= min)
                # so cuMotion IKs the pick on the clear wrist_2-positive family, not the ur_wrist_2-vs-forearm
                # self-colliding one that limit-cycled. (The contact's straight-line seeded-KDL IK is guarded
                # via the wrist_guard extra below.) None/off => byte-identical.
                pick_guard = self._pick_wrist_guard()
                if pick_guard is not None:
                    carry_cap, carry_w2min = pick_guard
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
                # RESTOCK_TRANSFER_CARRY=wrist_guard: the transfer place moves' KDL goal must land in the clear
                # wrist branch (warm2: the preplace IK chose w1 2.08 / w2 -1.02 and jammed at the stop). Passed
                # ONLY when active, so an unset flag keeps this begin() call byte-identical.
                guard = self._place_wrist_guard()
                extra = ({'wrist_guard': guard} if guard is not None and joint_space and not straight
                         and self.phase in GUARDED_PHASES else {})
                # sim-robust-transfer-preplace-reach-v1: alongside the wrist-guard, reject a clear-but-deep
                # shoulder fold and seed the holdable upright branch for the live_pose transfer PREPLACE.
                # Only added when the wrist-guard itself is active for this move, so an unset key keeps the
                # begin() call byte-identical.
                if 'wrist_guard' in extra:
                    reach = self._preplace_reach_guard()
                    if reach is not None:
                        extra['reach_guard'] = reach
                # sim-robust-transfer-settle-oscillation-v1: the live_pose transfer PICK CONTACT descent is a
                # straight-line seeded-KDL IK (contact_straight_line); guard that IK to the clear wrist branch
                # so a contact config outside the region is re-seeded from its wrist-flipped twin (or fails by
                # the named carry-wrist-region reason) instead of silently descending into the self-collision
                # stop, and so a MOVE_RETRY of the contact phase re-plans onto the clear branch from the live
                # pose. With the approach guarded (above) the contact start is clear and this accepts first try
                # (byte-identical). Off/non-pick => pick_guard is None => this never fires.
                if pick_guard is not None and straight_contact:
                    extra['wrist_guard'] = pick_guard
                self.motion.begin(target, s.joint_positions, straight_line=straight,
                                  joint_space=joint_space, wrist1_cap=carry_cap, wrist2_min=carry_w2min,
                                  slow=(place_move or self._reactive_arm_slow),
                                  goal_tol=(0.12 if place_move else None),
                                  near_start=place_move, **extra)
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
            # sim-pick-contact-seal-fallback-v1: once suction is enabled, evaluate guard 3b ONCE per contact
            # tick (before polling the ongoing press, so the dwell counts ticks, not the sparse step
            # boundaries). seal_accept behaves like the plugin latch below -- it drives the contact
            # termination and the gated SIM_ATTACH_FALLBACK forms the weld Isaac-side. Inert (False) when the
            # key is off or no seal evidence is present => byte-identical.
            seal_accept = (self.phase == 'contact' and self._contact_suction and not s.contact
                           and self._seal_diag_contact(s))
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
                # sim-robust-collect-stow-stall-v1: a terminal move_group planner crash/hang fails fast with
                # a named cause BEFORE the sim-deadline retry (which would re-send to the dead planner).
                crash = self._planner_crash_failfast_or_none()
                if crash is not None:
                    return crash
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
                # A rejected/aborted ExecuteTrajectory (trajectory_host goal rejection or execution
                # fault) is NOT a planning collision: surface the port's distinct reason when it has one
                # (sim-robust-pick-plan-rejection-v1), else the generic collision/plan-rejection string.
                _take = getattr(self.motion, 'take_exec_abort_reason', None)
                _distinct = _take() if callable(_take) else None
                return self._fail(_distinct or 'collision planning or execution failed')
            if outcome not in ('running', 'succeeded'):
                return self._fail('invalid motion outcome')
            if outcome == 'succeeded':
                # sim-collect-retreat-abort-v2: the collect deck-place retreat LIFT-OFF just completed (the
                # empty gripper is now clear ABOVE the carton/shelf/cage). Do NOT transition out of retreat
                # yet -- re-issue the SAME retreat phase, which now (self._retreat_lifted) folds to the clear
                # preplace shape in free space. Scoped to the collect-deck retreat with the lift-off on.
                if (self.phase == 'retreat' and self._collect_deck_place()
                        and self.collect_retreat_liftoff and not self._retreat_lifted):
                    self._retreat_lifted = True
                    self.issued = False
                    print('COLLECT_RETREAT_LIFTOFF done (gripper raised clear of the carton/shelf/cage) '
                          '-> fold to clear preplace shape', flush=True)
                    return self._result()
                # sim-preplace-per-joint-gains-v1: the WRIST-FIRST sub-move just completed (wrists at the
                # standoff, hold joints still tucked). Do NOT advance to place yet -- re-issue the SAME
                # preplace phase, which now (self._preplace_wristed) drives every joint to the standoff with
                # the wrists already set. Scoped to the collect deck place with wrist-first on.
                if (self.phase == 'preplace' and self.preplace_wrist_first and not self._preplace_wristed):
                    self._preplace_wristed = True
                    self.issued = False
                    print('PREPLACE_WRIST_FIRST done (wrists at standoff) -> move hold joints to standoff',
                          flush=True)
                    return self._result()
                # Contact-feedback descent (firstmate run-202): the step settled. If the seal has NOT
                # latched and the capped travel is not exhausted, step the contact pose a bit further
                # toward the carton and re-issue -- this drives to actual contact regardless of the
                # run-to-run residual variance a fixed over-press cannot absorb, and the per-step settle
                # gives the plugin a stationary pad to latch (suction stays on). Hitting the cap without
                # contact is a mission fault (the latch never formed); SEAL_DIAG has logged the gap.
                if (self.phase == 'contact' and self.contact_straight_line
                        and self.contact_overpress_step_m > 0 and not (s.contact or seal_accept)):
                    # Freeze-settle (firstmate run-203, stations verdict): the native plugin seals a
                    # STATIONARY pad every time, but the trajectory settle tolerance lets the link finish
                    # with non-zero velocity -- which does NOT seal. So when a step reaches its pose
                    # without a latch, HOLD (do not re-issue) for CONTACT_FREEZE_TICKS so the drive settles
                    # the link to ~0 velocity at the measured joints, giving the plugin (or the gated
                    # SIM_ATTACH_FALLBACK) its stationary settle window to latch. Only if it still has not
                    # latched after the settle do we step further toward the carton.
                    if self._contact_freeze < CONTACT_FREEZE_TICKS:
                        self._contact_freeze += 1
                        return self._result(self._contact_reissue_controls(s))
                    self._contact_freeze = 0
                    # sim-pick-contact-seal-fallback-v1 guard 3a: the settle has elapsed and we are about to
                    # press deeper. If the plugin FRESHLY reported not-enabled (the cw63 lost-command case),
                    # do NOT press on with no chance of a latch: fail fast by name, or (reissue_close_once)
                    # re-send the close ONCE and hold for a second settle; a second miss fails. Inert when
                    # the key is off => the historical over-press step (byte-identical).
                    #
                    # sim-gripper-status-relay-v1 (coldwin65): the guard must read an AUTHORITATIVE Isaac
                    # gripper state, not a starved host snapshot. When no fresh status reached the host
                    # (gripper_status_time absent/too old -- the whole coldwin65 cause), REFUSE TO JUDGE: log
                    # CONTACT_STATUS_STALE with the age and fall through to the bounded over-press -- never
                    # abort CONTACT_SUCTION_NOT_ENABLED on staleness alone (which falsely blamed the enable
                    # command for a feedback-relay gap). A genuine FRESH not-enabled still fails fast.
                    if self.contact_require_suction_enabled:
                        present, age = self._gripper_status_fresh(s)
                        if not present:
                            if not self._contact_status_stale_logged:
                                self._contact_status_stale_logged = True
                                print('CONTACT_STATUS_STALE age=%s bound=%.2fs step=%.1fmm: no authoritative '
                                      'Isaac gripper status reached the host (gripper_status_time absent) -- '
                                      'refusing to judge suction-enabled on staleness alone; continuing the '
                                      'bounded over-press (enable the feedback relay: '
                                      'RESTOCK_CONTACT_STATUS_FEEDBACK=1)'
                                      % (('%.3f' % age) if age is not None else 'never',
                                         self.contact_status_max_age_s, self._contact_step * 1000),
                                      flush=True)
                        elif not s.gripper_enabled:
                            self._contact_suction_miss += 1
                            diag = ('gap_mm=%s pad_deg=%s hit_target=%s enabled=%s age=%.3fs step=%.1fmm'
                                    % (s.seal_gap_mm, s.seal_pad_angle_deg, s.seal_hit_target,
                                       s.gripper_enabled, age, self._contact_step * 1000))
                            if self.contact_reissue_close_once and self._contact_suction_miss == 1:
                                print('CONTACT_SUCTION_REISSUE miss=1 re-sending the gripper close once (%s)'
                                      % diag, flush=True)
                                return self._result(({'kind': 'gripper', 'enabled': True},))
                            return self._fail('CONTACT_SUCTION_NOT_ENABLED at over-press step %d (%s): the '
                                              'gripper reported not enabled/closing with a FRESH Isaac status '
                                              '-- the close did not take (suction off / plugin never latched); '
                                              'refusing to press further into the carton'
                                              % (self._contact_suction_miss, diag))
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
                # sim-carton-level-place-on-cart-v1: release only after a LEVEL set-down. Before advancing
                # place -> release, confirm the carried carton is parallel to the ground at the completed
                # place, so a tilted set-down fails by name instead of opening the gripper on an off-level
                # carton. Inert when carton_level / trip is off, or the up-axis was never captured.
                if (self.phase == 'place' and self._carton_level is not None
                        and self._carton_level.trip and self._carton_up_in_tool is not None):
                    tilt_msg = self._carton_tilt_trip(s.joint_positions)
                    if tilt_msg is not None:
                        return self._fail('place not level before release: ' + tilt_msg)
                    print('PLACE_LEVEL_CONFIRMED phase=place (carton parallel to the ground before release)',
                          flush=True)
                self._transition(next_phase[self.phase])
                return self._result()
            # Contact-terminated grasp descent (firstmate key contact-move-termination): while the
            # contact move runs, raise the contact signal from MEASURED contact so trajectory_host
            # succeeds the goal holding the touch pose -- the gripper is physically against the carton
            # and cannot reach the commanded free-space pose (run-132 GOAL_MISS, stalled ~3.8deg short).
            # Every free-space move (approach/lift/preplace/place/retreat) keeps its exact pose-settle:
            # only 'contact' emits this, and only once measured contact is present.
            # sim-pick-contact-seal-fallback-v1: seal_accept (guard 3b) drives the SAME contact-termination
            # as the plugin latch -- trajectory_host succeeds the goal holding the touch pose; the gated
            # SIM_ATTACH_FALLBACK forms the weld Isaac-side (both require suction enabled). Off => seal_accept
            # is always False => byte-identical.
            if self.phase == 'contact' and (s.contact or seal_accept):
                return self._result(({'kind': 'contact'},))
            # sim-pick-contact-seal-fallback-v1 (item 2): re-publish the gripper-enable control on this
            # running contact tick (opt-in) so the one-shot enable survives the last-value status relay.
            return self._result(self._contact_reissue_controls(s))
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
                # sim-carton-level-carry-v1 / sim-carry-level-lift-v1: tuck/pan carry the welded carton ->
                # keep it parallel to the ground through the swing (level_path => leveled WAYPOINT path +
                # CARTON_LEVEL_PLAN_REJECTED gate; else the single leveled-endpoint goal; off =>
                # byte-identical). Plan the fold gently so the carton's inertia does not displace the
                # wrist off the body-clear plan (run-246). retreat_tuck is empty-handed -> never leveled.
                if self.phase in ('tuck', 'pan'):
                    return self._issue_level_carried_move(goal, s.joint_positions,
                                                          slow=True, goal_tol=None)
                self.motion.begin_joint_goal(goal, s.joint_positions, slow=self._reactive_arm_slow)
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
                # sim-robust-collect-stow-stall-v1: a terminal move_group planner crash/hang fails fast with
                # a named cause before the staged retry (which would re-send to the dead planner).
                crash = self._planner_crash_failfast_or_none()
                if crash is not None:
                    return crash
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
                if self.phase == 'tuck' and self._repositions_before_place():
                    nxt = 'transit'   # fm390 / service-leg: hold the carry fold; the base drives next
                elif self.phase == 'tuck' and self._live_pose_transfer():
                    nxt = 'preplace'  # live_pose (no service carry): no TRANSIT hold, no collect-cart pan
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
            # sim-release-threshold-carton-frame-v1: once the weld detach is confirmed, the opt-in
            # free-and-resting + clearance-lift sub-state owns the transition to retreat (hold -> prove the
            # carton is free+resting -> clear the gripper straight up -> retreat, or RELEASE_NOT_FREE). Off =>
            # _released_detached is never set => this is never reached => byte-identical.
            if self._release_free is not None and self._released_detached:
                return self._release_free_step(now, s)
            # sim-place-release-serialization-v1: a CONTACT-TRIGGERED release arrived MID place-execute (the
            # place ExecuteTrajectory is still in flight). Serialize the release as a proper phase transition:
            # (1) cancel the in-flight place move -- the cancel is the stop, so the arm never keeps pressing;
            # (2) WAIT (bounded) for the cancel's terminal, settled result; (3) only THEN detach (the planning-
            # scene mutation), from the not-in-flight state. coldwin43 skipped (1)+(2) and detached straight
            # into state=execute_result -> 'planning-scene mutation issued while a plan/execute is in flight'.
            # Inert (never pending) on the normal place->release path, where the place move already polled to
            # 'succeeded' (terminal) before the transition => byte-identical.
            if self._release_cancel_pending:
                cancel = getattr(self.motion, 'cancel', None)
                outcome_fn = getattr(self.motion, 'cancel_outcome', None)
                if not callable(cancel) or not callable(outcome_fn):
                    # A motion port without the cancel protocol cannot serialize a mid-flight contact release;
                    # fail BY NAME rather than risk the detach-in-flight invariant (never silently detach).
                    return self._fail('contact-release needs a cancellable motion port '
                                      '(cancel/cancel_outcome) to serialize the release detach')
                # sim-place-release-cancel-handshake-v1: opt-in. The place descent is a DRIVEN trajectory on
                # the sim FollowJointTrajectory executor, so the default ExecuteTrajectory ``cancel()`` is
                # BOTH dropped in the goal-accept race (``self.handle is None`` at the stall -- cw74/cw75) AND
                # unsafe (the sim trajectory_host treats a goal cancel as a FAULT that latches the global
                # stop). When on, STOP the descent on its OWN channel -- emit the contact-hold control so
                # trajectory_host SUCCEEDS holding the current pose (terminal, no fault, no stop) and mark the
                # port cancel-requested (``settle_in_place``, no ``cancel_goal_async``) so ``cancel_outcome``
                # reads the resolved execute future -- and, if the terminal still never arrives but the arm is
                # PROVABLY stopped, log RELEASE_CANCEL_STOPPED_FALLBACK and proceed. Off => the legacy
                # ``cancel()`` path below, byte-identical.
                _handshake = (self._carton_level is not None
                              and getattr(self._carton_level, 'release_cancel_handshake', False))
                if not self._release_cancel_issued:
                    self._release_cancel_issued = True
                    if _handshake:
                        settle = getattr(self.motion, 'settle_in_place', None)
                        if callable(settle):
                            settle()      # arm cancel_outcome to read the terminal; NO cancel_goal_async
                        self._release_stopped_ticks = 0
                        self._release_prev_q = None
                        self._release_prev_q_sim = None
                        print('PLACE_CONTACT_RELEASE_CANCEL phase=release now=%.3f: hold-and-settle the '
                              'in-flight place descent on the arm_contact channel (an ExecuteTrajectory '
                              'cancel is dropped in the goal-accept race and the sim trajectory executor '
                              'treats it as a FAULT+global-stop) before the release detach; waiting for the '
                              'terminal result' % now, flush=True)
                        return self._result(({'kind': 'contact'},))
                    cancel()   # request the cancel; the arm holds its last commanded pose. NO emergency stop.
                    print('PLACE_CONTACT_RELEASE_CANCEL phase=release now=%.3f: cancelling the in-flight place '
                          'execute (the cancel is the stop -- the arm stops pressing the welded carton into '
                          'the deck) before the release detach; waiting for the terminal result' % now,
                          flush=True)
                    return self._result()
                outcome = outcome_fn()
                if outcome is None:
                    # Still settling: the place execute has not reached a terminal result yet.
                    if _handshake:
                        # Stopped-arm fallback: if the arm is PROVABLY stopped (measured max joint speed below
                        # threshold for N consecutive ticks) proceed to detach from the stopped pose, forcing
                        # the port terminal so the detach's _PLAN_IN_FLIGHT_STATES guard passes. NEVER detach
                        # into a moving arm (the speed gate is the proof).
                        stopped_speed = self._release_arm_stopped(now, s)
                        if stopped_speed is not None:
                            force_terminal = getattr(self.motion, 'force_terminal_settled', None)
                            if callable(force_terminal):
                                force_terminal()
                            print('RELEASE_CANCEL_STOPPED_FALLBACK phase=release now=%.3f max_joint_speed='
                                  '%.5f rad/s <= %.5f for %d ticks: the place-cancel terminal never arrived '
                                  '(%.2fs sim) but the arm is provably stopped -> gripper-open + detach from '
                                  'the stopped pose'
                                  % (now, stopped_speed,
                                     float(getattr(self._carton_level, 'release_cancel_stopped_vel', 0.02)),
                                     self._release_stopped_ticks,
                                     now - self._release_cancel_sim_start), flush=True)
                            self.evidence.append({'release_cancel_stopped_fallback': {
                                'sim_time': float(now), 'max_joint_speed': float(stopped_speed),
                                'vel_threshold': float(getattr(self._carton_level,
                                                               'release_cancel_stopped_vel', 0.02)),
                                'ticks': int(self._release_stopped_ticks)}})
                            self._release_cancel_pending = False
                            # fall through to the gripper-open + detach below (issued stays False)
                        elif now - self._release_cancel_sim_start > self.release_cancel_timeout_s:
                            return self._fail('contact-release place hold-and-settle did not reach a terminal '
                                              'result and the arm never read stopped within %.2fs sim -- not '
                                              'detaching into an in-flight planning scene'
                                              % self.release_cancel_timeout_s)
                        else:
                            return self._result()
                    else:
                        # Wait a bounded SIM-clock budget, then FAIL BY NAME -- never detach blindly.
                        if now - self._release_cancel_sim_start > self.release_cancel_timeout_s:
                            return self._fail('contact-release place-cancel did not reach a terminal result '
                                              'within %.2fs sim of the cancel -- not detaching into an '
                                              'in-flight planning scene' % self.release_cancel_timeout_s)
                        return self._result()
                elif outcome == 'error':
                    # The place execute terminated with a broken result: report, do not detach blindly.
                    return self._fail('contact-release place-cancel terminated with an error -- not '
                                      'detaching into a broken motion terminal')
                else:
                    # Terminal: 'cancelled' (the normal cancel outcome) or 'succeeded' (the place completed /
                    # the contact-hold succeeded). Either is a settled, stopped arm NOT in flight -> proceed to
                    # the gripper-open + detach from the settled pose. The adapter has moved its state out of
                    # the in-flight set, so the detach's _PLAN_IN_FLIGHT_STATES guard now passes.
                    print('PLACE_CONTACT_RELEASE_SETTLED phase=release now=%.3f outcome=%s (place execute '
                          'terminal, arm settled) -> gripper-open + detach from the settled pose'
                          % (now, outcome), flush=True)
                    self.evidence.append({'place_contact_release_settled':
                                          {'outcome': outcome, 'sim_time': float(now)}})
                    self._release_cancel_pending = False
                # fall through to the gripper-open + detach below (same tick; issued is still False)
            if not self.issued:
                if s.valid_until_steady is not None:
                    require_fresh(s.valid_until_steady,self.steady_clock)
                self.issued = True
                self.release_time = now
                # sim-release-open-reissue-v1: arm the re-issue window at the FIRST gripper-open (not at
                # controller construction), so the bounded confirm budget starts here. No-op when off.
                if self._release_reissue is not None:
                    self._release_reissue.reset(now)
                return self._result(({'kind': 'gripper', 'enabled': False},))
            # sim-release-open-reissue-v1 (coldwin84): the gripper-open control is a ONE-SHOT, and coldwin84
            # lost the single open on the last-value /manipulation/status relay (0 Isaac-side command:open)
            # so the SIM_ATTACH weld never released and the run e-stopped. When on, RE-EMIT the gripper-open
            # -- which on the Isaac side is BOTH open AND the weld detach (cad_tool.command(False) ->
            # open_gripper + _release_fallback) -- EVERY tick until the per-tick feedback confirms gripper
            # open + carton not held (seal released), then proceed to the cuMotion detach; fail by name if
            # unconfirmed within the bounded sim window. Off => the legacy single-shot path below =>
            # byte-identical. The cuMotion planning-scene detach (update_payload) is the post-confirmation
            # 'proceed' step -- issued ONCE from the confirmed-released state, never per tick.
            if self._release_reissue is not None:
                dec = self._release_reissue.observe(
                    now=now, held=s.contact, gripper_enabled=s.gripper_enabled,
                    gripper_status_time=s.gripper_status_time)
                if dec.log is not None:
                    print(dec.log, flush=True)
                if dec.action == 'confirmed':
                    self.held = False
                    if not self.motion.update_payload(False, self.profile):
                        return self._fail('payload collision detachment not acknowledged')
                    print('RELEASE_CONFIRMED phase=release now=%.3f ticks=%d elapsed=%.2fs held=%s '
                          'gripper_open=%s status_fresh=%s (seal released -> cuMotion detach)'
                          % (now, dec.ticks, dec.elapsed_s, dec.held, dec.gripper_open,
                             dec.gripper_status_fresh), flush=True)
                    self.evidence.append({'release_confirmed': {
                        'sim_time': float(now), 'ticks': int(dec.ticks), 'elapsed_s': float(dec.elapsed_s)}})
                    self._begin_retreat_after_detach(now)
                    return self._result()
                if dec.action == 'timeout':
                    return self._fail('RELEASE_NOT_CONFIRMED: gripper-open + weld-detach re-issued every '
                                      'tick for %d tick(s) / %.2fs sim but the feedback never confirmed '
                                      'gripper open + carton not held (held=%s gripper_open=%s '
                                      'status_fresh=%s); not restarting the stage with the carton welded'
                                      % (dec.ticks, dec.elapsed_s, dec.held, dec.gripper_open,
                                         dec.gripper_status_fresh))
                # Still held: re-emit the gripper-open + weld-detach control this tick (resilient to the
                # last-value relay drop). No require_fresh (the first open validated the registration;
                # mirrors contact_reissue_suction_each_tick, which re-publishes without re-validating).
                return self._result(({'kind': 'gripper', 'enabled': False},))
            if not s.contact:
                self.held = False
                if not self.motion.update_payload(False, self.profile):
                    return self._fail('payload collision detachment not acknowledged')
                self._begin_retreat_after_detach(now)
                return self._result()
            # sim-robust-cart-place-descent-v1: the gripper-open was emitted but the measured contact has
            # not cleared -- the SIM_ATTACH weld / suction is still holding the carton. self.elapsed counts
            # SIM seconds in this release phase (reset on entry, advanced at the top of step). Fail fast with
            # a named reason rather than carry the carton back up on retreat/stow or stall to phase_timeout.
            if self.release_detach_timeout_s and self.elapsed > self.release_detach_timeout_s:
                return self._fail('release did not detach the held carton within %.1fs sim of the '
                                  'gripper-open command (weld/suction still engaged)'
                                  % self.release_detach_timeout_s)
            return self._result()
        if self.phase == 'stow':
            if s.contact or self.held:
                return self._fail('cannot stow with unexpected held payload')
            if not self.issued:
                if s.valid_until_steady is not None:
                    require_fresh(s.valid_until_steady,self.steady_clock)
                self.motion.begin_joint_goal(self.transport_joints,s.joint_positions,
                                             slow=self._reactive_arm_slow)
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
                # sim-robust-collect-stow-stall-v1: a terminal move_group planner crash/hang fails fast with
                # a named cause before the staged retry. This is the coldliveapproach failure: the stow plan
                # goal was never accepted (move_group SIGABRT-hung, no respawn), so re-sending it only
                # consumed the per-phase timeout -- fail fast named instead.
                crash = self._planner_crash_failfast_or_none()
                if crash is not None:
                    return crash
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
                        # fm461: for the bay transfer with a ground-truth target, mark the verify mode
                        # explicitly as ground_truth (NOT camera-measured) in the evidence + run log so the
                        # report/mission.json never misrepresents this as a sensor verification.
                        gt_transfer = self.transfer_ground_truth and self.operation == 'transfer'
                        self.evidence.append({'source': 'ground_truth_place_check', 'location': self.destination,
                                              'verify_mode': 'ground_truth' if gt_transfer else 'ground_truth_place',
                                              'simulation_time': float(now), 'carton_pose_station': tuple(c),
                                              'support_pose_station': tuple(self.target_surface),
                                              'xy_offset': float(xy), 'bottom_gap': float(bottom_gap)})
                        if self.collect_ground_truth and self.operation == 'collect':
                            self.evidence[-1]['collect_mode'] = gtc.MODE   # collect-gt: carried to mission.json
                        if gt_transfer:
                            print('GROUND_TRUTH_VERIFY transfer destination=%s carton=%s bay_target=%s '
                                  'xy_offset=%.3f bottom_gap=%.3f (ground-truth, not camera)'
                                  % (self.destination, tuple(round(v, 3) for v in c),
                                     tuple(round(v, 3) for v in self.target_surface), xy, bottom_gap), flush=True)
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
