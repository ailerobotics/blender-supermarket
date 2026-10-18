"""Bounded transfer state machine; all runtime evidence must originate in sensors."""
from dataclasses import asdict
import math
import time
from ..restock_camera.freshness import require_fresh

from .fit import axis_fits
from .types import MotionPort, ProductProfile, Result, Sensors

# Controller ticks to HOLD a settled contact step before re-checking the latch, so the arm link
# velocity decays to ~0 (the native SurfaceGripper seals a STATIONARY pad but not a moving one --
# stations verdict, firstmate run-203). Each tick is a sensor frame = many physics steps under pacing.
CONTACT_FREEZE_TICKS = 5


class TransferController:
    def __init__(self, motion: MotionPort, profiles: dict[str, ProductProfile], *,
                 cart_location: str = 'rollwagen', max_age: float = 0.5,
                 phase_timeout: float = 30.0, fixture_mode: bool = False,
                 frame: str = 'ur_base_link', pose_policy=None, transport_joints=None,
                 steady_clock=time.monotonic, contact_straight_line: bool = False,
                 contact_overpress_step_m: float = 0.0, contact_overpress_cap_m: float = 0.0,
                 staged_cart_swing: bool = False, cart_pan_joint0: float = 0.0,
                 cart_tuck_shape=None, carry_wrist1_cap=None,
                 cart_cage_place: bool = False, cage_axis=(1.0, 0.0, 0.0),
                 cage_mouth_standoff: float = 0.55, cage_insert_gap: float = 0.03):
        self.steady_clock=steady_clock
        # fm327 belt-and-braces wrist_1 cap for the task-space LIFT carry move (None = off).
        self.carry_wrist1_cap = None if carry_wrist1_cap is None else float(carry_wrist1_cap)
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
        self.cart_location, self.max_age = cart_location, max_age
        self.phase_timeout, self.fixture_mode = phase_timeout, fixture_mode
        self.phase = 'idle'
        self.held = False
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
        self.verify_ids = set()
        self.release_time = None

    def _fresh(self, stamp, now):
        return math.isfinite(stamp) and 0 <= now - stamp <= self.max_age

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
        if (sensors.parked is not True or type(sensors.contact) is not bool
                or not self._fresh(sensors.feedback_time, now)
                or not self._fresh(sensors.collision_time, now)
                or len(sensors.joint_positions) != 6
                or not all(math.isfinite(x) for x in sensors.joint_positions)):
            return self._fail('base moving or stale/missing feedback/collision world')
        if self.held and self.phase != 'release' and not sensors.contact:
            return self._fail('lost grasp')
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
            if not all(self._pose_ok(v) for v in self.poses.values()):
                import math as _m
                bad = {k: round(_m.dist(v[:3], (0, 0, 0)), 3) for k, v in self.poses.items()
                       if not self._pose_ok(v)}
                print('REACH_ENVELOPE_FAIL max_reach=%s bad=%s source=%s target=%s'
                      % (self.profile.max_reach, bad, tuple(round(c, 3) for c in source.pose[:3]),
                         tuple(round(c, 3) for c in target.pose[:3])), flush=True)
                return self._fail('approach outside reach envelope')
            self.target_pose = (*target.pose[:2], target.pose[2] + p.dimensions[2], *target.pose[3:])
            self.target_surface = target.pose
            self.source_pose = source.pose
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
                                  joint_space=joint_space, wrist1_cap=carry_cap,
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
            outcome = self.motion.poll()
            if outcome == 'failed':
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
            outcome = self.motion.poll()
            if outcome == 'failed':
                return self._fail('staged cart-swing %s move failed' % self.phase)
            if outcome == 'succeeded':
                self._transition({'tuck': 'pan', 'pan': 'preplace', 'retreat_tuck': 'stow'}[self.phase])
            elif outcome != 'running':
                return self._fail('invalid staged cart-swing motion outcome')
            return self._result()
        if self.phase == 'grasp':
            if not self.issued:
                if s.valid_until_steady is not None:
                    require_fresh(s.valid_until_steady,self.steady_clock)
                self.issued = True
                return self._result(({'kind': 'gripper', 'enabled': True},))
            if s.contact:
                self.held = True
                if not self.motion.update_payload(True, self.profile):
                    return self._fail('payload collision attachment not acknowledged')
                self._transition('lift')
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
            outcome=self.motion.poll()
            if outcome == 'failed':
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
