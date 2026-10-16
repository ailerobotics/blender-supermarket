"""Bounded transfer state machine; all runtime evidence must originate in sensors."""
from dataclasses import asdict
import math
import time
from ..restock_camera.freshness import require_fresh

from .types import MotionPort, ProductProfile, Result, Sensors


class TransferController:
    def __init__(self, motion: MotionPort, profiles: dict[str, ProductProfile], *,
                 cart_location: str = 'rollwagen', max_age: float = 0.5,
                 phase_timeout: float = 30.0, fixture_mode: bool = False,
                 frame: str = 'ur_base_link', pose_policy=None, transport_joints=None,
                 steady_clock=time.monotonic):
        self.steady_clock=steady_clock
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
        self.cart_location, self.max_age = cart_location, max_age
        self.phase_timeout, self.fixture_mode = phase_timeout, fixture_mode
        self.phase = 'idle'
        self.held = False
        self.evidence = []
        self.reason = ''
        self.last_time = None

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
            if (len(target.free_dimensions) != 3 or not self._pose_ok(target.pose)
                    or any(not math.isfinite(a) or a < b + .02
                           for a, b in zip(target.free_dimensions, self.profile.dimensions))):
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
            if not all(self._pose_ok(v) for v in self.poses.values()):
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
                self.motion.begin(self.poses[self.phase], s.joint_positions)
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
                if s.valid_until_steady is None:
                    return self._result()
                require_fresh(s.valid_until_steady, self.steady_clock)
                self._contact_suction = True
                return self._result(({'kind': 'gripper', 'enabled': True},))
            outcome = self.motion.poll()
            if outcome == 'failed':
                return self._fail('collision planning or execution failed')
            if outcome not in ('running', 'succeeded'):
                return self._fail('invalid motion outcome')
            if outcome == 'succeeded':
                next_phase = dict(approach='contact', contact='grasp', lift='preplace',
                                  preplace='place', place='release',
                                  retreat='stow' if self.transport_joints is not None else 'verify')
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
