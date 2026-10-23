"""Quasi-static CPU fixtures: these do not certify PhysX dynamics or sensing."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec, wrap
from twin.navigation import GridMap, Odometry, Observation
from twin.navigation.articulated import (ArticulatedMap,ArticulatedPlanner,
    ArticulatedController,TrainState,advance,dubins_connections)


class ArticulatedTests(unittest.TestCase):
    def setUp(self):
        self.spec=MobileSpec()
        self.world=ArticulatedMap(GridMap((-15,-15,15,15)))

    def test_actual_train_straight_through_2_5m_portal(self):
        world=ArticulatedMap(GridMap((-8,-8,8,8),((-8,-.15,-1.25,.15),(1.25,-.15,8,.15))))
        state=TrainState(0,4,-math.pi/2,-math.pi/2)
        samples=world.rollout(state,8,0,extra_margin=.02)
        self.assertTrue(samples)
        self.assertTrue(all(world.valid(s,extra_margin=.02) for s in samples))

    def test_narrow_aisle_straight_passes_but_tow_turn_fails(self):
        world=ArticulatedMap(GridMap((-8,-8,8,8),((-8,-8,-.75,8),(.75,-8,8,8))))
        state=TrainState(0,4,-math.pi/2,-math.pi/2)
        self.assertTrue(world.rollout(state,4,0))
        self.assertFalse(world.rollout(state,3,1/1.5))
        self.assertFalse(world.rollout(state,4,0,extra_margin=.05))

    def test_cart_is_independent_and_hitches_remain_coincident(self):
        state=TrainState(0,0,0,0)
        for _ in range(150):
            state=advance(state,.02,1/1.5,self.spec)
            base,cart=state.bodies(self.spec)
            self.assertLess(math.dist(base.point(self.spec.hitch_x),cart.point(self.spec.cart_hitch_x)),1e-10)
        self.assertGreater(abs(state.articulation),.5)
        self.assertLess(abs(state.articulation),self.spec.max_articulation)

    def test_dubins_shots_reach_pose_with_legal_curvature(self):
        start=TrainState(0,0,.3,.3)
        goal=(3,-2,-1.2)
        for shot in dubins_connections(start,goal):
            s=start
            for length,k in shot:
                self.assertLessEqual(abs(k),1/1.5)
                for _ in range(max(1,math.ceil(length/.02))):
                    s=advance(s,length/max(1,math.ceil(length/.02)),k,self.spec)
            self.assertLess(math.dist(s.pose[:2],goal[:2]),.001)
            self.assertLess(abs(wrap(s.yaw-goal[2])),1e-8)

    def test_planner_aligns_cart_and_returns(self):
        planner=ArticulatedPlanner(self.world,timeout=5)
        route=planner.plan(TrainState(0,0,0,0),(5,3,0),goal_articulation=.08)
        self.assertTrue(route.states,route.reason)
        self.assertLess(abs(route.states[-1].articulation),.08)
        back=planner.plan(route.states[-1],(0,0,0),goal_articulation=.08)
        self.assertTrue(back.states,back.reason)
        self.assertLess(abs(back.states[-1].articulation),.08)

    def test_articulated_braking_and_joint_limit_stop(self):
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,0,0),True)
        od=Odometry((0,0,0),.1,.2,source='test_fixture',position_uncertainty=.02)
        for points,angle in [(((-1.5,0),),0),((),1.2)]:
            obs=Observation(.1,points,True,True,angle,'test_fixture',10)
            self.assertEqual(c.gate_velocity(.1,od,obs,.2,0).velocity,(0,0))

    def test_no_pivot_reverse_or_too_tight_radius(self):
        self.assertFalse(self.world.rollout(TrainState(0,0,0,0),-1,0))
        self.assertFalse(self.world.rollout(TrainState(0,0,0,0),1,1))
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,0,0),True)
        od=Odometry((0,0,0),.1,source='test_fixture')
        obs=Observation(.1,(),True,True,0,'test_fixture',10)
        self.assertEqual(c.gate_velocity(.1,od,obs,0,.1).reason,'turn_radius')

    def test_feedback_controller_curved_goal(self):
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,3,0),True)
        state=TrainState(0,0,0,0); v=w=0
        for i in range(2000):
            now=(i+1)*.05
            od=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            result=c.step(.05,od,obs,now=now)
            v,w=result.velocity
            self.assertGreaterEqual(v,0)
            self.assertLessEqual(abs(w)*1.5,v+1e-9)
            state=advance(state,v*.05,w/v if v else 0,self.spec)
            if result.status!='running': break
        self.assertEqual(result.status,'succeeded',result.reason)

    def test_transient_measured_turn_radius_keeps_route(self):
        # A transient measured-curvature trip (base momentarily under commanded speed on a
        # tight turn) must keep the planned route and coast, not discard it + full stop/replan.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,3,0),True)
        state=TrainState(0,0,0,0); v=w=0.0; now=0.0
        for i in range(400):
            now=(i+1)*.05
            od=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            v,w=r.velocity
            if c.route is not None and c.route.states and r.reason=='tracking':
                break
            state=advance(state,v*.05,w/v if v else 0,self.spec)
        self.assertTrue(c.route and c.route.states,'controller never reached tracking')
        route=c.route
        # Inject odometry: base just above the 0.02 gate but its MEASURED arc is well past the
        # 1/1.5 min-radius cap -> the braking-curvature gate trips measured_turn_radius.
        now+=.05
        tight=Odometry(state.pose,now,0.05,0.05*(1/1.5+0.3),'test_fixture',.02)
        obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
        r=c.step(.05,tight,obs,now=now)
        self.assertEqual(r.reason,'measured_turn_radius')
        self.assertIs(c.route,route)          # route PRESERVED (no discard, no replan)
        self.assertNotEqual(r.status,'failed')  # non-terminal wait, resumes the route on recovery
        # Contrast: a genuine obstacle refusal still drops the route for replanning.
        now+=.05
        ring=tuple((state.x+.1*math.cos(a),state.y+.1*math.sin(a))
                   for a in [k*math.pi/8 for k in range(16)])
        od=Odometry(state.pose,now,v,w,'test_fixture',.02)
        obs=Observation(now,ring,True,True,state.articulation,'test_fixture',10)
        c.step(.05,od,obs,now=now)
        self.assertIsNone(c.route)            # dropped -> will replan

    def _reach_tracking(self, c):
        state=TrainState(0,0,0,0); v=w=0.0; now=0.0
        for i in range(400):
            now=(i+1)*.05
            od=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            v,w=r.velocity
            if c.route is not None and c.route.states and r.reason=='tracking':
                break
            state=advance(state,v*.05,w/v if v else 0,self.spec)
        self.assertTrue(c.route and c.route.states,'controller never reached tracking')
        return state, now

    def test_cart_yaw_rebase_matches_live_aligned_entry(self):
        # fm451: re-base the expected cart_yaw from the base's ACTUAL aligned entry. The nearest route
        # point's cart_yaw becomes the live cart_yaw (entry divergence ~0), differing from the authored
        # (swung) value; the re-based tail stays in the empty-map envelope so it is used.
        from twin.navigation.articulated import Route
        c = ArticulatedController(GridMap((-15, -15, 15, 15)), allow_test_fixtures=True,
                                  lane_cart_yaw_rebase=True)
        authored = [TrainState(i * 0.4, 0.0, -0.30, 0.20) for i in range(8)]   # authored cart_yaw swung +0.20
        route = Route(authored, [0.0] * 7, 'validated_lane')
        base = TrainState(1.2, 0.0, -0.30, -0.30)                              # at index 3, cart ALIGNED
        rebased = c._rebase_cart_yaw(route, base, c.world)
        self.assertAlmostEqual(rebased.states[3].cart_yaw, base.cart_yaw, places=9)  # entry divergence ~0
        self.assertNotAlmostEqual(rebased.states[3].cart_yaw, authored[3].cart_yaw)  # changed from authored
        self.assertEqual(len(rebased.states), len(authored))                   # same length/centreline

    def test_cart_yaw_rebase_keeps_authored_when_it_would_breach(self):
        # fm451 safety: if re-basing from a departed cart would put the swept envelope into an obstacle,
        # the authored route is kept (so a genuinely off-lane cart still trips stop_replan, never masked).
        from twin.navigation.articulated import Route
        grid = GridMap((-15, -15, 15, 15), ((0.0, -1.2, 4.0, -0.2),))          # wall just below the route
        c = ArticulatedController(grid, allow_test_fixtures=True, lane_cart_yaw_rebase=True)
        authored = [TrainState(i * 0.4, 0.0, 0.0, 0.0) for i in range(8)]
        route = Route(authored, [0.0] * 7, 'validated_lane')
        base = TrainState(1.2, 0.0, 0.0, -1.4)                                 # cart yawed hard toward the wall
        rebased = c._rebase_cart_yaw(route, base, c.world)
        self.assertIs(rebased, route)                                          # breach -> authored kept

    def test_track_envelope_caps_tol_by_local_lateral_slack(self):
        # fm449: with lane_track_envelope the stop_replan tol = min(lane_track_tol, local_slack - 0.05),
        # floored 0.10. Open space -> wide slack (tol loosens toward lane_track_tol); a wall beside a route
        # point -> narrower slack (tol tightens). Off (default) is the flat tol everywhere.
        openc = ArticulatedController(GridMap((-15, -15, 15, 15)), allow_test_fixtures=True,
                                      lane_track_tol=0.35, lane_track_envelope=True)
        open_slack = openc._lateral_slack(TrainState(0, 0, 0, 0))
        self.assertGreater(open_slack, 0.4)                      # empty map: wide lateral headroom
        walled = ArticulatedController(GridMap((-15, -15, 15, 15), ((-2, 0.55, 2, 2.0),)),
                                       allow_test_fixtures=True, lane_track_tol=0.35, lane_track_envelope=True)
        wall_slack = walled._lateral_slack(TrainState(0, 0, 0, 0))   # wall 0.55 m to the +y (lateral) side
        self.assertLess(wall_slack, open_slack)                  # the wall tightens the clear lateral offset
        self.assertLess(wall_slack, 0.35)                        # -> effective tol would tighten below 0.35
        # memoised: a second call returns the same value
        self.assertEqual(walled._lateral_slack(TrainState(0, 0, 0, 0)), wall_slack)

    def test_persistent_measured_turn_radius_faults_fast(self):
        # fm446 (C): a measured_turn_radius that trips EVERY tick (a sub-min-radius turn the coupled base
        # cannot make) must FAIL FAST with a named fault, not livelock. measured_turn_fault_s=1.0 => after
        # ~1 s (20 ticks at dt .05) of net over-rotation coast it terminates 'measured_turn_radius_persistent'.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True,measured_turn_fault_s=1.0)
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        reasons=[]
        for _ in range(40):
            now+=.05
            tight=Odometry(state.pose,now,0.05,0.05*(1/1.5+0.3),'test_fixture',.02)   # over-rotating arc
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,tight,obs,now=now)
            reasons.append((r.status,r.reason))
            if r.status=='failed':
                break
        self.assertEqual(reasons[-1],('failed','measured_turn_radius_persistent'))
        self.assertLessEqual(len(reasons),25)                 # ~20 ticks (1.0 s / 0.05), bounded fast

    def test_measured_turn_fault_off_by_default_coasts(self):
        # Default (no measured_turn_fault_s) is byte-identical: persistent measured_turn_radius keeps
        # coasting (never the new fault) -- the legacy behaviour.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        for _ in range(40):
            now+=.05
            tight=Odometry(state.pose,now,0.05,0.05*(1/1.5+0.3),'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,tight,obs,now=now)
            self.assertNotEqual(r.reason,'measured_turn_radius_persistent')

    def test_tight_turn_commands_reduced_speed(self):
        # The tracking command slows for tight turns so the coupled base can keep up: the
        # feedback curved-goal run must never command max_speed while turning hard.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,3,0),True)
        state=TrainState(0,0,0,0); v=w=0.0
        turning_speeds=[]
        for i in range(2000):
            now=(i+1)*.05
            od=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            v,w=r.velocity
            if v>1e-6 and abs(w)/v>0.5*(1/1.5):      # a meaningfully curved command
                turning_speeds.append(v)
            state=advance(state,v*.05,w/v if v else 0,self.spec)
            if r.status!='running': break
        self.assertTrue(turning_speeds,'no curved commands observed')
        self.assertLess(max(turning_speeds),c.max_speed-1e-9)

    def test_seed_replay_checks_full_margin_and_obstacles(self):
        from twin.navigation.articulated import Route
        state=TrainState(0,0,0,0)
        states=[state]
        for _ in range(50): states.append(advance(states[-1],.08,0,self.spec))
        seed=Route(states,[0.0]*50,'fixture')
        planner=ArticulatedPlanner(self.world)
        checked=planner.validate_seed(seed,state,(4,0,0),extra_margin=.02)
        self.assertTrue(checked.states)
        rejected=planner.validate_seed(seed,state,(4,0,0),dynamic=((2,0),),extra_margin=.02)
        self.assertEqual(rejected.reason,'seed_collision')
        # A path valid without localization allowance is NOT silently accepted.
        wall=ArticulatedMap(GridMap((-10,-10,10,10),((-10,.72,10,1.0),)))
        self.assertTrue(ArticulatedPlanner(wall).validate_seed(seed,state,(4,0,0),extra_margin=0).states)
        self.assertFalse(ArticulatedPlanner(wall).validate_seed(seed,state,(4,0,0),extra_margin=.02).states)

    def test_coupled_gate_is_direction_aware_rear_clears_ahead_stops(self):
        # The coupled gate already rolls the braking sweep FORWARD along the commanded curvature
        # (it never bounds the arc with an omnidirectional disk): a static return in the train's
        # rear hemisphere does not trip, while the same return AHEAD, inside the forward braking
        # rollout, does. This locks the coupled equivalent of the direction-aware base zone.
        spec = MobileSpec()
        world = ArticulatedMap(GridMap((-20, -20, 20, 20)))
        c = ArticulatedController(GridMap((-20, -20, 20, 20)), spec, allow_test_fixtures=True)
        c.start((5, 0, 0), True)
        od = Odometry((0, 0, 0), 0.1, 0.0, 0.0, 'test_fixture', 0.05)
        # The train spans robot nose (+0.5) to cart rear (~-1.77 with margins). An obstacle BEYOND
        # the cart rear (-2.0) is behind the whole no-reverse train -> no trip; one just beyond the
        # nose, inside the forward braking rollout (~0.7 m), stops.
        rear = Observation(0.1, ((-2.0, 0.0),), True, True, 0.0, 'test_fixture', 10.0)
        ahead = Observation(0.1, ((0.7, 0.0),), True, True, 0.0, 'test_fixture', 10.0)
        self.assertEqual(c.gate_velocity(0.1, od, rear, 0.25, 0.0).reason, 'tracking')
        self.assertEqual(c.gate_velocity(0.1, od, ahead, 0.25, 0.0).reason, 'obstacle_stop')

    # --- fm-sim-robust-outbound: coupled net-progress monitor (warm10 outbound stall) -----------------
    # warm10/warm9 outbound pinned at a fixed pose while the follower kept COMMANDING forward
    # (reason='tracking' on most ticks) and the base jittered in place (measured |rv| noise, ~0 NET
    # displacement). The _frozen_jitter helper replays that signature: the base pose does NOT advance,
    # but odometry reports a non-zero, jittering measured (rv, rw) exactly as the trace recorded.
    _JITTER_RV = (0.03, 0.20, 0.012, 0.05, 0.04)   # warm10 nav-trace rv: mean ~0.026, spikes to ~0.23
    _JITTER_RW = (0.05, 0.17, 0.03, 0.11, 0.08)

    def _drive_frozen_jitter(self, c, state, now, ticks):
        """Feed `ticks` steps with the base pose FROZEN at `state` but a jittering measured velocity.
        Returns (list of (status, reason), last now, forward_commands)."""
        out=[]; forward=[]
        for i in range(ticks):
            now+=.05
            rv=self._JITTER_RV[i%len(self._JITTER_RV)]
            rw=self._JITTER_RW[i%len(self._JITTER_RW)]
            od=Odometry(state.pose,now,rv,rw,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            out.append((r.status,r.reason))
            forward.append(r.velocity[0])
            if r.status=='failed':
                break
        return out, now, forward

    def test_coupled_no_progress_faults_fast_on_frozen_base(self):
        # With the monitor ON, a forward-commanding tow pinned with jittering velocity (warm10) is
        # caught on the SIM clock: it ESCALATES (forces the committed creep) then FAULTS fast so the
        # stage StageRetry-replans -- in ~fault_s SIM seconds, not the 120 s WALL nav-stuck budget.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True,
                                coupled_progress_escape=True,coupled_progress_min_m=0.05,
                                coupled_progress_escape_s=0.5,coupled_progress_fault_s=1.0)
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        out,now,_=self._drive_frozen_jitter(c,state,now,ticks=60)
        self.assertEqual(out[-1],('failed','coupled_no_progress'))
        self.assertLessEqual(len(out),25)            # ~20 ticks (1.0 s / 0.05 dt): bounded, fast
        self.assertGreater(c.progress_escapes,0)     # the creep was forced BEFORE the fault (escalate)

    def test_coupled_progress_escape_off_by_default_livelocks(self):
        # Default (monitor OFF) is byte-identical: the SAME frozen-jitter signature never produces the
        # new fault -- it livelocks exactly as warm10 did (only the WALL nav-stuck would eventually fail
        # it). Proves we reproduce the stall and that the fix is opt-in.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True)
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        out,_,_=self._drive_frozen_jitter(c,state,now,ticks=200)
        self.assertEqual(len(out),200)               # ran to the end, never terminated
        self.assertFalse(any(reason=='coupled_no_progress' for _,reason in out))
        self.assertEqual(c.progress_escapes,0)

    def test_coupled_progress_monitor_does_not_false_trip_on_a_healthy_tow(self):
        # A coupled tow that actually MOVES re-anchors well within the SIM budgets, so with the
        # PRODUCTION defaults (escape 3 s / fault 8 s / 5 cm) the monitor never escalates nor faults a
        # healthy drive to the curved goal -- slow (curve-floor) sections still clear >=min_m in time.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True,coupled_progress_escape=True)
        c.start((5,3,0),True)
        state=TrainState(0,0,0,0); v=w=0.0; r=None
        for i in range(2000):
            now=(i+1)*.05
            od=Odometry(state.pose,now,v,w,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            v,w=r.velocity
            state=advance(state,v*.05,w/v if v else 0,self.spec)
            if r.status!='running': break
        self.assertEqual(r.status,'succeeded',r.reason)
        self.assertEqual(c.progress_escapes,0)       # never needed the escape on a healthy tow

    def test_coupled_progress_monitor_resets_after_a_non_tracking_gap(self):
        # Only CONTIGUOUS tracking-with-no-progress accrues: if the base's tracking ticks are separated by
        # a long (> gap) SIM wait (the controller sat in planning_stop/blocked, which return before the
        # monitor), each resumed tick re-anchors a fresh window -> a legitimate wait is never charged
        # against the resumed drive, so a frozen-but-intermittently-tracking base does NOT false-fault.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True,
                                coupled_progress_escape=True,coupled_progress_min_m=0.05,
                                coupled_progress_escape_s=0.5,coupled_progress_fault_s=1.0)
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        for _ in range(40):
            now+=2.0        # a >0.5 s SIM gap between tracking ticks: the monitor sees a fresh window each time
            od=Odometry(state.pose,now,0.0,0.0,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            self.assertNotEqual(r.status,'failed')      # never faults despite being frozen
        self.assertEqual(c._prog_stuck_s,0.0)           # the window reset on every gap

    def test_coupled_progress_escape_rejects_invalid_params(self):
        for kw in (dict(coupled_progress_min_m=0.0), dict(coupled_progress_min_m=-1.0),
                   dict(coupled_progress_escape_s=0.0), dict(coupled_progress_fault_s=0.0),
                   dict(coupled_progress_min_m=float('nan')),
                   dict(coupled_progress_escape_s=5.0,coupled_progress_fault_s=1.0)):  # escape > fault
            with self.assertRaises(ValueError):
                ArticulatedController(self.world.grid,allow_test_fixtures=True,**kw)

    # --- fm-sim-robust-outbound-stall-recovery: bounded break-away recovery -----------------------------

    def _recovery_controller(self, **over):
        kw=dict(coupled_progress_escape=True, coupled_progress_recovery=True,
                coupled_progress_min_m=0.05, coupled_progress_escape_s=0.5, coupled_progress_fault_s=1.0,
                coupled_breakaway_speed_mps=0.18, coupled_breakaway_window_s=1.0,
                coupled_breakaway_pulse_s=0.2, coupled_breakaway_attempts=3)
        kw.update(over)
        return ArticulatedController(self.world.grid, allow_test_fixtures=True, **kw)

    def test_coupled_breakaway_recovery_breaks_a_stuck_tow_free(self):
        # The coldlivefull outbound: a coupled tow pinned (jittering in place, ~0 net displacement) while
        # commanding forward. With recovery ON, the monitor ENTERS a break-away maneuver (a continuous
        # pulsed committed forward) rather than the single weak creep + fail. Here the surge breaks the base
        # free (the fixture lets it move once the surge commands), so the stage recovers: it never faults and
        # a break-away success is recorded.
        c=self._recovery_controller()
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        v=w=0.0; freed=False; out=[]
        for i in range(300):
            now+=.05
            rv=(v if freed else self._JITTER_RV[i%len(self._JITTER_RV)])
            rw=(w if freed else self._JITTER_RW[i%len(self._JITTER_RW)])
            od=Odometry(state.pose,now,rv,rw,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            out.append((r.status,r.reason))
            v,w=r.velocity
            if r.status!='running':
                break
            if c._recovery_active:
                freed=True          # the surge is now commanding -> let the base start to move
            if freed:
                state=advance(state,v*.05,w/v if v else 0,self.spec)
        self.assertGreaterEqual(c.breakaway_recoveries,1)          # a recovery was entered
        self.assertGreaterEqual(c.breakaway_successes,1)           # it broke the base free
        self.assertFalse(any(reason=='coupled_no_progress' for _,reason in out))  # never faulted
        self.assertFalse(c._recovery_active)                       # recovery cleared on break-away

    def test_coupled_breakaway_commands_continuous_pulsed_forward(self):
        # The maneuver's substance: while recovering, EVERY commanded velocity is a forward surge (> 0) --
        # never the v=0 stop that the measured_turn_radius coast returns and that re-rests the stuck tow --
        # and it PULSES between the high and the relaxed (0.4x) level to rock the contact.
        c=self._recovery_controller(coupled_breakaway_attempts=5)   # keep recovering long enough to sample
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        surges=[]
        for i in range(120):
            now+=.05
            rv=self._JITTER_RV[i%len(self._JITTER_RV)]
            rw=self._JITTER_RW[i%len(self._JITTER_RW)]
            od=Odometry(state.pose,now,rv,rw,'test_fixture',.02)
            obs=Observation(now,(),True,True,state.articulation,'test_fixture',10)
            r=c.step(.05,od,obs,now=now)
            if c._recovery_active and r.reason=='tracking':
                surges.append(r.velocity[0])
            if r.status!='running':
                break
        self.assertTrue(surges)
        self.assertTrue(all(s>0 for s in surges))                  # no v=0 stop-go thrash during recovery
        hi=0.18; lo=0.18*0.4
        self.assertTrue(any(abs(s-hi)<1e-6 for s in surges))       # the high pulse level appears
        self.assertTrue(any(abs(s-lo)<1e-6 for s in surges))       # the relaxed pulse level appears

    def test_coupled_breakaway_exhausts_attempts_then_faults_by_name(self):
        # A base that never breaks free: the recovery runs a BOUNDED number of attempt windows then FAULTS
        # 'coupled_no_progress' (same name/StageRetry contract), fast on the SIM clock -- not the 120 s WALL
        # nav-stuck. The bound is escape_s + attempts * window_s.
        c=self._recovery_controller()
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        out,now,_=self._drive_frozen_jitter(c,state,now,ticks=400)
        self.assertEqual(out[-1],('failed','coupled_no_progress'))
        self.assertEqual(c.breakaway_recoveries,1)
        self.assertEqual(c.breakaway_successes,0)
        # bounded: 0.5 s to enter + 3 windows x 1.0 s = 3.5 s SIM / 0.05 dt ~= 70 ticks (+ small slack)
        self.assertLessEqual(len(out),80)

    def test_coupled_breakaway_recovery_off_is_byte_identical(self):
        # Recovery OFF (default): the escalate->fault ladder is unchanged and the break-away counters stay 0.
        c=ArticulatedController(self.world.grid,allow_test_fixtures=True,
                                coupled_progress_escape=True,coupled_progress_min_m=0.05,
                                coupled_progress_escape_s=0.5,coupled_progress_fault_s=1.0)
        self.assertFalse(c.coupled_progress_recovery)
        c.start((5,3,0),True)
        state,now=self._reach_tracking(c)
        out,now,_=self._drive_frozen_jitter(c,state,now,ticks=60)
        self.assertEqual(out[-1],('failed','coupled_no_progress'))
        self.assertGreater(c.progress_escapes,0)          # the legacy forced creep still ran
        self.assertEqual(c.breakaway_recoveries,0)        # recovery never engaged
        self.assertEqual(c.breakaway_successes,0)

    def test_coupled_breakaway_rejects_invalid_params(self):
        for kw in (dict(coupled_breakaway_speed_mps=0.0), dict(coupled_breakaway_speed_mps=-1.0),
                   dict(coupled_breakaway_speed_mps=1.5), dict(coupled_breakaway_speed_mps=float('nan')),
                   dict(coupled_breakaway_window_s=0.0), dict(coupled_breakaway_pulse_s=0.0),
                   dict(coupled_breakaway_attempts=0), dict(coupled_breakaway_attempts=21),
                   dict(coupled_breakaway_attempts=2.5), dict(coupled_breakaway_speed_mps=True),
                   dict(coupled_breakaway_attempts=True)):
            with self.assertRaises(ValueError):
                ArticulatedController(self.world.grid,allow_test_fixtures=True,**kw)

    def test_seed_rejects_teleport(self):
        from twin.navigation.articulated import Route
        a=TrainState(0,0,0,0)
        b=TrainState(0,.08,0,0)
        result=ArticulatedPlanner(self.world).validate_seed(Route([a,b],[0],'fixture'),a,b.pose)
        self.assertEqual(result.reason,'seed_kinematics_mismatch')


if __name__=='__main__': unittest.main()
