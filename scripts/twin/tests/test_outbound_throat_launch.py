"""Recorded-trace proof for sim-robust-throat-momentum-launch-v1.

Drives the REAL ``ArticulatedController`` coupled authored-lane follower against the REAL recorded
coldwin18 coupled-OUTBOUND approach to the sales_backroom_portal throat. coldwin18 = the COMPLETE
coldwin16 config + the PR258 pre-throat alignment gate: the base entered the throat ALIGNED
(yaw ~ -1 deg, throat_misaligned=0) but then PINNED from rest at x=-3.935 while aligned
(NAV_BREAKAWAY enter x3, coupled_no_progress). CPU scene inspection of the mission.usda around the
throat finds NO door threshold / sill / jamb / wall / floor seam -- the pin is a from-low-momentum
PhysX wheel/caster contact, stochastic across byte-identical runs; the passers rolled through it at
cruise carrying straight-travel momentum. The momentum launch (``heading_hold_launch_m`` +
``heading_hold_launch_mps``) proves the brief's acceptance for carrying that momentum deterministically:

  * on the recorded coldwin18 creep the controller FINISHES alignment before the launch window and,
    once aligned inside the launch window, commands NO steering (curvature == 0) at CRUISE (>= the
    launch speed), so the trailing casters stay aligned and the wheels carry momentum across the pin;
  * the launch never commands a negative linear velocity, and never below the follower's own speed
    (it only ADDS momentum, never crawls);
  * the launch window straddles the throat (fires both before AND after the lip x), so the straight
    cruise spans the recorded pin x=-3.935;
  * a base still MISALIGNED on entry keeps the bounded PR258 alignment gate (never a silent wedge);
  * key OFF (``heading_hold_launch_m`` = 0) is byte-identical to the PR258 gate command stream;
  * the launch_m < prethroat_m and requires-prethroat invariants are enforced at construction.

CPU-only; no Isaac/GPU/ROS. Run with:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -I -m pytest \
    scripts/twin/tests/test_outbound_throat_launch.py -q
"""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.mobile import MobileSpec
from twin.navigation import GridMap, Odometry, Observation
from twin.navigation.articulated import ArticulatedController, ArticulatedMap, TrainState, advance
from twin.navigation.lanes import TowLaneGraph
from twin.navigation.core import wrap

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'outbound_throat_launch_nav.json'
BOUND_DEG = 1.0
CREEP = 0.14
PRETHROAT_M = 0.8          # the pre-throat alignment window along the travel axis (PR258)
LAUNCH_M = 0.25            # the momentum launch window either side of the throat
LAUNCH_MPS = 0.18          # the straight-cruise launch speed (the passers' measured cruise)
THROAT = [-3.875, 15.26]


def _fixture():
    return json.loads(FIXTURE.read_text())


def _controller(**kw):
    fix = _fixture()
    spec = MobileSpec()
    world = GridMap((-20, 5, 10, 25))
    graph = TowLaneGraph.from_config([fix['lane']])
    graph.expand_all(ArticulatedMap(world, spec))
    params = dict(allow_test_fixtures=True, lane_graph=graph, tow_lane_goal_stop=True,
                  lane_trust_unknown=True, goal_timeout=900.0, blocked_timeout=120.0,
                  lane_track_tol=0.40)
    params.update(kw)
    c = ArticulatedController(world, spec, **params)
    c.start(tuple(fix['lane']['goal']), True)
    return c, spec


def _gate_on(**kw):
    p = dict(coupled_heading_hold=True, heading_hold_bound_deg=BOUND_DEG, heading_hold_creep_mps=CREEP,
             heading_hold_prethroat_m=PRETHROAT_M, heading_hold_throat=THROAT,
             heading_hold_align_timeout_s=0.0)
    p.update(kw)
    return _controller(**p)


def _launch_on(**kw):
    return _gate_on(heading_hold_launch_m=LAUNCH_M, heading_hold_launch_mps=LAUNCH_MPS, **kw)


def _reach_tracking(c, spec):
    fix = _fixture()
    sx, sy, syaw = fix['lane']['start']
    state = TrainState(sx, sy, syaw, 0.0)
    v = w = 0.0
    now = 0.0
    for i in range(600):
        now = (i + 1) * .05
        od = Odometry(state.pose, now, v, w, 'test_fixture', .02)
        obs = Observation(now, (), True, True, 0.0, 'test_fixture', 10)
        r = c.step(.05, od, obs, now=now)
        v, w = r.velocity
        if c.route is not None and c.route.states and r.reason == 'tracking':
            return now
        state = advance(state, v * .05, w / v if v else 0.0, spec)
    raise AssertionError('controller never reached tracking on the lane')


def _replay_open(c, spec, rows, start_now):
    """Open-loop: feed the recorded poses + recorded forward speed; collect the controller's command.
    The launch overrides are a faithful function of the recorded (aligned) heading and x."""
    cmds = []
    now = start_now
    prev = None
    for row in rows:
        if prev is not None:
            now += max(1e-3, row['world_t'] - prev['world_t'])
        prev = row
        yaw = math.radians(row['yaw_deg'])
        od = Odometry((row['x'], row['y'], yaw), now, row['lin'], row['rw'], 'test_fixture', .02)
        obs = Observation(now, (), True, True, 0.0, 'test_fixture', 10)
        res = c.step(.05, od, obs, now=now)
        cmds.append((row['x'], row['yaw_deg'], res.reason, res.velocity[0], res.velocity[1]))
    return cmds


class ThroatLaunchTests(unittest.TestCase):
    def setUp(self):
        self.fix = _fixture()
        self.axis = self.fix['lane']['goal'][2]            # travel axis = goal heading (0 deg east)
        self.rows = self.fix['coldwin18']['rows']
        self.tx = THROAT[0]

    def _ahead(self, x):
        return self.tx - x                                 # axis east: +x travel, throat ahead when >0

    # ---- (1) the launch fires inside the window and commands straight cruise ----------------------

    def test_launch_commands_straight_cruise_in_window(self):
        c, spec = _launch_on()
        now = _reach_tracking(c, spec)
        cmds = _replay_open(c, spec, self.rows, now)
        self.assertGreaterEqual(c.heading_launches, 1, 'the momentum launch never fired on the aligned creep')
        launched = False
        for x, ydeg, reason, v, w in cmds:
            if reason != 'tracking':
                continue
            ae = wrap(math.radians(ydeg) - self.axis)
            in_launch = abs(self._ahead(x)) <= LAUNCH_M
            if in_launch and abs(ae) <= math.radians(BOUND_DEG):
                launched = True
                # NO steering inside the launch window: commanded angular velocity is 0 (curvature 0).
                self.assertEqual(w, 0.0, f'launch commanded steering at x={x:.3f} (w={w:.5f})')
                # CRUISE: at least the launch speed (carries momentum), never above the cap.
                self.assertGreaterEqual(v, LAUNCH_MPS - 1e-9,
                                        f'launch crawled below cruise at x={x:.3f} (v={v:.4f})')
                self.assertLessEqual(v, spec.max_speed + 1e-9)
        self.assertTrue(launched, 'no aligned-in-window tick exercised the launch')

    # ---- (2) alignment finishes BEFORE the launch window -----------------------------------------

    def test_alignment_complete_before_launch_window(self):
        # On the recorded coldwin18 creep the base is already aligned (|yaw| < bound) by the time it
        # reaches the launch window, so the alignment gate (heading_holds) must NOT be what fires inside
        # the launch window -- the launch (heading_launches) owns it. Prove: every tick inside the launch
        # window advances heading_launches (straight cruise), not heading_holds (creep + corrective turn).
        c, spec = _launch_on()
        now = _reach_tracking(c, spec)
        now2 = now
        prev = None
        steered_in_launch = 0
        for row in self.rows:
            if prev is not None:
                now2 += max(1e-3, row['world_t'] - prev['world_t'])
            prev = row
            yaw = math.radians(row['yaw_deg'])
            od = Odometry((row['x'], row['y'], yaw), now2, row['lin'], row['rw'], 'test_fixture', .02)
            obs = Observation(now2, (), True, True, 0.0, 'test_fixture', 10)
            h0, l0 = c.heading_holds, c.heading_launches
            res = c.step(.05, od, obs, now=now2)
            in_launch = abs(self._ahead(row['x'])) <= LAUNCH_M
            if in_launch and res.reason == 'tracking' and c.heading_holds > h0:
                steered_in_launch += 1
        self.assertEqual(steered_in_launch, 0,
                         'the alignment gate (creep + corrective turn) fired inside the launch window; '
                         'alignment did not finish before the launch')

    # ---- (3) the launch window straddles the throat (spans the recorded pin x=-3.935) -------------

    def test_launch_spans_the_recorded_pin(self):
        c, spec = _launch_on()
        now = _reach_tracking(c, spec)
        cmds = _replay_open(c, spec, self.rows, now)
        # the recorded pin x=-3.935 is WEST of the throat (-3.875): ahead=+0.06 <= LAUNCH_M, so it is
        # inside the launch window and crossed with straight cruise.
        pin = next((t for t in cmds if abs(t[0] - (-3.935)) < 1e-3 and t[2] == 'tracking'
                    and abs(wrap(math.radians(t[1]) - self.axis)) <= math.radians(BOUND_DEG)), None)
        self.assertIsNotNone(pin, 'no aligned tracking tick at the recorded pin x=-3.935')
        self.assertEqual(pin[4], 0.0, 'the launch steered at the pin x=-3.935')
        self.assertGreaterEqual(pin[3], LAUNCH_MPS - 1e-9, 'the launch crawled at the pin x=-3.935')

    # ---- (4) forward-only, and never below the follower's own speed ------------------------------

    def test_launch_forward_only_and_never_crawls(self):
        c, spec = _launch_on()
        now = _reach_tracking(c, spec)
        on = _replay_open(c, spec, self.rows, now)
        # the same approach with the launch OFF (PR258 gate only) -> the follower's own v per tick, in the
        # SAME tick order (both replays feed the identical recorded poses open-loop).
        off_c, spec2 = _gate_on()
        now_off = _reach_tracking(off_c, spec2)
        off = _replay_open(off_c, spec2, self.rows, now_off)
        self.assertEqual(len(on), len(off))
        for (x, ydeg, reason, v, w), (_, _, _, voff, _) in zip(on, off):
            self.assertGreaterEqual(v, 0.0, f'launch commanded negative linear velocity at x={x:.3f}')
            if reason == 'tracking' and abs(self._ahead(x)) <= LAUNCH_M \
                    and abs(wrap(math.radians(ydeg) - self.axis)) <= math.radians(BOUND_DEG):
                self.assertGreaterEqual(v + 1e-9, voff,
                                        f'launch reduced the follower speed at x={x:.3f} (crawl, not momentum)')

    # ---- (5) key OFF is byte-identical to the PR258 gate -----------------------------------------

    def test_key_off_byte_identical(self):
        on, spec = _gate_on(heading_hold_launch_m=0.0)       # launch explicitly off
        n1 = _reach_tracking(on, spec)
        a = _replay_open(on, spec, self.rows, n1)
        off, spec2 = _gate_on()                              # launch key absent entirely
        n2 = _reach_tracking(off, spec2)
        b = _replay_open(off, spec2, self.rows, n2)
        self.assertEqual([t[3:] for t in a], [t[3:] for t in b],
                         'heading_hold_launch_m=0 differs from the PR258 gate (not byte-identical)')
        self.assertEqual(on.heading_launches, 0)
        self.assertEqual(off.heading_launches, 0)

    # ---- (6) construction invariants -------------------------------------------------------------

    def test_launch_requires_prethroat_and_is_shorter(self):
        with self.assertRaises(ValueError):           # launch without the prethroat gate
            _controller(coupled_heading_hold=True, heading_hold_launch_m=0.25)
        with self.assertRaises(ValueError):           # launch_m >= prethroat_m (no alignment runway)
            _gate_on(heading_hold_launch_m=PRETHROAT_M)
        with self.assertRaises(ValueError):           # negative / out of range
            _gate_on(heading_hold_launch_mps=2.0)


if __name__ == '__main__':
    unittest.main()
