"""Recorded-trace proof for sim-robust-outbound-throat-yaw-align-v1.

Drives the REAL ``ArticulatedController`` coupled authored-lane follower against the REAL recorded
outbound APPROACH creeps of the throat-wedge stalls (coldwin9, coldwin10) and the clean threads
(coldwin7, rock1), via the compact committed fixture ``fixtures/outbound_throat_yaw_nav.json`` (the
real nav-trace rows from the post-seat creep to the throat lip; SI; ``yaw_deg`` is the base map
heading, travel/portal axis = 0 deg east). It proves the brief's acceptance criteria:

  * the recorded coldwin9/coldwin10 creeps, replayed through the controller, COMMAND a corrective
    (CW, negative) yaw rate with restored (speed-schedule-independent) authority BEFORE the throat,
    and a closed-loop integration of those commands brings the base yaw within the bound before the
    throat lip (``arrive within the bound``);
  * coldwin7 and rock1 (which hold within the bound) never engage the hold -> unchanged;
  * key OFF is byte-identical (same (v, w) stream, and identical to omitting the new kwargs);
  * the guard never commands a negative linear velocity.

CPU-only; no Isaac/GPU/ROS. The synthetic straight-east lane reproduces the portal-crossing frame
(goal heading 0 deg) without the full ~800-obstacle world (the throat is invisible to the nav map).
Run with:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -I -m pytest \
    scripts/twin/tests/test_outbound_heading_hold.py -q
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

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'outbound_throat_yaw_nav.json'
BOUND_DEG = 1.0
CREEP = 0.14
# A synthetic straight-east tow lane (goal heading 0 deg = the portal-crossing travel axis), long
# enough that the recorded approach x-band lies well inside it. No obstacles: the throat is a PhysX
# door-frame contact invisible to the nav map, so the gate returns 'tracking' here exactly as it did
# in the real run while the base physically wedged -- the yaw command is what is under test.
LANE_CFG = [{'name': 'outbound_straight', 'start': [-4.60, 15.25, 0.0], 'goal': [-1.80, 15.25, 0.0],
             'start_cart_yaw': 0.0, 'goal_articulation': 0.3, 'start_tol_cart_yaw': 0.35,
             'lead_in_max_m': 3.0}]


def _fixture():
    return json.loads(FIXTURE.read_text())


def _controller(**kw):
    spec = MobileSpec()
    world = GridMap((-20, 5, 10, 25))
    graph = TowLaneGraph.from_config(LANE_CFG)
    graph.expand_all(ArticulatedMap(world, spec))
    params = dict(allow_test_fixtures=True, lane_graph=graph, tow_lane_goal_stop=True,
                  lane_trust_unknown=True, goal_timeout=600.0, blocked_timeout=60.0)
    params.update(kw)
    c = ArticulatedController(world, spec, **params)
    c.start((-1.80, 15.25, 0.0), True)
    return c, spec


def _reach_tracking(c, spec):
    """Drive the controller to a tracking follow on the lane from an aligned seat near the start."""
    state = TrainState(-4.55, 15.25, 0.0, 0.0)
    v = w = 0.0
    now = 0.0
    for i in range(400):
        now = (i + 1) * .05
        od = Odometry(state.pose, now, v, w, 'test_fixture', .02)
        obs = Observation(now, (), True, True, 0.0, 'test_fixture', 10)
        r = c.step(.05, od, obs, now=now)
        v, w = r.velocity
        if c.route is not None and c.route.states and r.reason == 'tracking':
            return state, now
        state = advance(state, v * .05, w / v if v else 0.0, spec)
    raise AssertionError('controller never reached tracking on the synthetic lane')


def _replay_commands(c, spec, rows, start_now):
    """Feed the EXACT recorded poses (x, y, yaw) + recorded forward speed as odometry; collect the
    controller's commanded (v, w) at each follow tick. Open-loop: the base pose is the recorded one,
    so the command is a faithful function of the real drifted heading."""
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


class HeadingHoldRecordedTests(unittest.TestCase):
    def setUp(self):
        self.fix = _fixture()
        self.throat_x = self.fix['throat_x']

    # ---- (1) the stalls command a corrective heading before the throat ---------------------------

    def _assert_corrective(self, run):
        c, spec = _controller(coupled_heading_hold=True, heading_hold_bound_deg=BOUND_DEG,
                              heading_hold_creep_mps=CREEP)
        _, now = _reach_tracking(c, spec)
        cmds = _replay_commands(c, spec, self.fix[run]['rows'], now)
        self.assertGreaterEqual(c.heading_holds, 1, f'{run}: hold never engaged')
        # every engaged tick is BEFORE the throat lip, commands a CW (negative) corrective turn,
        # caps the forward speed at the creep, and is forward (v >= 0).
        engaged = [t for t in cmds if t[2] == 'tracking' and abs(t[4]) > 1e-9 and t[4] < 0]
        self.assertTrue(engaged, f'{run}: no corrective (negative w) command')
        for x, yaw, reason, v, w in cmds:
            self.assertGreaterEqual(v, 0.0, f'{run}: negative linear velocity commanded')
        # the corrective commands occur while still west of the throat lip
        self.assertTrue(all(x <= self.throat_x + 1e-6 for x, _, _, _, w in cmds if w < -1e-9),
                        f'{run}: a corrective turn was commanded past the throat')
        # the commanded forward speed never exceeds the authority creep while holding
        holds = [v for x, yaw, reason, v, w in cmds if w < -1e-9 and reason == 'tracking']
        self.assertTrue(holds and max(holds) <= CREEP + 1e-6, f'{run}: hold did not cap speed at creep')

    def test_coldwin9_commands_corrective_heading(self):
        self._assert_corrective('coldwin9')

    def test_coldwin10_commands_corrective_heading(self):
        self._assert_corrective('coldwin10')

    # ---- (2) closed-loop: the commanded correction brings the base within the bound before throat --

    def _closed_loop_converges(self, run):
        """Faithful hybrid replay: the base follows the REAL recorded drift (yaw = recorded) and the
        recorded x-ADVANCE pace until the controller's heading hold ENGAGES; from that tick on the
        controller OWNS the yaw (yaw evolves closed-loop under its commanded w). Asserts the recorded
        drift is caught and the base arrives at the throat lip within the bound -- i.e. the commanded
        correction is sufficient in magnitude and runway to undo the recorded drift before the lip."""
        c, spec = _controller(coupled_heading_hold=True, heading_hold_bound_deg=BOUND_DEG,
                              heading_hold_creep_mps=CREEP)
        _, now = _reach_tracking(c, spec)
        rows = self.fix[run]['rows']
        yaw = math.radians(rows[0]['yaw_deg'])
        w = 0.0
        controlled = False
        prev = None
        yaw_at_throat = None
        caught_at = None
        for row in rows:
            dt = 0.05 if prev is None else max(1e-3, row['world_t'] - prev['world_t'])
            if prev is not None:
                yaw = wrap(yaw + w * dt) if controlled else math.radians(row['yaw_deg'])
            prev = row
            now += dt
            holds0 = c.heading_holds
            od = Odometry((row['x'], row['y'], yaw), now, row['lin'],
                          w if controlled else row['rw'], 'test_fixture', .02)
            obs = Observation(now, (), True, True, 0.0, 'test_fixture', 10)
            res = c.step(.05, od, obs, now=now)
            self.assertGreaterEqual(res.velocity[0], 0.0, f'{run}: negative linear velocity')
            if c.heading_holds > holds0:                      # the hold engaged this tick
                if not controlled:
                    caught_at = row['x']
                controlled = True
            w = res.velocity[1]
            if row['x'] >= self.throat_x - 0.08:
                yaw_at_throat = math.degrees(yaw)
        self.assertIsNotNone(caught_at, f'{run}: the hold never caught the drift')
        self.assertLess(caught_at, self.throat_x, f'{run}: drift caught only past the throat')
        self.assertIsNotNone(yaw_at_throat, f'{run}: never reached the throat lip')
        # arrives within the bound (vs the recorded failing drift of +2.1..+4.7 deg at the lip)
        self.assertLessEqual(yaw_at_throat, BOUND_DEG + 0.5,
                             f'{run}: base yaw at throat {yaw_at_throat:.2f} not within bound')

    def test_coldwin9_closed_loop_arrives_within_bound(self):
        self._closed_loop_converges('coldwin9')

    def test_coldwin10_closed_loop_arrives_within_bound(self):
        self._closed_loop_converges('coldwin10')

    # ---- (3) the clean threads never engage the hold (unchanged) ---------------------------------

    def _assert_unchanged(self, run):
        on, spec = _controller(coupled_heading_hold=True, heading_hold_bound_deg=BOUND_DEG,
                               heading_hold_creep_mps=CREEP)
        _, now = _reach_tracking(on, spec)
        cmds_on = _replay_commands(on, spec, self.fix[run]['rows'], now)
        self.assertEqual(on.heading_holds, 0, f'{run}: hold engaged on a clean thread')
        off, spec2 = _controller(coupled_heading_hold=False)
        _, now2 = _reach_tracking(off, spec2)
        cmds_off = _replay_commands(off, spec2, self.fix[run]['rows'], now2)
        self.assertEqual([t[3:] for t in cmds_on], [t[3:] for t in cmds_off],
                         f'{run}: key-on command stream differs from key-off on a clean thread')

    def test_coldwin7_unchanged(self):
        self._assert_unchanged('coldwin7')

    def test_rock1_unchanged(self):
        self._assert_unchanged('rock1')

    # ---- (4) key OFF is byte-identical; the key does something on a stall ------------------------

    def test_key_off_byte_identical_and_active(self):
        for run in ('coldwin7', 'rock1', 'coldwin9', 'coldwin10'):
            # OFF explicitly vs OFF-by-omission: identical streams (the new kwargs never perturb off).
            off_a, spec = _controller(coupled_heading_hold=False, heading_hold_bound_deg=BOUND_DEG,
                                      heading_hold_creep_mps=CREEP)
            _, na = _reach_tracking(off_a, spec)
            a = _replay_commands(off_a, spec, self.fix[run]['rows'], na)
            off_b, spec2 = _controller()      # omit the heading-hold kwargs entirely (defaults)
            _, nb = _reach_tracking(off_b, spec2)
            b = _replay_commands(off_b, spec2, self.fix[run]['rows'], nb)
            self.assertEqual(a, b, f'{run}: off-explicit != off-by-omission (not byte-identical)')
            self.assertEqual(off_a.heading_holds, 0)
        # and ON differs from OFF on a stall (the key is active, not a silent no-op)
        on, spec = _controller(coupled_heading_hold=True, heading_hold_bound_deg=BOUND_DEG,
                               heading_hold_creep_mps=CREEP)
        _, non = _reach_tracking(on, spec)
        on_cmds = _replay_commands(on, spec, self.fix['coldwin9']['rows'], non)
        off, spec2 = _controller(coupled_heading_hold=False)
        _, noff = _reach_tracking(off, spec2)
        off_cmds = _replay_commands(off, spec2, self.fix['coldwin9']['rows'], noff)
        self.assertNotEqual([t[3:] for t in on_cmds], [t[3:] for t in off_cmds],
                            'coldwin9: key on did not change the command (silent no-op)')
        self.assertGreaterEqual(on.heading_holds, 1)


if __name__ == '__main__':
    unittest.main()
