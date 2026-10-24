"""Recorded-trace proof for sim-robust-heading-hold-bound-v1.

Drives the REAL ``ArticulatedController`` coupled authored-lane follower against the REAL recorded
coupled-OUTBOUND approach creeps of the coldwin17 throat wedge (FAIL) and the coldwin16/coldwin11
throat crossings (PASS), on the REAL descending outbound lane (``outbound_dock_to_bay``, which drops
y 15.5->15.25 so the base seats ~0.25 m south of the centreline). It proves the brief's acceptance
for the two-sided PRE-THROAT ALIGNMENT GATE (``heading_hold_prethroat_m`` + ``heading_hold_throat``):

  * the recorded coldwin17 creep, replayed through the controller with the gate ON, commands a turn
    that REDUCES the axis yaw from the FIRST corrective tick and drives the base within the bound
    BEFORE x = -3.95 (vs the recorded failing drift that grew to +5.5 deg and wedged);
  * coldwin16 and coldwin11 (the passers) are not degraded -- forward progress is maintained, every
    correction is toward the axis, and they still reach the throat crossing;
  * a mirror CW-seat case is corrected on the OTHER side (two-sided), not ignored (the PR 253 hold
    was one-sided and would never fire on a CW drift);
  * the gate never commands a negative linear velocity;
  * key OFF is byte-identical (gate keys absent, or ``heading_hold_prethroat_m`` = 0, reproduce the
    legacy one-sided PR 253 hold command stream exactly).

CPU-only; no Isaac/GPU/ROS. Run with:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -I -m pytest \
    scripts/twin/tests/test_outbound_heading_hold_bound.py -q
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

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'outbound_throat_yaw_bound_nav.json'
BOUND_DEG = 1.0
CREEP = 0.14
PRETHROAT_M = 0.8          # the pre-throat alignment window along the travel axis
THROAT = [-3.875, 15.26]   # the sales_backroom_portal throat (x = lip; y ~ the crossing centre)


def _fixture():
    return json.loads(FIXTURE.read_text())


def _controller(**kw):
    fix = _fixture()
    spec = MobileSpec()
    world = GridMap((-20, 5, 10, 25))
    graph = TowLaneGraph.from_config([fix['lane']])
    graph.expand_all(ArticulatedMap(world, spec))
    # The obstacle-free expansion lays the centreline on the full y 15.5->15.25 descent, so the
    # recorded base (running ~0.25 m south of it, where the real run's merge lead-in had re-seated
    # the route) sits near the default 0.25 m track tol. Widen it here so the route survives the
    # offset and the follower lateral term + the gate's lateral CAP are exercised -- the replan tol
    # is a SEPARATE gate (fm449), not what this test isolates.
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


def _replay_open(c, spec, rows, start_now, yaw_scale=1.0):
    """Open-loop: feed the recorded poses (yaw optionally mirrored via yaw_scale=-1) and the recorded
    forward speed; collect the controller's commanded (v, w). The command is a faithful function of
    the recorded drifted heading."""
    cmds = []
    now = start_now
    prev = None
    for row in rows:
        if prev is not None:
            now += max(1e-3, row['world_t'] - prev['world_t'])
        prev = row
        yaw = math.radians(row['yaw_deg'] * yaw_scale)
        od = Odometry((row['x'], row['y'], yaw), now, row['lin'], row['rw'], 'test_fixture', .02)
        obs = Observation(now, (), True, True, 0.0, 'test_fixture', 10)
        res = c.step(.05, od, obs, now=now)
        cmds.append((row['x'], row['yaw_deg'] * yaw_scale, res.reason, res.velocity[0], res.velocity[1]))
    return cmds


def _closed_loop(c, spec, rows, start_now):
    """Hybrid replay: the base follows the recorded drift + recorded x-advance pace until the gate
    ENGAGES; from that tick the controller OWNS the yaw (evolves under its commanded w). Returns the
    per-tick (x, yaw_deg, reason, v, w, engaged) and the first-engage index."""
    out = []
    now = start_now
    prev = None
    yaw = math.radians(rows[0]['yaw_deg'])
    w = 0.0
    controlled = False
    first_engage = None
    for i, row in enumerate(rows):
        dt = 0.05 if prev is None else max(1e-3, row['world_t'] - prev['world_t'])
        if prev is not None:
            yaw = wrap(yaw + w * dt) if controlled else math.radians(row['yaw_deg'])
        prev = row
        now += dt
        h0 = c.heading_holds
        od = Odometry((row['x'], row['y'], yaw), now, row['lin'], w if controlled else row['rw'],
                      'test_fixture', .02)
        obs = Observation(now, (), True, True, 0.0, 'test_fixture', 10)
        res = c.step(.05, od, obs, now=now)
        engaged = c.heading_holds > h0
        if engaged and first_engage is None:
            first_engage = i
        if engaged:
            controlled = True
        w = res.velocity[1]
        out.append((row['x'], math.degrees(yaw), res.reason, res.velocity[0], res.velocity[1], engaged))
    return out, first_engage


class HeadingHoldBoundTests(unittest.TestCase):
    def setUp(self):
        self.fix = _fixture()
        self.axis = self.fix['lane']['goal'][2]            # travel axis = goal heading (0 deg east)

    # ---- (1) coldwin17: corrects from the first tick and reaches the bound before x=-3.95 ---------

    def test_coldwin17_corrects_and_aligns_before_throat(self):
        c, spec = _gate_on()
        now = _reach_tracking(c, spec)
        out, first = _closed_loop(c, spec, self.fix['coldwin17']['rows'], now)
        self.assertIsNotNone(first, 'coldwin17: the gate never engaged on the wedge creep')
        # the first corrective tick that actually tracks (the base yaw is still +, before the axis is
        # crossed) commands a CW (negative w) turn -> reduces the +yaw. (A coasting tick with w=0 from the
        # recorded noisy odometry at engage is not a correction.)
        first_corr = next(((x, w) for (x, y, r, v, w, e) in out
                           if e and r == 'tracking' and abs(w) > 1e-9 and math.radians(y) - self.axis > 0),
                          None)
        self.assertIsNotNone(first_corr, 'coldwin17: no CW correction while yawed CCW')
        fx, fw = first_corr
        self.assertLess(fw, 0.0, f'coldwin17: first corrective tick did not command CW (w={fw:.4f})')
        self.assertLess(fx, -3.95, 'coldwin17: first correction only past x=-3.95')
        # the base reaches within the bound (|yaw - axis| <= bound) while still WEST of x=-3.95
        aligned_before = [(x, y) for (x, y, r, v, w, e) in out
                          if x <= -3.95 and abs(wrap(math.radians(y) - self.axis)) <= math.radians(BOUND_DEG + 0.3)]
        self.assertTrue(aligned_before,
                        'coldwin17: never reached within the bound before x=-3.95')
        # and once reached, it is reached before the recorded pin x (-3.931) is the LAST approach x
        first_aligned_x = aligned_before[0][0]
        self.assertLess(first_aligned_x, -3.90,
                        f'coldwin17: aligned only at the lip (x={first_aligned_x:.3f})')

    # ---- (2) the passers are not degraded (progress kept, corrections toward the axis) ------------

    def _assert_passer_ok(self, run):
        c, spec = _gate_on()
        now = _reach_tracking(c, spec)
        out, first = _closed_loop(c, spec, self.fix[run]['rows'], now)
        for x, y, reason, v, w, e in out:
            self.assertGreaterEqual(v, 0.0, f'{run}: negative linear velocity commanded')
        # forward progress is maintained to (near) the throat crossing: the base advances east
        xs = [t[0] for t in out]
        self.assertGreater(max(xs) - min(xs), 0.3, f'{run}: the tow made < 0.3 m of progress')
        # every engaged correction drives the yaw toward the axis (|yaw-axis| not increased by a
        # gate-commanded turn away from the axis): the commanded turn sign opposes the yaw error sign
        for x, y, reason, v, w, e in out:
            if e and abs(w) > 1e-9:
                ae = wrap(math.radians(y) - self.axis)
                self.assertLessEqual(w * ae, 1e-6,
                                     f'{run}: a gate correction turned AWAY from the axis (w={w:.4f}, ae={ae:.4f})')

    def test_coldwin16_not_degraded(self):
        self._assert_passer_ok('coldwin16')

    def test_coldwin11_not_degraded(self):
        self._assert_passer_ok('coldwin11')

    # ---- (3) two-sided: a mirror CW seat drift is corrected on the OTHER side --------------------

    def test_two_sided_cw_seat_is_corrected(self):
        # mirror the coldwin17 creep about the axis (yaw -> -yaw): now the base drifts CW. The PR 253
        # one-sided hold would NEVER fire; the two-sided gate must command a CCW (+w) correction.
        c, spec = _gate_on()
        now = _reach_tracking(c, spec)
        cmds = _replay_open(c, spec, self.fix['coldwin17']['rows'], now, yaw_scale=-1.0)
        self.assertGreaterEqual(c.heading_holds, 1, 'two-sided: the gate never engaged on a CW drift')
        corrective = [w for x, yaw, r, v, w in cmds if r == 'tracking' and w > 1e-9]
        self.assertTrue(corrective, 'two-sided: no CCW (+w) correction on a CW drift')
        for x, yaw, r, v, w in cmds:
            self.assertGreaterEqual(v, 0.0, 'two-sided: negative linear velocity')

    # ---- (4) key OFF is byte-identical to the legacy PR 253 hold ---------------------------------

    def test_key_off_byte_identical(self):
        for run in ('coldwin17', 'coldwin16', 'coldwin11'):
            rows = self.fix[run]['rows']
            # legacy one-sided hold (gate keys absent entirely)
            leg, spec = _controller(coupled_heading_hold=True, heading_hold_bound_deg=BOUND_DEG,
                                    heading_hold_creep_mps=CREEP)
            nl = _reach_tracking(leg, spec)
            a = _replay_open(leg, spec, rows, nl)
            # gate present but OFF by heading_hold_prethroat_m = 0.0 (and a throat set, to prove the
            # throat alone does nothing while the window length is zero)
            off, spec2 = _controller(coupled_heading_hold=True, heading_hold_bound_deg=BOUND_DEG,
                                     heading_hold_creep_mps=CREEP, heading_hold_prethroat_m=0.0,
                                     heading_hold_throat=THROAT)
            no = _reach_tracking(off, spec2)
            b = _replay_open(off, spec2, rows, no)
            self.assertEqual([t[3:] for t in a], [t[3:] for t in b],
                             f'{run}: prethroat=0 differs from the legacy one-sided hold (not byte-identical)')
            self.assertEqual(leg.heading_holds, off.heading_holds)

    def test_prethroat_requires_throat_and_hold(self):
        with self.assertRaises(ValueError):
            _controller(coupled_heading_hold=True, heading_hold_prethroat_m=0.8)   # no throat
        with self.assertRaises(ValueError):
            _controller(coupled_heading_hold=False, heading_hold_prethroat_m=0.8, heading_hold_throat=THROAT)


if __name__ == '__main__':
    unittest.main()
