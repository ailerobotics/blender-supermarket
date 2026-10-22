"""fm482 (sim-video-couple-speed-v1) CPU tests: the couple-standoff arrival hysteresis, the coupled
creep escape, their factory/config wiring and the mission.json evidence. Deterministic fixtures, never
evidence of a physics run (the recorded-run proof is scripts/check_couple_speed.py)."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation.core import Controller, Footprint, GridMap, Odometry, Observation
from twin.navigation.articulated import ArticulatedController
from twin.navigation.factory import nav_factory
from twin.restock.inventory import Job
from twin.restock.runtime import Packet, RuntimeAdapter

APPROACH = (-4.46, 16.0, 0.0)     # the mission's couple_approach -> couple_standoff leg
STANDOFF = (-4.06, 15.25, 0.0)
DT = 1 / 60


def _grid():
    return GridMap((-8.0, 12.0, 0.0, 19.0), resolution=0.2)


def _footprint():
    return Footprint(0.65, 0.62, 1.12, margin=0.05)


class _PivotPlant:
    """Diff base whose measured point sits `o` (body frame) off the turning pivot, so an in-place turn
    translates the measured pose (run-350/352/356 fit: ~8 mm)."""
    def __init__(self, pose, o):
        self.o = o
        c, s = math.cos(pose[2]), math.sin(pose[2])
        self.cx, self.cy, self.th = pose[0] - (c*o[0]-s*o[1]), pose[1] - (s*o[0]+c*o[1]), pose[2]
        self.v = self.w = 0.0

    def pose(self):
        c, s = math.cos(self.th), math.sin(self.th)
        return (self.cx + c*self.o[0] - s*self.o[1], self.cy + s*self.o[0] + c*self.o[1], self.th)

    def step(self, v, w):
        a = self.pose()
        self.th += w*DT
        self.cx += v*DT*math.cos(self.th)
        self.cy += v*DT*math.sin(self.th)
        b = self.pose()
        self.v, self.w = math.hypot(b[0]-a[0], b[1]-a[1])/DT, w


def _drive(hysteresis, *, o=(0.0, 0.008), cap_s=60.0):
    c = Controller(_grid(), _footprint(), allow_test_fixtures=True, goal_timeout=1e4,
                   arrival_xy_hysteresis=hysteresis)
    c.start(STANDOFF, False, arrival_xy_tol=0.05)
    plant = _PivotPlant(APPROACH, o)
    for tick in range(int(cap_s/DT)):
        now = (tick+1)*DT
        p = plant.pose()
        r = c.step(DT, Odometry(p, now, plant.v, plant.w, 'test_fixture'),
                   Observation(now, (), False, True, source='test_fixture', coverage_radius=10.0), now=now)
        if r.status == 'succeeded':
            return True, now, p
        plant.step(r.linear, r.angular)
    return False, cap_s, plant.pose()


class ArrivalHysteresisTests(unittest.TestCase):
    def test_legacy_arrival_dithers_on_the_tolerance_circle(self):
        # Reproduces the warm-run couple standoff: captured 0.6 mm inside the circle, the turn to the goal
        # yaw pushes the measured point back out and the drive branch turns it back -- no arrival.
        arrived, _, pose = _drive(0.0)
        self.assertFalse(arrived)
        self.assertAlmostEqual(math.dist(pose[:2], STANDOFF[:2]), 0.05, delta=0.003)

    def test_hysteresis_latches_capture_and_arrives_in_seconds(self):
        arrived, t, pose = _drive(0.03)
        self.assertTrue(arrived)
        self.assertLess(t, 25.0)                                    # ~13 s drive + ~3 s turn/settle
        self.assertLessEqual(abs(pose[2] - STANDOFF[2]), 0.08)
        self.assertLessEqual(math.dist(pose[:2], STANDOFF[:2]), 0.05 + 0.03)

    def test_latch_holds_inside_band_and_releases_beyond_it(self):
        c = Controller(_grid(), _footprint(), allow_test_fixtures=True, arrival_xy_hysteresis=0.03)
        c.start(STANDOFF, False, arrival_xy_tol=0.05)
        obs = lambda t: Observation(t, (), False, True, source='test_fixture', coverage_radius=10.0)
        def at(dx, t, yaw=-0.5):
            return c.step(DT, Odometry((STANDOFF[0]+dx, STANDOFF[1], yaw), t, 0.0, 0.0, 'test_fixture'),
                          obs(t), now=t)
        r = at(0.04, 0.1)                       # captured: turn in place toward the goal yaw
        self.assertEqual((r.linear, r.reason), (0.0, 'tracking')); self.assertGreater(r.angular, 0)
        r = at(0.065, 0.2)                      # drifted outside tol but inside tol+h: still latched
        self.assertTrue(c._xy_latched); self.assertGreater(r.angular, 0); self.assertEqual(r.linear, 0.0)
        at(0.10, 0.3)                           # beyond tol+h: released, back to the drive branch
        self.assertFalse(c._xy_latched)

    def test_default_is_the_legacy_per_tick_test(self):
        c = Controller(_grid(), _footprint(), allow_test_fixtures=True)
        self.assertEqual(c.arrival_xy_hysteresis, 0.0)
        c.start(STANDOFF, False, arrival_xy_tol=0.05)
        r = c.step(DT, Odometry((STANDOFF[0]+0.06, STANDOFF[1], -0.5), 0.1, 0.0, 0.0, 'test_fixture'),
                   Observation(0.1, (), False, True, source='test_fixture', coverage_radius=10.0), now=0.1)
        self.assertFalse(c._xy_latched)
        self.assertLess(r.angular, 0)           # drive branch: turns to face the goal POINT (behind-left)

    def test_restart_clears_the_latch(self):
        c = Controller(_grid(), _footprint(), allow_test_fixtures=True, arrival_xy_hysteresis=0.03)
        c.start(STANDOFF, False); c._xy_latched = True
        c.start(APPROACH, False)
        self.assertFalse(c._xy_latched)

    def test_invalid_hysteresis_refused(self):
        for bad in (-0.01, 0.6, float('nan')):
            with self.assertRaises(ValueError):
                Controller(_grid(), _footprint(), allow_test_fixtures=True, arrival_xy_hysteresis=bad)


class CreepEscapeTests(unittest.TestCase):
    GOAL = (-2.0, 15.25, 0.0)
    POSE = (-3.935, 15.265, math.radians(0.8))   # run-355 stall state

    def _gate(self, escape, odom_lin, odom_ang, *, cmd=(0.23, 0.0), obstacles=()):
        c = ArticulatedController(_grid(), allow_test_fixtures=True, coupled_creep_escape=escape)
        c.start(self.GOAL, True)
        r = c.gate_velocity(1.0, Odometry(self.POSE, 1.0, odom_lin, odom_ang, 'test_fixture'),
                            Observation(1.0, obstacles, True, True, 0.0, 'test_fixture', 10.0), *cmd)
        return r, c

    def test_noise_band_ratio_refuses_without_escape(self):
        r, _ = self._gate(False, 0.03, 0.05)                # measured k = 1.67 at a crawl
        self.assertEqual(r.reason, 'measured_turn_radius')

    def test_escape_keeps_creep_with_clamped_arc_in_noise_band(self):
        r, c = self._gate(True, 0.03, 0.05, cmd=(0.23, 0.15))
        self.assertEqual(r.reason, 'tracking')
        self.assertGreaterEqual(r.linear, 0.03)
        self.assertLessEqual(abs(r.angular) * 1.5, r.linear + 1e-9)
        self.assertEqual(c.creep_escapes, 1)

    def test_escape_raises_a_sub_floor_command_to_the_creep_floor(self):
        r, _ = self._gate(True, 0.03, 0.05, cmd=(0.01, 0.0))
        self.assertEqual(r.reason, 'tracking'); self.assertAlmostEqual(r.linear, 0.03)

    def test_genuine_arc_at_tow_speed_still_refuses(self):
        for escape in (False, True):
            r, _ = self._gate(escape, 0.20, 0.30)           # k = 1.5 above the band
            self.assertEqual(r.reason, 'measured_turn_radius')

    def test_obstacle_ahead_still_stops_with_escape(self):
        r, _ = self._gate(True, 0.03, 0.05, obstacles=((self.POSE[0] + 0.60, self.POSE[1]),))
        self.assertEqual(r.reason, 'obstacle_stop')

    def test_stop_command_is_never_turned_into_motion(self):
        r, c = self._gate(True, 0.03, 0.05, cmd=(0.0, 0.0))
        self.assertEqual(r.linear, 0.0); self.assertEqual(c.creep_escapes, 0)

    def test_default_off_and_floor_validated(self):
        c = ArticulatedController(_grid(), allow_test_fixtures=True)
        self.assertFalse(c.coupled_creep_escape)
        for bad in (0.0, -0.01, 0.2, float('nan')):
            with self.assertRaises(ValueError):
                ArticulatedController(_grid(), allow_test_fixtures=True, coupled_creep_floor=bad)


MANIFEST = {'map_bounds': [-6., 6., -4., 4.], 'static_obstacles': []}


def _factory(**nav):
    nav = dict({'environment': MANIFEST}, **nav)
    return nav_factory(app=None, assembly=object(), plan=NS(nav_mode='reference'),
                       config={'navigation': nav}, output=tempfile.gettempdir())


class FactoryWiringTests(unittest.TestCase):
    def test_keys_reach_the_controller(self):
        c = _factory(arrival_xy_hysteresis_m=0.03, coupled_creep_escape=True, coupled_creep_floor_mps=0.04)
        self.assertEqual(c.arrival_xy_hysteresis, 0.03)
        self.assertTrue(c.coupled_creep_escape); self.assertEqual(c.coupled_creep_floor, 0.04)

    def test_absent_keys_are_off(self):
        c = _factory()
        self.assertEqual(c.arrival_xy_hysteresis, 0.0); self.assertFalse(c.coupled_creep_escape)

    def test_non_boolean_or_non_numeric_refused(self):
        with self.assertRaises(ValueError):
            _factory(coupled_creep_escape='true')
        with self.assertRaises(ValueError):
            _factory(arrival_xy_hysteresis_m=True)


class RuntimeStandoffTests(unittest.TestCase):
    def setUp(self):
        self.p = Packet('f1', 'test_fixture', .1, None, None, None, None, True, False, 0., True)
        self.port = NS(advance=lambda dt: self.p, controls=lambda c: None, stop=lambda: None)
        self.manip = NS(start=lambda a, j: None, cart_location='rollwagen',
                        step=lambda dt, s: NS(status='running', reason='', controls=()))
        self.mobile = NS(park=lambda: None, hold=lambda: None, release=lambda: None,
                         command=lambda *a, **k: None, hold_cart_pin=lambda: None,
                         attach=lambda obs: None, coupling_state=lambda o: {'healthy': True})
        self.job = Job('j', 'milk', 1, 'back', 'shelf')

    def _adapter(self, *, controller=None, **kw):
        starts = []
        nav = NS(start=lambda goal, coupled, **k: starts.append((tuple(goal), k)),
                 step=lambda dt, p: NS(status='succeeded', linear=0., angular=0., reason=''),
                 cancel=lambda: None)
        if controller is not None:
            nav.controller = controller
        poses = {k: [0, 0, 0] for k in ('home', 'pickup', 'shelf_approach')}
        poses.update(cart_dock=[-4.46, 15.25, 0.0], couple_approach=list(APPROACH),
                     couple_standoff=list(STANDOFF))
        return RuntimeAdapter(nav, self.mobile, self.manip, self.port, poses, fixture_mode=True, **kw), starts

    def test_standoff_tolerance_override_reaches_the_standoff_leg(self):
        adapter, starts = self._adapter(couple_standoff_tolerance={'xy_m': 0.05, 'yaw_rad': 0.12})
        adapter.start('couple', self.job)
        self.assertEqual(starts[-1][1], {})                          # approach leg unchanged
        adapter.step(.1)
        self.assertEqual(starts[-1], (STANDOFF, {'arrival_xy_tol': 0.05, 'arrival_yaw_tol': 0.12}))

    def test_default_standoff_leg_is_byte_identical(self):
        adapter, starts = self._adapter()
        adapter.start('couple', self.job); adapter.step(.1)
        self.assertEqual(starts[-1], (STANDOFF, {'arrival_xy_tol': 0.05}))

    def test_invalid_standoff_tolerance_refused(self):
        for bad in ({'xy': 0.05}, {'xy_m': 0.3}, {'yaw_rad': 0.0}, {'yaw_rad': True}):
            with self.assertRaises(ValueError):
                self._adapter(couple_standoff_tolerance=bad)

    def test_couple_evidence_names_the_switches_only_when_on(self):
        on, _ = self._adapter(controller=NS(arrival_xy_hysteresis=0.03))
        off, _ = self._adapter()
        for adapter, expect in ((on, 'standoff xy-hysteresis 0.03 m'), (off, None)):
            self.p = replace(self.p, observation_id='a', simulation_time=.1, coupled=False)
            adapter.time = 0.0; adapter.last_packet_id = None
            adapter.start('couple', self.job)
            r = None
            for i in range(6):
                self.p = replace(self.p, observation_id=f'x{i}{id(adapter)}', simulation_time=adapter.time + .1,
                                 coupled=i >= 2)
                r = adapter.step(.1)
                if r.status == 'succeeded':
                    break
            self.assertEqual(r.status, 'succeeded')
            self.assertEqual(r.evidence.get('couple_arrival_mode'), expect)
            if expect is None:
                self.assertEqual(set(r.evidence), {'observation_id', 'source', 'simulation_time', 'coupled'})

    def test_outbound_evidence_records_creep_escape(self):
        adapter, _ = self._adapter(controller=NS(arrival_xy_hysteresis=0.0, coupled_creep_escape=True,
                                                 creep_escapes=7))
        self.p = replace(self.p, coupled=True)
        adapter.start('outbound', self.job)
        r = adapter.step(.1)
        self.assertEqual(r.status, 'succeeded')
        self.assertEqual(r.evidence['outbound_brake_mode'], 'coupled creep-escape (7 noise-band ticks)')

    def test_standoff_tolerance_named_in_couple_mode(self):
        adapter, _ = self._adapter(controller=NS(arrival_xy_hysteresis=0.03),
                                   couple_standoff_tolerance={'xy_m': 0.05, 'yaw_rad': 0.12})
        self.assertEqual(adapter._nav_workarounds('couple'),
                         {'couple_arrival_mode': 'standoff xy-hysteresis 0.03 m; standoff tol xy 0.05 m / yaw 0.12 rad'})

    def test_walkthrough_cut_puts_the_modes_on_screen(self):
        # The demo cut's own label extractor must surface both modes as caption labels of their stages.
        from twin.demo import run_evidence as E
        events = [{'state': 'couple_succeeded', 'evidence': {'source': 'sensor',
                                                            'couple_arrival_mode': 'standoff xy-hysteresis 0.03 m'}},
                  {'state': 'outbound_succeeded', 'evidence': {'source': 'sensor',
                                                              'outbound_brake_mode': 'coupled creep-escape (7 noise-band ticks)'}}]
        run = NS(mission={'state': 'failed', 'events': events}, events=events, caption=None)
        labels = {(l.stage, l.key, l.value) for l in E.workaround_labels(run)}
        self.assertIn(('couple', 'couple_arrival_mode', 'standoff xy-hysteresis 0.03 m'), labels)
        self.assertIn(('outbound', 'outbound_brake_mode', 'coupled creep-escape (7 noise-band ticks)'), labels)


if __name__ == '__main__':
    unittest.main()
