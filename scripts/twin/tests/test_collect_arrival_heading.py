"""sim-collect-arrival-heading-v1 CPU proofs for the coldwin29 collect stall.

The cold COLLECT of coldwin29 drove to the place_approach standoff and then sat there for ~160 s until
the 300 s collect stage timeout fired; it never entered PREPLACE. The saved run
(coldwin29-fail1-saved) shows the base pinned near (-4.47, 15.56) with its heading frozen at ~-1.71 rad
while the nav goal heading was pi -- a 1.43 rad error that never nulled. The place_approach pose was
authored for the PRE-304 EAST-facing cart (its standoff faces the cart mouth at heading pi); PR 304
rotated the dock to the SOUTH-facing cart, whose mouth opens -y, so the standoff must face +pi/2. The
stale pi heading is ~1.43 rad from the heading the base reaches by driving south into the standoff
(~-pi/2), and the arrival test needs xy AND yaw within tol at once: the in-place rotation drifts the
differential base off the xy circle, the drive branch pulls it back, and the heading limit-cycles.

These tests (a) REPLAY the recorded final-leg poses through the real Controller arrival logic and show the
stall reproduces -- it never latches arrived_stopped, and the new ARRIVAL_HEADING_STALL guard NAMES it in
sim-time well before the 300 s stage budget (legacy: silent to goal_timeout); and (b) show the re-authored
spin-free lead-in makes the final heading come from the drive, so a kinematic drive to the corrected
standoff arrives_stopped. Deterministic fixtures, never evidence of a physics run.
"""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.navigation.core import Controller, Footprint, GridMap, Odometry, Observation
from twin.restock.isaac_bundle import derive_place_approach, derive_place_approach_stage

DT = 1 / 60
FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'coldwin29_collect_finalleg.json'
# The run's actual final-leg nav goal (place_approach) and tolerances (reposition leg: xy 0.15, yaw 0.20;
# the global arrival XY hysteresis was on). The stale pi heading is the bug.
STALE_GOAL = (-4.5, 15.4, math.pi)
XY_TOL, YAW_TOL, HYST = 0.15, 0.20, 0.03


def _grid():
    return GridMap((-8.0, 12.0, 0.0, 19.0), resolution=0.2)


def _footprint():
    return Footprint(0.65, 0.62, 1.12, margin=0.05)


def _controller(goal, *, stall_s, static_drive=True):
    c = Controller(_grid(), _footprint(), allow_test_fixtures=True, goal_timeout=1e4,
                   arrival_xy_hysteresis=HYST, arrival_heading_stall_s=stall_s)
    c.start(goal, False, arrival_xy_tol=XY_TOL, arrival_yaw_tol=YAW_TOL, static_drive=static_drive)
    return c


def _replay(goal, stall_s):
    """Feed the recorded coldwin29 final-leg poses through the real arrival logic. Returns
    (arrived, stall_reason_or_None, stall_t_or_None)."""
    ticks = json.loads(FIXTURE.read_text())
    c = _controller(goal, stall_s=stall_s)
    arrived = False
    for t, x, y, yaw, lin, ang in ticks:
        obs = Observation(t, (), False, True, source='test_fixture', coverage_radius=10.0)
        r = c.step(DT, Odometry((x, y, yaw), t, lin, ang, 'test_fixture'), obs, now=t)
        if r.status == 'succeeded' and r.reason == 'arrived_stopped':
            arrived = True
            break
        if r.status == 'failed':
            return arrived, r.reason, t
    return arrived, None, None


class ReplayReproducesStall(unittest.TestCase):
    def test_stale_pi_heading_never_latches_arrival(self):
        # On the REAL recorded poses the arrival never latches arrived_stopped (legacy stall_s=0: the
        # controller just returns 'tracking' forever, which is why collect ran out its 300 s stage budget).
        arrived, reason, _ = _replay(STALE_GOAL, stall_s=0.0)
        self.assertFalse(arrived, 'the stale pi heading must NOT latch arrival on the recorded poses')
        self.assertIsNone(reason, 'legacy (stall_s=0) stays silently tracking -- no named failure')

    def test_arrival_heading_stall_names_it_in_sim_time(self):
        # With the guard on (default 20 s sim), the same recorded stall NAMES itself -- a terminal
        # 'arrival_heading_stall' fires well before the 300 s stage timeout (the leg starts at sim ~119).
        arrived, reason, t = _replay(STALE_GOAL, stall_s=20.0)
        self.assertFalse(arrived)
        self.assertEqual(reason, 'arrival_heading_stall')
        self.assertLess(t, 160.0, 'must name the stall long before the 300 s collect stage timeout')
        self.assertGreater(t, 119.0 + 20.0 - 1.0, 'must not fire before the no-progress budget elapses')

    def test_guard_does_not_fire_when_heading_is_already_aligned(self):
        # Control: feed the SAME recorded xy track but with a goal heading equal to the base's actual
        # heading (so yaw_err ~ 0). The guard must NEVER fire and the near-goal aligned arrival branch
        # (settling_at_goal / arrived_stopped) must be reached -- proof the guard keys on the HEADING
        # error, not merely on sitting near the goal. (The recorded base never fully stops -- it jitters
        # above the stopped thresholds -- so it settles rather than latching; that is not the heading
        # stall, which is exactly the point.)
        ticks = json.loads(FIXTURE.read_text())
        # the base holds ~-1.705 rad through the stall; use that as the goal heading
        goal = (-4.473, 15.556, -1.705)
        c = _controller(goal, stall_s=20.0)
        reached_arrival_branch = False
        for t, x, y, yaw, lin, ang in ticks:
            obs = Observation(t, (), False, True, source='test_fixture', coverage_radius=10.0)
            r = c.step(DT, Odometry((x, y, yaw), t, lin, ang, 'test_fixture'), obs, now=t)
            self.assertNotEqual(r.reason, 'arrival_heading_stall',
                                'the heading guard must not fire when the heading is aligned')
            if r.reason in ('settling_at_goal', 'arrived_stopped'):
                reached_arrival_branch = True
        self.assertTrue(reached_arrival_branch,
                        'an aligned heading must reach the near-goal arrival branch, never the guard')


class LeadInGeometryIsSpinFree(unittest.TestCase):
    def test_south_dock_lead_in_final_heading_equals_goal(self):
        # The re-authored derive_place_approach_stage makes the FINAL leg a straight drive whose direction
        # equals place_approach[2], so the base arrives aligned (no spin). For the PR 304 south dock the
        # standoff faces +pi/2 and the lead-in is south of it.
        pickup = (-5.73, 16.0, 0.0)
        deck = (-4.6, 15.92)
        pa = derive_place_approach(deck, -math.pi / 2, 1.0)
        stage = derive_place_approach_stage(pa, pickup, lead_in=0.6)
        drive_in = math.atan2(pa[1] - stage[1], pa[0] - stage[0])
        self.assertAlmostEqual(abs(((pa[2] - drive_in + math.pi) % (2 * math.pi)) - math.pi), 0.0, places=6)

    def test_stale_pi_vs_corrected_heading_error(self):
        # The stale config heading (pi) is ~1.43 rad from the drive-in heading of the run's actual final
        # leg (stage (-4.5,16,0) -> place_approach (-4.5,15.4): a straight drive south, -pi/2). The
        # corrected heading (= the drive-in) nulls that error.
        stage_xy, pa_xy = (-4.5, 16.0), (-4.5, 15.4)
        drive_in = math.atan2(pa_xy[1] - stage_xy[1], pa_xy[0] - stage_xy[0])
        self.assertAlmostEqual(drive_in, -math.pi / 2, places=6)
        stale_err = abs(((math.pi - drive_in + math.pi) % (2 * math.pi)) - math.pi)
        self.assertGreater(stale_err, 1.4)  # the ~1.43 rad coldwin29 stall
        self.assertLess(abs(((drive_in - drive_in + math.pi) % (2 * math.pi)) - math.pi), 1e-9)


class KinematicDriveConverges(unittest.TestCase):
    """A unicycle integration of the Controller's own commands from the staged start. The corrected goal
    (heading == final-leg drive-in) arrives_stopped; it proves the fix closes the loop, not just the
    classifier. Honest limits: no payload dynamics, no gate obstacles (empty static leg)."""

    def _drive(self, start, goal, max_s=60.0):
        c = _controller(goal, stall_s=0.0, static_drive=True)  # guard off here: test convergence, not naming
        x, y, yaw = start
        t = 0.0
        while t < max_s:
            obs = Observation(t, (), False, True, source='test_fixture', coverage_radius=10.0)
            r = c.step(DT, Odometry((x, y, yaw), t, 0.0, 0.0, 'test_fixture'), obs, now=t)
            if r.status == 'succeeded':
                return True, t, (x, y, yaw)
            if r.status == 'failed':
                return False, t, (x, y, yaw)
            v = r.linear if r.linear is not None else 0.0
            w = r.angular if r.angular is not None else 0.0
            yaw = math.atan2(math.sin(yaw + w * DT), math.cos(yaw + w * DT))
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            t += DT
        return False, t, (x, y, yaw)

    def test_corrected_drive_in_heading_arrives(self):
        # Staged start = the run's staging waypoint (-4.5, 16, 0); corrected place_approach heading = the
        # final-leg drive-in (-pi/2). The base drives straight south and arrives aligned.
        arrived, t, pose = self._drive((-4.5, 16.0, 0.0), (-4.5, 15.4, -math.pi / 2))
        self.assertTrue(arrived, f'corrected heading must arrive; stopped at {pose} t={t:.1f}')
        self.assertLess(t, 30.0)


class ArrivalHeadingCheckCLI(unittest.TestCase):
    """Guard scripts/check_collect_arrival_heading.py (the offline gate the integration lane runs): the
    stale coldwin29 pose+stage must FAIL (named spin), the derived lead-in must PASS. Committed-files only."""

    REPO = Path(__file__).resolve().parents[3]
    CHECK = REPO / 'scripts' / 'check_collect_arrival_heading.py'
    SCEN = REPO / 'scripts' / 'restock_scenario.json'
    LANES = REPO / 'docs' / 'examples' / 'cart-side-bay-mission-fragment.json'

    def _run(self, *extra):
        import subprocess
        return subprocess.run(
            [sys.executable, '-I', str(self.CHECK), '--scenario', str(self.SCEN),
             '--lanes', str(self.LANES), *extra],
            capture_output=True, text=True, cwd=str(self.REPO))

    def test_clean_derivation_passes(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('COLLECT_ARRIVAL_HEADING PASS', r.stdout)
        self.assertIn('"place_approach": [-4.6, 14.92, 1.5708]', r.stdout)

    def test_stale_coldwin29_pose_fails_named(self):
        r = self._run('--authored-place-approach=-4.5,15.4,3.14159265', '--authored-stage=-4.5,16.0,0.0')
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('FAIL authored_place_approach_heading_is_spin_free', r.stdout)
        self.assertIn('"spin_deg": 90.0', r.stdout)


if __name__ == '__main__':
    unittest.main()
