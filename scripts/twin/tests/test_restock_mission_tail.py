"""Guard the mission TAIL (transfer succeeded -> verify -> uncouple-at-parking return -> mission end) for the
backroom-bay demo with the CPU dry-run (``scripts/check_mission_tail.py``, sim-video-verify-return-v1 / vr1).

The fixture ``fixtures/mission_tail_bay.json`` is the tail-relevant excerpt of the run-356 runtime config +
the fm459 bay scenario + the backroom static map, unchanged; the tests add the vr1 switches on top. The BT
dry-runs need py_trees (skipped without it, like the other BT tests); the geometry/gate tests do not.

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts:<bt-pythonpath> python3 -m pytest \
        scripts/twin/tests/test_restock_mission_tail.py -q
"""
import copy
import io
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import check_mission_tail as tail  # noqa: E402
from twin.restock.ground_truth_verify import (GroundTruthInventoryVerifier,  # noqa: E402
                                              GroundTruthVerifyError, ground_truth_place_actions,
                                              mission_ground_truth_sources, mission_modes)
from twin.restock.inventory import Inventory, Job  # noqa: E402
from twin.restock.mission import Mission, Result  # noqa: E402

FIXTURE = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'mission_tail_bay.json').read_text())
HAVE_PT = tail.have_py_trees()


def _config(**extra):
    config = copy.deepcopy(FIXTURE['config'])
    config.update(extra)
    return config


def _rows(rep, prefix):
    return {name: (verdict, detail) for verdict, name, detail in rep.rows if name.startswith(prefix)}


class GroundTruthVerifierGeometry(unittest.TestCase):
    """The mission-level ground-truth verify against the bay top (0.7 x 0.5 at z 0.73) and the placed
    tray (0.6 x 0.4 x 0.16): rests-on-bay accepted, off-bay refused fast and named, two frames."""

    def setUp(self):
        self.scenario = FIXTURE['scenario']
        self.unit = [-2.0, 16.42, 0.73 + 0.08, 0.0]
        self.contact = [False]
        self.v = tail.gt_verifier_from_config(_config(verify_ground_truth={'enabled': True}), self.scenario,
                                              lambda: tuple(self.unit), read_contact=lambda: self.contact[0])
        self.job = Job(**self.scenario['job'])

    def test_disabled_without_the_switch(self):
        self.assertIsNone(tail.gt_verifier_from_config(_config(), self.scenario, lambda: None))

    def test_two_frames_then_labelled_evidence(self):
        self.v.begin(self.job, 10.0)
        self.assertIsNone(self.v(10.0))                  # not after the boundary
        self.assertIsNone(self.v(10.1))                  # first confirmation
        ev = self.v(10.2)
        self.assertEqual((ev['source'], ev['verify_mode']), ('ground_truth', 'ground_truth'))
        self.assertEqual((ev['sku'], ev['quantity'], ev['location']), ('juice_orange', 1, 'G3_back_s2'))
        self.assertIs(ev['transfer_confirmed'], True)
        self.assertEqual(len(ev['confirming_observation_ids']), 2)
        self.assertIn('NOT a camera measurement', ev['note'])
        self.assertFalse(self.v.active)

    def test_tray_off_the_bay_fails_fast_named(self):
        self.unit[2] = 0.08                              # on the floor
        self.v.begin(self.job, 0.0)
        self.assertIsNone(self.v(0.1))
        self.assertIsNone(self.v(0.2))
        with self.assertRaisesRegex(GroundTruthVerifyError, 'not resting on the bay'):
            self.v(0.3)

    def test_still_held_is_refused(self):
        self.contact[0] = True
        self.v.begin(self.job, 0.0)
        with self.assertRaisesRegex(GroundTruthVerifyError, 'still held'):
            self.v(0.1)

    def test_settling_band_waits(self):
        self.unit[2] += 0.05                             # 5 cm above the top: not on, not clearly off
        self.v.begin(self.job, 0.0)
        for k in range(1, 6):
            self.assertIsNone(self.v(0.1 * k))
        self.assertTrue(self.v.active)

    def test_tray_on_bay_cases(self):
        rep = tail.Report(io.StringIO())
        tail.check_tray_on_bay(_config(verify_ground_truth={'enabled': True}), self.scenario, rep)
        failed = [name for verdict, name, _ in rep.rows if verdict == 'FAIL']
        self.assertEqual(failed, [])


class GroundTruthSwitches(unittest.TestCase):
    def test_default_config_is_byte_identical(self):
        # sim-robust-vram-headroom-default-on-v1: the VRAM headroom monitor is default-on, so its
        # provenance is the ONE expected default mode; every ground-truth/return switch stays absent.
        modes = mission_modes({})
        self.assertEqual(modes.get('vram_headroom_mode'), 'on')
        self.assertEqual({k for k in modes if k != 'vram_headroom_mode' and k != 'vram_headroom'}, set())
        # opting the monitor out restores the fully-empty (byte-identical) modes dict.
        self.assertEqual(mission_modes({'vram_headroom': False}), {})
        self.assertEqual(ground_truth_place_actions({}), ())
        self.assertEqual(mission_ground_truth_sources({}), ())

    def test_run_config_modes(self):
        # opt the default-on VRAM monitor out here so the assertion isolates the ground-truth/return
        # modes (the monitor's own provenance is covered by test_restock_vram_headroom.py).
        config = _config(verify_ground_truth={'enabled': True}, vram_headroom=False)
        self.assertEqual(mission_modes(config), {'verify_mode': 'ground_truth',
                                                 'transfer_verify_mode': 'ground_truth',
                                                 'return_mode': 'uncouple_at_cart_parking'})
        self.assertEqual(ground_truth_place_actions(config), ('transfer',))
        self.assertEqual(mission_ground_truth_sources(config), ('ground_truth',))
        # the lidar yaw-0 fix is ON by default now (vr1), so a fix run records NO lidar provenance (it is
        # the norm); only the EXCEPTIONAL legacy wiring is recorded as provenance, never a captioned *_mode.
        self.assertNotIn('lidar_sensor_to_base', mission_modes(dict(config, lidar_sensor_yaw_fix=True)))
        self.assertNotIn('lidar_sensor_to_base', mission_modes(config))         # absent => default fix
        legacy = mission_modes(dict(config, lidar_sensor_yaw_fix=False))['lidar_sensor_to_base']
        self.assertIn('legacy', legacy)
        self.assertFalse(any(k.endswith('_mode') and 'lidar' in k for k in
                             mission_modes(dict(config, lidar_sensor_yaw_fix=False))))

    def test_mission_refuses_ground_truth_evidence_unless_opted_in(self):
        class Adapter:
            def start(self, action, job): pass
            def stop(self): pass
            def step(self, dt):
                return Result('succeeded', {'source': 'ground_truth', 'observation_id': 'gt:1',
                                            'simulation_time': 0.1, 'released': True})
        with tempfile.TemporaryDirectory() as tmp:
            for sources, ok in (((), False), (('ground_truth',), True)):
                db = Inventory(str(Path(tmp) / f'{ok}.sqlite'))
                db.seed('backstock_00', 'juice_orange', 1)
                m = Mission(db, Job(**FIXTURE['scenario']['job']), Adapter(), start_at='transfer',
                            ground_truth_sources=sources)
                m.start()
                out = m.step_stage(0.1)
                self.assertEqual(out == 'succeeded', ok, out)
                db.close()
        with self.assertRaises(ValueError):
            Mission(None, Job(**FIXTURE['scenario']['job']), Adapter(), ground_truth_sources=('sensor_x',))


class ConfigChecks(unittest.TestCase):
    def test_run356_config_needs_the_verify_switch(self):
        # The run-356 config as-is: everything in place except a way for VERIFY to confirm on the
        # uncommissioned bay -- the dry-run names the missing key.
        rep = tail.Report(io.StringIO())
        tail.check_config(_config(), FIXTURE['scenario'], rep)
        rows = _rows(rep, 'config_')
        self.assertEqual(rows['config_verify_stage_can_confirm'][0], 'FAIL')
        self.assertIn('verify_ground_truth', rows['config_verify_stage_can_confirm'][1]['fix'])
        others = [n for n, (v, _) in rows.items() if v == 'FAIL' and n != 'config_verify_stage_can_confirm']
        self.assertEqual(others, [])
        self.assertEqual(rows['config_return_lane_resolves'][1]['lane'], 'return_bay_to_parking')

    def test_with_the_verify_switch_config_passes(self):
        rep = tail.Report(io.StringIO())
        tail.check_config(_config(verify_ground_truth={'enabled': True}), FIXTURE['scenario'], rep)
        self.assertEqual([n for v, n, _ in rep.rows if v == 'FAIL'], [])

    def test_cart_side_bay_config_accepts_home_empty_return(self):
        # sim-cart-side-bay-v1: the side bay leaves the cart parked off-axis and returns the base HOME
        # EMPTY, so uncouple_on_return MUST be false (the legacy uncouple-at-parking check would false-fail
        # that). check_config must instead pass config_side_bay_return_home_empty and NOT emit the legacy row.
        poses = copy.deepcopy(FIXTURE['config']['mission_poses'])
        poses['cart_side_bay'] = [-4.3, 6.95, -1.5707963267948966]
        config = _config(verify_ground_truth={'enabled': True}, shelf_staging=True,
                         cart_side_bay=True, uncouple_on_return=False, mission_poses=poses)
        rep = tail.Report(io.StringIO())
        tail.check_config(config, FIXTURE['scenario'], rep)
        rows = _rows(rep, 'config_')
        self.assertEqual([n for v, n, _ in rep.rows if v == 'FAIL'], [])
        self.assertEqual(rows['config_side_bay_return_home_empty'][0], 'PASS')
        self.assertNotIn('config_uncouple_at_parking_enabled', rows)


@unittest.skipUnless(HAVE_PT, 'py_trees not installed (BT pythonpath)')
class TailDryRun(unittest.TestCase):
    """The real tree drives transfer -> verify -> return -> finalize to mission success, and the latent
    failure modes found by this dry-run stay fixed."""

    def _run(self, config, **inject):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mission, summary, rig = tail.run_tail(config, FIXTURE['scenario'], tmp.name, **inject)
        mj = json.loads((Path(tmp.name) / 'mission.json').read_text())
        lines = (Path(tmp.name) / 'bt_events.jsonl').read_text().splitlines()
        return mission, summary, rig, mj, lines

    def test_all_dryrun_checks_pass_with_the_vr1_switches(self):
        rep = tail.Report(io.StringIO())
        with tempfile.TemporaryDirectory() as tmp:
            tail.check_tail(_config(verify_ground_truth={'enabled': True}), FIXTURE['scenario'], rep, tmp)
            tail.check_default_off(_config(verify_ground_truth={'enabled': True}), FIXTURE['scenario'], rep, tmp)
        failed = [(n, d) for v, n, d in rep.rows if v == 'FAIL']
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len([1 for v, n, _ in rep.rows if v == 'PASS' and n.startswith('tail_')]), 12)

    def test_nominal_tail_events_and_mission_json(self):
        mission, summary, rig, mj, lines = self._run(_config(verify_ground_truth={'enabled': True}))
        self.assertEqual(mission.state, 'complete')
        self.assertEqual(mj['state'], 'complete')
        self.assertEqual((mj['verify_mode'], mj['transfer_verify_mode'], mj['return_mode']),
                         ('ground_truth', 'ground_truth', 'uncouple_at_cart_parking'))
        states = [e['state'] for e in summary['events']]
        for s in ('transfer_succeeded', 'inventory_verified', 'verify_succeeded', 'return_succeeded'):
            self.assertIn(s, states)
        self.assertEqual(states[-1], 'complete')
        # the uncouple happens in RETURN at cart_parking, AFTER the transfer + verify at the park pose
        self.assertLess(states.index('verify_succeeded'), states.index('return'))
        self.assertEqual(rig.mobile.calls, ['hold_cart_pin', 'uncouple'])
        # walkthrough timeline: every tail stage begins + succeeds with each step on the sim clock
        timeline = [json.loads(line) for line in lines]
        begins = [e['data']['stage'] for e in timeline if e['type'] == 'stage_begin']
        self.assertEqual(begins, ['transfer', 'verify', 'return'])
        ret_steps = [e['data']['label'] for e in timeline
                     if e['type'] == 'stage_step' and e['data']['stage'] == 'return']
        self.assertEqual(ret_steps, ['tow→cart park', 'brake cart', 'uncouple', 'back away', 'drive→home'])
        self.assertTrue(all('sim_time' in e['data'] for e in timeline if e['type'].startswith('stage_')))
        self.assertEqual(timeline[-1]['type'], 'mission_complete')
        self.assertEqual([e['type'] for e in mj['stage_timeline']], [e['type'] for e in timeline
                                                                     if e['type'] in tail_types()])

    def test_run356_config_as_is_times_out_in_verify(self):
        # Reproduces the predicted first live attempt without the vr1 verify switch.
        mission, _summary, _rig, _mj, _lines = self._run(_config())
        self.assertEqual(mission.state, 'failed')
        self.assertIn('verify timed out', mission.reason)

    def test_default_off_reproduces_transfer_evidence_mismatch(self):
        config = _config()
        config['transfer_ground_truth'] = dict(config['transfer_ground_truth'], enabled=False)
        mission, *_ = self._run(config)
        self.assertEqual(mission.state, 'failed')
        self.assertIn('transfer evidence mismatch', mission.reason)

    def test_never_settling_train_fails_named_not_forced(self):
        # The uncouple settle wait is bounded: a train that never reads stationary is refused with a
        # NAMED reason (never force-detached), and the mission ends instead of hanging.
        mission, _s, rig, _mj, lines = self._run(_config(verify_ground_truth={'enabled': True}),
                                                 unsettle_after_pin=10 ** 7)
        self.assertEqual(mission.state, 'failed')
        self.assertNotIn('uncouple', rig.mobile.calls)
        failed = [json.loads(line)['data']['reason'] for line in lines
                  if json.loads(line)['type'] == 'stage_failed']
        self.assertIn('uncouple refused: train not stationary/parked', failed[0])

    def test_backaway_waypoint_is_driven_uncoupled(self):
        poses = dict(FIXTURE['config']['mission_poses'], cart_parking_backaway=[-0.6, 13.8, math.pi])
        mission, _s, rig, _mj, _l = self._run(_config(verify_ground_truth={'enabled': True}, mission_poses=poses))
        self.assertEqual(mission.state, 'complete')
        self.assertEqual([(g['goal'], g['coupled']) for g in rig.nav.goals],
                         [([-1.2, 15.25, 0.0], True), ([-0.6, 13.8, math.pi], False),
                          ([-5.73, 16.0, 0.0], False)])


def tail_types():
    from twin.restock.bt.driver import STAGE_EVENT_TYPES
    return STAGE_EVENT_TYPES


class HomeLegNavigation(unittest.TestCase):
    """The lidar wiring proof and the uncoupled home leg with the REAL controller on the bay map."""

    def test_legacy_lidar_wiring_rotates_returns_and_the_fix_does_not(self):
        from twin.restock.tail_nav import LIDAR_XYZ, lidar_wall_probe
        self.assertEqual(lidar_wall_probe((LIDAR_XYZ[0], LIDAR_XYZ[1], 0.0)), (3.0, 3.0))
        lo, hi = lidar_wall_probe(LIDAR_XYZ)          # the legacy (x, y, z) tuple read as (x, y, yaw)
        self.assertGreater(hi - lo, 1.0)

    def test_lidar_sensor_frame_orientation_locked_and_fix_is_default(self):
        """REGRESSION LOCK (sim-restock-lidar-yaw-fix-eval): the lidar_frame is a FIXED, rpy-0 child of
        base_link -- its physical yaw about base_link is exactly 0, and the URDF z is a mount HEIGHT, not a
        rotation. So the range sensor's ``sensor_to_base`` YAW must always be 0 (never the z), and the fix
        is the DEFAULT. If anything regresses the frame to carry a rotation, or wires the z back in as yaw
        by default, this fails."""
        from twin.urdf_probe import parse_urdf
        from twin.restock.isaac_bundle import lidar_sensor_to_base
        from twin.mobile_assets import build_mobile_urdf
        # Generate the real mobile URDF from a minimal UR source chain and read the lidar_frame joint, so
        # the asset-level invariant (fixed, rpy-0) is checked on the actual generator output.
        source = '''<robot name="test_fixture">
          <link name="world"/><link name="pedestal_link"/><link name="pedestal_top"/>
          <link name="ur_base_link"/><link name="arm"/><link name="ur_tool0"/>
          <joint name="pedestal_joint" type="fixed"><parent link="world"/><child link="pedestal_link"/></joint>
          <joint name="pedestal_top_joint" type="fixed"><parent link="pedestal_link"/><child link="pedestal_top"/></joint>
          <joint name="ur_base_joint" type="fixed"><parent link="pedestal_top"/><child link="ur_base_link"/></joint>
          <joint name="shoulder" type="revolute"><parent link="ur_base_link"/><child link="arm"/>
          <limit lower="-3" upper="3" effort="100" velocity="1"/></joint>
          <joint name="tool0_joint" type="fixed"><parent link="arm"/><child link="ur_tool0"/>
          <origin xyz="0 0 0.3" rpy="0 0 0"/></joint>
          <ros2_control name="fixture"><joint name="shoulder"><state_interface name="position">
          <param name="initial_value">-2.5</param></state_interface></joint></ros2_control>
          </robot>'''
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'arm.urdf'
            src.write_text(source)
            out = Path(tmp) / 'mobile.urdf'
            build_mobile_urdf(str(src), str(out))
            joint = parse_urdf(str(out)).joints_by_child['lidar_frame']
            self.assertEqual(joint.jtype, 'fixed')
            self.assertEqual(tuple(round(v, 6) for v in joint.origin_rpy), (0.0, 0.0, 0.0),
                             'lidar_frame must be physically UNROTATED; its z is a mount height, not a yaw')
            self.assertNotEqual(joint.origin_xyz[2], 0.0,
                                'the frame carries a nonzero z (the value the legacy bug read as yaw)')
            origin = joint.origin_xyz
        # The production resolver: absent/true => yaw 0 (the fix, default); explicit false => legacy z-as-yaw.
        self.assertEqual(lidar_sensor_to_base(origin, {}), (origin[0], origin[1], 0.0))
        self.assertEqual(lidar_sensor_to_base(origin, {'lidar_sensor_yaw_fix': True}),
                         (origin[0], origin[1], 0.0))
        self.assertEqual(lidar_sensor_to_base(origin, {'lidar_sensor_yaw_fix': False}),
                         (origin[0], origin[1], origin[2]))

    def test_recommended_return_poses_reach_home_uncoupled(self):
        # The vr1 return poses (docs/examples/mission-tail-fragment.json): from the cart parking spot,
        # back away south-east clear of the parked cart, then drive west to the spec home. Legacy lidar
        # wiring (how the run is wired today); ~20 s CPU. Without the back-away waypoint the braking
        # capsule clips the parked cart on the turn-in-place and the leg ends obstacle_stop_timeout.
        from twin.restock.tail_nav import simulate_home_leg
        fragment = json.loads((Path(__file__).resolve().parents[3] / 'docs' / 'examples'
                               / 'mission-tail-fragment.json').read_text())
        poses = fragment['mission_poses']
        r = simulate_home_leg(FIXTURE['config'], FIXTURE['scenario'], FIXTURE['environment'],
                              camera_config=FIXTURE['camera_config'], home=poses['home'],
                              backaway=poses['cart_parking_backaway'], lidar_yaw_fix=False)
        self.assertEqual((r['status'], r['reason'], r['leg']), ('succeeded', 'arrived_stopped', 1), r)
        self.assertLess(math.dist(r['pose'][:2], poses['home'][:2]), 0.15)
        self.assertLess(r['t'], 120.0)                # well inside the return stage budget


class CartBoxesYaw(unittest.TestCase):
    """``tail_nav.cart_boxes`` at an arbitrary parked yaw (sim-cart-boxes-yaw-v1). PR 271 re-heads the
    rollwagen parking spot SOUTH on the portal axis (``cart_parking`` yaw = +pi/2), so the helper must no
    longer refuse a non-axis-aligned pose; it stays byte-identical inside the old near-axis band."""

    CX, CY = -3.875, 16.3

    @staticmethod
    def _true_footprint_corners(cx, cy, cyaw):
        """The physically-correct parked cart: the canonical part boxes rigidly ROTATED about the centre.
        Every corner must fall inside the union of the returned (conservative, axis-aligned) boxes."""
        from twin.restock.tail_nav import _cart_local_boxes
        c, s = math.cos(cyaw), math.sin(cyaw)
        pts = []
        for lx0, ly0, _z0, lx1, ly1, _z1, _path in _cart_local_boxes():
            for lx in (lx0, lx1):
                for ly in (ly0, ly1):
                    pts.append((cx + c * lx - s * ly, cy + s * lx + c * ly))
        return pts

    @staticmethod
    def _union_contains(boxes, pts, eps=1e-9):
        return all(any(b[0] - eps <= px <= b[3] + eps and b[1] - eps <= py <= b[4] + eps for b in boxes)
                   for px, py in pts)

    def test_byte_identical_inside_the_near_axis_band(self):
        """Within the old tolerance (|sin(cyaw)| <= 0.2, yaw ~ 0 or ~ pi) the result is exactly the historical
        axis-aligned approximation -- every old near-axis call is unchanged to the bit."""
        from twin.restock.tail_nav import cart_boxes, _cart_boxes_axis_aligned
        for yaw in (0.0, 0.1, -0.1, 0.2, -0.2, math.pi, math.pi - 0.1, math.pi + 0.1):
            self.assertLessEqual(abs(math.sin(yaw)), 0.2 + 1e-12, yaw)
            self.assertEqual(cart_boxes(self.CX, self.CY, yaw),
                             _cart_boxes_axis_aligned(self.CX, self.CY, yaw), f'yaw={yaw}')

    def test_south_headed_pose_no_longer_refuses(self):
        """The PR 271 south-headed parking pose (yaw = +pi/2) raised ``cart_boxes approximates a
        near-axis-aligned parked cart only`` before this change; it now returns the full box set."""
        from twin.restock.tail_nav import cart_boxes
        boxes = cart_boxes(self.CX, self.CY, math.pi / 2)
        self.assertEqual(len(boxes), len(cart_boxes(self.CX, self.CY, 0.0)))     # same parts, 13 boxes
        self.assertTrue(all(len(b) == 7 and b[0] <= b[3] and b[1] <= b[4] for b in boxes))

    def test_contains_true_rotated_footprint_at_cardinal_and_oblique_yaws(self):
        """yaw = 0, pi/2, pi, -pi/2 and an oblique yaw: the returned boxes CONTAIN the true rotated cart
        footprint (conservative), the property the uncoupled home-leg raycaster relies on."""
        from twin.restock.tail_nav import cart_boxes
        for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2, 0.6):
            boxes = cart_boxes(self.CX, self.CY, yaw)
            self.assertTrue(self._union_contains(boxes, self._true_footprint_corners(self.CX, self.CY, yaw)),
                            f'footprint not contained at yaw={yaw}')

    def test_rotated_boxes_never_smaller_than_the_axis_aligned_parts(self):
        """Rotating a part's rectangle and taking its axis-aligned bounds never SHRINKS it: each box's xy
        area at an off-axis yaw is >= the canonical part area (the 'never smaller than today' guarantee)."""
        from twin.restock.tail_nav import _cart_boxes_rotated, _cart_local_boxes
        local = _cart_local_boxes()
        for yaw in (math.pi / 2, -math.pi / 2, 0.6, 2.4):
            rotated = _cart_boxes_rotated(self.CX, self.CY, yaw)
            self.assertEqual([b[6] for b in rotated], [b[6] for b in local])     # same parts, same order
            for rb, lb in zip(rotated, local):
                area_rot = (rb[3] - rb[0]) * (rb[4] - rb[1])
                area_loc = (lb[3] - lb[0]) * (lb[4] - lb[1])
                self.assertGreaterEqual(area_rot + 1e-12, area_loc, f'{rb[6]} shrank at yaw={yaw}')

    def test_home_leg_boots_from_south_headed_parking_pose(self):
        """End to end: ``simulate_home_leg`` (which calls ``cart_boxes`` for the parked obstacle) runs from
        the dock-align fragment's south-headed ``cart_parking`` instead of raising at setup -- the preflight
        refusal the integration lane hit assembling coldwin20. We assert only that the drive executes and
        the parked cart is placed as a live obstacle (routing success is the lane's concern, not this gate)."""
        from twin.restock.tail_nav import simulate_home_leg, cart_boxes, HITCH_TO_CART_M
        fragment = json.loads((Path(__file__).resolve().parents[3] / 'docs' / 'examples'
                               / 'tow-lanes-mission-fragment.json').read_text())
        parking = fragment['mission_poses']['cart_parking']
        self.assertAlmostEqual(parking[2], math.pi / 2)                           # PR 271 south-headed yaw
        cart = (parking[0] - HITCH_TO_CART_M * math.cos(parking[2]),
                parking[1] - HITCH_TO_CART_M * math.sin(parking[2]), parking[2])
        self.assertEqual(len(cart_boxes(*cart)), 13)                              # no raise at the parked pose
        r = simulate_home_leg(FIXTURE['config'], FIXTURE['scenario'], FIXTURE['environment'],
                              camera_config=FIXTURE['camera_config'], start=parking, max_s=5.0)
        self.assertIn('status', r)                                                # the leg ran; cart_boxes did not refuse


def _ref_free(grid, p, radius, dynamic=()):
    """The ORIGINAL ``GridMap.free`` body (full obstacle scan), the oracle the vectorised cull must match."""
    x, y = p[:2]
    a, b, c, d = grid.bounds
    if not a + radius < x < c - radius or not b + radius < y < d - radius:
        return False
    for xmin, ymin, xmax, ymax in grid.obstacles:
        if math.hypot(max(xmin - x, 0, x - xmax), max(ymin - y, 0, y - ymax)) <= radius:
            return False
    return all(math.hypot(x - q[0], y - q[1]) > radius for q in dynamic)


def _ref_raycast(lo, hi, paths, origin, direction, rmax):
    """The ORIGINAL ``BoxRaycaster.raycast_closest`` body (full 3-D scan of every box), the oracle the
    culled + batched implementation must match bit for bit."""
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    par = np.abs(d) <= 1e-12
    dd = np.where(par, 1.0, d)
    t1, t2 = (lo - o) / dd, (hi - o) / dd
    inside = (lo <= o) & (o <= hi)
    tlo = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    thi = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    tmin, tmax = np.maximum(tlo.max(axis=1), 0.0), np.minimum(thi.min(axis=1), rmax)
    ok = tmin <= tmax
    if not ok.any():
        return {'hit': False}
    i = np.where(ok)[0][np.argmin(tmin[ok])]
    return {'hit': True, 'distance': float(tmin[i]), 'collision': paths[i], 'rigidBody': paths[i]}


class TailCheckSpeedupEquivalence(unittest.TestCase):
    """sim-mission-tail-check-speed-v1: the tail gate was sped up ~17x (100 s -> ~6 s) by culling/batching
    the geometry primitives, NOT by weakening the check. These regressions pin the optimisations to the
    exact pre-speedup result -- ``GridMap.free`` and ``BoxRaycaster`` must stay bit-identical to the
    original full scans, and the batched ``prime_read`` must leave ``simulate_home_leg`` byte-identical for
    BOTH the old axis-aligned parking and the PR 271 south-headed (pi/2) parking."""

    def test_gridmap_free_bit_identical_to_full_scan(self):
        from twin.navigation.core import GridMap
        rng = random.Random(20260203)
        mism = 0
        for _ in range(120):
            obs = []
            for _ in range(rng.randint(0, 40)):
                x0, y0 = rng.uniform(-10, 10), rng.uniform(-10, 10)
                obs.append((x0, y0, x0 + rng.uniform(.01, 5), y0 + rng.uniform(.01, 5)))
            g = GridMap((-12, -12, 12, 12), obs, resolution=rng.choice([0.1, 0.2, 0.5]))
            for _ in range(250):
                p = (rng.uniform(-12, 12), rng.uniform(-12, 12))
                r = rng.choice([1e-4, .05, .2, .34, .5, 1., 2.])
                dyn = tuple((rng.uniform(-12, 12), rng.uniform(-12, 12)) for _ in range(rng.randint(0, 3)))
                mism += g.free(p, r, dyn) != _ref_free(g, p, r, dyn)
        # boundary-exact points (a disk edge grazing an obstacle corner) must not drift either
        for _ in range(4000):
            x0, y0 = rng.uniform(-5, 5), rng.uniform(-5, 5)
            w, h = rng.uniform(.1, 4), rng.uniform(.1, 4)
            g = GridMap((-20, -20, 20, 20), [(x0, y0, x0 + w, y0 + h)])
            r = rng.choice([.2, .34, .5])
            for px in (x0 - r, x0 - r - 1e-12, x0 - r + 1e-12, x0 + w + r, x0 + w, x0):
                for py in (y0, y0 + h / 2, y0 - r, y0 + h + r):
                    mism += g.free((px, py), r) != _ref_free(g, (px, py), r)
        self.assertEqual(mism, 0)

    def test_boxraycaster_bit_identical_to_full_scan(self):
        from twin.restock.tail_nav import BoxRaycaster
        rng = random.Random(7)
        mism = 0
        for _ in range(250):
            boxes = []
            for k in range(rng.randint(0, 60)):
                x0, y0, z0 = rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-.5, 2)
                boxes.append((x0, y0, z0, x0 + rng.uniform(.05, 3), y0 + rng.uniform(.05, 3),
                              z0 + rng.uniform(.05, 2), f'/b{k}'))
            rc = BoxRaycaster(boxes)
            lo = np.array([b[:3] for b in boxes]) if boxes else np.empty((0, 3))
            hi = np.array([b[3:6] for b in boxes]) if boxes else np.empty((0, 3))
            paths = [b[6] for b in boxes]
            for _ in range(50):
                ox, oy = rng.uniform(-11, 11), rng.uniform(-11, 11)
                oz = rng.choice([0.15, 0.5, 1.0, rng.uniform(-1, 3)])
                ang = rng.uniform(-math.pi, math.pi)
                rmax = rng.choice([10., 5., 2.])
                dirc = (math.cos(ang), math.sin(ang), 0.0)
                got = rc.raycast_closest((ox, oy, oz), dirc, rmax)
                ref = _ref_raycast(lo, hi, paths, (ox, oy, oz), dirc, rmax)
                mism += got.get('hit') != ref.get('hit') or (got.get('hit') and (
                    got['distance'] != ref['distance'] or got['collision'] != ref['collision']))
        self.assertEqual(mism, 0)

    def test_prime_read_cache_matches_live_including_axis_aligned(self):
        """The batched ``prime_read`` serves the exact same hit as the per-beam path for every bearing a
        read requests, and the rare axis-aligned bearings (not cached) fall back to the identical live path."""
        from twin.restock.tail_nav import BoxRaycaster, static_boxes
        boxes = static_boxes(FIXTURE['environment'], FIXTURE['scenario'], FIXTURE.get('camera_config'))
        for x, y, yaw in [(-1.2, 15.25, 0.0), (-2.0, 14.0, math.pi), (-3.0, 14.5, 1.3),
                          (-2.0, 15.0, math.pi / 2), (-2.5, 14.2, 2.7)]:
            rc = BoxRaycaster(boxes)
            rc.set_dynamic([])
            rc.prime_read(x, y, yaw, (0.15, 0.5, 1.0), 180, 10.0)
            inc = 2 * math.pi / 180
            for i in range(180):
                a = yaw - math.pi + i * inc
                dx, dy = math.cos(a), math.sin(a)
                for z in (0.15, 0.5, 1.0):
                    cached = rc._read_cache.get((x, y, z, dx, dy, 10.0))
                    live = BoxRaycaster(boxes).raycast_closest((x, y, z), (dx, dy, 0.0), 10.0)
                    got = cached if cached is not None else live
                    self.assertEqual(got.get('hit'), live.get('hit'))
                    if got.get('hit'):
                        self.assertEqual(got['distance'], live['distance'])
                        self.assertEqual(got['collision'], live['collision'])

    def test_home_leg_identical_with_and_without_the_batched_prime(self):
        """Disabling ``prime_read`` forces the slow per-beam path; the home-leg verdict (status, reason,
        arrival time, pose, reason histogram) must be byte-identical with it on -- for BOTH an axis-aligned
        ``cart_parking`` (the fixture) and the PR 271 south-headed pi/2 parking (the dock fragment)."""
        from twin.restock import tail_nav
        fragment = json.loads((Path(__file__).resolve().parents[3] / 'docs' / 'examples'
                               / 'tow-lanes-mission-fragment.json').read_text())
        cases = [('axis_aligned', None), ('pi_over_2', tuple(fragment['mission_poses']['cart_parking']))]
        for label, start in cases:
            kw = dict(camera_config=FIXTURE['camera_config'], max_s=8.0, heartbeat_s=0, start=start)
            fast = tail_nav.simulate_home_leg(FIXTURE['config'], FIXTURE['scenario'],
                                              FIXTURE['environment'], **kw)
            orig_prime = tail_nav.BoxRaycaster.prime_read
            tail_nav.BoxRaycaster.prime_read = lambda *a, **k: None               # force the live per-beam path
            try:
                slow = tail_nav.simulate_home_leg(FIXTURE['config'], FIXTURE['scenario'],
                                                  FIXTURE['environment'], **kw)
            finally:
                tail_nav.BoxRaycaster.prime_read = orig_prime
            self.assertEqual(fast, slow, label)
