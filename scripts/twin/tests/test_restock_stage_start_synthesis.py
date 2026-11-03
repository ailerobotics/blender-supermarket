"""CPU proof for sim-stage-start-synthesis-v1: outbound / park / verify / return each start in isolation.

For each of the four late stages the start-phase shortcut now synthesizes a consistent entry state
(base on the right lane pose, cart coupled behind, carton loaded level on the cart or placed on the bay,
and -- through ``Mission`` -- the skipped prefix recorded as honest ``*_shortcut`` events). These tests
assert, on the REAL backroom-bay tail config/scenario/environment (``fixtures/mission_tail_bay.json``, the
same inputs the integration pre-boot gates read):

  * the synthesized state satisfies that stage's ENTRY precondition (coupled + loaded/placed) and its
    ACTIONS evidence contract is reachable from it;
  * ``Mission`` accepts the start_at (and still REFUSES one past the verify stock-commit gate);
  * the synthesis spawns the coupled train exactly on the configured TOW LANE the live drive follows
    (outbound at the lane start, park at the lane goal), so no off-lane spawn;
  * the preflight gates still pass on the synthesized config: ``config_preflight`` is unaffected (start_at
    / start_staging are not batch keys), and ``check_mission_tail`` reports no FAIL.

No Isaac, no GPU, no py_trees required (the BT dry-runs inside check_mission_tail SKIP without py_trees;
the config + geometry gates run regardless).

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_restock_stage_start_synthesis.py -q
"""
import copy
import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # scripts/ -> twin.*, check_mission_tail

from twin.restock.mission import ACTIONS, Mission, bounded_stages  # noqa: E402
from twin.restock.inventory import Job  # noqa: E402
from twin.restock.shortcut import (  # noqa: E402
    plan_start_state, resolve_start_staging, staged_scenario, stage_isolation, isolation_matrix,
    cart_pose_from_hitch, couple_start_poses, _derive_stage_base_pose, _outbound_lane_cfg)
from twin.navigation.lanes import couple_seat_from_cart  # noqa: E402

FIXTURE = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'mission_tail_bay.json').read_text())
SHELF_STAGING_FRAGMENT = json.loads(
    (Path(__file__).resolve().parents[3] / 'docs' / 'examples' / 'tow-lanes-mission-fragment.json').read_text())
# Late stages this lane adds, and the start_at each is driven from (return cannot be a start_at: Mission
# refuses one past verify, the stock-commit gate, so return is reached via start_at=verify).
NEW_STAGES = ('outbound', 'park', 'verify', 'return')


def _synth_config(stage, **extra):
    """The fixture runtime config with this stage's isolation flags applied (what the bench would run)."""
    iso = stage_isolation(stage)
    cfg = copy.deepcopy(FIXTURE['config'])
    cfg.update(extra)
    cfg['start_at'] = iso['start_at']
    cfg['start_staging'] = iso['start_staging']
    cfg['stop_after'] = iso['stop_after']
    cfg.setdefault('verify_ground_truth', {'enabled': True})
    return cfg, iso


class SynthesizedStateSatisfiesPreconditions(unittest.TestCase):
    """Each stage's synthesized entry state matches its ACTIONS precondition (coupled + loaded/placed)."""

    def setUp(self):
        self.scenario = FIXTURE['scenario']

    def test_outbound_and_park_spawn_loaded_and_coupled_on_the_cart(self):
        for stage, skipped in (('outbound', ('collect', 'couple')),
                               ('park', ('collect', 'couple', 'outbound'))):
            cfg, iso = _synth_config(stage)
            variant, record = staged_scenario(self.scenario, cfg)
            plan = plan_start_state(variant, stage, config=cfg)
            # precondition: "carton loaded AND the cart physically coupled behind the robot"
            self.assertTrue(plan.couple_cart, stage)
            self.assertEqual(plan.carton_target, 'cart', stage)
            self.assertEqual(plan.skipped, skipped, stage)
            self.assertTrue(record['coupled'], stage)
            # the loaded carton sits on the cart upper shelf (support z + half the carton height)
            h = self.scenario['handling_unit']['dimensions_m'][2]
            self.assertAlmostEqual(plan.carton_position[2],
                                   self.scenario['initial_staging']['cart_support_z_world'] + h / 2, msg=stage)

    def test_verify_spawns_placed_on_the_bay_coupled_arm_empty(self):
        cfg, iso = _synth_config('verify')
        variant, record = staged_scenario(self.scenario, cfg)
        plan = plan_start_state(variant, 'verify', config=cfg)
        # precondition: "the carton placed on the bay top, still coupled (transfer done)"
        self.assertTrue(plan.couple_cart)
        self.assertEqual(plan.carton_target, 'bay')
        self.assertEqual(plan.skipped, ('collect', 'couple', 'outbound', 'park', 'transfer'))
        from twin.restock.shortcut import _bay_target_world
        bx, by, top = _bay_target_world(cfg, self.scenario)   # config GT target wins over the raw bay centre
        h = self.scenario['handling_unit']['dimensions_m'][2]
        self.assertAlmostEqual(plan.carton_position[0], bx)
        self.assertAlmostEqual(plan.carton_position[1], by)
        self.assertAlmostEqual(plan.carton_position[2], top + h / 2)   # resting on the bay top

    def test_return_entry_is_the_verify_placed_coupled_state(self):
        # return's precondition ("stock committed at verify, still coupled at the bay") is reached by
        # synthesizing the verify-entry placed+coupled state and running the one instant verify.
        iso = stage_isolation('return')
        self.assertEqual((iso['start_at'], iso['stop_after']), ('verify', 'return'))
        cfg, _ = _synth_config('return')
        variant, _ = staged_scenario(self.scenario, cfg)
        plan = plan_start_state(variant, cfg['start_at'], config=cfg)
        self.assertEqual(plan.carton_target, 'bay')
        self.assertTrue(plan.couple_cart)


class MissionAcceptsTheStartPoints(unittest.TestCase):
    """Mission accepts every synthesized start_at and bounds the run to the one stage, and keeps the
    verify stock-commit gate intact (start_at past verify is refused -- the safety bound is not widened)."""

    def _mission(self, start_at, stop_after):
        job = Job(**FIXTURE['scenario']['job'])
        stages = bounded_stages(None, stop_after, start_at)
        return Mission(SimpleNamespace(), job, SimpleNamespace(), stages=stages, start_at=start_at)

    def test_each_new_stage_constructs_and_records_the_skipped_prefix(self):
        for stage in NEW_STAGES:
            iso = stage_isolation(stage)
            m = self._mission(iso['start_at'], iso['stop_after'])
            self.assertIsNotNone(m.shortcut, stage)
            self.assertEqual(m.shortcut['start_at'], iso['start_at'], stage)
            # the recorded skipped prefix is exactly the in-order prefix before the start_at stage
            self.assertEqual(tuple(m.shortcut['skipped']), ACTIONS[:ACTIONS.index(iso['start_at'])], stage)
            # the run is bounded to end at this stage
            self.assertEqual(m.stages[-1], stage)

    def test_start_at_past_the_verify_commit_gate_is_refused(self):
        job = Job(**FIXTURE['scenario']['job'])
        with self.assertRaisesRegex(ValueError, 'verify stock-commit gate'):
            Mission(SimpleNamespace(), job, SimpleNamespace(), start_at='return')


class SpawnsOnTheConfiguredTowLane(unittest.TestCase):
    """The synthesis spawns the coupled train exactly on the TOW LANE the live coupled drive follows:
    outbound at the lane start, park at the lane goal, cart coupled directly behind. A bench therefore
    never spawns off-lane (which would stall the follower on lane_not_acquired)."""

    def test_outbound_base_is_the_outbound_lane_start(self):
        cfg = FIXTURE['config']
        lane = _outbound_lane_cfg(cfg)
        self.assertEqual(lane['name'], 'outbound_dock_to_bay')
        base, cart_yaw = _derive_stage_base_pose(dict(cfg, start_at='outbound'), 'outbound')
        self.assertEqual([round(v, 6) for v in base], [round(v, 6) for v in lane['start']])
        self.assertAlmostEqual(cart_yaw, lane['start_cart_yaw'])
        # the authored cart staged behind the base equals the scenario's recorded cart_body_pose
        cart = cart_pose_from_hitch(base, 0.0)
        self.assertAlmostEqual(cart[0], FIXTURE['scenario']['initial_staging']['cart_body_pose'][0], places=2)
        self.assertAlmostEqual(cart[1], FIXTURE['scenario']['initial_staging']['cart_body_pose'][1], places=2)

    def test_park_base_is_the_outbound_lane_goal(self):
        cfg = FIXTURE['config']
        lane = _outbound_lane_cfg(cfg)
        base, _ = _derive_stage_base_pose(dict(cfg, start_at='park'), 'park')
        self.assertEqual([round(v, 6) for v in base], [round(v, 6) for v in lane['goal']])

    def test_outbound_park_refuse_without_a_tow_lane(self):
        cfg = copy.deepcopy(FIXTURE['config'])
        cfg['navigation'] = dict(cfg.get('navigation') or {})
        cfg['navigation'].pop('tow_lanes', None)
        for stage in ('outbound', 'park'):
            with self.assertRaises(ValueError):
                _derive_stage_base_pose(dict(cfg, start_at=stage), stage)


class PreflightGatesPassOnTheSynthesizedConfig(unittest.TestCase):
    """config_preflight is indifferent to start_at/start_staging (not batch keys), and check_mission_tail
    reports no FAIL on the synthesized config -- the integration pre-boot gates still pass."""

    def test_config_preflight_unaffected_by_the_synthesis_keys(self):
        from twin.restock_manipulation.config_preflight import check_activation
        base = copy.deepcopy(FIXTURE['config'])
        synth = dict(base, start_at='outbound', start_staging='dock_coupled', stop_after='outbound')
        required = ['navigation', 'nav_mode', 'nav_factory']   # a representative runtime require set
        host = {}
        # start_at/start_staging are not registered batch keys, so they never change the verdict:
        # the synthesized config yields exactly the same activation problems as the base config.
        self.assertEqual(check_activation(host, synth, required), check_activation(host, base, required))

    def test_check_mission_tail_has_no_failures_on_the_synthesized_verify_config(self):
        import check_mission_tail as tail
        cfg, _ = _synth_config('verify')
        rep = tail.run_checks(copy.deepcopy(cfg), FIXTURE['scenario'])
        failed = [name for verdict, name, _ in rep.rows if verdict == 'FAIL']
        self.assertEqual(failed, [], failed)


class DocMatrixStaysConsistent(unittest.TestCase):
    def test_every_stage_isolates_and_stop_after_is_universal(self):
        matrix = isolation_matrix()
        self.assertEqual([r['stage'] for r in matrix], list(ACTIONS))
        for r in matrix:
            self.assertTrue(r['can_isolate'], r['stage'])
            self.assertEqual(r['stop_after'], r['stage'])


class CoupleStartSeatsAtTheDockAlignedLane(unittest.TestCase):
    """SCOPE ADDITION (fm steering): the ``start_at='couple'`` shortcut must spawn the post-collect train
    at the dock the live outbound lane expects, DERIVED from the lane ``couple_dock`` through PR 282's
    single source (``navigation.lanes.couple_seat_from_cart``), so ``start_at=couple --stop-after outbound``
    acquires the lane instead of seating the base ~1 m / 90 deg off (coldwin23-2). Proven for BOTH docks:
    the dock-align SOUTH dock (the shelf-staging fragment) and the legacy EAST dock."""

    EAST_CONFIG = {
        'start_at': 'couple',
        'navigation': {'tow_lanes': [{
            'name': 'outbound_dock_to_staging', 'start': [-4.46, 15.25, 0.0], 'goal': [-2.0, 15.25, 0.0],
            'start_cart_yaw': 0.0, 'couple_dock': [-5.73, 15.25, 0.0]}]}}

    def _dock_cases(self):
        south = copy.deepcopy(SHELF_STAGING_FRAGMENT)
        south['start_at'] = 'couple'
        return (('south', south), ('east', copy.deepcopy(self.EAST_CONFIG)))

    def test_couple_seat_within_leadin_tol_of_lane_start_for_both_docks(self):
        for name, cfg in self._dock_cases():
            lane = _outbound_lane_cfg(cfg)
            poses = couple_start_poses(cfg)
            self.assertIsNotNone(poses, name)
            seat = poses['robot_seat']
            # the seat is PR 282's single-source derivation from the lane's own couple_dock
            expect = couple_seat_from_cart(tuple(lane['couple_dock']),
                                           articulation=lane.get('couple_dock_articulation', 0.0))
            self.assertEqual([round(v, 6) for v in seat], [round(v, 6) for v in expect], name)
            # and it lies within the lane's lead-in tolerance (start_tol_xy + lead_in_max_m) of the start
            tol = lane.get('start_tol_xy', 0.25) + lane.get('lead_in_max_m', 0.0)
            self.assertLessEqual(math.dist(seat[:2], lane['start'][:2]), tol, name)
            self.assertLessEqual(abs(math.atan2(math.sin(seat[2] - lane['start'][2]),
                                                math.cos(seat[2] - lane['start'][2]))),
                                 lane.get('start_tol_yaw', 0.15), name)

    def test_staged_scenario_spawns_robot_at_seat_and_cart_at_dock(self):
        for name, cfg in self._dock_cases():
            lane = _outbound_lane_cfg(cfg)
            variant, record = staged_scenario(FIXTURE['scenario'], cfg)
            st = variant['initial_staging']
            poses = couple_start_poses(cfg)
            self.assertEqual([round(v, 6) for v in st['robot_base_pose']],
                             [round(v, 6) for v in poses['robot_seat']], name)
            self.assertEqual([round(v, 6) for v in st['cart_body_pose']],
                             [round(v, 6) for v in lane['couple_dock']], name)
            # the cart is left UNCOUPLED for the real couple stage (the shortcut authors no hitch)
            self.assertFalse(record['coupled'], name)
            self.assertEqual(record['robot_base_pose_origin'], 'derived_from_couple_dock', name)
            self.assertEqual(record['couple_seat'], [round(v, 4) for v in poses['robot_seat']], name)
            # the carton stays loaded on the cart (couple's precondition), not placed on a bay
            plan = plan_start_state(variant, 'couple', config=cfg)
            self.assertEqual(plan.carton_target, 'cart', name)
            self.assertFalse(plan.couple_cart, name)

    def test_couple_on_a_legacy_east_lane_is_byte_identical(self):
        # sim-couple-shortcut-settle-dt-v1: the dock is now resolved from the ONE source (lane couple_dock,
        # else derived from the lane start + start_cart_yaw). The backroom-bay fixture lane has no
        # couple_dock but a start/start_cart_yaw whose DERIVED dock equals the scenario cart spawn (legacy
        # EAST), so couple_start_poses resolves that dock yet staged_scenario is byte-identical (nothing to
        # re-head). A SOUTH lane (below) re-heads the spawn.
        cfg = copy.deepcopy(FIXTURE['config'])
        cfg['start_at'] = 'couple'
        scenario = FIXTURE['scenario']
        poses = couple_start_poses(cfg)
        self.assertIsNotNone(poses)
        self.assertEqual([round(v, 4) for v in poses['cart_dock']],
                         [round(float(v), 4) for v in scenario['initial_staging']['cart_body_pose']])
        variant, record = staged_scenario(scenario, cfg)
        self.assertIs(variant, scenario)      # dock == scenario spawn => no rewrite
        self.assertIsNone(record)

    def test_couple_on_a_south_lane_without_couple_dock_dock_aligns_from_lane_start(self):
        # The side-bay lane carries only start + start_cart_yaw (SOUTH), no explicit couple_dock. The
        # derived dock re-heads the spawn south so the coupled train faces the lane heading (fixes
        # coldwin25e, where the cart spawned EAST and the south lane was never acquired).
        cfg = copy.deepcopy(FIXTURE['config'])
        cfg['start_at'] = 'couple'
        cfg['navigation']['tow_lanes'] = [{'name': 'outbound_dock_to_side_bay',
                                           'start': [-3.875, 14.5, -math.pi / 2],
                                           'start_cart_yaw': -math.pi / 2,
                                           'goal': [-4.3, 6.95, -math.pi / 2]}]
        poses = couple_start_poses(cfg)
        self.assertIsNotNone(poses)
        self.assertAlmostEqual(poses['cart_dock'][2], -math.pi / 2, places=5)  # cart faces SOUTH
        self.assertAlmostEqual(poses['cart_dock'][0], -3.875, places=3)
        self.assertAlmostEqual(poses['cart_dock'][1], 15.77, places=2)  # == towstaging3's couple_dock
        self.assertEqual([round(v, 4) for v in poses['robot_seat']],
                         [round(v, 4) for v in cfg['navigation']['tow_lanes'][0]['start']])
        variant, record = staged_scenario(FIXTURE['scenario'], cfg)
        self.assertIsNot(variant, FIXTURE['scenario'])
        self.assertAlmostEqual(variant['initial_staging']['cart_body_pose'][2], -math.pi / 2, places=5)
        self.assertEqual(record['start_at'], 'couple')


if __name__ == '__main__':
    unittest.main()
