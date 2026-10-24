"""CPU contract tests for the bounded, observable move_group respawn policy
(sim-robust-movegroup-respawn-reliability-v1). No ROS/launch/GPU.

These pin: the respawn limit is honoured (give-up at exactly max_retries + 1), the give-up
reason is named and carries the count, the config -> plan threading respects the configured
delay + limit, the supervisor is observe-only (no respawn/kill, no launch actions => no double
supervision), and an absent config block is byte-identical to the bounded default.
"""
from types import SimpleNamespace as S
import unittest

from twin.restock_host.movegroup_respawn import (
    DEFAULT_MAX_RETRIES, DEFAULT_RESPAWN_DELAY_S, RespawnSupervisor,
    resolve_respawn, respawn_decision, respawn_log_line)


class RespawnDecisionTests(unittest.TestCase):
    def test_limit_honoured_gives_up_at_max_plus_one(self):
        # max_retries=2 (the proof budget): deaths 1,2 respawn; death 3 gives up.
        self.assertEqual(respawn_decision(1, 2), ('respawn', 1, 2))
        self.assertEqual(respawn_decision(2, 2), ('respawn', 2, 2))
        self.assertEqual(respawn_decision(3, 2), ('give_up', 2, 2))
        self.assertEqual(respawn_decision(9, 2), ('give_up', 2, 2))

    def test_zero_budget_gives_up_on_first_death(self):
        self.assertEqual(respawn_decision(1, 0), ('give_up', 0, 0))

    def test_give_up_reason_named_with_count(self):
        line = respawn_log_line(3, 2, 3.0, pid=4242, returncode=-6)
        self.assertIn('MOVE_GROUP_RESPAWN_GIVE_UP', line)
        self.assertIn('move_group', line)
        self.assertIn('2/2', line)                 # respawns used
        self.assertIn('will NOT be relaunched', line)
        self.assertIn('RESPAWN_WAIT_CAP', line)    # names the owning client fast-fail
        self.assertIn('pid=4242', line)
        self.assertIn('exit=-6', line)

    def test_respawn_line_carries_attempt_and_delay(self):
        line = respawn_log_line(1, 2, 3.0)
        self.assertIn('MOVE_GROUP_RESPAWN ', line)
        self.assertNotIn('GIVE_UP', line)
        self.assertIn('attempt 1/2', line)
        self.assertIn('respawn_delay=3.0s', line)


class ResolveRespawnTests(unittest.TestCase):
    def test_absent_block_is_default(self):
        self.assertEqual(resolve_respawn({}), (DEFAULT_MAX_RETRIES, DEFAULT_RESPAWN_DELAY_S))
        self.assertEqual(resolve_respawn({'other': 1}), (DEFAULT_MAX_RETRIES, DEFAULT_RESPAWN_DELAY_S))

    def test_explicit_values_respected_and_clamped(self):
        self.assertEqual(resolve_respawn({'movegroup_respawn': {'max_retries': 2, 'delay': 5.0}}),
                         (2, 5.0))
        # negatives clamp to zero; respawn_delay is an accepted alias for delay.
        self.assertEqual(resolve_respawn({'movegroup_respawn': {'max_retries': -4, 'respawn_delay': -1.0}}),
                         (0, 0.0))


class SupervisorTests(unittest.TestCase):
    def _run(self, max_retries, deaths):
        captured = []
        sup = RespawnSupervisor(max_retries, 3.0, captured.append)
        for i in range(deaths):
            sup.on_exit(S(pid=1000 + i, returncode=-6), context=None)
        return sup, captured

    def test_counts_and_gives_up_at_budget(self):
        sup, lines = self._run(2, 3)
        self.assertEqual(sup.deaths, 3)
        self.assertTrue(sup.gave_up)
        self.assertIn('attempt 1/2', lines[0])
        self.assertIn('attempt 2/2', lines[1])
        self.assertIn('MOVE_GROUP_RESPAWN_GIVE_UP', lines[2])

    def test_within_budget_never_gives_up(self):
        sup, lines = self._run(3, 2)
        self.assertFalse(sup.gave_up)
        self.assertTrue(all('GIVE_UP' not in line for line in lines))

    def test_observe_only_no_double_supervision(self):
        """The supervisor must not itself respawn or kill: the only respawner is ROS 2 launch."""
        sup = RespawnSupervisor(2, 3.0, lambda _l: None)
        # on_exit returns NO launch actions (would otherwise be a second teardown/respawner).
        self.assertIsNone(sup.on_exit(S(pid=1, returncode=-6), context=None))
        for forbidden in ('respawn', 'kill', 'spawn', 'restart', 'terminate'):
            self.assertFalse(hasattr(sup, forbidden),
                             f'supervisor must be observe-only; found a {forbidden} method')


if __name__ == '__main__':
    unittest.main()
