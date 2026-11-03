"""sim-collect-reposition-block-v1: the mid-collect carry reposition must HOLD the base pinned on a
blocked/stopped tick instead of issuing a zero-velocity mobile.command (which releases the base brake
and let the carried carton's moment drift the base into the live backstock Support -> permanent
blocked_no_route, boxc1a). No Isaac/ROS/GPU imports."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.runtime import Packet, RuntimeAdapter
from twin.restock.inventory import Job


def _blocked(linear=0.0, angular=0.0, reason='blocked_no_route'):
    return NS(status='running', linear=linear, angular=angular, reason=reason)


class RepositionHoldTests(unittest.TestCase):
    def _adapter(self, *, hold, reason='blocked_no_route', linear=0.0, angular=0.0):
        calls = {'hold': 0, 'command': 0}
        mobile = NS(park=lambda: None,
                    hold=lambda: calls.__setitem__('hold', calls['hold'] + 1),
                    release=lambda: None,
                    command=lambda *a, **k: calls.__setitem__('command', calls['command'] + 1),
                    attach=lambda obs: None, coupling_state=lambda o: {'healthy': True})
        p = Packet('frame-1', 'test_fixture', 0.1, None, None, None, None, True, False, 0.0, True)
        port = NS(advance=lambda dt: p, controls=lambda c: None, stop=lambda: None)
        nav = NS(start=lambda g, c, **kw: None,
                 step=lambda dt, pk: _blocked(linear, angular, reason), cancel=lambda: None)
        manip = NS(start=lambda a, j: None, cart_location='rollwagen',
                   step=lambda dt, s: NS(status='running', reason='', controls=()))
        poses = {k: [0, 0, 0] for k in ('home', 'pickup', 'cart_dock', 'shelf_approach')}
        a = RuntimeAdapter(nav, mobile, manip, port, poses, fixture_mode=True,
                           hold_base_on_reposition_block=hold)
        a.start('collect', Job('j', 'milk', 1, 'back', 'shelf'))
        a.phase = 'navigate'
        a._collect_reposition_pending = True
        return a, calls

    def test_blocked_reposition_holds_base_when_enabled(self):
        # The exact boxc1a tick: uncoupled collect carry reposition, planner returns no route (v=w=0).
        a, calls = self._adapter(hold=True)
        r = a._step(0.1)
        self.assertEqual((r.status, r.reason), ('running', 'blocked_no_route'))
        self.assertEqual(calls['hold'], 1)       # base pinned -> cannot drift off the departure pose
        self.assertEqual(calls['command'], 0)    # NOT released to a zero-velocity "drive"

    def test_blocked_reposition_default_off_is_byte_identical(self):
        # Default (flag off): the same tick issues mobile.command(0,0,...) exactly as before.
        a, calls = self._adapter(hold=False)
        a._step(0.1)
        self.assertEqual(calls['command'], 1)
        self.assertEqual(calls['hold'], 0)

    def test_driving_reposition_tick_still_commands(self):
        # A tick that commands ANY motion (tracking / turn-in-place) drives through mobile.command even
        # with the hold enabled -- the hold only engages on a FULL stop (linear==0 AND angular==0).
        a, calls = self._adapter(hold=True, reason='tracking', linear=0.12, angular=0.0)
        a._step(0.1)
        self.assertEqual(calls['command'], 1)
        self.assertEqual(calls['hold'], 0)
        a2, c2 = self._adapter(hold=True, reason='tracking', linear=0.0, angular=0.2)  # turn in place
        a2._step(0.1)
        self.assertEqual(c2['command'], 1)
        self.assertEqual(c2['hold'], 0)

    def test_predicate_scoped_to_uncoupled_collect_reposition(self):
        # The hold predicate is False outside the mid-collect carry reposition, so every other navigate
        # leg (couple/outbound/park/return, or collect before the reposition) is byte-identical.
        a, _ = self._adapter(hold=True)
        self.assertTrue(a._hold_base_on_nav_hold(_blocked()))
        a._collect_reposition_pending = False
        self.assertFalse(a._hold_base_on_nav_hold(_blocked()))   # not repositioning yet
        a._collect_reposition_pending = True
        a.action = 'outbound'
        self.assertFalse(a._hold_base_on_nav_hold(_blocked()))   # a different stage
        a.action = 'collect'
        self.assertFalse(a._hold_base_on_nav_hold(_blocked(linear=0.05)))  # a driving tick

    def test_predicate_off_by_default(self):
        a, _ = self._adapter(hold=False)
        self.assertFalse(a.hold_base_on_reposition_block)
        self.assertFalse(a._hold_base_on_nav_hold(_blocked()))


if __name__ == '__main__':
    unittest.main()
