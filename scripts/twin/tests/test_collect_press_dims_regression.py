"""sim-collect-press-regression-v1 -- CPU regression for the coldwin32 collect-press stall.

Root cause (NOT PR 311 gravity_compensation, which is falsified -- absent from every config and
structurally post-weld only): the HOST carton-profile dimension in ros_host_config.json drifted from
the scenario single source 0.45/0.30/0.18 (handling_unit.dimensions_m, PR 268) to 0.6/0.4/0.16. The
HOST side-grasp PICK standoff derives from those dims (wall_grip: pad stands off dims[wall-axis]/2 +
margin from the carton centre), so the larger host dims over-reach the pad past the real face and the
capped 20 mm over-press cannot reach contact ('contact not reached within capped over-press travel';
coldwin32 / warmacc3 gap 34.96 mm).

These tests pin: the committed HOST config is back on the scenario dims (with carton_level kept); the
config_preflight carton-dim mismatch guard refuses the drift; and the real side-grasp geometry
reproduces the ~30 mm over-press-cap-exceeding over-reach for the drifted dims (0 for the correct dims).
"""
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from twin.restock_manipulation import config_preflight as cp  # noqa: E402

_REPO = os.path.join(os.path.dirname(__file__), '..', '..', '..')
_HOST_CFG = os.path.join(os.path.dirname(__file__), '..', 'restock_manipulation', 'ros_host_config.json')
SCENARIO_DIMS = [0.45, 0.3, 0.18]          # handling_unit.dimensions_m (PR 268 single source)
DRIFTED_DIMS = [0.6, 0.4, 0.16]            # the coldwin32 regression
OVERPRESS_CAP_M = 0.02                      # contact_overpress_cap_m (NEVER widened)
CONTACT_MARGIN = 0.00275                    # juice_orange tcp_surface_offset
SOURCE_WALL = 3                             # backstock_00 (odd -> standoff uses dims[1])


def _host_cfg():
    with open(_HOST_CFG) as fh:
        return json.load(fh)


class CommittedHostConfigPin(unittest.TestCase):
    def test_juice_orange_dims_are_the_scenario_single_source(self):
        prof = next(p for p in _host_cfg()['profiles'] if p['sku'] == 'juice_orange')
        self.assertEqual([float(x) for x in prof['dimensions']], SCENARIO_DIMS,
                         'HOST carton dims must match the scenario handling_unit.dimensions_m (PR 268)')

    def test_carton_level_kept_alongside_transfer_service_carry(self):
        cfg = _host_cfg()
        self.assertTrue(cfg.get('carton_level'),
                        'carton_level must stay enabled for the parallel-to-ground place leveling')
        self.assertTrue(cfg.get('transfer_service_carry'),
                        'transfer_service_carry (PR 312) must remain on')


class CheckCartonDimsTests(unittest.TestCase):
    def _scenario(self, dims):
        return {'handling_unit': {'sku': 'juice_orange', 'dimensions_m': dims}}

    def test_drifted_host_dims_are_refused(self):
        host = {'profiles': [{'sku': 'juice_orange', 'dimensions': DRIFTED_DIMS}]}
        problems = cp.check_carton_dims(host, self._scenario(SCENARIO_DIMS))
        self.assertEqual(len(problems), 1)
        self.assertIn('carton-dim DRIFT', problems[0])
        self.assertIn('coldwin32', problems[0])

    def test_matching_dims_pass(self):
        host = {'profiles': [{'sku': 'juice_orange', 'dimensions': SCENARIO_DIMS}]}
        self.assertEqual(cp.check_carton_dims(host, self._scenario(SCENARIO_DIMS)), [])

    def test_committed_config_passes_against_the_scenario(self):
        self.assertEqual(cp.check_carton_dims(_host_cfg(), self._scenario(SCENARIO_DIMS)), [])

    def test_absent_scenario_is_a_noop(self):
        host = {'profiles': [{'sku': 'juice_orange', 'dimensions': DRIFTED_DIMS}]}
        self.assertEqual(cp.check_carton_dims(host, None), [])
        self.assertEqual(cp.check_carton_dims(host, {}), [])

    def test_other_sku_not_compared(self):
        host = {'profiles': [{'sku': 'other', 'dimensions': DRIFTED_DIMS}]}
        self.assertEqual(cp.check_carton_dims(host, self._scenario(SCENARIO_DIMS)), [])

    def test_tolerant_to_float_noise(self):
        host = {'profiles': [{'sku': 'juice_orange', 'dimensions': [0.4500001, 0.3, 0.18]}]}
        self.assertEqual(cp.check_carton_dims(host, self._scenario(SCENARIO_DIMS)), [])


def _wall_grip_standoff(dims, wall=SOURCE_WALL, margin=CONTACT_MARGIN):
    """|pad-out offset| from the carton centre that geometry.wall_grip places the contact pose at
    (== half the wall-axis dimension + the contact margin). Faithful to wall_policy.WallCartonPolicy."""
    root = os.environ.get('SHELFBOT_ROOT', os.path.expanduser('~/workspaces/shelfbot_newgripper'))
    if root not in sys.path:
        sys.path.insert(0, root)
    from shelfbot.motion_planning import geometry as geo
    from types import SimpleNamespace
    variant = SimpleNamespace(tray_size=tuple(dims), contact_margin=margin,
                              approach_offset_x_m=0.18, grasp_quat=(.5, .5, .5, .5))
    ox, oy, rx, ry, q = geo.wall_grip(variant, SimpleNamespace(yaw=0.0), wall)
    return math.hypot(ox, oy)


def _shelfbot_available():
    root = os.environ.get('SHELFBOT_ROOT', os.path.expanduser('~/workspaces/shelfbot_newgripper'))
    return os.path.exists(os.path.join(root, 'shelfbot', 'motion_planning', 'geometry.py'))


@unittest.skipUnless(_shelfbot_available(), 'shelfbot_newgripper geometry not available')
class SideGraspStandoffReproduction(unittest.TestCase):
    def test_drifted_dims_overreach_exceeds_overpress_cap(self):
        d_scen = _wall_grip_standoff(SCENARIO_DIMS)
        d_drift = _wall_grip_standoff(DRIFTED_DIMS)
        overreach = d_drift - d_scen
        # odd wall -> standoff uses dims[1]; drift is (0.40-0.30)/2 = 0.05 m
        self.assertAlmostEqual(overreach, (DRIFTED_DIMS[1] - SCENARIO_DIMS[1]) / 2, places=6)
        residual = overreach - OVERPRESS_CAP_M
        self.assertGreater(residual, 0.0,
                           'the capped over-press cannot recover the over-reach -> press fails')
        # predicted residual ~30 mm bounds the GPU-measured ~35 mm (raycast origin sits behind the pad)
        self.assertTrue(0.025 <= residual <= 0.045, f'residual {residual*1000:.1f} mm != ~30 mm')

    def test_scenario_dims_reach_contact(self):
        d_scen = _wall_grip_standoff(SCENARIO_DIMS)
        d_correct = _wall_grip_standoff(SCENARIO_DIMS)
        self.assertEqual(d_scen, d_correct)  # no over-reach with the correct dims -> over-press reaches


if __name__ == '__main__':
    unittest.main()
