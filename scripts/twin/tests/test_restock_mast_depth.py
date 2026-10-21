"""Mast depth camera (nvblox second feed) + cold-collect confirmation diagnosis.

CPU geometric design predictions only: offline ray/visibility and a finite-resolution voxel
replay, never rendered RGB-D, inventory or physics evidence.
"""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock_camera import mast_depth as M
from twin.restock_camera.config import camera_config
from twin.restock_camera.geometry import Intrinsics, project, transform_points, look_at
from twin.restock_camera.visibility import box_surface, sample_plane, inspect_view, ray_hits
from twin.navigation.observed_space import ObservedSpace


def design(mast_depth=False):
    scenario = dict(handling_unit=dict(local_wall=3, grasp_mode='side', dimensions_m=[.6, .4, .16]),
        initial_staging=dict(support_center_xy=[-5.73, 17.1], unit_bottom_z=.6,
            target_surface_pose_world=[-3.05, 4.826, .73, 0], cart_load_center_body_xy=[.08, 0],
            cart_support_z_world=.225))
    return camera_config(scenario, dict(spec=dict(arm_mount_z=.5)), mast_depth=mast_depth)


class MastIntrinsicsAndMount(unittest.TestCase):
    def test_realsense_class_centred_pinhole(self):
        k = M.mast_intrinsics()
        self.assertEqual((k.width, k.height), (M.MAST_WIDTH, M.MAST_HEIGHT))
        self.assertEqual((k.cx, k.cy), (k.width / 2, k.height / 2))   # Isaac/native pinhole requires centred
        self.assertAlmostEqual(k.fx, k.fy)                            # square pixels
        hfov = math.degrees(2 * math.atan((k.width / 2) / k.fx))
        self.assertAlmostEqual(hfov, M.MAST_HFOV_DEG, places=6)       # 87 deg D435 depth field
        self.assertLess(k.near, k.far)
        self.assertGreaterEqual(k.near, 0.28)                         # D435 min range class

    def test_mount_is_body_fixed_high_and_looks_over_the_arm(self):
        tf = M.mast_world_from_optical((0., 0., 0.))                  # base == world here
        self.assertGreater(tf[2, 3], 1.4)                            # high on the mast, above the arm
        forward = tf[:3, 2]
        self.assertGreater(forward[1], 0.5)                          # toward the arm's working side (+Y)
        self.assertLess(forward[2], -0.3)                           # and tilted down onto the workspace

    def test_native_spec_entry_has_collision_housing_on_robot_base(self):
        camera, solids = M.mast_camera_entry()
        self.assertEqual(camera['parent'], 'robot_base')             # body-fixed, travels with the rig
        self.assertEqual(camera['role'], 'nvblox_depth')             # not a QR/decode station camera
        self.assertTrue(all(s['parent'] == 'robot_base' and s['collision'] for s in solids))
        self.assertTrue(any(s['id'] == 'mast_housing' for s in solids))


class CameraConfigSwitch(unittest.TestCase):
    def test_off_is_byte_identical_default(self):
        import json
        self.assertEqual(json.dumps(design(False), sort_keys=True),
                         json.dumps(design(), sort_keys=True))
        self.assertNotIn('mast', design(False)['cameras'])

    def test_on_adds_only_the_mast_depth_camera_and_no_support(self):
        off, on = design(False), design(True)
        self.assertEqual(set(on['cameras']) - set(off['cameras']), {'mast'})
        self.assertEqual(set(on['supports']), set(off['supports']))   # no QR support for the depth cam
        self.assertEqual(on['cameras']['mast']['parent'], 'robot_base')
        # Every other camera and every support is unchanged by the switch.
        for cid in off['cameras']:
            self.assertEqual(on['cameras'][cid], off['cameras'][cid])

    def test_nvblox_feed_is_camera_1_distinct_from_the_wrist(self):
        feed = M.nvblox_feed()
        self.assertEqual(feed['index'], 1)                           # wrist is camera_0
        self.assertNotIn(feed['world_depth'], ('world_depth',))      # distinct from the wrist output
        self.assertNotEqual(feed['frame_id'], 'restock_camera')


class CarriedCartonObstacleMap(unittest.TestCase):
    """Deliverable 1: the mast feed sees the place/transfer region while the carton is held."""

    def setUp(self):
        # Robot-relative (base-frame) transfer/place-re-park geometry. The carton is held in
        # transport close to the body over the arm base; the shelf slot + approach are ahead on
        # the arm's working side. World poses: carry along outbound, place re-park at the shelf
        # approach [-3.05, 6.0, pi], transfer at the slot [-3.05, 4.826, 0.73] (mission-scenario).
        self.base = (0., 0., 0.)
        self.carton = box_surface('carton', (0.0, 0.30, 0.70), (0.6, 0.4, 0.16))
        self.place = sample_plane((0.0, 1.15, 0.73), (0.5, 0.3), 0.0, count=5)      # shelf slot
        self.approach = sample_plane((0.0, 0.90, 0.78), (0.6, 0.5), 0.0, count=5)   # arm sweep volume

    def test_mast_sees_place_region_and_approach_with_carton_held(self):
        # Carton MASKED out of the feed (not an occluder for nvblox): the mast sees the place
        # region and the approach volume from its body-fixed vantage over the arm.
        for region, name in ((self.place, 'place'), (self.approach, 'approach')):
            view = M.sees_points(self.base, region, [])
            self.assertTrue(view['all_samples_visible'], name)
            self.assertEqual(view['in_fov_and_range'], view['samples'], name)

    def test_held_carton_does_not_blind_the_body_mounted_mast(self):
        # Even treating the held carton as a hard occluder, the mast (high, body-fixed) keeps a
        # clear sightline to the place region -- unlike the eye-in-hand wrist the carton fills.
        view = M.sees_points(self.base, self.place, [self.carton])
        self.assertEqual(view['visible_samples'], view['samples'])
        self.assertNotIn('carton', view['occluders'])

    def test_wrist_eye_in_hand_is_blocked_by_the_held_carton(self):
        # The wrist rides the tool; the held carton sits centimetres in front of the lens and
        # fills its narrow field -- this is finding B' (nvblox starves, ESDF freezes).
        k = M.mast_intrinsics()
        wrist = look_at((0.0, 0.15, 0.72), (0.0, 0.30, 0.70))   # lens just behind the carton face
        near_carton = box_surface('carton', (0.0, 0.30, 0.70), (0.6, 0.4, 0.16))
        blocked = sum(ray_hits(wrist[:3, 3], p, near_carton.triangles) for p in self.place)
        self.assertEqual(blocked, len(self.place))              # every place sample occluded


class EsdfKeepsUpdating(unittest.TestCase):
    """Deliverable 1: the ESDF (observed voxel map) keeps updating across the held-carton window."""

    def test_mast_feed_stays_fresh_and_nonempty_while_wrist_is_frozen(self):
        stamps = [10.0 + 0.1 * i for i in range(8)]                 # the held-carton window
        mast_updates, wrist_updates = 0, 0
        for now in stamps:
            mast = ObservedSpace([M.proof_depth_frame(self.base_pose(), now)],
                                 required_sensors={'mast'}, resolution=.10, stale_after=.3)
            self.assertEqual(mast.freshness(now), '')              # feed is fresh every tick
            if mast.occupied or mast.free:
                mast_updates += 1
            wrist = ObservedSpace([M.proof_depth_frame(self.base_pose(), now, full_frame_masked=True)],
                                  required_sensors={'mast'}, resolution=.10, stale_after=.3)
            if wrist.occupied or wrist.free:
                wrist_updates += 1
        self.assertEqual(mast_updates, len(stamps))                # mast: ESDF updates every tick
        self.assertEqual(wrist_updates, 0)                         # carton-filled wrist: frozen

    def test_masked_carton_pixels_neither_clear_nor_occupy(self):
        # The carton silhouette is masked: it must not clear free space nor assert an obstacle;
        # the surrounding scene still integrates, so the map updates without hallucinating.
        frame = M.proof_depth_frame(self.base_pose(), 10.0, carton_masked=True)
        space = ObservedSpace([frame], required_sensors={'mast'}, resolution=.10, stale_after=.3)
        self.assertTrue(space.free)                                # scene around the carton still seen
        self.assertEqual(space.self_hits, set())                   # masked pixels asserted nothing

    @staticmethod
    def base_pose():
        return (0., 0., 0.)


class ColdCollectConfirmationCause(unittest.TestCase):
    """Deliverable 2: name why cold-collect grasp confirmation fails -- a geometry proof.

    run-327/328 (cold): the collect capture via the FIXED source camera never establishes a
    support registration and the manip reports 'no initial sensor frame'; the host relabels it
    'no established support registration to hold under occlusion'. The question: is the fixed
    source camera's view occluded (a camera-placement problem), or is it timing/sequencing?

    This proves the source camera retains a clear, in-FOV sightline to BOTH support markers and
    the witness at the parked collect pose, and that a south-approaching grasp arm does not block
    them (the markers/witness sit at y~17.4, directly under the overhead lens). So a clean cold
    registration frame IS geometrically available; the failure is the absence of an established
    registration to HOLD through the grasp/press occlusion window the scenario itself declares
    (occlusion_window_s). That is collect-provider/orchestrator sequencing, owned by the
    integration lane -- NOT a camera-placement problem, so no confirmation camera is added.
    """

    def setUp(self):
        # Fixed source camera (commissioned pose) and the backstock support geometry.
        self.eye = np.array([-5.73, 17.3, 1.45])
        target = np.array([-5.73, 17.1, 0.68])
        self.W = look_at(self.eye, target)
        self.K = Intrinsics(2560, 1920, 2000., 2000., 1280., 960., .15, 3.5)
        support = np.array([-5.73, 17.1, 0.6])
        self.markers = [support + [-.30, .30, 0], support + [.30, .30, 0]]
        self.witness = support + [0, .30, 0]
        self.carton_top = np.array([-5.73, 17.1, 0.76])

    def _visible(self, point, occluders):
        uv, valid = project(transform_points(np.linalg.inv(self.W), [point]), self.K)
        blocked = any(ray_hits(self.eye, point, o.triangles) for o in occluders)
        return bool(valid[0]) and not blocked

    def test_source_registration_targets_are_clear_with_arm_stowed(self):
        # Parked, arm stowed: a clean source registration frame is available (markers+witness in
        # FOV and unoccluded). The cold run could establish the hold here, before the arm reaches.
        for target in (*self.markers, self.witness, self.carton_top):
            self.assertTrue(self._visible(target, []))

    def test_grasp_arm_does_not_occlude_the_support_markers_or_witness(self):
        # A conservative south-approaching grasp arm (from the robot base at y=16 up to the
        # carton) and the pre-weld gripper press over the carton: neither blocks the markers or
        # witness. So the fixed source view is adequate GIVEN a prior established registration --
        # a new/re-aimed camera would not change the cold-start failure.
        arm = box_surface('arm', (-5.73, 16.6, 1.0), (0.25, 1.3, 0.25))         # base->carton reach
        press = box_surface('press', (-5.73, 17.0, 0.9), (0.25, 0.3, 0.25))     # gripper over carton
        for target in (*self.markers, self.witness):
            self.assertTrue(self._visible(target, [arm, press]))

    def test_the_press_can_transiently_block_only_the_carton_top(self):
        # The only element the grasp press can transiently occlude is the carton-top target --
        # by which point the registration must already be established. This is the declared
        # occlusion window, not a missing camera sightline.
        press = box_surface('press', (-5.73, 17.0, 0.95), (0.4, 0.4, 0.3))
        self.assertFalse(self._visible(self.carton_top, [press]))               # carton top blocked
        for target in (*self.markers, self.witness):
            self.assertTrue(self._visible(target, [press]))                     # registration still clear


if __name__ == '__main__':
    unittest.main()
