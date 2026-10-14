"""CPU contract tests for the /tf sim-time-backbone relay payload (Firstmate msg162 (b)).

nvblox and the incorporation ack look up restock_camera_arm <- ur_base_link at the depth stamp, so
the relayed /tf must carry exactly those frame ids (prim names), published before the masked depth
for the tick. The USD transform read is GPU-only; the frame-id + payload assembly is pure and tested
here. The /tf-before-depth ORDERING is structural: NativeTransport._publish_relay runs inside
isaac_port.advance (which calls native_transport.tick), ahead of the manipulation runtime's
sensing_bridge.publish on the same tick.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.native_transport import tf_payload, _frame_id

UR_BASE = '/MobileRobot/ur_base_link'
ARM_OPTICAL = '/MobileRobot/front_camera_frame/arm_camera/restock_camera_arm'


class FrameId(unittest.TestCase):
    def test_frame_id_is_the_prim_name(self):
        self.assertEqual(_frame_id(UR_BASE), 'ur_base_link')
        self.assertEqual(_frame_id(ARM_OPTICAL), 'restock_camera_arm')
        self.assertEqual(_frame_id('/MobileRobot/gripper_pad_tcp'), 'gripper_pad_tcp')


class TfPayload(unittest.TestCase):
    def test_carries_restock_camera_arm_as_child_of_ur_base_link(self):
        children = [('/MobileRobot/gripper_pad_tcp', (0.1, 0.0, 0.0), (0, 0, 0, 1.0)),
                    ('/MobileRobot/gripper_tcp', (0.2, 0.0, 0.0), (0, 0, 0, 1.0)),
                    (ARM_OPTICAL, (0.3, 0.0, 0.4), (0, 0, 0, 1.0))]
        p = tf_payload(UR_BASE, children, 5.0)
        self.assertEqual(p['t'], 5.0)
        pairs = [(row[0], row[1]) for row in p['tf']]
        # every frame is published relative to the single arm-base parent, by prim name
        self.assertTrue(all(parent == 'ur_base_link' for parent, _ in pairs), pairs)
        self.assertIn(('ur_base_link', 'restock_camera_arm'), pairs)
        self.assertIn(('ur_base_link', 'gripper_pad_tcp'), pairs)

    def test_row_shape_is_parent_child_then_translation_then_xyzw_quat(self):
        p = tf_payload(UR_BASE, [(ARM_OPTICAL, (0.3, 0.1, 0.4), (0.0, 0.0, 0.7071, 0.7071))], 2.0)
        row = p['tf'][0]
        self.assertEqual(len(row), 9)
        self.assertEqual(row[:2], ['ur_base_link', 'restock_camera_arm'])
        self.assertEqual(row[2:5], [0.3, 0.1, 0.4])            # translation x,y,z
        self.assertEqual(row[5:], [0.0, 0.0, 0.7071, 0.7071])  # quat x,y,z,w


class FrameIdContractWithNvblox(unittest.TestCase):
    """nvblox looks up depth_frame_id <- global_frame; the relay must publish exactly those names.
    Verify the depth header frame_id == the relay _frame_id of the arm optical prim, and the relay
    parent == the nvblox global_frame (config.FRAME) 'ur_base_link' with no prim-path prefix."""

    def test_arm_depth_frame_id_equals_relay_child_frame_id(self):
        depth_frame_id = 'restock_camera_arm'                 # masked_depth.ArmDepthSubprocessBridge default
        # masked_depth builds the optical prim at camera_path + '/' + frame_id
        optical_path = '/MobileRobot/front_camera_frame/arm_camera/' + depth_frame_id
        self.assertEqual(_frame_id(optical_path), depth_frame_id)   # exact string match, no prefix

    def test_relay_parent_equals_nvblox_global_frame_with_no_prefix(self):
        from twin.restock_host.config import FRAME
        self.assertEqual(FRAME, 'ur_base_link')
        self.assertEqual(_frame_id('/MobileRobot/ur_base_link'), FRAME)


class OneSharedStampConversion(unittest.TestCase):
    """tf2 accepts an exact-newest match only at NANOSECOND resolution, so the masked depth and the
    /tf for a tick must convert the identical world.current_time float to the IDENTICAL (sec,nanosec).
    Both go through control_ipc._secs_to_time now; lock that no 1 ns future skew can appear."""

    def test_depth_and_relay_stamps_land_on_identical_nanoseconds(self):
        from twin.restock.control_ipc import _secs_to_time
        for t in (0.0, 0.0166667, 0.2833333481103182, 1.7500000912696123, 5.2):
            ds, dn = _secs_to_time(t)          # depth path (masked_depth header)
            ts, tn = _secs_to_time(t)          # /tf + /clock + /joint_states path (control_ipc serve)
            self.assertEqual(ds * 10**9 + dn, ts * 10**9 + tn)
            self.assertTrue(0 <= dn < 10**9)

    def test_masked_depth_uses_the_shared_helper_not_an_inline_round(self):
        src = (Path(__file__).resolve().parents[1] / 'restock' / 'masked_depth.py').read_text()
        self.assertIn('from .control_ipc import _secs_to_time', src)
        self.assertNotIn("int(round((stamp_s - sec)", src)   # the old divergent inline is gone


if __name__ == '__main__':
    unittest.main()
