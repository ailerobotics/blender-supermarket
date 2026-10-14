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


if __name__ == '__main__':
    unittest.main()
