"""ROS-message contract fixtures; no simulator, ROS server or live node is used."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
try:
    from builtin_interfaces.msg import Time
    from nav2_msgs.action import FollowPath
except ImportError:
    Time=FollowPath=None

from twin.navigation import GridMap,Odometry
from twin.navigation.articulated import ArticulatedController
from twin.navigation.nav2 import ArticulatedNav2Adapter
from twin.navigation.sensors import Scan,observation


@unittest.skipUnless(FollowPath is not None,'requires task-owned Jazzy message packages')
class Nav2HandoffTests(unittest.TestCase):
    def bridge(self):
        b=object.__new__(ArticulatedNav2Adapter)
        b.safe=ArticulatedController(GridMap((-10,-10,10,10)))
        b.goal_future=b.goal_handle=None
        b.node=NS(get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:Time(sec=0,nanosec=100000000))))
        b.rclpy=NS(spin_once=lambda *a,**k:None,time=NS(Time=lambda:None))
        b.executor=NS(spin_once=lambda **k:None)
        b.now=b.localization_stamp=b.command_stamp=.1
        b.localization_uncertainty=b.minimum_localization_uncertainty=.02
        b.command=(.1,0)
        b.sent=[]
        def send(msg):
            b.sent.append(msg)
            return NS(add_done_callback=lambda cb:None)
        b.client=NS(server_is_ready=lambda:True,send_goal_async=send)
        b.odom=Odometry((0,0,0),.1,position_uncertainty=.02)
        b.scan=Scan(.1,(10.0,)*360,-math.pi,math.pi/180,10)
        obs=observation(b.scan,b.odom,coupled=True,arm_transport=True)
        b._state=lambda *a:(b.odom,obs)
        b.footprints=[]
        b._publish_footprint=lambda o,s:b.footprints.append((o,s))
        tf=NS(header=NS(stamp=NS(sec=0,nanosec=100000000)),
              transform=NS(rotation=NS(x=0,y=0,z=0,w=1),translation=NS(x=0,y=0)))
        b.buffer=NS(lookup_transform=lambda *a:tf)
        b.start((2,0,0),True)
        return b

    def test_checked_path_submitted_only_after_stop(self):
        b=self.bridge()
        result=b.step(.05,b.odom,b.scan,coupled=True,arm_transport=True)
        self.assertEqual(result.reason,'planning_stop');self.assertFalse(b.sent)
        result=b.step(.05,b.odom,b.scan,coupled=True,arm_transport=True)
        self.assertEqual(result.reason,'articulated_path_submitted')
        self.assertIsInstance(b.sent[0],FollowPath.Goal)
        self.assertEqual(b.sent[0].path.header.frame_id,'map')
        self.assertGreater(len(b.sent[0].path.poses),2)
        self.assertAlmostEqual(b.sent[0].path.poses[-1].pose.position.x,2,places=2)
        self.assertEqual(result.velocity,(0,0))

    def test_nav2_command_cannot_bypass_articulated_radius(self):
        b=self.bridge();b.pending=False;b.command=(.1,.3);b.command_stamp=.1
        result=b.step(.05,b.odom,b.scan,coupled=True,arm_transport=True)
        self.assertEqual(result.reason,'turn_radius')
        self.assertEqual(result.velocity,(0,0))


if __name__=='__main__': unittest.main()
