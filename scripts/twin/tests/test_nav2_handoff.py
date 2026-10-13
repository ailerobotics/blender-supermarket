"""ROS-message contract fixtures; no simulator, ROS server or live node is used."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import tempfile

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

    def test_required_map_rejected_before_path_submission(self):
        b=self.bridge();b.safe.require_temporal_map=True
        for supplied,reason in ((None,'missing_navigation_map'),(object(),'invalid_navigation_map')):
            result=b.step(.05,b.odom,b.scan,coupled=True,arm_transport=True,temporal_map=supplied)
            self.assertEqual(result.reason,reason)
            self.assertEqual(result.velocity,(0,0));self.assertFalse(b.sent)

    def test_replan_waits_for_old_action_result_before_new_path(self):
        b=self.bridge();b.pending=False;b.replan_requested=True
        canceled=[]
        b.goal_handle=NS(cancel_goal_async=lambda:canceled.append(True))
        result=b.step(.05,b.odom,b.scan,coupled=True,arm_transport=True)
        self.assertEqual(result.reason,'waiting_old_path_cancellation')
        self.assertEqual(result.velocity,(0,0));self.assertEqual(canceled,[True])
        b.step(.05,b.odom,b.scan,coupled=True,arm_transport=True)
        self.assertEqual(canceled,[True]);self.assertFalse(b.sent)
        b._finished(NS(result=lambda:NS(status=5)))
        self.assertIsNone(b.outcome)
        self.assertEqual(b.step(.05,b.odom,b.scan,coupled=True,arm_transport=True).reason,'planning_stop')
        self.assertEqual(b.step(.05,b.odom,b.scan,coupled=True,arm_transport=True).reason,'articulated_path_submitted')
        self.assertEqual(len(b.sent),1)

    def test_official_nvblox_plugin_config_keeps_unknown_and_no_reverse(self):
        import yaml
        from twin.navigation.nav2 import write_assets
        from twin.mobile import MobileSpec
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as output:
            grid=GridMap((-2,-2,2,2))
            _,params=write_assets(output,grid,ArticulatedController(grid).footprint,
                articulated_spec=MobileSpec(),nvblox_topic='/restock_sim/navigation/nvblox_node/static_map_slice')
            config=yaml.safe_load(params.read_text())
            for name in ('local_costmap','global_costmap'):
                p=config[name][name]['ros__parameters']
                self.assertTrue(p['track_unknown_space'])
                self.assertEqual(p['nvblox_layer']['plugin'],'nvblox::nav2::NvbloxCostmapLayer')
                self.assertEqual(p['nvblox_layer']['nvblox_map_slice_topic'],
                                 '/restock_sim/navigation/nvblox_node/static_map_slice')
            self.assertFalse(config['planner_server']['ros__parameters']['GridBased']['allow_unknown'])
            self.assertFalse(config['controller_server']['ros__parameters']['FollowPath']['allow_reversing'])


if __name__=='__main__': unittest.main()
