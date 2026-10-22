"""ROS Jazzy/Nav2 route for Isaac 5.1, imported only in an isolated simulation.

Nav2 owns AMCL, Smac Hybrid planning and regulated pure pursuit. Its cmd_vel is
an untrusted proposal; only step() results may be sent to the mobile base.
"""
import math
import os
from dataclasses import replace
from pathlib import Path
from .core import Odometry, Result, wrap
from .sensors import observation


def require_isolation():
    if os.environ.get('ROS_DOMAIN_ID') != '88' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise RuntimeError('navigation requires ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1')
    if os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE','LOCALHOST') != 'LOCALHOST':
        raise RuntimeError('ROS discovery must be loopback only')


def write_assets(directory,grid,footprint,*,coupled=True,template=None,articulated_spec=None,
                 nvblox_topic=None):
    """Generate map, Nav2 params and a replan tree; no installs or simulator boot."""
    import yaml
    if template is None:
        from ament_index_python.packages import get_package_share_directory
        template=Path(get_package_share_directory('nav2_bringup'))/'params/nav2_params.yaml'
    config=yaml.safe_load(Path(template).read_text())
    out=Path(directory)
    out.mkdir(parents=True,exist_ok=True)
    x0,y0,x1,y1=grid.bounds
    width,height=math.ceil((x1-x0)/grid.resolution),math.ceil((y1-y0)/grid.resolution)
    pixels=bytearray()
    for j in reversed(range(height)):
        for i in range(width):
            p=x0+(i+0.5)*grid.resolution,y0+(j+0.5)*grid.resolution
            # Mark all cells touching obstacles occupied, rather than aliasing thin walls.
            pixels.append(254 if grid.free(p,grid.resolution/math.sqrt(2)) else 0)
    image=out/'map.pgm'
    image.write_bytes(f'P5\n{width} {height}\n255\n'.encode()+pixels)
    mapfile=out/'map.yaml'
    mapfile.write_text(yaml.safe_dump(dict(image=str(image.resolve()),resolution=grid.resolution,
        origin=[x0,y0,0.0],negate=0,occupied_thresh=0.65,free_thresh=0.196,mode='trinary')))
    def params(name):
        return config.setdefault(name,{}).setdefault('ros__parameters',{'use_sim_time':True})
    for name,value in config.items():
        if 'ros__parameters' in value:
            value['ros__parameters']['use_sim_time']=True
    params('amcl').update(base_frame_id='base_link',odom_frame_id='odom',global_frame_id='map',
                          scan_topic='scan',tf_broadcast=True,set_initial_pose=False,
                          update_min_d=0.01,update_min_a=0.01)
    params('map_server')['yaml_filename']=str(mapfile.resolve())
    params('planner_server').update(planner_plugins=['GridBased'],GridBased={
        'plugin':'nav2_smac_planner::SmacPlannerHybrid','motion_model_for_search':'DUBIN',
        'minimum_turning_radius':max(0.4,footprint.min_turn_radius),
        'angle_quantization_bins':72,'allow_unknown':False,'tolerance':0.08})
    params('controller_server').update(controller_frequency=20.0,controller_plugins=['FollowPath'],
        goal_checker_plugins=['goal_checker'],goal_checker={
        'plugin':'nav2_controller::SimpleGoalChecker','xy_goal_tolerance':0.08,
        'yaw_goal_tolerance':0.08,'stateful':False},FollowPath={
        'plugin':'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
        'desired_linear_vel':0.25,'lookahead_dist':0.6,'use_rotate_to_heading':False,
        'allow_reversing':False,'use_collision_detection':True,
        'max_allowed_time_to_collision_up_to_carrot':2.0})
    radius=footprint.radius(coupled)+0.10  # localization allowance also checked at runtime
    if articulated_spec is not None:
        # Global costmap is advisory only for the FollowPath route. The actual
        # base/cart path and every actuator command pass the articulated checker.
        radius=(articulated_spec.transport_radius+0.05+0.02)/math.cos(math.pi/16)
        params('controller_server')['FollowPath'].update(
            lookahead_dist=0.10,min_lookahead_dist=0.10,max_lookahead_dist=0.10,
            use_velocity_scaled_lookahead_dist=False,use_collision_detection=False,
            use_cost_regulated_linear_velocity_scaling=False)
    for name in ('global_costmap','local_costmap'):
        p=config[name][name]['ros__parameters']
        p.update(use_sim_time=True,robot_base_frame='base_link',robot_radius=radius,
                 resolution=grid.resolution,footprint_padding=0.0,track_unknown_space=True)
        p.pop('footprint',None)
        p['inflation_layer']['inflation_radius']=radius+0.35
        if name=='local_costmap':
            p.update(width=max(8,math.ceil(radius*4)),height=max(8,math.ceil(radius*4)))
            p['plugins']=['obstacle_layer','inflation_layer']
            p.pop('voxel_layer',None)
        p['obstacle_layer']={'plugin':'nav2_costmap_2d::ObstacleLayer',
            'observation_sources':'scan','scan':{'topic':'scan','data_type':'LaserScan',
            'clearing':True,'marking':True,'raytrace_max_range':10.0,'obstacle_max_range':10.0}}
        if 'obstacle_layer' not in p['plugins']:
            p['plugins'].insert(-1,'obstacle_layer')
        if nvblox_topic is not None:
            if not nvblox_topic.startswith('/restock_sim/navigation/'):
                raise ValueError('navigation-scoped nvblox topic required')
            p['nvblox_layer']={'plugin':'nvblox::nav2::NvbloxCostmapLayer',
                'nvblox_map_slice_topic':nvblox_topic,
                'nav2_costmap_global_frame':p['global_frame'],
                'convert_to_binary_costmap':True}
            p['plugins'].insert(-1,'nvblox_layer')
            # The raw-slice actuator gate also checks unknown: layer maximum
            # merging must never promote a static-map free cell over unknown.
    if 'velocity_smoother' in config:
        params('velocity_smoother').update(max_velocity=[0.25,0.0,0.35],
            min_velocity=[0.0,0.0,-0.35],max_accel=[0.4,0.0,0.5],
            max_decel=[-0.4,0.0,-0.5],velocity_timeout=0.3)
    if 'collision_monitor' in config:
        p=params('collision_monitor')
        p.update(base_frame_id='base_link',source_timeout=0.3)
        if 'FootprintApproach' in p:
            p['FootprintApproach']['footprint_topic']='local_costmap/published_footprint'
            p['FootprintApproach']['min_points']=1
    tree=out/'navigate.xml'
    tree.write_text('''<root BTCPP_format="4" main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence>
      <RateController hz="1.0">
        <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
      </RateController>
      <FollowPath path="{path}" controller_id="FollowPath" goal_checker_id="goal_checker"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
''')
    params('bt_navigator')['default_nav_to_pose_bt_xml']=str(tree.resolve())
    params('bt_navigator')['robot_base_frame']='base_link'
    result=out/'nav2.yaml'
    result.write_text(yaml.safe_dump(config))
    return mapfile,result


class Nav2Adapter:
    """Call publish_sensors after each physics tick, then step after ROS spin.

    Wheel odometry is unlocalized, scan ranges are physical observations. AMCL is
    the sole map->odom publisher. The base consumes returned velocities only.
    """
    def __init__(self,safety_controller):
        require_isolation()
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.qos import qos_profile_sensor_data
        from geometry_msgs.msg import Twist,PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry as RosOdom
        from sensor_msgs.msg import LaserScan
        from rosgraph_msgs.msg import Clock
        from nav2_msgs.action import NavigateToPose
        from tf2_ros import Buffer,TransformListener,TransformBroadcaster
        self.rclpy=rclpy
        self.context=rclpy.context.Context()
        rclpy.init(context=self.context)
        self.node=rclpy.create_node('restock_navigation_adapter',namespace='restock',context=self.context,
            cli_args=['--ros-args','-r','/tf:=/restock/tf','-r','/tf_static:=/restock/tf_static'])
        from rclpy.executors import SingleThreadedExecutor
        self.executor=SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.node.set_parameters([rclpy.parameter.Parameter('use_sim_time',value=True)])
        self.safe=safety_controller
        self.command=(0.0,0.0)
        self.command_stamp=-math.inf
        self.localization_stamp=-math.inf
        self.localization_uncertainty=math.inf
        self.minimum_localization_uncertainty=0.10
        self.goal_handle=None
        self.goal_future=None
        self.outcome=None
        self.now=0.0
        self.odom_pub=self.node.create_publisher(RosOdom,'odom',10)
        self.scan_pub=self.node.create_publisher(LaserScan,'scan',qos_profile_sensor_data)
        self.clock_pub=self.node.create_publisher(Clock,'/clock',10)
        self.initial_pub=self.node.create_publisher(PoseWithCovarianceStamped,'initialpose',10)
        self.node.create_subscription(Twist,'cmd_vel',self._command,10)
        self.node.create_subscription(PoseWithCovarianceStamped,'amcl_pose',self._localized,10)
        self.buffer=Buffer(node=self.node)
        self.listener=TransformListener(self.buffer,self.node)
        self.broadcaster=TransformBroadcaster(self.node)
        self.client=ActionClient(self.node,NavigateToPose,'navigate_to_pose')

    def _command(self,msg):
        self.command=(msg.linear.x,msg.angular.z)
        self.command_stamp=self.now

    def _localized(self,msg):
        self.localization_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.localization_uncertainty=2*math.sqrt(max(0.0,msg.pose.covariance[0],msg.pose.covariance[7]))

    @staticmethod
    def _pose(message,pose):
        message.position.x,message.position.y=float(pose[0]),float(pose[1])
        message.orientation.z=math.sin(pose[2]/2)
        message.orientation.w=math.cos(pose[2]/2)

    def _safe_start(self,goal,coupled,goal_articulation,strict_xy_arrival=False):
        # Thread the per-goal articulation tolerance + the grasp-critical strict XY arrival to the safety
        # controller so the NavigationBridge start(goal, coupled, goal_articulation=, strict_xy_arrival=)
        # contract holds. Nav2's own action governs arrival on the live route; this only tightens the
        # safety-fallback controller's arrival so a grasp-critical leg is consistent across both.
        kw={}
        if goal_articulation is not None:
            kw['goal_articulation']=goal_articulation
        if strict_xy_arrival:
            kw['strict_xy_arrival']=True
        self.safe.start(goal,coupled,**kw)

    def initialize_pose(self,pose):
        from geometry_msgs.msg import PoseWithCovarianceStamped
        msg=PoseWithCovarianceStamped()
        msg.header.frame_id='map'
        msg.header.stamp=self.node.get_clock().now().to_msg()
        self._pose(msg.pose.pose,pose)
        msg.pose.covariance[0]=msg.pose.covariance[7]=0.04
        msg.pose.covariance[35]=0.02
        self.initial_pub.publish(msg)

    def publish_sensors(self,odom,scan):
        from nav_msgs.msg import Odometry as RosOdom
        from sensor_msgs.msg import LaserScan
        from rosgraph_msgs.msg import Clock
        from geometry_msgs.msg import TransformStamped
        if odom.source != 'wheel_odometry' or scan.source != 'simulated_lidar':
            raise ValueError('ROS route requires measured wheel odometry and simulated lidar')
        if abs(odom.stamp-scan.stamp)>0.05:
            raise ValueError('scan/encoder time mismatch')
        self.now=odom.stamp
        stamp=self.rclpy.time.Time(seconds=odom.stamp).to_msg()
        clock=Clock(); clock.clock=stamp
        self.clock_pub.publish(clock)
        msg=RosOdom(); msg.header.stamp=stamp; msg.header.frame_id='odom'; msg.child_frame_id='base_link'
        self._pose(msg.pose.pose,odom.pose)
        msg.twist.twist.linear.x=odom.linear; msg.twist.twist.angular.z=odom.angular
        msg.pose.covariance[0]=msg.pose.covariance[7]=0.01; msg.pose.covariance[35]=0.02
        self.odom_pub.publish(msg)
        tf=TransformStamped(); tf.header=msg.header; tf.child_frame_id='base_link'
        tf.transform.translation.x=float(odom.pose[0]); tf.transform.translation.y=float(odom.pose[1])
        tf.transform.rotation=msg.pose.pose.orientation
        self.broadcaster.sendTransform(tf)
        msg=LaserScan(); msg.header.stamp=self.rclpy.time.Time(seconds=scan.stamp).to_msg(); msg.header.frame_id='base_link'
        msg.angle_min=float(scan.angle_min); msg.angle_increment=float(scan.angle_increment)
        msg.angle_max=float(scan.angle_min+(len(scan.ranges)-1)*scan.angle_increment)
        msg.range_min=0.001; msg.range_max=float(scan.range_max); msg.ranges=list(scan.ranges)
        self.scan_pub.publish(msg)
        self.executor.spin_once(timeout_sec=0.0)

    def start(self,goal,coupled,*,goal_articulation=None,strict_xy_arrival=False):
        from nav2_msgs.action import NavigateToPose
        if self.goal_future is not None or self.goal_handle is not None:
            raise RuntimeError('cancel previous goal and wait for its result before starting another')
        self._safe_start(goal,coupled,goal_articulation,strict_xy_arrival)
        self.outcome=None; self.command=(0.0,0.0); self.command_stamp=-math.inf
        if not self.client.server_is_ready():
            self.outcome='nav2_unavailable'
            return
        msg=NavigateToPose.Goal(); msg.pose.header.frame_id='map'
        msg.pose.header.stamp=self.node.get_clock().now().to_msg()
        self._pose(msg.pose.pose,goal)
        self.goal_future=self.client.send_goal_async(msg)
        self.goal_future.add_done_callback(self._accepted)

    def _accepted(self,future):
        self.goal_future=None
        self.goal_handle=future.result()
        if not self.goal_handle.accepted:
            self.goal_handle=None; self.outcome='goal_rejected'
        else:
            self.goal_handle.get_result_async().add_done_callback(self._finished)

    def _finished(self,future):
        self.outcome='arrived' if future.result().status==4 else 'nav2_failed'
        self.goal_handle=None
        self.command=(0.0,0.0)

    def cancel(self):
        self.command=(0.0,0.0)
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        return self.safe.cancel()

    def step(self,dt,odometry,scan,*,coupled,arm_transport,articulation=0.0,temporal_map=None):
        if not math.isfinite(dt) or dt <= 0 or dt > self.safe.stale_after:
            return Result(reason='invalid_control_period')
        self.executor.spin_once(timeout_sec=0.0)
        if self.safe.goal is None:
            return Result(status='failed',reason='no_goal')
        if self.safe.terminal:
            return self.safe.terminal
        self.safe.elapsed+=dt
        if self.safe.elapsed>self.safe.goal_timeout:
            self.cancel()
            return Result(status='failed',reason='goal_timeout')
        if self.outcome not in (None,'arrived'):
            self.safe.terminal=Result(status='failed',reason=self.outcome)
            return self.safe.terminal
        try:
            tf=self.buffer.lookup_transform('map','base_link',self.rclpy.time.Time())
            q=tf.transform.rotation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            stamp=tf.header.stamp.sec+tf.header.stamp.nanosec*1e-9
            pose=(tf.transform.translation.x,tf.transform.translation.y,yaw)
            if self.now-self.localization_stamp>1.0 or self.localization_uncertainty>0.25:
                raise ValueError('localization missing or uncertain')
            odom=Odometry(pose,stamp,odometry.linear,odometry.angular,
                          position_uncertainty=max(self.minimum_localization_uncertainty,self.localization_uncertainty))
            obs=observation(scan,odom,coupled=coupled,arm_transport=arm_transport,articulation=articulation)
            if temporal_map is not None:obs=replace(obs,temporal_map=temporal_map)
        except Exception:
            return self.safe._wait(dt,'localization_unavailable')
        reason=self.safe._inputs(self.now,odom,obs)
        if reason:
            return self.safe._wait(dt,reason)
        if self.outcome=='arrived':
            if (math.dist(pose[:2],self.safe.goal[:2])<=0.08 and abs(wrap(pose[2]-self.safe.goal[2]))<=0.08
                    and abs(odom.linear)<=0.02 and abs(odom.angular)<=0.03):
                clear=self.safe.gate_velocity(self.now,odom,obs,0.0,0.0)
                if clear.reason!='tracking':
                    return self.safe._wait(dt,clear.reason)
                self.safe.terminal=Result(status='succeeded',reason='arrived_stopped')
                return self.safe.terminal
            return self.safe._wait(dt,'arrival_outside_tolerance')
        if self.now-self.command_stamp>self.safe.stale_after:
            return self.safe._wait(dt,'stale_nav2_command')
        result=self.safe.gate_velocity(self.now,odom,obs,*self.command)
        if result.reason!='tracking':
            return self.safe._wait(dt,result.reason)
        self.safe.blocked=0.0
        return result

    def close(self):
        self.cancel()
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.shutdown()


class ArticulatedNav2Adapter(Nav2Adapter):
    """Use Nav2 FollowPath on a fully validated articulated path, never NavigateToPose.

    AMCL/TF/scan and the velocity gate are the same as Nav2Adapter. Construct with
    ArticulatedController and generate assets with articulated_spec=controller.spec.
    Missing localization stops before planning. The caller applies every returned
    zero and keeps a watchdog; raw ROS cmd_vel must never connect to the wheels.
    """
    def __init__(self,safety_controller):
        from .articulated import ArticulatedController
        if not isinstance(safety_controller,ArticulatedController):
            raise TypeError('ArticulatedController required')
        super().__init__(safety_controller)
        from rclpy.action import ActionClient
        from nav2_msgs.action import FollowPath
        from geometry_msgs.msg import Polygon
        self.client.destroy()
        self.client=ActionClient(self.node,FollowPath,'follow_path')
        self.footprint_pub=self.node.create_publisher(Polygon,'local_costmap/footprint',10)
        self.minimum_localization_uncertainty=0.02
        self.pending=False
        self.stop_sent=False

    def start(self,goal,coupled,*,goal_articulation=None,strict_xy_arrival=False):
        if self.goal_future is not None or self.goal_handle is not None:
            raise RuntimeError('wait for previous action result before starting another')
        if not coupled:
            raise ValueError('use Nav2Adapter for an uncoupled mission')
        self._safe_start(goal,coupled,goal_articulation,strict_xy_arrival)
        self.outcome=None
        self.command=(0.0,0.0)
        self.command_stamp=-math.inf
        self.pending=True
        self.stop_sent=False
        self.replan_requested=False
        self.replan_cancel_sent=False

    def _finished(self,future):
        super()._finished(future)
        if self.replan_requested:self.outcome=None

    def _state(self,raw,scan,coupled,arm_transport,articulation):
        tf=self.buffer.lookup_transform('map','base_link',self.rclpy.time.Time())
        q=tf.transform.rotation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        stamp=tf.header.stamp.sec+tf.header.stamp.nanosec*1e-9
        if self.now-self.localization_stamp>1.0 or self.localization_uncertainty>0.25:
            raise ValueError('localization missing or uncertain')
        odom=Odometry((tf.transform.translation.x,tf.transform.translation.y,yaw),
                       stamp,raw.linear,raw.angular,
                       position_uncertainty=max(.02,self.localization_uncertainty))
        return odom,observation(scan,odom,coupled=coupled,arm_transport=arm_transport,
                                 articulation=articulation)

    def _publish_footprint(self,odom,obs):
        from geometry_msgs.msg import Polygon,Point32
        from .articulated import TrainState
        from twin.mobile import occupied_envelope
        s=TrainState(*odom.pose,wrap(odom.pose[2]-obs.articulation))
        points=occupied_envelope(*s.bodies(self.safe.spec),self.safe.spec,
                                  margin=.05+max(.10 if obs.temporal_map is not None else .02,
                                                 odom.position_uncertainty))
        msg=Polygon()
        for x,y in points:
            dx,dy=x-s.x,y-s.y
            msg.points.append(Point32(x=float(dx*math.cos(s.yaw)+dy*math.sin(s.yaw)),
                                      y=float(-dx*math.sin(s.yaw)+dy*math.cos(s.yaw)),z=0.0))
        self.footprint_pub.publish(msg)

    def step(self,dt,odometry,scan,*,coupled,arm_transport,articulation=0.0,temporal_map=None):
        if not math.isfinite(dt) or not 0<dt<=self.safe.stale_after:
            return Result(reason='invalid_control_period')
        if self.safe.terminal:
            return self.safe.terminal
        self.executor.spin_once(timeout_sec=0.0)
        if self.safe.require_temporal_map and temporal_map is None:
            return self.safe._wait(dt,'missing_navigation_map')
        if temporal_map is not None:
            from .temporal import TemporalMap
            if not isinstance(temporal_map,TemporalMap):
                return self.safe._wait(dt,'invalid_navigation_map')
        if self.replan_requested:
            self.safe.elapsed+=dt
            if self.safe.elapsed>self.safe.goal_timeout:return self.cancel()
            self.command=(0.,0.)
            if self.goal_future is not None:
                return Result(reason='waiting_old_path_acceptance')
            if self.goal_handle is not None:
                if not self.replan_cancel_sent:
                    self.goal_handle.cancel_goal_async();self.replan_cancel_sent=True
                return Result(reason='waiting_old_path_cancellation')
            self.pending=True;self.stop_sent=False;self.replan_requested=False
            self.replan_cancel_sent=False;self.outcome=None;self.safe.route=None
        try:
            odom,obs=self._state(odometry,scan,coupled,arm_transport,articulation)
            if temporal_map is not None:obs=replace(obs,temporal_map=temporal_map)
            reason=self.safe._inputs(self.now,odom,obs)
            if reason:
                return self.safe._wait(dt,reason)
            self._publish_footprint(odom,obs)
        except Exception:
            return self.safe._wait(dt,'localization_unavailable')
        if self.pending:
            self.safe.elapsed+=dt
            if self.safe.elapsed>self.safe.goal_timeout:
                return self.cancel()
            if not self.stop_sent:
                self.stop_sent=True
                return Result(reason='planning_stop')
            if abs(odom.linear)>.02 or abs(odom.angular)>.03:
                return Result(reason='waiting_stopped_for_plan')
            if not self.client.server_is_ready():
                return self.safe._wait(dt,'nav2_unavailable')
            from .articulated import TrainState
            state=TrainState(*odom.pose,wrap(odom.pose[2]-obs.articulation))
            if temporal_map is not None:
                reason=temporal_map.readiness(self.now)
                if reason:return self.safe._wait(dt,reason)
            route=self.safe.plan_observed_route(state,obs,odom.position_uncertainty)
            if not route.states:
                return self.safe._wait(dt,route.reason)
            from nav2_msgs.action import FollowPath
            from geometry_msgs.msg import PoseStamped
            msg=FollowPath.Goal()
            msg.path.header.frame_id='map'
            msg.path.header.stamp=self.node.get_clock().now().to_msg()
            for s in route.states:
                pose=PoseStamped();pose.header=msg.path.header
                self._pose(pose.pose,s.pose)
                msg.path.poses.append(pose)
            msg.controller_id='FollowPath'
            msg.goal_checker_id='goal_checker'
            self.safe.route=route
            self.pending=False
            self.goal_future=self.client.send_goal_async(msg)
            self.goal_future.add_done_callback(self._accepted)
            return Result(reason='articulated_path_submitted')
        result=super().step(dt,odometry,scan,coupled=coupled,
                            arm_transport=arm_transport,articulation=articulation,temporal_map=temporal_map)
        if temporal_map is not None and result.reason in ('occupied_navigation_cell','unknown_navigation_cell',
                                                           'outside_navigation_map','obstacle_stop'):
            self.replan_requested=True
            self.command=(0.,0.)
        return result
