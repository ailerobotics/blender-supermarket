#!/usr/bin/env python3
"""Host-Jazzy bridge for an Isaac process using its own Python/ROS ABI.

Domain 88, loopback only. Inputs are simulator wheel odometry, base-frame scan,
and timestamped measured rig state. Only safe_cmd_vel may drive simulated wheels.
No sensor, localization or robot pose is synthesized by this process.
"""
import argparse
import json
import math
from pathlib import Path
import time

from twin.navigation import GridMap,Odometry
from twin.navigation.articulated import ArticulatedController
from twin.navigation.nav2 import ArticulatedNav2Adapter,require_isolation
from twin.navigation.sensors import Scan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--integration-scenario',type=Path,help='Include staging support obstacle')
    p.add_argument('--initial-pose',nargs=3,type=float,required=True,
                   help='Declared deployment prior for AMCL, not live world feedback')
    args=p.parse_args()
    require_isolation()
    from twin.navigation.scenario import environment_grid
    scenario=json.loads(args.integration_scenario.read_text()) if args.integration_scenario else None
    grid=environment_grid(json.loads(args.environment.read_text()),scenario)
    controller=ArticulatedController(grid)
    adapter=ArticulatedNav2Adapter(controller)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry as RosOdometry
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float64MultiArray,String
    node=adapter.node
    velocity=node.create_publisher(Twist,'safe_cmd_vel',10)
    status=node.create_publisher(String,'navigation_status',10)
    latest={};last_wall={};pending_goal=[]

    def receive(name,msg):
        latest[name]=msg;last_wall[name]=time.monotonic()

    for topic,msgtype,key in [('encoder_odom',RosOdometry,'odom'),
                              ('scan_input',LaserScan,'scan'),
                              ('measured_rig_state',Float64MultiArray,'state')]:
        node.create_subscription(msgtype,topic,lambda msg,k=key:receive(k,msg),qos_profile_sensor_data)

    def goal(msg):
        try:
            data=json.loads(msg.data)
            if len(data['pose'])!=3 or not all(math.isfinite(v) for v in data['pose']):
                raise ValueError('finite map pose required')
            if data.get('coupled') is not True:
                raise ValueError('this bridge handles coupled towing')
            pending_goal[:]=[data]
        except (ValueError,KeyError,TypeError) as exc:
            status.publish(String(data=json.dumps({'status':'failed','reason':str(exc)})))
    node.create_subscription(String,'navigation_goal',goal,10)
    last_stamp=None;last_status=None;initializations=0;last_initialization=-math.inf

    def tick():
        nonlocal last_stamp,last_status,initializations,last_initialization
        command=Twist()
        result={'status':'running','reason':'missing_or_stale_sensor_input'}
        try:
            if any(k not in latest or time.monotonic()-last_wall[k]>.3 for k in ('odom','scan','state')):
                return
            om,sm=latest['odom'],latest['scan']
            stamp=om.header.stamp.sec+om.header.stamp.nanosec*1e-9
            if stamp==last_stamp:
                result['reason']='waiting_new_encoder_sample'
                return
            dt=.05 if last_stamp is None else stamp-last_stamp
            last_stamp=stamp
            if om.header.frame_id!='odom' or om.child_frame_id!='base_link' or sm.header.frame_id!='base_link':
                raise ValueError('sensor frame mismatch')
            q=om.pose.pose.orientation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            odom=Odometry((om.pose.pose.position.x,om.pose.pose.position.y,yaw),stamp,
                           om.twist.twist.linear.x,om.twist.twist.angular.z,'wheel_odometry')
            scan=Scan(sm.header.stamp.sec+sm.header.stamp.nanosec*1e-9,tuple(sm.ranges),
                      sm.angle_min,sm.angle_increment,sm.range_max)
            state=latest['state'].data  # [sim_stamp, coupled(0/1), arm_transport(0/1), articulation]
            if (len(state)!=4 or not all(math.isfinite(v) for v in state)
                    or not 0<=stamp-state[0]<=.1 or state[1] not in (0,1) or state[2] not in (0,1)):
                raise ValueError('invalid or stale measured rig state')
            adapter.publish_sensors(odom,scan)
            if not math.isfinite(adapter.localization_stamp) and initializations<5 and stamp-last_initialization>1:
                adapter.initialize_pose(args.initial_pose)
                initializations+=1;last_initialization=stamp
            if pending_goal and adapter.goal_future is None and adapter.goal_handle is None:
                data=pending_goal.pop(0);adapter.start(data['pose'],True)
            if controller.goal is None:
                result={'status':'running','reason':'waiting_goal'}
                return
            decision=adapter.step(dt,odom,scan,coupled=bool(state[1]),
                                   arm_transport=bool(state[2]),articulation=state[3])
            command.linear.x=float(decision.linear);command.angular.z=float(decision.angular)
            result={'status':decision.status,'reason':decision.reason}
        except (ValueError,RuntimeError,TypeError) as exc:
            result={'status':'failed','reason':str(exc)}
        finally:
            velocity.publish(command)
            if result!=last_status:
                status.publish(String(data=json.dumps(result)));last_status=result

    next_tick=time.monotonic()
    try:
        while adapter.context.ok():
            adapter.executor.spin_once(timeout_sec=.01)
            if time.monotonic()>=next_tick:
                tick();next_tick=time.monotonic()+.05
    except KeyboardInterrupt:
        pass
    finally:
        velocity.publish(Twist());adapter.close()


if __name__=='__main__':
    main()
