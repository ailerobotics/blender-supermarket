#!/usr/bin/env python3
"""Separate-process dynamic /tf publisher with a controllable stamp lag.

Reproduces the host-DDS-relay TF race for the nvblox depth path: the integration mission
relays /tf and masked depth through two independent host-DDS processes, so a depth frame
can reach nvblox before the matching /tf for its stamp. This child publishes the
ur_base_link -> restock_camera_arm transform on /tf (dynamic, NOT /tf_static) from its own
process, stamping every transform ``lag_ms`` behind the current simulation clock. The
newest transform available to tf2 is therefore always ``lag`` old, so a lookup at a depth
frame's stamp cannot resolve until the clock advances ``lag`` further -- exactly the
"depth before its /tf" buffer miss, with ``lag`` the controlled independent-relay delay.

Runs until SIGTERM. Sim time comes from /clock (the probe drives it); domain 88 isolated.
"""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-frame', default='ur_base_link')
    p.add_argument('--child-frame', default='restock_camera_arm')
    p.add_argument('--lag-ms', type=float, required=True)
    p.add_argument('--z', type=float, default=2.0)         # camera height above ur_base_link
    p.add_argument('--rate-hz', type=float, default=50.0)  # dense enough to bracket depth stamps
    a = p.parse_args()
    lag_ns = int(a.lag_ms * 1e6)

    import rclpy
    from rclpy.parameter import Parameter
    from rosgraph_msgs.msg import Clock
    from geometry_msgs.msg import TransformStamped
    from builtin_interfaces.msg import Time as RTime
    from tf2_ros import TransformBroadcaster

    rclpy.init()
    node = rclpy.create_node('tf_lag_publisher', namespace='/restock_sim',
                             parameter_overrides=[Parameter('use_sim_time', value=True)])
    latest = {'ns': 0}
    node.create_subscription(Clock, '/clock',
                             lambda m: latest.__setitem__('ns', m.clock.sec * 1_000_000_000 + m.clock.nanosec), 10)
    bc = TransformBroadcaster(node)

    def tick():
        now_ns = latest['ns']
        if now_ns <= 0:
            return
        stamp_ns = max(0, now_ns - lag_ns)  # publish transforms lagging the clock by `lag`
        tf = TransformStamped()
        tf.header.stamp = RTime(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
        tf.header.frame_id, tf.child_frame_id = a.parent_frame, a.child_frame
        tf.transform.translation.z = a.z
        tf.transform.rotation.x, tf.transform.rotation.w = 1.0, 0.0  # 180 deg about X -> look down
        bc.sendTransform(tf)

    node.create_timer(1.0 / a.rate_hz, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
