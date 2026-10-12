"""Bounded graph-only endpoint probe. No motion or mapper service is called."""
import time

from .config import ENDPOINTS, NS


def missing_endpoints(services, actions):
    """Accept graph name/type pairs, requiring exact action versus service kind."""
    service_map, action_map = dict(services), dict(actions)
    return [NS + '/' + name + ' (' + kind + ')'
            for name, kind in ENDPOINTS.items()
            if kind not in (action_map if '/action/' in kind else service_map).get(NS + '/' + name, [])]


def probe(timeout=5.0):
    import rclpy
    from rclpy.action import get_action_server_names_and_types_by_node
    rclpy.init()
    node = rclpy.create_node('host_readiness', namespace=NS)
    try:
        deadline = time.monotonic() + timeout
        while True:
            actions = []
            for name, namespace in node.get_node_names_and_namespaces():
                actions.extend(get_action_server_names_and_types_by_node(node, name, namespace))
            missing = missing_endpoints(node.get_service_names_and_types(), actions)
            if not missing or time.monotonic() >= deadline:
                return {'missing_endpoints': missing, 'runtime_verified': False,
                        'reason': 'Discovery only; require fresh measured joints/TF/depth, parked map acknowledgement and both attachment acknowledgements before motion'}
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0, deadline-time.monotonic())))
    finally:
        node.destroy_node()
        rclpy.shutdown()
