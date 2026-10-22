"""fm362: every WALL-clocked timeout/ack gate on the post-grasp mission path is sim-only env-overridable
so run-290 can widen the whole render-paced timing-gate family in one batch. Defaults byte-identical
when the env is unset; the fault/gap logging is unchanged. (Root fix later: sim-clock these gates.)"""
import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# (module path, attribute, default, env var, widened value)
GATES = [
    ('twin.restock.native_transport', 'ARM_HEARTBEAT_WALL_BACKSTOP_S', 30.0, 'RESTOCK_ARM_HEARTBEAT_WALL_BACKSTOP_S', 90.0),
    ('twin.restock.native_channel', '_HEARTBEAT_UNAVAILABLE_S', 10.0, 'RESTOCK_HEARTBEAT_UNAVAILABLE_S', 30.0),
    ('twin.restock.native_channel', '_STAGE_UNAVAILABLE_S', 3.0, 'RESTOCK_STAGE_UNAVAILABLE_S', 12.0),
    ('twin.restock_manipulation.ros_adapter', 'PLAN_TOTAL_WALL_CAP_S', 22.0, 'RESTOCK_PLAN_WALL_CAP_S', 66.0),
    ('twin.restock_manipulation.ros_adapter', 'RESPAWN_WAIT_CAP_S', 45.0, 'RESTOCK_RESPAWN_WAIT_CAP_S', 120.0),
    ('twin.restock_manipulation.ros_adapter', 'MOVE_WALL_TIMEOUT_S', 25.0, 'RESTOCK_MOVE_WALL_TIMEOUT_S', 75.0),
    ('twin.restock.station_host', 'STATION_PENDING_WALL_S', 2.0, 'RESTOCK_STATION_PENDING_WALL_S', 8.0),
    ('twin.restock_manipulation.isaac_sensors', 'SENSOR_WALL_BACKSTOP_S', 60.0, 'RESTOCK_SENSOR_WALL_BACKSTOP_S', 180.0),
    # fm364/run-293: the host /sensors stream wall backstop (RosTransferBridge default), the 3rd missed gate.
    ('twin.restock_manipulation.ros_adapter', 'SENSOR_STREAM_WALL_BACKSTOP_S', 30.0, 'RESTOCK_SENSOR_STREAM_WALL_BACKSTOP_S', 180.0),
]


def _value_with(modpath, attr, env):
    mod = importlib.import_module(modpath)
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        return getattr(importlib.reload(mod), attr)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        importlib.reload(mod)  # restore defaults for the rest of the suite


class WallGateOverrideTests(unittest.TestCase):
    def test_defaults_byte_identical_when_unset(self):
        for modpath, attr, default, env, _ in GATES:
            self.assertEqual(_value_with(modpath, attr, {env: None}), default,
                             f'{modpath}.{attr} default drifted')

    def test_env_widens_each_gate(self):
        for modpath, attr, _, env, wide in GATES:
            self.assertEqual(_value_with(modpath, attr, {env: str(wide)}), wide,
                             f'{modpath}.{attr} did not take the env widen')

    def test_clamped_sane(self):
        # An absurd value is clamped (not blindly trusted).
        v = _value_with('twin.restock_manipulation.ros_adapter', 'PLAN_TOTAL_WALL_CAP_S',
                        {'RESTOCK_PLAN_WALL_CAP_S': '99999'})
        self.assertLessEqual(v, 300.0)


# (attribute, default, env var, widened value) -- the sim-robust-collect-place-v1 COUNT caps on the
# plan path (ints, not wall seconds), so they ride their own test rather than the float GATES sweep.
COUNT_GATES = [
    ('PLAN_TOTAL_ATTEMPTS_CAP', 10, 'RESTOCK_PLAN_ATTEMPTS_CAP', 8),
    ('REPLAN_MAX_JOINT_SPACE', 3, 'RESTOCK_REPLAN_MAX_JOINT_SPACE', 2),
]


class PlanCountCapOverrideTests(unittest.TestCase):
    MOD = 'twin.restock_manipulation.ros_adapter'

    def test_defaults_when_unset(self):
        for attr, default, env, _ in COUNT_GATES:
            self.assertEqual(_value_with(self.MOD, attr, {env: None}), default, f'{attr} default drifted')

    def test_env_overrides_each_cap(self):
        for attr, _, env, wide in COUNT_GATES:
            self.assertEqual(_value_with(self.MOD, attr, {env: str(wide)}), wide, f'{attr} ignored its env')

    def test_clamped_sane(self):
        # Absurd values are clamped, never blindly trusted.
        self.assertLessEqual(_value_with(self.MOD, 'PLAN_TOTAL_ATTEMPTS_CAP',
                                         {'RESTOCK_PLAN_ATTEMPTS_CAP': '99999'}), 64)
        # Joint-space re-plans can never exceed the overall REPLAN_MAX.
        from twin.restock_manipulation.ros_adapter import REPLAN_MAX
        self.assertLessEqual(_value_with(self.MOD, 'REPLAN_MAX_JOINT_SPACE',
                                         {'RESTOCK_REPLAN_MAX_JOINT_SPACE': '99999'}), REPLAN_MAX)


if __name__ == '__main__':
    unittest.main()
