"""fm366: close the wall-clock-bound SHADOWING family. ONE sweep that, under the full sim env block,
imports every bound module and instantiates the real StationClient the way the mission does, then
asserts the EFFECTIVE value of every wall-clock bound == the widened value -- failing by name on any
bound still pinned to a ctor/config default. A companion run asserts byte-identical with no env.

Objects that cannot be built on CPU (RosMotionPort/RosTransferBridge/RosPayloadSync import rclpy in
their ctors) are asserted at their nearest real seam: the module constant / env_wins the ctor uses.
StationClient (the confirmed shadow -- isaac_factory passed config over the env) IS instantiated for
real with a stub channel, so a reintroduced shadow fails this test.

Run in a clean subprocess per env so import-time constants are read fresh (no reload pollution)."""
import json
import os
import subprocess
import sys
import unittest

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# The sim-only widen block (what the run-295 orchestrator sets). Each -> the EFFECTIVE value expected.
SIM_ENV = {
    'RESTOCK_PLAN_WALL_CAP_S': '120', 'RESTOCK_MOVE_WALL_TIMEOUT_S': '75',
    'RESTOCK_RESPAWN_WAIT_CAP_S': '120', 'RESTOCK_ARM_HEARTBEAT_WALL_BACKSTOP_S': '90',
    'RESTOCK_HEARTBEAT_UNAVAILABLE_S': '30', 'RESTOCK_STAGE_UNAVAILABLE_S': '12',
    'RESTOCK_STATION_PENDING_WALL_S': '8', 'RESTOCK_SENSOR_WALL_BACKSTOP_S': '180',
    'RESTOCK_SENSOR_STREAM_WALL_BACKSTOP_S': '180', 'RESTOCK_FEEDBACK_SIM_STALE_S': '1.0',
    'RESTOCK_FEEDBACK_WALL_BACKSTOP_S': '90', 'RESTOCK_STARTUP_GRACE_S': '20',
    'RESTOCK_PAYLOAD_ACK_WALL_S': '15',
}

# Probe snippet: import every bound module, instantiate the real StationClient, dump effective values.
# StationClient is passed the UN-widened real-rig values (3.0/10.0) so a shadow (passed value winning)
# is caught: under the env these must come back widened, not 3.0/10.0.
PROBE = r'''
import json
from twin.restock import trajectory_host as th
from twin.restock_manipulation import ros_adapter as ra
from twin.restock import native_transport as nt
from twin.restock import native_channel as nc
from twin.restock import station_host as sh
from twin.restock_manipulation import isaac_sensors as isens
from twin.restock_manipulation import ros_payload as rp
sc = nc.StationClient(object(), stage_clock=lambda: 0.0, stage_unavailable_s=3.0, heartbeat_s=10.0)
out = {
  'FEEDBACK_SIM_STALE_S': th.FEEDBACK_SIM_STALE_S,
  'FEEDBACK_WALL_BACKSTOP_S': th.FEEDBACK_WALL_BACKSTOP_S,
  'PLAN_TOTAL_WALL_CAP_S': ra.PLAN_TOTAL_WALL_CAP_S,
  'MOVE_WALL_TIMEOUT_S': ra.MOVE_WALL_TIMEOUT_S,
  'RESPAWN_WAIT_CAP_S': ra.RESPAWN_WAIT_CAP_S,
  'SENSOR_STREAM_WALL_BACKSTOP_S': ra.SENSOR_STREAM_WALL_BACKSTOP_S,
  'STARTUP_GRACE_S': ra.STARTUP_GRACE_S,
  'ARM_HEARTBEAT_WALL_BACKSTOP_S': nt.ARM_HEARTBEAT_WALL_BACKSTOP_S,
  'STATION_PENDING_WALL_S': sh.STATION_PENDING_WALL_S,
  'SENSOR_WALL_BACKSTOP_S': isens.SENSOR_WALL_BACKSTOP_S,
  # env_wins seams the un-instantiable ctors use (RosMotionPort/RosTransferBridge): env must win over a
  # passed caller value (here the real-rig-ish 22/25/5).
  'motionport_plan_wall_cap': ra.env_wins('RESTOCK_PLAN_WALL_CAP_S', 60.0, ra.PLAN_TOTAL_WALL_CAP_S),
  'motionport_timeout': ra.env_wins('RESTOCK_MOVE_WALL_TIMEOUT_S', 25.0, ra.MOVE_WALL_TIMEOUT_S),
  'bridge_startup_grace': ra.env_wins('RESTOCK_STARTUP_GRACE_S', 5.0, ra.STARTUP_GRACE_S),
  # RosPayloadSync ack: the env-governed module value (ctor uses it when set, else the passed config).
  'payload_ack': (float(rp.PAYLOAD_ACK_WALL_S) if rp.PAYLOAD_ACK_WALL_S is not None else 15.0),
  # StationClient REAL ctor (the confirmed shadow): env must win over the passed 3.0/10.0.
  'stationclient_stage_unavailable_s': sc.stage_unavailable_s,
  'stationclient_heartbeat_s': sc.heartbeat_s,
}
print(json.dumps(out))
'''


def _probe(extra_env):
    env = dict(os.environ)
    env['PYTHONPATH'] = SCRIPTS
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    for k in SIM_ENV:                      # start from a clean slate for every bound
        env.pop(k, None)
    env.update(extra_env)
    out = subprocess.run([sys.executable, '-c', PROBE], env=env, capture_output=True, text=True)
    assert out.returncode == 0, f'probe failed: {out.stderr[-2000:]}'
    return json.loads(out.stdout.strip().splitlines()[-1])


class WallBoundSweep(unittest.TestCase):
    def test_every_bound_widened_under_sim_env(self):
        v = _probe(SIM_ENV)
        expected = {
            'FEEDBACK_SIM_STALE_S': 1.0, 'FEEDBACK_WALL_BACKSTOP_S': 90.0,
            'PLAN_TOTAL_WALL_CAP_S': 120.0, 'MOVE_WALL_TIMEOUT_S': 75.0,
            'RESPAWN_WAIT_CAP_S': 120.0, 'SENSOR_STREAM_WALL_BACKSTOP_S': 180.0,
            'STARTUP_GRACE_S': 20.0, 'ARM_HEARTBEAT_WALL_BACKSTOP_S': 90.0,
            'STATION_PENDING_WALL_S': 8.0, 'SENSOR_WALL_BACKSTOP_S': 180.0,
            'motionport_plan_wall_cap': 120.0, 'motionport_timeout': 75.0,
            'bridge_startup_grace': 20.0, 'payload_ack': 15.0,
            'stationclient_stage_unavailable_s': 12.0, 'stationclient_heartbeat_s': 30.0,
        }
        for k, exp in expected.items():
            self.assertAlmostEqual(v[k], exp, msg=f'{k}: still a ctor/config default ({v[k]}) not widened {exp}')

    def test_stationclient_shadow_regression_guard(self):
        # The confirmed shadow: with the env set, the REAL StationClient ctor must override the passed
        # config value (3.0). This assertion targets the effective attr, not a module constant.
        v = _probe({'RESTOCK_STAGE_UNAVAILABLE_S': '12'})
        self.assertEqual(v['stationclient_stage_unavailable_s'], 12.0)

    def test_byte_identical_when_env_unset(self):
        v = _probe({})
        defaults = {
            'FEEDBACK_SIM_STALE_S': 0.2, 'FEEDBACK_WALL_BACKSTOP_S': 30.0,
            'PLAN_TOTAL_WALL_CAP_S': 22.0, 'MOVE_WALL_TIMEOUT_S': 25.0,
            'RESPAWN_WAIT_CAP_S': 45.0, 'SENSOR_STREAM_WALL_BACKSTOP_S': 30.0,
            'STARTUP_GRACE_S': 5.0, 'ARM_HEARTBEAT_WALL_BACKSTOP_S': 30.0,
            'STATION_PENDING_WALL_S': 2.0, 'SENSOR_WALL_BACKSTOP_S': 60.0,
            'motionport_plan_wall_cap': 60.0, 'motionport_timeout': 25.0,
            'bridge_startup_grace': 5.0, 'payload_ack': 15.0,
            'stationclient_stage_unavailable_s': 3.0, 'stationclient_heartbeat_s': 10.0,
        }
        for k, exp in defaults.items():
            self.assertAlmostEqual(v[k], exp, msg=f'{k}: default drifted ({v[k]} != {exp})')


if __name__ == '__main__':
    unittest.main()
