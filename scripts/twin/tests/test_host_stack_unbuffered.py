"""fm359: the host-stack stdout capture must be UNBUFFERED, so a transient can never again truncate
host-stack.log mid-buffer (run-283 lost the pan failure reason -- the log ended at PLAN_DELIVERY
because piped-to-file python block-buffers stdout and the teardown kill dropped the last block)."""
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
SCRIPTS = os.path.abspath(os.path.join(HERE, '..', '..'))   # .../scripts
sys.path.insert(0, SCRIPTS)

import run_mobile_restock as rmr  # noqa: E402


class HostStackUnbufferedTests(unittest.TestCase):
    def test_launch_host_stack_sets_pythonunbuffered(self):
        import tempfile
        captured = {}

        class _FakeChild:
            pid = 4321

        def fake_popen(cmd, **kw):
            captured['env'] = kw.get('env')
            return _FakeChild()

        orig = rmr.subprocess.Popen
        rmr.subprocess.Popen = fake_popen
        try:
            with tempfile.TemporaryDirectory() as d:
                log = os.path.join(d, 'host-stack.log')
                child, handle = rmr._launch_host_stack(
                    repo=SCRIPTS + '/..', plugin_env='/x/plugin.sh',
                    nvidia_runtime='/x/rt', host_config='/x/host.json', logpath=log)
                handle.close()
        finally:
            rmr.subprocess.Popen = orig
        env = captured['env']
        self.assertIsNotNone(env)
        self.assertEqual(env.get('PYTHONUNBUFFERED'), '1')   # the fm359 fix
        self.assertEqual(env.get('ROS_DOMAIN_ID'), '88')     # existing behavior preserved
        # The sanitizing of the Isaac interpreter env is preserved (no leaked PYTHONPATH/HOME).
        self.assertNotIn('PYTHONHOME', env)

    def test_launch_script_runs_host_unbuffered(self):
        script = os.path.join(SCRIPTS, 'restock_runtime', 'launch_restock_host_stack.sh')
        text = open(script).read()
        self.assertIn('python3 -u ', text)   # host python unbuffered at the exec


if __name__ == '__main__':
    unittest.main()
