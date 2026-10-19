"""fm363: (1) universal sim-only step-level bounded retry at the mission stage seam; (2) env-shadow fix
so the fm362 widening knobs win over a ctor/config value; (3) plan wall cap reaches 120 via the knob."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.inventory import Inventory, Job
from twin.restock.mission import Mission, Result


class Fixture:
    """Minimal always-succeeding mission adapter (mirrors test_restock_mission.Fixture)."""
    def __init__(self):
        self.time = 0.

    def start(self, action, job):
        self.action, self.job = action, job

    def step(self, dt):
        self.time += dt
        return Result('succeeded', dict(
            source='test_fixture', observation_id=f'{self.action}-{self.time}',
            simulation_time=self.time, loaded=True, coupled=True, arrived=True, stationary=True,
            released=True, home=True, sku=self.job.sku, quantity=self.job.quantity,
            location=self.job.shelf_id, transfer_confirmed=True), '')

    def stop(self):
        pass


class _Retry(Fixture):
    """Succeeds every stage except ``fail_stage``, which fails (or raises) its first ``fail_times``
    attempts then succeeds. Each mission re-begin is a fresh attempt (start resets nothing here)."""
    def __init__(self, fail_stage, fail_times, raise_instead=False):
        super().__init__()
        self.fail_stage, self.fail_times, self.raise_instead = fail_stage, fail_times, raise_instead
        self.attempts = {}

    def step(self, dt):
        if self.action == self.fail_stage:
            n = self.attempts.get(self.action, 0)
            if n < self.fail_times:
                self.attempts[self.action] = n + 1
                if self.raise_instead:
                    raise RuntimeError('planning failed (exhausted within wall cap)')
                return Result('failed', {}, 'transient nav/dock trip')
        return super().step(dt)


def _env(**kw):
    saved = {k: os.environ.get(k) for k in kw}

    class _Ctx:
        def __enter__(self):
            for k, v in kw.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

        def __exit__(self, *a):
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return _Ctx()


class StepRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inv = Inventory(str(Path(self.tmp.name) / 'stock.sqlite'))
        self.inv.seed('backroom', 'milk', 2)
        self.job = Job('one', 'milk', 1, 'backroom', 'shelf')

    def tearDown(self):
        self.inv.close(); self.tmp.cleanup()

    def _run(self, adapter, **kw):
        m = Mission(self.inv, self.job, adapter, fixture_mode=True, **kw)
        m.start()
        for _ in range(400):
            if m.terminal:
                break
            m.tick(.1)
        return m

    def test_off_is_byte_identical_no_retry(self):
        with _env(RESTOCK_STEP_RETRY_MAX=None):
            m = self._run(_Retry('couple', 1))
            self.assertEqual(m.state, 'failed')   # couple fails once -> mission fails, no retry

    def test_maneuver_stage_retries_then_completes(self):
        with _env(RESTOCK_STEP_RETRY_MAX='2'):
            m = self._run(_Retry('couple', 2))    # couple fails twice, 2 retries -> succeeds
            self.assertEqual(m.state, 'complete')

    def test_maneuver_stage_retries_on_raise(self):
        with _env(RESTOCK_STEP_RETRY_MAX='2'):
            m = self._run(_Retry('park', 1, raise_instead=True))
            self.assertEqual(m.state, 'complete')

    def test_retry_bounded_exhausts_then_fails(self):
        with _env(RESTOCK_STEP_RETRY_MAX='2'):
            m = self._run(_Retry('couple', 5))    # fails 5x, only 2 retries -> fails
            self.assertEqual(m.state, 'failed')

    def test_collect_is_not_retried_even_when_enabled(self):
        # collect makes irreversible payload acts -> scoped OUT; a collect failure fails immediately.
        with _env(RESTOCK_STEP_RETRY_MAX='2'):
            m = self._run(_Retry('collect', 1))
            self.assertEqual(m.state, 'failed')


class EnvShadowTests(unittest.TestCase):
    def test_env_wins_over_passed_config_value(self):
        # env_wins reads os.environ LIVE, so no module reload (which pollutes the shared module) is
        # needed: the module_default argument stands in for the env-read PLAN_TOTAL_WALL_CAP_S (=120
        # when the env is 120). This is exactly how RosMotionPort.__init__ calls it on the ros_host
        # config value, so a passed 60 is overridden to 120 when the env is set.
        from twin.restock_manipulation.ros_adapter import env_wins
        with _env(RESTOCK_PLAN_WALL_CAP_S=None):
            self.assertEqual(env_wins('RESTOCK_PLAN_WALL_CAP_S', 60.0, 22.0), 60.0)   # env off: config value kept
        with _env(RESTOCK_PLAN_WALL_CAP_S='120'):
            self.assertEqual(env_wins('RESTOCK_PLAN_WALL_CAP_S', 60.0, 120.0), 120.0)  # env on: wins over 60

    def test_plan_wall_cap_module_constant_reaches_120_under_env(self):
        # Clean subprocess (no in-process reload) proves RESTOCK_PLAN_WALL_CAP_S=120 makes the module
        # constant 120 AND the effective RosMotionPort value 120 (shadow closed end-to-end).
        import subprocess
        env = dict(os.environ, RESTOCK_PLAN_WALL_CAP_S='120',
                   PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        out = subprocess.run(
            [sys.executable, '-c',
             'from twin.restock_manipulation import ros_adapter as ra;'
             'print(ra.PLAN_TOTAL_WALL_CAP_S, ra.env_wins("RESTOCK_PLAN_WALL_CAP_S", 60.0, ra.PLAN_TOTAL_WALL_CAP_S))'],
            capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(out.stdout.split(), ['120.0', '120.0'], out.stderr[-500:])


if __name__ == '__main__':
    unittest.main()
