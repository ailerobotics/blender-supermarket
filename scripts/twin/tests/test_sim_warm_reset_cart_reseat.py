"""Warm re-seed start-pose RESTORATION (sim-warm-reset-cart-reseat-v1); CPU, no GPU/Isaac.

The warm two-run's second attempt must begin from the SAME configured start pose the cold start
verifies, not from where attempt 1 left the train. warmacc4's refusal:

    WARM_RESET step=reset_stage_and_reseed REFUSED: reseat: START_STATE_SETTLE_MISMATCH: the
    synthesized cart settled at heading -99.1 deg, not the configured dock -90.0 deg (residual 9.1 deg
    > 5.0 deg); the couple standoff would be derived from the wrong cart axis
    ... decision=cold_boot

These prove the fix on a FAKE stage: attempt 1 leaves the cart at -99.1 deg and the carton displaced,
and the re-seed restores BOTH (and the base) to the configured pose + zeroes velocities, so PR 297's
heading_check then passes; a body that cannot be restored still refuses LOUDLY by name; and the
production reseat op restores the cart to the configured dock BEFORE the (Isaac-owned) verify runs.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock import warm_reseat as wr          # noqa: E402
from twin.restock import warm_reset_provider as wrp  # noqa: E402
from twin.restock import shortcut as sc              # noqa: E402
from twin.restock import start_settle as ss          # noqa: E402

DOCK_YAW = math.radians(-90.0)       # the configured 304 dock heading
NUDGED_YAW = math.radians(-99.1)     # where attempt 1's host e-stop left the cart (warmacc4)


# --------------------------------------------------------------------------- fakes

class _FakeBody:
    """A live-body stand-in: stores the world pose set on it and records velocity zeroing."""
    def __init__(self, position, yaw):
        self._pos = [float(v) for v in position]
        self._quat = list(wr.wxyz_from_yaw(yaw))
        self.lin = self.ang = None
        self.set_calls = 0

    def get_world_pose(self):
        return list(self._pos), list(self._quat)

    def set_world_pose(self, position=None, orientation=None):
        self.set_calls += 1
        if position is not None:
            self._pos = [float(v) for v in position]
        if orientation is not None:
            self._quat = [float(v) for v in orientation]

    def set_linear_velocity(self, v):
        self.lin = [float(x) for x in v]

    def set_angular_velocity(self, v):
        self.ang = [float(x) for x in v]


class _StuckBody(_FakeBody):
    """A body whose set_world_pose is a no-op -- it cannot be moved back (=> loud refuse)."""
    def set_world_pose(self, position=None, orientation=None):
        self.set_calls += 1  # ignores the request


class _NoSetterBody:
    """A body with no set_world_pose handle at all."""
    def __init__(self, position, yaw):
        self._pos = [float(v) for v in position]
        self._quat = list(wr.wxyz_from_yaw(yaw))

    def get_world_pose(self):
        return list(self._pos), list(self._quat)


# --------------------------------------------------------------------------- pure restore + verify

def test_reseed_restores_cart_carton_and_base_to_configured_pose():
    # attempt 1 left the cart yawed to -99.1 deg, the carton bumped off its slot, the base drifted.
    cart = _FakeBody((-9.0, 16.0, 0.20), NUDGED_YAW)
    carton = _FakeBody((-9.3, 16.4, 0.95), math.radians(12.0))   # displaced
    base = _FakeBody((-8.6, 15.6, 0.0), math.radians(4.0))       # drifted + yawed

    targets = [
        wr.planar_target(cart, (-9.0, 16.0, DOCK_YAW), rest_z=0.20, name='cart'),
        wr.captured_target(carton, ((-9.0, 16.2, 0.95), wr.wxyz_from_yaw(0.0)), name='carton'),
        wr.planar_target(base, (-8.5, 15.5, 0.0), rest_z=0.0, name='base'),
    ]
    records = wr.restore_targets(targets)

    # Cart restored to the configured dock heading, velocities zeroed.
    assert math.degrees(wr.yaw_from_wxyz(cart.get_world_pose()[1])) == pytest.approx(-90.0, abs=1e-6)
    assert cart.lin == [0.0, 0.0, 0.0] and cart.ang == [0.0, 0.0, 0.0]
    # Carton restored to its captured (configured) pose.
    assert carton.get_world_pose()[0] == [-9.0, 16.2, 0.95]
    assert math.degrees(wr.yaw_from_wxyz(carton.get_world_pose()[1])) == pytest.approx(0.0, abs=1e-6)
    # Base restored (moved by its drift), read-back ok.
    assert base.get_world_pose()[0][:2] == [-8.5, 15.5]
    assert all(r.readback_ok for r in records)
    # The record names the correction the re-seed made to the cart.
    cart_rec = next(r for r in records if r.name == 'cart')
    assert cart_rec.moved_yaw_deg == pytest.approx(9.1, abs=0.05)


def test_pr297_heading_check_passes_after_restore_and_reproduces_warmacc4_refusal():
    # BEFORE the fix: the cart sat at -99.1 deg and PR 297's heading_check refused (the recorded line).
    ok, residual, reason = ss.heading_check(NUDGED_YAW, DOCK_YAW, math.radians(5.0))
    assert not ok and 'START_STATE_SETTLE_MISMATCH' in reason
    assert math.degrees(residual) == pytest.approx(9.1, abs=0.05)

    # AFTER the fix: restore the cart to the configured dock, then heading_check passes.
    cart = _FakeBody((-9.0, 16.0, 0.20), NUDGED_YAW)
    wr.restore_target(wr.planar_target(cart, (-9.0, 16.0, DOCK_YAW), rest_z=0.20, name='cart'))
    restored_yaw = wr.yaw_from_wxyz(cart.get_world_pose()[1])
    ok, residual, reason = ss.heading_check(restored_yaw, DOCK_YAW, math.radians(5.0))
    assert ok and reason is None and residual == pytest.approx(0.0, abs=1e-6)


def test_unrestorable_cart_refuses_loudly_by_name():
    cart = _StuckBody((-9.0, 16.0, 0.20), NUDGED_YAW)
    with pytest.raises(wr.WarmReseatRefused) as ei:
        wr.restore_target(wr.planar_target(cart, (-9.0, 16.0, DOCK_YAW), rest_z=0.20, name='cart'))
    assert 'WARM_RESEAT_UNRESTORED' in str(ei.value) and "'cart'" in str(ei.value)


def test_unrestorable_base_names_the_base_code():
    base = _StuckBody((-8.6, 15.6, 0.0), math.radians(40.0))   # a drifted/flipped base that won't move
    with pytest.raises(wr.WarmReseatRefused) as ei:
        wr.restore_target(wr.planar_target(base, (-8.5, 15.5, 0.0), rest_z=0.0, name='base'))
    assert 'START_STATE_BASE_UNRESTORED' in str(ei.value)


def test_body_without_set_world_pose_refuses():
    body = _NoSetterBody((-9.0, 16.0, 0.20), NUDGED_YAW)
    with pytest.raises(wr.WarmReseatRefused) as ei:
        wr.restore_target(wr.planar_target(body, (-9.0, 16.0, DOCK_YAW), rest_z=0.20, name='cart'))
    assert 'no set_world_pose handle' in str(ei.value)


def test_configured_planar_reads_initial_staging_dock_heading():
    scenario = {'initial_staging': {'cart_body_pose': [-9.0, 16.0, DOCK_YAW],
                                    'robot_base_pose': [-8.5, 15.5, 0.0]}}
    assert wr.configured_planar(scenario, 'cart_body_pose') == (-9.0, 16.0, pytest.approx(DOCK_YAW))
    assert wr.configured_planar(scenario, 'robot_base_pose') == (-8.5, 15.5, 0.0)
    assert wr.configured_planar({}, 'cart_body_pose') is None
    assert wr.configured_planar({'initial_staging': {'cart_body_pose': [1, 2]}}, 'cart_body_pose') is None


# --------------------------------------------------------------- production reseat op wiring

class _FakeWorld:
    def __init__(self):
        self.current_time = 0.0

    def reset(self):
        self.current_time = 0.0


class _FakeStage:
    def Traverse(self):
        return []


class _FakeAssembly:
    def __init__(self, cart, robot):
        self.stage = _FakeStage()
        self.world = _FakeWorld()
        self.cart_body = cart
        self.robot = robot
        self.evidence = {'composition': {'stock': '/World/stock'}}


def test_production_reseat_restores_cart_before_isaac_verify(monkeypatch):
    # Dock-aligned warm run: attempt 1 left the cart at -99.1 deg; the production reseat op must restore
    # it to the configured dock BEFORE the Isaac-owned verify_dock_seat reads it.
    cart = _FakeBody((-9.0, 16.0, 0.20), NUDGED_YAW)
    robot = _FakeBody((-8.6, 15.6, 0.0), math.radians(3.0))
    assembly = _FakeAssembly(cart, robot)
    config = {'couple_dock_align': True}
    scenario = {'initial_staging': {'cart_body_pose': [-9.0, 16.0, DOCK_YAW],
                                    'robot_base_pose': [-8.5, 15.5, 0.0]}}

    seen = {}

    def _fake_verify(a, c, s):
        seen['cart_yaw_deg'] = math.degrees(wr.yaw_from_wxyz(a.cart_body.get_world_pose()[1]))
        seen['base_xy'] = tuple(a.robot.get_world_pose()[0][:2])

    monkeypatch.setattr(sc, 'verify_dock_seat', _fake_verify)

    ops = wrp._build_production_ops(app=None, assembly=assembly, parts={}, plan=None, config=config,
                                    output='/x', scenario=scenario, hooks={})
    ops.reseat(attempt=2, now=2.0)

    assert seen['cart_yaw_deg'] == pytest.approx(-90.0, abs=1e-6)   # restored before verify saw it
    assert seen['base_xy'] == (-8.5, 15.5)
    assert cart.lin == [0.0, 0.0, 0.0]


def test_production_reseat_refusal_is_wrapped_as_a_named_cold_boot(monkeypatch):
    # An unrestorable cart makes the provider's __call__ surface a named 'reseat: ...' refusal -- the
    # loud cold-boot the launcher records (never a silent reuse of a mis-seated train).
    cart = _StuckBody((-9.0, 16.0, 0.20), NUDGED_YAW)
    robot = _FakeBody((-8.5, 15.5, 0.0), 0.0)
    assembly = _FakeAssembly(cart, robot)
    config = {'couple_dock_align': True}
    scenario = {'initial_staging': {'cart_body_pose': [-9.0, 16.0, DOCK_YAW],
                                    'robot_base_pose': [-8.5, 15.5, 0.0]}}
    ops = wrp._build_production_ops(app=None, assembly=assembly, parts={}, plan=None, config=config,
                                    output='/x', scenario=scenario, hooks={})
    provider = wrp.WarmResetProvider(ops=ops, baseline_prims=frozenset())
    with pytest.raises(wrp.WarmResetRefused) as ei:
        provider(attempt=2, now=2.0)
    assert 'reseat: WARM_RESEAT_UNRESTORED' in str(ei.value)
