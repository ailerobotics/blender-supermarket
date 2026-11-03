"""CPU fixtures only: no simulated trajectory is claimed as physics evidence."""
import math
from dataclasses import replace
import pytest
from twin.mobile import (MobileSpec, Pose2, DockObservation, Coupler, wheel_targets,
                         occupied_envelope, swept_envelope, transport_ready)


def observation():
    return DockObservation(Pose2(0, 0, 0), Pose2(-1.27, 0, 0), .28, .28, 0, 0, 0, 0)


def inside(point, hull):
    return all((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0]) >= -1e-8
               for a, b in zip(hull, hull[1:]+hull[:1]))


def test_drive_units_limits_and_interlocks():
    s = MobileSpec()
    assert wheel_targets(.2, 0) == (0, 0)
    assert wheel_targets(.16, 0, transport_ready=True) == (1, 1)
    l, r = wheel_targets(0, .2, transport_ready=True)
    assert r > 0 and l == -r
    assert max(map(abs, wheel_targets(100, 100, transport_ready=True))) <= s.max_wheel_speed
    assert wheel_targets(-.2, .3, coupled=True, transport_ready=True) == (0, 0)
    assert wheel_targets(.2, 0, coupled=True, articulation=1.2, transport_ready=True) == (0, 0)
    with pytest.raises(ValueError):
        wheel_targets(float('nan'), 0)


@pytest.mark.parametrize('change', [dict(age=.2), dict(age=-1), dict(cart_speed=.1),
    dict(robot_yaw_rate=.1), dict(cart_hitch_z=.30), dict(robot_hitch_z=float('nan')),
    dict(cart=Pose2(-1.4, 0, 0)), dict(cart=Pose2(-1.27, 0, .2))])
def test_bad_dock_never_calls_physics(change):
    c = Coupler()
    calls = []
    with pytest.raises(ValueError):
        c.attach(replace(observation(), **change), lambda: calls.append(True))
    assert not c.attached and not calls


@pytest.mark.parametrize('change,category,needle', [
    (dict(age=.2), 'stale', 'age=0.200'),
    (dict(robot_hitch_z=float('nan')), 'stale', 'non-finite'),
    (dict(cart_speed=.1), 'moving', 'cart_speed=0.1000'),
    (dict(robot_yaw_rate=.1), 'moving', 'robot_yaw_rate=0.1000'),
    (dict(cart=Pose2(-1.4, 0, 0)), 'distant', 'hitch gap='),      # 130 mm > 25 mm (XY)
    (dict(cart_hitch_z=.30), 'distant', 'hitch Z gap=20.0 mm'),
    (dict(cart=Pose2(-1.27, 0, .2)), 'misaligned', 'yaw=11.46 deg')])
def test_dock_reject_names_the_failing_gate_with_numbers(change, category, needle):
    c = Coupler()
    obs = replace(observation(), **change)
    reasons = c.dock_reject_reasons(obs)
    cats = [cat for cat, _ in reasons]
    assert category in cats, (category, reasons)
    assert any(needle in detail for _, detail in reasons), (needle, reasons)
    with pytest.raises(ValueError) as ei:
        c.attach(obs, lambda: None)
    # The legacy phrase is kept (greps/ledger), then the specific categories + numbers appended.
    assert 'dock rejected: stale, moving, distant, or misaligned' in str(ei.value)
    assert category in str(ei.value) and needle in str(ei.value)


def test_good_dock_has_no_reject_reasons():
    assert Coupler().dock_reject_reasons(observation()) == []


def test_coupling_transaction_and_stopped_detach():
    c = Coupler()
    def failure():
        raise RuntimeError('physics failed')
    with pytest.raises(RuntimeError):
        c.attach(observation(), failure)
    assert not c.attached
    c.attach(observation(), lambda: None)
    with pytest.raises(ValueError):
        c.detach(replace(observation(), robot_speed=.1), lambda: None)
    assert c.attached
    with pytest.raises(RuntimeError):
        c.detach(observation(), failure)
    assert c.attached
    c.detach(observation(), lambda: None)
    assert not c.attached


def test_envelope_contains_independent_cart_rotation_and_sweep():
    r0, c0 = Pose2(0, 0, 0), Pose2(-1.27, 0, 0)
    r1, c1 = Pose2(.3, .2, .8), Pose2(-1.1, -.2, -.5)
    hull = swept_envelope([(r0, c0), (r1, c1)])
    for i in range(101):
        t = i/100
        def lerp(a, b):
            return Pose2(a.x+t*(b.x-a.x), a.y+t*(b.y-a.y), a.yaw+t*(b.yaw-a.yaw))
        for p in occupied_envelope(lerp(r0, r1), lerp(c0, c1), margin=0):
            assert inside(p, hull)
    assert min(p[0] for p in hull) < -1.67
    assert max(p[1] for p in occupied_envelope(r0)) < .75


def test_transport_feedback_required():
    assert transport_ready({'a': .01}, {'a': 0}, .64)
    assert not transport_ready({}, {'a': 0}, .64)
    assert not transport_ready({'a': 0}, {'a': 0}, .8)
    assert not transport_ready({'a': float('nan')}, {'a': 0}, .6)


@pytest.mark.parametrize('changes', [dict(mass=-1), dict(wheel_radius=0),
    dict(hitch_x=0), dict(cart_hitch_x=.2), dict(arm_mount_z=.1)])
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        MobileSpec(**changes)
