from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twin.restock.settle_metrics import settle_report


def _window(n, step=1 / 60, mover=None):
    rows = []
    for i in range(n):
        t = i * step
        rows.append(dict(simulation_time=t,
            robot=dict(pose_xyz_yaw=[0.0, 0.0, 0.24, 0.0],
                       linear_velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0]),
            cart=dict(pose_xyz_yaw=(mover(i) if mover else [1.0, 0.0, 0.2, 0.0]),
                      linear_velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])))
    return rows


def test_frozen_bodies_settle_even_with_biased_reported_velocity():
    rows = _window(31)
    # A large solver-biased reported velocity must NOT fail a body that does not move.
    for r in rows:
        r['cart']['linear_velocity'] = [0.0144, 0.0, 0.0]
    settled, report = settle_report(rows)
    assert settled is True
    assert report['cart']['net_linear_rate_m_s'] == 0.0
    # The biased value is still surfaced for transparency.
    assert report['cart']['reported_solver_speed_biased'] > 0.01


def test_steady_translation_above_tolerance_is_not_settled():
    # Cart translating at 2 cm/s net over the window must fail the 5 mm/s gate.
    rows = _window(31, mover=lambda i: [1.0 + 0.02 * (i / 60.0), 0.0, 0.2, 0.0])
    settled, report = settle_report(rows)
    assert settled is False
    assert report['cart']['net_linear_rate_m_s'] > 0.005


def test_in_place_jitter_is_not_counted_as_motion():
    # Alternating +/- 1 mm jitter: large per-step rate, ~zero net displacement.
    rows = _window(31, mover=lambda i: [1.0 + (0.001 if i % 2 else -0.001), 0.0, 0.2, 0.0])
    settled, report = settle_report(rows)
    assert settled is True
    assert report['cart']['peak_step_linear_rate_m_s'] > 0.005
    assert report['cart']['net_linear_rate_m_s'] < 0.005


def test_rejects_non_monotonic_window():
    rows = _window(2)
    rows[1]['simulation_time'] = rows[0]['simulation_time']
    try:
        settle_report(rows)
    except ValueError:
        return
    raise AssertionError('expected non-monotonic window rejection')
