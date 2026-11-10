"""Analytic controls for the crossing event used by the physics regression."""

import pytest

from test_manifold_physics import _interpolated_crossing_time


@pytest.mark.parametrize("dt", [0.002, 0.001, 0.0005, 0.00025])
def test_linear_event_location_is_independent_of_sample_phase(dt):
    exact = 3.863270565
    start = int(exact / dt) * dt
    result = _interpolated_crossing_time(start, dt, exact - start, exact - start - dt)
    assert result == pytest.approx(exact, abs=1e-12)


def test_event_location_rejects_non_bracketing_samples():
    with pytest.raises(ValueError):
        _interpolated_crossing_time(0.0, 0.1, 1.0, 0.5)


def test_curved_event_location_converges_to_known_root():
    exact = 0.731
    errors = []
    for dt in (0.1, 0.05, 0.025):
        start = int(exact / dt) * dt
        before = exact * exact - start * start
        after = exact * exact - (start + dt) ** 2
        errors.append(abs(_interpolated_crossing_time(start, dt, before, after) - exact))
    assert max(errors) < 0.005
    assert errors[-1] < errors[0]
