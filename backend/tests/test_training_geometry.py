"""Training proposals must be valid at their actual float32 rollout position."""

import pytest
import numpy as np

from src.training_geometry import sample_regular_training_start


def test_rejects_a_start_that_rounds_to_the_known_cusp(runtime_core_module):
    rc = runtime_core_module
    draws = iter([complex(0.25 + 1e-10, 0.0), complex(-0.5, 0.1)])
    start = sample_regular_training_start(lambda: next(draws), epsilon=1e-4)
    # This invokes the same Rust geometry gate as the rollout. Testing only
    # the original double-precision proposal would miss its float32 reset.
    metric = rc.manifold_induced_metric(start, rc.ManifoldConfig(epsilon=1e-4))
    assert metric[0][0] > 0.0
    assert start.real == -0.5
    assert start.imag == float(np.float32(0.1))


def test_regular_start_is_not_relocated(runtime_core_module):
    draws = 0

    def draw() -> complex:
        nonlocal draws
        draws += 1
        return complex(-0.5, 0.1)

    start = sample_regular_training_start(draw, epsilon=1e-4)
    assert start.real == -0.5
    assert draws == 1


def test_invalid_proposals_exhaust_a_bounded_budget(runtime_core_module):
    draws = 0

    def draw() -> complex:
        nonlocal draws
        draws += 1
        return complex(0.25, 0.0)

    with pytest.raises(RuntimeError, match="16.*Regular training start") as error:
        sample_regular_training_start(draw, epsilon=1e-4)
    assert draws == 16
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "geometry not regular" in str(error.value.__cause__)
