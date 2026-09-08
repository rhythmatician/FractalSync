"""Pending same-integrator analytic versus sampled-field experiment for #120."""

import pytest


@pytest.mark.skip(
    reason="Blocked on an analytic-field injection seam in the Rust integrator, not Python imports."
)
def test_analytic_vs_sampled_energy_refinement_pending() -> None:
    pytest.fail(
        "Measure energy across timestep refinement with both fields using the same kernel."
    )
