"""Pending #120 measurements beyond existing SPD coverage."""

import pytest


@pytest.mark.skip(
    reason="Pending #120: spectrum and tangent/normal anisotropy measurements are not implemented."
)
def test_metric_spectrum_and_anisotropy_pending() -> None:
    pytest.fail(
        "Compute eigenvalues, condition number, and anisotropy before removing the skip."
    )
