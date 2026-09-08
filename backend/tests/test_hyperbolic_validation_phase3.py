"""Pending cross-scale measurements for #120."""

import pytest


@pytest.mark.skip(
    reason="Pending #120: cross-scale displacement and control authority measurements are not implemented."
)
def test_cross_scale_measurements_pending() -> None:
    pytest.fail("Implement cross-scale measurements before removing the skip.")
