"""Pending replay diagnostics beyond existing Shore-crossing tests for #120."""

import pytest


@pytest.mark.skip(
    reason="Pending #120: instrumented #82/#111 replay diagnostics are not implemented."
)
def test_instrumented_replays_pending() -> None:
    pytest.fail(
        "Implement work, dissipation, and derivative/integrator diagnostics before removing the skip."
    )
