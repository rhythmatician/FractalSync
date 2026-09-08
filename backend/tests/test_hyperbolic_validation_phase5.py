"""Pending runtime measurements beyond existing Rust/Python parity for #120."""

import pytest


@pytest.mark.skip(
    reason="Pending #120: runtime cost and additional Rust/Python/WASM measurements are not implemented."
)
def test_runtime_measurements_pending() -> None:
    pytest.fail("Implement missing runtime measurements before removing the skip.")
