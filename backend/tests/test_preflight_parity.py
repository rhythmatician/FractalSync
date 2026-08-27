"""Tests for scripts/preflight_parity.py — the pre-training parity guardrail.

These tests exercise the same mandatory checks the preflight script runs
before training (carrier parity, mirror parity, shared phase source) so a
diverged mirror is caught in CI as well as at train time.

Run: python -m pytest backend/tests/test_preflight_parity.py -q
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Import the preflight module (scripts/ is not a package).
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
preflight = importlib.import_module("preflight_parity")


@pytest.fixture(scope="module")
def rc(runtime_core_module):  # noqa: F811 - provided by backend/conftest.py
    """Return the *real* compiled runtime_core extension.

    Some legacy test modules install a lightweight fake ``runtime_core`` into
    ``sys.modules`` at import time. If that pollution reaches us, drop the
    cached entry and re-import so these parity tests always exercise the
    actual Rust bindings.
    """
    mod = runtime_core_module
    if not hasattr(mod, "mandelbrot_cardioid_proximity_batch"):
        sys.modules.pop("runtime_core", None)
        importlib.invalidate_caches()
        mod = importlib.import_module("runtime_core")
    return mod


class TestPreflightChecks:
    def test_carrier_parity(self, rc):
        ok, max_err = preflight.check_carrier_parity(rc)
        assert ok, f"carrier parity failed: max abs err {max_err:.3e}"
        assert max_err <= preflight.CARRIER_TOL

    def test_mirror_parity(self, rc):
        ok, max_err = preflight.check_mirror_parity(rc)
        assert ok, f"mirror parity failed: max abs err {max_err:.3e}"
        assert max_err <= preflight.MIRROR_TOL

    def test_shared_phase_source(self, rc):
        ok, max_err = preflight.check_shared_phase_source(rc)
        assert ok, f"shared phase source mismatch: max abs err {max_err:.3e}"

    def test_run_preflight_all_pass(self, rc):
        all_ok, results = preflight.run_preflight(verbose=False)
        assert all_ok
        statuses = {name: status for name, status, _ in results}
        for name, _, _ in results:
            if name.startswith(("a)", "b)", "c)")):
                assert statuses[name] == "PASS", f"{name} did not pass"
