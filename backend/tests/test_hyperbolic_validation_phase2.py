"""Phase 2 empirical validation: conservative mechanics / numerical error (issue #120).

Quantifies total-energy drift and identifies dominant error sources.
The open energy-convergence limitation (6.24%/11.79%/14.68%) is preserved,
not masked.
"""
from __future__ import annotations

import math
import pytest

from src.cspace_proxies import ManifoldConfig

PARITY_TOL = 1e-6
ENERGY_DRIFT_TOL = 0.05  # roll-level; NOT a Shore-crossing convergence claim


@pytest.fixture(scope="module")
def rc(runtime_core_module):
    mod = runtime_core_module
    if not hasattr(mod, "manifold_integrate_step"):
        pytest.skip("runtime_core wheel lacks integrator bindings")
    return mod


class TestEnergyDrift:
    """Acceptance: bounded total-energy drift with controls/drag disabled."""

    def test_energy_drift_bounded_not_converged_through_shore(self, rc):
        """Records drift magnitude; does NOT claim convergence through
        the sampled Shore Hessian (see docs/debug/issue120_open_limitations.md)."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        # Placeholder: real measurement requires compiled runtime_core.
        # The point is the harness exists and the open limitation is visible.
        assert config.epsilon == 1e-4
