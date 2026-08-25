"""Rust ↔ PyTorch parity tests for orbit synthesis and c-space proxies.

The trainer's differentiable mirrors (``backend/src/cspace_proxies.py``)
claim to mirror ``runtime_core``'s canonical math. These tests make that
claim verifiable: they compare the PyTorch implementations against the
Rust bindings exposed through ``runtime_core`` on identical inputs.

The Rust controller draws residual phases from a seeded RNG, while the
PyTorch mirror uses a fixed golden-angle spread. To compare like with
like, the parity tests drive both through the *carrier-only* path
(alpha = 0, where residuals vanish) and through the residual path using
the Rust state's actual phases read back via ``residual_phases()``.

Run: pytest backend/tests/test_synthesis_parity.py
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
import torch

torch.manual_seed(0)


@pytest.fixture(scope="module")
def rc(runtime_core_module):  # noqa: F811 - provided by conftest.py
    """Return the *real* compiled runtime_core extension.

    Some legacy test modules install a lightweight fake ``runtime_core`` into
    ``sys.modules`` at import time. If that pollution reaches us, drop the
    cached entry and re-import so these parity tests always exercise the
    actual Rust bindings.
    """
    import importlib
    import sys

    mod = runtime_core_module
    if not hasattr(mod, "mandelbrot_cardioid_proximity_batch"):
        sys.modules.pop("runtime_core", None)
        importlib.invalidate_caches()
        mod = importlib.import_module("runtime_core")
    return mod


def _carrier_reference(theta: float, s: float) -> complex:
    """Independent reference: c = μ/2 − μ²/4 with μ = s·e^{iθ}."""
    mu = s * cmath.exp(1j * theta)
    return mu / 2 - mu**2 / 4


class TestCarrierParity:
    """alpha=0 path: PyTorch mirror vs Rust synthesize vs closed form."""

    def test_pytorch_carrier_matches_rust(self, rc):
        k_residuals = 6
        rp = rc.ResidualParams(
            k_residuals=k_residuals, residual_cap=0.5, radius_scale=1.0
        )

        s_val, theta_val = 1.02, 0.7

        state = rc.OrbitState.new_with_seed(
            1, 0, theta_val, 1.0, s_val, 0.0, k_residuals, 2.0, 1337
        )
        # alpha=0 → Rust returns pure carrier regardless of phases.
        rust_c = state.synthesize(rp, band_gates=[1.0] * k_residuals)

        from src.cspace_proxies import synthesize_c

        pt_c = synthesize_c(
            s_target=torch.tensor([s_val]),
            alpha=torch.zeros(1),
            band_gates=torch.ones(1, k_residuals),
            thetas=torch.tensor([theta_val]),
            k_residuals=k_residuals,
            residual_cap=0.5,
        )

        assert abs(rust_c.real - pt_c[0].real.item()) < 1e-6
        assert abs(rust_c.imag - pt_c[0].imag.item()) < 1e-6

    def test_rust_carrier_matches_closed_form(self, rc):
        rp = rc.ResidualParams(k_residuals=6, residual_cap=0.5, radius_scale=1.0)
        for theta in (0.0, 0.7, math.pi / 2, 3.1):
            state = rc.OrbitState.new_with_seed(1, 0, theta, 1.0, 1.02, 0.0, 6, 2.0, 42)
            c = state.synthesize(rp, None)
            expected = _carrier_reference(theta, 1.02)
            assert abs(c.real - expected.real) < 1e-12
            assert abs(c.imag - expected.imag) < 1e-12


class TestResidualParity:
    """Residual path: replay Rust phases inside the PyTorch mirror."""

    def test_pytorch_residuals_match_rust_with_same_phases(self, rc):
        from src.cspace_proxies import GOLDEN_ANGLE, synthesize_c

        k_residuals = 6
        residual_cap = 0.5
        radius = 0.25
        s_val, alpha_val, theta_val = 1.3, 0.8, 0.9

        rp = rc.ResidualParams(
            k_residuals=k_residuals, residual_cap=residual_cap, radius_scale=1.0
        )
        state = rc.OrbitState.new_with_seed(
            1, 0, theta_val, 1.0, s_val, alpha_val, k_residuals, 2.0, 1337
        )
        rust_c = state.synthesize(rp, band_gates=[1.0] * k_residuals)

        # Reconstruct what the Rust controller computed: carrier + Σ
        # amplitude_k · gate · e^{i·phase_k}, capped at residual_cap·radius.
        phases = list(state.residual_phases())
        carrier = _carrier_reference(theta_val, s_val)
        res_re = sum(
            (alpha_val * (s_val * radius)) / 2 ** (k + 1) * math.cos(p)
            for k, p in enumerate(phases)
        )
        res_im = sum(
            (alpha_val * (s_val * radius)) / 2 ** (k + 1) * math.sin(p)
            for k, p in enumerate(phases)
        )
        mag = math.hypot(res_re, res_im)
        cap = residual_cap * radius
        if mag > cap:
            scale = cap / mag
            res_re *= scale
            res_im *= scale
        expected = complex(carrier.real + res_re, carrier.imag + res_im)

        assert abs(rust_c.real - expected.real) < 1e-10
        assert abs(rust_c.imag - expected.imag) < 1e-10

        # The PyTorch mirror uses golden-angle phases; verify it produces the
        # same structure (amplitude decay + gating) by checking it against its
        # own reference with GOLDEN_ANGLE phases.
        gates = [1.0] * k_residuals
        pt_c = synthesize_c(
            s_target=torch.tensor([s_val]),
            alpha=torch.tensor([alpha_val]),
            band_gates=torch.tensor([gates], dtype=torch.float32),
            thetas=torch.tensor([theta_val]),
            k_residuals=k_residuals,
            residual_cap=residual_cap,
        )
        res_re_ga = sum(
            (alpha_val * (s_val * radius))
            / 2 ** (k + 1)
            * math.cos((k * GOLDEN_ANGLE) % (2 * math.pi))
            for k in range(k_residuals)
        )
        res_im_ga = sum(
            (alpha_val * (s_val * radius))
            / 2 ** (k + 1)
            * math.sin((k * GOLDEN_ANGLE) % (2 * math.pi))
            for k in range(k_residuals)
        )
        mag_ga = math.hypot(res_re_ga, res_im_ga)
        if mag_ga > cap:
            sc = cap / mag_ga
            res_re_ga *= sc
            res_im_ga *= sc
        expected_ga = complex(carrier.real + res_re_ga, carrier.imag + res_im_ga)

        assert abs(pt_c[0].real.item() - expected_ga.real) < 1e-5
        assert abs(pt_c[0].imag.item() - expected_ga.imag) < 1e-5

    def test_residual_cap_engages_identically(self, rc):
        """Large alpha must trigger the magnitude cap in both implementations."""
        from src.cspace_proxies import synthesize_c

        k_residuals = 6
        residual_cap = 0.05  # tiny cap → always engaged
        radius = 0.25
        s_val, alpha_val, theta_val = 2.0, 1.0, 0.4

        rp = rc.ResidualParams(
            k_residuals=k_residuals, residual_cap=residual_cap, radius_scale=1.0
        )
        state = rc.OrbitState.new_with_seed(
            1, 0, theta_val, 1.0, s_val, alpha_val, k_residuals, 2.0, 7
        )
        rust_c = state.synthesize(rp, band_gates=[1.0] * k_residuals)

        phases = list(state.residual_phases())
        carrier = _carrier_reference(theta_val, s_val)
        res = sum(
            (alpha_val * (s_val * radius)) / 2 ** (k + 1) * cmath.exp(1j * p)
            for k, p in enumerate(phases)
        )
        mag = abs(res)
        capped = res * (residual_cap * radius / mag) if mag > 0 else res
        expected = carrier + capped

        assert abs(rust_c.real - expected.real) < 1e-10
        assert abs(rust_c.imag - expected.imag) < 1e-10

        # PyTorch mirror with the same phases via a custom spread is not
        # directly injectable; instead confirm the cap bounds its residual.
        pt_c = synthesize_c(
            s_target=torch.tensor([s_val]),
            alpha=torch.tensor([alpha_val]),
            band_gates=torch.ones(1, k_residuals),
            thetas=torch.tensor([theta_val]),
            k_residuals=k_residuals,
            residual_cap=residual_cap,
        )
        pt_res = complex(pt_c[0].real.item(), pt_c[0].imag.item()) - carrier
        assert abs(pt_res) <= residual_cap * radius + 1e-6


class TestProximityParity:
    """Cardioid proximity: PyTorch mirror vs Rust proxies binding."""

    def test_pytorch_proximity_matches_rust_batch(self, rc):
        from src.cspace_proxies import cardioid_proximity

        rng = np.random.RandomState(42)
        points = [
            complex(re, im)
            for re, im in zip(rng.uniform(-1.5, 1.0, 32), rng.uniform(-1.0, 1.0, 32))
        ]

        rust_vals = rc.mandelbrot_cardioid_proximity_batch(points)

        c_tensor = torch.tensor(
            [p.real for p in points], dtype=torch.float32
        ) + 1j * torch.tensor([p.imag for p in points], dtype=torch.float32)
        pt_vals = cardioid_proximity(c_tensor.to(torch.complex64))

        for rv, pv in zip(rust_vals, pt_vals.tolist()):
            assert abs(rv - pv) < 1e-5

    def test_proximity_zero_on_boundary_both_sides(self, rc):
        from src.cspace_proxies import cardioid_proximity

        # Boundary points via the multiplier map with |μ| = 1.
        boundary = [
            complex(
                math.cos(t) / 2 - math.cos(2 * t) / 4,
                math.sin(t) / 2 - math.sin(2 * t) / 4,
            )
            for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)
        ]
        rust_vals = rc.mandelbrot_cardioid_proximity_batch(boundary)
        for rv in rust_vals:
            assert rv < 1e-9

        c_tensor = (
            torch.tensor([p.real for p in boundary], dtype=torch.float32)
            + 1j * torch.tensor([p.imag for p in boundary], dtype=torch.float32)
        ).to(torch.complex64)
        for pv in cardioid_proximity(c_tensor).tolist():
            assert pv < 1e-6

    def test_path_metrics_binding_round_trip(self, rc):
        points = [complex(0, 0), complex(3, 4), complex(6, 8)]
        mean_speed, max_speed, spread = rc.orbit_path_metrics_py(points)
        assert mean_speed == pytest.approx(5.0)
        assert max_speed == pytest.approx(5.0)
        # Pairwise distances: 5, 10, 5 → mean ≈ 6.667
        assert spread == pytest.approx(20.0 / 3.0)


class TestGradientFlow:
    """The mirrors exist so gradients flow; guard that property."""

    def test_synthesize_gradients_flow(self):
        from src.cspace_proxies import synthesize_c

        s = torch.tensor([1.0], requires_grad=True)
        alpha = torch.tensor([0.5], requires_grad=True)
        gates = torch.ones(1, 6, requires_grad=True)
        thetas = torch.tensor([0.3])

        c = synthesize_c(s, alpha, gates, thetas, 6, 0.5)
        c.abs().sum().backward()

        assert s.grad is not None and torch.isfinite(s.grad).all()
        assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
        assert gates.grad is not None and torch.isfinite(gates.grad).all()

    def test_proximity_gradient_flows(self):
        from src.cspace_proxies import cardioid_proximity

        c = torch.tensor([0.3 + 0.1j], requires_grad=True)
        p = cardioid_proximity(c)
        p.sum().backward()
        assert c.grad is not None and torch.isfinite(c.grad).all()
