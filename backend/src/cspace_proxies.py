"""Differentiable PyTorch mirrors of runtime-core's canonical math.

These functions re-implement, as tensor ops, the Rust reference
implementations in ``runtime_core``:

- :func:`synthesize_c` mirrors ``runtime_core::controller::synthesize``
  for the main cardioid (lobe=1).
- :func:`cardioid_proximity` mirrors
  ``runtime_core::proxies::mandelbrot_cardioid_proximity``.

They exist so gradients can flow during training. Because they are pure
functions of tensors, they can be parity-tested against the Rust bindings
(see ``backend/tests/test_synthesis_parity.py``). Any change to the Rust
canonical math must be mirrored here and the parity tests updated — the
Rust-first policy makes ``runtime-core`` the source of truth.
"""

from __future__ import annotations

import numpy as np
import torch

# Golden-angle spread of residual phases, matching the trainer's historical
# convention. The Rust controller uses per-state random phases; this fixed
# spread is the deterministic training-time approximation.
GOLDEN_ANGLE = 2.399963229728653


def synthesize_c(
    s_target: torch.Tensor,
    alpha: torch.Tensor,
    band_gates: torch.Tensor,
    thetas: torch.Tensor,
    k_residuals: int,
    residual_cap: float,
) -> torch.Tensor:
    """Differentiable c-space synthesis mirroring runtime-core's synthesize().

    Mirrors ``runtime_core::controller::synthesize`` for the main cardioid
    (lobe=1): carrier = μ/2 − μ²/4 with μ = s·e^{iθ}, plus k residual
    epicycles with amplitude α·s·radius/2^(k+1), gated per band and capped.

    All ops are real-tensor ops so gradients flow back through s, alpha,
    and band_gates. Returns a complex tensor of shape (batch,).
    """
    batch_size = s_target.shape[0]
    device = s_target.device

    # Carrier: mu = s * e^{i theta}; c = mu/2 - mu^2/4
    cos_t = torch.cos(thetas)
    sin_t = torch.sin(thetas)
    mu_re = s_target * cos_t
    mu_im = s_target * sin_t
    # mu^2 = (mu_re^2 - mu_im^2) + i(2 mu_re mu_im)
    mu2_re = mu_re * mu_re - mu_im * mu_im
    mu2_im = 2.0 * mu_re * mu_im
    carrier_re = 0.5 * mu_re - 0.25 * mu2_re
    carrier_im = 0.5 * mu_im - 0.25 * mu2_im

    # Cardioid radius used by runtime-core for lobe == 1 is 0.25.
    radius = 0.25

    residual_re = torch.zeros(batch_size, device=device, dtype=s_target.dtype)
    residual_im = torch.zeros(batch_size, device=device, dtype=s_target.dtype)
    for k in range(k_residuals):
        gate = band_gates[:, k]
        amplitude = (alpha * (s_target * radius)) / (2.0 ** (k + 1))
        phase_k = float((k * GOLDEN_ANGLE) % (2.0 * np.pi))  # golden-angle spread
        residual_re = residual_re + amplitude * gate * np.cos(phase_k)
        residual_im = residual_im + amplitude * gate * np.sin(phase_k)

    # Cap residual magnitude at residual_cap * radius.
    mag = torch.sqrt(residual_re**2 + residual_im**2)
    cap = residual_cap * radius
    scale = torch.where(
        mag > cap,
        cap / (mag + 1e-12),
        torch.ones_like(mag),
    )
    residual_re = residual_re * scale
    residual_im = residual_im * scale

    c_re = carrier_re + residual_re
    c_im = carrier_im + residual_im
    return torch.complex(c_re.float(), c_im.float())


def cardioid_proximity(c: torch.Tensor) -> torch.Tensor:
    """Differentiable distance proxy to the main cardioid boundary.

    Mirrors ``runtime_core::proxies::mandelbrot_cardioid_proximity``:
    w = sqrt(1 − 4c); mu = 1 − w; return ||mu| − 1|. Zero on the boundary.
    """
    inner = 1.0 - 4.0 * c
    w = torch.sqrt(inner.to(torch.complex64))
    mu = 1.0 - w
    return torch.abs(torch.abs(mu) - 1.0)
