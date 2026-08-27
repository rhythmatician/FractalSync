"""Differentiable PyTorch mirrors of runtime-core's canonical math.

These functions re-implement, as tensor ops, the Rust reference
implementations in ``runtime_core``:

- :func:`synthesize_c` mirrors ``runtime_core::controller::synthesize``
  for the main cardioid (lobe=1).
- :func:`cardioid_proximity` mirrors
  ``runtime_core::proxies::mandelbrot_cardioid_proximity``.

They exist so gradients can flow during training. The residual phases used
by :func:`synthesize_c` come from ``runtime_core.residual_phases_for_seed_py``
— the same single source of truth the runtime controller uses — so training
and runtime share identical phase statistics (no golden-angle approximation).
Any change to the Rust canonical math must be mirrored here and the parity
tests updated — the Rust-first policy makes ``runtime-core`` the source of
truth.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Golden-angle spread of residual phases. Retained only as a backward-
# compatible fallback when no explicit phases are supplied. The primary path
# now passes phases from ``runtime_core.residual_phases_for_seed_py`` so
# training and runtime share identical phase statistics.
GOLDEN_ANGLE = 2.399963229728653


def synthesize_c(
    s_target: torch.Tensor,
    alpha: torch.Tensor,
    band_gates: torch.Tensor,
    thetas: torch.Tensor,
    k_residuals: int,
    residual_cap: float,
    phases: list[float] | None = None,
) -> torch.Tensor:
    """Differentiable c-space synthesis mirroring runtime-core's synthesize().

    Mirrors ``runtime_core::controller::synthesize`` for the main cardioid
    (lobe=1): carrier = μ/2 − μ²/4 with μ = s·e^{iθ}, plus k residual
    epicycles with amplitude α·s·radius/2^(k+1), gated per band and capped.

    All ops are real-tensor ops so gradients flow back through s, alpha,
    and band_gates. Returns a complex tensor of shape (batch,).

    ``phases`` are the residual phases to use. When provided (recommended),
    they come from ``runtime_core.residual_phases_for_seed_py`` so training
    uses the *exact* phase statistics as the runtime controller — eliminating
    the historical golden-angle vs seeded-RNG parity gap. When omitted, the
    legacy golden-angle spread is used for backward compatibility.
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

    if phases is None:
        phases = [
            float((k * GOLDEN_ANGLE) % (2.0 * np.pi)) for k in range(k_residuals)
        ]

    residual_re = torch.zeros(batch_size, device=device, dtype=s_target.dtype)
    residual_im = torch.zeros(batch_size, device=device, dtype=s_target.dtype)
    for k in range(k_residuals):
        gate = band_gates[:, k]
        amplitude = (alpha * (s_target * radius)) / (2.0 ** (k + 1))
        phase_k = float(phases[k])
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

    .. deprecated::
        Sunset per issue #88: the minimaps (mip pyramid S field) are the
        shore-distance oracle. Use :func:`shore_proximity` instead.
    """
    inner = 1.0 - 4.0 * c
    w = torch.sqrt(inner.to(torch.complex64))
    mu = 1.0 - w
    return torch.abs(torch.abs(mu) - 1.0)


def shore_proximity(c: torch.Tensor, level: int = 2) -> torch.Tensor:
    """Shore proximity sampled from the Map's mip pyramid (issue #88).

    Reads the baked S field (gradient-magnitude proximity G/(G+G0), already
    normalized [0, 1]) at each point of `c` via the runtime-core minimap
    reader. This is non-differentiable w.r.t. c — it is a *supervision
    signal*, not a gradient path. The Rust reader is the single source of
    truth; there is deliberately no tensor re-implementation.

    Args:
        c: complex tensor of points in c-space.
        level: mip level to sample (default 2 — the finest selected rung).

    Returns a float tensor of shore-proximity values in [0, 1].
    """
    import runtime_core

    flat = c.detach().cpu().reshape(-1)
    re = flat.real.tolist()
    im = flat.imag.tolist()
    values = runtime_core.minimap_shore_proximity_batch_py(re, im, level)
    return torch.tensor(
        values, dtype=torch.float32, device=c.device
    ).reshape(c.shape if c.dim() > 0 else ())


# ---------------------------------------------------------------------------
# PlayerState momentum integrator mirror
#
# Mirrors ``runtime_core::controller::PlayerState::step`` — the integrator the
# BROWSER actually executes. Constants must match controller.rs exactly:
#   drag = 0.90, accel_gain = omega_scale * 2.0 * dt, jitter amp 0.004/(k+1).
# The contour-bias step (minimap) is a runtime-only refinement on top of the
# clamped motion; training supervises the no-pyramid fallback path, which is
# what the golden vectors record.
# ---------------------------------------------------------------------------

PLAYER_DRAG = 0.90
PLAYER_JITTER_AMP = 0.004


def player_step_sequence(
    s_target: torch.Tensor,
    alpha: torch.Tensor,
    omega_scale: torch.Tensor,
    band_gates: torch.Tensor,
    segment_ids: torch.Tensor,
    dt: float = 1.0 / 60.0,
    c0: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Differentiable replay of PlayerState momentum integration.

    Mirrors ``PlayerState::step`` per frame: target = lobe_point_at_angle(
    lobe=1, 0, alpha*2π, s); a = (target − c)·ω·2·dt + gate jitter;
    v = drag·v + a; c += v·dt. Velocity resets at segment boundaries.

    All ops are real-tensor ops so gradients flow back through s, alpha,
    omega_scale, and band_gates. Returns complex tensor of shape (N,).

    Parity is pinned by backend/tests/test_golden_parity.py against
    shared/golden_vectors.json player trajectories.
    """
    n = s_target.shape[0]
    device = s_target.device

    # Target boundary points for every frame (vectorized carrier formula).
    theta = alpha.reshape(-1).float() * 2.0 * math.pi
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    mu_re = s_target.reshape(-1).float() * cos_t
    mu_im = s_target.reshape(-1).float() * sin_t
    mu2_re = mu_re * mu_re - mu_im * mu_im
    mu2_im = 2.0 * mu_im * mu_re
    tgt_re = 0.5 * mu_re - 0.25 * mu2_re
    tgt_im = 0.5 * mu_im - 0.25 * mu2_im

    # Gate jitter phases: alpha*2π*(k+2), amplitude 0.004*gate/(k+1).
    k_idx = torch.arange(band_gates.shape[1], device=device, dtype=torch.float32)
    jit_phase = alpha.reshape(-1, 1).float() * 2.0 * math.pi * (k_idx + 2.0)
    jit_amp = PLAYER_JITTER_AMP * band_gates.float() / (k_idx + 1.0)
    jit_re = (jit_amp * torch.cos(jit_phase)).sum(dim=1)
    jit_im = (jit_amp * torch.sin(jit_phase)).sum(dim=1)

    # Per-frame acceleration gain.
    w = omega_scale.reshape(-1).float().clamp(0.1, 10.0)
    accel_gain = w * 2.0 * dt

    seg = segment_ids.reshape(-1)
    seg_boundary = torch.zeros(n, dtype=torch.bool, device=device)
    if n > 1:
        seg_boundary[1:] = seg[1:] != seg[:-1]

    # Sequential scan (differentiable through time).
    c_re = torch.zeros(n, device=device, dtype=torch.float32)
    c_im = torch.zeros(n, device=device, dtype=torch.float32)
    v_re = torch.zeros((), device=device, dtype=torch.float32)
    v_im = torch.zeros((), device=device, dtype=torch.float32)

    start_re, start_im = c0 if c0 is not None else (
        float(tgt_re[0].detach()), float(tgt_im[0].detach())
    )
    cur_re = torch.tensor(start_re, device=device, dtype=torch.float32)
    cur_im = torch.tensor(start_im, device=device, dtype=torch.float32)

    for i in range(n):
        if seg_boundary[i]:
            v_re = torch.zeros_like(v_re)
            v_im = torch.zeros_like(v_im)
            cur_re = tgt_re[i]
            cur_im = tgt_im[i]
        a_re = (tgt_re[i] - cur_re) * accel_gain[i] + jit_re[i]
        a_im = (tgt_im[i] - cur_im) * accel_gain[i] + jit_im[i]
        v_re = v_re * PLAYER_DRAG + a_re
        v_im = v_im * PLAYER_DRAG + a_im
        cur_re = cur_re + v_re * dt
        cur_im = cur_im + v_im * dt
        c_re[i] = cur_re
        c_im[i] = cur_im

    return torch.complex(c_re, c_im)
