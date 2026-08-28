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
# Gravity valley (orbit-controller/3): restoring acceleration toward the
# origin. Must match runtime-core controller.rs GRAVITY_ACCEL exactly.
GRAVITY_ACCEL = 0.01
# Music push (uphill toward the Shore). Must match minimap.rs
# MUSIC_PUSH_GAIN. Applied along the analytic cardioid normal in this
# mirror (the trainer supervises the no-pyramid path).
MUSIC_PUSH_GAIN = 0.55


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


# ---------------------------------------------------------------------------
# OrbitController mirror (the RUNTIME controller — May-proven baseline)
#
# Mirrors ``runtime_core::controller::OrbitController::step`` with all
# refinement flags OFF (the default runtime path). Constants must match
# controller.rs exactly:
#   boundary: theta_b = 2*pi*alpha; r = 0.25*(1-cos(theta_b));
#             c = r*e^(i*theta_b/2) * min(s, 1.5), s clamped [0.01, 3]
#   residuals: sum_k gate_k * 0.05 * e^(i*(k+2)*theta), theta += omega*dt
# ---------------------------------------------------------------------------

ORBIT_RESIDUAL_AMP = 0.05


def orbit_controller_sequence(
    s_target: torch.Tensor,
    alpha: torch.Tensor,
    omega: float,
    band_gates: torch.Tensor,
    segment_ids: torch.Tensor,
    dt: float = 1.0 / 60.0,
) -> torch.Tensor:
    """Differentiable replay of OrbitController::step (May baseline).

    This is THE training-time mirror of the controller the browser executes.
    Per frame: theta += omega*dt; c = mandelbrot_boundary(s, alpha) +
    sum_k gate_k*0.05*e^(i*(k+2)*theta). Fully differentiable w.r.t. s, alpha,
    and band_gates.

    Parity is pinned by backend/tests/test_golden_parity.py and by the
    preflight player-mirror check against runtime_core.OrbitController.
    """
    n = s_target.shape[0]
    device = s_target.device

    s = s_target.reshape(-1).float().clamp(0.01, 3.0)
    a = alpha.reshape(-1).float().clamp(0.0, 1.0)

    # May's mandelbrotBoundary(s, alpha), vectorized:
    # theta_b = 2*pi*alpha; r = 0.25*(1-cos(theta_b)); scale = min(s, 1.5)
    theta_b = a * 2.0 * math.pi
    r = 0.25 * (1.0 - torch.cos(theta_b))
    scale = torch.clamp(s, max=1.5)
    base_re = r * torch.cos(theta_b / 2.0) * scale
    base_im = r * torch.sin(theta_b / 2.0) * scale

    # Wobble phase: sequential scan of theta += omega*dt, reset per segment.
    seg = segment_ids.reshape(-1)
    seg_boundary = torch.zeros(n, dtype=torch.bool, device=device)
    if n > 1:
        seg_boundary[1:] = seg[1:] != seg[:-1]

    two_pi = 2.0 * math.pi
    theta = torch.zeros(n, device=device, dtype=torch.float32)
    th = torch.zeros((), device=device, dtype=torch.float32)
    for i in range(n):
        if seg_boundary[i]:
            th = torch.zeros_like(th)
        th = (th + omega * dt) % two_pi
        theta[i] = th

    # Residual epicycles: gate_k * 0.05 * e^(i*(k+2)*theta).
    k_idx = torch.arange(band_gates.shape[1], device=device, dtype=torch.float32)
    freqs = k_idx + 2.0
    phase = theta.reshape(-1, 1) * freqs.reshape(1, -1)
    gates = band_gates.float().clamp(0.0, 1.0)
    res_re = (gates * ORBIT_RESIDUAL_AMP * torch.cos(phase)).sum(dim=1)
    res_im = (gates * ORBIT_RESIDUAL_AMP * torch.sin(phase)).sum(dim=1)

    return torch.complex(base_re + res_re, base_im + res_im)


def orbit_controller_momentum_sequence(
    s_target: torch.Tensor,
    alpha: torch.Tensor,
    omega: float,
    band_gates: torch.Tensor,
    segment_ids: torch.Tensor,
    dt: float = 1.0 / 60.0,
    drag: float = 0.90,
    thrust: float | torch.Tensor = 0.0,
    initial_c: torch.Tensor | None = None,
    energy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable replay of OrbitController::step with momentum ON.

    Mirrors the Rust momentum path exactly (controller.rs, v3):
      theta += omega*dt
      target = mandelbrot_boundary(s, alpha) + residual epicycles
      a = (target - c) * 2*dt          # pull-as-acceleration
        - GRAVITY_ACCEL * c            # gravity valley (settle at origin)
        + thrust * tangent(target - c) # audio thrust (inertia)
      v = v*drag + a
      c += v*dt + MUSIC_PUSH_GAIN * energy * shore_normal(c)
    c starts at the first frame's boundary point; velocity resets at
    segment boundaries. Fully differentiable w.r.t. s, alpha, band_gates.

    Parity is pinned by preflight check (e2) against the Rust binding with
    set_momentum(true).
    """
    n = s_target.shape[0]
    device = s_target.device

    s = s_target.reshape(-1).float().clamp(0.01, 3.0)
    a = alpha.reshape(-1).float().clamp(0.0, 1.0)

    theta_b = a * 2.0 * math.pi
    r = 0.25 * (1.0 - torch.cos(theta_b))
    scale = torch.clamp(s, max=1.5)
    base_re = r * torch.cos(theta_b / 2.0) * scale
    base_im = r * torch.sin(theta_b / 2.0) * scale

    seg = segment_ids.reshape(-1)
    seg_boundary = torch.zeros(n, dtype=torch.bool, device=device)
    if n > 1:
        seg_boundary[1:] = seg[1:] != seg[:-1]

    two_pi = 2.0 * math.pi
    theta = torch.zeros(n, device=device, dtype=torch.float32)
    th = torch.zeros((), device=device, dtype=torch.float32)
    for i in range(n):
        if seg_boundary[i]:
            th = torch.zeros_like(th)
        th = (th + omega * dt) % two_pi
        theta[i] = th

    k_idx = torch.arange(band_gates.shape[1], device=device, dtype=torch.float32)
    freqs = k_idx + 2.0
    phase = theta.reshape(-1, 1) * freqs.reshape(1, -1)
    gates = band_gates.float().clamp(0.0, 1.0)
    res_re = (gates * ORBIT_RESIDUAL_AMP * torch.cos(phase)).sum(dim=1)
    res_im = (gates * ORBIT_RESIDUAL_AMP * torch.sin(phase)).sum(dim=1)

    tgt_re = base_re + res_re
    tgt_im = base_im + res_im

    # Sequential momentum integration (differentiable through time).
    c_re = torch.zeros(n, device=device, dtype=torch.float32)
    c_im = torch.zeros(n, device=device, dtype=torch.float32)
    v_re = torch.zeros((), device=device, dtype=torch.float32)
    v_im = torch.zeros((), device=device, dtype=torch.float32)

    # Rust: c starts at (0,0) Default unless domain-randomized.
    ic: torch.Tensor | None = None
    if initial_c is not None and isinstance(initial_c, torch.Tensor) and initial_c.numel() > 0:
        ic = initial_c
        if ic.is_complex() and ic.numel() == n:
            cur_re = ic[0].real.float()
            cur_im = ic[0].imag.float()
        elif ic.is_complex():
            cur_re = ic.real.float().squeeze()
            cur_im = ic.imag.float().squeeze()
        else:
            cur_re = torch.zeros((), device=device, dtype=torch.float32)
            cur_im = torch.zeros((), device=device, dtype=torch.float32)
    else:
        cur_re = torch.zeros((), device=device, dtype=torch.float32)
        cur_im = torch.zeros((), device=device, dtype=torch.float32)
    accel_gain = 2.0 * dt

    for i in range(n):
        if seg_boundary[i]:
            v_re = torch.zeros_like(v_re)
            v_im = torch.zeros_like(v_im)
            if ic is not None and ic.is_complex() and ic.numel() == n:
                cur_re = ic[i].real.float()
                cur_im = ic[i].imag.float()
        dx = tgt_re[i] - cur_re
        dy = tgt_im[i] - cur_im
        a_re = dx * accel_gain - GRAVITY_ACCEL * cur_re
        a_im = dy * accel_gain - GRAVITY_ACCEL * cur_im
        thi = thrust[i] if isinstance(thrust, torch.Tensor) and thrust.ndim > 0 else thrust
        if isinstance(thi, torch.Tensor):
            thi = float(thi.item())
        if thi != 0.0 and thi > 0.0:
            d = torch.sqrt(dx * dx + dy * dy + 1e-12)
            a_re = a_re + thi * (-dy / d)
            a_im = a_im + thi * (dx / d)
        v_re = v_re * drag + a_re
        v_im = v_im * drag + a_im
        cur_re = cur_re + v_re * dt
        cur_im = cur_im + v_im * dt
        # Music push: uphill along the analytic cardioid normal, scaled by
        # energy. The analytic normal of p(c) = ||mu|-1| points toward the
        # boundary; computed per-frame with a finite-difference of the
        # closed form (differentiable through cur via the closed form).
        if energy is not None:
            e_i = energy.reshape(-1).float()[i]
            inner = torch.sqrt((1.0 - 4.0 * cur_re).abs() + 4.0 * cur_im * cur_im + 1e-12)
            # d(mu)/dc direction: mu = 1 - sqrt(1-4c); the proximity
            # gradient points along -d|mu|/dc toward the boundary. Use the
            # analytic direction of the cardioid inward normal: from c
            # toward the nearest boundary point ~ direction of mu itself.
            mu_re = 0.5 - (cur_re * 0.5 - cur_re * cur_re * 0.25 + cur_im * cur_im * 0.25) / (inner + 1e-12)
            mu_im = -(cur_im * 0.5 - cur_re * cur_im * 0.5) / (inner + 1e-12)
            mu_norm = torch.sqrt(mu_re * mu_re + mu_im * mu_im + 1e-12)
            push = MUSIC_PUSH_GAIN * e_i * dt
            cur_re = cur_re + (mu_re / mu_norm) * push
            cur_im = cur_im + (mu_im / mu_norm) * push
        c_re[i] = cur_re
        c_im[i] = cur_im

    return torch.complex(c_re, c_im)
