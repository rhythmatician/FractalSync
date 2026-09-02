"""Differentiable PyTorch mirrors of runtime-core's canonical math.

These functions re-implement, as tensor ops, the Rust reference
implementations in ``runtime_core``:

- :func:`synthesize_c` mirrors ``runtime_core::controller::synthesize``
  for the main cardioid (lobe=1).
- :func:`cardioid_proximity` mirrors
  ``runtime_core::proxies::mandelbrot_cardioid_proximity``.
- :func:`manifold_integrate_step` mirrors
  ``runtime_core::manifold::integrate_step`` (issue #106).

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
from dataclasses import dataclass

import numpy as np
import torch

# Golden-angle spread of residual phases. Retained only as a backward-
# compatible fallback when no explicit phases are supplied. The primary path
# now passes phases from ``runtime_core.residual_phases_for_seed_py`` so
# training and runtime share identical phase statistics.
GOLDEN_ANGLE = 2.399963229728653


def canonical_hop_dt() -> float:
    """The canonical physics timestep, derived from the deployed contract.

    NEVER restate this as a literal (e.g. ``1.0 / 60.0``). The browser
    supplies ``AnalysisTick.dtSeconds`` = HOP_LENGTH / SAMPLE_RATE from
    the Rust timebase; the trainer must advance its mirrors with the same
    value or it supervises physics the runtime does not run (the #93
    incident: parity tests advanced both paths at 1/60 while the browser
    supplied 1024/48000).

    Derived from the installed runtime_core wheel so a constant change in
    Rust automatically changes the trainer's timestep.
    """
    import runtime_core

    return runtime_core.HOP_LENGTH / runtime_core.SAMPLE_RATE


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
        phases = [float((k * GOLDEN_ANGLE) % (2.0 * np.pi)) for k in range(k_residuals)]

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


def cardioid_mu(c: torch.Tensor) -> torch.Tensor:
    """Differentiable cardioid parameterization mu = 1 - sqrt(1 - 4c).

    Mirrors ``runtime_core::proxies`` cardioid math (the same mu that
    :func:`cardioid_proximity` reduces to ``||mu| - 1|``). The angle of mu
    is the position along the cardioid boundary nearest c — the second
    perceptual axis used by region-style losses.

    Single shared implementation: other modules must import this rather
    than re-deriving ``1 - sqrt(1-4c)`` inline (architecture guardrail,
    issue #90).
    """
    inner = 1.0 - 4.0 * c
    w = torch.sqrt(inner.to(torch.complex64))
    return 1.0 - w


def cardioid_proximity(c: torch.Tensor) -> torch.Tensor:
    """Differentiable distance proxy to the main cardioid boundary.

    Mirrors ``runtime_core::proxies::mandelbrot_cardioid_proximity``:
    w = sqrt(1 − 4c); mu = 1 − w; return ||mu| − 1|. Zero on the boundary.

    .. deprecated::
        Sunset per issue #88: the minimaps (mip pyramid S field) are the
        shore-distance oracle. Use :func:`shore_proximity` instead.
    """
    mu = cardioid_mu(c)
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
    return torch.tensor(values, dtype=torch.float32, device=c.device).reshape(
        c.shape if c.dim() > 0 else ()
    )


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
    dt: float,
    c0: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Differentiable replay of PlayerState momentum integration.

    ``dt`` is REQUIRED (no default): callers must derive it from the
    canonical tick contract (``canonical_hop_dt()`` or
    ``AnalysisTick.dtSeconds``), never restate a literal. A defaulted
    ``1/60`` here is what let the trainer and browser drift apart while
    parity stayed green (#93).

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

    start_re, start_im = (
        c0 if c0 is not None else (float(tgt_re[0].detach()), float(tgt_im[0].detach()))
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
    dt: float,
) -> torch.Tensor:
    """Differentiable replay of OrbitController::step (May baseline).

    ``dt`` is REQUIRED (no default): derive it from the canonical tick
    contract (``canonical_hop_dt()``), never restate a literal.

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
    dt: float,
    drag: float = 0.90,
    thrust: float | torch.Tensor = 0.0,
    initial_c: torch.Tensor | None = None,
    energy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable replay of OrbitController::step with momentum ON.

    Mirrors the Rust momentum path (controller.rs, orbit-controller/3):
      theta += omega*dt
      target = mandelbrot_boundary(s, alpha) + residual epicycles
      a = (target - c) * 2*dt          # pull-as-acceleration
        - GRAVITY_ACCEL * c            # gravity valley (settle at origin)
        + thrust * tangent(target - c) # audio thrust (inertia)
      v = v*drag + a
      c += v*dt
      c += MUSIC_PUSH_GAIN * energy * shore_normal(c)  # NO dt factor
    c starts at the first frame's boundary point; velocity resets at
    segment boundaries. Fully differentiable w.r.t. s, alpha, band_gates.

    The shore normal is the analytic direction toward the cardioid boundary
    (mu itself, with mu = 1 - sqrt(1-4c)). Rust's
    ``cardioid_fallback_step`` sign-flips the gradient of p(c) =
    ||mu|-1| because p DECREASES toward the boundary, so the shoreward
    direction is -grad p = +mu_hat. This mirror applies that direction
    directly: same boundary, no extra dt factor (Rust's music push is a
    per-frame displacement of ``MUSIC_PUSH_GAIN * energy``, not per-
    second; matching that keeps the trainer's force balance the same as
    the runtime's). Parity vs the Rust binding is pinned by preflight
    checks (e) and (e3).
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
    if (
        initial_c is not None
        and isinstance(initial_c, torch.Tensor)
        and initial_c.numel() > 0
    ):
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
        thi = (
            thrust[i]
            if isinstance(thrust, torch.Tensor) and thrust.ndim > 0
            else thrust
        )
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
            inner = torch.sqrt(
                (1.0 - 4.0 * cur_re).abs() + 4.0 * cur_im * cur_im + 1e-12
            )
            # d(mu)/dc direction: mu = 1 - sqrt(1-4c); the proximity
            # gradient points along -d|mu|/dc toward the boundary. Use the
            # analytic direction of the cardioid inward normal: from c
            # toward the nearest boundary point ~ direction of mu itself.
            mu_re = 0.5 - (
                cur_re * 0.5 - cur_re * cur_re * 0.25 + cur_im * cur_im * 0.25
            ) / (inner + 1e-12)
            mu_im = -(cur_im * 0.5 - cur_re * cur_im * 0.5) / (inner + 1e-12)
            mu_norm = torch.sqrt(mu_re * mu_re + mu_im * mu_im + 1e-12)
            # Per-frame displacement (matches Rust MUSIC_PUSH_GAIN * energy;
            # NO dt factor — the music push is a force, not a velocity).
            push = MUSIC_PUSH_GAIN * e_i
            cur_re = cur_re + (mu_re / mu_norm) * push
            cur_im = cur_im + (mu_im / mu_norm) * push
        c_re[i] = cur_re
        c_im[i] = cur_im

    return torch.complex(c_re, c_im)


# ---------------------------------------------------------------------------
# Rust-oracle forward (non-differentiable)
#
# `orbit_controller_momentum_sequence` above is a *differentiable surrogate*
# of the runtime physics: it re-implements the analytic cardioid push in
# PyTorch so gradients can flow. The actual runtime routes the per-frame
# contour step through `runtime_core.contour_biased_step_py` (the Rust
# function the browser executes) which is NOT PyTorch-differentiable.
#
# `orbit_controller_oracle_sequence` is the non-differentiable twin: it
# runs the same momentum integrator but DELEGATES the contour step to the
# Rust binding. Use this for:
#   - parity / sanity checks (e3 in preflight)
#   - trainer forward simulation that exactly matches the deployed
#     physics (the model is still trained by gradient descent through the
#     differentiable integrator; only the contour step is non-grad)
#
# Architecture rationale: the analytic surrogate above is known to drift
# from the real field-and-fallback path in regions where the S field is
# non-uniform (analytic push always points at the cusp; real push points
# at the nearest boundary on the loaded mip pyramid). When the surrogate
# is the trainer's forward simulation, the model learns to exploit the
# surrogate's behavior — physics the browser doesn't reproduce. The
# oracle forward keeps the model honest: it sees exactly what the
# browser will see, even though the loss gradients only flow through
# the integrator and not the contour step.
# ---------------------------------------------------------------------------


def orbit_controller_oracle_sequence(
    s_target: torch.Tensor,
    alpha: torch.Tensor,
    omega: float,
    band_gates: torch.Tensor,
    segment_ids: torch.Tensor,
    dt: float,
    drag: float = 0.90,
    thrust: float | torch.Tensor = 0.0,
    initial_c: torch.Tensor | None = None,
    energy: torch.Tensor | None = None,
    h: torch.Tensor | None = None,
    level: int = 0,
    d_star: float = 0.5,
    max_step: float = 0.05,
) -> torch.Tensor:
    """Non-differentiable replay of the Rust OrbitController's
    momentum+shore_bias path. Routes the per-frame contour step through
    ``runtime_core.contour_biased_step_py`` so the forward trajectory
    exactly matches what the browser executes.

    The integrator part (accel from target-gravity, velocity drag, thrust)
    is still differentiable w.r.t. s, alpha, band_gates because those
    flow through PyTorch ops before the contour step. The contour step
    itself is detached (it sits behind a PyO3 boundary) so the gradient
    stops there — but the model still learns, because the integrator
    dominates the signal: the contour step is mostly a small correction
    on top of the integrator's drift.

    Args mirror ``orbit_controller_momentum_sequence`` with three
    additions for the shore-bias path: ``h`` (transient signal, [0,1]),
    ``level`` (mip level for the contour step), and ``d_star`` (target
    shore-proximity for the servo).
    """
    # Note: ``runtime_core`` is imported lazily inside
    # ``_ContourStep.forward`` (line ~746) where the actual binding is
    # used. Importing here would be a no-op binding lookup that ruff
    # flags as F401.
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

    c_re = torch.zeros(n, device=device, dtype=torch.float32)
    c_im = torch.zeros(n, device=device, dtype=torch.float32)
    v_re = torch.zeros((), device=device, dtype=torch.float32)
    v_im = torch.zeros((), device=device, dtype=torch.float32)

    # Per-frame energy / h schedules. Default to 0 if not supplied so
    # the oracle matches the no-shore-bias path when those signals are
    # missing (e.g. parity regression suites).
    if energy is None:
        energy_t = torch.zeros(n, device=device, dtype=torch.float32)
    else:
        energy_t = energy.reshape(-1).float()
    if h is None:
        h_t = torch.zeros(n, device=device, dtype=torch.float32)
    else:
        h_t = h.reshape(-1).float()

    ic: torch.Tensor | None = None
    if (
        initial_c is not None
        and isinstance(initial_c, torch.Tensor)
        and initial_c.numel() > 0
    ):
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
        thi = (
            thrust[i]
            if isinstance(thrust, torch.Tensor) and thrust.ndim > 0
            else thrust
        )
        if isinstance(thi, torch.Tensor):
            thi = float(thi.item())
        if thi != 0.0 and thi > 0.0:
            d = torch.sqrt(dx * dx + dy * dy + 1e-12)
            a_re = a_re + thi * (-dy / d)
            a_im = a_im + thi * (dx / d)
        v_re = v_re * drag + a_re
        v_im = v_im * drag + a_im
        proposed_re = v_re * dt
        proposed_im = v_im * dt

        e_i = energy_t[i] if i < energy_t.shape[0] else torch.tensor(0.0, device=device)
        h_i = h_t[i] if i < h_t.shape[0] else torch.tensor(0.0, device=device)
        d_star_t = torch.tensor(d_star, device=device, dtype=torch.float32)
        max_step_t = torch.tensor(max_step, device=device, dtype=torch.float32)
        level_t = torch.tensor(float(level), device=device, dtype=torch.float32)
        # DEFER to the Rust binding: this is the whole point of the
        # oracle. ``_ContourStep`` is a torch.autograd.Function whose
        # forward calls the real Rust ``contour_biased_step_py`` and
        # whose backward uses an identity surrogate (gradients pass
        # through unchanged) — sufficient for the integrator above
        # to learn s/alpha/thrust adjustments, and exact for the
        # forward trajectory.
        delta_re, delta_im = _ContourStep.apply(
            cur_re,
            cur_im,
            proposed_re,
            proposed_im,
            h_i,
            d_star_t,
            max_step_t,
            level_t,
            e_i,
        )
        # ``+`` keeps the autograd graph connected: the integrator
        # above (target - c) -> c -> losses backpropagates through
        # ``cur_re + delta_re`` to cur_re's history.
        cur_re = cur_re + delta_re
        cur_im = cur_im + delta_im
        c_re[i] = cur_re
        c_im[i] = cur_im

    return torch.complex(c_re, c_im)


class _ContourStep(torch.autograd.Function):
    """Custom autograd bridge for ``contour_biased_step_py``.

    The forward call hits the Rust binding (so the trajectory is
    bit-for-bit the runtime physics the browser executes). The
    backward uses an identity surrogate — gradients of the loss
    with respect to the contour step's output pass through
    unchanged — which is enough for the integrator above
    (target − c, gravity, thrust) to propagate learning signal
    back to s, alpha, omega_scale, and band_gates. The contour
    step is treated as a fixed physics function the model learns
    to anticipate, not to optimize through (the true gradient
    is undefined across a PyO3 boundary).
    """

    @staticmethod
    def forward(
        ctx,
        c_re: torch.Tensor,
        c_im: torch.Tensor,
        u_re: torch.Tensor,
        u_im: torch.Tensor,
        h: torch.Tensor,
        d_star: torch.Tensor,
        max_step: torch.Tensor,
        level: torch.Tensor,
        energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import runtime_core

        rust_step = getattr(runtime_core, "contour_biased_step_py", None)
        if rust_step is None:
            # Binding unavailable (e.g. fake runtime_core in lightweight test
            # environments): fall back to the proposed step unchanged so the
            # integrator still produces a deterministic trajectory.
            new_re = float(c_re.item()) + float(u_re.item())
            new_im = float(c_im.item()) + float(u_im.item())
        else:
            new_re, new_im = rust_step(
                float(c_re.item()),
                float(c_im.item()),
                float(u_re.item()),
                float(u_im.item()),
                float(h.item()),
                float(d_star.item()),
                float(max_step.item()),
                int(level.item()),
                float(energy.item()),
            )
        # Save the *deltas* so the autograd graph has something to
        # backprop through: the backward computes gradient of the
        # loss with respect to the integrator inputs (c_re, c_im,
        # u_re, u_im) by routing the upstream grad through identity.
        cur_re_f = float(c_re.item())
        cur_im_f = float(c_im.item())
        ctx.save_for_backward(
            torch.tensor(new_re - cur_re_f, device=c_re.device, dtype=torch.float32),
            torch.tensor(new_im - cur_im_f, device=c_re.device, dtype=torch.float32),
        )
        return (
            torch.tensor(new_re - cur_re_f, device=c_re.device, dtype=torch.float32),
            torch.tensor(new_im - cur_im_f, device=c_re.device, dtype=torch.float32),
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_delta_re: torch.Tensor, grad_delta_im: torch.Tensor
    ):
        # Identity surrogate: gradient of the loss with respect to
        # the new c equals the gradient of the loss with respect to
        # the old c (the contour step is treated as a small constant
        # correction that doesn't change the gradient direction).
        grad_c_re = grad_delta_re
        grad_c_im = grad_delta_im
        grad_u_re = grad_delta_re
        grad_u_im = grad_delta_im
        # The remaining args are not differentiated (they're controls
        # that the Rust step uses internally but the trainer does
        # not optimize through).
        return grad_c_re, grad_c_im, grad_u_re, grad_u_im, None, None, None, None, None


# ---------------------------------------------------------------------------
# Manifold physics (issue #106) — training mirror.
#
# Mirrors ``runtime_core::manifold::integrate_step`` so gradients flow
# through the full differential geometry (metric, Christoffel, potential,
# forces). The Rust integrator is the canonical source of truth; the
# trainer's differentiable surrogate must reproduce it within tolerance.
# ---------------------------------------------------------------------------


@dataclass
class ManifoldConfig:
    """Mirror of ``runtime_core::manifold::ManifoldConfig`` (issue #106).

    Defaults match the Rust ``Default`` impl exactly; the trainer must not
    invent its own values (Rust-first parity, ADR 0001).
    """

    d_ref: float = 0.1
    epsilon: float = 1e-4
    lambda_sq: float = 1.0
    kappa: float = 1.0


@dataclass
class ManifoldEnergyInfo:
    """Mirror of ``runtime_core::manifold::EnergyInfo``."""

    kinetic: float
    potential: float
    total: float
    delta_total: float
    delta_kinetic: float


def manifold_integrate_step(
    c_re: torch.Tensor,
    c_im: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    qx: torch.Tensor,
    qy: torch.Tensor,
    beta: float,
    dt: float,
    config: ManifoldConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ManifoldEnergyInfo]:
    """Replay of ``runtime_core::manifold::integrate_step`` (issue #106).

    The forward call hits the Rust binding (scalar signature:
    ``manifold_integrate_step(c_re, c_im, vx, vy, qx, qy, beta, dt,
    ManifoldConfig)``) so the trajectory is bit-for-bit the runtime
    physics. Gradients pass through via the identity surrogate in
    :class:`_ManifoldStep` — the same pattern as the contour-step bridge
    above: the integrator dominates the gradient signal, and the true
    gradient is undefined across a PyO3 boundary.

    Returns ``(c_new_re, c_new_im, v_new_x, v_new_y, energy_info)`` where
    the first four are 0-dim tensors on the input device and
    ``energy_info`` is a :class:`ManifoldEnergyInfo` diagnostic computed
    from the Rust energy bindings (same quantities the Rust
    ``integrate_step`` reports).
    """
    device = c_re.device
    new_re, new_im, new_vx, new_vy = _ManifoldStep.apply(
        c_re.reshape(()),
        c_im.reshape(()),
        vx.reshape(()),
        vy.reshape(()),
        qx.reshape(()),
        qy.reshape(()),
        torch.tensor(float(beta), device=device, dtype=torch.float32),
        torch.tensor(float(dt), device=device, dtype=torch.float32),
        config,
    )
    # Energy diagnostics from the Rust bindings (detached, same math as
    # runtime_core::manifold::integrate_step's EnergyInfo).
    import runtime_core

    rc_config = runtime_core.ManifoldConfig(
        config.d_ref, config.epsilon, config.lambda_sq, config.kappa
    )
    c_new = complex(new_re.detach().item(), new_im.detach().item())
    c_old = complex(float(c_re.detach()), float(c_im.detach()))
    v_new = (new_vx.detach().item(), new_vy.detach().item())
    v_old = (float(vx.detach()), float(vy.detach()))
    k_new = runtime_core.manifold_kinetic_energy(v_new[0], v_new[1], c_new, rc_config)
    k_old = runtime_core.manifold_kinetic_energy(v_old[0], v_old[1], c_old, rc_config)
    u_new = runtime_core.manifold_potential_energy(c_new, rc_config)
    u_old = runtime_core.manifold_potential_energy(c_old, rc_config)
    e_new = k_new + u_new
    e_old = k_old + u_old
    info = ManifoldEnergyInfo(
        kinetic=k_new,
        potential=u_new,
        total=e_new,
        delta_total=e_new - e_old,
        delta_kinetic=k_new - k_old,
    )
    return new_re, new_im, new_vx, new_vy, info


class _ManifoldStep(torch.autograd.Function):
    """Custom autograd bridge for ``runtime_core.manifold_integrate_step``.

    Forward calls the Rust scalar binding; backward routes the upstream
    gradient through identity (same surrogate as :class:`_ContourStep`).
    The Rust binding returns ``(new_re, new_im, new_vx, new_vy, EnergyInfo)``;
    we return the deltas so ``old + delta`` keeps the autograd graph
    connected, mirroring the contour-step bridge.
    """

    @staticmethod
    def forward(
        ctx,
        c_re: torch.Tensor,
        c_im: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        qx: torch.Tensor,
        qy: torch.Tensor,
        beta: torch.Tensor,
        dt: torch.Tensor,
        config: ManifoldConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        import runtime_core

        rust_step = getattr(runtime_core, "manifold_integrate_step", None)
        if rust_step is None:
            # Binding unavailable (e.g. fake runtime_core in lightweight
            # test environments): fall back to plain semi-implicit Euler
            # with no forces so the trajectory stays deterministic.
            new_re = float(c_re.item()) + float(vx.item()) * float(dt.item())
            new_im = float(c_im.item()) + float(vy.item()) * float(dt.item())
            new_vx = float(vx.item())
            new_vy = float(vy.item())
            info = ManifoldEnergyInfo(0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            rc_config = runtime_core.ManifoldConfig(
                config.d_ref, config.epsilon, config.lambda_sq, config.kappa
            )
            new_re, new_im, new_vx, new_vy, rc_info = rust_step(
                float(c_re.item()),
                float(c_im.item()),
                float(vx.item()),
                float(vy.item()),
                float(qx.item()),
                float(qy.item()),
                float(beta.item()),
                float(dt.item()),
                rc_config,
            )
            info = ManifoldEnergyInfo(
                kinetic=rc_info.kinetic,
                potential=rc_info.potential,
                total=rc_info.total,
                delta_total=rc_info.delta_total,
                delta_kinetic=rc_info.delta_kinetic,
            )
        device = c_re.device
        ctx.manifold_info = info
        return (
            torch.tensor(new_re, device=device, dtype=torch.float32),
            torch.tensor(new_im, device=device, dtype=torch.float32),
            torch.tensor(new_vx, device=device, dtype=torch.float32),
            torch.tensor(new_vy, device=device, dtype=torch.float32),
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_re: torch.Tensor,
        grad_im: torch.Tensor,
        grad_vx: torch.Tensor,
        grad_vy: torch.Tensor,
    ):
        # Identity surrogate: gradient w.r.t. the new state routes back to
        # the old state. The c-gradient is ALSO routed to the generalized
        # force (qx, qy): the force direction is what carries the learning
        # signal from the model's (s, alpha) target into the trajectory,
        # so dropping it would sever the gradient path entirely (the
        # target only affects c through Q). beta, dt, and config are not
        # differentiated.
        return grad_re, grad_im, grad_vx, grad_vy, grad_re, grad_im, None, None, None


def orbit_controller_manifold_sequence(
    s_target: torch.Tensor,
    alpha: torch.Tensor,
    omega: float,
    band_gates: torch.Tensor,
    segment_ids: torch.Tensor,
    dt: float,
    energy: torch.Tensor | None = None,
    initial_c: torch.Tensor | None = None,
    initial_v: tuple[float, float] = (0.0, 0.0),
    manifold_drag: float = 0.1,
    config: ManifoldConfig | None = None,
) -> tuple[torch.Tensor, list[ManifoldEnergyInfo]]:
    """Differentiable replay of ``OrbitController::step_manifold`` (#106).

    Mirrors the Rust manifold path per frame:
      theta += omega*dt
      target = mandelbrot_boundary(s, alpha) + residual epicycles
      Q = force_mag * dt * (target - c) / |target - c|   # generalized force
        with force_mag = clamp(omega, 0.1, 10) * clamp(energy, 0, 1) * 2
      (c, v) = manifold_integrate_step(c, v, Q, beta, dt, config)

    The target/residual/force-direction math is pure PyTorch (fully
    differentiable w.r.t. s, alpha, band_gates); the integrator itself
    routes through the Rust binding via :class:`_ManifoldStep` with an
    identity-surrogate backward — the same architecture as the
    shore-bias oracle path.

    Returns ``(trajectory, energy_infos)``: a complex tensor of shape (N,)
    and the per-frame :class:`ManifoldEnergyInfo` diagnostics.
    """
    import runtime_core  # noqa: F401 - availability check for early failure

    cfg = config if config is not None else ManifoldConfig()
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

    if energy is None:
        energy_t = torch.zeros(n, device=device, dtype=torch.float32)
    else:
        energy_t = energy.reshape(-1).float().clamp(0.0, 1.0)

    # Force magnitude: clamp(omega, 0.1, 10) * clamp(energy, 0, 1) * 2
    # (matches controller.rs step_manifold exactly).
    force_mag = max(0.1, min(10.0, omega)) * 2.0

    c_re = torch.zeros(n, device=device, dtype=torch.float32)
    c_im = torch.zeros(n, device=device, dtype=torch.float32)

    ic: torch.Tensor | None = None
    if (
        initial_c is not None
        and isinstance(initial_c, torch.Tensor)
        and initial_c.numel() > 0
    ):
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

    v_re = torch.tensor(initial_v[0], device=device, dtype=torch.float32)
    v_im = torch.tensor(initial_v[1], device=device, dtype=torch.float32)

    infos: list[ManifoldEnergyInfo] = []
    for i in range(n):
        if seg_boundary[i]:
            v_re = torch.zeros_like(v_re)
            v_im = torch.zeros_like(v_im)
            if ic is not None and ic.is_complex() and ic.numel() == n:
                cur_re = ic[i].real.float()
                cur_im = ic[i].imag.float()
        dx = tgt_re[i] - cur_re
        dy = tgt_im[i] - cur_im
        dist = torch.sqrt(dx * dx + dy * dy)
        # Q = force_mag * dt * unit(target - c); zero when at the target.
        if float(dist.detach()) > 1e-9:
            e_i = energy_t[i]
            q_re = (dx / dist) * (force_mag * e_i * dt)
            q_im = (dy / dist) * (force_mag * e_i * dt)
        else:
            q_re = torch.zeros((), device=device, dtype=torch.float32)
            q_im = torch.zeros((), device=device, dtype=torch.float32)

        new_re, new_im, v_re, v_im, info = manifold_integrate_step(
            cur_re,
            cur_im,
            v_re,
            v_im,
            q_re,
            q_im,
            beta=manifold_drag,
            dt=dt,
            config=cfg,
        )
        infos.append(info)
        cur_re = new_re
        cur_im = new_im
        c_re[i] = cur_re
        c_im[i] = cur_im

    return torch.complex(c_re, c_im), infos
