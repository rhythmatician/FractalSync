"""
Thin helpers around the runtime_core Rust bindings.

This module centralises the shared constants and constructors so the
backend never drifts from the Rust source of truth. Frontend must use
matching values exposed by the wasm bindings.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence
import logging

import runtime_core as rc

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = rc.SAMPLE_RATE
HOP_LENGTH: int = rc.HOP_LENGTH
N_FFT: int = rc.N_FFT
WINDOW_FRAMES: int = rc.WINDOW_FRAMES
DEFAULT_K_RESIDUALS: int = rc.DEFAULT_K_RESIDUALS
DEFAULT_RESIDUAL_CAP: float = rc.DEFAULT_RESIDUAL_CAP
DEFAULT_RESIDUAL_OMEGA_SCALE: float = rc.DEFAULT_RESIDUAL_OMEGA_SCALE
DEFAULT_BASE_OMEGA: float = rc.DEFAULT_BASE_OMEGA
DEFAULT_ORBIT_SEED: int = rc.DEFAULT_ORBIT_SEED


def make_feature_extractor(
    include_delta: bool = False,
    include_delta_delta: bool = False,
) -> rc.FeatureExtractor:
    """Create a FeatureExtractor configured with the shared defaults.

    NOTE: the previous Python fallback was removed; the Rust implementation
    is now the single source of truth. If the runtime_core bindings fail to
    build or the feature extractor is broken, tests/CI will fail so we can
    address the root cause promptly.
    """
    return rc.FeatureExtractor(
        SAMPLE_RATE,
        HOP_LENGTH,
        N_FFT,
        include_delta,
        include_delta_delta,
    )


def make_residual_params(
    k_residuals: int = DEFAULT_K_RESIDUALS,
    residual_cap: float = DEFAULT_RESIDUAL_CAP,
    radius_scale: float = 1.0,
) -> rc.ResidualParams:
    return rc.ResidualParams(
        k_residuals=k_residuals,
        residual_cap=residual_cap,
        radius_scale=radius_scale,
    )


def make_lobe_state(n_lobes: int = 2) -> rc.LobeState:
    """Construct a Rust-backed LobeState via runtime_core bindings.

    This replaces the former Python fallback `src.lobe_state` so the
    runtime-core implementation is the single source of truth.
    """
    return rc.LobeState(n_lobes)


def make_orbit_state(
    *,
    lobe: int = 1,
    sub_lobe: int = 0,
    theta: float = 0.0,
    omega: float = DEFAULT_BASE_OMEGA,
    s: float = 1.02,
    alpha: float = 0.3,
    k_residuals: int = DEFAULT_K_RESIDUALS,
    residual_omega_scale: float = DEFAULT_RESIDUAL_OMEGA_SCALE,
    seed: Optional[int] = DEFAULT_ORBIT_SEED,
) -> rc.OrbitState:
    """Construct a deterministic orbit state using the Rust implementation."""
    return rc.OrbitState(
        lobe=lobe,
        sub_lobe=sub_lobe,
        theta=theta,
        omega=omega,
        s=s,
        alpha=alpha,
        k_residuals=k_residuals,
        residual_omega_scale=residual_omega_scale,
        seed=seed,
    )


def step_orbit(
    state: rc.OrbitState,
    dt: float,
    residual_params: Optional[rc.ResidualParams] = None,
    band_gates: Optional[Sequence[float]] = None,
    distance_field: Optional[object] = None,
    h: float = 0.0,
    d_star: Optional[float] = None,
    max_step: Optional[float] = None,
) -> rc.Complex:
    """
    Step the orbit forward by dt, passing band gates and optionally a distance field
    and contour integrator options.

    This wrapper adapts to whichever binding signature is available on the installed
    runtime_core extension (backwards compatible).
    """
    rp = residual_params or make_residual_params()
    gates = list(band_gates) if band_gates is not None else None

    # Try the full signature first
    try:
        return state.step(dt, rp, gates, distance_field, h, d_star, max_step)
    except TypeError:
        # Old bindings: try without integrator args
        try:
            return state.step(dt, rp, gates, distance_field)
        except TypeError:
            # Fallback to older signature without distance_field
            return state.step(dt, rp, gates)


def synthesize(
    state: rc.OrbitState,
    residual_params: Optional[rc.ResidualParams] = None,
    band_gates: Optional[Iterable[float]] = None,
) -> rc.Complex:
    rp = residual_params or make_residual_params()
    return state.synthesize(rp, list(band_gates) if band_gates is not None else None)
