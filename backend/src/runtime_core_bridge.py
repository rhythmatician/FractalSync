"""
Thin helpers around the runtime_core Rust bindings.

This module centralises the shared constants and constructors so the
backend never drifts from the Rust source of truth. Frontend must use
matching values exposed by the wasm bindings.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union
import logging

import runtime_core as rc
from .python_feature_extractor import PythonFeatureExtractor

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
) -> Union[rc.FeatureExtractor, PythonFeatureExtractor]:
    """Create a FeatureExtractor configured with the shared defaults.

    TODO(CRITICAL): Fix Rust extractor hanging bug
    Issue: runtime_core.FeatureExtractor.extract_windowed_features() hangs indefinitely
    Root cause: Unknown - likely PyO3 parameter conversion or GIL deadlock
    Tested: Hangs on inputs as small as 10 samples, all other Rust functions work
    Workaround: Python fallback extractor (this function returns PythonFeatureExtractor)

    To fix:
    1. Debug PyO3 Vec<f32> parameter conversion (might need numpy integration)
    2. Check for infinite loops in features.rs extract_features()
    3. Verify GIL is properly released during computation
    4. Test with minimal reproducible example in Rust unit tests

    Once fixed, change this function to return rc.FeatureExtractor directly.
    """
    logger.warning(
        "Using Python fallback feature extractor due to Rust implementation bug. "
        "Performance will be degraded. See TODO in runtime_core_bridge.py"
    )
    return PythonFeatureExtractor(
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        include_delta=include_delta,
        include_delta_delta=include_delta_delta,
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
) -> rc.Complex:
    """
    Step the orbit forward by dt, passing band gates and optionally a distance field
    if supported by the installed runtime_core extension.

    This wrapper tries the 4-arg form first (with distance_field) and falls back to
    the 3-arg form if the extension doesn't accept the extra parameter.
    """
    rp = residual_params or make_residual_params()
    gates = list(band_gates) if band_gates is not None else None
    try:
        # Prefer signature with distance_field if available
        return state.step(dt, rp, gates, distance_field)  # type: ignore[arg-type]
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
