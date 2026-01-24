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
import numpy as np


class FeatureExtractorAdapter:
    """Adapter to present the same interface as PythonFeatureExtractor while
    delegating to the Rust-backed `rc.FeatureExtractor` for heavy lifting.
    """

    def __init__(self, inner: rc.FeatureExtractor):
        self._inner = inner
        self.feature_mean = None
        self.feature_std = None

    def num_features_per_frame(self) -> int:
        return self._inner.num_features_per_frame()

    def extract_windowed_features(self, audio, window_frames: int):
        # Rust binding returns nested lists; convert to numpy array
        windows = self._inner.extract_windowed_features(list(audio), window_frames)
        if len(windows) == 0:
            return np.empty(
                (0, self.num_features_per_frame() * window_frames), dtype=np.float64
            )
        return np.array(windows, dtype=np.float64)

    def compute_normalization_stats(self, all_features: list[np.ndarray]):
        if not all_features:
            return
        concatenated = np.concatenate(all_features, axis=0)
        self.feature_mean = np.mean(concatenated, axis=0)
        self.feature_std = np.std(concatenated, axis=0) + 1e-8

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return features
        return (features - self.feature_mean) / self.feature_std


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
) -> FeatureExtractorAdapter:
    """Create a FeatureExtractor configured with the shared defaults.

    This returns an adapter exposing the same Python-friendly API as the
    original `PythonFeatureExtractor` (notably `compute_normalization_stats`) but
    delegates the heavy-lifting to the Rust implementation.
    """
    inner = rc.FeatureExtractor(
        SAMPLE_RATE,
        HOP_LENGTH,
        N_FFT,
        include_delta,
        include_delta_delta,
    )
    return FeatureExtractorAdapter(inner)


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

    # Call runtime-core's OrbitState.step with the full, canonical signature.
    return state.step(dt, rp, gates, distance_field, h, d_star, max_step)


def synthesize(
    state: rc.OrbitState,
    residual_params: Optional[rc.ResidualParams] = None,
    band_gates: Optional[Iterable[float]] = None,
) -> rc.Complex:
    rp = residual_params or make_residual_params()
    return state.synthesize(rp, list(band_gates) if band_gates is not None else None)
