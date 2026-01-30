"""
Thin helpers around the runtime_core Rust bindings.

This module centralises the shared constants and constructors so the
backend never drifts from the Rust source of truth. Frontend must use
matching values exposed by the wasm bindings.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence
import logging
import numpy as np

from runtime_core import (
    Complex,
    FeatureExtractor,
    ResidualParams,
    OrbitState,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    DEFAULT_K_RESIDUALS,
    DEFAULT_RESIDUAL_CAP,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    DEFAULT_BASE_OMEGA,
    DEFAULT_ORBIT_SEED,
)

logger = logging.getLogger(__name__)


class FeatureExtractorBridge:
    """Adapter that provides a stable feature extractor API.

    - Uses the Rust extractor for `extract_windowed_features` when available
      and verified by the sanity check (for performance).
    - Always uses the Python extractor for normalization-related methods
      (compute_normalization_stats, normalize_features) to ensure functionality
      parity and avoid placing extra surface area in the Rust code while the
      extractor issue is being investigated.
    """

    def __init__(
        self,
        rust_extractor: FeatureExtractor,
    ) -> None:
        self._rust = rust_extractor
        self.feature_mean = None
        self.feature_std = None

    # Basic shape/info
    def num_features_per_frame(self) -> int:
        return int(self._rust.num_features_per_frame())

    # Extraction: uses Rust extractor; raises RuntimeError if unavailable or fails
    def extract_windowed_features(self, audio, window_frames: int):
        result = self._rust.extract_windowed_features(list(audio), window_frames)
        return np.array(result, dtype=np.float64)

    # Normalization helpers
    def compute_normalization_stats(self, all_features: list):
        """Compute mean and std for normalization across dataset."""
        if not all_features:
            return

        concatenated = np.concatenate(all_features, axis=0)
        self.feature_mean = np.mean(concatenated, axis=0)
        self.feature_std = np.std(concatenated, axis=0) + 1e-8

    def normalize_features(self, features):
        """Normalize features using computed stats."""
        if self.feature_mean is None or self.feature_std is None:
            return features
        return (features - self.feature_mean) / self.feature_std


def make_feature_extractor(
    include_delta: bool = False,
    include_delta_delta: bool = False,
) -> FeatureExtractorBridge:
    """Create a FeatureExtractor configured with the shared defaults."""

    logger.info("Using Rust FeatureExtractor for extraction)")
    rust_fx = FeatureExtractor(
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        include_delta=include_delta,
        include_delta_delta=include_delta_delta,
    )
    return FeatureExtractorBridge(rust_fx)


def make_residual_params(
    k_residuals: int = DEFAULT_K_RESIDUALS,
    residual_cap: float = DEFAULT_RESIDUAL_CAP,
    radius_scale: float = 1.0,
) -> ResidualParams:
    return ResidualParams(
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
) -> OrbitState:
    """Construct a deterministic orbit state using the Rust implementation.

    Use positional arguments to avoid relying on a keyword name that may not
    be present in all generated Python bindings. If `seed` is provided the
    Rust constructor that accepts a seed will be used.
    """
    if seed is None:
        return OrbitState(
            lobe,
            sub_lobe,
            theta,
            omega,
            s,
            alpha,
            k_residuals,
            residual_omega_scale,
        )
    # Prefer explicit constructor that accepts a seed when available
    if hasattr(OrbitState, "new_with_seed"):
        return OrbitState.new_with_seed(
            lobe,
            sub_lobe,
            theta,
            omega,
            s,
            alpha,
            k_residuals,
            residual_omega_scale,
            seed,
        )
    raise RuntimeError(
        "make_orbit_state: seed provided but OrbitState.new_with_seed() not available"
    )


def step_orbit(
    state: OrbitState,
    dt: float,
    residual_params: Optional[ResidualParams] = None,
    band_gates: Optional[Sequence[float]] = None,
) -> Complex:
    rp = residual_params or make_residual_params()
    return state.step(
        dt, rp, band_gates=list(band_gates) if band_gates is not None else None
    )


def synthesize(
    state: OrbitState,
    residual_params: Optional[ResidualParams] = None,
    band_gates: Optional[Iterable[float]] = None,
) -> Complex:
    rp = residual_params or make_residual_params()
    return state.synthesize(rp, list(band_gates) if band_gates is not None else None)
