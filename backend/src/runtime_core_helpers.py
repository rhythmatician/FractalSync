"""Helpers that call into the `runtime_core` native extension directly.

This module provides small wrappers that replace the legacy
`runtime_core_bridge` surface. They are intended to be a minimal,
stable, and easily-reviewable migration target while we phase out the
bridge module.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

import runtime_core as rc

logger = logging.getLogger(__name__)

SAMPLE_RATE = rc.SAMPLE_RATE
HOP_LENGTH = rc.HOP_LENGTH
N_FFT = rc.N_FFT
WINDOW_FRAMES = rc.WINDOW_FRAMES
DEFAULT_K_RESIDUALS = rc.DEFAULT_K_RESIDUALS
DEFAULT_RESIDUAL_CAP = rc.DEFAULT_RESIDUAL_CAP
DEFAULT_RESIDUAL_OMEGA_SCALE = rc.DEFAULT_RESIDUAL_OMEGA_SCALE
DEFAULT_BASE_OMEGA = rc.DEFAULT_BASE_OMEGA
DEFAULT_ORBIT_SEED = rc.DEFAULT_ORBIT_SEED


def _rust_extractor_sanity_check(
    include_delta: bool, include_delta_delta: bool, timeout: float = 2.0
) -> bool:
    import sys
    import subprocess

    code = (
        "import runtime_core as rc, sys\n"
        f"fe = rc.FeatureExtractor(sr={SAMPLE_RATE}, hop_length={HOP_LENGTH}, n_fft={N_FFT}, include_delta={include_delta}, include_delta_delta={include_delta_delta})\n"
        f"samples = [0.0]*{max(16, HOP_LENGTH)}\n"
        f"res = fe.extract_windowed_features(samples, {WINDOW_FRAMES})\n"
        "print('RUST_SANITY_OK')\n"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0 and "RUST_SANITY_OK" in proc.stdout:
            return True
        logger.warning(
            "Rust extractor sanity subprocess failed: rc=%s stdout=%r stderr=%r",
            proc.returncode,
            proc.stdout,
            proc.stderr,
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning(
            "Rust extractor sanity subprocess timed out after %s seconds", timeout
        )
        return False
    except Exception as exc:
        logger.exception("Unexpected error while probing Rust extractor: %s", exc)
        return False


class FeatureExtractorProxy:
    """Thin adapter around `rc.FeatureExtractor` to provide the same
    `feature_mean`/`feature_std` attributes expected by the backend.
    """

    def __init__(self, fe: rc.FeatureExtractor):
        self._fe = fe
        self.feature_mean: Optional[NDArray[np.float64]] = None
        self.feature_std: Optional[NDArray[np.float64]] = None

    def num_features_per_frame(self) -> int:
        return self._fe.num_features_per_frame()

    def extract_windowed_features(
        self,
        audio: Sequence[float] | NDArray[np.floating],
        window_frames: int,
    ) -> NDArray[np.float64]:
        result = self._fe.extract_windowed_features(list(audio), window_frames)
        return np.array(result, dtype=np.float64)

    # Provide normalization helpers matching the old bridge API so callers
    # don't need to change during migration.
    def compute_normalization_stats(
        self, all_features: list[NDArray[np.float64]]
    ) -> None:
        if not all_features:
            return
        concatenated = np.concatenate(all_features, axis=0)
        self.feature_mean = np.mean(concatenated, axis=0)
        self.feature_std = np.std(concatenated, axis=0) + 1e-8

    def normalize_features(
        self, features: Sequence[float] | NDArray[np.floating]
    ) -> NDArray[np.float64]:
        array = np.asarray(features, dtype=np.float64)
        if self.feature_mean is None or self.feature_std is None:
            return array
        return (array - self.feature_mean) / self.feature_std


def make_feature_extractor(
    include_delta: bool = False, include_delta_delta: bool = False
) -> FeatureExtractorProxy:
    """Create a runtime_core.FeatureExtractor and wrap it with a small
    proxy object. We run a short sanity subprocess check to avoid calling
    the Rust extractor from the main process if it is known to hang.
    Raises RuntimeError if the Rust extractor fails the sanity check.
    """
    if _rust_extractor_sanity_check(include_delta, include_delta_delta):
        logger.info("Using Rust FeatureExtractor for extraction (sanity check passed)")
        fe = rc.FeatureExtractor(
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            include_delta=include_delta,
            include_delta_delta=include_delta_delta,
        )
        return FeatureExtractorProxy(fe)
    logger.error(
        "Rust FeatureExtractor sanity check failed; refusing to fall back to Python extractor"
    )
    raise RuntimeError(
        "Rust FeatureExtractor sanity check failed. "
        "This indicates a problem with the runtime_core native extension. "
        "Rebuild with 'maturin develop --release' from the runtime-core directory, "
        "or investigate the Rust extractor hang. Fallback to Python extractor is disabled by policy."
    )


# Simple convenience wrappers mirroring the bridge
def make_residual_params(
    k_residuals: int = DEFAULT_K_RESIDUALS,
    residual_cap: float = DEFAULT_RESIDUAL_CAP,
    radius_scale: float = 1.0,
) -> rc.ResidualParams:
    return rc.ResidualParams(
        k_residuals=k_residuals, residual_cap=residual_cap, radius_scale=radius_scale
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
    if seed is None:
        return rc.OrbitState(
            lobe, sub_lobe, theta, omega, s, alpha, k_residuals, residual_omega_scale
        )
    if hasattr(rc.OrbitState, "new_with_seed"):
        return rc.OrbitState.new_with_seed(
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
    state: rc.OrbitState,
    dt: float,
    residual_params: Optional[rc.ResidualParams] = None,
    band_gates: Optional[Sequence[float]] = None,
) -> complex:
    rp = residual_params or make_residual_params()
    return state.step(
        dt, rp, band_gates=list(band_gates) if band_gates is not None else None
    )  # type: ignore[return-value]  # PyO3 returns a Complex object


def synthesize(
    state: rc.OrbitState,
    residual_params: Optional[rc.ResidualParams] = None,
    band_gates: Optional[Iterable[float]] = None,
) -> complex:
    rp = residual_params or make_residual_params()
    return state.synthesize(rp, list(band_gates) if band_gates is not None else None)  # type: ignore[return-value]  # PyO3 returns a Complex
