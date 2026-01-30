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


def _rust_extractor_sanity_check(
    include_delta: bool, include_delta_delta: bool, timeout: float = 2.0
) -> bool:
    """Attempt to construct and exercise the Rust FeatureExtractor in a subprocess.

    Using `subprocess` avoids Windows handle duplication / spawn issues (e.g.
    ``OSError: [WinError 6] The handle is invalid``) and bypasses pickling
    limitations that arise when passing local functions or complex objects to
    `multiprocessing.Process`.

    Returns True if the child process completes successfully within `timeout`.
    """
    import sys
    import subprocess

    # Small, self-contained python snippet executed in a fresh process.
    code = (
        "import runtime_core as rc, sys\n"
        f"fe = rc.FeatureExtractor(sr={SAMPLE_RATE}, hop_length={HOP_LENGTH}, n_fft={N_FFT}, include_delta={include_delta}, include_delta_delta={include_delta_delta})\n"
        f"samples = [0.0]*{max(16, HOP_LENGTH)}\n"
        f"res = fe.extract_windowed_features(samples, {WINDOW_FRAMES})\n"
        "# If we reach here the Rust extractor executed successfully\n"
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
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error while probing Rust extractor: %s", exc)
        return False


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
        rust_extractor: Optional[rc.FeatureExtractor],
    ) -> None:
        self._rust = rust_extractor
        self.feature_mean = None
        self.feature_std = None

    # Basic shape/info
    def num_features_per_frame(self) -> int:
        if self._rust is not None:
            return int(self._rust.num_features_per_frame())
        raise RuntimeError("num_features_per_frame: Rust extractor not available")

    # Extraction: uses Rust extractor; raises RuntimeError if unavailable or fails
    def extract_windowed_features(self, audio, window_frames: int):
        # Accept numpy arrays or Python sequences
        if self._rust is not None:
            try:
                # Rust binding expects a sequence of floats; list() is safe for numpy arrays
                result = self._rust.extract_windowed_features(
                    list(audio), window_frames
                )
                # Rust returns a list-of-lists (Vec<Vec<f64>>); convert to numpy array
                return np.array(result, dtype=np.float64)
            except Exception:
                raise RuntimeError("extract_windowed_features: Rust extractor failed")
        raise RuntimeError("extract_windowed_features: Rust extractor not available")

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
    """Create a FeatureExtractor configured with the shared defaults.

    Ensure the native Rust `runtime_core.FeatureExtractor` works for extraction

    A short, timed sanity check is run to avoid calling the Rust extractor from
    the main process if it is known to hang (see earlier TODO in this file).
    """

    if _rust_extractor_sanity_check(include_delta, include_delta_delta):
        logger.info("Using Rust FeatureExtractor for extraction (sanity check passed)")
        rust_fx = rc.FeatureExtractor(
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            include_delta=include_delta,
            include_delta_delta=include_delta_delta,
        )
        return FeatureExtractorBridge(rust_fx)
    else:
        logger.error(
            "Falling back to Python FeatureExtractor for extraction (sanity check failed)"
        )
        raise RuntimeError("make_feature_extractor: Rust extractor unavailable")


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
    """Construct a deterministic orbit state using the Rust implementation.

    Use positional arguments to avoid relying on a keyword name that may not
    be present in all generated Python bindings. If `seed` is provided the
    Rust constructor that accepts a seed will be used.
    """
    if seed is None:
        return rc.OrbitState(
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
        "make_orbit_state: seed provided but rc.OrbitState.new_with_seed() not available"
    )


def step_orbit(
    state: rc.OrbitState,
    dt: float,
    residual_params: Optional[rc.ResidualParams] = None,
    band_gates: Optional[Sequence[float]] = None,
) -> rc.Complex:
    rp = residual_params or make_residual_params()
    return state.step(
        dt, rp, band_gates=list(band_gates) if band_gates is not None else None
    )


def synthesize(
    state: rc.OrbitState,
    residual_params: Optional[rc.ResidualParams] = None,
    band_gates: Optional[Iterable[float]] = None,
) -> rc.Complex:
    rp = residual_params or make_residual_params()
    return state.synthesize(rp, list(band_gates) if band_gates is not None else None)
