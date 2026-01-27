"""
Thin helpers around the runtime_core Rust bindings.

This module centralises the shared constants and constructors so the
backend never drifts from the Rust source of truth. Frontend must use
matching values exposed by the wasm bindings.
"""

from __future__ import annotations

from typing import Union
import logging

import runtime_core as rc
from .python_feature_extractor import PythonFeatureExtractor

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = rc.SAMPLE_RATE
HOP_LENGTH: int = rc.HOP_LENGTH
N_FFT: int = rc.N_FFT
WINDOW_FRAMES: int = rc.WINDOW_FRAMES
DEFAULT_HEIGHT_ITERATIONS: int = rc.DEFAULT_HEIGHT_ITERATIONS
DEFAULT_HEIGHT_EPSILON: float = rc.DEFAULT_HEIGHT_EPSILON
DEFAULT_HEIGHT_GAIN: float = rc.DEFAULT_HEIGHT_GAIN


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


def height_field(
    c: rc.Complex,
    iterations: int = DEFAULT_HEIGHT_ITERATIONS,
    epsilon: float = DEFAULT_HEIGHT_EPSILON,
) -> rc.HeightFieldSample:
    return rc.height_field(c, iterations, epsilon)


def height_controller_step(
    c: rc.Complex,
    delta_model: rc.Complex,
    target_height: float,
    normal_risk: float,
    height_gain: float = DEFAULT_HEIGHT_GAIN,
    iterations: int = DEFAULT_HEIGHT_ITERATIONS,
    epsilon: float = DEFAULT_HEIGHT_EPSILON,
) -> rc.HeightControllerStep:
    return rc.height_controller_step(
        c, delta_model, target_height, normal_risk, height_gain, iterations, epsilon
    )
