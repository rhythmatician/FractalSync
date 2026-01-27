"""Type stubs for runtime_core (Rust PyO3 extension)."""

# flake8: noqa

from __future__ import annotations

from typing import Sequence
from numpy.typing import NDArray

# Constants
SAMPLE_RATE: int
HOP_LENGTH: int
N_FFT: int
WINDOW_FRAMES: int
DEFAULT_HEIGHT_ITERATIONS: int
DEFAULT_HEIGHT_EPSILON: float
DEFAULT_HEIGHT_GAIN: float


class Complex:
    """Complex number with real/imag attributes."""

    real: float
    imag: float

    def __init__(self, real: float, imag: float) -> None: ...
    def __repr__(self) -> str: ...
    def __add__(self, other: Complex) -> Complex: ...
    def __sub__(self, other: Complex) -> Complex: ...
    def __mul__(self, other: Complex) -> Complex: ...
    def __complex__(self) -> complex: ...


class HeightFieldSample:
    """Height-field sample (height + gradient)."""

    height: float
    gradient: Complex


class HeightControllerStep:
    """Controller step result."""

    new_c: Complex
    delta: Complex
    height: float
    gradient: Complex


class FeatureExtractor:
    """Audio feature extractor."""

    def __init__(
        self,
        sr: int = 48000,
        hop_length: int = 1024,
        n_fft: int = 4096,
        include_delta: bool = False,
        include_delta_delta: bool = False,
    ) -> None: ...

    def num_features_per_frame(self) -> int: ...
    def extract_windowed_features(
        self,
        audio: Sequence[float],
        window_frames: int = 10,
    ) -> NDArray: ...


def height_field(
    c: Complex,
    iterations: int = DEFAULT_HEIGHT_ITERATIONS,
    epsilon: float = DEFAULT_HEIGHT_EPSILON,
) -> HeightFieldSample: ...


def height_controller_step(
    c: Complex,
    delta_model: Complex,
    target_height: float,
    normal_risk: float,
    height_gain: float = DEFAULT_HEIGHT_GAIN,
    iterations: int = DEFAULT_HEIGHT_ITERATIONS,
    epsilon: float = DEFAULT_HEIGHT_EPSILON,
) -> HeightControllerStep: ...
