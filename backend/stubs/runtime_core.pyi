"""Type stubs for runtime_core (Rust PyO3 extension)."""

# flake8: noqa

from __future__ import annotations

from typing import Optional, Sequence
from numpy.typing import NDArray

# Constants
SAMPLE_RATE: int
HOP_LENGTH: int
N_FFT: int
WINDOW_FRAMES: int
DEFAULT_K_RESIDUALS: int
DEFAULT_RESIDUAL_CAP: float
DEFAULT_RESIDUAL_OMEGA_SCALE: float
DEFAULT_BASE_OMEGA: float
DEFAULT_ORBIT_SEED: int

class Complex:
    """Complex number with re and im attributes."""

    re: float
    im: float

    def __init__(self, re: float, im: float) -> None: ...
    def __repr__(self) -> str: ...
    def __add__(self, other: Complex) -> Complex: ...
    def __sub__(self, other: Complex) -> Complex: ...
    def __mul__(self, other: Complex) -> Complex: ...
    def __complex__(self) -> complex: ...

    # Allow attribute access as real/imag for compatibility
    @property
    def real(self) -> float: ...
    @property
    def imag(self) -> float: ...

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

    # Normalization statistics provided after compute_normalization_stats
    feature_mean: Sequence[float]
    feature_std: Sequence[float]

    def num_features_per_frame(self) -> int: ...
    def extract_windowed_features(
        self,
        audio: Sequence[float],
        window_frames: int = 10,
    ) -> NDArray: ...
    def compute_normalization_stats(self, features: Sequence[NDArray]) -> None: ...
    def normalize_features(self, features: NDArray) -> NDArray: ...

class ResidualParams:
    """Residual orbit parameters."""

    def __init__(
        self,
        k_residuals: int = 6,
        residual_cap: float = 0.5,
        radius_scale: float = 1.0,
    ) -> None: ...

    k_residuals: int
    residual_cap: float
    radius_scale: float

    class DistanceField:
        """Distance field for Mandelbrot boundary proximity lookup."""

        def __init__(
            self,
            field: Sequence[float],
            resolution: int,
            real_range: tuple[float, float],
            imag_range: tuple[float, float],
            max_distance: float,
            slowdown_threshold: float,
        ) -> None: ...
        def lookup(self, c: Complex) -> float: ...
        def get_velocity_scale(self, c: Complex) -> float: ...

class OrbitState:
    """Mandelbrot orbit state."""

    def __init__(
        self,
        lobe: int = 1,
        sub_lobe: int = 0,
        theta: float = 0.0,
        omega: float = 0.15,
        s: float = 1.02,
        alpha: float = 0.3,
        k_residuals: int = 6,
        residual_omega_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None: ...

    lobe: int
    sub_lobe: int
    theta: float
    omega: float
    s: float
    alpha: float
    k_residuals: int
    residual_omega_scale: float

    def step(
        self,
        dt: float,
        residual_params: ResidualParams,
        band_gates: Optional[list[float]] = None,
        distance_field: Optional[DistanceField] = None,
    ) -> Complex: ...
    def synthesize(
        self,
        residual_params: ResidualParams,
        band_gates: Optional[list[float]] = None,
    ) -> Complex: ...
    def clone(self) -> OrbitState: ...

# Geometry functions

def lobe_point_at_angle(
    period: int,
    sub_lobe: int,
    theta: float,
    s: float = 1.0,
) -> Complex: ...
