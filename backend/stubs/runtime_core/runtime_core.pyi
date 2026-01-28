# Module-style runtime_core stub, used via package-style re-export in __init__.pyi

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
    re: float
    im: float

    def __init__(self, re: float, im: float) -> None: ...
    def __repr__(self) -> str: ...
    def __add__(self, other: Complex) -> Complex: ...
    def __sub__(self, other: Complex) -> Complex: ...
    def __mul__(self, other: Complex) -> Complex: ...
    def __complex__(self) -> complex: ...
    @property
    def real(self) -> float: ...
    @property
    def imag(self) -> float: ...

class FeatureExtractor:
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
    def compute_normalization_stats(self, features: NDArray) -> None: ...
    def normalize_features(self, features: NDArray) -> NDArray: ...
    def test_simple(self) -> list[float]: ...
    
    feature_mean: Optional[NDArray]
    feature_std: Optional[NDArray]

class ResidualParams:
    def __init__(
        self, k_residuals: int = 6, residual_cap: float = 0.5, radius_scale: float = 1.0
    ) -> None: ...
    k_residuals: int
    residual_cap: float
    radius_scale: float

class OrbitState:
    def __init__(
        self,
        lobe: int,
        sub_lobe: int,
        theta: float,
        omega: float,
        s: float,
        alpha: float,
        k_residuals: int,
        residual_omega_scale: float,
        seed: Optional[int] = None,
    ) -> None: ...
    @staticmethod
    def new_with_seed(
        lobe: int,
        sub_lobe: int,
        theta: float,
        omega: float,
        s: float,
        alpha: float,
        k_residuals: int,
        residual_omega_scale: float,
        seed: int,
    ) -> OrbitState: ...
    @staticmethod
    def new_default_seeded(seed: int) -> OrbitState: ...
    def carrier(self) -> Complex: ...
    def residual_phases(self) -> list[float]: ...
    def residual_omegas(self) -> list[float]: ...
    def advance(self, dt: float) -> None: ...
    def step(
        self,
        dt: float,
        residual_params: ResidualParams,
        band_gates: Optional[list[float]] = None,
    ) -> Complex: ...
    def synthesize(
        self,
        residual_params: ResidualParams,
        band_gates: Optional[list[float]] = None,
    ) -> Complex: ...

def lobe_point_at_angle(
    period: int, sub_lobe: int, theta: float, s: float = 1.0
) -> Complex: ...
