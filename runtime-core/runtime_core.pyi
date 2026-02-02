"""Type stubs for the ``runtime_core`` native extension.

This file is the *authoritative* type stub included in the `runtime_core`
wheel so that type-checkers (e.g. mypy), editors and CI can inspect the
Python API exposed by the Rust PyO3 bindings. It documents the public
Python-facing surface implemented in ``src/pybindings.rs`` and is used
for static verification only — it carries no runtime behaviour.

Maintenance & workflow
- During backend development, prefer editing ``backend/stubs/runtime_core/runtime_core.pyi``
  for fast iteration and local testing.
- When making a release or preparing CI, ensure this file is updated so
  the wheel contains the same, authoritative ``.pyi`` that CI installs.
- Keep the declarations in sync with the Rust bindings in
  ``runtime-core/src/pybindings.rs``; update tests when adding or
  removing public symbols.

Note: This file exists solely to aid static tools and should not contain
executable code or runtime imports.
"""

from typing import Optional, Sequence, Union
from numpy.typing import NDArray
import numpy as np

SAMPLE_RATE: int
HOP_LENGTH: int
N_FFT: int
WINDOW_FRAMES: int
DEFAULT_K_RESIDUALS: int
DEFAULT_RESIDUAL_CAP: float
DEFAULT_RESIDUAL_OMEGA_SCALE: float
DEFAULT_BASE_OMEGA: float
DEFAULT_ORBIT_SEED: int

class FeatureExtractor:
    def __init__(
        self,
        sr: int = ...,
        hop_length: int = ...,
        n_fft: int = ...,
        include_delta: bool = ...,
        include_delta_delta: bool = ...,
    ) -> None: ...
    def num_features_per_frame(self) -> int: ...
    def extract_windowed_features(
        self,
        audio: Union[Sequence[float], NDArray[np.floating]],
        window_frames: int = ...,
    ) -> NDArray: ...
    def test_simple(self) -> list[float]: ...
    def compute_normalization_stats(
        self,
        all_features: Union[Sequence[Sequence[float]], Sequence[NDArray[np.floating]]],
    ) -> None: ...
    def normalize_features(
        self, features: Union[Sequence[float], NDArray[np.floating]]
    ) -> list[float]: ...
    @property
    def feature_mean(self) -> Optional[list[float]]: ...
    @property
    def feature_std(self) -> Optional[list[float]]: ...

class ResidualParams:
    def __init__(
        self,
        k_residuals: int = ...,
        residual_cap: float = ...,
        radius_scale: float = ...,
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
    ) -> None: ...

    lobe: int
    sub_lobe: int
    theta: float
    omega: float
    s: float
    alpha: float
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
    ) -> "OrbitState": ...
    @staticmethod
    def new_default_seeded(seed: int) -> "OrbitState": ...
    def carrier(self) -> complex: ...
    def residual_phases(self) -> list[float]: ...
    def residual_omegas(self) -> list[float]: ...
    def advance(self, dt: float) -> None: ...
    def step(
        self,
        dt: float,
        residual_params: ResidualParams,
        band_gates: Optional[list[float]] = ...,
    ) -> complex: ...
    def synthesize(
        self, residual_params: ResidualParams, band_gates: Optional[list[float]] = ...
    ) -> complex: ...

class RuntimeVisualMetrics:
    edge_density: float
    color_uniformity: float
    brightness_mean: float
    brightness_std: float
    brightness_range: float
    mandelbrot_membership: bool
    ...

def load_distance_field_py(path: str) -> None: ...
def set_distance_field_py(
    data: Sequence[Sequence[float]], xmin: float, xmax: float, ymin: float, ymax: float
) -> None: ...
def sample_distance_field_py(
    x_coords: Sequence[float], y_coords: Sequence[float]
) -> list[float]: ...
def get_builtin_distance_field_py(
    name: str,
) -> tuple[int, int, float, float, float, float]: ...
def compute_runtime_visual_metrics(
    image: Sequence[float],
    width: int,
    height: int,
    channels: int,
    c_real: float,
    c_imag: float,
    max_iter: int = ...,
) -> RuntimeVisualMetrics: ...
def lobe_point_at_angle(
    period: int, sub_lobe: int, theta: float, s: float = ...
) -> complex: ...
