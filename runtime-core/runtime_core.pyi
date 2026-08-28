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

# Module-level version constants (stamped into model metadata and golden
# vectors; used by the browser to reject stale models and by the parity
# guardrail to reject stale goldens).
CONTROLLER_VERSION: str
FEATURE_VERSION: str

# Built-in numeric constants (mirrors runtime-core/src/controller.rs).
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

class PlayerState:
    """c-space integrator with persistent momentum and contour-biased dynamics.

    Mirrors ``runtime_core::controller::PlayerState``; the same struct the
    browser instantiates. Use this when you want identical physics semantics
    on the Python side (parity tests, deterministic replay, debugging).
    """

    def __init__(self, lobe: int, sub_lobe: int, s: float, alpha: float) -> None: ...
    @property
    def c_re(self) -> float: ...
    @property
    def c_im(self) -> float: ...
    @property
    def speed(self) -> float: ...
    def apply_controls(self, s: float, alpha: float, omega_scale: float) -> None: ...
    def set_lobe(self, lobe: int, sub_lobe: int) -> None: ...
    def set_level(self, level: int) -> None: ...
    def set_d_star(self, d_star: float) -> None: ...
    def set_max_step(self, max_step: float) -> None: ...
    def set_energy(self, energy: float) -> None: ...
    def step(
        self,
        dt: float,
        h: float,
        band_gates: Optional[list[float]] = ...,
    ) -> tuple[float, float]: ...

class OrbitController:
    """May-proven orbit controller with opt-in momentum and shore-bias refinements.

    Mirrors ``runtime_core::controller::OrbitController``. The default
    construction produces a flags-off controller that is bit-identical to the
    pre-fe1087b TypeScript baseline; enable ``set_momentum`` and/or
    ``set_shore_bias`` to layer the PlayerState ideas.
    """

    def __init__(self, s: float, alpha: float, omega: float) -> None: ...
    @property
    def theta(self) -> float: ...
    def apply_controls(self, s: float, alpha: float) -> None: ...
    def set_momentum(self, on: bool) -> None: ...
    def set_drag(self, drag: float) -> None: ...
    def set_thrust(self, thrust: float) -> None: ...
    def set_energy(self, energy: float) -> None: ...
    def set_shore_bias(self, on: bool) -> None: ...
    def set_d_star(self, d_star: float) -> None: ...
    def set_max_step(self, max_step: float) -> None: ...
    def set_level(self, level: int) -> None: ...
    def set_c(self, re: float, im: float) -> None: ...
    def step(
        self,
        dt: float,
        band_gates: Optional[list[float]] = ...,
        h: float = ...,
    ) -> tuple[float, float]: ...

class RuntimeVisualMetrics:
    edge_density: float
    color_uniformity: float
    brightness_mean: float
    brightness_std: float
    brightness_range: float
    mandelbrot_membership: bool
    ...

def set_distance_field_py(
    data: Sequence[Sequence[float]], xmin: float, xmax: float, ymin: float, ymax: float
) -> None: ...
def sample_distance_field_py(
    coords: Sequence[complex],
) -> list[float]: ...
def get_builtin_distance_field_py(
    name: str,
) -> tuple[int, int, float, float, float, float]: ...
def residual_phases_for_seed_py(seed: int, k_residuals: int) -> list[float]: ...
def load_mip_pyramid_py(
    f_bin_path: str, s_bin_path: str, meta_path: str
) -> tuple[int, float, float, float, float]: ...
def install_pyramid_py(
    levels_data: Sequence[Sequence[float]],
    widths: Sequence[int],
    heights: Sequence[int],
    re_min: float,
    re_max: float,
    im_min: float,
    im_max: float,
) -> int: ...
def clear_pyramid_py() -> None: ...
def player_observation_py(c_re: float, c_im: float) -> list[float]: ...
def minimap_slope_py(c_re: float, c_im: float, level: int) -> tuple[float, float]: ...
def minimap_shore_proximity_batch_py(
    re: Sequence[float], im: Sequence[float], level: int
) -> list[float]: ...
def contour_biased_step_py(
    c_re: float,
    c_im: float,
    u_re: float,
    u_im: float,
    h: float,
    d_star: float,
    max_step: float,
    level: int,
    energy: float = ...,
) -> tuple[float, float]: ...
def mandelbrot_distance_estimate(
    coords: Union[Sequence[complex], tuple[Sequence[float], Sequence[float]]],
    ys: Optional[Sequence[float]] = ...,
) -> list[float]: ...
def compute_runtime_visual_metrics(
    image: Sequence[float],
    width: int,
    height: int,
    channels: int,
    c: complex,
    max_iter: int = ...,
) -> RuntimeVisualMetrics: ...
def lobe_point_at_angle(
    period: int, sub_lobe: int, theta: float, s: float = ...
) -> complex: ...
def mandelbrot_cardioid_proximity_batch(
    coords: Sequence[complex],
) -> list[float]: ...
def orbit_path_metrics_py(
    coords: Sequence[complex],
) -> tuple[float, float, float]: ...
