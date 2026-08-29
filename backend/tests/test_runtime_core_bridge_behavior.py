import sys
import types
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# If `runtime_core` is not available (typical when running tests locally
# without building the native extension), install a lightweight fake module
# so importing `runtime_core_helpers` succeeds. Tests below only exercise the
# Python-side adapter behaviour and do not require the real C-extension.
if "runtime_core" not in sys.modules:
    fake_rc = types.ModuleType("runtime_core")
    fake_rc.SAMPLE_RATE = 48000
    fake_rc.HOP_LENGTH = 1024
    fake_rc.N_FFT = 4096
    fake_rc.WINDOW_FRAMES = 10
    fake_rc.DEFAULT_K_RESIDUALS = 6
    fake_rc.DEFAULT_RESIDUAL_CAP = 1.0
    fake_rc.DEFAULT_RESIDUAL_OMEGA_SCALE = 1.0
    fake_rc.DEFAULT_BASE_OMEGA = 1.0
    fake_rc.DEFAULT_ORBIT_SEED = 123

    # Minimal lobe_point_at_angle function
    def _lobe_point_at_angle(lobe: int, sub_lobe: int, theta: float):
        return types.SimpleNamespace(real=0.0, imag=0.0)

    fake_rc.lobe_point_at_angle = _lobe_point_at_angle

    # Minimal class required by make_residual_params() call in ControlTrainer
    class _ResidualParams:
        def __init__(
            self,
            k_residuals: int = 6,
            residual_cap: float = 1.0,
            radius_scale: float = 1.0,
        ):
            self.k_residuals = k_residuals
            self.residual_cap = residual_cap
            self.radius_scale = radius_scale

    fake_rc.ResidualParams = _ResidualParams

    # Minimal FeatureExtractor stub
    class _FeatureExtractor:
        def __init__(
            self,
            sr=48000,
            hop_length=1024,
            n_fft=4096,
            include_delta=False,
            include_delta_delta=False,
        ):
            pass

        def num_features_per_frame(self):
            return 6

        def extract_windowed_features(self, audio, window_frames):
            # produce a small array with shape (n_windows, 6*window_frames)
            n_windows = max(1, len(audio) // (hop_length := 1024))
            return [[0.0] * (6 * window_frames) for _ in range(n_windows)]

    fake_rc.FeatureExtractor = _FeatureExtractor

    # Minimal orbit state to support synthesize()/step() calls in trainer
    class _FakeOrbitState:
        def __init__(
            self,
            lobe,
            sub_lobe,
            theta,
            omega,
            s,
            alpha,
            k_residuals,
            residual_omega_scale,
        ):
            pass

        @staticmethod
        def new_with_seed(
            lobe,
            sub_lobe,
            theta,
            omega,
            s,
            alpha,
            k_residuals,
            residual_omega_scale,
            seed,
        ):
            return _FakeOrbitState(
                lobe,
                sub_lobe,
                theta,
                omega,
                s,
                alpha,
                k_residuals,
                residual_omega_scale,
            )

        def step(self, dt, residual_params, band_gates=None):
            return types.SimpleNamespace(real=0.0, imag=0.0)

        def synthesize(self, residual_params, band_gates=None):
            return types.SimpleNamespace(real=0.0, imag=0.0)

    fake_rc.OrbitState = _FakeOrbitState

    # Minimal Complex type for geometry/return values used by synthesize/step
    class _Complex:
        def __init__(self, re: float, im: float):
            self.re = re
            self.im = im

        @property
        def real(self):
            return self.re

        @property
        def imag(self):
            return self.im

    fake_rc.Complex = _Complex

    # Version stamps required by src/data_loader.py's import contract
    fake_rc.FEATURE_VERSION = "fake-feature-version"
    fake_rc.CONTROLLER_VERSION = "fake-controller-version"

    sys.modules["runtime_core"] = fake_rc

from src.runtime_core_helpers import FeatureExtractorProxy

# Provide a minimal 'librosa' shim so importing modules that reference it at
# import-time (e.g., DataLoader) doesn't fail in lightweight test envs.
# Only install the shim when librosa is genuinely missing — never shadow a
# real librosa, and always restore the real one after these tests finish so
# later test modules see the genuine library.
_real_librosa = None
if "librosa" not in sys.modules:
    try:
        _real_librosa = importlib.import_module("librosa")
    except Exception:
        _real_librosa = None
if _real_librosa is None and "librosa" not in sys.modules:
    fake_librosa = types.ModuleType("librosa")

    def _load_dummy(path, sr, mono=True, duration=None):
        # return a short silent array and the sampling rate
        return np.zeros(min(sr * 2, 1024), dtype=np.float32), sr

    fake_librosa.load = _load_dummy
    sys.modules["librosa"] = fake_librosa

# Provide a minimal `cv2` shim to avoid OpenCV import errors in the visual metrics
# module when running tests in lightweight environments.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

from src.control_trainer import ControlTrainer
from src.control_model import AudioToControlModel


def test_num_features_per_frame_callable_and_attr():
    class FakeRustMethod:
        def num_features_per_frame(self):
            return 6

    be = FeatureExtractorProxy(FakeRustMethod())
    assert be.num_features_per_frame() == 6

    class FakeRustAttr:
        num_features_per_frame = 12

    be2 = FeatureExtractorProxy(FakeRustAttr())
    assert be2.num_features_per_frame() == 12


class DummyVisualMetrics:
    def render_julia_set(self, seed, width, height, max_iter, **kwargs):
        # return a small dummy image
        return np.zeros((height, width, 3), dtype=np.uint8)

    def compute_all_metrics(self, image, prev_image=None):
        return {"temporal_change": 0.0}

    def mandelbrot_distance_estimate(self, c_values):
        return torch.zeros(len(c_values))


def test_control_trainer_falls_back_on_num_features_exception():
    # Small model for quick test
    model = AudioToControlModel(
        window_frames=10, n_features_per_frame=6, hidden_dims=[16, 16], k_bands=6
    )

    class BadFE:
        def num_features_per_frame(self):
            raise RuntimeError("boom")

    # Create trainer with the bad feature extractor. The c-space proxy
    # fast path routes the contour step through the real Rust binding
    # (contour_biased_step_py), which the fake runtime_core module does
    # not implement — disable it so this test exercises only the
    # num_features_per_frame fallback behavior.
    trainer = ControlTrainer(
        model=model,
        visual_metrics=DummyVisualMetrics(),
        feature_extractor=BadFE(),
        device="cpu",
        learning_rate=1e-4,
        use_curriculum=False,
        num_workers=0,
        k_residuals=6,
        use_cspace_proxies=False,
    )

    # Create synthetic feature tensor (11 frames, input dim = 6*10)
    features = np.zeros((11, 60), dtype=np.float32)
    all_features_tensor = torch.tensor(features, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(all_features_tensor), batch_size=4)

    # Should not raise despite feature extractor raising in num_features_per_frame()
    avg_losses = trainer.train_epoch(dataloader, epoch=0)
    assert "loss" in avg_losses
    assert avg_losses["loss"] == avg_losses["loss"]  # finite number (not NaN)
