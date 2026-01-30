from __future__ import annotations

import numpy as np
from runtime_core import FeatureExtractor, SAMPLE_RATE, HOP_LENGTH, N_FFT


class FeatureExtractorBridge:
    """Thin adapter providing a stable feature extractor API backed by the
    Rust `FeatureExtractor` implementation.

    This bridge exists to ease migration; callers are encouraged to use
    `runtime_core.FeatureExtractor` directly where possible.
    """

    def __init__(self, rust_extractor: FeatureExtractor) -> None:
        self._rust = rust_extractor
        self.feature_mean = None
        self.feature_std = None

    def num_features_per_frame(self) -> int:
        return int(self._rust.num_features_per_frame())

    def extract_windowed_features(self, audio, window_frames: int):
        result = self._rust.extract_windowed_features(list(audio), window_frames)
        return np.array(result, dtype=np.float64)

    def compute_normalization_stats(self, all_features: list):
        if not all_features:
            return
        concatenated = np.concatenate(all_features, axis=0)
        self.feature_mean = np.mean(concatenated, axis=0)
        self.feature_std = np.std(concatenated, axis=0) + 1e-8

    def normalize_features(self, features):
        if self.feature_mean is None or self.feature_std is None:
            return features
        return (features - self.feature_mean) / self.feature_std


def make_feature_extractor(
    include_delta: bool = False,
    include_delta_delta: bool = False,
) -> FeatureExtractorBridge:
    """Create a `FeatureExtractorBridge` configured with the shared defaults."""

    rust_fx = FeatureExtractor(
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        include_delta=include_delta,
        include_delta_delta=include_delta_delta,
    )
    return FeatureExtractorBridge(rust_fx)
