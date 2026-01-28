"""Stubs for src.python_feature_extractor used by tests."""

from __future__ import annotations

from typing import Sequence, Any
import numpy as np

class PythonFeatureExtractor:
    def __init__(
        self,
        sr: int = ...,
        hop_length: int = ...,
        n_fft: int = ...,
        include_delta: bool = False,
        include_delta_delta: bool = False,
    ) -> None: ...
    def extract_windowed_features(
        self, audio: Sequence[float], window_frames: int = ...
    ) -> np.ndarray: ...
    def num_features_per_frame(self) -> int: ...
    def test_simple(self) -> Any: ...
