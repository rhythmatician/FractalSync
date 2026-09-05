"""Parity checks for the diagnostic NumPy feature oracle."""

import numpy as np
import runtime_core as rc

from src.python_feature_extractor import DiagnosticFeatureParityOracle


def _test_audio(sample_rate: int = 48_000, duration: float = 0.2) -> np.ndarray:
    sample_count = int(sample_rate * duration)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    return (
        0.3 * np.sin(2.0 * np.pi * 440.0 * time)
        + 0.2 * np.sin(2.0 * np.pi * 880.0 * time)
        + 0.1 * np.sin(2.0 * np.pi * 220.0 * time)
    ).astype(np.float32)


def test_diagnostic_oracle_matches_rust_extractor() -> None:
    """The test compares both implementations on the same generated signal."""
    audio = _test_audio()
    window_frames = 4
    rust = rc.FeatureExtractor()
    oracle = DiagnosticFeatureParityOracle()

    rust_windows = np.asarray(
        rust.extract_windowed_features(audio, window_frames), dtype=np.float64
    )
    oracle_windows = oracle.extract_windowed_features(audio, window_frames)

    np.testing.assert_allclose(oracle_windows, rust_windows, rtol=1e-5, atol=1e-7)
