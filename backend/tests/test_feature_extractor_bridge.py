import numpy as np

from src.runtime_core_bridge import make_feature_extractor


def test_feature_extractor_bridge_api():
    fe = make_feature_extractor(include_delta=False, include_delta_delta=False)
    # Sanity: basic API
    assert hasattr(fe, "extract_windowed_features")
    assert hasattr(fe, "compute_normalization_stats")
    assert hasattr(fe, "normalize_features")

    # Generate a short sine and extract
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    windows = fe.extract_windowed_features(audio, window_frames=10)
    # windows is 2D numpy array
    assert windows.ndim == 2

    # Compute normalization stats and normalize
    fe.compute_normalization_stats([windows])
    normed = fe.normalize_features(windows)
    assert normed.shape == windows.shape
