import numpy as np
from src.runtime_core_bridge import make_feature_extractor
from src.runtime_core_bridge import SAMPLE_RATE


def test_extract_numpy_vs_list_equal():
    # Short deterministic audio
    duration_sec = 0.1
    n_samples = int(SAMPLE_RATE * duration_sec)
    audio = (np.random.RandomState(0).randn(n_samples) * 0.1).astype(np.float32)

    extractor = make_feature_extractor()

    # Using ndarray (contiguous float32)
    arr = np.ascontiguousarray(audio, dtype=np.float32)
    features_arr = extractor.extract_windowed_features(arr, window_frames=10)

    # Using Python list (older code path)
    features_list = extractor.extract_windowed_features(list(arr), window_frames=10)

    assert features_arr.shape == features_list.shape
    assert np.allclose(features_arr, features_list)


def test_extract_with_float64_array():
    # Ensure float64 arrays are accepted and cast internally
    duration_sec = 0.05
    n_samples = int(SAMPLE_RATE * duration_sec)
    audio64 = (np.random.RandomState(1).randn(n_samples) * 0.1).astype(np.float64)

    extractor = make_feature_extractor()
    features64 = extractor.extract_windowed_features(audio64, window_frames=10)

    # shape sanity check
    assert features64.ndim == 2
    assert features64.shape[1] == extractor.num_features_per_frame() * 10
