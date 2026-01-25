import numpy as np
from src.normalization import safe_zscore, apply_runtime_normalization


def test_safe_zscore_basic():
    feats = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    mean = np.array([1.5, 3.0, 4.5])
    std = np.array([0.5, 1.0, 1.5])
    out = safe_zscore(feats, mean, std)
    assert out.shape == feats.shape
    # Manual check for first element
    assert abs(out[0, 0] - ((1.0 - 1.5) / 0.5)) < 1e-6


def test_safe_zscore_handles_zero_std():
    feats = np.array([[1.0, 2.0]])
    mean = np.array([1.0, 2.0])
    std = np.array([0.0, 0.0])
    out = safe_zscore(feats, mean, std)
    # when std is zero we fall back to dividing by 1.0
    assert np.allclose(out, np.zeros_like(feats))


def test_apply_runtime_normalization_applies_and_raises():
    feats = np.array([[1.0, 2.0]])
    metadata = {
        "input_normalization": {"type": "zscore", "applied_by": "runtime"},
        "feature_mean": [0.0, 1.0],
        "feature_std": [1.0, 1.0],
    }
    out = apply_runtime_normalization(metadata, feats)
    assert np.allclose(out, np.array([[1.0, 1.0]]))

    # Missing fields should raise
    bad_meta = {"input_normalization": {"type": "zscore", "applied_by": "runtime"}}
    try:
        apply_runtime_normalization(bad_meta, feats)
        raised = False
    except ValueError:
        raised = True
    assert raised
