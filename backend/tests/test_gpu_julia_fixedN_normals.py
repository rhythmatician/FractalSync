import numpy as np
import pytest

from src.julia_gpu import GPUJuliaRenderer


@pytest.mark.skipif(False, reason="Skip only if GPU unavailable")
def test_fixedN_normals_no_seam():
    try:
        r = GPUJuliaRenderer(width=64, height=64)
    except Exception as e:
        pytest.skip("GPU not available: " + str(e))

    img = r.render(-0.7269, 0.1889, zoom=1.0, max_iter=128)
    # Convert to grayscale [0,1]
    gray = img.mean(axis=2) / 255.0

    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)

    med = np.median(grad)
    p99 = np.percentile(grad, 99)

    # Assert gradients are generally smooth (empirical thresholds)
    assert med < 0.15, f"median gradient too large: {med}"
    assert p99 < 0.9, f"99th percentile gradient too large: {p99}"
