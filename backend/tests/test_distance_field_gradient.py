import numpy as np

from src.distance_field_loader import DistanceField


def test_distance_field_gradient_on_array():
    # small synthetic array with linear gradient in x direction
    w, h = 8, 6
    arr = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            arr[y, x] = x / float(w - 1)

    df = DistanceField(arr=arr, real_range=(0.0, 1.0), imag_range=(0.0, 1.0))

    gx, gy = df.gradient(0.5, 0.5)
    # gradient in x roughly equals 1.0 over domain
    assert abs(gx - 1.0) < 0.5
    # gradient in y should be near zero
    assert abs(gy) < 0.5


def test_distance_field_gradient_fallback():
    df = DistanceField(arr=None)
    # the analytic estimator is smooth; gradient should be finite
    gx, gy = df.gradient(-0.5, 0.0)
    assert isinstance(gx, float)
    assert isinstance(gy, float)
