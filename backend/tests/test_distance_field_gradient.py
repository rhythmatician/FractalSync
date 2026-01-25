import numpy as np

from src.distance_field_loader import DistanceField


def test_distance_field_gradient_on_array():
    # small synthetic square array with linear gradient in x direction (8x8)
    w = h = 8
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
