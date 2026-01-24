import numpy as np

import runtime_core as rc


def test_sample_bilinear_batch_matches_distancefield():
    res = 4
    field = np.arange(res * res, dtype=np.float32).reshape((res, res))
    flat = list(field.ravel())
    df = rc.DistanceField(flat, res, (0.0, 4.0), (0.0, 4.0), 1.0, 0.05)

    # Points to test (real, imag)
    pts = [(0.5, 0.5), (1.25, 2.75), (3.0, 3.0), (0.0, 0.0), (3.999, 3.999)]
    reals = [p[0] for p in pts]
    imags = [p[1] for p in pts]

    batch_vals = rc.sample_bilinear_batch(flat, res, 0.0, 4.0, 0.0, 4.0, reals, imags)

    individual = [df.sample_bilinear(r, i) for (r, i) in pts]

    for b, ind in zip(batch_vals, individual):
        assert abs(b - ind) < 1e-6
