import numpy as np
import pathlib
import sys

# Ensure repo root on path
ROOT = pathlib.Path(__file__).parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_distance_field import _try_compute_dem_gpu, Coords


def test_try_compute_dem_gpu_smoke():
    res = 32
    coords = Coords(xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0)
    out = _try_compute_dem_gpu(
        res=res,
        coords=coords,
        max_iter=64,
        dem_bailout=10.0,
        dem_eps=1e-12,
        batch=16,
        local_size=8,
    )
    # GPU path may not be available in CI / headless machines; accept None or valid array
    assert out is None or (isinstance(out, np.ndarray) and out.shape == (res, res))
