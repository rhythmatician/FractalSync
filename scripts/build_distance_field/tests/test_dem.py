import sys
import pathlib
import numpy as np

# Ensure repository root is on sys.path so the 'scripts' package can be imported
ROOT = pathlib.Path(__file__).parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_distance_field import (
    compute_dem_for_point,
    apply_dem_to_sdf,
    build_mask_cpu,
    build_signed_distance,
    DemParams,
    Coords,
)


def test_compute_dem_basic():
    c = complex(1.0, 0.0)
    # small bailout to force early escape
    params = DemParams(
        enabled=True, bailout=1.5, max_iter=20, eps=1e-12, blend=1.0, band=1.0
    )
    de = compute_dem_for_point(c, params)
    assert de is not None
    # Expected approx: positive finite
    assert de > 0.0 and not np.isnan(de)


def test_apply_dem_integration():
    res = 64
    xmin, xmax = -2.5, 1.5
    ymin, ymax = -2.0, 2.0
    coords = Coords(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    inside = build_mask_cpu(res, coords, 256, 2.0)
    signed, dx, dy = build_signed_distance(inside, coords)

    # Use parameters that will compute DEM for a wide band and force escape quickly
    params = DemParams(
        enabled=True, bailout=2.0, max_iter=256, eps=1e-12, blend=1.0, band=1.0
    )
    out = apply_dem_to_sdf(
        signed,
        coords,
        dx,
        dy,
        params,
    )

    # ensure interior remains non-positive
    interior_mask = signed <= 0.0
    assert np.all(out[interior_mask] <= 0.0)

    # ensure at least one exterior pixel changed due to DEM
    exterior_mask = signed > 0.0
    changed = np.any(np.abs(out[exterior_mask] - signed[exterior_mask]) > 1e-6)
    assert changed
