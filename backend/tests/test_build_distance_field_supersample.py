import sys
from pathlib import Path
import numpy as np
from scipy import ndimage

# Ensure repository root is on sys.path so `scripts` package is importable
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from scripts.build_distance_field import Coords, build_signed_distance


def generate_circle_mask(res, xmin, xmax, ymin, ymax, cx=0.0, cy=0.0, r=0.5):
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(xs, ys)
    D = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    inside = D <= r
    return inside


def analytic_circle_signed(x, y, cx, cy, r):
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r


def downsample_signed(signed_high, ss, out_res):
    zoom_factor = 1.0 / float(ss)
    signed = ndimage.zoom(signed_high, (zoom_factor, zoom_factor), order=3)
    signed = signed[:out_res, :out_res]
    return signed


def test_supersampled_reduces_error():
    xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0
    res = 256
    ss = 4

    # low-res mask and SDF
    low_mask = generate_circle_mask(res, xmin, xmax, ymin, ymax, cx=0.0, cy=0.0, r=0.5)
    coords = Coords(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    signed_low, dx_low, dy_low = build_signed_distance(low_mask, coords)

    # high-res mask and SDF
    high_res = res * ss
    high_mask = generate_circle_mask(
        high_res, xmin, xmax, ymin, ymax, cx=0.0, cy=0.0, r=0.5
    )
    signed_high, dx_high, dy_high = build_signed_distance(high_mask, coords)

    # downsample high-res SDF
    signed_ds = downsample_signed(signed_high, ss, res)

    # analytic distances at low-res grid
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(xs, ys)
    analytic = analytic_circle_signed(X, Y, 0.0, 0.0, 0.5)

    # compare absolute unsigned distances
    err_low = np.abs(np.abs(signed_low) - np.abs(analytic))
    err_ds = np.abs(np.abs(signed_ds) - np.abs(analytic))

    max_low = err_low.max()
    max_ds = err_ds.max()

    # Supersampled max error should be strictly smaller
    assert (
        max_ds < max_low
    ), f"Supersampled error {max_ds} not less than low-res {max_low}"
