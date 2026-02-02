"""Build a signed distance field for the Mandelbrot set.

Generates a signed distance field (float32) where positive values indicate
outside distance to the boundary and negative values indicate inside depth.
Saves a .npy file and a metadata .json alongside it.

Usage (from repo root):
    python scripts/build_distance_field.py --out data/mandelbrot_distance_512.npy --res 512 --xmin -2.5 --xmax 1.5 --ymin -2.0 --ymax 2.0 --max-iter 1024

This script depends on numpy and scipy.
"""

# TODO: Use GPU acceleration for high res builds (Important: OpenGL, not CUDA)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--res", type=int, default=2048)
    p.add_argument("--xmin", type=float, default=-2.5)
    p.add_argument("--xmax", type=float, default=1.5)
    p.add_argument("--ymin", type=float, default=-2.0)
    p.add_argument("--ymax", type=float, default=2.0)
    p.add_argument("--max-iter", type=int, default=1024)
    p.add_argument("--bailout", type=float, default=4.0)
    return p.parse_args()


def build_mask(
    res: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int,
    bailout: float,
):
    w = res
    h = res
    xs = np.linspace(xmin, xmax, w)
    ys = np.linspace(ymin, ymax, h)
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    escaped = np.zeros(C.shape, dtype=bool)

    for i in range(max_iter):
        mask = ~escaped
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped_now = np.abs(Z) > bailout
        escaped = escaped | escaped_now
        if escaped.all():
            break

    inside = ~escaped
    return inside.astype(np.bool_)


def build_signed_distance(inside_mask: np.ndarray, pixel_scale: float):
    # distance to outside for interior points
    dist_out = ndimage.distance_transform_edt(~inside_mask) * pixel_scale
    # distance to inside for outside points
    dist_in = ndimage.distance_transform_edt(inside_mask) * pixel_scale
    # signed distance: positive outside, negative inside
    signed = dist_out.astype(np.float32) - dist_in.astype(np.float32)
    return signed


def main():
    args = parse_args()
    out = args.out
    res = args.res

    print(
        f"Building mask for res={res} in box x=[{args.xmin},{args.xmax}] y=[{args.ymin},{args.ymax}] max_iter={args.max_iter}"
    )
    inside = build_mask(
        res, args.xmin, args.xmax, args.ymin, args.ymax, args.max_iter, args.bailout
    )

    # pixel scale: Euclidean distance per pixel in X - use (res - 1) because
    # numpy.linspace includes both endpoints so pixel spacing is (xmax-xmin)/(res-1).
    # Assume square pixels and use the X axis spacing as the pixel scale.
    if res <= 1:
        raise ValueError("res must be > 1")
    pixel_scale = (args.xmax - args.xmin) / float(res - 1)

    print("Computing signed distance transform...")
    signed = build_signed_distance(inside, pixel_scale)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), signed)
    # Also write raw little-endian float32 binary suitable for embedding in Rust.
    with open(out.with_suffix(".bin"), "wb") as f:
        f.write(signed.astype("<f4").tobytes())

    meta = {
        "xmin": args.xmin,
        "xmax": args.xmax,
        "ymin": args.ymin,
        "ymax": args.ymax,
        "res": res,
        "max_iter": args.max_iter,
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Saved signed distance field to {out.with_suffix('.npy')} and metadata {out.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
