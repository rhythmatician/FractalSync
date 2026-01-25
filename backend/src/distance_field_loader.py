"""Load Mandelbrot distance field for runtime use.

Provides a small, well-tested wrapper with `lookup` and `get_velocity_scale` helpers
that the rest of the codebase expects. Falls back to a simple analytic estimator when
Numpy files are not present so tests and CI remain robust.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple
import runtime_core as rc
import numpy as np


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def compute_escape_time_estimate(
    c_real: float, c_imag: float, max_iter: int = 256
) -> float:
    """Rudimentary escape-time estimator returning a normalized escape time in [0, 1].

    This is an inexpensive analytic estimator used by small tests and utilities
    when a precomputed distance field is not required. It is intentionally
    inexpensive (low iterations) and coarse."""
    z_r, z_i = 0.0, 0.0
    for i in range(max_iter):
        z_r2 = z_r * z_r
        z_i2 = z_i * z_i
        if z_r2 + z_i2 > 256.0:
            # Normalize iteration count into [0,1] (quick escape -> small value)
            return float(i) / float(max_iter)
        # z = z^2 + c
        z_i = 2.0 * z_r * z_i + c_imag
        z_r = z_r2 - z_i2 + c_real
    # did not escape -> treat as inside (1.0)
    return 1.0


class DistanceField:
    """Distance field wrapper used by runtime and tests.

    Attributes:
        arr: 2D numpy array of escape times in [0, 1] where 1.0 is inside/on boundary
        real_range: (min, max) of real axis
        imag_range: (min, max) of imag axis
        slowdown_threshold: escape time threshold above which slowdown begins
    """

    def __init__(
        self,
        arr: Optional[np.ndarray] = None,
        real_range: Tuple[float, float] = (-2.5, 1.0),
        imag_range: Tuple[float, float] = (-1.5, 1.5),
        slowdown_threshold: float = 0.02,
    ) -> None:
        self.arr = arr
        self.real_range = real_range
        self.imag_range = imag_range
        self.slowdown_threshold = float(slowdown_threshold)

    def _lookup_array(self, x: float, y: float) -> float:
        """Bilinearly sample the numpy array at coordinate (x, y) and return escape_time.

        If no array is present, fall back to the analytic estimator.
        """
        if self.arr is None:
            return compute_escape_time_estimate(x, y, max_iter=256)

        h, w = self.arr.shape
        rx = (x - self.real_range[0]) / (self.real_range[1] - self.real_range[0])
        ry = (y - self.imag_range[0]) / (self.imag_range[1] - self.imag_range[0])

        # Clamp to [0,1]
        rx = float(np.clip(rx, 0.0, 1.0))
        ry = float(np.clip(ry, 0.0, 1.0))

        fx = rx * (w - 1)
        fy = ry * (h - 1)
        ix = int(np.floor(fx))
        iy = int(np.floor(fy))

        ix1 = min(ix + 1, w - 1)
        iy1 = min(iy + 1, h - 1)

        dx = fx - ix
        dy = fy - iy

        v00 = float(self.arr[iy, ix])
        v10 = float(self.arr[iy, ix1])
        v01 = float(self.arr[iy1, ix])
        v11 = float(self.arr[iy1, ix1])

        # Bilinear interpolation
        v0 = v00 * (1 - dx) + v10 * dx
        v1 = v01 * (1 - dx) + v11 * dx
        v = v0 * (1 - dy) + v1 * dy
        # Ensure finite and clamped
        if not np.isfinite(v):
            return 1.0
        return float(np.clip(v, 0.0, 1.0))

    def lookup(self, real: float, imag: float) -> float:
        """Return escape_time in [0, 1] for coordinate (real, imag)."""
        return self._lookup_array(real, imag)

    def get_velocity_scale(self, real: float, imag: float) -> float:
        """Compute velocity scale in [0, 1] given an escape time.

        Values near the boundary (escape_time near 1.0) will yield scale < 1.0.
        """
        et = self.lookup(real, imag)
        t = float(et)
        if t <= self.slowdown_threshold:
            return 1.0
        # Map [threshold, 1] -> [0,1] then apply smoothstep and invert
        u = (t - self.slowdown_threshold) / (1.0 - self.slowdown_threshold)
        s = _smoothstep(u)
        scale = 1.0 - s
        return float(np.clip(scale, 0.0, 1.0))

    def gradient(self, real: float, imag: float) -> Tuple[float, float]:
        """Return finite-difference gradient (gx, gy) of the distance field at (real, imag).

        This mirrors the sampling used by :meth:`lookup` and uses one grid-cell offsets
        to compute central differences. If no precomputed array is present, fall back
        to numeric differences of the analytic escape-time estimator.
        """
        if self.arr is None:
            # Analytic fallback: use small epsilon relative to the analytic domain
            eps = 1e-4
            fcx = compute_escape_time_estimate(real + eps, imag)
            fmx = compute_escape_time_estimate(real - eps, imag)
            fcy = compute_escape_time_estimate(real, imag + eps)
            fmy = compute_escape_time_estimate(real, imag - eps)
            gx = (fcx - fmx) / (2.0 * eps)
            gy = (fcy - fmy) / (2.0 * eps)
            return float(gx), float(gy)

        h, w = self.arr.shape
        real_scale = (self.real_range[1] - self.real_range[0]) / float(w)
        imag_scale = (self.imag_range[1] - self.imag_range[0]) / float(h)

        left = self._lookup_array(real - real_scale, imag)
        right = self._lookup_array(real + real_scale, imag)
        down = self._lookup_array(real, imag - imag_scale)
        up = self._lookup_array(real, imag + imag_scale)

        gx = (right - left) / (2.0 * real_scale)
        gy = (up - down) / (2.0 * imag_scale)
        return float(gx), float(gy)


def load_distance_field_for_runtime(path: str) -> DistanceField:
    """Load a distance field from `path` (without extension) and return a DistanceField.

    Accepts either a directory/prefix like `mandelbrot_distance_field/data` or a
    path that includes `.npy`/`.json` suffixes.
    """
    p = Path(path)
    npy_path = p.with_suffix(".npy")
    json_path = p.with_suffix(".json")

    arr = None
    real_range = (-2.5, 1.0)
    imag_range = (-1.5, 1.5)
    slowdown_threshold = 0.02

    if not (npy_path.exists() and json_path.exists()):
        raise FileNotFoundError(
            "Precomputed mandelbrot distance field not found. Generate it with: 'python -m mandelbrot_distance_field'"
        )

    arr = np.load(str(npy_path))
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Return a wrapper object usable both from Python tests and runtime binding
    return DistanceField(
        arr=arr,
        real_range=tuple(meta.get("real_range", real_range)),
        imag_range=tuple(meta.get("imag_range", imag_range)),
        slowdown_threshold=meta.get("slowdown_threshold", slowdown_threshold),
    )
