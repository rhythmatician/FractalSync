"""Quality gate metrics: ΔV, coverage/entropy, transient detection.

These helpers are small, pure functions intended for unit tests and the
quality gate runner script.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np

# Use canonical ΔV implementation from visual_metrics to avoid duplicating
# normalization semantics across the codebase.
from src.visual_metrics import proxy_delta_v


def window_coverage_entropy(
    c_seq: Iterable[Tuple[float, float]], bins: Tuple[int, int] = (32, 32)
) -> Tuple[float, float]:
    """Compute coverage fraction and entropy (nats) over a sequence of c (real, imag).

    Returns (coverage_fraction, entropy)
    """
    arr = np.asarray(list(c_seq), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    xs = arr[:, 0]
    ys = arr[:, 1]
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.5, 1.5

    xi = np.floor(
        (np.clip(xs, x_min, x_max) - x_min) / (x_max - x_min) * bins[0]
    ).astype(int)
    yi = np.floor(
        (np.clip(ys, y_min, y_max) - y_min) / (y_max - y_min) * bins[1]
    ).astype(int)
    xi = np.clip(xi, 0, bins[0] - 1)
    yi = np.clip(yi, 0, bins[1] - 1)

    idx = xi * bins[1] + yi
    counts = np.bincount(idx, minlength=bins[0] * bins[1]).astype(float)
    visited = counts > 0
    coverage_frac = float(np.sum(visited) / counts.size)

    probs = counts[counts > 0] / counts[counts > 0].sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-12))) if probs.size > 0 else 0.0
    return coverage_frac, entropy


def detect_transients(spectral_flux: Iterable[float], thresh: float = 0.5) -> List[int]:
    """Return indices where spectral_flux exceeds thresh (simple transient detector)."""
    arr = np.asarray(list(spectral_flux), dtype=np.float32)
    return list(np.where(arr > thresh)[0])
