"""Convenience helpers for calling runtime_core.mandelbrot_distance_estimate
from a variety of Python types (torch.Tensor, numpy.ndarray, sequences).

The goal is to centralize coercion logic so callers (e.g., LossVisualMetrics)
don't replicate it and so we can optimize zero-copy paths for numpy/torch.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore[assignment]

import runtime_core


def _to_xy_from_numpy(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) as 1D numpy float arrays for a numpy input.

    Supports:
    - complex dtype arrays of shape (N,) or (N,)
    - real arrays of shape (N, 2)
    """
    if np.iscomplexobj(arr):
        xs = arr.real.ravel()
        ys = arr.imag.ravel()
        return xs, ys
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].ravel(), arr[:, 1].ravel()
    raise TypeError("Unsupported numpy array shape for mandelbrot input")


def mandelbrot_distance_estimate_from_any(
    points: Union[Sequence[complex], np.ndarray, "torch.Tensor"],
    max_iter: int = 8192,
    bailout: float = 1e6,
    perimeter_samples: int = 512,
    clamp: float = 0.06,
) -> list[float]:
    """Coerce `points` into x and y coordinate lists and call runtime_core.

    Accepts:
    - torch.Tensor of dtype complex64 or shape (N,2) real pairs (CPU tensors)
    - numpy.ndarray of complex dtype (N,) or real shape (N,2)
    - Python sequence of complex numbers

    Returns:
        list of floats (signed distances) as returned by runtime_core.
    """
    # torch.Tensor pathway (zero-copy to numpy when possible)
    if torch is not None and isinstance(points, torch.Tensor):
        if points.device.type != "cpu":
            points = points.cpu()
        if points.is_complex():
            arr = points.detach().numpy()
            xs, ys = _to_xy_from_numpy(arr)
        elif points.dim() == 2 and points.shape[1] == 2:
            arr = points.detach().numpy()
            xs, ys = arr[:, 0].ravel(), arr[:, 1].ravel()
        else:
            raise TypeError("Unsupported torch tensor shape/dtype for mandelbrot input")
    elif isinstance(points, np.ndarray):
        xs, ys = _to_xy_from_numpy(points)
    else:
        # sequence of complex numbers
        xs = np.array([float(complex(p).real) for p in points], dtype=float)
        ys = np.array([float(complex(p).imag) for p in points], dtype=float)

    # Call the runtime-core binding with Python lists (cheap views via .tolist()).
    # The binding takes (coords) or (xs, ys); max_iter/bailout/perimeter_samples/
    # clamp use the binding's internal defaults (8192 / 1e6 / 512).
    xs_list = xs.tolist()
    ys_list = ys.tolist()
    return runtime_core.mandelbrot_distance_estimate(xs_list, ys_list)
