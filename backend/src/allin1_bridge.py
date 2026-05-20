"""Optional integration bridge for `allin1` (All-In-One Music Structure Analyzer).

This module does not require `allin1` to be installed; it uses an import
guard and returns `None` when unavailable. It provides helpers to fetch
frame-level activations and resample them to arbitrary time bases used in
`SongAnalyzer`.

Design goals:
- Fail gracefully when `allin1` is absent (no hard dependency).
- Keep runtime overhead minimal (use CPU by default).
- Provide small, unit-testable functions with no heavy ML work.
"""

from typing import Dict, Any
import importlib
import numpy as np

_ALLIN1 = None


def _try_import_allin1():
    global _ALLIN1
    if _ALLIN1 is not None:
        return _ALLIN1
    try:
        _ALLIN1 = importlib.import_module("allin1")
    except Exception:
        _ALLIN1 = None
    return _ALLIN1


class AllInOneUnavailable(Exception):
    pass


def available() -> bool:
    """Return True if `allin1` is importable."""
    return _try_import_allin1() is not None


def analyze_get_activations(
    path: str, device: str = "cpu", model: str = "harmonix-all"
) -> Dict[str, Any]:
    """Call `allin1.analyze` and return a dict of activations.

    Returns a dictionary with keys:
      - activations: dict with frame-level arrays (numpy arrays)
      - frame_rate: sampling rate of activations (Hz), default 100

    Raises AllInOneUnavailable if `allin1` isn't installed.
    Raises RuntimeError on unexpected failures from `allin1`.
    """
    allin1 = _try_import_allin1()
    if allin1 is None:
        raise AllInOneUnavailable("`allin1` is not installed")

    try:
        res = allin1.analyze(path, include_activations=True, model=model, device=device)
    except Exception as e:
        # Bubble up with clearer message
        raise RuntimeError(f"allin1.analyze failed: {e}") from e

    activ = getattr(res, "activations", None)
    if activ is None:
        raise RuntimeError("allin1.analyze did not return activations")

    # Convert to numpy, if they're not already
    activ_np = {k: np.asarray(v) for k, v in activ.items()}

    return {"activations": activ_np, "frame_rate": 100}


def resample_activation_to_times(
    activ: np.ndarray, src_fps: float, dst_times: np.ndarray, src_start: float = 0.0
) -> np.ndarray:
    """Resample frame-level activation `activ` (sampled at `src_fps`) to `dst_times` (seconds).

    Uses linear interpolation; values outside source range are left as 0.

    activ: shape [T]
    src_fps: frames per second for activ
    dst_times: array of times (seconds) to sample the activations at
    src_start: time of first frame (default 0.0)

    Returns: array of shape dst_times.shape
    """
    if activ.ndim != 1:
        # For safety only support 1D activations in this helper
        activ = activ.squeeze()

    n = activ.shape[0]
    src_times = src_start + np.arange(n) / float(src_fps)

    if dst_times.size == 0:
        return np.zeros(0, dtype=float)

    res = np.interp(dst_times, src_times, activ, left=0.0, right=0.0)
    return res
