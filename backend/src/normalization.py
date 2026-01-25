"""Runtime normalization helpers (z-score) for model inputs.

These helpers are intentionally small and dependency-free (numpy only) so they
can be used from training, export, and lightweight runtime code paths.
"""

from typing import Tuple

import numpy as np


def safe_zscore(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Z-score normalize `features` using `mean` and `std`.

    Args:
        features: (..., D) array of features
        mean: (D,) array of per-feature means
        std: (D,) array of per-feature stds
        eps: minimum std to avoid division by zero

    Returns:
        normalized features with same shape as `features`.
    """
    if features.shape[-1] != mean.shape[0] or mean.shape[0] != std.shape[0]:
        raise ValueError("feature dimension mismatch between features, mean, and std")
    std_safe = np.where(std < eps, 1.0, std)
    return (features - mean) / std_safe


def apply_runtime_normalization(metadata: dict, features: np.ndarray) -> np.ndarray:
    """Apply runtime normalization if model metadata indicates it.

    Checks `metadata["input_normalization"]` and uses `feature_mean` and
    `feature_std` fields from metadata if present. If normalization is not
    required or metadata is missing, returns the original features unchanged.
    """
    inorm = metadata.get("input_normalization") if metadata else None
    if not inorm or inorm.get("applied_by") != "runtime":
        return features

    mean = metadata.get("feature_mean")
    std = metadata.get("feature_std")
    if mean is None or std is None:
        raise ValueError(
            "Runtime normalization requested but model metadata missing 'feature_mean' or 'feature_std'"
        )
    mean_arr = np.array(mean, dtype=float)
    std_arr = np.array(std, dtype=float)
    return safe_zscore(np.asarray(features, dtype=float), mean_arr, std_arr)
