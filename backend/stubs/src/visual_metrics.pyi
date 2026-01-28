"""Stubs for src.visual_metrics used by tests."""

from __future__ import annotations

import numpy as np

class VisualMetrics:
    def __init__(self) -> None: ...
    def compute_all_metrics(self, image: np.ndarray) -> dict[str, float]: ...
