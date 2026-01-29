"""Stubs for src.visual_metrics used by tests."""

from __future__ import annotations

from typing import TypeAlias
import numpy as np

class LossVisualMetrics:
    def __init__(self) -> None: ...
    def compute_all_metrics(
        self, image: np.ndarray, prev_image: np.ndarray | None = None
    ) -> dict[str, float]: ...

VisualMetrics: TypeAlias = LossVisualMetrics
