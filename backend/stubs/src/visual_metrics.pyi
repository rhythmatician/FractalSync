"""Stubs for src.visual_metrics used by tests."""

from __future__ import annotations

from typing import TypeAlias
import numpy as np
import torch

class LossVisualMetrics:
    def __init__(self) -> None: ...
    def compute_all_metrics(
        self, image: np.ndarray, prev_image: np.ndarray | None = None
    ) -> dict[str, float]: ...
    def render_julia_set(
        self,
        seed: complex,
        width: int = 64,
        height: int = 64,
        zoom: float = 1.0,
        max_iter: int = 100,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> np.ndarray: ...
    @staticmethod
    def mandelbrot_distance_estimate(
        c: torch.Tensor | list[complex],
    ) -> torch.Tensor: ...

VisualMetrics: TypeAlias = LossVisualMetrics
