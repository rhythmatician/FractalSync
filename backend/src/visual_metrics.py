"""
Visual metrics computation for evaluating Julia set renderings.
Measures perceptual qualities like roughness, smoothness, brightness, etc.
"""

from typing import Optional

import cv2
import numpy as np


class LossVisualMetrics:
    """Compute loss-facing visual metrics from rendered Julia sets."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def compute_all_metrics(
        self, image: np.ndarray, prev_image: Optional[np.ndarray] = None
    ) -> dict:
        """
        Compute loss-facing visual metrics from image.

        Args:
            image: Current image array (H, W, 3) or (H, W) in [0, 255] or [0, 1]
            prev_image: Previous image for temporal metrics (optional)

        Returns:
            Dictionary of metric values
        """
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # Convert to grayscale for some metrics
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        metrics = {}

        # Temporal change rate (loss metric)
        if prev_image is not None:
            if prev_image.max() > 1.0:
                prev_image = prev_image.astype(np.float32) / 255.0
            if len(prev_image.shape) == 3:
                prev_gray = np.mean(prev_image, axis=2)
            else:
                prev_gray = prev_image

            metrics["temporal_change"] = self._compute_temporal_change(gray, prev_gray)
        else:
            metrics["temporal_change"] = 0.0

        return metrics

    def _compute_temporal_change(
        self, current: np.ndarray, previous: np.ndarray
    ) -> float:
        """
        Compute temporal change rate between frames.

        Args:
            current: Current frame
            previous: Previous frame

        Returns:
            Change rate [0, 1]
        """
        # Ensure same shape
        if current.shape != previous.shape:
            # Resize if needed
            h, w = current.shape[:2]
            prev_resized = cv2.resize(previous, (w, h))
        else:
            prev_resized = previous

        # Compute difference
        diff = np.abs(current - prev_resized)
        change_rate = np.mean(diff)

        return float(change_rate)

    def render_julia_set(
        self,
        seed_real: float,
        seed_imag: float,
        width: int = 512,
        height: int = 512,
        zoom: float = 1.0,
        max_iter: int = 100,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> np.ndarray:
        """
        Render Julia set for metrics computation.
        This is a CPU-based renderer for training.

        Args:
            seed_real: Real part of Julia seed
            seed_imag: Imaginary part of Julia seed
            width: Image width
            height: Image height
            zoom: Zoom level
            max_iter: Maximum iterations
            center_x: Center X coordinate
            center_y: Center Y coordinate

        Returns:
            Rendered image array (H, W, 3) in [0, 255]
        """
        # Create coordinate arrays
        x = np.linspace(center_x - 2.0 / zoom, center_x + 2.0 / zoom, width)
        y = np.linspace(center_y - 2.0 / zoom, center_y + 2.0 / zoom, height)

        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        # Julia set iteration
        Z = C.copy()
        iterations = np.zeros_like(C, dtype=np.int32)

        c = seed_real + 1j * seed_imag

        for i in range(max_iter):
            mask = np.abs(Z) <= 2.0
            Z[mask] = Z[mask] ** 2 + c
            iterations[mask] = i + 1

        # Normalize iterations to [0, 1]
        normalized = iterations.astype(np.float32) / max_iter

        # Apply color mapping (simple grayscale for now)
        # In practice, this would use hue/saturation/brightness from model
        image = (normalized * 255).astype(np.uint8)

        # Convert to RGB
        image_rgb = np.stack([image, image, image], axis=2)

        return image_rgb


# Backwards-compatible alias for loss metrics.
VisualMetrics = LossVisualMetrics
