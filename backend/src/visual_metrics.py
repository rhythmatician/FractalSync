"""
Visual metrics computation for evaluating Julia set renderings.
Measures perceptual qualities like roughness, smoothness, brightness, etc.
"""

from typing import Optional

import cv2
import numpy as np
import torch
import runtime_core


def sample_distance_field(c: complex) -> float:
    """Sample the precomputed signed distance field at complex coordinate c.

    This implementation requires the Rust runtime-core sampler to be available
    and will raise if `runtime_core.sample_distance_field_py` is not exposed.

    Returns unsigned distance (abs of signed distance) as float.
    """
    real = float(c.real)
    imag = float(c.imag)

    sampled_list = runtime_core.sample_distance_field_py([real], [imag])
    sampled = sampled_list[0]
    return abs(sampled)


def _sample_distance_field(c_complex: torch.Tensor) -> torch.Tensor:
    """Sample the precomputed signed distance field at complex coordinates c_complex.

    This implementation requires the Rust runtime-core sampler to be available
    and will raise if `runtime_core.sample_distance_field_py` is not exposed.

    Returns unsigned distances (abs of signed distance) as float tensor (N,).
    """
    real = c_complex.real.to(torch.float32)
    imag = c_complex.imag.to(torch.float32)

    # Use Rust sampler if available for speed
    xs = real.detach().cpu().numpy().tolist()
    ys = imag.detach().cpu().numpy().tolist()
    sampled_list = runtime_core.sample_distance_field_py(xs, ys)
    sampled = torch.tensor(sampled_list, dtype=torch.float32, device=c_complex.device)
    return sampled.abs()


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
        width: int = 64,
        height: int = 64,
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

    @staticmethod
    def mandelbrot_distance_estimate(
        c: torch.Tensor,
        max_iter=128,
        bailout=10.0,
        eps=1e-8,
    ) -> torch.Tensor:
        """Estimate distance to the Mandelbrot boundary for a batch of points.

        Accepts either:
        - a complex-valued tensor of shape (batch,) (dtype=torch.cfloat or torch.cdouble),
        - a real tensor of shape (batch, 2) where columns are (real, imag),
        - a real tensor of shape (batch,) (imag part assumed 0), or
        - a scalar/python complex which will be converted to a single-element tensor.

        This estimator samples the precomputed signed distance field via the
        runtime-core sampler (fast, non-differentiable). If the sampler is not
        available the function will raise an error instructing developers to
        rebuild the runtime-core Python extension.

        Returns a real float tensor of shape (batch,) with non-negative distances.
        """
        if c.dtype.is_complex:
            c_complex = c.view(-1)
        else:
            # handle (N, 2) real/imag pairs
            if c.dim() == 2 and c.shape[1] == 2:
                real = c[:, 0].to(torch.get_default_dtype())
                imag = c[:, 1].to(torch.get_default_dtype())
                c_complex = torch.complex(real, imag).to(torch.complex64)
            else:
                raise TypeError(
                    "Unsupported tensor shape for mandelbrot_distance_estimate: expected (N,2) or (N,)"
                )

        sampled = _sample_distance_field(c_complex)
        return sampled.clamp_min(0.0)
