"""
Visual metrics computation for evaluating Julia set renderings.
Measures perceptual qualities like roughness, smoothness, brightness, etc.
"""

from typing import Optional

import cv2
import numpy as np


class VisualMetrics:
    """Compute perceptual metrics from rendered Julia sets."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def compute_all_metrics(
        self, image: np.ndarray, prev_image: Optional[np.ndarray] = None
    ) -> dict:
        """
        Compute all visual metrics from image.

        Args:
            image: Current image array (H, W, 3) or (H, W) in [0, 255] or [0, 1]
            prev_image: Previous image for temporal metrics (optional)

        Returns:
            Dictionary of metric values
        """
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            # Normalize by actual max value, not fixed 255
            # This handles cases where max_iter is low
            image = image.astype(np.float32) / (image.max() + 1e-8)

        # Convert to grayscale for some metrics
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        metrics = {}

        # Edge density (roughness/complexity)
        metrics["edge_density"] = self._compute_edge_density(gray)

        # Color uniformity (smoothness)
        if len(image.shape) == 3:
            metrics["color_uniformity"] = self._compute_color_uniformity(image)
        else:
            metrics["color_uniformity"] = self._compute_color_uniformity(
                np.stack([gray, gray, gray], axis=2)
            )

        # Brightness distribution
        metrics["brightness_mean"] = float(np.mean(gray))
        metrics["brightness_std"] = float(np.std(gray))
        metrics["brightness_range"] = float(np.max(gray) - np.min(gray))

        # Temporal change rate
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

        # Connectedness (fractal dimension approximation)
        metrics["connectedness"] = self._compute_connectedness(gray)

        return metrics

    def _compute_edge_density(self, gray: np.ndarray) -> float:
        """
        Compute edge density using Canny edge detection.
        Higher values indicate more roughness/complexity.

        Args:
            gray: Grayscale image [0, 1]

        Returns:
            Edge density [0, 1]
        """
        # Scale to [0, 255] for OpenCV
        gray_uint8 = (gray * 255).astype(np.uint8)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_uint8, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Compute edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        return float(edge_density)

    def _compute_color_uniformity(self, image: np.ndarray) -> float:
        """
        Compute color uniformity (inverse of variance).
        Higher values indicate smoother, more uniform colors.

        Args:
            image: RGB image [0, 1]

        Returns:
            Color uniformity [0, 1]
        """
        # Compute local variance using a small kernel
        # Lower variance = more uniform

        # Convert to grayscale for simplicity
        gray = np.mean(image, axis=2)

        # Compute local variance
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

        # Convolve to get local mean
        local_mean = cv2.filter2D(gray, -1, kernel)

        # Compute local variance
        local_var = cv2.filter2D((gray - local_mean) ** 2, -1, kernel)

        # Average variance (lower is more uniform)
        avg_variance = float(np.mean(local_var))

        # Convert to uniformity score [0, 1] (inverse relationship)
        uniformity = 1.0 / (1.0 + avg_variance * 10.0)

        return float(uniformity)

    def _compute_temporal_change(
        self, current: np.ndarray, previous: np.ndarray
    ) -> float:
        """
        Compute temporal change rate between frames using the canonical ΔV
        computation (normalizes frames before computing mean absolute difference).

        Args:
            current: Current frame
            previous: Previous frame

        Returns:
            Change rate [0, 1]
        """
        # Ensure same shape
        if current.shape != previous.shape:
            h, w = current.shape[:2]
            prev_resized = cv2.resize(previous, (w, h))
        else:
            prev_resized = previous

        # Convert to grayscale if needed
        if len(current.shape) == 3:
            curr_gray = np.mean(current, axis=2)
        else:
            curr_gray = current
        if len(prev_resized.shape) == 3:
            prev_gray = np.mean(prev_resized, axis=2)
        else:
            prev_gray = prev_resized

        # Use same normalization as proxy_delta_v logic: normalize by combined min/max
        lo = float(min(float(curr_gray.min()), float(prev_gray.min())))
        hi = float(max(float(curr_gray.max()), float(prev_gray.max())))
        if hi - lo < 1e-12:
            # constant image -> midpoint
            p = np.full_like(prev_gray, 0.5)
            c = np.full_like(curr_gray, 0.5)
        else:
            p = (prev_gray - lo) / (hi - lo)
            c = (curr_gray - lo) / (hi - lo)

        return float(np.mean(np.abs(c - p)))

    def _compute_connectedness(self, gray: np.ndarray) -> float:
        """
        Compute connectedness using fractal dimension approximation.
        Higher values indicate more connected structures.

        Args:
            gray: Grayscale image [0, 1]

        Returns:
            Connectedness score [0, 1]
        """
        # Threshold to binary
        threshold = 0.5
        binary = (gray > threshold).astype(np.uint8)

        # Count connected components
        num_labels, labels = cv2.connectedComponents(binary)

        # Compute component sizes
        component_sizes = []
        for label in range(1, num_labels):  # Skip background (label 0)
            component_size = np.sum(labels == label)
            component_sizes.append(component_size)

        if len(component_sizes) == 0:
            return 0.0

        # Connectedness: ratio of largest component to total area
        total_area = binary.shape[0] * binary.shape[1]
        largest_component = max(component_sizes)
        connectedness = largest_component / total_area

        # Also consider number of components (fewer = more connected)
        num_components_factor = 1.0 / (1.0 + len(component_sizes) / 100.0)

        # Combined score
        score = connectedness * num_components_factor

        return float(score)

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


# Canonical ΔV helper available for other modules
def proxy_delta_v(prev: np.ndarray, curr: np.ndarray) -> float:
    """Compute scalar ΔV between two proxy frames (canonical implementation).

    Both frames are 2D arrays (H, W) or flattened. Returns normalized mean
    absolute difference in [0, 1].
    """
    prev_arr = np.asarray(prev, dtype=np.float32)
    curr_arr = np.asarray(curr, dtype=np.float32)
    if prev_arr.shape != curr_arr.shape:
        raise ValueError("proxy shapes must match")

    lo = min(float(prev_arr.min()), float(curr_arr.min()))
    hi = max(float(prev_arr.max()), float(curr_arr.max()))
    if hi - lo < 1e-12:
        p = np.full_like(prev_arr, 0.5)
        c = np.full_like(curr_arr, 0.5)
    else:
        p = (prev_arr - lo) / (hi - lo)
        c = (curr_arr - lo) / (hi - lo)
    return float(np.mean(np.abs(c - p)))
