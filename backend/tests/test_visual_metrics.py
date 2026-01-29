"""
Unit tests for visual metrics computation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visual_metrics import LossVisualMetrics  # noqa: E402


def _compute_runtime_metrics(runtime_core, image: np.ndarray, c_real: float, c_imag: float):
    if image.max() > 1.0:
        image = image.astype(np.float64) / 255.0
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    flat = image.astype(np.float64).reshape(-1).tolist()
    return runtime_core.compute_runtime_visual_metrics(
        flat, width, height, channels, c_real, c_imag, 50
    )


class TestLossVisualMetrics:
    """Test loss-only visual metrics computation."""

    def test_metrics_initialization(self):
        """Test creating LossVisualMetrics instance."""
        metrics = LossVisualMetrics()
        assert metrics is not None

    def test_temporal_change(self):
        """Test temporal change metric between frames."""
        metrics_calc = LossVisualMetrics()

        image1 = np.random.rand(64, 64).astype(np.float32)
        image2 = image1 + np.random.randn(64, 64).astype(np.float32) * 0.1

        results = metrics_calc.compute_all_metrics(image2, prev_image=image1)

        assert "temporal_change" in results
        assert results["temporal_change"] > 0.0
        assert results["temporal_change"] < 1.0

    def test_render_julia_set(self):
        """Test Julia set rendering."""
        metrics_calc = LossVisualMetrics()

        image = metrics_calc.render_julia_set(
            seed_real=-0.7, seed_imag=0.27, width=128, height=128, zoom=1.0
        )

        assert image.shape == (128, 128, 3)
        assert image.dtype == np.uint8
        assert image.min() >= 0
        assert image.max() <= 255


class TestRuntimeVisualMetrics:
    """Test Rust-backed runtime visual metrics."""

    def test_runtime_metrics_ranges(self, runtime_core_module):
        """Ensure runtime metrics are computed and within expected ranges."""
        image = np.random.rand(32, 32, 3).astype(np.float32)
        metrics = _compute_runtime_metrics(runtime_core_module, image, 0.0, 0.0)

        assert 0.0 <= metrics.edge_density <= 1.0
        assert 0.0 <= metrics.color_uniformity <= 1.0
        assert 0.0 <= metrics.brightness_mean <= 1.0
        assert metrics.brightness_std >= 0.0
        assert 0.0 <= metrics.brightness_range <= 1.0
        assert metrics.mandelbrot_membership is True

    def test_mandelbrot_membership_outside(self, runtime_core_module):
        """Ensure membership returns False for an escaping point."""
        image = np.zeros((8, 8), dtype=np.float32)
        metrics = _compute_runtime_metrics(runtime_core_module, image, 2.0, 0.0)
        assert metrics.mandelbrot_membership is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
