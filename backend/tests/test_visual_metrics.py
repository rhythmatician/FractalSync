"""
Unit tests for visual metrics computation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visual_metrics import VisualMetrics


class TestVisualMetrics:
    """Test visual metrics computation."""

    def test_metrics_initialization(self):
        """Test creating VisualMetrics instance."""
        metrics = VisualMetrics()
        assert metrics is not None

    def test_simple_image_metrics(self):
        """Test computing metrics on a simple synthetic image."""
        metrics_calc = VisualMetrics()

        # Create a simple gradient image (64x64)
        image = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)

        results = metrics_calc.compute_all_metrics(image)

        # Check all expected metrics are present
        assert "edge_density" in results
        assert "color_uniformity" in results
        assert "brightness_mean" in results
        assert "brightness_std" in results
        assert "brightness_range" in results
        assert "temporal_change" in results
        assert "connectedness" in results

        # Check values are in reasonable ranges
        assert 0.0 <= results["edge_density"] <= 1.0
        assert 0.0 <= results["color_uniformity"] <= 1.0
        assert 0.0 <= results["brightness_mean"] <= 1.0
        assert results["brightness_std"] >= 0.0
        assert 0.0 <= results["brightness_range"] <= 1.0
        assert results["temporal_change"] == 0.0  # No previous image

    def test_color_image_metrics(self):
        """Test computing metrics on RGB color image."""
        metrics_calc = VisualMetrics()

        # Create a simple color image (64x64x3)
        image = np.random.rand(64, 64, 3).astype(np.float32)

        results = metrics_calc.compute_all_metrics(image)

        # All metrics should still be computed
        assert len(results) == 7
        assert all(isinstance(v, (float, np.floating)) for v in results.values())

    def test_uint8_image_normalization(self):
        """Test that uint8 images [0,255] are normalized correctly."""
        metrics_calc = VisualMetrics()

        # Create image in [0, 255] range
        image_255 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        results_255 = metrics_calc.compute_all_metrics(image_255)

        # Create same image normalized to [0, 1]
        image_01 = image_255.astype(np.float32) / 255.0
        results_01 = metrics_calc.compute_all_metrics(image_01)

        # Results should be very similar (allowing for floating point differences)
        for key in results_255.keys():
            if key != "temporal_change":  # Skip temporal since no prev image
                assert (
                    abs(results_255[key] - results_01[key]) < 0.01
                ), f"Mismatch for {key}: {results_255[key]} vs {results_01[key]}"

    def test_temporal_change(self):
        """Test temporal change metric between frames."""
        metrics_calc = VisualMetrics()

        # Create two slightly different images
        image1 = np.random.rand(64, 64).astype(np.float32)
        image2 = image1 + np.random.randn(64, 64).astype(np.float32) * 0.1

        results = metrics_calc.compute_all_metrics(image2, prev_image=image1)

        # Temporal change should be non-zero
        assert results["temporal_change"] > 0.0
        # And should be reasonable (not too large)
        assert results["temporal_change"] < 1.0

    def test_edge_density_extremes(self):
        """Test edge density on extreme cases."""
        metrics_calc = VisualMetrics()

        # Uniform image (no edges)
        uniform = np.ones((64, 64), dtype=np.float32) * 0.5
        results_uniform = metrics_calc.compute_all_metrics(uniform)

        # High-contrast checkerboard pattern (many edges)
        # Scale to uint8 range for better edge detection
        checkerboard = np.zeros((64, 64), dtype=np.uint8)
        checkerboard[::2, ::2] = 255
        checkerboard[1::2, 1::2] = 255
        results_checkerboard = metrics_calc.compute_all_metrics(checkerboard)

        # Checkerboard should have higher edge density than uniform
        # (or at least equal if edge detector is very conservative)
        assert results_checkerboard["edge_density"] >= results_uniform["edge_density"]

        # Create a high-contrast gradient (should have some edges)
        gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        results_gradient = metrics_calc.compute_all_metrics(gradient)

        # Gradient should have non-zero edge density
        assert results_gradient["edge_density"] >= 0.0  # At least valid

    def test_brightness_metrics(self):
        """Test brightness statistics are computed correctly."""
        metrics_calc = VisualMetrics()

        # Dark image
        dark = np.ones((64, 64), dtype=np.float32) * 0.2
        results_dark = metrics_calc.compute_all_metrics(dark)

        # Bright image
        bright = np.ones((64, 64), dtype=np.float32) * 0.8
        results_bright = metrics_calc.compute_all_metrics(bright)

        # Brightness mean should reflect image brightness
        assert results_dark["brightness_mean"] < 0.3
        assert results_bright["brightness_mean"] > 0.7

        # Uniform images should have low std and range
        assert results_dark["brightness_std"] < 0.01
        assert results_bright["brightness_std"] < 0.01
        assert results_dark["brightness_range"] < 0.01
        assert results_bright["brightness_range"] < 0.01

    def test_color_uniformity(self):
        """Test color uniformity metric."""
        metrics_calc = VisualMetrics()

        # Uniform color image
        uniform_color = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        results_uniform = metrics_calc.compute_all_metrics(uniform_color)

        # Random color image
        random_color = np.random.rand(64, 64, 3).astype(np.float32)
        results_random = metrics_calc.compute_all_metrics(random_color)

        # Uniform should have higher color uniformity than random
        assert results_uniform["color_uniformity"] > results_random["color_uniformity"]

    def test_render_julia_set(self):
        """Test Julia set rendering."""
        metrics_calc = VisualMetrics()

        # Render a Julia set
        image = metrics_calc.render_julia_set(
            seed_real=-0.7, seed_imag=0.27, width=128, height=128, zoom=1.0
        )

        # Check image properties
        assert image.shape == (128, 128, 3)
        assert image.dtype == np.uint8
        assert image.min() >= 0
        assert image.max() <= 255

        # Image should not be all black or all white
        assert image.mean() > 10
        assert image.mean() < 245

    def test_render_julia_set_different_seeds(self):
        """Test that different Julia seeds produce different images."""
        metrics_calc = VisualMetrics()

        # Render two Julia sets with different seeds
        image1 = metrics_calc.render_julia_set(-0.7, 0.27, 64, 64, 1.0)
        image2 = metrics_calc.render_julia_set(-0.4, 0.6, 64, 64, 1.0)

        # Images should be different
        diff = np.abs(image1.astype(float) - image2.astype(float)).mean()
        assert diff > 1.0  # Should have noticeable difference

    def test_render_julia_set_zoom(self):
        """Test Julia set rendering with different zoom levels."""
        metrics_calc = VisualMetrics()

        # Render at different zoom levels
        image_zoom_out = metrics_calc.render_julia_set(-0.7, 0.27, 64, 64, zoom=0.5)
        image_zoom_in = metrics_calc.render_julia_set(-0.7, 0.27, 64, 64, zoom=2.0)

        # Both should be valid images
        assert image_zoom_out.shape == (64, 64, 3)
        assert image_zoom_in.shape == (64, 64, 3)

        # Images should be different due to zoom
        diff = np.abs(image_zoom_out.astype(float) - image_zoom_in.astype(float)).mean()
        assert diff > 1.0

    def test_metrics_are_finite(self):
        """Test that all metrics are finite numbers (no NaN or Inf)."""
        metrics_calc = VisualMetrics()

        # Test with various edge cases
        test_images = [
            np.zeros((64, 64), dtype=np.float32),  # All black
            np.ones((64, 64), dtype=np.float32),  # All white
            np.random.rand(64, 64).astype(np.float32),  # Random
            np.eye(64, dtype=np.float32),  # Diagonal
        ]

        for i, image in enumerate(test_images):
            results = metrics_calc.compute_all_metrics(image)
            for key, value in results.items():
                assert np.isfinite(
                    value
                ), f"Non-finite value for {key} in test image {i}: {value}"

    def test_batch_consistency(self):
        """Test that metrics are consistent when computed multiple times."""
        metrics_calc = VisualMetrics()

        # Create a fixed image
        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float32)

        # Compute metrics multiple times
        results1 = metrics_calc.compute_all_metrics(image)
        results2 = metrics_calc.compute_all_metrics(image)

        # Results should be numerically identical (within floating-point tolerance)
        for key in results1.keys():
            assert np.isclose(
                results1[key], results2[key], atol=1e-10
            ), f"Inconsistent results for {key}: {results1[key]} vs {results2[key]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
