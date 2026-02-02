"""
Tests for build_distance_field.py GPU acceleration.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from build_distance_field import build_mask, build_mask_gpu, build_signed_distance


def test_build_mask_cpu():
    """Test CPU-based mask generation."""
    res = 64
    inside = build_mask(
        res=res,
        xmin=-2.5,
        xmax=1.5,
        ymin=-2.0,
        ymax=2.0,
        max_iter=256,
        bailout=4.0,
    )
    
    assert inside.shape == (res, res)
    assert inside.dtype == np.bool_
    # The Mandelbrot set should have both inside and outside points
    assert inside.any()
    assert (~inside).any()


def test_build_mask_gpu_fallback():
    """Test GPU mask generation (will fall back to CPU in most test environments)."""
    res = 64
    inside = build_mask_gpu(
        res=res,
        xmin=-2.5,
        xmax=1.5,
        ymin=-2.0,
        ymax=2.0,
        max_iter=256,
        bailout=4.0,
    )
    
    assert inside.shape == (res, res)
    assert inside.dtype == np.bool_
    assert inside.any()
    assert (~inside).any()


def test_cpu_gpu_consistency():
    """Test that CPU and GPU implementations produce similar results."""
    res = 64
    params = {
        "res": res,
        "xmin": -2.5,
        "xmax": 1.5,
        "ymin": -2.0,
        "ymax": 2.0,
        "max_iter": 256,
        "bailout": 4.0,
    }
    
    cpu_mask = build_mask(**params)
    gpu_mask = build_mask_gpu(**params)
    
    # They should be very similar (or identical if GPU falls back to CPU)
    # Allow for minor differences due to floating point precision
    agreement = np.sum(cpu_mask == gpu_mask) / (res * res)
    assert agreement > 0.99, f"CPU and GPU masks only agree on {agreement*100:.1f}% of pixels"


def test_signed_distance_transform():
    """Test signed distance transform computation."""
    # Create a simple test mask (a square in the middle)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 20:44] = True
    
    pixel_scale = 1.0
    signed = build_signed_distance(mask, pixel_scale)
    
    assert signed.shape == mask.shape
    assert signed.dtype == np.float32
    
    # Points inside should have negative values
    assert signed[32, 32] < 0
    
    # Points outside should have positive values
    assert signed[0, 0] > 0
    assert signed[63, 63] > 0


def test_mandelbrot_properties():
    """Test that the Mandelbrot set has expected properties."""
    res = 128
    inside = build_mask(
        res=res,
        xmin=-2.5,
        xmax=1.5,
        ymin=-2.0,
        ymax=2.0,
        max_iter=512,
        bailout=4.0,
    )
    
    # The origin (0, 0) should be inside the set
    # Map (0, 0) to pixel coordinates
    x_coord = int((0.0 - (-2.5)) / (1.5 - (-2.5)) * res)
    y_coord = int((0.0 - (-2.0)) / (2.0 - (-2.0)) * res)
    assert inside[y_coord, x_coord], "Origin should be inside Mandelbrot set"
    
    # Point far outside should not be in the set
    # (-2.0, 0.0) is outside
    x_outside = int((-2.0 - (-2.5)) / (1.5 - (-2.5)) * res)
    y_outside = int((0.0 - (-2.0)) / (2.0 - (-2.0)) * res)
    assert not inside[y_outside, x_outside], "Point at (-2.0, 0.0) should be outside"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
