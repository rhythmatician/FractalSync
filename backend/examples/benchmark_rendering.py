#!/usr/bin/env python3
"""
Quick benchmark comparing old vs new rendering performance.
Run with: python examples/benchmark_rendering.py
"""

import time
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from src.julia_gpu import GPUJuliaRenderer  # noqa: E402
from src.visual_metrics import VisualMetrics  # noqa: E402


def benchmark_old_rendering(num_samples: int = 10):
    """Benchmark old 128x128 rendering with visual_metrics."""
    visual_metrics = VisualMetrics()

    seeds = [
        (-0.7, 0.27),
        (-0.4, 0.6),
        (0.0, 0.0),
        (-0.8, 0.156),
        (-0.162, 1.04),
    ] * 2  # 10 samples total

    start = time.time()
    for seed_real, seed_imag in seeds[:num_samples]:
        visual_metrics.render_julia_set(
            seed_real=seed_real,
            seed_imag=seed_imag,
            width=128,
            height=128,
            max_iter=100,
        )
    old_time = time.time() - start

    return old_time


def benchmark_new_rendering(num_samples: int = 10, use_gpu: bool = True):
    """Benchmark new 64x64 rendering with GPU acceleration."""
    renderer = GPUJuliaRenderer(width=64, height=64)

    seeds = [
        (-0.7, 0.27),
        (-0.4, 0.6),
        (0.0, 0.0),
        (-0.8, 0.156),
        (-0.162, 1.04),
    ] * 2  # 10 samples total

    start = time.time()
    for seed_real, seed_imag in seeds[:num_samples]:
        renderer.render(
            seed_real=seed_real,
            seed_imag=seed_imag,
            max_iter=50,
        )
    new_time = time.time() - start

    return new_time


if __name__ == "__main__":
    print("Julia Set Rendering Benchmark")
    print("=" * 50)

    num_samples = 10
    print(f"\nBenchmarking {num_samples} samples...")

    # Old rendering
    print("\nOld (128x128, max_iter=100, visual_metrics):")
    old_time = benchmark_old_rendering(num_samples)
    print(f"  Total time: {old_time:.2f}s")
    print(f"  Per sample: {old_time/num_samples:.3f}s")

    # New rendering
    print("\nNew (64x64, max_iter=50, julia_gpu):")
    new_time = benchmark_new_rendering(num_samples)
    print(f"  Total time: {new_time:.2f}s")
    print(f"  Per sample: {new_time/num_samples:.3f}s")

    # Calculate speedup
    speedup = old_time / new_time
    print(f"\nSpeedup: {speedup:.1f}x faster")

    print("\n" + "=" * 50)
    print("For full training (e.g., 100 epochs x 32 samples/batch):")
    total_renders_old = 100 * 32  # epochs * batch_size
    total_renders_new = total_renders_old
    print(f"  Old: {total_renders_old * (old_time/num_samples)/60:.1f} minutes")
    print(f"  New: {total_renders_new * (new_time/num_samples)/60:.1f} minutes")
    print(
        f"  Savings: {(total_renders_old * (old_time/num_samples) - total_renders_new * (new_time/num_samples))/60:.1f} minutes"
    )
