"""
Benchmark script to compare CPU vs GPU performance for distance field generation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_distance_field import build_mask, build_mask_gpu


def benchmark_resolution(res, max_iter):
    """Benchmark both CPU and GPU at a given resolution."""
    print(f"\n{'='*60}")
    print(f"Resolution: {res}x{res}, Max iterations: {max_iter}")
    print('='*60)
    
    params = {
        "res": res,
        "xmin": -2.5,
        "xmax": 1.5,
        "ymin": -2.0,
        "ymax": 2.0,
        "max_iter": max_iter,
        "bailout": 4.0,
    }
    
    # CPU benchmark
    print("Running CPU implementation...")
    start = time.time()
    cpu_mask = build_mask(**params)
    cpu_time = time.time() - start
    print(f"  CPU time: {cpu_time:.3f}s")
    
    # GPU benchmark (will fall back to CPU if GPU unavailable)
    print("Running GPU implementation...")
    start = time.time()
    result = build_mask_gpu(**params)
    gpu_time = time.time() - start
    
    # Handle tuple return value
    if isinstance(result, tuple):
        gpu_mask, actually_used_gpu = result
        if actually_used_gpu:
            print(f"  GPU time: {gpu_time:.3f}s (actual GPU)")
        else:
            print(f"  GPU time: {gpu_time:.3f}s (fell back to CPU)")
    else:
        gpu_mask = result
        print(f"  GPU time: {gpu_time:.3f}s")
    
    # Speedup
    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"  Speedup: {speedup:.2f}x")
    
    # Check agreement
    import numpy as np
    agreement = np.sum(cpu_mask == gpu_mask) / (res * res)
    print(f"  Agreement: {agreement*100:.2f}%")


if __name__ == "__main__":
    print("Distance Field Generation Benchmark")
    print("CPU vs GPU (OpenGL) comparison")
    
    # Test various resolutions
    resolutions = [
        (128, 256),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]
    
    for res, max_iter in resolutions:
        try:
            benchmark_resolution(res, max_iter)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("\n" + "="*60)
    print("Note: In environments without GPU/display, GPU will fall back to CPU.")
    print("Run on a system with OpenGL support to see actual GPU acceleration.")
    print("="*60)
