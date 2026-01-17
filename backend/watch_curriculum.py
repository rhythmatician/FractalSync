"""
Watch curriculum Julia sets as an animated video.
Shows the Julia sets the model will be trained on, flowing through parameter space.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mandelbrot_orbits import generate_curriculum_sequence
from src.julia_gpu import GPUJuliaRenderer
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def watch_curriculum(
    n_samples: int = 500,
    fps: int = 30,
    julia_size: int = 512,
    output_file: str = "curriculum_viz/curriculum.gif",
):
    """
    Create an animated GIF of curriculum Julia sets.

    Args:
        n_samples: Number of curriculum samples to generate
        fps: Frames per second for animation
        julia_size: Resolution of each Julia set render
        output_file: Output GIF file path
    """
    print(f"Generating curriculum sequence ({n_samples} samples)...")
    positions, velocities = generate_curriculum_sequence(n_samples)

    c_real = positions[:, 0]
    c_imag = positions[:, 1]
    v_real = velocities[:, 0]
    v_imag = velocities[:, 1]

    print(f"Real range: [{c_real.min():.4f}, {c_real.max():.4f}]")
    print(f"Imag range: [{c_imag.min():.4f}, {c_imag.max():.4f}]")

    print(f"\nInitializing Julia renderer ({julia_size}x{julia_size})...")
    renderer = GPUJuliaRenderer(width=julia_size, height=julia_size)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    frames = []

    print(f"Rendering {n_samples} Julia sets...")
    for idx in range(n_samples):
        if (idx + 1) % max(1, n_samples // 10) == 0:
            print(f"  {idx + 1}/{n_samples}...")

        c = complex(c_real[idx], c_imag[idx])
        vel_mag = np.sqrt(v_real[idx] ** 2 + v_imag[idx] ** 2)

        # Render Julia set
        try:
            julia_img = renderer.render(c.real, c.imag, zoom=1.0, max_iter=50)
        except Exception as e:
            print(f"    Warning: render failed at idx {idx}: {e}")
            # Create blank image
            julia_img = np.zeros((julia_size, julia_size, 3), dtype=np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(julia_img)

        # Add text overlay
        draw = ImageDraw.Draw(img)
        text = f"Sample {idx + 1}/{n_samples}\nc = {c.real:.4f} {c.imag:+.4f}i\nv = {vel_mag:.4f}"

        # Simple text rendering (no fancy font needed)
        try:
            draw.text((10, 10), text, fill=(255, 255, 255))
        except:
            # Fallback if font loading fails
            pass

        frames.append(img)

    print(f"\nSaving animation to {output_file}...")
    if frames:
        # Save as GIF with loop
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,  # Convert fps to milliseconds per frame
            loop=0,  # Loop forever
            optimize=False,
        )
        print(f"✓ Animation saved: {output_file}")
        print(f"  Duration: {len(frames) / fps:.1f} seconds at {fps} fps")
    else:
        print("Error: No frames were rendered")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch curriculum Julia sets")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--size", type=int, default=512, help="Julia set resolution")
    parser.add_argument(
        "--output",
        type=str,
        default="curriculum_viz/curriculum.gif",
        help="Output file",
    )

    args = parser.parse_args()

    watch_curriculum(
        n_samples=args.samples,
        fps=args.fps,
        julia_size=args.size,
        output_file=args.output,
    )
