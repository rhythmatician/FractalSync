"""
Visualize the curriculum learning trajectories through Mandelbrot parameter space.
Shows Julia sets and their positions as the model is trained.
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Tuple, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mandelbrot_orbits import generate_curriculum_sequence
from src.julia_gpu import GPUJuliaRenderer

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec


def visualize_curriculum(
    n_samples: int = 500,
    render_interval: int = 10,
    output_dir: str = "curriculum_viz",
):
    """
    Visualize curriculum learning trajectory.

    Args:
        n_samples: Total curriculum samples to generate
        render_interval: Render Julia set every N samples
        output_dir: Directory to save visualizations
    """
    print(f"Generating curriculum data ({n_samples} samples)...")
    positions, velocities = generate_curriculum_sequence(n_samples)

    print("Initializing Julia renderer...")
    renderer = GPUJuliaRenderer(width=128, height=128)

    os.makedirs(output_dir, exist_ok=True)

    # Extract real and imaginary parts
    c_real = positions[:, 0]
    c_imag = positions[:, 1]
    v_real = velocities[:, 0]
    v_imag = velocities[:, 1]

    print(f"Curriculum statistics:")
    print(
        f"  Real: min={c_real.min():.4f}, max={c_real.max():.4f}, mean={c_real.mean():.4f}"
    )
    print(
        f"  Imag: min={c_imag.min():.4f}, max={c_imag.max():.4f}, mean={c_imag.mean():.4f}"
    )
    print(
        f"  Velocity magnitude: min={np.sqrt(v_real**2 + v_imag**2).min():.4f}, "
        f"max={np.sqrt(v_real**2 + v_imag**2).max():.4f}"
    )

    # Create figure with trajectory and sample Julia sets
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Main plot: trajectory in parameter space
    ax_main = fig.add_subplot(gs[0:2, 0:2])

    # Plot Mandelbrot set boundary
    mandelbrot_x = []
    mandelbrot_y = []
    for theta in np.linspace(0, 2 * np.pi, 1000):
        c = 0.5 * complex(np.cos(theta), np.sin(theta)) - 0.25 * complex(
            np.cos(2 * theta), np.sin(2 * theta)
        )
        mandelbrot_x.append(c.real)
        mandelbrot_y.append(c.imag)

    ax_main.plot(
        mandelbrot_x,
        mandelbrot_y,
        "k-",
        alpha=0.3,
        linewidth=0.5,
        label="Mandelbrot boundary",
    )

    # Plot curriculum trajectory
    scatter = ax_main.scatter(
        c_real,
        c_imag,
        c=np.arange(len(c_real)),
        cmap="viridis",
        s=10,
        alpha=0.6,
        label="Curriculum path",
    )
    cbar = plt.colorbar(scatter, ax=ax_main)
    cbar.set_label("Sample index")

    # Add direction arrows
    arrow_interval = max(1, n_samples // 20)
    for i in range(0, n_samples - arrow_interval, arrow_interval):
        arrow = FancyArrowPatch(
            (c_real[i], c_imag[i]),
            (c_real[i + arrow_interval], c_imag[i + arrow_interval]),
            arrowstyle="-|>",
            mutation_scale=15,
            alpha=0.3,
            color="red",
            linewidth=0.5,
        )
        ax_main.add_patch(arrow)

    ax_main.set_xlabel("Real")
    ax_main.set_ylabel("Imaginary")
    ax_main.set_title(f"Curriculum Trajectory ({n_samples} samples)")
    ax_main.grid(True, alpha=0.2)
    ax_main.legend()

    # Sample Julia sets at key points
    sample_indices = [
        0,
        n_samples // 4,
        n_samples // 2,
        3 * n_samples // 4,
        n_samples - 1,
    ]

    for idx, sample_idx in enumerate(sample_indices):
        ax_julia = fig.add_subplot(gs[idx // 2, 2 + (idx % 2)])

        c = complex(c_real[sample_idx], c_imag[sample_idx])
        print(f"  Rendering Julia set {idx + 1}/5 at c={c:.4f}...")

        try:
            julia_image = renderer.render(c.real, c.imag, zoom=1.0, max_iter=50)
            ax_julia.imshow(
                julia_image, extent=[-2, 2, -2, 2], origin="lower", cmap="hot"
            )
            ax_julia.set_title(f"Julia @{sample_idx}\nc={c.real:.3f}{c.imag:+.3f}i")
            ax_julia.axis("off")
        except Exception as e:
            print(f"    Error: {e}")

    # Velocity magnitude over time
    ax_vel = fig.add_subplot(gs[2, 0:2])
    vel_mag = np.sqrt(v_real**2 + v_imag**2)
    ax_vel.plot(vel_mag, linewidth=1)
    ax_vel.set_xlabel("Sample index")
    ax_vel.set_ylabel("Velocity magnitude")
    ax_vel.set_title("Curriculum Velocity Over Time")
    ax_vel.grid(True, alpha=0.2)

    # Real/Imag over time
    ax_comp = fig.add_subplot(gs[2, 2:4])
    ax_comp.plot(c_real, label="Real", alpha=0.7, linewidth=1)
    ax_comp.plot(c_imag, label="Imag", alpha=0.7, linewidth=1)
    ax_comp.set_xlabel("Sample index")
    ax_comp.set_ylabel("Value")
    ax_comp.set_title("Real/Imaginary Components")
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.2)

    # Save
    output_path = os.path.join(output_dir, "curriculum_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    visualize_curriculum(n_samples=500)
