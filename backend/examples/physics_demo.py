"""
Example script demonstrating physics-based model usage.

This script shows how to:
1. Create preset Mandelbrot orbits
2. Initialize a physics-based model
3. Simulate velocity predictions and position integration
4. Visualize the trajectory

Run: python examples/physics_demo.py
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mandelbrot_orbits import (
    get_preset_orbit,
    list_preset_names,
    generate_curriculum_sequence,
)


def demo_preset_orbits():
    """Demonstrate preset Mandelbrot orbit generation."""
    print("=" * 60)
    print("Demo 1: Preset Mandelbrot Orbits")
    print("=" * 60)

    # List available presets
    presets = list_preset_names()
    print(f"\nAvailable presets: {presets}")

    # Get and visualize cardioid orbit
    cardioid = get_preset_orbit("cardioid")
    print(f"\nCardioid orbit:")
    print(f"  Name: {cardioid.name}")
    print(f"  Points: {len(cardioid.points)}")
    print(f"  Closed: {cardioid.closed}")

    # Sample points
    samples = cardioid.sample(n_samples=100)
    print(f"  Sampled {len(samples)} points")

    # Compute velocities
    velocities = cardioid.compute_velocities(n_samples=100)
    print(f"  Computed {len(velocities)} velocity vectors")
    print(
        f"  Average velocity magnitude: {np.mean(np.linalg.norm(velocities, axis=1)):.4f}"
    )

    return cardioid, samples, velocities


def demo_curriculum_sequence():
    """Demonstrate curriculum learning sequence generation."""
    print("\n" + "=" * 60)
    print("Demo 2: Curriculum Learning Sequence")
    print("=" * 60)

    # Generate curriculum sequence
    n_samples = 500
    positions, velocities = generate_curriculum_sequence(n_samples)

    print(f"\nGenerated {n_samples} curriculum samples")
    print(f"Positions shape: {positions.shape}")
    print(f"Velocities shape: {velocities.shape}")

    print(f"\nPosition statistics:")
    print(f"  Real range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Imag range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")

    print(f"\nVelocity statistics:")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(velocities, axis=1)):.4f}")
    print(f"  Max magnitude: {np.max(np.linalg.norm(velocities, axis=1)):.4f}")

    return positions, velocities


def demo_physics_integration():
    """Demonstrate physics-based velocity integration."""
    print("\n" + "=" * 60)
    print("Demo 3: Physics Integration Simulation")
    print("=" * 60)

    # Simulate a simple trajectory with velocity integration
    n_steps = 100
    dt = 0.1
    damping = 0.95

    # Initial conditions
    position = np.array([0.0, 0.0])
    velocity = np.array([0.1, 0.05])

    positions = [position.copy()]
    velocities = [velocity.copy()]

    print("\nSimulating physics integration:")
    print(f"  Steps: {n_steps}")
    print(f"  dt: {dt}")
    print(f"  Damping: {damping}")

    for step in range(n_steps):
        # Apply damping
        velocity *= damping

        # Integrate position
        position += velocity * dt

        # Constrain to |c| < 2
        magnitude = np.linalg.norm(position)
        if magnitude > 2.0:
            position = position / magnitude * 2.0
            # Reflect velocity
            velocity = -velocity * 0.5

        positions.append(position.copy())
        velocities.append(velocity.copy())

    positions = np.array(positions)
    velocities = np.array(velocities)

    print(f"\nFinal position: [{positions[-1, 0]:.4f}, {positions[-1, 1]:.4f}]")
    print(f"Final velocity magnitude: {np.linalg.norm(velocities[-1]):.4f}")
    print(
        f"Position stayed within bounds: {np.all(np.linalg.norm(positions, axis=1) <= 2.01)}"
    )


def main():
    """Run all demos."""
    print("\nPhysics-Based Model Demo")
    print("=" * 60)

    # Demo 1: Preset orbits
    demo_preset_orbits()

    # Demo 2: Curriculum sequence
    demo_curriculum_sequence()

    # Demo 3: Physics integration
    demo_physics_integration()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNext steps:")
    print(
        "1. Train a physics model: python train_physics.py --data-dir data/audio --use-curriculum"
    )
    print("2. Compare with orbit control model: python train_orbit.py --data-dir data/audio")
    print("3. Read documentation: backend/docs/PHYSICS_MODEL.md")


if __name__ == "__main__":
    main()
