#!/usr/bin/env python3
"""Test the new mathematical curriculum and check julia parameter coverage."""

import torch
import numpy as np
from src.mandelbrot_orbits import (
    generate_curriculum_sequence,
    generate_dynamic_curriculum_orbits,
)
from src.model import AudioToVisualModel

# Generate curriculum
print("Generating curriculum with new mathematical orbits...")
positions, velocities = generate_curriculum_sequence(
    n_samples=2000, use_curriculum=True
)

print(f"Curriculum positions shape: {positions.shape}")
print(f"Curriculum velocities shape: {velocities.shape}")

# Check coverage
print("\nCurriculum coverage statistics:")
print(f"  Real: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]")
print(f"    - stdev: {positions[:, 0].std():.4f}")
print(f"  Imag: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")
print(f"    - stdev: {positions[:, 1].std():.4f}")

# Show orbit names and their coverage
print("\nOrbit coverage breakdown:")
orbits = generate_dynamic_curriculum_orbits()
for name, orbit in orbits.items():
    samples = orbit.sample(100)
    print(
        f"  {name:25s}: Real [{samples[:, 0].min():.3f}, {samples[:, 0].max():.3f}]  Imag [{samples[:, 1].min():.3f}, {samples[:, 1].max():.3f}]"
    )

# Try loading and running the latest model
print("\nLoading trained model...")
try:
    checkpoint = torch.load("checkpoints/checkpoint_epoch_10.pt", map_location="cpu")
    model = AudioToVisualModel(input_dim=60, output_dim=9)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run on curriculum data
    with torch.no_grad():
        # Normalize positions to audio feature range (0-1)
        positions_norm = (positions + 2.0) / 4.0  # Assume [-2, 2] -> [0, 1]
        positions_norm = np.clip(positions_norm, 0, 1).astype(np.float32)

        # Create simple audio features (use positions as dummy features)
        audio_features = np.zeros((positions.shape[0], 60), dtype=np.float32)
        audio_features[:, :2] = positions_norm

        inputs = torch.tensor(audio_features, dtype=torch.float32)
        outputs = model(inputs)

        print(f"  Model output shape: {outputs.shape}")
        print("  Output statistics:")
        for i in range(outputs.shape[1]):
            print(
                f"    Output {i}: [{outputs[:, i].min():.4f}, {outputs[:, i].max():.4f}] mean={outputs[:, i].mean():.4f}"
            )

except Exception as e:
    print(f"  Error loading model: {e}")

print("\n[OK] Curriculum test complete!")
