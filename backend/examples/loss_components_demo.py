"""
Demo script showing how the advanced loss components work.

This demonstrates the behavior of each loss component with sample data.
"""

import torch
import numpy as np
from pathlib import Path

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loss_components import (
    MembershipProximityLoss,
    EdgeDensityCorrelationLoss,
    LobeVarietyLoss,
    NeighborhoodPenaltyLoss,
)


def demo_membership_proximity_loss():
    """Demonstrate membership proximity loss behavior."""
    print("\n" + "=" * 60)
    print("DEMO 1: Membership Proximity Loss")
    print("=" * 60)
    
    loss_fn = MembershipProximityLoss(target_membership=0.75, max_iter=50, weight=1.0)
    
    # Test points: inside M, near boundary, outside M
    test_points = [
        (0.0, 0.0, "Origin (deep inside M)"),
        (-0.5, 0.0, "Cardioid lobe center"),
        (-1.0, 0.0, "Period-2 bulb center"),
        (0.3, 0.3, "Near boundary"),
        (1.5, 0.0, "Far outside M"),
    ]
    
    print("\nMembership proxy for test points:")
    for real, imag, label in test_points:
        c_real = torch.tensor([real])
        c_imag = torch.tensor([imag])
        membership = loss_fn.compute_membership_proxy(c_real, c_imag)
        print(f"  {label:30s}: membership = {membership.item():.3f}")
    
    # Test with different audio intensities
    print("\nLoss with varying audio intensity (point outside M):")
    c_real = torch.tensor([1.5])
    c_imag = torch.tensor([0.0])
    
    for intensity in [0.0, 0.25, 0.5, 0.75, 1.0]:
        audio_intensity = torch.tensor([intensity])
        loss = loss_fn(c_real, c_imag, audio_intensity)
        print(f"  Intensity {intensity:.2f}: loss = {loss.item():.4f}")
    
    print("\nKey insight: Higher intensity → higher loss for points outside M")


def demo_edge_density_correlation_loss():
    """Demonstrate edge density correlation loss behavior."""
    print("\n" + "=" * 60)
    print("DEMO 2: Edge Density Correlation Loss")
    print("=" * 60)
    
    loss_fn = EdgeDensityCorrelationLoss(weight=1.0)
    
    # Positive correlation (what we want)
    edge_density_pos = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    spectral_centroid_pos = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    loss_pos = loss_fn(edge_density_pos, spectral_centroid_pos)
    
    # Negative correlation (what we don't want)
    edge_density_neg = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
    spectral_centroid_neg = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    loss_neg = loss_fn(edge_density_neg, spectral_centroid_neg)
    
    # No correlation
    edge_density_rand = torch.tensor([0.3, 0.7, 0.2, 0.8, 0.5])
    spectral_centroid_rand = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    loss_rand = loss_fn(edge_density_rand, spectral_centroid_rand)
    
    print("\nCorrelation scenarios:")
    print(f"  Positive correlation (desired): loss = {loss_pos.item():.4f}")
    print(f"  Negative correlation (avoid):   loss = {loss_neg.item():.4f}")
    print(f"  No correlation (neutral):       loss = {loss_rand.item():.4f}")
    
    print("\nKey insight: Negative loss values indicate positive correlation")
    print("             (which is what we want - detail matches brightness)")


def demo_lobe_variety_loss():
    """Demonstrate lobe variety loss behavior."""
    print("\n" + "=" * 60)
    print("DEMO 3: Lobe Variety Loss")
    print("=" * 60)
    
    loss_fn = LobeVarietyLoss(history_size=50, n_clusters=5, weight=1.0)
    
    # Scenario 1: Staying in one place (high loss)
    print("\nScenario 1: Staying in one place")
    for _ in range(20):
        c_real = torch.tensor([0.0])
        c_imag = torch.tensor([0.0])
        loss = loss_fn(c_real, c_imag)
    variety_stuck = loss_fn.compute_variety_score()
    print(f"  Variety score: {variety_stuck:.4f}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Scenario 2: Exploring different regions (low loss)
    loss_fn_explore = LobeVarietyLoss(history_size=50, n_clusters=5, weight=1.0)
    print("\nScenario 2: Exploring different lobes")
    lobes = [
        (0.0, 0.0),     # Cardioid
        (-1.0, 0.0),    # Period-2 bulb
        (-0.1, 0.8),    # Period-3 bulb
        (0.3, 0.3),     # Near boundary
        (-0.8, 0.2),    # Period-4 region
    ]
    
    for _ in range(4):  # Cycle through lobes multiple times
        for real, imag in lobes:
            c_real = torch.tensor([real])
            c_imag = torch.tensor([imag])
            loss = loss_fn_explore(c_real, c_imag)
    
    variety_explore = loss_fn_explore.compute_variety_score()
    print(f"  Variety score: {variety_explore:.4f}")
    print(f"  Loss: {loss.item():.4f}")
    
    print(f"\nKey insight: Exploring increases variety ({variety_explore:.3f}) vs. ")
    print(f"             staying in place ({variety_stuck:.3f}), lowering loss")


def demo_neighborhood_penalty_loss():
    """Demonstrate neighborhood penalty loss behavior."""
    print("\n" + "=" * 60)
    print("DEMO 4: Neighborhood Penalty Loss")
    print("=" * 60)
    
    loss_fn = NeighborhoodPenaltyLoss(window_size=10, min_radius=0.3, weight=1.0)
    
    # Scenario 1: Small movements (high loss)
    print("\nScenario 1: Small movements (clustered)")
    for i in range(10):
        c_real = torch.tensor([0.0 + i * 0.01])
        c_imag = torch.tensor([0.0 + i * 0.01])
        loss = loss_fn(c_real, c_imag)
    radius_small = loss_fn.compute_neighborhood_radius()
    print(f"  Neighborhood radius: {radius_small:.4f}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Scenario 2: Large movements (low loss)
    loss_fn_large = NeighborhoodPenaltyLoss(window_size=10, min_radius=0.3, weight=1.0)
    print("\nScenario 2: Large movements (spread out)")
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    for angle in angles:
        c_real = torch.tensor([np.cos(angle) * 0.5])
        c_imag = torch.tensor([np.sin(angle) * 0.5])
        loss = loss_fn_large(c_real, c_imag)
    radius_large = loss_fn_large.compute_neighborhood_radius()
    print(f"  Neighborhood radius: {radius_large:.4f}")
    print(f"  Loss: {loss.item():.4f}")
    
    print(f"\nKey insight: Large movements ({radius_large:.3f} radius) have lower loss")
    print(f"             than small movements ({radius_small:.3f} radius)")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("ADVANCED LOSS COMPONENTS DEMO")
    print("=" * 60)
    print("\nThis demo shows how each loss component behaves with sample data.")
    
    demo_membership_proximity_loss()
    demo_edge_density_correlation_loss()
    demo_lobe_variety_loss()
    demo_neighborhood_penalty_loss()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The four loss components work together to:

1. Membership Proximity: Keep visuals interesting during intense audio
   - Prevents sparse/empty fractals when music is loud
   
2. Edge Density Correlation: Match visual detail with audio brightness
   - Jagged fractals for bright audio, smooth for dark audio
   
3. Lobe Variety: Encourage exploration of different regions
   - Rewards visiting diverse lobes across the song
   
4. Neighborhood Penalty: Prevent staying in one place
   - Encourages movement on shorter timescales

Together, these create visuals that are both emotionally coherent
with the music and visually varied throughout the song.
    """)


if __name__ == "__main__":
    main()
