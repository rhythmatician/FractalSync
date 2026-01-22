"""
Unit tests for advanced loss components.
"""

import numpy as np
import pytest
import torch

from backend.src.loss_components import (
    MembershipProximityLoss,
    EdgeDensityCorrelationLoss,
    LobeVarietyLoss,
    NeighborhoodPenaltyLoss,
)


class TestMembershipProximityLoss:
    """Tests for membership proximity loss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = MembershipProximityLoss(target_membership=0.75, max_iter=50, weight=1.0)

        # Create sample c values - some inside M, some outside
        c_real = torch.tensor([0.0, -0.5, 0.3, -1.0])
        c_imag = torch.tensor([0.0, 0.0, 0.3, 0.0])
        audio_intensity = torch.tensor([0.5, 0.8, 1.0, 0.3])

        loss = loss_fn(c_real, c_imag, audio_intensity)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_membership_proxy(self):
        """Test membership proxy computation."""
        loss_fn = MembershipProximityLoss(max_iter=50)

        # Point clearly inside M (origin)
        c_real_inside = torch.tensor([0.0])
        c_imag_inside = torch.tensor([0.0])
        membership_inside = loss_fn.compute_membership_proxy(c_real_inside, c_imag_inside)
        assert membership_inside.item() > 0.9  # Should not escape

        # Point clearly outside M
        c_real_outside = torch.tensor([2.0])
        c_imag_outside = torch.tensor([2.0])
        membership_outside = loss_fn.compute_membership_proxy(c_real_outside, c_imag_outside)
        assert membership_outside.item() < 0.1  # Should escape quickly

    def test_intensity_weighting(self):
        """Test that higher intensity increases loss."""
        loss_fn = MembershipProximityLoss(
            target_membership=0.75, weight=1.0, use_boundary_mode=False
        )

        # Point outside M
        c_real = torch.tensor([1.5, 1.5])
        c_imag = torch.tensor([0.0, 0.0])

        # Low intensity
        low_intensity = torch.tensor([0.1, 0.1])
        loss_low = loss_fn(c_real, c_imag, low_intensity)

        # High intensity
        high_intensity = torch.tensor([1.0, 1.0])
        loss_high = loss_fn(c_real, c_imag, high_intensity)

        # Higher intensity should produce higher loss
        assert loss_high.item() > loss_low.item()

    def test_boundary_mode(self):
        """Test boundary proximity mode for beautiful Julia sets."""
        loss_fn = MembershipProximityLoss(
            target_membership=0.75,
            boundary_membership=0.95,
            boundary_width=0.1,
            max_iter=50,
            weight=1.0,
            use_boundary_mode=True,
        )

        # Point on/near boundary (should have low loss at high intensity)
        # Using c = -0.75 which is near the boundary
        c_real_boundary = torch.tensor([-0.75])
        c_imag_boundary = torch.tensor([0.0])
        high_intensity = torch.tensor([1.0])

        loss_boundary = loss_fn(c_real_boundary, c_imag_boundary, high_intensity)

        # Point deep inside M (should have higher loss at high intensity)
        c_real_inside = torch.tensor([0.0])
        c_imag_inside = torch.tensor([0.0])
        loss_inside = loss_fn(c_real_inside, c_imag_inside, high_intensity)

        # Boundary point should have lower or similar loss during high intensity
        # (we want to encourage boundary proximity)
        assert loss_boundary.item() >= 0.0
        assert loss_inside.item() >= 0.0

    def test_boundary_vs_simple_mode(self):
        """Test that boundary mode behaves differently from simple mode."""
        # Create two loss functions with different modes
        loss_fn_boundary = MembershipProximityLoss(
            target_membership=0.75,
            boundary_membership=0.95,
            max_iter=50,
            weight=1.0,
            use_boundary_mode=True,
        )
        
        loss_fn_simple = MembershipProximityLoss(
            target_membership=0.75,
            max_iter=50,
            weight=1.0,
            use_boundary_mode=False,
        )

        # Test with high intensity and c deep inside M
        c_real = torch.tensor([0.0])
        c_imag = torch.tensor([0.0])
        high_intensity = torch.tensor([1.0])

        loss_boundary = loss_fn_boundary(c_real, c_imag, high_intensity)
        loss_simple = loss_fn_simple(c_real, c_imag, high_intensity)

        # Boundary mode should penalize being deep inside M at high intensity
        # Simple mode should have zero loss (membership > target)
        assert loss_boundary.item() > 0.0  # Penalizes deep inside
        assert loss_simple.item() == 0.0  # No penalty (above threshold)


class TestEdgeDensityCorrelationLoss:
    """Tests for edge density correlation loss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = EdgeDensityCorrelationLoss(weight=1.0)

        edge_density = torch.tensor([0.1, 0.3, 0.5, 0.7])
        spectral_centroid = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0])

        loss = loss_fn(edge_density, spectral_centroid)

        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_positive_correlation(self):
        """Test that positive correlation gives negative loss."""
        loss_fn = EdgeDensityCorrelationLoss(weight=1.0)

        # Perfectly correlated
        edge_density = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        spectral_centroid = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])

        loss = loss_fn(edge_density, spectral_centroid)

        # Positive correlation should give negative loss (we want to maximize)
        assert loss.item() < 0.0

    def test_negative_correlation(self):
        """Test that negative correlation gives positive loss."""
        loss_fn = EdgeDensityCorrelationLoss(weight=1.0)

        # Negatively correlated
        edge_density = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
        spectral_centroid = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])

        loss = loss_fn(edge_density, spectral_centroid)

        # Negative correlation should give positive loss (we want to minimize)
        assert loss.item() > 0.0


class TestLobeVarietyLoss:
    """Tests for lobe variety loss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = LobeVarietyLoss(history_size=10, n_clusters=3, weight=1.0)

        c_real = torch.tensor([0.0, 0.1, 0.2])
        c_imag = torch.tensor([0.0, 0.1, 0.2])

        loss = loss_fn(c_real, c_imag)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_history_tracking(self):
        """Test that history is tracked correctly."""
        loss_fn = LobeVarietyLoss(history_size=5, n_clusters=3, weight=1.0)

        # Add several points
        c_real = torch.tensor([0.0, 0.1, 0.2])
        c_imag = torch.tensor([0.0, 0.1, 0.2])
        loss_fn.update_history(c_real, c_imag)

        assert len(loss_fn.c_history) == 3

        # Add more to exceed history size
        c_real = torch.tensor([0.3, 0.4, 0.5])
        c_imag = torch.tensor([0.3, 0.4, 0.5])
        loss_fn.update_history(c_real, c_imag)

        # Should be trimmed to history_size
        assert len(loss_fn.c_history) == 5

    def test_variety_score(self):
        """Test variety score computation."""
        loss_fn = LobeVarietyLoss(history_size=10, n_clusters=3, weight=1.0)

        # Add points with low variety (clustered)
        for _ in range(5):
            c_real = torch.tensor([0.0])
            c_imag = torch.tensor([0.0])
            loss_fn.update_history(c_real, c_imag)

        variety_low = loss_fn.compute_variety_score()

        # Reset and add points with high variety (spread out)
        loss_fn.c_history = []
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
        for real, imag in points:
            c_real = torch.tensor([real])
            c_imag = torch.tensor([imag])
            loss_fn.update_history(c_real, c_imag)

        variety_high = loss_fn.compute_variety_score()

        # Spread out points should have higher variety
        assert variety_high > variety_low


class TestNeighborhoodPenaltyLoss:
    """Tests for neighborhood penalty loss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = NeighborhoodPenaltyLoss(window_size=10, min_radius=0.1, weight=1.0)

        c_real = torch.tensor([0.0, 0.05, 0.1])
        c_imag = torch.tensor([0.0, 0.05, 0.1])

        loss = loss_fn(c_real, c_imag)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_small_neighborhood_penalty(self):
        """Test that staying in small neighborhood is penalized."""
        loss_fn = NeighborhoodPenaltyLoss(window_size=10, min_radius=0.5, weight=1.0)

        # Add points clustered tightly
        for _ in range(10):
            c_real = torch.tensor([0.0, 0.01, 0.02])
            c_imag = torch.tensor([0.0, 0.01, 0.02])
            loss = loss_fn(c_real, c_imag)

        # Should have non-zero loss for small neighborhood
        assert loss.item() > 0.0

    def test_large_neighborhood_no_penalty(self):
        """Test that large neighborhood has no penalty."""
        loss_fn = NeighborhoodPenaltyLoss(window_size=10, min_radius=0.1, weight=1.0)

        # Add points spread out
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
        for real, imag in points:
            c_real = torch.tensor([real])
            c_imag = torch.tensor([imag])
            loss = loss_fn(c_real, c_imag)

        # Should have zero loss for large neighborhood
        assert loss.item() == 0.0

    def test_radius_computation(self):
        """Test neighborhood radius computation."""
        loss_fn = NeighborhoodPenaltyLoss(window_size=10, min_radius=0.1, weight=1.0)

        # Add points with known distribution
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        for real, imag in points:
            c_real = torch.tensor([real])
            c_imag = torch.tensor([imag])
            loss_fn.update_recent(c_real, c_imag)

        radius = loss_fn.compute_neighborhood_radius()

        # Centroid is at (0.5, 0.5), distances are all sqrt(0.5) ≈ 0.707
        assert 0.6 < radius < 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
