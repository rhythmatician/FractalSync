"""Test boundary proximity and directional consistency losses."""

import torch
from src.physics_trainer import BoundaryProximityLoss, DirectionalConsistencyLoss


class TestBoundaryProximityLoss:
    """Test Mandelbrot boundary proximity loss."""

    def test_boundary_loss_creation(self):
        """Test creating the loss function."""
        loss_fn = BoundaryProximityLoss(weight=0.2, target_iters=30, max_iters=100)
        assert loss_fn.weight == 0.2
        assert loss_fn.target_iters == 30
        assert loss_fn.max_iters == 100

    def test_mandelbrot_escape_time_inside_set(self):
        """Test escape time for points inside Mandelbrot set."""
        loss_fn = BoundaryProximityLoss(max_iters=100)

        # c = 0 is deep inside the set (should take max_iters)
        c_real = torch.tensor([0.0])
        c_imag = torch.tensor([0.0])
        escape_time = loss_fn.mandelbrot_escape_time(c_real, c_imag)

        assert escape_time[0] == 100  # Never escapes

    def test_mandelbrot_escape_time_outside_set(self):
        """Test escape time for points outside Mandelbrot set."""
        loss_fn = BoundaryProximityLoss(max_iters=100)

        # c = 2 + 0i is far outside the set (should escape quickly)
        c_real = torch.tensor([2.0])
        c_imag = torch.tensor([0.0])
        escape_time = loss_fn.mandelbrot_escape_time(c_real, c_imag)

        assert escape_time[0] < 10  # Escapes quickly

    def test_boundary_proximity_batch(self):
        """Test boundary proximity loss with batch input."""
        loss_fn = BoundaryProximityLoss(weight=0.2, target_iters=30)

        # Mix of inside, boundary, and outside points
        c_real = torch.tensor([0.0, -0.5, 2.0, -1.0])
        c_imag = torch.tensor([0.0, 0.5, 0.0, 0.0])

        loss = loss_fn(c_real, c_imag)

        # Loss should be a scalar tensor
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_boundary_proximity_at_target(self):
        """Test that loss increases for points far from target iteration count."""
        loss_fn = BoundaryProximityLoss(weight=1.0, target_iters=30, max_iters=100)

        # Test that loss is computed (point deep in set has high loss)
        c_real = torch.tensor([0.0])
        c_imag = torch.tensor([0.0])
        loss_deep = loss_fn(c_real, c_imag)

        # Test that loss is computed (point far outside has high loss)
        c_real = torch.tensor([2.0])
        c_imag = torch.tensor([0.0])
        loss_outside = loss_fn(c_real, c_imag)

        # Both should have positive loss (not at target)
        assert loss_deep.item() > 0.5
        assert loss_outside.item() > 0.5


class TestDirectionalConsistencyLoss:
    """Test directional consistency loss."""

    def test_directional_loss_creation(self):
        """Test creating the loss function."""
        loss_fn = DirectionalConsistencyLoss(weight=0.15)
        assert loss_fn.weight == 0.15

    def test_same_direction_zero_loss(self):
        """Test that same direction gives zero loss."""
        loss_fn = DirectionalConsistencyLoss(weight=1.0)

        # Same velocity direction
        current = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        previous = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

        loss = loss_fn(current, previous)

        # Should be near zero (same direction)
        assert loss.item() < 0.01

    def test_opposite_direction_high_loss(self):
        """Test that opposite direction gives high loss."""
        loss_fn = DirectionalConsistencyLoss(weight=1.0)

        # Opposite velocity direction (180 degree flip)
        current = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        previous = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])

        loss = loss_fn(current, previous)

        # Should be high (opposite direction)
        assert loss.item() > 0.9  # Near 1.0 for perfect flip

    def test_perpendicular_direction_zero_loss(self):
        """Test that perpendicular motion gives zero loss."""
        loss_fn = DirectionalConsistencyLoss(weight=1.0)

        # Perpendicular velocity direction (90 degrees)
        current = torch.tensor([[1.0, 0.0]])
        previous = torch.tensor([[0.0, 1.0]])

        loss = loss_fn(current, previous)

        # Should be zero (dot product = 0, no flip penalty)
        assert loss.item() < 0.01

    def test_directional_consistency_batch(self):
        """Test directional consistency with batch."""
        loss_fn = DirectionalConsistencyLoss(weight=0.15)

        # Mix of directions
        current = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        previous = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        loss = loss_fn(current, previous)

        # Loss should be scalar
        assert loss.shape == ()
        assert loss.item() >= 0.0
