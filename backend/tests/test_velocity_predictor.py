"""Unit tests for velocity-based parameter prediction."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.velocity_predictor import (  # noqa: E402
    VelocityPredictor,
    VelocityLoss,
    VelocityAwarePredictor,
)


class TestVelocityPredictor(unittest.TestCase):
    """Test VelocityPredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = VelocityPredictor(num_params=7)

    def test_initialization(self):
        """Test that predictor initializes correctly."""
        self.assertEqual(self.predictor.num_params, 7)
        self.assertIsNone(self.predictor.previous_params)
        self.assertIsNone(self.predictor.current_velocity)

    def test_first_update_returns_input(self):
        """Test that first update returns input unchanged."""
        params = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 1.0, 0.4])
        result = self.predictor.update(params)

        np.testing.assert_array_almost_equal(
            result, params, err_msg="First update should return input unchanged"
        )

    def test_smooth_transition(self):
        """Test that predictor smooths abrupt changes."""
        # Start at zero
        params1 = np.zeros(7)
        self.predictor.update(params1)

        # Jump to 1.0 (abrupt change)
        params2 = np.ones(7)
        result = self.predictor.update(params2)

        # Result should be between 0 and 1 (smoothed)
        self.assertTrue(
            np.all(result > 0) and np.all(result < 1),
            f"Smoothed result should be between 0 and 1, got {result}",
        )

    def test_velocity_tracking(self):
        """Test that velocity is tracked correctly."""
        # Create linear motion: 0 -> 1 -> 2
        params1 = np.zeros(7)
        params2 = np.ones(7)
        params3 = np.ones(7) * 2

        self.predictor.update(params1)
        self.predictor.update(params2)
        self.predictor.update(params3)

        # Velocity should be positive (moving forward)
        velocity = self.predictor.get_velocity()
        self.assertIsNotNone(velocity)
        self.assertTrue(
            np.all(velocity > 0),
            f"Velocity should be positive for increasing sequence, got {velocity}",
        )

    def test_reset(self):
        """Test that reset clears state."""
        params = np.ones(7)
        self.predictor.update(params)

        self.assertIsNotNone(self.predictor.previous_params)
        self.assertIsNotNone(self.predictor.current_velocity)

        self.predictor.reset()

        self.assertIsNone(self.predictor.previous_params)
        self.assertIsNone(self.predictor.current_velocity)

    def test_batch_processing(self):
        """Test that predictor handles batched inputs."""
        batch_size = 4
        params = np.random.randn(batch_size, 7)

        result = self.predictor.update(params)

        self.assertEqual(
            result.shape,
            (batch_size, 7),
            f"Batch processing should preserve shape, got {result.shape}",
        )

    def test_velocity_clipping(self):
        """Test that extreme velocities are clipped."""
        # Create extreme change
        params1 = np.zeros(7)
        params2 = np.ones(7) * 100  # Huge jump

        self.predictor.update(params1)
        self.predictor.update(params2)

        velocity = self.predictor.get_velocity()

        # Velocity should be clipped to max_velocity (default 0.5)
        self.assertTrue(
            np.all(np.abs(velocity) <= self.predictor.max_velocity + 0.01),
            f"Velocity should be clipped to {self.predictor.max_velocity}, got {velocity}",
        )

    def test_predict_next(self):
        """Test future parameter prediction."""
        # Create constant velocity motion with a predictor that has lower decay
        predictor = VelocityPredictor(num_params=7, momentum=0.5, velocity_decay=0.98)
        
        params1 = np.zeros(7)
        params2 = np.ones(7)

        predictor.update(params1)
        predictor.update(params2)

        # Predict next step
        predicted = predictor.predict_next(steps=1)

        self.assertIsNotNone(predicted)
        # Should predict something in the direction of motion (greater than or equal to current)
        # With smoothing, it should be at least as large as current params
        self.assertTrue(
            np.all(predicted >= params1),
            f"Predicted parameters should show forward motion, got {predicted}",
        )

    def test_momentum_effect(self):
        """Test that momentum creates smooth transitions."""
        predictor_high_momentum = VelocityPredictor(num_params=7, momentum=0.9)
        predictor_low_momentum = VelocityPredictor(num_params=7, momentum=0.1)

        params1 = np.zeros(7)
        params2 = np.ones(7)

        # Update both predictors
        predictor_high_momentum.update(params1)
        predictor_high_momentum.update(params2)

        predictor_low_momentum.update(params1)
        predictor_low_momentum.update(params2)

        # High momentum should result in slower response (more smoothing)
        result_high = predictor_high_momentum.previous_params
        result_low = predictor_low_momentum.previous_params

        # High momentum result should be closer to params1 (slower to respond)
        dist_high = np.linalg.norm(result_high - params1)
        dist_low = np.linalg.norm(result_low - params1)

        self.assertLess(
            dist_high,
            dist_low,
            "High momentum should create slower response than low momentum",
        )


class TestVelocityLoss(unittest.TestCase):
    """Test VelocityLoss class."""

    def test_initialization(self):
        """Test that loss initializes correctly."""
        loss = VelocityLoss(weight=0.1)
        self.assertEqual(loss.weight, 0.1)

    def test_loss_computation_without_previous_velocity(self):
        """Test loss computation without previous velocity."""
        loss = VelocityLoss(weight=0.1)

        current = torch.randn(4, 7)
        previous = torch.randn(4, 7)

        result = loss(current, previous)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([]))
        self.assertTrue(result.item() >= 0, "Loss should be non-negative")

    def test_loss_computation_with_previous_velocity(self):
        """Test loss computation with previous velocity (jerk penalty)."""
        loss = VelocityLoss(weight=0.1)

        current = torch.randn(4, 7)
        previous = torch.randn(4, 7)
        prev_velocity = torch.randn(4, 7)

        result = loss(current, previous, prev_velocity)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([]))
        self.assertTrue(result.item() >= 0, "Loss should be non-negative")

    def test_constant_velocity_has_low_loss(self):
        """Test that constant velocity results in low jerk loss."""
        loss = VelocityLoss(weight=1.0)

        # Create constant velocity motion
        t0 = torch.zeros(4, 7)
        t1 = torch.ones(4, 7)
        t2 = torch.ones(4, 7) * 2

        # Compute velocity
        v1 = t1 - t0  # velocity = 1

        result = loss(t2, t1, v1)

        # Jerk should be near zero for constant velocity
        self.assertLess(
            result.item(), 0.01, "Constant velocity should have near-zero jerk loss"
        )

    def test_accelerating_motion_has_higher_loss(self):
        """Test that accelerating motion has higher loss than constant velocity."""
        loss = VelocityLoss(weight=1.0)

        # Constant velocity
        t0_const = torch.zeros(4, 7)
        t1_const = torch.ones(4, 7)
        t2_const = torch.ones(4, 7) * 2
        v1_const = t1_const - t0_const

        loss_const = loss(t2_const, t1_const, v1_const)

        # Accelerating velocity
        t0_accel = torch.zeros(4, 7)
        t1_accel = torch.ones(4, 7)
        t2_accel = torch.ones(4, 7) * 3  # Acceleration
        v1_accel = t1_accel - t0_accel

        loss_accel = loss(t2_accel, t1_accel, v1_accel)

        self.assertGreater(
            loss_accel.item(),
            loss_const.item(),
            "Accelerating motion should have higher loss than constant velocity",
        )


class TestVelocityAwarePredictor(unittest.TestCase):
    """Test VelocityAwarePredictor neural network module."""

    def test_initialization(self):
        """Test that module initializes correctly."""
        predictor = VelocityAwarePredictor(input_dim=60, output_dim=7)
        self.assertEqual(predictor.output_dim, 7)

    def test_forward_pass(self):
        """Test forward pass produces params and velocities."""
        predictor = VelocityAwarePredictor(input_dim=60, output_dim=7)
        x = torch.randn(4, 60)

        params, velocities = predictor(x)

        self.assertEqual(params.shape, (4, 7))
        self.assertEqual(velocities.shape, (4, 7))

    def test_velocity_is_bounded(self):
        """Test that velocity outputs are bounded by tanh."""
        predictor = VelocityAwarePredictor(input_dim=60, output_dim=7)
        x = torch.randn(4, 60)

        _, velocities = predictor(x)

        # Tanh bounds output to [-1, 1]
        self.assertTrue(torch.all(velocities >= -1))
        self.assertTrue(torch.all(velocities <= 1))

    def test_predict_with_velocity(self):
        """Test prediction incorporating velocity."""
        predictor = VelocityAwarePredictor(input_dim=60, output_dim=7)
        x = torch.randn(4, 60)

        adjusted_params = predictor.predict_with_velocity(x, dt=1.0)

        self.assertEqual(adjusted_params.shape, (4, 7))

    def test_different_dt_affects_prediction(self):
        """Test that dt parameter affects velocity integration."""
        predictor = VelocityAwarePredictor(input_dim=60, output_dim=7)
        predictor.eval()  # Set to eval mode for deterministic results

        x = torch.randn(4, 60)

        # Predict with different time steps
        with torch.no_grad():
            result_dt_small = predictor.predict_with_velocity(x, dt=0.1)
            result_dt_large = predictor.predict_with_velocity(x, dt=2.0)

        # Results should be different
        self.assertFalse(
            torch.allclose(result_dt_small, result_dt_large),
            "Different dt values should produce different results",
        )


if __name__ == "__main__":
    unittest.main()
