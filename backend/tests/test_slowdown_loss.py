"""Unit tests for TrajectorySlowdownLoss."""

import torch
from src.control_trainer import TrajectorySlowdownLoss


def test_slowdown_loss_empty_inputs():
    """Test slowdown loss with empty inputs."""
    loss_fn = TrajectorySlowdownLoss(weight=1.0, threshold=0.02)

    # Empty inputs
    result = loss_fn([], [])
    assert torch.isfinite(result), f"Expected finite, got {result}"
    assert result.item() == 0.0, f"Expected 0.0, got {result.item()}"


def test_slowdown_loss_single_trajectory():
    """Test slowdown loss with a single trajectory."""
    loss_fn = TrajectorySlowdownLoss(weight=1.0, threshold=0.02)

    # Single trajectory with 3 steps
    velocities = [[0.1, 0.2]]
    distances = [[0.5, 0.8]]

    result = loss_fn(velocities, distances)
    assert torch.isfinite(result), f"Expected finite, got {result}"
    print(f"Single trajectory result: {result.item()}")


def test_slowdown_loss_with_nan_values():
    """Test slowdown loss handles NaN gracefully."""
    loss_fn = TrajectorySlowdownLoss(weight=1.0, threshold=0.02)

    # Include NaN in velocities
    velocities = [[0.1, float("nan"), 0.3]]
    distances = [[0.5, 0.8, 0.3]]

    result = loss_fn(velocities, distances)
    assert torch.isfinite(result), f"Expected finite, got {result}"
    print(f"With NaN velocities result: {result.item()}")


def test_slowdown_loss_with_inf_values():
    """Test slowdown loss handles Inf gracefully."""
    loss_fn = TrajectorySlowdownLoss(weight=1.0, threshold=0.02)

    # Include Inf in distances
    velocities = [[0.1, 0.2, 0.3]]
    distances = [[0.5, float("inf"), 0.3]]

    result = loss_fn(velocities, distances)
    assert torch.isfinite(result), f"Expected finite, got {result}"
    print(f"With Inf distances result: {result.item()}")


def test_slowdown_loss_multiple_trajectories():
    """Test slowdown loss with multiple trajectories."""
    loss_fn = TrajectorySlowdownLoss(weight=1.0, threshold=0.02)

    # Multiple trajectories
    velocities = [
        [0.1, 0.2, 0.15],
        [0.05, 0.3],
        [0.2, 0.2, 0.2],
    ]
    distances = [
        [0.5, 0.8, 0.3],
        [0.9, 0.1],
        [0.1, 0.05, 0.02],
    ]

    result = loss_fn(velocities, distances)
    assert torch.isfinite(result), f"Expected finite, got {result}"
    print(f"Multiple trajectories result: {result.item()}")


def test_slowdown_loss_high_velocity_near_boundary():
    """Test that high velocity near boundary (small distance) increases loss."""
    loss_fn = TrajectorySlowdownLoss(weight=1.0, threshold=0.02)

    # High velocity near boundary (distance=0.01, velocity=0.5)
    velocities_near = [[0.5]]
    distances_near = [[0.01]]

    # Same velocity far from boundary (distance=0.9, velocity=0.5)
    velocities_far = [[0.5]]
    distances_far = [[0.9]]

    result_near = loss_fn(velocities_near, distances_near)
    result_far = loss_fn(velocities_far, distances_far)

    print(f"Near boundary (d=0.01, v=0.5): {result_near.item()}")
    print(f"Far from boundary (d=0.9, v=0.5): {result_far.item()}")

    # Near boundary should have higher loss
    assert (
        result_near.item() > result_far.item()
    ), f"Near boundary ({result_near.item()}) should be > far ({result_far.item()})"


def test_slowdown_loss_zero_velocity():
    """Test slowdown loss with zero velocity."""
    loss_fn = TrajectorySlowdownLoss(weight=1.0, threshold=0.02)

    velocities = [[0.0, 0.0, 0.0]]
    distances = [[0.1, 0.5, 0.9]]

    result = loss_fn(velocities, distances)
    assert torch.isfinite(result), f"Expected finite, got {result}"
    assert result.item() == 0.0, f"Expected 0.0 for zero velocity, got {result.item()}"
    print(f"Zero velocity result: {result.item()}")


if __name__ == "__main__":
    print("Running slowdown loss tests...\n")

    test_slowdown_loss_empty_inputs()
    print("✓ Empty inputs\n")

    test_slowdown_loss_single_trajectory()
    print("✓ Single trajectory\n")

    test_slowdown_loss_with_nan_values()
    print("✓ NaN handling\n")

    test_slowdown_loss_with_inf_values()
    print("✓ Inf handling\n")

    test_slowdown_loss_multiple_trajectories()
    print("✓ Multiple trajectories\n")

    test_slowdown_loss_high_velocity_near_boundary()
    print("✓ High velocity near boundary\n")

    test_slowdown_loss_zero_velocity()
    print("✓ Zero velocity\n")

    print("All tests passed!")
