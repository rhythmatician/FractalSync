"""
Example: Using VelocityPredictor for smooth real-time inference.

This demonstrates how to use the VelocityPredictor class to smooth
model predictions during real-time inference, creating more natural
transitions between visual parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.velocity_predictor import VelocityPredictor


def example_real_time_smoothing():
    """Example of using velocity predictor for real-time smoothing."""
    
    # Initialize predictor for 7 visual parameters
    predictor = VelocityPredictor(
        num_params=7,
        momentum=0.8,      # Higher = smoother but slower response
        velocity_decay=0.95,  # Decay factor to prevent oscillation
        max_velocity=0.5,  # Clip extreme velocity changes
    )
    
    print("Real-time inference with velocity smoothing")
    print("=" * 50)
    
    # Simulate model predictions over time
    # In practice, these would come from your trained model
    raw_predictions = [
        np.array([0.5, 0.3, 0.7, 0.8, 0.6, 1.0, 0.4]),  # Frame 1
        np.array([0.5, 0.3, 0.7, 0.8, 0.6, 1.0, 0.4]),  # Frame 2 (same)
        np.array([0.8, 0.9, 0.2, 0.3, 0.9, 2.0, 0.8]),  # Frame 3 (big jump)
        np.array([0.8, 0.9, 0.2, 0.3, 0.9, 2.0, 0.8]),  # Frame 4 (same)
        np.array([0.6, 0.5, 0.5, 0.6, 0.7, 1.5, 0.6]),  # Frame 5 (moderate)
    ]
    
    print("\nFrame | Raw Prediction (julia_real) | Smoothed (julia_real)")
    print("-" * 60)
    
    for i, raw_pred in enumerate(raw_predictions, 1):
        # Apply velocity smoothing
        smoothed_pred = predictor.update(raw_pred)
        
        # Print comparison for first parameter (julia_real)
        print(f"  {i}   |      {raw_pred[0]:.3f}           |      {smoothed_pred[0]:.3f}")
    
    print("\nNotice how the smoothed values change gradually,")
    print("even when raw predictions have sudden jumps.")


def example_batch_smoothing():
    """Example of using velocity predictor with batched predictions."""
    
    # Initialize predictor
    predictor = VelocityPredictor(num_params=7, momentum=0.7)
    
    print("\n\nBatch processing with velocity smoothing")
    print("=" * 50)
    
    # Simulate batch predictions (batch_size=4)
    batch_predictions = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ])
    
    # Process batch
    smoothed_batch = predictor.update(batch_predictions)
    
    print(f"\nOriginal batch shape: {batch_predictions.shape}")
    print(f"Smoothed batch shape: {smoothed_batch.shape}")
    print(f"\nBatch processing maintains shape and applies smoothing per-sample")


def example_prediction():
    """Example of predicting future parameters based on velocity."""
    
    # Initialize predictor
    predictor = VelocityPredictor(num_params=7, momentum=0.6)
    
    print("\n\nPredicting future parameters from velocity")
    print("=" * 50)
    
    # Create a sequence with clear trend
    sequence = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ]
    
    # Build up velocity
    for params in sequence:
        predictor.update(params)
    
    # Predict next values
    predicted = predictor.predict_next(steps=1)
    
    print("\nSequence: 0.0 → 0.1 → 0.2")
    print(f"Predicted next (1 step): {predicted[0][0]:.3f}")
    print(f"Actual trend would be: 0.3")
    print("\nNote: Prediction accounts for momentum/smoothing,")
    print("so it may differ from pure linear extrapolation.")


def example_reset():
    """Example of resetting predictor state for new sequences."""
    
    predictor = VelocityPredictor(num_params=7)
    
    print("\n\nResetting predictor between sequences")
    print("=" * 50)
    
    # First sequence
    params1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    smoothed1 = predictor.update(params1)
    print(f"\nSequence 1 - First frame: {smoothed1[0]:.3f}")
    
    params2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    smoothed2 = predictor.update(params2)
    print(f"Sequence 1 - Second frame: {smoothed2[0]:.3f} (smoothed)")
    
    # Reset for new sequence
    predictor.reset()
    print("\n>>> Predictor reset <<<")
    
    # New sequence starts fresh
    params3 = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    smoothed3 = predictor.update(params3)
    print(f"\nSequence 2 - First frame: {smoothed3[0]:.3f} (no smoothing)")
    print("First frame after reset returns input unchanged")


if __name__ == "__main__":
    example_real_time_smoothing()
    example_batch_smoothing()
    example_prediction()
    example_reset()
    
    print("\n" + "=" * 50)
    print("Examples complete!")
    print("=" * 50)
