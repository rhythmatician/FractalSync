"""Test suite for AudioDrivenMomentumLoss."""

import sys
from pathlib import Path

import torch
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics_trainer import AudioDrivenMomentumLoss


class TestAudioDrivenMomentumLoss:
    """Tests for audio-driven momentum loss."""

    def test_initialization(self):
        """Test loss can be instantiated."""
        loss = AudioDrivenMomentumLoss(weight=0.1)
        assert loss.weight == 0.1

    def test_zero_loss_when_aligned(self):
        """Test loss is low when velocity matches audio energy."""
        loss_fn = AudioDrivenMomentumLoss(weight=1.0)

        # Same pattern should normalize identically
        velocity_mag = torch.tensor([0.1, 0.2, 0.3])
        audio_energy = torch.tensor([0.1, 0.2, 0.3])

        loss = loss_fn(velocity_mag, audio_energy)
        assert loss.item() < 0.01

    def test_high_loss_when_misaligned(self):
        """Test loss is high when velocity opposes audio energy."""
        loss_fn = AudioDrivenMomentumLoss(weight=1.0)

        # Opposite patterns should give high loss
        velocity_mag = torch.tensor([0.1, 0.5, 0.9])
        audio_energy = torch.tensor([0.9, 0.5, 0.1])

        loss = loss_fn(velocity_mag, audio_energy)
        # After normalization: vel [0, 0.5, 1], energy [1, 0.5, 0]
        # Mismatches: |0-1| + |0.5-0.5| + |1-0| = 2.0, mean = 0.667
        assert loss.item() > 0.5

    def test_silent_pattern_match(self):
        """Test that matching silent patterns give low loss."""
        loss_fn = AudioDrivenMomentumLoss(weight=1.0)

        # Silent: low velocity matches low energy pattern
        velocity_mag = torch.tensor([0.01, 0.01, 0.02])
        audio_energy = torch.tensor([0.01, 0.01, 0.02])

        loss = loss_fn(velocity_mag, audio_energy)
        # Same pattern should normalize identically, giving ~0 loss
        assert loss.item() < 0.01

    def test_loud_pattern_match(self):
        """Test that loud patterns with similar structure give low loss."""
        loss_fn = AudioDrivenMomentumLoss(weight=1.0)

        # Loud: high velocity matches high energy with same structure
        velocity_mag = torch.tensor([0.7, 0.8, 0.9])
        audio_energy = torch.tensor([0.7, 0.8, 0.9])

        loss = loss_fn(velocity_mag, audio_energy)
        assert loss.item() < 0.01

    def test_batch_processing(self):
        """Test loss works with batches of different sizes."""
        loss_fn = AudioDrivenMomentumLoss(weight=0.5)

        for batch_size in [1, 4, 16, 32]:
            velocity_mag = torch.rand(batch_size)
            audio_energy = torch.rand(batch_size)

            loss = loss_fn(velocity_mag, audio_energy)
            assert loss.shape == torch.Size([])
            assert loss.item() >= 0

    def test_weight_scaling(self):
        """Test that weight parameter scales loss correctly."""
        velocity_mag = torch.tensor([0.1, 0.9])
        audio_energy = torch.tensor([0.9, 0.1])

        loss_w1 = AudioDrivenMomentumLoss(weight=1.0)(velocity_mag, audio_energy)
        loss_w2 = AudioDrivenMomentumLoss(weight=2.0)(velocity_mag, audio_energy)

        assert abs(loss_w2.item() - 2 * loss_w1.item()) < 1e-5

    def test_gradient_flow(self):
        """Test gradients flow through the loss."""
        loss_fn = AudioDrivenMomentumLoss(weight=1.0)

        velocity_mag = torch.tensor(
            [0.1, 0.5, 0.9], requires_grad=True, dtype=torch.float32
        )
        audio_energy = torch.tensor([0.2, 0.6, 0.8], dtype=torch.float32)

        loss = loss_fn(velocity_mag, audio_energy)
        loss.backward()

        assert velocity_mag.grad is not None
        # Loss has mismatch (not 0), so gradients should flow
        assert loss.item() > 0

    def test_symmetric_processing(self):
        """Test loss treats velocity and audio energy symmetrically in normalization."""
        loss_fn = AudioDrivenMomentumLoss(weight=1.0)

        # If we scale both proportionally, loss shouldn't change much
        velocity_mag = torch.tensor([0.2, 0.5, 0.8])
        audio_energy = torch.tensor([0.3, 0.6, 0.9])

        loss1 = loss_fn(velocity_mag, audio_energy)

        # Scale both by 2
        velocity_mag_scaled = velocity_mag * 2
        audio_energy_scaled = audio_energy * 2

        loss2 = loss_fn(velocity_mag_scaled, audio_energy_scaled)

        # Normalized values should be identical
        assert abs(loss1.item() - loss2.item()) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
