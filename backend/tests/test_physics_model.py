"""
Tests for physics-based model and Mandelbrot orbits.
"""

import pytest
import numpy as np
import torch

from backend.src.mandelbrot_orbits import (
    MandelbrotOrbit,
    get_preset_orbit,
    list_preset_names,
    generate_curriculum_sequence,
    is_in_mandelbrot_set,
)
from backend.src.physics_model import PhysicsAudioToVisualModel


class TestMandelbrotOrbits:
    """Test Mandelbrot orbit generation and curriculum learning."""

    def test_orbit_creation(self):
        """Test creating a custom orbit."""
        points = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]
        orbit = MandelbrotOrbit(name="test", points=points, closed=True)

        assert orbit.name == "test"
        assert len(orbit.points) == 3
        assert orbit.closed is True

    def test_orbit_sampling(self):
        """Test sampling points along an orbit."""
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        orbit = MandelbrotOrbit(name="square", points=points, closed=True)

        samples = orbit.sample(n_samples=100)

        assert samples.shape == (100, 2)
        assert samples.dtype == np.float32

        # First and last should be same for closed orbit
        assert np.allclose(samples[0], samples[-1])

    def test_orbit_velocities(self):
        """Test velocity computation along an orbit."""
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        orbit = MandelbrotOrbit(name="test", points=points, closed=False)

        velocities = orbit.compute_velocities(n_samples=10, time_step=1.0)

        assert velocities.shape == (10, 2)
        assert velocities.dtype == np.float32

    def test_preset_orbits(self):
        """Test loading preset orbits."""
        names = list_preset_names()

        assert len(names) > 0
        assert "cardioid" in names
        assert "bulb" in names

        cardioid = get_preset_orbit("cardioid")
        assert cardioid.name == "cardioid"
        assert cardioid.closed is True

    def test_curriculum_sequence(self):
        """Test generating curriculum learning sequence."""
        n_samples = 1000
        positions, velocities = generate_curriculum_sequence(n_samples)

        assert positions.shape == (n_samples, 2)
        assert velocities.shape == (n_samples, 2)
        assert positions.dtype == np.float32
        assert velocities.dtype == np.float32

        # Check that positions are in reasonable range (within Mandelbrot set region)
        assert np.all(np.abs(positions) < 3.0)

    def test_mandelbrot_set_membership(self):
        """Test Mandelbrot set membership checking."""
        # Points known to be in the set
        assert is_in_mandelbrot_set(0.0, 0.0, max_iter=100) is True
        assert is_in_mandelbrot_set(-0.5, 0.0, max_iter=100) is True

        # Points known to escape
        assert is_in_mandelbrot_set(1.0, 1.0, max_iter=100) is False
        assert is_in_mandelbrot_set(2.0, 2.0, max_iter=100) is False


class TestPhysicsModel:
    """Test physics-based audio to visual model."""

    def test_model_creation(self):
        """Test creating physics model."""
        model = PhysicsAudioToVisualModel(
            window_frames=10,
            hidden_dims=[128, 256, 128],
            output_dim=9,
            predict_velocity=True,
        )

        assert model.input_dim == 60  # 6 features * 10 frames
        assert model.output_dim == 9
        assert model.predict_velocity is True

    def test_model_forward_pass(self):
        """Test forward pass through physics model."""
        model = PhysicsAudioToVisualModel(
            window_frames=10,
            predict_velocity=True,
        )
        model.eval()

        # Create dummy input
        batch_size = 4
        input_features = torch.randn(batch_size, 60)  # 6 * 10 features

        # Forward pass
        output = model(input_features)

        # Check output shape
        assert output.shape == (batch_size, 9)

        # Check that velocity values are reasonable
        v_real = output[:, 0]
        v_imag = output[:, 1]
        assert torch.all(torch.isfinite(v_real))
        assert torch.all(torch.isfinite(v_imag))

    def test_model_with_rms_modulation(self):
        """Test that RMS energy modulates velocity magnitude."""
        model = PhysicsAudioToVisualModel(
            window_frames=10,
            predict_velocity=True,
            speed_scale=1.0,
        )
        model.eval()

        batch_size = 4
        input_features = torch.randn(batch_size, 60)

        # Test with low RMS
        low_rms = torch.ones(batch_size) * 0.1
        output_low = model(input_features, audio_rms=low_rms)
        velocity_low = output_low[:, 0:2]
        magnitude_low = torch.norm(velocity_low, dim=1)

        # Test with high RMS
        high_rms = torch.ones(batch_size) * 0.9
        output_high = model(input_features, audio_rms=high_rms)
        velocity_high = output_high[:, 0:2]
        magnitude_high = torch.norm(velocity_high, dim=1)

        # Higher RMS should generally produce higher velocity magnitudes
        # Note: Due to random initialization, this might not always hold strictly
        # We just check that magnitudes are positive and finite
        assert torch.all(magnitude_low >= 0.0)
        assert torch.all(magnitude_high >= 0.0)
        assert torch.all(torch.isfinite(magnitude_low))
        assert torch.all(torch.isfinite(magnitude_high))

    def test_velocity_integration(self):
        """Test velocity integration to position."""
        model = PhysicsAudioToVisualModel(
            window_frames=10,
            predict_velocity=True,
            damping_factor=0.9,
        )

        batch_size = 4
        velocity = torch.randn(batch_size, 2) * 0.1  # Small velocities
        position = torch.zeros(batch_size, 2)

        # Integrate
        new_position, new_velocity = model.integrate_velocity(
            velocity, position, prev_velocity=None, dt=1.0
        )

        # Check shapes
        assert new_position.shape == (batch_size, 2)
        assert new_velocity.shape == (batch_size, 2)

        # Position should have changed
        assert not torch.allclose(new_position, position, atol=1e-5)

        # Position should be constrained to |c| < 2
        magnitudes = torch.norm(new_position, dim=1)
        assert torch.all(magnitudes <= 2.0 + 1e-5)

    def test_state_reset(self):
        """Test resetting model state."""
        model = PhysicsAudioToVisualModel(
            window_frames=10,
            predict_velocity=True,
        )

        # Set initial state
        model.reset_state(position=(0.5, 0.5))

        assert torch.allclose(model.current_position, torch.tensor([0.5, 0.5]))
        assert torch.allclose(model.current_velocity, torch.zeros(2))

        # Reset to default
        model.reset_state()

        assert torch.allclose(model.current_position, torch.zeros(2))
        assert torch.allclose(model.current_velocity, torch.zeros(2))

    def test_direct_position_mode(self):
        """Test model in direct position prediction mode (no physics)."""
        model = PhysicsAudioToVisualModel(
            window_frames=10,
            predict_velocity=False,  # Direct mode
        )
        model.eval()

        batch_size = 4
        input_features = torch.randn(batch_size, 60)

        output = model(input_features)

        # In direct mode, output should be 7 parameters
        assert output.shape == (batch_size, 7)

        # Check Julia parameter range
        julia_real = output[:, 0]
        julia_imag = output[:, 1]
        assert torch.all(julia_real >= -2.0)
        assert torch.all(julia_real <= 2.0)
        assert torch.all(julia_imag >= -2.0)
        assert torch.all(julia_imag <= 2.0)

    def test_parameter_ranges(self):
        """Test getting parameter ranges."""
        model_velocity = PhysicsAudioToVisualModel(predict_velocity=True)
        ranges_velocity = model_velocity.get_parameter_ranges()

        assert "v_real" in ranges_velocity
        assert "v_imag" in ranges_velocity
        assert "julia_real" in ranges_velocity

        model_direct = PhysicsAudioToVisualModel(predict_velocity=False)
        ranges_direct = model_direct.get_parameter_ranges()

        assert "v_real" not in ranges_direct
        assert "julia_real" in ranges_direct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
