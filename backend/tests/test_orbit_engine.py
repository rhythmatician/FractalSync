"""Unit tests for orbit engine module."""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orbit_engine import OrbitEngine, create_synthetic_dataset  # noqa: E402


class TestOrbitEngine(unittest.TestCase):
    """Test orbit engine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = OrbitEngine(n_audio_features=6, sr=22050, hop_length=512)

    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.n_audio_features, 6)
        self.assertEqual(self.engine.sr, 22050)
        self.assertEqual(self.engine.hop_length, 512)

    def test_generate_synthetic_trajectory_velocity(self):
        """Test synthetic trajectory generation with velocity correlation."""
        n_samples = 100

        audio_features, visual_params = self.engine.generate_synthetic_trajectory(
            orbit_name="cardioid_boundary",
            n_samples=n_samples,
            audio_correlation="velocity",
        )

        # Check shapes
        self.assertEqual(audio_features.shape, (n_samples, 6))
        self.assertEqual(visual_params.shape, (n_samples, 2))

        # Check value ranges (features should be in [0, 1])
        self.assertTrue(np.all(audio_features >= 0))
        self.assertTrue(np.all(audio_features <= 1))

    def test_generate_synthetic_trajectory_position(self):
        """Test synthetic trajectory generation with position correlation."""
        n_samples = 100

        audio_features, visual_params = self.engine.generate_synthetic_trajectory(
            orbit_name="period2_boundary",
            n_samples=n_samples,
            audio_correlation="position",
        )

        # Check shapes
        self.assertEqual(audio_features.shape, (n_samples, 6))
        self.assertEqual(visual_params.shape, (n_samples, 2))

    def test_generate_synthetic_trajectory_acceleration(self):
        """Test synthetic trajectory generation with acceleration correlation."""
        n_samples = 100

        audio_features, visual_params = self.engine.generate_synthetic_trajectory(
            orbit_name="period3_boundary",
            n_samples=n_samples,
            audio_correlation="acceleration",
        )

        # Check shapes
        self.assertEqual(audio_features.shape, (n_samples, 6))
        self.assertEqual(visual_params.shape, (n_samples, 2))

    def test_invalid_correlation_type(self):
        """Test that invalid correlation type raises error."""
        with self.assertRaises(ValueError):
            self.engine.generate_synthetic_trajectory(
                orbit_name="cardioid_boundary",
                n_samples=100,
                audio_correlation="invalid",
            )

    def test_velocity_correlated_features(self):
        """Test velocity-correlated feature generation."""
        velocities = np.random.randn(100, 2).astype(np.float32)

        features = self.engine._generate_velocity_correlated_features(velocities, 100)

        # Check shape
        self.assertEqual(features.shape, (100, 6))

        # Check range
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))

    def test_position_correlated_features(self):
        """Test position-correlated feature generation."""
        positions = np.random.randn(100, 2).astype(np.float32)

        features = self.engine._generate_position_correlated_features(positions, 100)

        # Check shape
        self.assertEqual(features.shape, (100, 6))

        # Check range
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))

    def test_acceleration_correlated_features(self):
        """Test acceleration-correlated feature generation."""
        accelerations = np.random.randn(100, 2).astype(np.float32)

        features = self.engine._generate_acceleration_correlated_features(
            accelerations, 100
        )

        # Check shape
        self.assertEqual(features.shape, (100, 6))

        # Check range
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))

    def test_compute_accelerations(self):
        """Test acceleration computation from velocities."""
        velocities = np.array(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32
        )

        accelerations = self.engine._compute_accelerations(velocities)

        # Check shape
        self.assertEqual(accelerations.shape, velocities.shape)

        # First three accelerations should be approximately [1, 0]
        for i in range(3):
            self.assertAlmostEqual(accelerations[i, 0], 1.0, places=5)
            self.assertAlmostEqual(accelerations[i, 1], 0.0, places=5)

    def test_generate_mixed_curriculum(self):
        """Test mixed curriculum generation."""
        n_samples = 500

        audio_features, visual_params, metadata = self.engine.generate_mixed_curriculum(
            n_samples=n_samples
        )

        # Check shapes
        self.assertEqual(audio_features.shape[0], n_samples)
        self.assertEqual(audio_features.shape[1], 6)
        self.assertEqual(visual_params.shape, (n_samples, 2))
        self.assertEqual(len(metadata), n_samples)

        # Check metadata structure
        if len(metadata) > 0:
            self.assertIn("orbit", metadata[0])
            self.assertIn("correlation", metadata[0])
            self.assertIn("sample_idx", metadata[0])

    def test_generate_mixed_curriculum_with_specific_orbits(self):
        """Test mixed curriculum with specific orbit names."""
        n_samples = 200
        orbit_names = ["cardioid_boundary", "period2_boundary"]

        audio_features, visual_params, metadata = self.engine.generate_mixed_curriculum(
            n_samples=n_samples, orbit_names=orbit_names
        )

        # Check shapes
        self.assertEqual(audio_features.shape[0], n_samples)

        # Check that only specified orbits are used
        used_orbits = set(m["orbit"] for m in metadata)
        self.assertTrue(used_orbits.issubset(set(orbit_names)))

    def test_generate_windowed_features(self):
        """Test windowed feature generation."""
        # Create simple features
        audio_features = np.random.rand(100, 6).astype(np.float32)
        visual_params = np.random.rand(100, 2).astype(np.float32)
        window_frames = 10

        windowed_audio, windowed_visual = self.engine.generate_windowed_features(
            audio_features, visual_params, window_frames
        )

        # Check shapes
        # Should have (100 - 10 + 1) = 91 windows
        expected_windows = 100 - window_frames + 1
        self.assertEqual(windowed_audio.shape[0], expected_windows)
        self.assertEqual(windowed_audio.shape[1], 6 * window_frames)  # Flattened
        self.assertEqual(windowed_visual.shape, (expected_windows, 2))

    def test_generate_windowed_features_short_audio(self):
        """Test windowed features with audio shorter than window."""
        # Create very short features (5 frames, window is 10)
        audio_features = np.random.rand(5, 6).astype(np.float32)
        visual_params = np.random.rand(5, 2).astype(np.float32)
        window_frames = 10

        windowed_audio, windowed_visual = self.engine.generate_windowed_features(
            audio_features, visual_params, window_frames
        )

        # Should still produce valid output (with padding)
        self.assertGreater(windowed_audio.shape[0], 0)
        self.assertEqual(windowed_audio.shape[1], 6 * window_frames)

    def test_create_synthetic_dataset(self):
        """Test convenience function for dataset creation."""
        n_samples = 1000
        window_frames = 10

        windowed_audio, windowed_visual, metadata = create_synthetic_dataset(
            n_samples=n_samples, window_frames=window_frames, n_audio_features=6
        )

        # Check that we got valid data
        self.assertGreater(windowed_audio.shape[0], 0)
        self.assertEqual(windowed_audio.shape[1], 6 * window_frames)
        self.assertEqual(windowed_visual.shape[1], 2)
        self.assertIsInstance(metadata, list)

    def test_different_audio_feature_counts(self):
        """Test engine with different numbers of audio features."""
        for n_features in [4, 6, 8, 12]:
            engine = OrbitEngine(n_audio_features=n_features)

            audio_features, visual_params = engine.generate_synthetic_trajectory(
                orbit_name="cardioid_boundary",
                n_samples=50,
                audio_correlation="velocity",
            )

            self.assertEqual(audio_features.shape[1], n_features)


if __name__ == "__main__":
    unittest.main()
