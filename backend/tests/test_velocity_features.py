"""Unit tests for velocity-based audio features."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_features import AudioFeatureExtractor  # noqa: E402
from src.model import AudioToVisualModel  # noqa: E402


class TestVelocityFeatures(unittest.TestCase):
    """Test velocity (delta) feature extraction."""

    def test_base_features_only(self):
        """Test extraction with only base features (no deltas)."""
        extractor = AudioFeatureExtractor(
            sr=22050, include_delta=False, include_delta_delta=False
        )

        # Verify feature count
        self.assertEqual(extractor.get_num_features(), 6)

        # Extract features from synthetic audio
        audio = np.random.randn(22050).astype(np.float32)
        features = extractor.extract_features(audio)

        # Should have 6 features
        self.assertEqual(features.shape[0], 6)

    def test_delta_features(self):
        """Test extraction with delta (velocity) features."""
        extractor = AudioFeatureExtractor(
            sr=22050, include_delta=True, include_delta_delta=False
        )

        # Verify feature count: 6 base + 6 delta = 12
        self.assertEqual(extractor.get_num_features(), 12)

        # Extract features from synthetic audio
        audio = np.random.randn(22050).astype(np.float32)
        features = extractor.extract_features(audio)

        # Should have 12 features (6 base + 6 delta)
        self.assertEqual(features.shape[0], 12)

        # First 6 should be base features, next 6 should be deltas
        base_features = features[:6, :]
        delta_features = features[6:12, :]

        # Delta features should have same number of frames
        self.assertEqual(base_features.shape[1], delta_features.shape[1])

    def test_delta_delta_features(self):
        """Test extraction with delta and delta-delta features."""
        extractor = AudioFeatureExtractor(
            sr=22050, include_delta=True, include_delta_delta=True
        )

        # Verify feature count: 6 base + 6 delta + 6 delta-delta = 18
        self.assertEqual(extractor.get_num_features(), 18)

        # Extract features from synthetic audio
        audio = np.random.randn(22050).astype(np.float32)
        features = extractor.extract_features(audio)

        # Should have 18 features
        self.assertEqual(features.shape[0], 18)

    def test_windowed_features_with_delta(self):
        """Test windowed feature extraction with delta features."""
        extractor = AudioFeatureExtractor(sr=22050, include_delta=True)

        # Extract windowed features
        audio = np.random.randn(110250).astype(np.float32)  # 5 seconds
        window_frames = 10

        windowed = extractor.extract_windowed_features(audio, window_frames=window_frames)

        # Each window should have 12 features * 10 frames = 120 dimensions
        expected_dim = 12 * window_frames
        self.assertEqual(
            windowed.shape[1],
            expected_dim,
            f"Expected {expected_dim}-dim features with delta, got {windowed.shape[1]}",
        )

    def test_windowed_features_with_all_deltas(self):
        """Test windowed feature extraction with all delta features."""
        extractor = AudioFeatureExtractor(
            sr=22050, include_delta=True, include_delta_delta=True
        )

        # Extract windowed features
        audio = np.random.randn(110250).astype(np.float32)  # 5 seconds
        window_frames = 10

        windowed = extractor.extract_windowed_features(audio, window_frames=window_frames)

        # Each window should have 18 features * 10 frames = 180 dimensions
        expected_dim = 18 * window_frames
        self.assertEqual(
            windowed.shape[1],
            expected_dim,
            f"Expected {expected_dim}-dim features with all deltas, got {windowed.shape[1]}",
        )


class TestModelWithVelocityFeatures(unittest.TestCase):
    """Test model compatibility with velocity features."""

    def test_model_with_base_features(self):
        """Test model with base features only."""
        model = AudioToVisualModel(window_frames=10, num_features_per_frame=6)

        # Input should be 6 * 10 = 60
        self.assertEqual(model.input_dim, 60)

        # Test forward pass
        batch_size = 4
        input_tensor = torch.randn(batch_size, 60)

        output = model(input_tensor)

        # Output should be (batch_size, 7)
        self.assertEqual(output.shape, (batch_size, 7))

    def test_model_with_delta_features(self):
        """Test model with delta features."""
        model = AudioToVisualModel(window_frames=10, num_features_per_frame=12)

        # Input should be 12 * 10 = 120
        self.assertEqual(model.input_dim, 120)

        # Test forward pass
        batch_size = 4
        input_tensor = torch.randn(batch_size, 120)

        output = model(input_tensor)

        # Output should be (batch_size, 7)
        self.assertEqual(output.shape, (batch_size, 7))

    def test_model_with_all_deltas(self):
        """Test model with delta and delta-delta features."""
        model = AudioToVisualModel(window_frames=10, num_features_per_frame=18)

        # Input should be 18 * 10 = 180
        self.assertEqual(model.input_dim, 180)

        # Test forward pass
        batch_size = 4
        input_tensor = torch.randn(batch_size, 180)

        output = model(input_tensor)

        # Output should be (batch_size, 7)
        self.assertEqual(output.shape, (batch_size, 7))

    def test_model_rejects_wrong_input_dim(self):
        """Test that model validates input dimension."""
        model = AudioToVisualModel(window_frames=10, num_features_per_frame=12)

        # Try with wrong input dimension (60 instead of 120)
        batch_size = 4
        wrong_input = torch.randn(batch_size, 60)

        with self.assertRaises(ValueError) as context:
            model(wrong_input)

        self.assertIn("Expected input dim 120", str(context.exception))


class TestEndToEndWithVelocity(unittest.TestCase):
    """End-to-end tests with velocity features."""

    def test_feature_extraction_to_model(self):
        """Test complete pipeline from audio to model output."""
        # Create extractor with delta features
        extractor = AudioFeatureExtractor(sr=22050, include_delta=True)

        # Generate synthetic audio
        audio = np.random.randn(110250).astype(np.float32)  # 5 seconds

        # Extract windowed features
        window_frames = 10
        features = extractor.extract_windowed_features(audio, window_frames=window_frames)

        # Create model with matching dimensions
        num_features_per_frame = extractor.get_num_features()
        model = AudioToVisualModel(
            window_frames=window_frames,
            num_features_per_frame=num_features_per_frame,
        )

        # Convert to tensor and run through model
        features_tensor = torch.from_numpy(features).float()
        
        # Test with a single batch
        batch = features_tensor[:4]  # Take first 4 samples
        
        output = model(batch)

        # Verify output shape
        self.assertEqual(output.shape, (4, 7))

        # Verify output is in valid ranges
        # Julia real/imag: [-2, 2]
        self.assertTrue(torch.all(output[:, 0] >= -2) and torch.all(output[:, 0] <= 2))
        self.assertTrue(torch.all(output[:, 1] >= -2) and torch.all(output[:, 1] <= 2))

        # Color hue/sat/bright: [0, 1]
        self.assertTrue(torch.all(output[:, 2] >= 0) and torch.all(output[:, 2] <= 1))
        self.assertTrue(torch.all(output[:, 3] >= 0) and torch.all(output[:, 3] <= 1))
        self.assertTrue(torch.all(output[:, 4] >= 0) and torch.all(output[:, 4] <= 1))

        # Zoom: positive
        self.assertTrue(torch.all(output[:, 5] > 0))

        # Speed: [0, 1]
        self.assertTrue(torch.all(output[:, 6] >= 0) and torch.all(output[:, 6] <= 1))


if __name__ == "__main__":
    unittest.main()
