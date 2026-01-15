"""Unit tests for audio feature extraction and batching pipeline."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_features import AudioFeatureExtractor  # noqa: E402


class TestAudioFeatureExtraction(unittest.TestCase):
    """Test audio feature extraction and windowing."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = AudioFeatureExtractor(sr=22050, hop_length=512, n_fft=2048)

    def test_extract_windowed_features_dimensions(self):
        """Test that windowed features have correct dimensions."""
        # Create a mock audio array of various lengths
        test_cases = [
            (22050, "1 second"),  # 1 second at 22050 Hz
            (110250, "5 seconds"),  # 5 seconds
            (220500, "10 seconds"),  # 10 seconds
        ]

        for audio_len, description in test_cases:
            with self.subTest(length=description):
                # Create synthetic audio
                audio = np.random.randn(audio_len).astype(np.float32)

                # Extract windowed features
                features = self.extractor.extract_windowed_features(
                    audio, window_frames=10
                )

                # Verify shape: each sample should be 6 features * 10 frames = 60 dims
                self.assertEqual(
                    features.shape[1],
                    60,
                    f"Expected 60-dim features for {description}, got {features.shape[1]}",
                )

                # Should have at least one window
                self.assertGreater(
                    features.shape[0],
                    0,
                    f"Should have at least one feature window for {description}",
                )

    def test_extract_windowed_features_short_audio(self):
        """Test that short audio (< window_frames) still produces correct dimensions."""
        # Very short audio: 0.5 seconds
        audio = np.random.randn(11025).astype(np.float32)

        features = self.extractor.extract_windowed_features(audio, window_frames=10)

        # Should still produce 60-dim output
        self.assertEqual(
            features.shape[1],
            60,
            f"Short audio should still produce 60-dim features, got {features.shape[1]}",
        )

        # Should have at least one window
        self.assertGreaterEqual(features.shape[0], 1)

    def test_extract_features_basic(self):
        """Test basic feature extraction without windowing."""
        audio = np.random.randn(22050).astype(np.float32)

        features = self.extractor.extract_features(audio)

        # Should extract 6 features
        self.assertEqual(features.shape[0], 6, "Should extract exactly 6 features")

        # Should have frames proportional to audio length
        self.assertGreater(
            features.shape[1],
            0,
            "Should have at least one frame of features",
        )

    def test_different_window_sizes(self):
        """Test that different window_frames values work correctly."""
        audio = np.random.randn(110250).astype(np.float32)  # 5 seconds

        for window_frames in [5, 10, 15, 20]:
            with self.subTest(window_frames=window_frames):
                features = self.extractor.extract_windowed_features(
                    audio, window_frames=window_frames
                )

                expected_dim = 6 * window_frames
                self.assertEqual(
                    features.shape[1],
                    expected_dim,
                    f"With window_frames={window_frames}, expected {expected_dim}-dim features",
                )


class TestBatchingPipeline(unittest.TestCase):
    """Test batching and DataLoader compatibility."""

    def test_dataloader_batch_dimensions(self):
        """Test that DataLoader batches maintain correct dimensions."""
        from torch.utils.data import DataLoader

        # Create synthetic features: 330 samples of 60-dim vectors
        features = np.random.randn(330, 60).astype(np.float32)

        # Create a simple tensor dataset
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(features))

        # Create dataloader with batch size 32
        dataloader = DataLoader(dataset, batch_size=32)

        # Check batches
        for batch_idx, batch in enumerate(dataloader):
            batch_tensor = batch[0]  # DataLoader wraps in tuple

            # Each batch should have shape (batch_size, 60) or less for final batch
            self.assertEqual(
                batch_tensor.shape[1],
                60,
                f"Batch {batch_idx} has wrong feature dimension: {batch_tensor.shape}",
            )

            # Batch size should be <= 32
            self.assertLessEqual(
                batch_tensor.shape[0],
                32,
                f"Batch {batch_idx} exceeds batch size: {batch_tensor.shape[0]}",
            )

    def test_features_stay_60d_through_processing(self):
        """Test that features remain 60-dimensional through entire pipeline."""
        # Simulate feature extraction from multi-minute audio
        extractor = AudioFeatureExtractor(sr=22050, hop_length=512)

        # 3 audio files: 3, 5, and 10 minutes
        audio_lengths = [
            3 * 60 * 22050,  # 3 minutes
            5 * 60 * 22050,  # 5 minutes
            10 * 60 * 22050,  # 10 minutes
        ]

        all_features = []
        for audio_len in audio_lengths:
            audio = np.random.randn(audio_len).astype(np.float32)
            features = extractor.extract_windowed_features(audio, window_frames=10)

            # Each should be 60-dimensional
            self.assertEqual(
                features.shape[1],
                60,
                f"60-dimensional features required, got {features.shape[1]}",
            )

            all_features.append(features)

        # Concatenate all features
        combined = np.vstack(all_features)

        # Create DataLoader
        from torch.utils.data import DataLoader

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(combined))
        dataloader = DataLoader(dataset, batch_size=32)

        # Verify each batch maintains 60-dim
        batch_count = 0
        for batch in dataloader:
            batch_tensor = batch[0]
            self.assertEqual(
                batch_tensor.shape[1],
                60,
                f"Batch {batch_count} lost dimensionality: {batch_tensor.shape}",
            )
            batch_count += 1

        self.assertGreater(batch_count, 0, "Should have processed multiple batches")


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end tests for the entire audio processing pipeline."""

    def test_feature_consistency(self):
        """Test that features are consistent across extraction calls."""
        extractor = AudioFeatureExtractor(sr=22050, hop_length=512)

        # Same audio should produce same features
        audio = np.random.randn(110250).astype(np.float32)

        features1 = extractor.extract_windowed_features(audio, window_frames=10)
        features2 = extractor.extract_windowed_features(audio, window_frames=10)

        np.testing.assert_array_almost_equal(
            features1, features2, err_msg="Features should be deterministic"
        )

    def test_model_input_compatibility(self):
        """Test that extracted features are compatible with model input."""
        extractor = AudioFeatureExtractor(sr=22050, hop_length=512)

        # Extract features from synthetic audio (10 seconds)
        audio = np.random.randn(220500).astype(np.float32)
        features = extractor.extract_windowed_features(audio, window_frames=10)

        # Verify shape matches model expectation
        self.assertEqual(
            features.shape[1],
            60,
            "Features must be 60-dimensional for AudioToVisualModel",
        )

        # Convert to tensor (as trainer would do)
        features_tensor = torch.from_numpy(features).float()

        # Verify batch dimension
        self.assertEqual(
            features_tensor.shape[1],
            60,
            "Tensor features should maintain 60-dimensional feature space",
        )

        # Create batches (as dataloader would)
        batch_size = 32
        for i in range(0, len(features_tensor), batch_size):
            batch = features_tensor[i : i + batch_size]

            # Each batch should have correct feature dimension
            self.assertEqual(
                batch.shape[1],
                60,
                f"Batch has wrong dimension: {batch.shape}",
            )

    def test_long_audio_files(self):
        """Test processing of long audio files (like 5+ minute songs)."""
        extractor = AudioFeatureExtractor(sr=22050, hop_length=512)

        # Simulate a long song: 5 minutes
        audio = np.random.randn(5 * 60 * 22050).astype(np.float32)

        # Extract features
        features = extractor.extract_windowed_features(audio, window_frames=10)

        # Should produce many samples
        self.assertGreater(
            features.shape[0],
            1000,
            "Long audio should produce many windowed samples",
        )

        # But always 60-dimensional
        self.assertEqual(
            features.shape[1],
            60,
            "Features must always be 60-dimensional regardless of audio length",
        )

    def test_partial_batch_handling(self):
        """Test that trainer handles partial batches correctly.

        This is a regression test for the bug where the final batch
        of a dataset (size 30) would fail when previous_params from
        the prior batch (size 32) was used in smoothness_loss.
        """
        from src.audio_features import AudioFeatureExtractor
        from src.model import AudioToVisualModel
        from src.trainer import Trainer
        from src.visual_metrics import VisualMetrics
        from torch.utils.data import DataLoader, TensorDataset

        # Create a small dataset where last batch will be partial
        # 162 samples with batch_size=32 gives: 5 full batches + 1 batch of 2
        n_samples = 162
        input_dim = 60
        batch_size = 32

        # Create synthetic feature data
        features = torch.randn(n_samples, input_dim)
        dataset = TensorDataset(features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize dependencies
        feature_extractor = AudioFeatureExtractor(sr=22050)
        visual_metrics = VisualMetrics()

        # Initialize model and trainer
        model = AudioToVisualModel(window_frames=10)
        trainer = Trainer(
            model=model,
            feature_extractor=feature_extractor,
            visual_metrics=visual_metrics,
            learning_rate=0.0001,
        )

        # Try to train for one epoch - this should not crash on the partial batch
        try:
            epoch_losses = trainer.train_epoch(dataloader, epoch=1)

            # Verify we got a valid loss dictionary
            self.assertIsInstance(epoch_losses, dict)
            self.assertIn("loss", epoch_losses)
            self.assertIsInstance(epoch_losses["loss"], float)

            # Test passes if we reach here without exception
            self.assertTrue(True, "Partial batch handled successfully")

        except RuntimeError as e:
            if "size of tensor" in str(e):
                self.fail(
                    f"Partial batch caused tensor size mismatch: {e}\n"
                    "This likely means previous_params from a full batch (32) "
                    "was used with a partial batch (2)."
                )
            else:
                raise

    def test_partial_batch_handling_with_velocity_loss(self):
        """Test that trainer handles partial batches correctly with velocity loss enabled."""
        from src.audio_features import AudioFeatureExtractor
        from src.model import AudioToVisualModel
        from src.trainer import Trainer
        from src.visual_metrics import VisualMetrics
        from torch.utils.data import DataLoader, TensorDataset

        # Create a small dataset where last batch will be partial
        n_samples = 162
        input_dim = 60
        batch_size = 32

        # Create synthetic feature data
        features = torch.randn(n_samples, input_dim)
        dataset = TensorDataset(features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize dependencies
        feature_extractor = AudioFeatureExtractor(sr=22050)
        visual_metrics = VisualMetrics()

        # Initialize model and trainer WITH velocity loss
        model = AudioToVisualModel(window_frames=10)
        trainer = Trainer(
            model=model,
            feature_extractor=feature_extractor,
            visual_metrics=visual_metrics,
            learning_rate=0.0001,
            use_velocity_loss=True,
        )

        # Try to train for one epoch with velocity loss
        try:
            epoch_losses = trainer.train_epoch(dataloader, epoch=1)

            # Verify we got a valid loss dictionary
            self.assertIsInstance(epoch_losses, dict)
            self.assertIn("loss", epoch_losses)
            self.assertIn("velocity_loss", epoch_losses)
            self.assertIsInstance(epoch_losses["loss"], float)
            self.assertIsInstance(epoch_losses["velocity_loss"], float)

            # Test passes if we reach here without exception
            self.assertTrue(True, "Partial batch with velocity loss handled successfully")

        except RuntimeError as e:
            if "size of tensor" in str(e):
                self.fail(
                    f"Partial batch with velocity loss caused tensor size mismatch: {e}"
                )
            else:
                raise


if __name__ == "__main__":
    unittest.main()
