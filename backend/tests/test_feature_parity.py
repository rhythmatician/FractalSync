"""
Test feature extraction parity between backend and frontend.

This module generates reference feature data from backend extraction,
which is compared against frontend extraction in the TypeScript tests.
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.audio_features import AudioFeatureExtractor  # noqa: E402


# Use the actual Tool song for testing
TEST_AUDIO_FILE = (
    Path(__file__).parent.parent / "data" / "audio" / "TOOL - The Grudge (Audio).mp3"
)
REFERENCE_FEATURES_FILE = Path(__file__).parent / "fixtures" / "feature_baseline.json"


@pytest.fixture(scope="session")
def test_audio_path():
    """Get path to test audio file."""
    if not TEST_AUDIO_FILE.exists():
        pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}")
    return str(TEST_AUDIO_FILE)


@pytest.fixture(scope="session")
def audio_data(test_audio_path):
    """Load audio data once for the test session."""
    audio, _ = librosa.load(test_audio_path, sr=22050)
    return audio


@pytest.fixture(scope="session")
def feature_extractor():
    """Create feature extractor with default settings."""
    return AudioFeatureExtractor(
        include_delta=False,
        include_delta_delta=False,
    )


class TestFeatureExtraction:
    """Test backend feature extraction consistency."""

    def test_feature_extraction_deterministic(self, audio_data, feature_extractor):
        """
        Verify that feature extraction is deterministic
        (same audio → same features every time).
        """
        features1 = feature_extractor.extract_windowed_features(audio_data)
        features2 = feature_extractor.extract_windowed_features(audio_data)

        assert features1.shape == features2.shape
        np.testing.assert_array_almost_equal(features1, features2, decimal=10)

    def test_feature_shape(self, audio_data, feature_extractor):
        """
        Verify feature shape is correct: (n_samples, n_features * window_frames).
        """
        features = feature_extractor.extract_windowed_features(audio_data)

        # 6 base features, 10 frames per window
        expected_features_per_sample = 6 * 10

        assert features.ndim == 2
        assert features.shape[1] == expected_features_per_sample
        assert features.shape[0] > 0  # Should have samples

    def test_feature_normalization(self, audio_data, feature_extractor):
        """
        Verify features are normalized (zero mean, unit std) after normalization.
        """
        features = feature_extractor.extract_windowed_features(audio_data)
        feature_extractor.compute_normalization_stats([features])
        normalized = feature_extractor.normalize_features(features)

        # Check normalized features have reasonable range
        assert (
            np.abs(normalized.mean()) < 0.1
        ), "Normalized features should have near-zero mean"
        assert (
            np.abs(normalized.std() - 1.0) < 0.1
        ), "Normalized features should have unit std"

    def test_feature_range(self, audio_data, feature_extractor):
        """
        Verify raw features are in expected ranges.
        """
        features = feature_extractor.extract_windowed_features(audio_data)

        # Features should be non-negative or within audio feature ranges
        assert np.all(np.isfinite(features)), "All features should be finite"
        assert np.all(features >= 0), "Audio features should be non-negative"

    def test_generate_feature_baseline(self, audio_data, feature_extractor):
        """
        Generate and save feature baseline for frontend comparison.

        This creates a JSON file with:
        - Raw features from the test audio
        - Normalization parameters
        - Feature configuration

        The frontend test will extract the same audio and compare.
        """
        # Extract features
        features = feature_extractor.extract_windowed_features(audio_data)

        # Compute normalization
        feature_extractor.compute_normalization_stats([features])
        normalized = feature_extractor.normalize_features(features)

        # Create baseline data
        baseline = {
            "audio_file": TEST_AUDIO_FILE.name,
            "config": {
                "include_delta": False,
                "include_delta_delta": False,
                "window_frames": 10,
                "n_features": 6,
            },
            "statistics": {
                "n_samples": int(features.shape[0]),
                "n_features_per_sample": int(features.shape[1]),
            },
            "normalization": {
                "feature_mean": feature_extractor.feature_mean.tolist(),
                "feature_std": feature_extractor.feature_std.tolist(),
            },
            # Store first 10 samples of raw and normalized features for quick comparison
            "sample_raw": features[:10].tolist(),
            "sample_normalized": normalized[:10].tolist(),
        }

        # Ensure fixtures directory exists
        REFERENCE_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Write baseline
        with open(REFERENCE_FEATURES_FILE, "w") as f:
            json.dump(baseline, f, indent=2)

        # Verify it was written
        assert REFERENCE_FEATURES_FILE.exists()

        # Verify we can read it back
        with open(REFERENCE_FEATURES_FILE, "r") as f:
            loaded = json.load(f)

        assert loaded["statistics"]["n_samples"] == baseline["statistics"]["n_samples"]
        assert len(loaded["sample_raw"]) == 10
        assert len(loaded["sample_normalized"]) == 10

    def test_feature_baseline_matches_fresh_extraction(
        self, audio_data, feature_extractor
    ):
        """
        Verify that loading and using the baseline matches fresh extraction.
        """
        # Load baseline
        with open(REFERENCE_FEATURES_FILE, "r") as f:
            baseline = json.load(f)

        # Fresh extraction
        features = feature_extractor.extract_windowed_features(audio_data)
        feature_extractor.compute_normalization_stats([features])
        normalized = feature_extractor.normalize_features(features)

        # Compare first 10 samples
        baseline_raw = np.array(baseline["sample_raw"])
        baseline_norm = np.array(baseline["sample_normalized"])

        np.testing.assert_array_almost_equal(
            features[:10],
            baseline_raw,
            decimal=5,
            err_msg="Raw features should match baseline",
        )

        np.testing.assert_array_almost_equal(
            normalized[:10],
            baseline_norm,
            decimal=5,
            err_msg="Normalized features should match baseline",
        )
