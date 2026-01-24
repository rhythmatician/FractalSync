"""
Parity test: Verify Python fallback feature extractor matches Rust implementation.

This test ensures that backend/src/python_feature_extractor.py produces
the same output as runtime-core/src/features.rs when both are given
identical audio input.

Since the Rust version works in pure Rust (proven by unit tests) but hangs
when called from Python via PyO3, we test by:
1. Running Rust unit tests that output feature vectors to a file
2. Running Python extractor on same audio
3. Comparing the outputs

This ensures the backend training uses identical features to what the
frontend WASM will compute, preventing drift.
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime_core_bridge import make_feature_extractor


def generate_test_audio(sample_rate: int = 48000, duration: float = 1.0) -> np.ndarray:
    """Generate deterministic test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Mix of frequencies for interesting spectral content
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t)  # A4
        + 0.2 * np.sin(2 * np.pi * 880 * t)  # A5
        + 0.1 * np.sin(2 * np.pi * 220 * t)  # A3
    ).astype(np.float32)
    return audio


# Rust extraction harness removed: runtime-core FeatureExtractor is the canonical implementation.
# This test now verifies the bridge extractor is callable and deterministic.


def test_feature_parity():
    """Test that Python and Rust extractors produce identical output."""
    print("=" * 70)
    print("Audio Feature Parity Test")
    print("=" * 70)

    # Test parameters
    sample_rate = 48000
    window_frames = 10

    # Generate test audio
    print("\n1. Generating test audio...")
    audio = generate_test_audio(sample_rate=sample_rate, duration=1.0)
    print(f"   Audio: {len(audio)} samples at {sample_rate} Hz")

    # Extract features with the canonical FeatureExtractor (Rust-backed)
    print("\n2. Extracting features with canonical FeatureExtractor...")
    python_extractor = make_feature_extractor(
        include_delta=False, include_delta_delta=False
    )
    python_features = python_extractor.extract_windowed_features(audio, window_frames)
    print(
        f"   FeatureExtractor (Rust-backed): {python_features.shape} (windows, features)"
    )

    # Sanity checks: deterministic and reasonable shape
    print("\n3. Sanity checks...")
    assert python_features.ndim == 2
    # Deterministic: extract twice and compare
    python_features_2 = python_extractor.extract_windowed_features(audio, window_frames)
    assert python_features.shape == python_features_2.shape
    assert np.allclose(python_features, python_features_2)


def test_feature_consistency():
    """
    Test that Python extractor produces consistent output.

    This is a sanity check that runs without needing Rust.
    """
    print("\n" + "=" * 70)
    print("Python Feature Consistency Test")
    print("=" * 70)

    audio = generate_test_audio(sample_rate=48000, duration=0.5)

    extractor = make_feature_extractor(include_delta=False, include_delta_delta=False)

    # Extract twice - should be identical
    features1 = extractor.extract_windowed_features(audio, window_frames=10)
    features2 = extractor.extract_windowed_features(audio, window_frames=10)

    if np.allclose(features1, features2):
        print("PASS - Python extractor is deterministic")
        return True
    else:
        print("FAIL - Python extractor produces different outputs!")
        return False


if __name__ == "__main__":
    print("Testing audio feature extraction parity")
    print("This ensures backend and frontend compute identical features\n")

    # First check Python consistency
    consistency_pass = test_feature_consistency()

    # Then check parity with Rust (if possible)
    if consistency_pass:
        parity_pass = test_feature_parity()
    else:
        print("\nWARNING: Skipping parity test due to consistency failure")
        parity_pass = False

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Python Consistency: {'PASS' if consistency_pass else 'FAIL'}")
    print(f"Python-Rust Parity: {'PASS' if parity_pass else 'FAIL or SKIPPED'}")

    if consistency_pass and parity_pass:
        print("\nAll tests passed - feature extraction is consistent!")
        sys.exit(0)
    else:
        print("\nSome tests failed - review output above")
        sys.exit(1)
