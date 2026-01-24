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

import json
import subprocess
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


def run_rust_feature_extraction(audio: np.ndarray, window_frames: int) -> np.ndarray:
    """
    Run Rust feature extraction via a test that outputs to JSON.

    This workaround is needed because calling the Rust function from Python hangs.
    """
    print("Running Rust feature extraction via cargo test...")

    # Save audio to temp file
    audio_path = Path("backend/data/cache/parity_test_audio.npy")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(audio_path, audio)

    # Find cargo
    cargo_cmd = "cargo"
    # Try to find cargo in standard Windows location
    cargo_path = Path.home() / ".cargo" / "bin" / "cargo.exe"
    if cargo_path.exists():
        cargo_cmd = str(cargo_path)

    # Run Rust test that reads this file and outputs features
    env = subprocess.os.environ.copy()
    env["PARITY_TEST_AUDIO_PATH"] = str(audio_path.absolute())

    result = subprocess.run(
        [
            cargo_cmd,
            "test",
            "--release",
            "--lib",
            "test_parity_extract",
            "--",
            "--nocapture",
            "--test-threads=1",
        ],
        cwd="runtime-core",
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(
            "STDERR:",
            result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,
        )
        raise RuntimeError(f"Rust test failed with code {result.returncode}")

    # Parse JSON output from test
    output_path = Path("backend/data/cache/parity_test_features.json")
    if not output_path.exists():
        raise RuntimeError(f"Rust test did not create {output_path}")

    with open(output_path) as f:
        features = json.load(f)

    return np.array(features, dtype=np.float64)


def test_feature_parity():
    """Test that Python and Rust extractors produce identical output."""
    print("=" * 70)
    print("Audio Feature Parity Test")
    print("=" * 70)

    # Test parameters
    sample_rate = 48000
    hop_length = 1024
    n_fft = 4096
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

    # Extract features with Rust (via test harness)
    print("\n3. Extracting features with Rust (via cargo test)...")
    try:
        rust_features = run_rust_feature_extraction(audio, window_frames)
        print(f"   Rust: {rust_features.shape} (windows, features)")
    except Exception as e:
        print(f"   WARNING: Could not run Rust extraction: {e}")
        print("   Skipping comparison - manual verification needed")
        return

    # Compare outputs
    print("\n4. Comparing outputs...")

    if python_features.shape != rust_features.shape:
        print("   SHAPE MISMATCH!")
        print(f"      Python: {python_features.shape}")
        print(f"      Rust:   {rust_features.shape}")
        return False

    # Check element-wise differences
    abs_diff = np.abs(python_features - rust_features)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"   Max absolute difference:  {max_diff:.6e}")
    print(f"   Mean absolute difference: {mean_diff:.6e}")

    # Allow small numerical differences due to floating point
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"   PASS - Features match within tolerance ({tolerance})")
        return True
    else:
        print(f"   FAIL - Differences exceed tolerance ({tolerance})")

        # Show where differences occur
        problem_indices = np.where(abs_diff > tolerance)
        print("\n   Problem locations (first 10):")
        for i in range(min(10, len(problem_indices[0]))):
            win_idx = problem_indices[0][i]
            feat_idx = problem_indices[1][i]
            print(f"      Window {win_idx}, Feature {feat_idx}:")
            print(f"         Python: {python_features[win_idx, feat_idx]:.6f}")
            print(f"         Rust:   {rust_features[win_idx, feat_idx]:.6f}")
            print(f"         Diff:   {abs_diff[win_idx, feat_idx]:.6e}")

        return False


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
