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
from numpy.typing import NDArray

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_audio(
    sample_rate: int = 48000, duration: float = 1.0
) -> NDArray[np.float32]:
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

    #

    # Save audio to temp file (path relative to this test file to avoid depending on working directory)
    audio_path = (
        Path(__file__).parent.parent / "data" / "cache" / "parity_test_audio.npy"
    )
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(audio_path, audio)

    # Find cargo
    cargo_cmd = "cargo"
    # Try to find cargo in standard Windows location
    cargo_path = Path.home() / ".cargo" / "bin" / "cargo.exe"
    if cargo_path.exists():
        cargo_cmd = str(cargo_path)

    # Run Rust test that reads this file and outputs features
    import os

    env = os.environ.copy()
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
    output_path = (
        Path(__file__).parent.parent / "data" / "cache" / "parity_test_features.json"
    )
    if not output_path.exists():
        raise RuntimeError(f"Rust test did not create {output_path}")

    with open(output_path) as f:
        features = json.load(f)

    return np.array(features, dtype=np.float64)
