"""
Parity tracing harness: verify backend and frontend produce identical orbit synthesis results.

This module provides utilities to:
1. Extract runtime constants and feature extractor parameters
2. Run feature extraction with deterministic seeds
3. Perform orbit synthesis and compare numeric precision
4. Generate parity test reports

Usage:
    python -m backend.src.parity_trace \
        --audio-file data/audio/sample.wav \
        --seed 1337 \
        --output-report parity_results.json
"""

import json
import argparse
from typing import Dict, Any, cast

import numpy as np

from .runtime_core_bridge import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    WINDOW_FRAMES,
    DEFAULT_K_RESIDUALS,
    DEFAULT_RESIDUAL_CAP,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    DEFAULT_BASE_OMEGA,
    DEFAULT_ORBIT_SEED,
    make_feature_extractor,
    make_orbit_state,
    synthesize,
    step_orbit,
)


def dump_constants() -> Dict[str, Any]:
    """
    Dump all shared constants for parity verification.

    These constants must match between Python backend and JavaScript/WASM frontend.
    """
    return {
        "SAMPLE_RATE": SAMPLE_RATE,
        "HOP_LENGTH": HOP_LENGTH,
        "N_FFT": N_FFT,
        "WINDOW_FRAMES": WINDOW_FRAMES,
        "DEFAULT_K_RESIDUALS": DEFAULT_K_RESIDUALS,
        "DEFAULT_RESIDUAL_CAP": DEFAULT_RESIDUAL_CAP,
        "DEFAULT_RESIDUAL_OMEGA_SCALE": DEFAULT_RESIDUAL_OMEGA_SCALE,
        "DEFAULT_BASE_OMEGA": DEFAULT_BASE_OMEGA,
        "DEFAULT_ORBIT_SEED": DEFAULT_ORBIT_SEED,
    }


def extract_features_deterministic(
    audio_path: str, seed: int = DEFAULT_ORBIT_SEED
) -> np.ndarray:
    """
    Extract windowed audio features with deterministic settings.

    Args:
        audio_path: Path to audio file
        seed: Random seed for reproducibility (not used by extractor, for logging)

    Returns:
        Feature array of shape (n_windows, n_features * window_frames)
    """
    import librosa

    # Load audio at shared sample rate
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Create feature extractor
    extractor = make_feature_extractor()

    # Extract windowed features
    features = extractor.extract_windowed_features(
        audio.astype(np.float32), window_frames=WINDOW_FRAMES
    )

    return cast(np.ndarray, features)


def generate_orbit_sequence_deterministic(
    duration: float = 1.0,
    dt: float = 1.0 / 48000.0,
    seed: int = DEFAULT_ORBIT_SEED,
) -> Dict[str, Any]:
    """
    Generate deterministic orbit sequence for parity testing.

    Args:
        duration: Sequence duration in seconds
        dt: Time step (usually 1/sample_rate)
        seed: RNG seed for orbit state initialization

    Returns:
        Dict with sequence data and metadata
    """
    n_samples = int(duration / dt)

    # Create orbit state with deterministic seed
    state = make_orbit_state(
        lobe=1,
        sub_lobe=0,
        theta=0.0,
        omega=DEFAULT_BASE_OMEGA,
        s=1.02,
        alpha=0.3,
        k_residuals=DEFAULT_K_RESIDUALS,
        residual_omega_scale=DEFAULT_RESIDUAL_OMEGA_SCALE,
        seed=seed,
    )

    # Generate sequence
    c_sequence = []
    for i in range(n_samples):
        c = synthesize(state)
        c_sequence.append({"re": c.re, "im": c.im})
        # Advance state for next iteration
        step_orbit(state, dt)

    return {
        "duration": duration,
        "dt": dt,
        "n_samples": n_samples,
        "seed": seed,
        "sequence": c_sequence[:100],  # First 100 samples for report
    }


def compare_numeric_precision(
    reference: Dict[str, Any],
    test: Dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compare two numeric sequences for parity within tolerance.

    Args:
        reference: Reference sequence (e.g., from Python backend)
        test: Test sequence (e.g., from WASM frontend)
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dict with comparison results and any deviations
    """
    issues = []

    # Check metadata
    if reference.get("n_samples") != test.get("n_samples"):
        issues.append(
            f"Sample count mismatch: {reference['n_samples']} vs {test['n_samples']}"
        )

    # Compare sequences
    ref_seq = reference.get("sequence", [])
    test_seq = test.get("sequence", [])

    if len(ref_seq) != len(test_seq):
        issues.append(f"Sequence length mismatch: {len(ref_seq)} vs {len(test_seq)}")

    max_real_error = 0.0
    max_imag_error = 0.0

    for i, (ref_c, test_c) in enumerate(zip(ref_seq, test_seq)):
        real_error = abs(ref_c["real"] - test_c["real"])
        imag_error = abs(ref_c["imag"] - test_c["imag"])

        max_real_error = max(max_real_error, real_error)
        max_imag_error = max(max_imag_error, imag_error)

        # Check tolerance
        tol = atol + rtol * max(abs(ref_c["real"]), abs(test_c["real"]))
        if real_error > tol or imag_error > tol:
            if len(issues) < 5:  # Limit issue reporting
                issues.append(
                    f"Sample {i}: real_error={real_error:.2e}, imag_error={imag_error:.2e}"
                )

    return {
        "passed": len(issues) == 0,
        "max_real_error": max_real_error,
        "max_imag_error": max_imag_error,
        "issues": issues,
    }


def generate_parity_report(output_path: str = "parity_trace.json") -> None:
    """
    Generate a complete parity trace report.

    This report captures all shared constants and deterministic orbit sequences
    that can be compared with the frontend implementation.

    Args:
        output_path: Path to write JSON report
    """
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "runtime": "Python backend (runtime_core + PyO3)",
        "constants": dump_constants(),
        "orbit_sequence": generate_orbit_sequence_deterministic(
            duration=0.1, seed=DEFAULT_ORBIT_SEED
        ),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Parity trace report written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate parity trace for FractalSync"
    )
    parser.add_argument(
        "--output-report",
        default="parity_trace.json",
        help="Output path for parity trace JSON",
    )
    parser.add_argument(
        "--duration", type=float, default=0.1, help="Orbit sequence duration (seconds)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_ORBIT_SEED,
        help="Random seed for deterministic generation",
    )

    args = parser.parse_args()

    print("Generating parity trace...")
    generate_parity_report(output_path=args.output_report)
    print(f"Report written to: {args.output_report}")
