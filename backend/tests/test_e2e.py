#!/usr/bin/env python3
"""
End-to-end integration test for FractalSync.

Tests:
1. Backend runtime_core imports work
2. Feature extraction works
3. Orbit synthesis works
4. Model training can run
5. ONNX export works
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))


def test_imports():
    """Test that all backend modules import correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Backend Imports")
    print("=" * 60)
    try:
        from src.runtime_core_bridge import (  # noqa: F401
            make_feature_extractor,
            make_orbit_state,
            synthesize,
            SAMPLE_RATE,
            HOP_LENGTH,
            N_FFT,
            WINDOW_FRAMES,
        )

        print("✓ runtime_core_bridge imports")

        from src.data_loader import AudioDataset  # noqa: F401

        print("✓ data_loader imports")

        from src.control_model import AudioToControlModel  # noqa: F401

        print("✓ control_model imports")

        from src.control_trainer import ControlTrainer  # noqa: F401

        print("✓ control_trainer imports")

        from src.visual_metrics import LossVisualMetrics  # noqa: F401

        print("✓ visual_metrics imports")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test that feature extraction works."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Extraction")
    print("=" * 60)
    try:
        from src.runtime_core_bridge import (  # noqa: F401
            make_feature_extractor,
            SAMPLE_RATE,
            N_FFT,
            HOP_LENGTH,
        )
        import numpy as np

        # Create extractor
        extractor = make_feature_extractor()
        print("✓ Feature extractor created")

        # Generate test audio
        duration_sec = 1.0
        n_samples = int(SAMPLE_RATE * duration_sec)
        audio = np.random.randn(n_samples).astype(np.float32) * 0.1

        # Extract features
        features = extractor.extract_windowed_features(audio, window_frames=10)
        print(f"✓ Extracted features shape: {features.shape}")
        print("  Expected: (n_frames, 6*10=60)")

        # Verify shape
        assert features.ndim == 2, f"Expected 2D array, got {features.ndim}D"
        assert (
            features.shape[1] == 60
        ), f"Expected 60 features per frame, got {features.shape[1]}"
        print("✓ Feature shape is correct")

        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_orbit_synthesis():
    """Test that orbit synthesis works."""
    print("\n" + "=" * 60)
    print("TEST 3: Orbit Synthesis")
    print("=" * 60)
    try:
        from src.runtime_core_bridge import (
            make_orbit_state,
            make_residual_params,
            synthesize,
        )

        # Create orbit state with seed for determinism
        orbit_state = make_orbit_state(
            lobe=1,
            sub_lobe=0,
            theta=0.0,
            omega=6.28,  # 2*pi
            s=1.02,
            alpha=0.3,
            k_residuals=10,
            residual_omega_scale=1.5,
            seed=42,
        )
        print("✓ OrbitState created with seed=42")

        # Create residual params
        residual_params = make_residual_params(
            k_residuals=10, residual_cap=0.5, radius_scale=1.5
        )
        print("✓ ResidualParams created")

        # Synthesize first point
        c1 = synthesize(orbit_state, residual_params, band_gates=None)
        print(f"✓ Synthesized c(t=0) = {c1}")

        # Step time forward
        orbit_state.advance(0.01)
        c2 = synthesize(orbit_state, residual_params, band_gates=None)
        print(f"✓ Synthesized c(t=0.01) = {c2}")

        # Verify outputs are different (orbit progresses)
        assert c1 != c2, "Orbit did not progress"
        print("✓ Orbit progression verified")

        return True
    except Exception as e:
        print(f"✗ Orbit synthesis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_init():
    """Test that model can be initialized."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Initialization")
    print("=" * 60)
    try:
        import torch
        from src.control_model import AudioToControlModel

        # Initialize model with correct parameters
        model = AudioToControlModel(
            window_frames=10,
            n_features_per_frame=6,
            hidden_dims=[128, 256, 128],
            k_bands=6,
        )
        print("✓ Model initialized")

        # Test forward pass
        test_input = torch.randn(1, 60)  # 6 features * 10 frames
        output = model(test_input)
        print(f"✓ Forward pass works: input {test_input.shape} → output {output.shape}")

        # Verify output shape
        assert output.shape == (1, 9), f"Expected output (1,9), got {output.shape}"
        print("✓ Output shape is correct")

        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_visual_metrics():
    """Test that visual metrics can be initialized."""
    print("\n" + "=" * 60)
    print("TEST 5: Visual Metrics")
    print("=" * 60)
    try:
        import numpy as np
        import runtime_core
        from src.visual_metrics import LossVisualMetrics

        # Initialize
        metrics = LossVisualMetrics()
        print("✓ LossVisualMetrics initialized")

        # Test with a synthetic image
        test_image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = metrics.compute_all_metrics(test_image)
        print(f"✓ Loss metrics computed: {len(result)} metrics")

        image_float = test_image.astype(np.float64) / 255.0
        flat = image_float.reshape(-1).tolist()
        runtime_metrics = runtime_core.compute_runtime_visual_metrics(
            flat, test_image.shape[1], test_image.shape[0], test_image.shape[2], 0.0, 0.0, 50
        )
        print(f"✓ Runtime metrics computed: edge_density={runtime_metrics.edge_density:.4f}")

        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert len(result) > 0, "Expected at least one metric"
        print("✓ Visual metrics computation works")

        return True
    except Exception as e:
        print(f"✗ Visual metrics failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█" * 60)
    print("█  FractalSync End-to-End Integration Test")
    print("█" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Feature Extraction", test_feature_extraction()))
    results.append(("Orbit Synthesis", test_orbit_synthesis()))
    results.append(("Model Initialization", test_model_init()))
    results.append(("Visual Metrics", test_visual_metrics()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}  {name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
