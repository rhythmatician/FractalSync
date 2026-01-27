#!/usr/bin/env python3
"""
End-to-end integration test for FractalSync.

Tests:
1. Backend runtime_core imports work
2. Feature extraction works
3. Height-field controller works
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
            height_field,
            height_controller_step,
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

        from src.visual_metrics import VisualMetrics  # noqa: F401

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


def test_height_controller():
    """Test that height-field controller works."""
    print("\n" + "=" * 60)
    print("TEST 3: Height Controller")
    print("=" * 60)
    try:
        from src.runtime_core_bridge import height_controller_step
        import runtime_core as rc

        c = rc.Complex(-0.6, 0.0)
        delta = rc.Complex(0.01, -0.01)

        step1 = height_controller_step(c, delta, target_height=-0.5, normal_risk=0.1)
        step2 = height_controller_step(c, delta, target_height=-0.5, normal_risk=0.1)

        print("✓ Height controller step computed")
        print(f"✓ Step1 c = {step1.new_c}")
        print(f"✓ Step2 c = {step2.new_c}")

        assert step1.new_c.real == step2.new_c.real
        assert step1.new_c.imag == step2.new_c.imag

        return True
    except Exception as e:
        print(f"✗ Height controller failed: {e}")
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
        )
        print("✓ Model initialized")

        # Test forward pass
        test_input = torch.randn(1, 60)  # 6 features * 10 frames
        output = model(test_input)
        print(f"✓ Forward pass works: input {test_input.shape} → output {output.shape}")

        # Verify output shape
        assert output.shape == (1, 4), f"Expected output (1,4), got {output.shape}"
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
        from src.visual_metrics import VisualMetrics

        # Initialize
        metrics = VisualMetrics()
        print("✓ VisualMetrics initialized")

        # Test with a synthetic image
        test_image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = metrics.compute_all_metrics(test_image)
        print(f"✓ Metrics computed: {len(result)} metrics")

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
    results.append(("Height Controller", test_height_controller()))
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
