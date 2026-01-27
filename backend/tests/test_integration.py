#!/usr/bin/env python
"""
End-to-end test suite for FractalSync runtime-core integration.

Prerequisites:
  - Python runtime_core bindings built: `cd runtime-core && maturin develop --release`

Run this test to verify:
  1. Backend feature extraction works
  2. Height-field controller is deterministic
  3. Model training pipeline initializes
"""

import sys


def test_imports():
    """Test 1: Can we import runtime_core and bridge?"""
    print("\n[Test 1] Testing imports...")
    try:
        import runtime_core

        print("  [OK] runtime_core imported")
        print(f"    - SAMPLE_RATE: {runtime_core.SAMPLE_RATE}")
        print(f"    - DEFAULT_HEIGHT_ITERATIONS: {runtime_core.DEFAULT_HEIGHT_ITERATIONS}")

        from src.runtime_core_bridge import (  # noqa: F401
            make_feature_extractor,
            height_field,
            height_controller_step,
        )

        print("  [OK] runtime_core_bridge imported")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        print("    -> Run: cd runtime-core && maturin develop --release")
        return False


def test_feature_extraction():
    """Test 2: Can we extract features?"""
    print("\n[Test 2] Testing feature extraction...")
    try:
        import numpy as np
        from src.runtime_core_bridge import make_feature_extractor

        extractor = make_feature_extractor()
        print("  ✓ Feature extractor created")

        # Generate synthetic audio
        audio = np.random.randn(48000).astype(np.float32)  # 1 second at 48kHz
        features = extractor.extract_windowed_features(audio, window_frames=10)

        print(f"  ✓ Features extracted: shape {features.shape}")
        assert features.shape[1] > 0, "Feature dimension mismatch"
        return True
    except Exception as e:
        print(f"  ✗ Feature extraction failed: {e}")
        return False


def test_height_controller():
    """Test 3: Can we step the height controller deterministically?"""
    print("\n[Test 3] Testing height controller...")
    try:
        from src.runtime_core_bridge import height_controller_step
        import runtime_core as rc

        c = rc.Complex(-0.6, 0.0)
        delta = rc.Complex(0.01, -0.005)
        step1 = height_controller_step(c, delta, target_height=-0.5, normal_risk=0.1)
        step2 = height_controller_step(c, delta, target_height=-0.5, normal_risk=0.1)

        print("  ✓ Height controller step computed")
        print(f"    - Step1 c: {step1.new_c.real:.6f} + {step1.new_c.imag:.6f}i")
        print(f"    - Step2 c: {step2.new_c.real:.6f} + {step2.new_c.imag:.6f}i")

        # Check determinism
        if (
            abs(step1.new_c.real - step2.new_c.real) < 1e-10
            and abs(step1.new_c.imag - step2.new_c.imag) < 1e-10
        ):
            print("  ✓ Controller step is deterministic")
            return True
        print("  ✗ Controller step not deterministic")
        return False
    except Exception as e:
        print(f"  ✗ Height controller failed: {e}")
        return False


def test_model_init():
    """Test 4: Can we initialize the training model?"""
    print("\n[Test 4] Testing model initialization...")
    try:
        from src.control_model import AudioToControlModel

        model = AudioToControlModel(
            window_frames=10,
            n_features_per_frame=6,
            hidden_dims=[128, 256, 128],
        )
        print("  ✓ AudioToControlModel created")

        # Check parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    - Total parameters: {total_params:,}")
        return True
    except Exception as e:
        print(f"  ✗ Model initialization failed: {e}")
        return False


def test_visual_metrics():
    """Test 5: Can we compute visual metrics?"""
    print("\n[Test 5] Testing visual metrics...")
    try:
        import numpy as np
        from src.visual_metrics import VisualMetrics

        vm = VisualMetrics()
        print("  ✓ VisualMetrics created")

        # Create a synthetic image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Compute metrics
        metrics = vm.compute_all_metrics(image)
        print(f"    - Metrics computed: {len(metrics)} metrics")
        print(f"    - Edge density: {metrics['edge_density']:.4f}")

        return True
    except Exception as e:
        print(f"  ✗ Visual metrics failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("FractalSync Runtime-Core Integration Tests")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Feature Extraction", test_feature_extraction),
        ("Height Controller", test_height_controller),
        ("Model Initialization", test_model_init),
        ("Visual Metrics", test_visual_metrics),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All systems ready!")
        print("\nNext steps:")
        print("  1. Start backend API: cd backend && python api/server.py")
        print("  2. Run training: cd backend && python train.py --data-dir data/audio")
        print("  3. Start frontend: cd frontend && npm run dev")
        return 0
    else:
        print("\n⚠️  Some tests failed. See details above.")
        if not results.get("Imports"):
            print("\nCritical: runtime_core not available.")
            print("  Fix: cd runtime-core && maturin develop --release")
        return 1


if __name__ == "__main__":
    sys.exit(main())
