#!/usr/bin/env python
"""
Quick verification that runtime_core is working with all required functions.
"""

import sys


def main():
    print("=" * 70)
    print("FractalSync Runtime-Core Verification")
    print("=" * 70)

    # Test 1: Import runtime_core
    print("\n[1] Testing runtime_core module import...")
    try:
        import runtime_core

        print("[OK] runtime_core imported successfully")
        print(f"     - SAMPLE_RATE: {runtime_core.SAMPLE_RATE} Hz")
        print(f"     - HOP_LENGTH: {runtime_core.HOP_LENGTH}")
        print(f"     - N_FFT: {runtime_core.N_FFT}")
        print(f"     - DEFAULT_K_RESIDUALS: {runtime_core.DEFAULT_K_RESIDUALS}")
    except ImportError as e:
        print(f"[FAIL] Could not import runtime_core: {e}")
        return 1

    # Test 2: Import bridge module
    print("\n[2] Testing runtime_core_bridge import...")
    try:
        from backend.src.runtime_core_bridge import (
            make_feature_extractor,
            make_orbit_state,
            synthesize,
        )

        print("[OK] runtime_core_bridge functions imported")
    except ImportError as e:
        print(f"[FAIL] Could not import bridge functions: {e}")
        return 1

    # Test 3: Create feature extractor
    print("\n[3] Testing FeatureExtractor creation...")
    try:
        import numpy as np

        extractor = make_feature_extractor()
        print(f"[OK] FeatureExtractor created")

        # Test with audio
        audio = np.random.randn(48000).astype(np.float32)
        features = extractor.extract_windowed_features(audio, window_frames=10)
        print(f"     - Features shape: {features.shape}")
        print(f"     - Sample values: {features[0, :3]}")
    except Exception as e:
        print(f"[FAIL] Feature extraction failed: {e}")
        return 1

    # Test 4: Create orbit state and synthesize
    print("\n[4] Testing OrbitState and synthesis...")
    try:
        state = make_orbit_state(seed=12345)
        c = synthesize(state)
        print(f"[OK] Orbit synthesized")
        print(f"     - Complex value: {c.real:.6f} + {c.imag:.6f}i")
        print(f"     - Magnitude: {abs(c):.6f}")
    except Exception as e:
        print(f"[FAIL] Orbit synthesis failed: {e}")
        return 1

    # Test 5: Test determinism
    print("\n[5] Testing deterministic behavior...")
    try:
        state1 = make_orbit_state(seed=54321)
        c1 = synthesize(state1)

        state2 = make_orbit_state(seed=54321)
        c2 = synthesize(state2)

        diff_real = abs(c1.real - c2.real)
        diff_imag = abs(c1.imag - c2.imag)

        if diff_real < 1e-10 and diff_imag < 1e-10:
            print("[OK] Synthesis is deterministic")
            print(f"     - Difference (real): {diff_real}")
            print(f"     - Difference (imag): {diff_imag}")
        else:
            print("[FAIL] Synthesis not deterministic")
            return 1
    except Exception as e:
        print(f"[FAIL] Determinism test failed: {e}")
        return 1

    # Test 6: Test model integration
    print("\n[6] Testing model initialization...")
    try:
        from backend.src.control_model import AudioToControlModel

        model = AudioToControlModel(
            window_frames=10,
            n_features_per_frame=6,
            hidden_dims=[128, 256, 128],
            k_bands=6,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created with {total_params:,} parameters")
    except Exception as e:
        print(f"[FAIL] Model initialization failed: {e}")
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("[SUCCESS] All runtime_core components verified!")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Start the backend API: python backend/api/server.py")
    print("  2. Run training: python backend/train.py --data-dir data/audio")
    print("  3. Start the frontend: cd frontend && npm run dev")
    print("\nAll systems are ready for integration!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
