#!/usr/bin/env python
"""
End-to-end test suite for FractalSync runtime-core integration.

Prerequisites:
  - Python runtime_core bindings built: `Push-Location runtime-core; try { maturin develop --release } finally { Pop-Location }`
  - WASM bindings built: [OK] (already built in wasm-orbit/pkg/)

Run this test to verify:
  1. Backend feature extraction works
  2. Orbit synthesis is deterministic
  3. Model training pipeline initializes
  4. WASM module loads (frontend ready)
"""
from typing import Iterable, Optional

import sys
from pathlib import Path
from runtime_core import (
    ResidualParams,
    OrbitState,
    DEFAULT_K_RESIDUALS,
    DEFAULT_RESIDUAL_CAP,
    SAMPLE_RATE,
    DEFAULT_BASE_OMEGA,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    DEFAULT_ORBIT_SEED,
)


def test_imports():
    """Test 1: Can we import runtime_core and bridge?"""
    print("\n[Test 1] Testing imports...")
    try:

        print("  [OK] runtime_core imported")
        print(f"    - SAMPLE_RATE: {SAMPLE_RATE}")
        print(f"    - DEFAULT_K_RESIDUALS: {DEFAULT_K_RESIDUALS}")

        from runtime_core import FeatureExtractor  # noqa: F401

        print("  [OK] FeatureExtractor available in runtime_core")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        print(
            "    -> Run (PowerShell-safe): Push-Location runtime-core; try { maturin develop --release } finally { Pop-Location }"
        )
        return False


def test_feature_extraction():
    """Test 2: Can we extract features?"""
    print("\n[Test 2] Testing feature extraction...")
    try:
        import numpy as np
        from runtime_core import FeatureExtractor, SAMPLE_RATE, HOP_LENGTH, N_FFT

        extractor = FeatureExtractor(
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
        )
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


def test_orbit_synthesis():
    """Test 3: Can we synthesize orbits deterministically?"""
    print("\n[Test 3] Testing orbit synthesis...")
    try:

        def make_orbit_state(
            *,
            lobe: int = 1,
            sub_lobe: int = 0,
            theta: float = 0.0,
            omega: float = DEFAULT_BASE_OMEGA,
            s: float = 1.02,
            alpha: float = 0.3,
            k_residuals: int = DEFAULT_K_RESIDUALS,
            residual_omega_scale: float = DEFAULT_RESIDUAL_OMEGA_SCALE,
            seed: int = DEFAULT_ORBIT_SEED,
        ) -> OrbitState:
            """Construct a deterministic orbit state using the Rust implementation.

            Use positional arguments to avoid relying on a keyword name that may not
            be present in all generated Python bindings. If `seed` is provided the
            Rust constructor that accepts a seed will be used.
            """
            return OrbitState.new_with_seed(
                lobe,
                sub_lobe,
                theta,
                omega,
                s,
                alpha,
                k_residuals,
                residual_omega_scale,
                seed,
            )

        def synthesize(
            state: OrbitState,
            residual_params: Optional[ResidualParams] = None,
            band_gates: Optional[Iterable[float]] = None,
        ) -> complex:
            rp = residual_params or ResidualParams(
                k_residuals=DEFAULT_K_RESIDUALS,
                residual_cap=DEFAULT_RESIDUAL_CAP,
                radius_scale=1.0,
            )
            return state.synthesize(
                rp, list(band_gates) if band_gates is not None else None
            )

        # Create state with seed
        state1 = make_orbit_state(seed=1337)
        c1 = synthesize(state1)

        # Create another with same seed
        state2 = make_orbit_state(seed=1337)
        c2 = synthesize(state2)

        print("  ✓ Orbit state created (seed=1337)")
        print(f"    - First synthesis: {c1.real:.6f} + {c1.imag:.6f}i")
        print(f"    - Second synthesis: {c2.real:.6f} + {c2.imag:.6f}i")

        # Check determinism
        if abs(c1.real - c2.real) < 1e-10 and abs(c1.imag - c2.imag) < 1e-10:
            print("  ✓ Synthesis is deterministic")
            return True
        else:
            print("  ✗ Synthesis not deterministic")
            return False
    except Exception as e:
        print(f"  ✗ Orbit synthesis failed: {e}")
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
            context_dim=265,
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
        import runtime_core
        from src.visual_metrics import LossVisualMetrics

        vm = LossVisualMetrics()
        print("  ✓ LossVisualMetrics created")

        # Create a synthetic image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Compute loss metrics
        metrics = vm.compute_all_metrics(image)
        print(f"    - Loss metrics computed: {len(metrics)} metrics")
        print(f"    - Temporal change: {metrics['temporal_change']:.4f}")

        # Compute runtime metrics from Rust
        image_float = image.astype(np.float64) / 255.0
        flat = image_float.reshape(-1).tolist()
        runtime_metrics = runtime_core.compute_runtime_visual_metrics(
            flat, image.shape[1], image.shape[0], image.shape[2], 0.0, 0.0, 50
        )
        print(f"    - Edge density: {runtime_metrics.edge_density:.4f}")

        return True
    except Exception as e:
        print(f"  ✗ Visual metrics failed: {e}")
        return False


def test_wasm_artifacts():
    """Test 6: Are WASM artifacts built?"""
    print("\n[Test 6] Testing WASM artifacts...")
    try:
        pkg_dir = Path("wasm-orbit/pkg")
        files = {
            "orbit_synth_wasm.js": pkg_dir / "orbit_synth_wasm.js",
            "orbit_synth_wasm.wasm": pkg_dir / "orbit_synth_wasm_bg.wasm",
            "orbit_synth_wasm.d.ts": pkg_dir / "orbit_synth_wasm.d.ts",
        }

        all_exist = True
        for name, path in files.items():
            if path.exists():
                size = path.stat().st_size
                print(f"  ✓ {name}: {size:,} bytes")
            else:
                print(f"  ✗ {name}: NOT FOUND")
                all_exist = False

        return all_exist
    except Exception as e:
        print(f"  ✗ WASM check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("FractalSync Runtime-Core Integration Tests")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Feature Extraction", test_feature_extraction),
        ("Orbit Synthesis", test_orbit_synthesis),
        ("Model Initialization", test_model_init),
        ("Visual Metrics", test_visual_metrics),
        ("WASM Artifacts", test_wasm_artifacts),
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
        print("  1. Start backend API: python backend/api/server.py")
        print("  2. Run training: python backend/train.py --data-dir data/audio")
        print("  3. Start frontend: npm --prefix frontend run dev")
        return 0
    else:
        print("\n⚠️  Some tests failed. See details above.")
        if not results.get("Imports"):
            print("\nCritical: runtime_core not available.")
            print(
                "  Fix: Push-Location runtime-core; try { maturin develop --release } finally { Pop-Location }"
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
