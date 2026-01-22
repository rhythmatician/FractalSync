#!/usr/bin/env python3
"""
Quick diagnostic to check what's working and what needs fixing.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

print("\n" + "=" * 70)
print("DIAGNOSTIC: FractalSync Runtime Status")
print("=" * 70)

# Test 1: Check if runtime_core is installed
print("\n[1] Checking runtime_core installation...")
try:
    import runtime_core

    print(f"  ✓ runtime_core imported")
    print(f"    SAMPLE_RATE: {runtime_core.SAMPLE_RATE}")
    print(f"    HOP_LENGTH: {runtime_core.HOP_LENGTH}")
    runtime_core_available = True
except ImportError as e:
    print(f"  ✗ runtime_core NOT available: {e}")
    runtime_core_available = False

# Test 2: Check WASM build
print("\n[2] Checking WASM build...")
wasm_pkg_path = "wasm-orbit/pkg"
if os.path.exists(wasm_pkg_path):
    files = os.listdir(wasm_pkg_path)
    if any(f.endswith(".wasm") for f in files):
        print(f"  ✓ WASM binary found in {wasm_pkg_path}")
        print(f"    Files: {', '.join(sorted(files)[:5])}")
    else:
        print(f"  ✗ No .wasm file in {wasm_pkg_path}")
else:
    print(f"  ✗ {wasm_pkg_path} does not exist")

# Test 3: Backend modules (without runtime_core)
print("\n[3] Checking backend modules (independent of runtime_core)...")
try:
    from src.control_model import AudioToControlModel  # noqa: F401

    print(f"  ✓ control_model.AudioToControlModel imports")
except Exception as e:
    print(f"  ✗ control_model failed: {e}")

try:
    from src.visual_metrics import VisualMetrics  # noqa: F401

    print(f"  ✓ visual_metrics.VisualMetrics imports (class: {VisualMetrics.__name__})")
except Exception as e:
    print(f"  ✗ visual_metrics failed: {e}")

try:
    import torch  # noqa: F401

    print(f"  ✓ torch available")
except Exception as e:
    print(f"  ✗ torch not available: {e}")

try:
    import cv2  # noqa: F401

    print(f"  ✓ cv2 (OpenCV) available")
except Exception as e:
    print(f"  ✗ cv2 not available: {e}")

# Test 4: If runtime_core is available, test bridge
if runtime_core_available:
    print("\n[4] Testing runtime_core_bridge...")
    try:
        from src.runtime_core_bridge import (
            make_feature_extractor,
            make_orbit_state,
            synthesize,
        )

        print(f"  ✓ runtime_core_bridge imports work")

        # Test feature extraction
        extractor = make_feature_extractor()
        print(f"  ✓ Feature extractor created")

        # Test orbit synthesis
        state = make_orbit_state(seed=1337)
        print(f"  ✓ Orbit state created")

        c = synthesize(state)
        print(f"  ✓ Synthesis works: c = {c.real:.4f} + {c.imag:.4f}i")

    except Exception as e:
        print(f"  ✗ Bridge test failed: {e}")
        import traceback

        traceback.print_exc()
else:
    print("\n[4] Skipping runtime_core_bridge tests (runtime_core not available)")
    print("    Run: cd runtime-core && maturin develop --release")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)

if not runtime_core_available:
    print("1. Build runtime_core:")
    print("   $ rustup update  # Make sure Rust is up to date")
    print("   $ cd runtime-core")
    print("   $ maturin develop --release")
    print("")

print("2. Run full test suite:")
print("   $ python test_e2e.py")
print("")
print("3. Start backend API server:")
print("   $ cd backend")
print("   $ python api/server.py")
print("")
print("4. Start frontend dev server:")
print("   $ cd frontend")
print("   $ npm run dev")

print("=" * 70)
