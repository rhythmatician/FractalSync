#!/usr/bin/env python3
"""
Build runtime_core Python bindings using maturin.

Handles PATH setup for Rust tools, works around VS Code terminal PATH issues.
"""

import subprocess
import sys
import os
from pathlib import Path


def ensure_venv():
    """Verify Python venv is activated."""
    if not os.environ.get("VIRTUAL_ENV"):
        print("❌ Python venv not activated")
        print("   Run: .venv\\Scripts\\Activate.ps1")
        sys.exit(1)
    print(f"✓ Using venv: {os.environ['VIRTUAL_ENV']}")


def find_cargo():
    """Find cargo binary, adding to PATH if needed."""
    cargo_home = os.path.expanduser("~/.cargo/bin")

    # Try to find cargo in standard location
    if sys.platform == "win32":
        cargo_path = os.path.join(cargo_home, "cargo.exe")
    else:
        cargo_path = os.path.join(cargo_home, "cargo")

    if not os.path.exists(cargo_path):
        print(f"❌ Cargo not found at {cargo_path}")
        print("   Install Rust: https://rustup.rs/")
        sys.exit(1)

    # Add to PATH if not already there
    if cargo_home not in os.environ["PATH"]:
        os.environ["PATH"] = f"{cargo_home}{os.pathsep}{os.environ['PATH']}"

    print(f"✓ Found cargo at {cargo_path}")
    return cargo_path


def check_cargo_version():
    """Verify cargo version."""
    result = subprocess.run(["cargo", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {result.stdout.strip()}")
    else:
        print(f"❌ Failed to run cargo: {result.stderr}")
        sys.exit(1)


def check_rustc_version():
    """Verify rustc version."""
    result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {result.stdout.strip()}")
    else:
        print(f"❌ Failed to run rustc: {result.stderr}")
        sys.exit(1)


def check_maturin():
    """Verify maturin is installed."""
    result = subprocess.run(["maturin", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {result.stdout.strip()}")
    else:
        print("❌ maturin not installed")
        print("   Run: pip install maturin")
        sys.exit(1)


def build_runtime_core():
    """Build runtime_core with maturin."""
    print("\nBuilding runtime_core...")
    print("=" * 70)

    # Change to runtime-core directory
    runtime_core_dir = Path(__file__).parent / "runtime-core"
    os.chdir(runtime_core_dir)

    result = subprocess.run(["maturin", "develop", "--release"], text=True)

    if result.returncode != 0:
        print("=" * 70)
        print("❌ Build failed")
        sys.exit(1)

    print("=" * 70)
    print("✅ Build successful!")


def verify_install():
    """Verify runtime_core can be imported."""
    print("\nVerifying installation...")
    try:
        import runtime_core

        print(f"✓ runtime_core imported successfully")
        print(f"  SAMPLE_RATE: {runtime_core.SAMPLE_RATE}")
        print(f"  HOP_LENGTH: {runtime_core.HOP_LENGTH}")
        print(f"  N_FFT: {runtime_core.N_FFT}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import runtime_core: {e}")
        return False


def main():
    """Build runtime_core with proper setup."""
    print("\n" + "=" * 70)
    print("FractalSync: Build runtime_core Python Bindings")
    print("=" * 70 + "\n")

    ensure_venv()
    find_cargo()

    print("\nChecking tools...")
    check_cargo_version()
    check_rustc_version()
    check_maturin()

    build_runtime_core()

    if verify_install():
        print("\n" + "=" * 70)
        print("✅ SUCCESS! You can now run:")
        print("   python diagnostic.py")
        print("   python test_e2e.py")
        print("=" * 70)
        return 0
    else:
        print("\n⚠ Build completed but verification failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
