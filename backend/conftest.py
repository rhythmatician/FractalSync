"""Pytest fixtures for backend tests.

Includes a session-scoped `runtime_core_module` fixture that will build
and install the `runtime-core` PyO3 extension with `maturin develop` when
it is not already importable. This keeps developer workflow simple while
allowing CI to pre-build or rely on the fixture.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def runtime_core_module():
    """Ensure `runtime_core` is importable, building it with maturin if needed.

    Yields the imported `runtime_core` module. If `maturin` is not installed
    this will raise a helpful RuntimeError instructing the developer how to
    install it (recommended: `pip install -r backend/requirements-dev.txt`).
    """
    # Try importing runtime_core; if present we may still rebuild if the source changed
    installed = False
    try:
        import runtime_core  # type: ignore

        installed = True
    except Exception:
        installed = False

    # Ensure maturin is available (don't import the module just to avoid unused-import warnings)
    from importlib import util as _importlib_util

    if _importlib_util.find_spec("maturin") is None:
        raise RuntimeError(
            "maturin is required to build runtime-core; install it via `pip install -r backend/requirements-dev.txt`"
        )

    # Determine current runtime-core commit (if available)
    def _current_runtime_core_commit() -> str | None:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT / "runtime-core",
                stderr=subprocess.DEVNULL,
            )
            return out.decode().strip()
        except Exception:
            return None

    commit = _current_runtime_core_commit()
    marker = ROOT / "runtime-core" / "target" / ".last_built_commit"

    need_build = True
    if commit is not None and marker.exists():
        try:
            if marker.read_text().strip() == commit:
                need_build = False
        except Exception:
            need_build = True

    if not need_build and installed:
        # Already built for this commit and importable: use it
        yield runtime_core
        return

    # Prefer using the maturin Python API when available; it provides
    # better error objects and avoids invoking a subprocess when possible.
    # Prepare output directories required by the API
    wheel_dir = str((ROOT / "runtime-core" / "target" / "wheels").absolute())
    sdist_dir = str((ROOT / "runtime-core" / "target" / "sdist").absolute())
    (ROOT / "runtime-core" / "target" / "wheels").mkdir(parents=True, exist_ok=True)
    (ROOT / "runtime-core" / "target" / "sdist").mkdir(parents=True, exist_ok=True)

    try:
        import maturin

        maturin.build_editable(wheel_directory=wheel_dir)

        maturin.build_sdist(sdist_directory=sdist_dir)
        maturin.build_wheel(wheel_directory=wheel_dir)

    except Exception:
        # Fall back to the CLI which is reliable across maturin releases.
        cmd = [sys.executable, "-m", "maturin", "develop", "--release"]
        subprocess.run(cmd, cwd=ROOT / "runtime-core", check=True)

    # Update the build marker so future test sessions can skip rebuilding
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(commit or "unknown")
    except Exception:
        # Non-fatal: marker update is best-effort
        pass

    # Import the freshly-built extension
    import importlib

    importlib.invalidate_caches()
    import runtime_core

    yield runtime_core
