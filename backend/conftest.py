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
    # Try importing runtime_core; if present we will prefer a local-source build
    installed = False
    runtime_core_mod = None
    try:
        import runtime_core  # type: ignore

        runtime_core_mod = runtime_core
        installed = True
        # If the installed runtime_core is not the one built from our local source
        # (e.g., it's a site-packages wheel), force a rebuild so tests exercise
        # the current repository source.
        try:
            module_file = getattr(runtime_core_mod, "__file__", "")
            if module_file:
                module_path = Path(module_file).resolve()
                # If the installed module's path is not inside our repo's runtime-core dir, rebuild
                if str((ROOT / "runtime-core").resolve()) not in str(module_path):
                    installed = False
        except Exception:
            # Be conservative and rebuild if we can't determine origin
            installed = False
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
        import os

        rc_dir = ROOT / "runtime-core"
        pyproj = rc_dir / "pyproject.toml"
        if not pyproj.exists():
            raise RuntimeError(
                f"runtime-core pyproject.toml not found at expected location: {pyproj}"
            )

        # maturin's API expects to be called from the project dir so it can
        # locate `pyproject.toml`. To use debugpy, we temporarily chdir into
        # the runtime-core directory for the API invocation.
        try:
            old_cwd = Path.cwd()
            os.chdir(rc_dir)
            maturin.build_editable(wheel_directory=wheel_dir)
            maturin.build_sdist(sdist_directory=sdist_dir)
            maturin.build_wheel(wheel_directory=wheel_dir)

            # maturin's build API builds artifacts but does not always
            # perform an install into the current environment. If we built
            # a wheel, install the most recent wheel so the extension is
            # importable immediately.
            from pathlib import Path as _Path

            wheel_path_dir = _Path(wheel_dir)
            wheels = sorted(
                wheel_path_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime
            )
            if wheels:
                latest = wheels[-1]
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--force-reinstall",
                        str(latest),
                    ],
                    check=True,
                )
        finally:
            os.chdir(old_cwd)
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

    # Workaround: a stale or broken installed `runtime_core` package directory
    # (e.g., .venv/Lib/site-packages/runtime_core/__init__.py that expects a
    # submodule `runtime_core.runtime_core`) can shadow the compiled extension
    # we just built. Detect such directories and temporarily rename them so
    # the import picks up our built extension. We restore names on failure.
    renamed = []
    try:
        for sp in list(sys.path):
            try:
                sp_path = Path(sp)
            except Exception:
                continue
            pkg_dir = sp_path / "runtime_core"
            if pkg_dir.exists() and pkg_dir.is_dir():
                # If the package directory does not contain a compiled extension
                # file (pyd/so/dll), it's likely to be the culprit.
                has_ext = (
                    any(pkg_dir.glob("*.pyd"))
                    or any(pkg_dir.glob("*.so"))
                    or any(pkg_dir.glob("*.dll"))
                )
                if not has_ext:
                    bak = pkg_dir.with_name(pkg_dir.name + ".disabled_by_tests")
                    try:
                        pkg_dir.rename(bak)
                        renamed.append((pkg_dir, bak))
                        print(
                            f"[runtime_core_module] renamed conflicting installed package {pkg_dir} -> {bak}",
                            file=sys.stderr,
                        )
                    except Exception as e:
                        print(
                            f"[runtime_core_module] failed to rename {pkg_dir}: {e}",
                            file=sys.stderr,
                        )

        import runtime_core
    except Exception:
        # On failure, attempt to restore any renamed package directories and
        # re-raise the original exception so the test failure is visible.
        for orig, bak in renamed:
            try:
                if bak.exists():
                    bak.rename(orig)
            except Exception:
                pass
        raise

    # Successful import; yield the module for tests.
    yield runtime_core
