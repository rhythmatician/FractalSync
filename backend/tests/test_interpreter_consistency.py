"""Interpreter-consistency guard (multi-interpreter pitfall).

This repo commonly has TWO Python interpreters: a base install
(e.g. ``C:\\Python314``) and the project ``.venv``, each with its own
``runtime_core`` wheel install. Any test or script that shells out to bare
``python`` (PATH resolution) can silently run under the OTHER interpreter
with a STALE wheel — the test then validates bindings pytest never
imported, and parity checks pass against the wrong runtime_core.

This guard makes that an invalid repository state:

  1. No backend test may invoke bare ``python`` via subprocess or
     ``shell=True`` strings; subprocesses must reuse ``sys.executable``.
  2. The pytest interpreter must be able to import ``runtime_core`` from
     the same environment pytest itself runs in (no cross-env leakage).

CI is unaffected structurally (it installs into one interpreter), but this
guard keeps local Windows dev honest — where the two-interpreter setup is
the norm.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent

# Patterns that indicate a subprocess would resolve `python` from PATH
# instead of reusing the running interpreter.
BARE_PYTHON_PATTERNS = [
    # subprocess.run("python ...", shell=True) style strings
    r"""["']python(?:3(?:\.\d+)?)?\s""",
    # subprocess.run(["python", ...]) list forms
    r"""=\s*\[\s*["']python(?:3(?:\.\d+)?)?["']\s*,\s*["']""",
]

# Legitimate uses of the literal word `python` that are NOT interpreter
# invocations (docstrings, comments, module names, shebangs, etc.).
SAFE_CONTEXTS = [
    "sys.executable",
    "python_feature_extractor",
    "runtime_core_helpers",
]


def _is_bare_python_invocation(line: str) -> bool:
    """Heuristic: does this line spawn a bare `python` interpreter?"""
    stripped = line.strip()
    if stripped.startswith("#") or stripped.startswith('"""'):
        return False
    for pattern in BARE_PYTHON_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


class TestNoBarePythonSubprocess:
    """Backend tests must reuse sys.executable, never PATH python."""

    def test_no_bare_python_in_test_files(self) -> None:
        offenders: list[str] = []
        for test_file in sorted(TESTS_DIR.glob("test_*.py")):
            text = test_file.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if _is_bare_python_invocation(line):
                    offenders.append(f"{test_file.name}:{lineno}: {line.strip()}")
        assert not offenders, (
            "Tests spawning bare `python` found — these resolve from PATH "
            "and can run under a DIFFERENT interpreter with a STALE "
            "runtime_core wheel (the multi-interpreter pitfall). Use "
            "sys.executable instead:\n" + "\n".join(offenders)
        )

    def test_runtime_core_importable_from_pytest_interpreter(self):
        """The wheel pytest sees must be importable and self-consistent."""
        import runtime_core

        # Basic contract: the constants the parity system depends on exist.
        assert hasattr(runtime_core, "SAMPLE_RATE")
        assert hasattr(runtime_core, "HOP_LENGTH")
        assert hasattr(runtime_core, "FEATURE_VERSION")
        # The canonical timebase must be present (post-#93 contract).
        assert hasattr(runtime_core, "AnalysisTimebase"), (
            "runtime_core is missing AnalysisTimebase — the pytest "
            "interpreter has a stale wheel. Rebuild and reinstall: "
            "maturin build --release, then pip install --force-reinstall "
            "--no-deps into THIS interpreter."
        )

    def test_pytest_interpreter_matches_venv_when_present(self):
        """If a repo .venv exists, pytest should be running from it (or the
        developer must consciously opt out via env var)."""
        venv_python = BACKEND_DIR.parent / ".venv" / "Scripts" / "python.exe"
        venv_python_posix = BACKEND_DIR.parent / ".venv" / "bin" / "python"
        if not (venv_python.exists() or venv_python_posix.exists()):
            pytest_skip = True  # no venv; nothing to check
        else:
            pytest_skip = False
        if pytest_skip:
            return

        import os

        if os.environ.get("ALLOW_NON_VENV_PYTHON"):
            return

        exe = Path(sys.executable).resolve()
        venv_dir = (BACKEND_DIR.parent / ".venv").resolve()
        assert venv_dir in exe.parents or exe.parent == venv_dir, (
            f"pytest is running under {exe} but the repo has a .venv at "
            f"{venv_dir}. The two may hold different runtime_core wheels, "
            "which is exactly the multi-interpreter parity trap. Run tests "
            "via .venv\\Scripts\\python.exe -m pytest, or set "
            "ALLOW_NON_VENV_PYTHON=1 to override consciously."
        )
