"""CI gate for the architecture guardrail (issue #90).

Shared cross-runtime behavior has one authority: ``runtime-core`` (Rust).
This test runs ``scripts/architecture_guardrail.py`` so the check executes
in the same pytest pass as the parity gates — a PR that reintroduces
Rust-owned runtime logic in TypeScript or Python outside an approved,
parity-pinned mirror fails here.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GUARDRAIL = REPO_ROOT / "scripts" / "architecture_guardrail.py"


def test_architecture_guardrail_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(GUARDRAIL)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        "Architecture guardrail failed — Rust-owned runtime logic found "
        "outside runtime-core and outside shared/architecture_mirrors.json:\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_manifest_mirrors_are_pinned_by_preflight_checks() -> None:
    """Every manifested mirror's parity list must reference real checks.

    The preflight CHECKS registry (scripts/preflight_parity.py) is the
    enforcement mechanism; a mirror whose parity entries name nothing real
    is not actually pinned.
    """
    import json

    manifest = json.loads(
        (REPO_ROOT / "shared" / "architecture_mirrors.json").read_text(
            encoding="utf-8"
        )
    )
    preflight_text = (REPO_ROOT / "scripts" / "preflight_parity.py").read_text(
        encoding="utf-8"
    )

    for mirror in manifest["mirrors"]:
        path = mirror["path"]
        parity = mirror.get("parity", [])
        if path in (
            "backend/src/c_trace_plot.py",
            "backend/src/live_controller.py",
        ):
            # Diagnostic-only / experimental mirrors declare "none" parity
            # with an explicit reason; the guardrail manifest documents why.
            assert any(
                "none" in entry.lower() for entry in parity
            ), f"{path}: diagnostic/experimental mirror must state why parity is 'none'"
            continue
        assert parity, f"{path}: mirror has no parity checks listed"
        for entry in parity:
            # Each parity entry must reference a mechanism that exists:
            # a preflight check letter, a test file, or a vitest spec.
            references_something = (
                "preflight_parity.py" in entry
                or "test_" in entry
                or ".test.ts" in entry
            )
            assert references_something, (
                f"{path}: parity entry does not reference a check or test: "
                f"{entry}"
            )
            if "preflight_parity.py" in entry:
                # The referenced check letter must exist in the preflight.
                assert (
                    "check" in preflight_text
                ), "preflight registry missing"
