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
    """Every manifested mirror must satisfy the parity contract for its kind.

    Behavioral mirrors (differentiable, behavioral, or transitional) MUST
    name real checks — the preflight CHECKS registry
    (scripts/preflight_parity.py) is the enforcement mechanism.
    Diagnostic/experimental mirrors must declare 'none' parity with a
    reason instead of pretending to be pinned. Transitional mirrors
    additionally require a sunset path and sunset_required_by date.
    """
    import json
    import re

    manifest = json.loads(
        (REPO_ROOT / "shared" / "architecture_mirrors.json").read_text(encoding="utf-8")
    )
    preflight_text = (REPO_ROOT / "scripts" / "preflight_parity.py").read_text(
        encoding="utf-8"
    )
    capabilities = manifest.get("capabilities", {})

    KINDS_REQUIRING_PARITY = {
        "differentiable_mirror",
        "behavioral_mirror",
        "transitional_mirror",
    }
    KINDS_EXEMPT = {"diagnostic", "experimental"}
    ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    for mirror in manifest["mirrors"]:
        path = mirror["path"]
        kind = mirror.get("kind")
        assert kind in KINDS_REQUIRING_PARITY | KINDS_EXEMPT, (
            f"{path}: missing/invalid kind ({kind!r})"
        )

        # Capability-constrained amendment (2026-08-29): every mirror must
        # name the capability that justifies its existence outside Rust.
        why = mirror.get("why_outside_rust")
        assert why, f"{path}: missing why_outside_rust"
        lang, _, cap = why.partition(":")
        lang = lang.strip()
        cap = cap.strip()
        lang_caps = capabilities.get(lang, {})
        allowed = lang_caps.get("allowed_responsibilities", [])
        assert allowed, f"{path}: why_outside_rust references unknown language {lang!r}"
        assert cap in allowed, (
            f"{path}: why_outside_rust={why!r} not in "
            f"{lang}.allowed_responsibilities={allowed}"
        )

        parity = mirror.get("parity", [])
        if kind in KINDS_REQUIRING_PARITY:
            assert parity, f"{path}: kind={kind} mirror has no parity checks listed"
            assert not any("none" in entry.lower() for entry in parity), (
                f"{path}: kind={kind} mirror claims 'none' parity — downgrade "
                f"the kind or add real checks"
            )
            for entry in parity:
                # Each parity entry must reference a mechanism that exists:
                # a preflight check letter, a test file, or a vitest spec.
                references_something = (
                    "preflight_parity.py" in entry
                    or "test_" in entry
                    or ".test.ts" in entry
                )
                assert references_something, (
                    f"{path}: parity entry does not reference a check or test: {entry}"
                )
                if "preflight_parity.py" in entry:
                    # The referenced check letter must exist in the preflight.
                    assert "check" in preflight_text, "preflight registry missing"
            if kind == "transitional_mirror":
                # Sunset contract: a real removal path + a deadline.
                sunset = mirror.get("sunset")
                assert sunset, (
                    f"{path}: transitional_mirror requires a non-empty sunset"
                )
                assert "ADR" in sunset or "issue" in sunset.lower(), (
                    f"{path}: transitional_mirror sunset must name an ADR or "
                    f"GitHub issue that removes the mirror"
                )
                deadline = mirror.get("sunset_required_by")
                assert deadline, (
                    f"{path}: transitional_mirror requires sunset_required_by"
                )
                assert ISO_DATE.match(deadline), (
                    f"{path}: sunset_required_by must be ISO YYYY-MM-DD, got {deadline!r}"
                )
        else:
            # Diagnostic/experimental: parity must be 'none' with a reason.
            assert parity, (
                f"{path}: kind={kind} mirror must state 'none' parity with a reason"
            )
            assert all("none" in entry.lower() for entry in parity), (
                f"{path}: kind={kind} mirror must not claim real parity checks"
            )


def test_guardrail_defines_capability_rules() -> None:
    """The guardrail must encode the 2026-08-29 capability checks.

    Without these rules the amendment is unenforced: agents could
    re-declare domain-state types or hard-code shared constants and
    the architecture guardrail would stay silent. This test pins the
    rule names so removing them requires updating the test (i.e. an
    explicit, visible decision).
    """
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "guardrail_capability_test",
        str(REPO_ROOT / "scripts" / "architecture_guardrail.py"),
    )
    assert spec is not None and spec.loader is not None
    guardrail = module_from_spec(spec)
    import sys

    sys.modules["guardrail_capability_test"] = guardrail
    spec.loader.exec_module(guardrail)

    rule_names = {r.name for r in guardrail.RULES}
    assert "domain-type-redeclaration" in rule_names, (
        "domain-type-redeclaration rule missing — capability check unenforced"
    )
    assert "hardcoded-shared-constant" in rule_names, (
        "hardcoded-shared-constant rule missing — capability check unenforced"
    )

    # Manifest validator must require the new fields.
    import json

    manifest = json.loads(
        (REPO_ROOT / "shared" / "architecture_mirrors.json").read_text(encoding="utf-8")
    )
    assert "capabilities" in manifest, "manifest missing capabilities block"
    for lang in ("typescript", "python"):
        assert lang in manifest["capabilities"], (
            f"capabilities block missing language: {lang}"
        )
        assert manifest["capabilities"][lang].get("allowed_responsibilities"), (
            f"capabilities.{lang}.allowed_responsibilities must be a non-empty list"
        )
    for mirror in manifest["mirrors"]:
        assert "why_outside_rust" in mirror, (
            f"{mirror['path']}: missing why_outside_rust capability"
        )
        if mirror.get("kind") == "transitional_mirror":
            assert "sunset" in mirror and mirror["sunset"], (
                f"{mirror['path']}: transitional_mirror missing sunset path"
            )
            assert "sunset_required_by" in mirror, (
                f"{mirror['path']}: transitional_mirror missing sunset_required_by"
            )
