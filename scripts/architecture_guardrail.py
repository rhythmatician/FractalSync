"""Architecture guardrail: runtime-core is the single authority (issue #90).

Shared cross-runtime behavior (c-space/Mandelbrot geometry, controller and
motion synthesis, minimap/Shore physics, canonical audio feature formulas,
deterministic synthesis primitives) must be implemented in ``runtime-core``
(Rust). TypeScript and Python consume bindings; intentional mirrors are
allowed only through an explicit entry in ``shared/architecture_mirrors.json``
pinned by parity checks.

This script is a deterministic, cheap, text-based check. It guards
architectural concepts and named APIs with targeted evidence — not ordinary
math tokens — so it stays low-noise and hard to bypass accidentally.

Usage:
    python scripts/architecture_guardrail.py            # exit 0/1
    python scripts/architecture_guardrail.py --verbose  # list scanned files

Run in CI via backend/tests/test_architecture_guardrail.py and directly in
.github/workflows/pytest.yml.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "shared" / "architecture_mirrors.json"

# Directories scanned for violations (Rust crate and build output excluded).
SCAN_DIRS = ["frontend/src", "backend/src", "backend/api", "scripts", "shared"]
SCAN_SUFFIXES = {".ts", ".tsx", ".py"}

# Files always excluded from scanning: the guardrail itself (its rule
# definitions quote the patterns), the parity preflight (it *is* the parity
# enforcement), binding type declarations (wasm .d.ts files are generated
# declarations of the Rust API, not implementations), and the distance-field
# builder (the offline tool that bakes the field the Rust runtime samples).
ALWAYS_EXCLUDE = {
    "shared/architecture_mirrors.json",
    "scripts/architecture_guardrail.py",
    "scripts/preflight_parity.py",
    "scripts/build_distance_field/__init__.py",
}

# Suffixes that are binding declarations (generated from the Rust API).
DECLARATION_SUFFIXES = {".d.ts"}


@dataclass
class Rule:
    """One targeted-evidence rule.

    ``pattern`` is matched against file text (case-sensitive). ``evidence``
    requires ALL of its substrings on (roughly) the same logical construct —
    implemented as: every evidence string appears within a small window of
    the pattern match, or in the same file when ``file_level`` is set.
    ``skip_comments`` ignores matches on comment-only lines (prose may
    legitimately reference a Rust API; code may not re-implement it).
    ``skip_imports`` ignores matches on import lines (consuming a binding
    type via ``import`` or Python ``from … import …`` is not a
    re-declaration).
    """

    name: str
    pattern: str
    evidence: list[str] = field(default_factory=list)
    file_level: bool = False
    skip_comments: bool = False
    skip_imports: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Rules: each targets a *named concept* of the Rust-owned runtime, requiring
# corroborating evidence so ordinary math (e.g. a learning-rate sqrt) never
# trips it.
# ---------------------------------------------------------------------------

RULES: list[Rule] = [
    # 1. Competing controller implementations (not wrappers).
    Rule(
        name="controller-class",
        pattern=r"\bclass\s+\w*(OrbitController|OrbitSynthesizer|PlayerState)\w*",
        evidence=["step"],
        file_level=True,
        description=(
            "A class named like the Rust controller with a step method — "
            "outside runtime-core this must be a binding wrapper or a "
            "manifested mirror."
        ),
    ),
    # 2. Mandelbrot/cardioid boundary parameterization: the mu = 1 - sqrt(1-4c)
    #    closed form. Evidence: the literal 1-4c combination together with a
    #    sqrt over it. Guarded via the distinctive '1 - 4' / '1-4' coefficient
    #    AND a sqrt in the same statement neighborhood.
    Rule(
        name="cardioid-parameterization",
        pattern=r"1\.?\d*\s*-\s*4\.?\d*\s*\*\s*\w+",
        evidence=["sqrt"],
        file_level=True,
        description=(
            "Inline cardioid parameterization (mu = 1 - sqrt(1-4c)) outside "
            "an approved mirror. Import the shared helper "
            "(backend/src/cspace_proxies.cardioid_mu) or add a manifest entry."
        ),
    ),
    # 3. Cardioid boundary outline: c = e^{it}/2 - e^{2it}/4 (cos/sin pair).
    Rule(
        name="cardioid-outline",
        pattern=r"cos\(\s*2\.?\d*\s*\*\s*\w+\s*\)\s*/\s*4",
        evidence=["sin("],
        file_level=True,
        description=(
            "Cardioid boundary outline formula (c = e^{it}/2 - e^{2it}/4) "
            "outside an approved mirror."
        ),
    ),
    # 4. Residual epicycle synthesis: amplitude halving per harmonic with a
    #    per-band gate — the signature of controller::synthesize.
    Rule(
        name="residual-epicycles",
        pattern=r"2\.?\d*\s*\*\*\s*\(\s*\w+\s*\+\s*1\.?\d*\s*\)",
        evidence=["residual"],
        file_level=True,
        description=(
            "Residual epicycle amplitude ladder (alpha*s*radius/2^(k+1)) "
            "outside an approved mirror — synthesis semantics belong to "
            "runtime-core."
        ),
    ),
    # 5. Julia/Mandelbrot escape iteration: z = z^2 + c loop with bailout 2.
    #    Evidence: the quadratic update together with a bailout comparison.
    Rule(
        name="escape-iteration",
        pattern=r"\bz(\s*\*\s*z|\s*\*\*\s*2)\s*\+\s*(c|seed)",
        evidence=["2.0", "iter"],
        file_level=True,
        description=(
            "Julia/Mandelbrot escape iteration (z = z^2 + c with bailout) "
            "outside an approved mirror."
        ),
    ),
    # 6. Canonical feature formulas: the features/2 causal transform
    #    log1p(100x)/log1p(100) is a named contract constant.
    Rule(
        name="feature-causal-transform",
        pattern=r"log1p\s*\(\s*100",
        evidence=[],
        file_level=False,
        description=(
            "The features/2 causal fixed transform (log1p(100x)/log1p(100)) "
            "outside an approved mirror — feature formulas belong to "
            "runtime-core/src/features.rs."
        ),
    ),
    # 7. Named future concepts: PhaseTracker / CycleBank must be born in Rust.
    #    Only *implementation-shaped* declarations are flagged (class/def/
    #    function/interface/type). Binding consumers (``rc.PhaseTracker(...)``,
    #    ``import { PhaseTracker } from 'wasm-orbit'``) and wrapper adapters
    #    are legitimate — they consume the Rust authority, they do not
    #    replace it. Manifest adapters are exempt entirely, mirroring the
    #    controller-class rule.
    Rule(
        name="phase-tracker-concept",
        pattern=r"\b(class|def|function|interface|type)\s+\w*(PhaseTracker|CycleBank)\w*",
        evidence=[],
        file_level=False,
        description=(
            "PhaseTracker/CycleBank are planned Rust-owned concepts; a "
            "class/function declaration outside runtime-core must be a "
            "manifested mirror (binding consumers and adapters are fine)."
        ),
    ),
    # 8. Shore/minimap physics: contour-biased step / shore-normal push.
    #    Comment-only mentions are fine (documentation references the API);
    #    only code lines implementing or invoking the physics are flagged.
    Rule(
        name="shore-physics",
        pattern=r"\b(contour_biased_step|shore_normal|music_push|MUSIC_PUSH_GAIN)\b",
        evidence=[],
        file_level=False,
        skip_comments=True,
        description=(
            "Shore/minimap movement physics named API outside an approved "
            "mirror — the contour step lives in runtime-core/src/minimap.rs."
        ),
    ),
    # 9. Domain-state type re-declaration (capability-constrained amendment,
    #    2026-08-29, refined issue #93 strict review): TypeScript/Python
    #    files outside runtime-core/ that declare a class/interface/type
    #    whose name matches a Rust-exported domain-state type, AND that
    #    contains a structural-field token, are re-declaring canonical
    #    structure instead of consuming the binding. Evidence: the
    #    Rust-original snake_case field name AND the camelCase wire form
    #    BOTH appearing in the file (i.e. the file is hand-maintaining the
    #    shape rather than importing it from the generated .d.ts or the
    #    PyO3 binding). Per-type rules below — see ``DOMAIN_TYPE_EVIDENCE``
    #    for the field-token table. Binding declarations (*.d.ts) and
    #    approved adapters are exempt (they're the legitimate surface for
    #    type re-export / consumption).
    Rule(
        name="domain-type-redeclaration-OrbitState",
        pattern=r"\b(class|interface|type)\s+(OrbitState)\b",
        evidence=["theta", "omega", "lobe"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``OrbitState`` outside runtime-core/ "
            "— import the wasm-orbit binding type or the PyO3 stub "
            "instead of restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-PlayerState",
        pattern=r"\b(class|interface|type)\s+(PlayerState)\b",
        evidence=["c_re", "v_re", "theta"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``PlayerState`` outside runtime-core/ "
            "— import the binding type instead of restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-PlayerObservation",
        pattern=r"\b(class|interface|type)\s+(PlayerObservation)\b",
        evidence=["c_re", "c_im", "alpha"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``PlayerObservation`` outside "
            "runtime-core/ — import the binding type instead of "
            "restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-CycleHypothesis",
        pattern=r"\b(class|interface|type)\s+(CycleHypothesis)\b",
        evidence=["phase", "period", "confidence"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``CycleHypothesis`` outside "
            "runtime-core/ — import the binding type instead of "
            "restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-CycleBank",
        pattern=r"\b(class|interface|type)\s+(CycleBank)\b",
        evidence=["cycle", "bank", "phase"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``CycleBank`` outside runtime-core/ "
            "— import the binding type instead of restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-CycleMode",
        pattern=r"\b(class|interface|type)\s+(CycleMode)\b",
        evidence=["frequencyHz", "phase", "confidence"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``CycleMode`` outside runtime-core/ "
            "— import the wasm-generated ``CycleMode`` interface or the "
            "PyO3 stub instead of restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-CycleRelation",
        pattern=r"\b(class|interface|type)\s+(CycleRelation)\b",
        evidence=["generalizedPhase", "phaseStability", "freqResidual"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``CycleRelation`` outside runtime-core/ "
            "— import the binding type instead of restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-CycleObservation",
        pattern=r"\b(class|interface|type)\s+(CycleObservation)\b",
        evidence=["sampleIndex", "dtSeconds", "streamEpoch"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``CycleObservation`` outside "
            "runtime-core/ — build it via the canonical seam binding "
            "instead of restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-AnalysisTick",
        pattern=r"\b(class|interface|type)\s+(AnalysisTick)\b",
        evidence=["sampleIndex", "dtSeconds", "streamEpoch"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``AnalysisTick`` outside runtime-core/ "
            "— import the binding type from the wasm-orbit generated "
            "``.d.ts`` (via ``typescript_custom_section``) instead of "
            "hand-maintaining the camelCase interface."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-TimebaseDiagnostics",
        pattern=r"\b(class|interface|type)\s+(TimebaseDiagnostics)\b",
        evidence=["canonicalSampleIndex", "detectedGaps", "streamEpoch"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``TimebaseDiagnostics`` outside "
            "runtime-core/ — import the binding type from the wasm-orbit "
            "generated ``.d.ts`` instead of hand-maintaining the "
            "camelCase interface."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-ResidualParams",
        pattern=r"\b(class|interface|type)\s+(ResidualParams)\b",
        evidence=["k_residuals", "residual_cap", "radius_scale"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``ResidualParams`` outside "
            "runtime-core/ — import the binding type instead of "
            "restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-OrbitController",
        pattern=r"\b(class|interface|type)\s+(OrbitController)\b",
        evidence=["band_gates", "step(", "apply_controls"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``OrbitController`` outside "
            "runtime-core/ — import the binding type instead of "
            "restating its fields/methods."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-OrbitSynthesizer",
        pattern=r"\b(class|interface|type)\s+(OrbitSynthesizer)\b",
        evidence=["synthesize", "k_residuals", "residual_cap"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``OrbitSynthesizer`` outside "
            "runtime-core/ — import the binding type instead of "
            "restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-PhaseTracker",
        pattern=r"\b(class|interface|type)\s+(PhaseTracker)\b",
        evidence=["phase", "period", "tracker"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``PhaseTracker`` (planned) outside "
            "runtime-core/ — declare a binding consumer, not a fresh "
            "structural definition."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-FeatureExtractor",
        pattern=r"\b(class|interface|type)\s+(FeatureExtractor)\b",
        evidence=["n_fft", "hop_length", "feature_mean"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``FeatureExtractor`` outside "
            "runtime-core/ — import the binding type instead of "
            "restating its fields."
        ),
    ),
    Rule(
        name="domain-type-redeclaration-AnalysisTimebase",
        pattern=r"\b(class|interface|type)\s+(AnalysisTimebase)\b",
        evidence=["ingest", "flush", "diagnostics"],
        skip_imports=True,
        file_level=True,
        description=(
            "Re-declaration of Rust ``AnalysisTimebase`` outside "
            "runtime-core/ — import the binding type from the wasm-orbit "
            "generated ``.d.ts`` instead of hand-maintaining the class."
        ),
    ),
    # 10. Hard-coded shared constants (capability-constrained amendment,
    #     2026-08-29): magic-number copies of canonical constants in
    #     non-binding, non-manifested code. Bindings (wasm .d.ts,
    #     runtime-core-bridge-style files) are exempt because exposing
    #     the constant IS the binding. Pattern: an assignment statement
    #     whose LHS matches a canonical constant name; right-hand side
    #     contains the canonical literal value (48000 for SAMPLE_RATE,
    #     NORM_EPS epsilon-like 1e-5, etc).
    #     Extended (issue #93 review): also covers the TS constant names
    #     (RUNTIME_SAMPLE_RATE / HOP_LENGTH / CANONICAL_SAMPLE_RATE /
    #     CANONICAL_HOP_LENGTH) and the pipeline-version string — the
    #     browser must read constants from wasm.constants() and Python
    #     must ALIAS runtime_core values, never restate them.
    Rule(
        name="hardcoded-shared-constant",
        pattern=(
            r"\b(SAMPLE_RATE|WINDOW_FRAMES|K_RESIDUALS|NORM_EPS|"
            r"FEATURE_VERSION|CONTROLLER_VERSION|ANALYSIS_PIPELINE_VERSION|"
            r"RUNTIME_SAMPLE_RATE|CANONICAL_SAMPLE_RATE|CANONICAL_HOP_LENGTH)\s*[=:]\s*"
            r"(48000|48000\.0|6|1e-5|0\.00001|\d+)"
        ),
        evidence=[],
        file_level=False,
        description=(
            "Hard-coded canonical constant outside runtime-core/ and "
            "outside an approved mirror/binding. Read it from the Rust "
            "binding (runtime_core.SAMPLE_RATE, wasm constants(), etc.) "
            "instead of restating the value."
        ),
    ),
    # 10b. HOP_LENGTH restatement: the TS name collides with the Rust
    #     binding name, so the numeric rule above cannot distinguish a
    #     binding re-export from a literal. Flag any `HOP_LENGTH = <int>`
    #     assignment outside runtime-core; bindings re-export via
    #     `constants()` objects, not named const assignments.
    Rule(
        name="hardcoded-hop-length",
        pattern=r"\b(?:const|let|var)?\s*HOP_LENGTH\s*[=:]\s*\d+",
        evidence=[],
        file_level=False,
        description=(
            "Hard-coded HOP_LENGTH outside runtime-core/. Read the hop "
            "length from the wasm binding (constants().hop_length) — a "
            "TypeScript literal is a second authority that can drift from "
            "runtime-core."
        ),
    ),
    # 10c. Pipeline-version string restatement: the analysis pipeline
    #     version is Rust-owned ("analysis/1"); Python/TS must alias the
    #     binding value, never declare their own version string.
    Rule(
        name="hardcoded-pipeline-version",
        pattern=r"""PIPELINE_VERSION\s*[=:]\s*["'](?:timebase|analysis)/""",
        evidence=[],
        file_level=False,
        description=(
            "Hard-coded analysis pipeline version string. Alias the Rust "
            "constant (runtime_core.ANALYSIS_PIPELINE_VERSION / wasm "
            "constants().analysis_pipeline_version) instead of restating "
            "a version literal that can silently drift."
        ),
    ),
]

# Python AST refinement for the cardioid rule: only flag when the sqrt is
# applied to the same expression containing (1 - 4*x). Implemented below via
# a regex over the matched line's neighborhood instead of a full AST walk to
# keep the check dependency-free and identical across TS and Python.


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix in SCAN_SUFFIXES and p.is_file():
                files.append(p)
    return files


def _rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix().replace("\\", "/")


def _is_comment_line(line: str, suffix: str) -> bool:
    """True when the line is comment-only for the given language."""
    stripped = line.strip()
    if suffix in {".py"}:
        return stripped.startswith("#")
    return (
        stripped.startswith("//")
        or stripped.startswith("/*")
        or stripped.startswith("*")
    )


def _is_import_line(line: str, suffix: str) -> bool:
    """True when the line is an import (Python or TS) for the given suffix.

    Used by ``Rule.skip_imports`` to ignore ``import type { Foo } from '...'``
    and ``from … import Foo`` — those consume a binding type, they don't
    re-declare it. A multi-line ``from … import (\\n  Foo,\\n  Bar,\\n)`` is
    not fully handled here, but the pattern syntax check is per-line so
    each item is its own line and would still be matched — keep imports
    to single lines for canonical types if they need to be exempt.
    """
    stripped = line.strip()
    if suffix in {".py"}:
        return stripped.startswith("import ") or stripped.startswith("from ")
    return (
        stripped.startswith("import ")
        or stripped.startswith("import type ")
        or stripped.startswith("import {")
        or stripped.startswith("import type {")
        or stripped.startswith("export type ")
    )


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"manifest missing: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_paths(manifest: dict) -> set[str]:
    return {m["path"] for m in manifest.get("mirrors", [])}


def _check_file(path: Path, text: str, manifest: dict) -> list[str]:
    rel = _rel(path)
    if rel in ALWAYS_EXCLUDE:
        return []
    if path.name.endswith(".d.ts"):
        return []  # generated binding declarations, not implementations
    is_manifested = rel in _manifest_paths(manifest)
    is_adapter = rel in {a.replace("\\", "/") for a in manifest.get("adapters", [])}
    is_transitional = rel in {
        m["path"]
        for m in manifest.get("mirrors", [])
        if m.get("kind") == "transitional_mirror"
    }
    violations: list[str] = []
    lines = text.splitlines()
    for rule in RULES:
        rx = re.compile(rule.pattern)
        for i, line in enumerate(lines):
            if not rx.search(line):
                continue
            if rule.skip_comments and _is_comment_line(line, path.suffix):
                continue
            if rule.skip_imports and _is_import_line(line, path.suffix):
                continue
            if rule.file_level and rule.evidence:
                # Evidence must appear elsewhere in the file (same construct
                # family). If not present, the pattern alone is too generic —
                # skip.
                if not all(ev in text for ev in rule.evidence):
                    continue
            if is_manifested and not is_transitional:
                # Stable mirrors are allowed; the manifest's parity list
                # is the enforcement contract.
                continue
            if is_transitional:
                # Transitional mirrors are tracked debt. They are exempt
                # from the formula rules; their parity + sunset fields are
                # the enforcement contract instead.
                continue
            if is_adapter and rule.name in {
                "controller-class",
                "phase-tracker-concept",
                "domain-type-redeclaration-OrbitState",
                "domain-type-redeclaration-PlayerState",
                "domain-type-redeclaration-PlayerObservation",
                "domain-type-redeclaration-CycleHypothesis",
                "domain-type-redeclaration-CycleBank",
                "domain-type-redeclaration-CycleMode",
                "domain-type-redeclaration-CycleRelation",
                "domain-type-redeclaration-CycleObservation",
                "domain-type-redeclaration-AnalysisTick",
                "domain-type-redeclaration-TimebaseDiagnostics",
                "domain-type-redeclaration-ResidualParams",
                "domain-type-redeclaration-OrbitController",
                "domain-type-redeclaration-OrbitSynthesizer",
                "domain-type-redeclaration-PhaseTracker",
                "domain-type-redeclaration-FeatureExtractor",
                "domain-type-redeclaration-AnalysisTimebase",
            }:
                # Adapters may wrap the Rust controller class or declare
                # binding-consumer types for planned Rust concepts — that is
                # their whole job. They are still scanned for formula rules
                # and for hardcoded-shared-constant (so an adapter cannot
                # silently restate SAMPLE_RATE).
                continue
            violations.append(
                f"{rel}:{i + 1}: [{rule.name}] {rule.description}\n"
                f"    {line.strip()[:160]}"
            )
            break  # one report per rule per file
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="list scanned files")
    args = parser.parse_args(argv)

    manifest = _load_manifest()
    manifested = _manifest_paths(manifest)

    # Manifest sanity: every listed mirror must exist, declare a valid kind,
    # and satisfy the parity contract for that kind. Behavioral mirrors
    # (differentiable, behavioral, or transitional) MUST name real parity
    # checks; diagnostic and experimental entries must NOT pretend to have
    # them — they state "none" with a reason instead, so "parity required"
    # is actually enforced.
    #
    # Transitional mirrors add a sunset contract: parity is real (no
    # 'none') AND a non-empty sunset field naming the follow-up ADR/issue
    # AND an ISO `sunset_required_by` date. Transitional mirrors are also
    # exempt from the formula rules (their debt is the whole point); the
    # sunset path is what removes them.
    MIRROR_KINDS_REQUIRING_PARITY = {
        "differentiable_mirror",
        "behavioral_mirror",
        "transitional_mirror",
    }
    MIRROR_KINDS_EXEMPT = {"diagnostic", "experimental"}
    capabilities = manifest.get("capabilities", {})
    problems: list[str] = []
    for m in manifest.get("mirrors", []):
        if not (REPO_ROOT / m["path"]).exists():
            problems.append(f"manifest mirror path does not exist: {m['path']}")
        if not m.get("rust_authority"):
            problems.append(f"manifest mirror missing rust_authority: {m['path']}")
        kind = m.get("kind")
        if kind not in MIRROR_KINDS_REQUIRING_PARITY | MIRROR_KINDS_EXEMPT:
            problems.append(
                f"manifest mirror missing/invalid kind (expected one of "
                f"{sorted(MIRROR_KINDS_REQUIRING_PARITY | MIRROR_KINDS_EXEMPT)}): "
                f"{m['path']}"
            )
            continue
        # Capability-constrained amendment (2026-08-29): every mirror must
        # declare which capability justifies its existence outside Rust.
        why = m.get("why_outside_rust")
        if not why:
            problems.append(
                f"manifest mirror missing why_outside_rust capability: {m['path']}"
            )
        else:
            lang, _, cap = why.partition(":")
            lang = lang.strip()
            cap = cap.strip()
            lang_caps = capabilities.get(lang, {})
            allowed = lang_caps.get("allowed_responsibilities", [])
            if not allowed:
                problems.append(
                    f"manifest mirror why_outside_rust references unknown "
                    f"language {lang!r}: {m['path']}"
                )
            elif cap not in allowed:
                problems.append(
                    f"manifest mirror why_outside_rust={why!r} is not in "
                    f"{lang}.allowed_responsibilities={allowed}: {m['path']}"
                )
        parity = m.get("parity", [])
        if kind in MIRROR_KINDS_REQUIRING_PARITY:
            if not parity:
                problems.append(
                    f"manifest mirror kind={kind} requires a nonempty parity "
                    f"list: {m['path']}"
                )
            elif any("none" in entry.lower() for entry in parity):
                problems.append(
                    f"manifest mirror kind={kind} claims 'none' parity — "
                    f"downgrade the kind or add real checks: {m['path']}"
                )
            if kind == "transitional_mirror":
                if not m.get("sunset"):
                    problems.append(
                        f"manifest mirror kind=transitional_mirror requires "
                        f"a non-empty sunset field naming the follow-up ADR/"
                        f"issue: {m['path']}"
                    )
                if not m.get("sunset_required_by"):
                    problems.append(
                        f"manifest mirror kind=transitional_mirror requires "
                        f"a sunset_required_by ISO date: {m['path']}"
                    )
        else:
            if parity and not all("none" in entry.lower() for entry in parity):
                problems.append(
                    f"manifest mirror kind={kind} must declare 'none' parity "
                    f"with a reason, not real checks: {m['path']}"
                )

    files = _iter_files()
    violations: list[str] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        violations.extend(_check_file(p, text, manifest))

    if args.verbose:
        for p in files:
            tag = "MIRROR" if _rel(p) in manifested else "scan  "
            print(f"  {tag} {_rel(p)}")

    if problems:
        print("MANIFEST PROBLEMS:")
        for pr in problems:
            print(f"  {pr}")
        return 1

    if violations:
        print(
            "ARCHITECTURE VIOLATIONS — Rust-owned runtime logic outside "
            "runtime-core and outside shared/architecture_mirrors.json:"
        )
        for v in violations:
            print(f"  {v}")
        print(
            "\nFix: delegate to runtime-core bindings, import the shared "
            "helper, or add an explicit documented mirror entry to "
            "shared/architecture_mirrors.json with a parity check."
        )
        return 1

    print(
        f"architecture guardrail: OK ({len(files)} files scanned, "
        f"{len(manifested)} manifested mirrors)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
