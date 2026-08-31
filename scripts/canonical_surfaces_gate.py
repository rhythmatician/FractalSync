"""Cross-surface contract gate: parity of production paths, not just formulas.

The #93 incident: AnalysisTimebase was implemented in Rust, exported through
wasm (runtime surface), and fully tested in Rust — but the training surface
(backend) never consumed it. Component-level parity tests stayed green while
the real production paths diverged (training bypassed the timebase entirely;
the preflight advanced both controller paths at a hardcoded 1/60 the browser
no longer supplies).

This gate makes an incomplete cross-surface implementation an INVALID
repository state. It enforces shared/canonical_surfaces.json:

  1. Manifest integrity — every declared authority file, surface file, and
     required test must exist in the repository.
  2. Binding symmetry — every pyclass in runtime-core/src/pybindings.rs must
     have a wasm_bindgen export in wasm-orbit/src/lib.rs (and vice versa),
     unless exempted in the manifest. This is the structural check that
     would have failed #93 immediately: AnalysisTimebase was wasm-exported
     with no PyO3 training surface.
  3. Required-test reachability — each required test file must reference the
     subsystem's authority symbol, proving the test actually exercises the
     canonical implementation rather than a bypass.

Usage:
    python scripts/canonical_surfaces_gate.py            # exit 0/1
    python scripts/canonical_surfaces_gate.py --verbose

Run in CI via backend/tests/test_canonical_surfaces_gate.py and directly in
.github/workflows/pytest.yml.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "shared" / "canonical_surfaces.json"

PYBINDINGS = REPO_ROOT / "runtime-core" / "src" / "pybindings.rs"
WASM_BINDINGS = REPO_ROOT / "wasm-orbit" / "src" / "lib.rs"

# Classes that are binding-layer plumbing, not canonical subsystems.
_PLUMBING_CLASSES = {"Complex"}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        print(f"FAIL: manifest missing: {MANIFEST_PATH}")
        raise SystemExit(1)
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def extract_pyclasses(source: str) -> set[str]:
    """Names of structs annotated #[pyclass] in pybindings.rs."""
    return set(
        re.findall(
            r"#\[\s*pyclass[^\]]*\]\s*(?:#[^\]]*\]\s*)*pub\s+struct\s+(\w+)", source
        )
    )


def extract_wasm_classes(source: str) -> set[str]:
    """Names of structs annotated #[wasm_bindgen] in wasm-orbit/src/lib.rs."""
    return set(
        re.findall(
            r"#\[\s*wasm_bindgen[^\]]*\]\s*(?:#[^\]]*\]\s*)*pub\s+struct\s+(\w+)",
            source,
        )
    )


def check_binding_symmetry(manifest: dict) -> list[str]:
    """Every pyclass must have a wasm counterpart (and vice versa)."""
    failures: list[str] = []
    exemptions = set(manifest.get("binding_symmetry", {}).get("exemptions", []))

    py_src = PYBINDINGS.read_text(encoding="utf-8")
    wasm_src = WASM_BINDINGS.read_text(encoding="utf-8")
    py_classes = extract_pyclasses(py_src) - _PLUMBING_CLASSES
    wasm_classes = extract_wasm_classes(wasm_src) - _PLUMBING_CLASSES

    for name in sorted(py_classes - wasm_classes - exemptions):
        failures.append(
            f"binding symmetry: pyclass '{name}' (runtime-core/src/pybindings.rs) "
            f"has no wasm_bindgen export in wasm-orbit/src/lib.rs — a canonical "
            f"subsystem must be reachable from BOTH the runtime (wasm) and "
            f"training (PyO3) surfaces. Either export it, exempt it in "
            f"{MANIFEST_PATH.relative_to(REPO_ROOT)}, or it is not canonical."
        )
    for name in sorted(wasm_classes - py_classes - exemptions):
        failures.append(
            f"binding symmetry: wasm_bindgen struct '{name}' (wasm-orbit/src/lib.rs) "
            f"has no pyclass export in runtime-core/src/pybindings.rs — a canonical "
            f"subsystem must be reachable from BOTH the runtime (wasm) and "
            f"training (PyO3) surfaces. Either export it, exempt it in "
            f"{MANIFEST_PATH.relative_to(REPO_ROOT)}, or it is not canonical."
        )
    return failures


def check_manifest_integrity(manifest: dict) -> list[str]:
    """All referenced files must exist; required tests must reference the
    authority symbol."""
    failures: list[str] = []
    subsystems = manifest.get("subsystems", {})
    if not subsystems:
        failures.append("manifest declares no subsystems")

    for name, sub in subsystems.items():
        surfaces = sub.get("surfaces", {})
        if "rust_authority" not in surfaces:
            failures.append(f"{name}: no rust_authority surface declared")

        # Authority file must exist and be under runtime-core.
        authority = surfaces.get("rust_authority", "")
        authority_path = REPO_ROOT / authority
        if not authority_path.exists():
            failures.append(f"{name}: authority file missing: {authority}")
        elif "runtime-core" not in authority.replace("\\", "/"):
            failures.append(
                f"{name}: authority '{authority}' is not under runtime-core — "
                "canonical subsystems are Rust-owned (ADR 0001)"
            )

        # Every other declared surface file must exist.
        for surface_name, surface_path in surfaces.items():
            if surface_name == "rust_authority":
                continue
            # Surface values may be prose ("frontend → wasm ..."); extract
            # any repo-relative file paths mentioned.
            for token in re.findall(r"[\w\-/]+\.(?:rs|py|ts|tsx)", surface_path):
                if not (REPO_ROOT / token).exists():
                    failures.append(
                        f"{name}: surface '{surface_name}' references missing "
                        f"file: {token}"
                    )

        # Required tests must exist AND reference the authority symbol.
        for test_rel in sub.get("required_tests", []):
            test_path = REPO_ROOT / test_rel
            if not test_path.exists():
                failures.append(f"{name}: required test missing: {test_rel}")
                continue
            # The authority symbol is the last path component of the
            # authority (e.g. timebase.rs -> timebase, controller.rs ->
            # controller) or a declared class name.
            authority_stem = Path(authority).stem
            test_text = test_path.read_text(encoding="utf-8", errors="replace")
            if authority_stem not in test_text:
                failures.append(
                    f"{name}: required test '{test_rel}' does not reference "
                    f"the authority module '{authority_stem}' — a parity test "
                    "that bypasses the canonical implementation cannot "
                    "certify production-path parity"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()

    failures: list[str] = []
    failures += check_binding_symmetry(manifest)
    failures += check_manifest_integrity(manifest)

    if args.verbose:
        py_classes = extract_pyclasses(PYBINDINGS.read_text(encoding="utf-8"))
        wasm_classes = extract_wasm_classes(WASM_BINDINGS.read_text(encoding="utf-8"))
        print(f"pyclasses:      {sorted(py_classes)}")
        print(f"wasm_bindgen:   {sorted(wasm_classes)}")
        print(f"subsystems:     {sorted(manifest.get('subsystems', {}))}")

    if failures:
        print("CANONICAL SURFACES GATE FAILURE — invalid repository state:")
        for f in failures:
            print(f"  - {f}")
        print(
            "\nA canonical shared subsystem must be reachable from every "
            "declared surface through its real public seam. Component parity "
            "tests cannot catch a missing production path (see #93)."
        )
        return 1

    print("Canonical surfaces gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
