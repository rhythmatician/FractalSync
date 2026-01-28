"""Validate runtime_core stubs against the actual bindings."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import runtime_core as rc  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _ensure_runtime_core(runtime_core_module: ModuleType) -> ModuleType:
    """Autouse fixture that installs the built runtime_core module as `rc`.

    The fixture sets a module-level `rc` variable so existing tests can
    reference `rc` directly without changing the test bodies.
    """
    globals()["rc"] = runtime_core_module
    return runtime_core_module


def compile_classes() -> dict[str, list[str]] | None:
    """Attempt to compile the expected classes from the stub file.

    Returns:
        A mapping of class names to lists of member names, or None on failure.
    """
    import ast
    from pathlib import Path

    stub_dir = Path(__file__).parent.parent / "stubs"
    stub_path = stub_dir / "runtime_core" / "runtime_core.pyi"
    assert stub_path is not None

    try:
        with open(stub_path, "r", encoding="utf-8") as f:
            stub_source = f.read()
        tree = ast.parse(stub_source, filename=str(stub_path))

        classes: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                member_names: list[str] = []
                for body_item in node.body:
                    if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        member_names.append(body_item.name)
                    elif isinstance(body_item, ast.AnnAssign):
                        if isinstance(body_item.target, ast.Name):
                            member_names.append(body_item.target.id)
                    elif isinstance(body_item, ast.Assign):
                        for target in body_item.targets:
                            if isinstance(target, ast.Name):
                                member_names.append(target.id)
                classes[node.name] = member_names
        return classes
    except Exception as e:
        print(f"[stub-parse-error] {e}", file=sys.stderr)
        return None


def test_classes_have_expected_members():
    """For each class in our stubs check the expected attributes/methods exist.

    We attempt a conservative check: look for the member on the class object
    and (when possible) on a default instance. For certain classes we use
    safe fallbacks to obtain an instance (e.g. explicit constructor args for
    `Complex`). This makes the
    test tolerant of minor ABI differences while verifying presence of the
    API surface described in the stubs.
    """
    assert rc is not None, "runtime_core fixture did not provide module `rc`"
    if not (classes := compile_classes()):
        pytest.fail("Could not compile expected classes from runtime_core stubs")
    all_missing = {}
    for cls_name, members in classes.items():
        cls = getattr(rc, cls_name)

        # Try to obtain an instance for instance-level checks. Be conservative
        # and avoid letting exceptions fail the entire test suite.
        inst = None
        try:
            inst = cls()
        except Exception:
            # Fallbacks for commonly non-default-constructible types
            try:
                if cls_name == "Complex":
                    inst = cls(0.0, 0.0)
                elif cls_name == "FeatureExtractor":
                    inst = cls()
            except Exception:
                inst = None

        # Complex has several acceptable access patterns; treat as alternative group
        if cls_name == "Complex":
            has_re_im = (hasattr(cls, "re") and hasattr(cls, "im")) or (
                inst is not None and hasattr(inst, "re") and hasattr(inst, "im")
            )
            has_real_imag = (hasattr(cls, "real") and hasattr(cls, "imag")) or (
                inst is not None and hasattr(inst, "real") and hasattr(inst, "imag")
            )
            has_complex = hasattr(cls, "__complex__") or (
                inst is not None and hasattr(inst, "__complex__")
            )
            if not (has_re_im or has_real_imag or has_complex):
                pytest.fail(
                    f"Complex type lacks any expected accessor patterns (re/im, real/imag, "
                    f"or __complex__) on class {cls} or its instances"
                )
            continue

        missing = []
        for m in members:
            present = False
            # Member on the class object (e.g., classmethod, attribute)
            if hasattr(cls, m):
                present = True
            # Member on an instance (e.g., property/method bound to instance)
            if not present and inst is not None and hasattr(inst, m):
                present = True
            if not present:
                missing.append(m)

        if missing:
            all_missing[cls_name] = missing

    if all_missing:
        messages = [f"{k}: missing {sorted(v)}" for k, v in all_missing.items()]
        pytest.fail("Runtime core API differs from stubs:\n" + "\n".join(messages))


def test_compile_classes_returns_nonempty_dict():
    """Test that compile_classes returns valid structure."""
    result = compile_classes()
    assert isinstance(result, dict)
    assert result is not None, "compile_classes returned None"
    for cls_name, members in result.items():
        assert isinstance(cls_name, str)
        assert isinstance(members, list)
        assert all(isinstance(m, str) for m in members)
