"""Generate a .pyi file for runtime_core using metadata provided by the compiled bindings.

This is a pragmatic fallback to produce accurate stubs when introspection-only generation
by `pyo3-stubgen` is insufficient.
"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any

import runtime_core as rc


def generate_pyi(metadata: dict[str, Any]) -> str:
    out = [
        "# Auto-generated runtime_core stubs",
        "from __future__ import annotations",
        "from typing import Optional, Sequence, Any",
        "from numpy.typing import NDArray",
        "",
    ]

    funcs = metadata.get("functions", {})
    for name, sig in funcs.items():
        # Normalize ndarray typing to NDArray if present
        sig = sig.replace("-> ndarray", "-> NDArray")
        out.append(f"def {name}{sig}: ...\n")

    # Emit constants if present
    consts = metadata.get("constants", {})
    for k, v in consts.items():
        out.append(f"{k}: {v}")
    if consts:
        out.append("")

    for cls_name in (
        "Complex",
        "ResidualParams",
        "OrbitState",
        "FeatureExtractor",
        "RuntimeVisualMetrics",
    ):
        cls = metadata.get(cls_name, {})
        out.append(f"class {cls_name}:")
        attrs = cls.get("attributes", [])

        # Special-case Complex to expose re/im and real/imag
        if cls_name == "Complex":
            out.append("    re: Any")
            out.append("    im: Any")
            out.append("    real: Any")
            out.append("    imag: Any")
        else:
            for a in attrs:
                out.append(f"    {a}: Any")

        methods = cls.get("methods", {})
        # Treat constructors/newters as static methods
        static_methods = {"new_with_seed", "new_default_seeded"}
        for m, sig in methods.items():
            # Normalize ndarray typing
            sig = sig.replace("-> ndarray", "-> NDArray")
            if m == "__init__":
                if sig.startswith("("):
                    close_idx = sig.find(")")
                    args = sig[1:close_idx]
                    if args.strip() == "":
                        out.append("    def __init__(self) -> None: ...")
                    else:
                        out.append(f"    def __init__(self, {args}) -> None: ...")
                else:
                    out.append(f"    def __init__{sig}: ...")
            elif m in static_methods:
                out.append(f"    def {m}{sig}: ...")
            else:
                if sig.startswith("("):
                    close_idx = sig.find(")")
                    args = sig[1:close_idx]
                    rest = sig[close_idx + 1 :]
                    if args.strip() == "":
                        new_sig = f"(self){rest}"
                    else:
                        new_sig = f"(self, {args}){rest}"
                    out.append(f"    def {m}{new_sig}: ...")
                else:
                    out.append(f"    def {m}{sig}: ...")
        out.append("")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default="backend/stubs/runtime_core")
    args = parser.parse_args()

    meta = rc.export_binding_metadata()

    # Convert PyDict/PyObjects to native dict
    # The runtime binding returns built-in Python types so we can use them directly
    metadata = dict(meta)

    pyi_text = generate_pyi(metadata)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runtime_core.pyi"
    out_path.write_text(pyi_text, encoding="utf-8")
    print(f"Wrote {out_path}")

    # Do NOT overwrite runtime-core/runtime_core.pyi: that file is the
    # hand-authored canonical stub shipped in the wheel (see
    # runtime-core/pyproject.toml) and validated by
    # runtime-core/tests/test_stub_parity.rs. The auto-generated output is
    # lower fidelity (Any-typed attributes, missing constants) and must not
    # clobber it — doing so made the stub-verification workflow dirty on
    # every run.


if __name__ == "__main__":
    main()
