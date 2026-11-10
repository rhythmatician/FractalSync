"""Compose the default and optional stubs in a caller-selected output tree."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    default = (root / "runtime-core/runtime_core.pyi").read_text(encoding="utf-8")
    extra = (
        root / "runtime-core/experimental-stubs/runtime_core/__init__.pyi"
    ).read_text(encoding="utf-8")
    destination = args.output / "runtime_core"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "__init__.pyi").write_text(
        default.rstrip() + "\n\n" + extra, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
