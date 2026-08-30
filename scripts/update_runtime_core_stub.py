"""Sync backend/stubs/runtime_core/runtime_core.pyi from the canonical runtime-core/runtime_core.pyi.

Usage:
    python scripts/update_runtime_core_stub.py [--check]

If --check is provided, exit with code 0 when files are identical, 1 when different.
Otherwise, overwrite the backend stub with the canonical one.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "runtime-core" / "runtime_core.pyi"
BACKEND = ROOT / "backend" / "stubs" / "runtime_core" / "runtime_core.pyi"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check that backend stub matches canonical")
    args = parser.parse_args()

    canon = read(CANON)
    backend = read(BACKEND)

    if canon == backend:
        print("runtime_core stubs are in sync")
        return 0

    if args.check:
        print("runtime_core stubs differ: backend/stubs/runtime_core/runtime_core.pyi is out of date")
        return 1

    # Overwrite backend stub
    BACKEND.parent.mkdir(parents=True, exist_ok=True)
    BACKEND.write_text(canon, encoding="utf-8")
    print("Updated backend/stubs/runtime_core/runtime_core.pyi from canonical runtime-core/runtime_core.pyi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
