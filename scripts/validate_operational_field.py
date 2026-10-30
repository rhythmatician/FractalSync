"""Replay existing tests on an explicitly experimental, staged Rust wheel.

No wheel installation or test-body edits. The preparation happens before pytest.
Use the project venv, and pass a wheel built with python,experimental-operational-field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import zipfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reference-policy",type=Path)
    parser.add_argument("--staging", type=Path, help=argparse.SUPPRESS)
    args, pytest_args = parser.parse_known_args()
    if args.staging is None:
        # A loaded DLL cannot be deleted on Windows. Let the child exit before
        # the owning parent removes its temporary staging directory.
        with tempfile.TemporaryDirectory(prefix="operational-field-") as staging:
            return subprocess.run([
                sys.executable, str(Path(__file__).resolve()), str(args.wheel.resolve()),
                "--report", str(args.report.resolve()), "--staging", staging, *pytest_args,
                "--asset", str(args.asset.resolve()), *(["--reference-policy",str(args.reference_policy.resolve())] if args.reference_policy else []),
            ], check=False).returncode
    root = Path(__file__).resolve().parents[1]
    wheel = args.wheel.resolve()
    test = root / "backend/tests/test_manifold_physics.py"
    before = hashlib.sha256(test.read_bytes()).hexdigest()
    if args.staging is not None:
        staging = str(args.staging)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(staging)
        sys.path.insert(0, staging)
        sys.path.insert(1, str(root / "backend"))
        os.environ["RUNTIME_CORE_NO_BUILD"] = "1"
        import runtime_core
        import pytest

        class Outcomes:
            def __init__(self) -> None:
                self.tests: list[dict[str, str | float]] = []

            def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
                if report.when == "call" or report.failed:
                    self.tests.append({"nodeid": report.nodeid, "outcome": report.outcome, "phase": report.when, "seconds": report.duration})
                    if report.failed:
                        failure=str(report.longrepr)
                        for path,label in [(str(root),"${REPO}"),(staging,"${WHEEL}")]:
                            failure=failure.replace(path,label).replace(path.replace("\\","\\\\"),label)
                        self.tests[-1]["failure"]=failure[-3000:]

        outcomes = Outcomes()

        extension = next(Path(staging).rglob("*.pyd"))
        started = time.perf_counter()
        asset=args.asset.read_bytes()
        identity = runtime_core.experimental_load_operational_asset_with_policy(asset,args.reference_policy.read_text(encoding="utf-8")) if args.reference_policy else runtime_core.experimental_load_operational_asset(asset)
        preparation = time.perf_counter() - started
        started = time.perf_counter()
        selected_args=pytest_args or [str(test.relative_to(root)), "-q", "-k", "TestEnergyDrift or TestShoreCrossings or TestSignedSdfContinuity"]
        code = pytest.main(selected_args, plugins=[outcomes])
        record = {
            "profile": "experimental-operational-field/1",
            "field_identity": identity,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "test_source_sha256": before,
            "test_source_unchanged": before == hashlib.sha256(test.read_bytes()).hexdigest(),
            "rust_source_sha256": {
                str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [root / "runtime-core/src/operational_field.rs", root / "runtime-core/src/geometry_provider.rs", root / "runtime-core/src/pybindings.rs", root / "runtime-core/src/manifold.rs"]
            },
            "asset_loading_seconds": preparation,
            "asset_sha256":hashlib.sha256(asset).hexdigest(),
            "reference_policy_sha256":hashlib.sha256(args.reference_policy.read_bytes()).hexdigest() if args.reference_policy else None,
            "pytest_seconds": time.perf_counter() - started,
            "pytest_args": selected_args,
            "pytest_exit_code": int(code),
            "tests": outcomes.tests,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
