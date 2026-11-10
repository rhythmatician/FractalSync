"""Reproduce the historical adaptive-provider refusal, outside this engine.

Supply a clean checkout of the recorded adaptive source. This diagnostic expects
the old test to fail at the recorded point; it does not claim a provider fix.
"""

import argparse
import subprocess
from pathlib import Path

SOURCE = "e780e795e59f9426e34bb1c2bb764886f5b4dab9"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adaptive_checkout", type=Path)
    checkout = parser.parse_args().adaptive_checkout.resolve(strict=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if head != SOURCE:
        raise RuntimeError(f"expected adaptive source {SOURCE}, got {head}")
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "runtime-core"],
        cwd=checkout,
        check=True,
    )
    result = subprocess.run(
        [
            "rtk",
            "proxy",
            "cargo",
            "test",
            "--manifest-path",
            "runtime-core/Cargo.toml",
            "--release",
            "manifold::tests::controls_drive_can_cross_shore_without_wall",
            "--lib",
            "--",
            "--nocapture",
        ],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    expected = "c=(0.21474166530729011, 0.12774018997553147)"
    if (
        result.returncode == 0
        or "singularity=CutLocus" not in result.stdout
        or expected not in result.stdout
    ):
        raise RuntimeError(
            f"historical refusal changed or did not reproduce:\n{result.stdout}"
        )
    print(f"historical adaptive failure reproduced at {expected}; source={SOURCE}")


if __name__ == "__main__":
    main()
