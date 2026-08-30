"""CI wrapper for the canonical-surfaces gate.

Makes an incomplete cross-surface implementation an invalid repository
state: a canonical shared subsystem that exists on one required surface
(runtime wasm) but not another (training PyO3) fails CI — the #93 failure
mode. See scripts/canonical_surfaces_gate.py and
shared/canonical_surfaces.json.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GATE = REPO_ROOT / "scripts" / "canonical_surfaces_gate.py"


def test_canonical_surfaces_gate_passes():
    result = subprocess.run(
        [sys.executable, str(GATE)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        "canonical surfaces gate failed — invalid repository state:\n"
        f"{result.stdout}\n{result.stderr}"
    )
