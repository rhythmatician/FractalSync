import json
from pathlib import Path
import subprocess

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "model_io_contract.json"


def compute_hash():
    # reuse the script to compute
    out = subprocess.check_output(
        [
            "python",
            str(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "compute_contract_hash.py"
            ),
        ]
    )
    return out.decode().strip()


def test_contract_version_matches_hash():
    with open(CONTRACT, "r", encoding="utf-8") as f:
        cj = json.load(f)
    assert "version" in cj, "contract missing 'version' field"
    computed = compute_hash()
    assert (
        cj["version"] == computed
    ), f"contract version mismatch: file={cj['version']} computed={computed}"
