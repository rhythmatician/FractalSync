"""Update the 'version' field in contracts/model_io_contract.json to the SHA256 of the rest of the contract."""
from pathlib import Path
import json
from compute_contract_hash import compute_hash

CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "model_io_contract.json"


def update_version(write: bool = True) -> str:
    h = compute_hash()
    with open(CONTRACT, "r", encoding="utf-8") as f:
        cj = json.load(f)
    if cj.get("version") == h:
        print("version already up-to-date")
        return h
    cj["version"] = h
    if write:
        with open(CONTRACT, "w", encoding="utf-8") as f:
            json.dump(cj, f, indent=4, sort_keys=False)
            f.write("\n")
        print("Updated version to:", h)
    return h


if __name__ == "__main__":
    update_version()
