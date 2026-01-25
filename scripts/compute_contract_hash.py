"""Compute deterministic SHA256 hash of contracts/model_io_contract.json excluding 'version' field."""

from pathlib import Path
import json
import hashlib

CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "model_io_contract.json"


def load_contract():
    with open(CONTRACT, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical_json(obj) -> str:
    # deterministic serialization: sorted keys, no spaces
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_hash() -> str:
    cj = load_contract()
    # exclude 'version' field
    cj_nover = {k: v for k, v in cj.items() if k != "version"}
    canon = canonical_json(cj_nover)
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return h


if __name__ == "__main__":
    print(compute_hash())
