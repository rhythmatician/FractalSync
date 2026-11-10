"""Compare an actual frozen-asset rollout in native Python and Node WASM."""

import json
import math
import subprocess
import sys
from pathlib import Path

from runtime_core import BoundedOperationalRollout


def main() -> None:
    asset, policy = Path(sys.argv[1]), Path(sys.argv[2])
    engine = BoundedOperationalRollout(
        asset.read_bytes(),
        policy.read_text(),
        0.1,
        1e-4,
        1.0,
        0.1,
        1 / math.pi,
        0.001,
        0.0,
    )
    probe = json.loads(engine.probe(0.4, 0.05))
    engine.start(0.4, 0.05, -0.001, 0.0)
    for _ in range(10):
        state = json.loads(engine.step(0.0, 0.0))
    wasm = json.loads(
        subprocess.check_output(
            [
                "node",
                "scripts/check_bounded_asset.cjs",
                "tmp/final-wasm/orbit_synth_wasm.js",
                str(asset),
                str(policy),
            ],
            text=True,
        )
    )
    assert probe["field_id"] == wasm["probe"]["field_id"]
    assert state["physics_profile_id"] == wasm["state"]["physics_profile_id"]
    errors = [
        abs(probe["operational_value"] - wasm["probe"]["operational_value"]),
        abs(probe["potential"] - wasm["probe"]["potential"]),
    ]
    errors.extend(
        abs(a - b)
        for a, b in zip(
            sum(probe["metric"], []), sum(wasm["probe"]["metric"], []), strict=True
        )
    )
    for key in ("position", "velocity"):
        errors.extend(
            abs(a - b) for a, b in zip(state[key], wasm["state"][key], strict=True)
        )
    for key in ("mechanical_time", "integrated_work", "energy"):
        errors.append(abs(state[key] - wasm["state"][key]))
    assert all(math.isfinite(e) and e <= 1e-9 for e in errors)
    print(
        json.dumps(
            {
                "passed": True,
                "tolerance": 1e-9,
                "maximum_absolute_error": max(errors),
                "field_id": probe["field_id"],
                "physics_profile_id": state["physics_profile_id"],
                "steps": 10,
                "scope": "Node WASM and native Python, one frozen-asset trajectory",
            }
        )
    )


if __name__ == "__main__":
    main()
