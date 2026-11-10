"""Compare a named browser replay to the native frozen-asset rollout."""

import json
import math
import sys
from pathlib import Path

from runtime_core import BoundedOperationalRollout


def main() -> None:
    asset, policy, browser = (Path(arg) for arg in sys.argv[1:])
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
    result = json.loads(browser.read_text())["result"]
    other_probe, other_state = result["probe"], result["parity_state"]
    assert probe["field_id"] == other_probe["field_id"]
    assert state["physics_profile_id"] == other_state["physics_profile_id"]
    errors = [
        abs(probe["operational_value"] - other_probe["operational_value"]),
        abs(probe["potential"] - other_probe["potential"]),
    ]
    errors.extend(
        abs(a - b)
        for a, b in zip(
            sum(probe["metric"], []), sum(other_probe["metric"], []), strict=True
        )
    )
    for key in ("position", "velocity"):
        errors.extend(
            abs(a - b) for a, b in zip(state[key], other_state[key], strict=True)
        )
    for key in ("mechanical_time", "integrated_work", "energy"):
        errors.append(abs(state[key] - other_state[key]))
    assert all(math.isfinite(error) and error <= 1e-9 for error in errors)
    print(
        json.dumps(
            {
                "passed": True,
                "tolerance": 1e-9,
                "maximum_absolute_error": max(errors),
                "field_id": probe["field_id"],
                "physics_profile_id": state["physics_profile_id"],
                "steps": 10,
            }
        )
    )


if __name__ == "__main__":
    main()
