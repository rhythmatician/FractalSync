"""Replay the two retained absolute-energy limits on the frozen asset."""

import json
import math
import sys
from pathlib import Path

from runtime_core import BoundedOperationalRollout


def main() -> None:
    asset, policy = (Path(arg) for arg in sys.argv[1:])
    engine = BoundedOperationalRollout(
        asset.read_bytes(),
        policy.read_text(),
        0.1,
        1e-4,
        1.0,
        1.0,
        1 / math.pi,
        0.002,
        0.0,
    )
    results = []
    for name, start, steps in (
        ("high_curvature", [0.4, 0.05], 150),
        ("near_shore", [0.3, 0.05], 100),
    ):
        initial = json.loads(engine.start(start[0], start[1], 0.0, 0.02))
        maximum = 0.0
        for _ in range(steps):
            state = json.loads(engine.step(0.0, 0.0))
            maximum = max(maximum, abs(state["energy"] - initial["energy"]))
        drift = abs(state["energy"] - initial["energy"])
        assert maximum < 0.05 and drift < 0.05
        results.append(
            {
                "name": name,
                "maximum_absolute_excursion": maximum,
                "absolute_drift": drift,
                "limit": 0.05,
                "initial": initial,
                "final": state,
            }
        )
    print(
        json.dumps(
            {
                "field_id": engine.field_id(),
                "physics_profile_id": engine.physics_profile_id(),
                "runs": results,
            }
        )
    )


if __name__ == "__main__":
    main()
