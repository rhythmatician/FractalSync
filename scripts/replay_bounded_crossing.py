"""Replay the #148 crossing scenario on one explicitly supplied frozen asset."""

import argparse
import json
import math
from pathlib import Path

from runtime_core import BoundedOperationalRollout


def event_time(t, dt, before, after):
    if not before >= 0.0 > after:
        raise ValueError("event does not bracket the operational contour")
    return t + dt * before / (before - after)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("policy", type=Path)
    args = parser.parse_args()
    asset, policy = args.asset.read_bytes(), args.policy.read_text(encoding="utf-8")

    def create(dt):
        return BoundedOperationalRollout(
            asset, policy, 0.1, 1e-4, 1.0, 0.1, 1 / math.pi, dt, 0.0
        )

    probe = create(0.001)

    def sample(x, y):
        return json.loads(probe.probe(x, y))

    lo, hi, y = 0.2, 0.5, 0.05
    assert sample(lo, y)["operational_value"] < 0 < sample(hi, y)["operational_value"]
    for _ in range(60):
        mid = (lo + hi) / 2
        if sample(mid, y)["operational_value"] < 0:
            lo = mid
        else:
            hi = mid
    start = sample(0.4, y)
    crest = sample((lo + hi) / 2, y)
    barrier = crest["potential"] - start["potential"]
    assert barrier > 0

    def attempt(multiplier, dt):
        engine = create(dt)
        vx = -math.sqrt(2 * multiplier * barrier / start["metric"][0][0])
        state = json.loads(engine.start(0.4, y, vx, 0.0))
        initial_energy = state["energy"]
        max_energy_excursion = 0.0
        previous = start["operational_value"]
        minimum = previous
        for step in range(round(5 / dt)):
            state = json.loads(engine.step(0.0, 0.0))
            max_energy_excursion = max(
                max_energy_excursion,
                abs(state["energy"] - initial_energy) / max(abs(initial_energy), 1.0),
            )
            d = json.loads(engine.probe(*state["position"]))["operational_value"]
            minimum = min(minimum, d)
            if d < 0:
                return {
                    "crossed": True,
                    "time": event_time(step * dt, dt, previous, d),
                    "minimum": minimum,
                    "max_energy_excursion": max_energy_excursion,
                    "state": state,
                }
            previous = d
        return {
            "crossed": False,
            "minimum": minimum,
            "state": state,
            "max_energy_excursion": max_energy_excursion,
        }

    under = attempt(0.5, 0.001)
    attempts = [attempt(25.0, dt) for dt in (0.002, 0.001, 0.0005, 0.00025)]
    assert not under["crossed"] and under["minimum"] > 0
    assert all(attempt["crossed"] for attempt in attempts)
    assert attempts[1]["minimum"] < under["minimum"]
    assert abs(attempts[2]["time"] - attempts[3]["time"]) < 0.05
    assert all(result["max_energy_excursion"] < 0.05 for result in [under, *attempts])
    print(
        json.dumps(
            {
                "operational_crossing": "passed",
                "reference_crossing": "unverified",
                "reference_distance_accuracy": "unknown",
                "field_id": probe.field_id(),
                "under": under,
                "attempts": attempts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
