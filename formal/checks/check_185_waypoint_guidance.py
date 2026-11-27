"""Verify a retained-field waypoint replay and summarize paired outcomes."""

import argparse
import json
from collections import Counter
from math import hypot
from pathlib import Path
from statistics import median
from typing import TypedDict


class Frame(TypedDict):
    position: list[float]
    mechanical_time: float


class Route(TypedDict):
    id: str
    start: list[float]
    goal: list[float]


class Settings(TypedDict):
    forecast: int
    throttle: float


class Trace(TypedDict):
    frames: list[Frame]
    peak_throttle: float
    best_miss: float | None
    final_miss: float | None
    forecast_refusals: dict[str, int]
    executed_refusal: str | None
    unavailable_extra_pair: int
    selected_extra_pair: int


class Row(TypedDict):
    settings: Settings
    route: Route
    chart: str
    arm: str
    trace: Trace


class Replay(TypedDict):
    steps: int
    rows: list[Row]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    report: Replay = json.loads(parser.parse_args().replay.read_text(encoding="utf-8-sig"))
    grouped: dict[tuple[int, float, str, str], dict[str, Row]] = {}
    for row in report["rows"]:
        settings = row["settings"]
        route = row["route"]
        key = (settings["forecast"], settings["throttle"], row["chart"], route["id"])
        case = grouped.setdefault(key, {})
        assert row["arm"] not in case
        case[row["arm"]] = row
        trace = row["trace"]
        assert 0 <= trace["peak_throttle"] <= settings["throttle"]
        frames = trace["frames"]
        misses: list[float] = []
        if frames:
            assert frames[0]["position"] == route["start"]
            assert all(
                abs(frame["mechanical_time"] - index * 0.01) < 1e-12
                for index, frame in enumerate(frames)
            )
            misses = [
                hypot(*(a - b for a, b in zip(frame["position"], route["goal"], strict=True)))
                for frame in frames
            ]
            best_miss = trace["best_miss"]
            assert best_miss is not None and abs(min(misses) - best_miss) < 1e-12
        if trace["executed_refusal"] is None:
            assert len(frames) == report["steps"] + 1
            final_miss = trace["final_miss"]
            assert final_miss is not None and abs(misses[-1] - final_miss) < 1e-12
        else:
            assert len(frames) < report["steps"] + 1
            assert trace["final_miss"] is None

    aggregates: dict[tuple[int, float, str, str], list[tuple[str, Trace, Trace]]] = {}
    assert len(grouped) == 3 * 2 * 6
    for (forecast, throttle, chart, route_id), case in sorted(grouped.items()):
        assert set(case) == {"Map", "Multiplier", "PointTrap"}
        baseline = case["Map"]
        for arm, row in case.items():
            assert row["route"] == baseline["route"]
            trace = row["trace"]
            base_trace = baseline["trace"]
            if trace["frames"] and base_trace["frames"]:
                assert trace["frames"][0] == base_trace["frames"][0]
            if arm != "Map":
                aggregates.setdefault((forecast, throttle, chart, arm), []).append(
                    (route_id, base_trace, trace)
                )

    lines = [
        "| Forecast steps | Cap | Chart | Replacement | Median change | Range | Wins/ties/losses |"
        " Hits at .005 B/A | Hits at .010 B/A | Forecast refusals B/A | Real refusals B/A |"
        " Unavailable | Selected |",
        "| ---: | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for (forecast, throttle, chart, arm), pairs in aggregates.items():
        deltas = {}
        hits = {tolerance: [0, 0] for tolerance in (0.005, 0.01)}
        forecast_refusals: list[Counter[str]] = [Counter(), Counter()]
        real_refusals: list[Counter[str]] = [Counter(), Counter()]
        unavailable = selected = 0
        for route_id, baseline, alternative in pairs:
            b, a = baseline["final_miss"], alternative["final_miss"]
            if b is not None and a is not None:
                deltas[route_id] = a - b
            for side, trace in enumerate((baseline, alternative)):
                forecast_refusals[side].update(trace["forecast_refusals"])
                if refusal := trace["executed_refusal"]:
                    real_refusals[side][refusal] += 1
                for tolerance, counts in hits.items():
                    best_miss = trace["best_miss"]
                    if best_miss is not None and best_miss <= tolerance:
                        counts[side] += 1
            unavailable += alternative["unavailable_extra_pair"]
            selected += alternative["selected_extra_pair"]
        values = list(deltas.values())
        wins = sum(value < -1e-6 for value in values)
        losses = sum(value > 1e-6 for value in values)
        ties = len(values) - wins - losses
        effect = (
            f"{median(values):+.6f} | {min(values):+.6f} to {max(values):+.6f}"
            if values else "NA | NA"
        )
        forecasts = "/".join(str(sum(counter.values())) for counter in forecast_refusals)
        real = "/".join(str(sum(counter.values())) for counter in real_refusals)
        tight, loose = ("/".join(map(str, counts)) for counts in hits.values())
        lines.append(
            f"| {forecast} | {throttle:.2f} | {chart} | {arm} | {effect} |"
            f" {wins}/{ties}/{losses} | {tight} | {loose} | {forecasts} | {real} |"
            f" {unavailable} | {selected} |"
        )
        for label, offenders in (("forecast", forecast_refusals), ("executed", real_refusals)):
            if any(offenders):
                print(f"{forecast} {throttle} {chart} {arm} {label} refusals: {offenders}")
        print(f"{forecast} {throttle} {chart} {arm} deltas: {deltas}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
