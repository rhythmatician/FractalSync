"""Validate saved teacher timelines without running either upstream model."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import json
from math import isclose, isfinite
from pathlib import Path
from typing import TypedDict


class LabelProbability(TypedDict):
    label: str
    prob: float


class SegmentTiming(TypedDict):
    start: float
    end: float


class SegmentExport(SegmentTiming, total=False):
    label: str
    topk: list[LabelProbability]


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    labels: tuple[str, ...]


@dataclass(frozen=True)
class Window:
    """Native teacher evidence attached to an observation time."""

    time: float
    start: float
    end: float
    value: float | None


def read_segments(path: Path) -> list[Segment]:
    rows: list[SegmentExport] = json.loads(path.read_text(encoding="utf-8-sig"))
    if not rows or not isinstance(rows, list):
        raise ValueError(f"Expected a nonempty segment array in {path.name}")
    segments: list[Segment] = []
    previous_end = 0.0
    for row in rows:
        start, end = float(row["start"]), float(row["end"])
        if not (0 <= previous_end <= start < end and isfinite(end)):
            raise ValueError(f"Invalid or overlapping segment in {path.name}")
        if "label" in row:
            labels = (row["label"],)
        elif "topk" in row:
            choices = row["topk"]
            if not all(0 <= choice["prob"] <= 1 for choice in choices):
                raise ValueError(f"Invalid label probabilities in {path.name}")
            if sum(choice["prob"] for choice in choices) > 1 + 1e-6:
                raise ValueError(f"Probability mass exceeds one in {path.name}")
            labels = tuple(choice["label"] for choice in choices)
        else:
            raise ValueError(f"Missing hard label or top-k probabilities in {path.name}")
        if not labels or not all(labels) or len(set(labels)) != len(labels):
            raise ValueError(f"Invalid labels in {path.name}")
        segments.append(Segment(start, end, labels))
        previous_end = end
    return segments


def targets_at(
    windows: Iterable[Window], ticks: Iterable[int], sample_rate: int, maximum_age: float
) -> list[float | None]:
    """Select the latest native target with a valid support window.

    This produces offline labels. Support can extend past the query time.
    The live provenance rule below is checked independently.
    """
    if sample_rate <= 0 or maximum_age < 0 or not isfinite(maximum_age):
        raise ValueError("Invalid timeline sampling parameters")
    result: list[float | None] = []
    native = list(windows)
    previous_time = -1.0
    for row in native:
        if not 0 <= row.start <= row.time <= row.end < float("inf"):
            raise ValueError("Invalid teacher support")
        if row.time <= previous_time or (row.value is not None and not isfinite(row.value)):
            raise ValueError("Teacher records must be finite and strictly ordered")
        previous_time = row.time
    index = -1
    previous_tick = -1
    for tick in ticks:
        if not 0 <= tick or tick < previous_tick:
            raise ValueError("Canonical ticks must be ordered")
        previous_tick = tick
        time = tick / sample_rate
        while index + 1 < len(native) and native[index + 1].time <= time:
            index += 1
        current = native[index] if index >= 0 else None
        result.append(current.value if current and time - current.time <= maximum_age else None)
    return result


def usable_live(row: Window, tick: int, sample_rate: int) -> bool:
    """Check timing availability only. Teachers remain offline by contract."""
    return row.value is not None and row.end <= tick / sample_rate


def synthetic_probe() -> None:
    # The middle sample is missing. The last observation is a genuine zero.
    tension = [Window(0.5, 0, 1.5, 0.6), Window(1.0, 0, 1.5, None), Window(1.5, 1, 2, 0.0)]
    ticks = [0, 24_000, 36_000, 48_000, 60_000, 72_000, 96_000]
    assert targets_at(tension, ticks, 48_000, 0.3) == [None, 0.6, 0.6, None, None, 0.0, None]
    assert not usable_live(tension[0], 24_000, 48_000)
    assert usable_live(tension[0], 72_000, 48_000)
    assert not usable_live(tension[1], 96_000, 48_000)
    assert usable_live(tension[2], 96_000, 48_000)
    print("Synthetic alignment: verified future support, missing targets, zero tension, and staleness")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path)
    args = parser.parse_args()
    synthetic_probe()
    if args.segments_dir is None:
        return
    files = sorted(args.segments_dir.rglob("*.json"))
    if not files:
        raise ValueError("No saved segment exports found")
    lengths: list[float] = []
    counts: list[int] = []
    maximum_gap = 0.0
    for path in files:
        segments = read_segments(path)
        counts.append(len(segments))
        lengths.append(segments[-1].end)
        maximum_gap = max(maximum_gap, segments[0].start, *(
            right.start - left.end for left, right in zip(segments, segments[1:])
        ))
    print(f"Exports: {len(files)}; rows: {sum(counts)}; duration: {min(lengths):.3f} to {max(lengths):.3f} s")
    print(f"Contiguous: {isclose(maximum_gap, 0.0, abs_tol=1e-9)}; largest gap: {maximum_gap:.6f} s")


if __name__ == "__main__":
    main()
