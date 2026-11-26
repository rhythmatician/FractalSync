"""Audit training-only boundary targets made from saved SongFormer segments."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

import runtime_core as rust
import soundfile as sf

from check_teacher_timelines import Segment, SegmentExport, read_segments


ROOT = Path(__file__).resolve().parents[2]
RATE, HOP = rust.SAMPLE_RATE, rust.HOP_LENGTH
HORIZONS = tuple(milliseconds * RATE // 1000 for milliseconds in (500, 1000, 2000, 4000))
NAMES = ("boundary_now", "next_0.5s", "next_1s", "next_2s", "next_4s")


@dataclass(frozen=True)
class Join:
    sample: int
    left_label: str
    right_label: str

    @property
    def repeats_label(self) -> bool:
        return self.left_label == self.right_label

    @property
    def touches_silence(self) -> bool:
        return "silence" in (self.left_label, self.right_label)


@dataclass(frozen=True)
class Timeline:
    coverage: tuple[tuple[int, int], ...]
    joins: tuple[Join, ...]

    def target(self, lo: int, hi: int) -> tuple[Join, ...] | None:
        """Known hard joins in (lo, hi]. None means unknown, not a negative."""
        if not any(start <= lo <= hi < end for start, end in self.coverage):
            return None
        first = bisect_right(self.joins, lo, key=lambda join: join.sample)
        past = bisect_right(self.joins, hi, key=lambda join: join.sample)
        return self.joins[first:past]


def quantize(seconds: float) -> int:
    return round(seconds * RATE)


def build_timeline(segments: list[Segment], audio_end: int) -> Timeline:
    spans: list[tuple[int, int]] = []
    joins: list[Join] = []
    old: Segment | None = None
    for segment in segments:
        start, end = quantize(segment.start), min(quantize(segment.end), audio_end)
        if start >= audio_end:
            break
        if end <= start:
            raise ValueError("A segment vanishes on the sample clock")
        if old is not None and old.end == segment.start:
            initial, old_end = spans.pop()
            assert start == old_end
            spans.append((initial, end))
            joins.append(Join(start, old.labels[0], segment.labels[0]))
        else:
            # A source gap remains unknown even if its width rounds to zero samples.
            spans.append((start, end))
        old = segment
    return Timeline(tuple(spans), tuple(joins))


@dataclass
class Totals:
    positive: int = 0
    negative: int = 0
    masked: int = 0
    multi_join: int = 0
    only_same_label: int = 0
    any_silence_join: int = 0
    initial_source_mask: int = 0
    terminal_mask: int = 0

    def update(self, result: tuple[Join, ...] | None, lo: int, hi: int, timeline: Timeline) -> None:
        if result is None:
            self.masked += 1
            self.initial_source_mask += int(lo < timeline.coverage[0][0])
            self.terminal_mask += int(hi >= timeline.coverage[-1][1])
        elif not result:
            self.negative += 1
        else:
            self.positive += 1
            self.multi_join += int(len(result) > 1)
            self.only_same_label += int(all(join.repeats_label for join in result))
            self.any_silence_join += int(any(join.touches_silence for join in result))

    def report(self) -> dict:
        supported = self.positive + self.negative
        return {
            "positive": self.positive,
            "negative": self.negative,
            "masked": self.masked,
            "prevalence": self.positive / supported if supported else None,
            "always_negative_accuracy": self.negative / supported if supported else None,
            "positive_with_multiple_joins": self.multi_join,
            "positive_only_with_same_label_joins": self.only_same_label,
            "positive_touching_silence": self.any_silence_join,
            "initial_source_mask": self.initial_source_mask,
            "terminal_mask": self.terminal_mask,
            "other_gap_masks": self.masked - self.initial_source_mask - self.terminal_mask,
        }


def uncertainty(path: Path) -> list[dict]:
    rows: list[SegmentExport] = json.loads(path.read_text(encoding="utf-8-sig"))
    result = []
    for row in rows:
        if "topk" in row:
            choices = row["topk"]
            result.append({
                "start_seconds": row["start"], "end_seconds": row["end"], "top_k": choices,
                "omitted_probability_mass": max(0.0, 1 - sum(item["prob"] for item in choices)),
            })
    return result


def analyze(path: Path, audio: Path, representative: bool) -> dict:
    segments = read_segments(path)
    source = sf.info(str(audio))
    audio_end = source.frames * RATE // source.samplerate
    timeline = build_timeline(segments, audio_end)
    if not timeline.coverage:
        raise ValueError(f"No supported teacher/audio overlap for {path}")
    totals = [Totals() for _ in NAMES]
    for t in range(HOP, audio_end + 1, HOP):
        windows = ((t - HOP, t), *((t, t + h) for h in HORIZONS))
        assigned = [timeline.target(lo, hi) for lo, hi in windows]
        for (lo, hi), value, meter in zip(windows, assigned, totals, strict=True):
            meter.update(value, lo, hi, timeline)
        for short, long in zip(assigned[1:], assigned[2:]):
            if long is not None:
                assert short is not None and short == long[:len(short)]
    ticks = audio_end // HOP
    assert all(meter.masked + meter.negative + meter.positive == ticks for meter in totals)
    audio_hash = sha256()
    with audio.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            audio_hash.update(block)
    return {
        "group": path.parent.name,
        "export": path.name,
        "count_in_held_out_group_summaries": representative,
        "export_sha256": sha256(path.read_bytes()).hexdigest(),
        "local_audio_sha256": audio_hash.hexdigest(),
        "native_source_frames": source.frames,
        "native_source_rate": source.samplerate,
        "nominal_audio_end_sample": audio_end,
        "export_end_seconds": segments[-1].end,
        "audio_end_minus_export_end_ms": 1000 * (audio_end / RATE - segments[-1].end),
        "segments": len(segments),
        "native_segment_seconds": [[segment.start, segment.end] for segment in segments],
        "support_samples": timeline.coverage,
        "joins": [{"sample": join.sample, "left_label": join.left_label, "right_label": join.right_label}
                  for join in timeline.joins],
        "same_label_joins": sum(join.repeats_label for join in timeline.joins),
        "silence_joins": sum(join.touches_silence for join in timeline.joins),
        "minimum_join_spacing_seconds": min(
            ((right.sample - left.sample) / RATE for left, right in zip(timeline.joins, timeline.joins[1:])),
            default=None,
        ),
        "top_k_records": uncertainty(path),
        "targets": {name: meter.report() for name, meter in zip(NAMES, totals, strict=True)},
    }


def self_check() -> None:
    fixture = build_timeline(
        [Segment(0.25, 1.0, ("a",)), Segment(1.0, 2.0, ("a",)),
         Segment(2.25, 3.0, ("silence",)), Segment(3.0, 6.0, ("b",))],
        audio_end=5 * RATE,
    )
    first, second = fixture.joins
    assert first.repeats_label and second.touches_silence
    assert fixture.target(0, RATE) is None
    assert fixture.target(RATE - HOP, RATE) == (first,)
    assert fixture.target(RATE, RATE + HOP) == ()
    assert fixture.target(RATE, 2 * RATE) is None
    assert fixture.target(RATE, 3 * RATE) is None
    assert fixture.target(4 * RATE, 5 * RATE) is None
    assert fixture.target(3 * RATE - HOP, 5 * RATE - HOP) == (second,)
    assert quantize(0.5) == 24_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=ROOT / "outputs_songformer")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "backend/data/audio")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/boundaries-102/results.json")
    args = parser.parse_args()
    assert (RATE, HOP) == (48_000, 1024)
    self_check()
    exports = []
    for path in sorted(args.segments_dir.rglob("*.json")):
        group = path.parent.name
        representative = path.stem == group
        exports.append(analyze(path, args.audio_dir / f"{group}.mp3", representative))
    if not exports:
        raise ValueError("No saved segment exports found")
    groups = [row for row in exports if row["count_in_held_out_group_summaries"]]
    group_by_name = {row["group"]: row for row in groups}
    for row in exports:
        if row["count_in_held_out_group_summaries"]:
            continue
        other = group_by_name[row["group"]]
        assert row["native_segment_seconds"] == other["native_segment_seconds"]
        for a, b in zip(row["targets"].values(), other["targets"].values(), strict=True):
            assert all(a[k] == b[k] for k in ("positive", "negative", "masked"))
        labels = [item["right_label"] for item in row["joins"]]
        hard_labels = [item["right_label"] for item in other["joins"]]
        row["top1_disagreements_at_joins"] = sum(
            left != right for left, right in zip(labels, hard_labels, strict=True)
        )
    aggregate = {}
    for name in NAMES:
        rows = [item["targets"][name] for item in groups]
        positive = sum(x["positive"] for x in rows)
        negative = sum(x["negative"] for x in rows)
        count = positive + negative
        aggregate[name] = {
            "positive": positive, "negative": negative, "masked": sum(x["masked"] for x in rows),
            "prevalence": positive / count if count else None,
            "always_negative_accuracy": negative / count if count else None,
            "positive_with_multiple_joins": sum(x["positive_with_multiple_joins"] for x in rows),
            "positive_only_with_same_label_joins": sum(x["positive_only_with_same_label_joins"] for x in rows),
            "positive_touching_silence": sum(x["positive_touching_silence"] for x in rows),
        }
    assert all(isclose(row["prevalence"] + row["always_negative_accuracy"], 1)
               for row in aggregate.values() if row["prevalence"] is not None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"analysis_contract": rust.ANALYSIS_PIPELINE_VERSION, "sample_rate": RATE, "hop": HOP,
              "horizon_samples": HORIZONS, "unique_audio_groups": len(groups),
              "actual_student_ticks": False, "dense_boundary_logits": False,
              "aggregate": aggregate, "exports": exports}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{len(exports)} segment exports across {len(groups)} audio groups")
    for name, result in aggregate.items():
        print(name, result)
    print(args.output)


if __name__ == "__main__":
    main()
