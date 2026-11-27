"""Test whether onset density transfers to independently produced section joins."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import runtime_core as rust
import soundfile as sf

from check_101_onset_density import HOP, RATE, SONGS, WINDOW, Track, design, extract, fit_predict
from check_102_boundary_targets import HORIZONS, Timeline, build_timeline
from check_teacher_timelines import read_segments


ROOT = Path(__file__).resolve().parents[2]
HISTORY = "history"
DENSITY = "history+density"
REGIME = "history+regime"
CONFIGURATIONS = (HISTORY, DENSITY, REGIME)
OFFSET_TICKS = 12
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Example:
    track: Track
    timeline: Timeline
    teacher_sha256: str
    features: dict[str, FloatArray]

    @property
    def sample_indices(self) -> IntArray:
        return self.track.indices[WINDOW - 1 :]


def read_example(audio: Path, teacher: Path) -> Example:
    with sf.SoundFile(str(audio)) as source:
        audio_end = source.frames * RATE // source.samplerate
        seconds = (source.frames + source.samplerate - 1) // source.samplerate
    track = extract(audio, seconds)
    timeline = build_timeline(read_segments(teacher), audio_end)
    features = {name: design(track, name)[WINDOW - 1 :] for name in CONFIGURATIONS}
    return Example(track, timeline, sha256(teacher.read_bytes()).hexdigest(), features)


def target_rows(example: Example, horizon: int, offset_ticks: int = 0) -> IntArray:
    """A -1 row has unknown coverage and takes no part in fitting or scoring."""
    offset = offset_ticks * HOP
    labels = []
    for sample in example.sample_indices:
        start = int(sample) + offset
        joins = example.timeline.target(start, start + horizon)
        labels.append(-1 if joins is None else int(bool(joins)))
    return np.asarray(labels, dtype=np.int64)


def average_precision(labels: IntArray, scores: FloatArray) -> float:
    """Step-integrated precision–recall, grouping tied prediction thresholds."""
    ranked = np.argsort(-scores, kind="stable")
    ordered = scores[ranked]
    ends = np.flatnonzero(np.r_[ordered[:-1] != ordered[1:], True])
    true_positives = np.cumsum(labels[ranked])[ends]
    positives = int(true_positives[-1])
    if not 0 < positives < len(labels):
        raise ValueError("Average precision requires positives and negatives")
    at_threshold = true_positives / (ends + 1)
    increments = np.diff(true_positives, prepend=0)
    return float(increments @ at_threshold / positives)


def shifted_score_ap(labels: IntArray, scores: FloatArray) -> list[float]:
    # Shift the ordered sequence of supported rows only. A cyclic shift is a
    # temporal contrast; its overlapping rows do not supply independent trials.
    return [average_precision(labels, np.roll(scores, len(labels) * k // 10))
            for k in range(1, 10)]


def evaluate(examples: Sequence[Example], horizon: int) -> list[dict]:
    targets = [target_rows(example, horizon) for example in examples]
    folded = []
    for fold, example in enumerate(examples):
        y = targets[fold]
        supported = y >= 0
        score_y = y[supported]
        rows = []
        offset_targets = [target_rows(example, horizon, k) for k in (-OFFSET_TICKS, OFFSET_TICKS)]
        common_support = supported & np.logical_and.reduce([labels >= 0 for labels in offset_targets])
        others = [i for i in range(len(examples)) if i != fold]
        train_y = np.concatenate([targets[i][targets[i] >= 0] for i in others]).astype(np.float64)
        for configuration in CONFIGURATIONS:
            x = [item.features[configuration] for item in examples]
            train_x = np.concatenate([x[i][targets[i] >= 0] for i in others])
            scores = fit_predict(train_x, train_y, x[fold])
            ap = average_precision(score_y, scores[supported])
            shifts = shifted_score_ap(score_y, scores[supported])
            rows.append({
                "configuration": configuration,
                "average_precision": ap,
                "shifted_score_ap": shifts,
                "shifted_score_median_ap": float(np.median(shifts)),
                "ap_at_common_support": average_precision(y[common_support], scores[common_support]),
                "ap_at_minus_and_plus_offset": [
                    average_precision(labels[common_support], scores[common_support])
                    for labels in offset_targets
                ],
            })
        folded.append({
            "held_out_song": example.track.name,
            "rows": int(supported.sum()),
            "masked_rows": int((~supported).sum()),
            "positives": int(score_y.sum()),
            "prevalence": float(score_y.mean()),
            "offset_common_rows": int(common_support.sum()),
            "results": rows,
        })
    return folded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "backend/data/audio")
    parser.add_argument("--segments-dir", type=Path, default=ROOT / "outputs_songformer")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/onset-readiness/results.json")
    args = parser.parse_args()
    assert (rust.FEATURE_VERSION, rust.SAMPLE_RATE, rust.HOP_LENGTH) == ("features/2", RATE, HOP)
    assert average_precision(np.array([1, 0, 1, 0]), np.zeros(4)) == 0.5
    assert average_precision(np.array([1, 0, 1, 0]), np.array([3.0, 2.0, 3.0, 1.0])) == 1.0
    examples = []
    for name in SONGS:
        example = read_example(args.audio_dir / f"{name}.mp3",
                               args.segments_dir / name / f"{name}.json")
        examples.append(example)
        print(f"{name}: {len(example.sample_indices)} eligible clock rows, "
              f"{len(example.timeline.joins)} saved joins", flush=True)
    results = []
    for horizon in HORIZONS:
        folds = evaluate(examples, horizon)
        means = {config: float(np.mean([
            next(row["average_precision"] for row in fold["results"] if row["configuration"] == config)
            for fold in folds
        ])) for config in CONFIGURATIONS}
        results.append({"horizon_samples": horizon, "folds": folds, "macro_ap": means})
        print(f"{horizon / RATE:g} s | " + " ".join(f"{k}={v:.5f}" for k, v in means.items()), flush=True)
    report = {
        "schema": "onset-density-songformer-readiness/1",
        "contracts": {"features": rust.FEATURE_VERSION, "analysis": rust.ANALYSIS_PIPELINE_VERSION,
                      "cycle_bank": rust.CYCLE_BANK_VERSION, "sample_rate": RATE, "hop": HOP},
        "offset_ticks": OFFSET_TICKS,
        "tracks": [{"song": e.track.name, "audio_sha256": e.track.digest,
                    "teacher_sha256": e.teacher_sha256, "native_sample_rate": e.track.native_rate,
                    "teacher_joins": len(e.timeline.joins)} for e in examples],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
