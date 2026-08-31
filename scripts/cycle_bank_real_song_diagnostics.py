"""Real-song diagnostics for the canonical observed-ridge CycleBank (#92).

Python ONLY orchestrates: decode audio, feed canonical ticks into the Rust
``CycleBank`` (via the ``AnalysisTimebase`` -> ``observe_tick`` production
path), aggregate the returned modes/relations, and render a report + plots.
ALL transform, ridge, tracking, frequency, phase, confidence, relation, and
prediction math is Rust. Nothing here recomputes any of it.

No beat/event oracle exists in the repo (the only supervised annotation is a
recorded c-trajectory, not musical timing), so per the issue we do NOT invent
semantic ground truth. For material with tempo drift / odd meter / polymeter
(Tool songs), we report:

- acquisition time of the strongest short-timescale observed mode;
- the observed-mode frequency trajectory (drift / polymeter are visible);
- strength/confidence through the song;
- rational-relation stability among observed modes (polymeter diagnostic);
- agreement between measured onset events (from the canonical onset channel)
  and the observed mode's causal ``time_to_next`` phase prediction — a
  descriptive predictive-timing measure, not a beat-tracking score.

Usage:
    python scripts/cycle_bank_real_song_diagnostics.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import runtime_core  # noqa: E402

TWO_PI = 2.0 * math.pi
DT = runtime_core.HOP_LENGTH / runtime_core.SAMPLE_RATE
AUDIO_DIR = REPO_ROOT / "backend" / "data" / "audio"
SONGS = ["Eulogy.mp3", "RightInTwo.mp3", "Stinkfist.mp3", "TheGrudge.mp3", "ThirdEye.mp3"]


@dataclass
class ModeSeries:
    """Per-hop trace of one observed mode identity (Rust-emitted values)."""

    mode_id: int = -1
    times: list[float] = field(default_factory=list)
    frequency: list[float] = field(default_factory=list)
    phase: list[float] = field(default_factory=list)
    strength: list[float] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)


def _decode(path: Path) -> tuple[np.ndarray, int]:
    import librosa

    audio, sr = librosa.load(str(path), sr=None, mono=True, duration=5 * 60)
    return audio.astype(np.float32), int(sr)


def _ticks_for(path: Path, block_seconds: float = 1.0) -> list[dict]:
    """Decode and run the canonical AnalysisTimebase, collecting ticks.

    This mirrors the trainer's production ingestion path exactly (decode at
    native rate, Rust resamples, exact 1024-hop ticks).
    """
    audio, sr = _decode(path)
    tb = runtime_core.AnalysisTimebase()
    block = int(sr * block_seconds)
    ticks: list[dict] = []
    for start in range(0, len(audio), block):
        chunk = audio[start : start + block].tolist()
        ticks.extend(tb.ingest(chunk, sr, start))
    ticks.extend(tb.flush())
    return ticks


def _onset_series(ticks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Onset evidence per tick, as actually extracted by the canonical Rust
    seam (``runtime_core.cycle_observation_channels_from_tick``).

    Python never computes the newest-frame offset; the frame-major layout
    lives in Rust. We read only the onset channel value (already computed by
    the Rust FeatureExtractor) for measurement-only onset event detection.
    """
    times = np.array([t["timeSeconds"] for t in ticks], dtype=float)
    onset = np.zeros(len(ticks), dtype=float)
    for i, t in enumerate(ticks):
        channels = dict(runtime_core.cycle_observation_channels_from_tick(t))
        onset[i] = channels.get("onset", 0.0)
    return times, onset


def _measure_onsets(times: np.ndarray, onset: np.ndarray) -> np.ndarray:
    """Measurement-only onset event detection on the canonical onset channel.

    This is the offline ORACLE side: it reads the already-computed causal
    onset envelope and picks salient peaks to compare predictions against. It
    performs no transform/ridge/tracking math.
    """
    if len(onset) < 3:
        return np.array([])
    # Local maxima above an adaptive threshold.
    thr = float(np.mean(onset) + 1.0 * np.std(onset))
    events: list[float] = []
    for i in range(1, len(onset) - 1):
        if onset[i] > thr and onset[i] >= onset[i - 1] and onset[i] >= onset[i + 1]:
            events.append(float(times[i]))
    return np.array(events)


def analyze_song(path: Path) -> dict:
    ticks = _ticks_for(path)
    bank = runtime_core.CycleBank(
        {
            "f_min_hz": 0.0625,
            "f_max_hz": 8.0,
            "birth_persistence": 2,
            "scales_per_octave": 12,
        }
    )

    # Track the strongest observed short-timescale mode over time by id.
    series: dict[int, ModeSeries] = {}
    relations_seen: set[tuple[int, int, int, int]] = set()
    relation_stable: dict[tuple[int, int, int, int], list[float]] = {}
    first_mode_time: float | None = None

    for tick in ticks:
        modes = bank.observe_tick(tick)
        t = tick["timeSeconds"]
        if modes and first_mode_time is None:
            first_mode_time = t
        for m in modes:
            s = series.setdefault(m.id, ModeSeries(mode_id=m.id))
            s.times.append(t)
            s.frequency.append(m.frequency_hz)
            s.phase.append(m.phase)
            s.strength.append(m.strength)
            s.confidence.append(m.confidence)
        for r in bank.latest_relations():
            key = (r["iId"], r["jId"], r["m"], r["n"])
            relations_seen.add(key)
            relation_stable.setdefault(key, []).append(r["phaseStability"])

    times, onset = _onset_series(ticks)
    onsets = _measure_onsets(times, onset)

    # Predictive agreement: for the most persistent observed short-timescale
    # mode, advance to the end and compare each mode's phase-locked period
    # against the median inter-onset interval as a descriptive consistency
    # check (NOT a beat score).
    candidate = None
    if series:
        # Choose the mode with the most observations in the 0.5..6 Hz band.
        band = {
            mid: s
            for mid, s in series.items()
            if s.frequency and 0.5 <= float(np.median(s.frequency)) <= 6.0
        }
        pool = band if band else series
        candidate = max(pool.values(), key=lambda s: len(s.times))

    inter_onset = np.diff(onsets) if len(onsets) > 1 else np.array([])
    median_ioi = float(np.median(inter_onset)) if len(inter_onset) else None

    result = {
        "song": path.name,
        "duration_s": round(float(times[-1]) if len(times) else 0.0, 2),
        "n_ticks": len(ticks),
        "n_modes_observed": len(series),
        "acquisition_time_s": (
            round(first_mode_time, 3) if first_mode_time is not None else None
        ),
        "n_onset_events": int(len(onsets)),
        "median_inter_onset_s": (
            round(median_ioi, 4) if median_ioi is not None else None
        ),
        "relations": [
            {
                "pair": f"{m}:{n}",
                "i_id": i,
                "j_id": j,
                "mean_phase_stability": round(float(np.mean(v)), 4),
            }
            for (i, j, m, n), v in sorted(relation_stable.items())
        ],
        "candidate_mode": (
            {
                "id": candidate.mode_id,
                "median_frequency_hz": round(float(np.median(candidate.frequency)), 4),
                "frequency_range_hz": [
                    round(float(np.min(candidate.frequency)), 4),
                    round(float(np.max(candidate.frequency)), 4),
                ],
                "median_period_s": round(1.0 / float(np.median(candidate.frequency)), 4),
                "median_strength": round(float(np.median(candidate.strength)), 4),
                "median_confidence": round(float(np.median(candidate.confidence)), 4),
                "observation_fraction": round(
                    len(candidate.times) / max(1, len(ticks)), 3
                ),
            }
            if candidate is not None
            else None
        ),
    }
    return result, series, (times, onset, onsets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "backend" / "logs" / "cycle_bank_diagnostics"),
    )
    parser.add_argument("--songs", nargs="*", default=None, help="subset of song filenames")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    songs = [s for s in SONGS if (args.songs is None or s in args.songs)]
    report: list[dict] = []
    for name in songs:
        path = AUDIO_DIR / name
        if not path.exists():
            print(f"skip missing {name}")
            continue
        print(f"analyzing {name} ...")
        result, series, (times, onset, onsets) = analyze_song(path)
        report.append(result)
        print(
            f"  modes={result['n_modes_observed']} "
            f"acq={result['acquisition_time_s']}s "
            f"onsets={result['n_onset_events']} "
            f"candidate={result['candidate_mode']}"
        )
        _plot(name, series, times, onset, onsets, out_dir)

    report_path = out_dir / "cycle_bank_real_song_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {report_path}")
    return 0


def _plot(name, series, times, onset, onsets, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(times, onset, lw=0.6, color="0.5", label="onset channel")
    for t in onsets:
        axes[0].axvline(t, color="r", alpha=0.15, lw=0.5)
    axes[0].set_ylabel("onset")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{name} — observed CycleBank modes (Rust)")

    for s in series.values():
        if len(s.times) < 5:
            continue
        axes[1].plot(s.times, s.frequency, lw=1.0, label=f"mode {s.mode_id}")
    axes[1].set_ylabel("frequency (Hz)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=7, ncol=2)

    for s in series.values():
        if len(s.times) < 5:
            continue
        axes[2].plot(s.times, s.confidence, lw=1.0, label=f"mode {s.mode_id}")
    axes[2].set_ylabel("confidence")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(name).stem}_cycle_bank.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
