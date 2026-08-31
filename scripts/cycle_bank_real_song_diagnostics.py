"""Real-song diagnostics for the canonical observed-ridge CycleBank (#92).

This module evaluates the *causal predictive usefulness* of directly observed
CycleBank ridges without turning the diagnostic into a beat tracker.

Canonical estimation and prediction remain Rust-owned:

    native-rate PCM
        -> AnalysisTimebase
        -> canonical AnalysisTick
        -> CycleBank.observe_tick(...)
        -> CycleMode.phase_at(...) / CycleMode.time_to_next(...)

Python is only the offline evaluator/orchestrator. It may decode files, choose
measurement-only onset events, perform circular statistics over Rust-emitted
phases, aggregate errors, and draw plots. It does not reimplement the analytic
transform, ridge extraction/tracking, phase evolution, or event-time prediction.

The onset oracle is intentionally descriptive rather than semantic ground
truth. It is derived from the canonical Rust onset evidence and may inspect the
whole song because it belongs to the measurement side. Oracle event times are
never fed back into CycleBank.

The predictive evaluation is strictly causal:

1. Build the canonical tick stream.
2. Replay the same ticks through the Rust CycleBank and retain immutable
   per-tick CycleMode snapshots.
3. For each oracle event, use the latest CycleBank snapshot STRICTLY BEFORE
   the event. An event that lands exactly on analysis tick n therefore uses the
   state after tick n-1, never state that has already ingested the event.
4. Use an initial calibration interval only to select one observed mode and
   fit the event reference phase for that mode.
5. Freeze the selected mode id + reference phase.
6. For later events, call the frozen mode snapshot's Rust time_to_next()
   method and compare the predicted crossing time with the oracle event.

This is a diagnostic of "does one directly observed ridge form a useful causal
clock for these salient onset events?", not a claim that the selected ridge is
the beat, bar, meter, or phrase clock.

Usage:
    python scripts/cycle_bank_real_song_diagnostics.py
    python scripts/cycle_bank_real_song_diagnostics.py --songs ThirdEye.mp3
    python scripts/cycle_bank_real_song_diagnostics.py --calibration-events 24
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import runtime_core  # noqa: E402


AUDIO_DIR = REPO_ROOT / "backend" / "data" / "audio"
SONGS = [
    "Eulogy.mp3",
    "RightInTwo.mp3",
    "Stinkfist.mp3",
    "TheGrudge.mp3",
    "ThirdEye.mp3",
]

DEFAULT_CALIBRATION_EVENTS = 24
DEFAULT_MIN_CANDIDATE_HITS = 6
SHORT_TIMESCALE_MIN_HZ = 0.5
SHORT_TIMESCALE_MAX_HZ = 6.0

# Any iterable of onset times in seconds. Accepts both Python sequences and
# numpy arrays, since callers may have either (np.diff output vs. Python list).
OnsetTimes = Union[Sequence[float], NDArray[np.float64]]


@dataclass
class ModeSeries:
    """Whole-song plotting trace for one Rust mode identity within one epoch."""

    stream_epoch: int = 0
    mode_id: int = -1
    times: list[float] = field(default_factory=list)
    frequency: list[float] = field(default_factory=list)
    phase: list[float] = field(default_factory=list)
    strength: list[float] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TickSnapshot:
    """Immutable post-tick CycleBank snapshot.

    A prediction for an oracle event at time t MUST use a TickSnapshot whose
    time_seconds is strictly less than t.
    """

    time_seconds: float
    stream_epoch: int
    modes: tuple[Any, ...]


@dataclass
class _CandidateCalibration:
    phases: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)


def _decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode at native sample rate; Rust owns canonical resampling."""

    import librosa

    audio, sr = librosa.load(str(path), sr=None, mono=True, duration=5 * 60)
    return audio.astype(np.float32), int(sr)


def _ticks_for(path: Path, block_seconds: float = 1.0) -> list[dict[str, Any]]:
    """Run the canonical AnalysisTimebase over native-rate decoded PCM."""

    audio, sr = _decode(path)
    tb = runtime_core.AnalysisTimebase()
    block = max(1, int(sr * block_seconds))
    ticks: list[dict[str, Any]] = []
    for start in range(0, len(audio), block):
        chunk = audio[start : start + block].tolist()
        ticks.extend(tb.ingest(chunk, sr, start))
    ticks.extend(tb.flush())
    return ticks


def _onset_series(ticks: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Read the Rust-owned canonical onset evidence for measurement only."""

    times = np.array([float(t["timeSeconds"]) for t in ticks], dtype=float)
    onset = np.zeros(len(ticks), dtype=float)
    for i, tick in enumerate(ticks):
        channels = dict(runtime_core.cycle_observation_channels_from_tick(tick))
        onset[i] = float(channels.get("onset", 0.0))
    return times, onset


def _measure_onsets(times: np.ndarray, onset: np.ndarray) -> np.ndarray:
    """Pick salient local peaks from the canonical onset envelope.

    This is an offline measurement oracle. The adaptive threshold may use the
    complete song because these event times are never supplied to CycleBank.
    The resulting timestamps are quantized to the current analysis-hop clock,
    so 20/30/40 ms results are useful diagnostics rather than sub-hop acoustic
    ground truth.
    """

    if len(onset) < 3:
        return np.array([], dtype=float)

    threshold = float(np.mean(onset) + np.std(onset))
    events: list[float] = []
    for i in range(1, len(onset) - 1):
        if (
            onset[i] > threshold
            and onset[i] >= onset[i - 1]
            and onset[i] >= onset[i + 1]
        ):
            events.append(float(times[i]))
    return np.asarray(events, dtype=float)


def _run_cycle_bank(
    ticks: Sequence[dict[str, Any]],
) -> tuple[
    list[TickSnapshot],
    dict[tuple[int, int], ModeSeries],
    dict[tuple[int, int, int, int, int], list[float]],
    float | None,
]:
    """Replay canonical ticks through the Rust CycleBank exactly once.

    The returned snapshots are causal immutable views: although the complete
    list is retained offline, each snapshot was produced without future input.
    Evaluation helpers below mechanically use only snapshots strictly preceding
    each oracle event.
    """

    bank = runtime_core.CycleBank(
        {
            "f_min_hz": 0.0625,
            "f_max_hz": 8.0,
            "birth_persistence": 2,
            "scales_per_octave": 12,
        }
    )

    snapshots: list[TickSnapshot] = []
    series: dict[tuple[int, int], ModeSeries] = {}
    relation_stability: dict[tuple[int, int, int, int, int], list[float]] = {}
    first_mode_time: float | None = None

    for tick in ticks:
        t = float(tick["timeSeconds"])
        epoch = int(tick["streamEpoch"])
        modes = tuple(bank.observe_tick(tick))
        snapshots.append(TickSnapshot(t, epoch, modes))

        if modes and first_mode_time is None:
            first_mode_time = t

        for mode in modes:
            key = (epoch, int(mode.id))
            trace = series.setdefault(
                key,
                ModeSeries(stream_epoch=epoch, mode_id=int(mode.id)),
            )
            trace.times.append(t)
            trace.frequency.append(float(mode.frequency_hz))
            trace.phase.append(float(mode.phase))
            trace.strength.append(float(mode.strength))
            trace.confidence.append(float(mode.confidence))

        for relation in bank.latest_relations():
            rel_key = (
                epoch,
                int(relation["iId"]),
                int(relation["jId"]),
                int(relation["m"]),
                int(relation["n"]),
            )
            relation_stability.setdefault(rel_key, []).append(
                float(relation["phaseStability"])
            )

    return snapshots, series, relation_stability, first_mode_time


def _strictly_previous_snapshot(
    snapshots: Sequence[TickSnapshot],
    snapshot_times: Sequence[float],
    event_time: float,
) -> TickSnapshot | None:
    """Return the latest snapshot with snapshot.time_seconds < event_time.

    bisect_left is deliberate. bisect_right would allow the state *at* an onset
    tick to predict that same onset after the event evidence had already been
    ingested, which is exactly the causal leak this diagnostic exists to avoid.
    """

    index = bisect.bisect_left(snapshot_times, event_time) - 1
    if index < 0:
        return None
    return snapshots[index]


def _circular_mean_concentration(phases: Sequence[float]) -> tuple[float, float]:
    """Offline circular statistics over Rust-projected event phases."""

    if not phases:
        raise ValueError("phases must be non-empty")
    vector = sum(complex(math.cos(p), math.sin(p)) for p in phases)
    mean_phase = math.atan2(vector.imag, vector.real)
    concentration = abs(vector) / len(phases)
    return mean_phase, float(concentration)


def _mode_in_snapshot(snapshot: TickSnapshot, mode_id: int) -> Any | None:
    return next((m for m in snapshot.modes if int(m.id) == mode_id), None)


def _calibrate_observed_mode(
    onsets: OnsetTimes,
    snapshots: Sequence[TickSnapshot],
    *,
    calibration_events: int = DEFAULT_CALIBRATION_EVENTS,
    min_candidate_hits: int = DEFAULT_MIN_CANDIDATE_HITS,
    min_hz: float = SHORT_TIMESCALE_MIN_HZ,
    max_hz: float = SHORT_TIMESCALE_MAX_HZ,
) -> dict[str, Any]:
    """Select one directly observed mode using calibration data only.

    Calibration continues until `calibration_events` oracle events with at
    least one eligible strictly-preceding mode snapshot have been observed.
    Pre-acquisition events are skipped rather than causing future state to be
    consulted.

    Candidate score:
        event coverage * circular phase concentration * mean confidence

    The projected event phase for every candidate is obtained by calling the
    Rust CycleMode.phase_at() method. Python never reconstructs phase dynamics.
    """

    if calibration_events <= 0:
        raise ValueError("calibration_events must be > 0")
    if min_candidate_hits <= 0:
        raise ValueError("min_candidate_hits must be > 0")
    if not snapshots:
        return {
            "status": "insufficient_data",
            "reason": "no_cyclebank_snapshots",
            "n_calibration_opportunities": 0,
            "n_skipped_pre_acquisition": len(onsets),
        }

    epochs = {s.stream_epoch for s in snapshots}
    if len(epochs) != 1:
        return {
            "status": "unsupported",
            "reason": "multiple_stream_epochs",
            "n_calibration_opportunities": 0,
            "n_skipped_pre_acquisition": 0,
        }
    stream_epoch = next(iter(epochs))

    snapshot_times = [s.time_seconds for s in snapshots]
    stats: dict[int, _CandidateCalibration] = {}

    opportunities = 0
    skipped_pre_acquisition = 0
    last_calibration_event_index: int | None = None

    for event_index, raw_event_time in enumerate(onsets):
        event_time = float(raw_event_time)
        snapshot = _strictly_previous_snapshot(snapshots, snapshot_times, event_time)
        if snapshot is None:
            skipped_pre_acquisition += 1
            continue

        eligible = [
            mode
            for mode in snapshot.modes
            if min_hz <= float(mode.frequency_hz) <= max_hz
        ]
        if not eligible:
            skipped_pre_acquisition += 1
            continue

        opportunities += 1
        last_calibration_event_index = event_index
        delta = event_time - snapshot.time_seconds
        assert delta > 0.0, "strict predecessor must produce positive lead time"

        for mode in eligible:
            mode_id = int(mode.id)
            candidate = stats.setdefault(mode_id, _CandidateCalibration())
            # Canonical future phase prediction is Rust-owned.
            candidate.phases.append(float(mode.phase_at(delta)))
            candidate.confidences.append(float(mode.confidence))

        if opportunities >= calibration_events:
            break

    if opportunities < calibration_events or last_calibration_event_index is None:
        return {
            "status": "insufficient_data",
            "reason": "not_enough_calibration_events_with_modes",
            "stream_epoch": stream_epoch,
            "n_calibration_opportunities": opportunities,
            "n_calibration_events_requested": calibration_events,
            "n_skipped_pre_acquisition": skipped_pre_acquisition,
        }

    candidate_rows: list[dict[str, Any]] = []
    for mode_id, candidate in stats.items():
        hits = len(candidate.phases)
        if hits < min_candidate_hits:
            continue
        phi_event, concentration = _circular_mean_concentration(candidate.phases)
        mean_confidence = float(np.mean(candidate.confidences))
        coverage = hits / opportunities
        score = coverage * concentration * mean_confidence
        candidate_rows.append(
            {
                "mode_id": mode_id,
                "hits": hits,
                "coverage": coverage,
                "phi_event": phi_event,
                "phase_concentration": concentration,
                "mean_confidence": mean_confidence,
                "score": score,
            }
        )

    if not candidate_rows:
        return {
            "status": "insufficient_data",
            "reason": "no_candidate_met_min_hits",
            "stream_epoch": stream_epoch,
            "n_calibration_opportunities": opportunities,
            "n_calibration_events_requested": calibration_events,
            "n_skipped_pre_acquisition": skipped_pre_acquisition,
        }

    candidate_rows.sort(
        key=lambda row: (
            row["score"],
            row["hits"],
            row["phase_concentration"],
            row["mean_confidence"],
            -row["mode_id"],
        ),
        reverse=True,
    )
    winner = candidate_rows[0]

    return {
        "status": "ok",
        "reason": None,
        "stream_epoch": stream_epoch,
        "candidate_mode_id": winner["mode_id"],
        "phi_event": winner["phi_event"],
        "phase_concentration": winner["phase_concentration"],
        "mean_calibration_confidence": winner["mean_confidence"],
        "calibration_coverage": winner["coverage"],
        "candidate_score": winner["score"],
        "candidate_hits": winner["hits"],
        "n_calibration_opportunities": opportunities,
        "n_calibration_events_requested": calibration_events,
        "n_skipped_pre_acquisition": skipped_pre_acquisition,
        "last_calibration_event_index": last_calibration_event_index,
        "calibration_cutoff_s": float(onsets[last_calibration_event_index]),
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _predict_one_event(
    snapshot: TickSnapshot,
    *,
    event_time: float,
    mode_id: int,
    phi_event: float,
) -> tuple[float, float] | None:
    """Predict one event from an already-selected strictly preceding snapshot.

    Returns (predicted_time, signed_error_seconds), or None if the frozen mode
    is unavailable. The lead time itself comes from Rust time_to_next().
    """

    if not snapshot.time_seconds < event_time:
        raise ValueError("snapshot must strictly precede event_time")

    mode = _mode_in_snapshot(snapshot, mode_id)
    if mode is None:
        return None

    lead = mode.time_to_next(phi_event)
    if lead is None:
        return None

    predicted_time = snapshot.time_seconds + float(lead)
    return predicted_time, predicted_time - event_time


def _score_frozen_candidate(
    onsets: OnsetTimes,
    snapshots: Sequence[TickSnapshot],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Score later oracle events without changing the calibrated candidate."""

    if calibration.get("status") != "ok":
        return {
            "status": calibration.get("status", "insufficient_data"),
            "reason": calibration.get("reason", "calibration_failed"),
            "n_evaluation_events": 0,
            "n_predictions": 0,
            "prediction_coverage": 0.0,
        }

    mode_id = int(calibration["candidate_mode_id"])
    phi_event = float(calibration["phi_event"])
    last_calibration_event_index = int(calibration["last_calibration_event_index"])
    stream_epoch = int(calibration["stream_epoch"])

    snapshot_times = [s.time_seconds for s in snapshots]
    errors: list[float] = []
    skipped_no_snapshot = 0
    skipped_mode_missing = 0

    evaluation_events = list(onsets[last_calibration_event_index + 1 :])

    for raw_event_time in evaluation_events:
        event_time = float(raw_event_time)
        snapshot = _strictly_previous_snapshot(snapshots, snapshot_times, event_time)
        if snapshot is None:
            skipped_no_snapshot += 1
            continue
        if snapshot.stream_epoch != stream_epoch:
            # Frozen mode identity is epoch-local. Do not silently attach the
            # old id to a different post-reset physical mode.
            skipped_mode_missing += 1
            continue

        prediction = _predict_one_event(
            snapshot,
            event_time=event_time,
            mode_id=mode_id,
            phi_event=phi_event,
        )
        if prediction is None:
            skipped_mode_missing += 1
            continue

        _, error = prediction
        errors.append(error)

    abs_errors = [abs(e) for e in errors]
    n_eval = len(evaluation_events)
    n_predictions = len(errors)
    coverage = n_predictions / n_eval if n_eval else 0.0

    return {
        "status": "ok" if n_predictions else "insufficient_data",
        "reason": None if n_predictions else "frozen_candidate_produced_no_predictions",
        "n_evaluation_events": n_eval,
        "n_predictions": n_predictions,
        "prediction_coverage": coverage,
        "n_skipped_no_preceding_snapshot": skipped_no_snapshot,
        "n_skipped_candidate_missing": skipped_mode_missing,
        "signed_timing_error_s": _distribution(errors),
        "abs_timing_error_s": _distribution(abs_errors),
        "fraction_within": {
            "within_20_ms": (
                float(np.mean(np.asarray(abs_errors) <= 0.020)) if abs_errors else None
            ),
            "within_30_ms": (
                float(np.mean(np.asarray(abs_errors) <= 0.030)) if abs_errors else None
            ),
            "within_40_ms": (
                float(np.mean(np.asarray(abs_errors) <= 0.040)) if abs_errors else None
            ),
        },
    }


def _causal_timing_evaluation(
    onsets: OnsetTimes,
    snapshots: Sequence[TickSnapshot],
    *,
    calibration_events: int = DEFAULT_CALIBRATION_EVENTS,
    min_candidate_hits: int = DEFAULT_MIN_CANDIDATE_HITS,
) -> dict[str, Any]:
    """Calibrate on the past, freeze, then score causal future predictions."""

    calibration = _calibrate_observed_mode(
        onsets,
        snapshots,
        calibration_events=calibration_events,
        min_candidate_hits=min_candidate_hits,
    )

    result: dict[str, Any] = {
        "oracle": "canonical_onset_local_maxima",
        "oracle_time_resolution_note": (
            "Oracle events are sampled on the canonical analysis-hop timeline; "
            "20/30/40 ms rates are diagnostics, not sub-hop acoustic ground truth."
        ),
        "n_oracle_events": len(onsets),
        "calibration": calibration,
    }

    if calibration.get("status") != "ok":
        result["status"] = calibration.get("status", "insufficient_data")
        result["reason"] = calibration.get("reason", "calibration_failed")
        result["evaluation"] = _score_frozen_candidate(onsets, snapshots, calibration)
        return result

    evaluation = _score_frozen_candidate(onsets, snapshots, calibration)
    result["status"] = evaluation["status"]
    result["reason"] = evaluation["reason"]
    result["evaluation"] = evaluation
    return result


def _relation_report(
    relation_stability: dict[tuple[int, int, int, int, int], list[float]],
) -> list[dict[str, Any]]:
    return [
        {
            "stream_epoch": epoch,
            "pair": f"{m}:{n}",
            "i_id": i_id,
            "j_id": j_id,
            "mean_phase_stability": float(np.mean(values)),
        }
        for (epoch, i_id, j_id, m, n), values in sorted(relation_stability.items())
    ]


def _candidate_summary(
    series: dict[tuple[int, int], ModeSeries],
    timing: dict[str, Any],
    n_ticks: int,
) -> dict[str, Any] | None:
    calibration = timing.get("calibration", {})
    if calibration.get("status") != "ok":
        return None

    key = (
        int(calibration["stream_epoch"]),
        int(calibration["candidate_mode_id"]),
    )
    trace = series.get(key)
    if trace is None or not trace.frequency:
        return None

    return {
        "stream_epoch": trace.stream_epoch,
        "id": trace.mode_id,
        "median_frequency_hz": float(np.median(trace.frequency)),
        "frequency_range_hz": [
            float(np.min(trace.frequency)),
            float(np.max(trace.frequency)),
        ],
        "median_period_s": 1.0 / float(np.median(trace.frequency)),
        "median_strength": float(np.median(trace.strength)),
        "median_confidence": float(np.median(trace.confidence)),
        "observation_fraction": len(trace.times) / max(1, n_ticks),
    }


def analyze_song(
    path: Path,
    *,
    calibration_events: int = DEFAULT_CALIBRATION_EVENTS,
    min_candidate_hits: int = DEFAULT_MIN_CANDIDATE_HITS,
) -> tuple[
    dict[str, Any],
    dict[tuple[int, int], ModeSeries],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Run the production CycleBank path and the causal offline evaluation."""

    ticks = _ticks_for(path)
    times, onset = _onset_series(ticks)
    onsets = _measure_onsets(times, onset)

    snapshots, series, relation_stability, first_mode_time = _run_cycle_bank(ticks)
    timing = _causal_timing_evaluation(
        onsets,
        snapshots,
        calibration_events=calibration_events,
        min_candidate_hits=min_candidate_hits,
    )

    inter_onset = np.diff(onsets) if len(onsets) > 1 else np.array([], dtype=float)
    median_ioi = float(np.median(inter_onset)) if len(inter_onset) else None

    result = {
        "song": path.name,
        "duration_s": float(times[-1]) if len(times) else 0.0,
        "n_ticks": len(ticks),
        "n_modes_observed": len(series),
        "acquisition_time_s": first_mode_time,
        "n_onset_events": int(len(onsets)),
        "median_inter_onset_s": median_ioi,
        "relations": _relation_report(relation_stability),
        "candidate_mode": _candidate_summary(series, timing, len(ticks)),
        "timing_evaluation": timing,
    }
    return result, series, (times, onset, onsets)


def _plot(
    name: str,
    series: dict[tuple[int, int], ModeSeries],
    times: np.ndarray,
    onset: np.ndarray,
    onsets: np.ndarray,
    out_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(times, onset, lw=0.6, color="0.5", label="onset oracle source")
    for event_time in onsets:
        axes[0].axvline(event_time, color="r", alpha=0.15, lw=0.5)
    axes[0].set_ylabel("onset")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{name} — observed CycleBank modes (Rust)")

    for trace in series.values():
        if len(trace.times) < 5:
            continue
        axes[1].plot(
            trace.times,
            trace.frequency,
            lw=1.0,
            label=f"e{trace.stream_epoch}:m{trace.mode_id}",
        )
    axes[1].set_ylabel("frequency (Hz)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=7, ncol=2)

    for trace in series.values():
        if len(trace.times) < 5:
            continue
        axes[2].plot(
            trace.times,
            trace.confidence,
            lw=1.0,
            label=f"e{trace.stream_epoch}:m{trace.mode_id}",
        )
    axes[2].set_ylabel("confidence")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(name).stem}_cycle_bank.png", dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "backend" / "logs" / "cycle_bank_diagnostics"),
    )
    parser.add_argument(
        "--songs",
        nargs="*",
        default=None,
        help="subset of song filenames",
    )
    parser.add_argument(
        "--calibration-events",
        type=int,
        default=DEFAULT_CALIBRATION_EVENTS,
        help="number of causal onset opportunities used before freezing the candidate",
    )
    parser.add_argument(
        "--min-candidate-hits",
        type=int,
        default=DEFAULT_MIN_CANDIDATE_HITS,
        help="minimum calibration events supporting a candidate mode",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    songs = [s for s in SONGS if args.songs is None or s in args.songs]
    report: list[dict[str, Any]] = []

    for name in songs:
        path = AUDIO_DIR / name
        if not path.exists():
            print(f"skip missing {name}")
            continue

        print(f"analyzing {name} ...")
        result, series, (times, onset, onsets) = analyze_song(
            path,
            calibration_events=args.calibration_events,
            min_candidate_hits=args.min_candidate_hits,
        )
        report.append(result)

        timing = result["timing_evaluation"]
        calibration = timing.get("calibration", {})
        evaluation = timing.get("evaluation", {})
        print(
            f"  modes={result['n_modes_observed']} "
            f"acq={result['acquisition_time_s']}s "
            f"onsets={result['n_onset_events']} "
            f"candidate={calibration.get('candidate_mode_id')} "
            f"R={calibration.get('phase_concentration')} "
            f"coverage={evaluation.get('prediction_coverage')} "
            f"median_abs={evaluation.get('abs_timing_error_s', {}).get('median')}"
        )

        _plot(name, series, times, onset, onsets, out_dir)

    report_path = out_dir / "cycle_bank_real_song_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
