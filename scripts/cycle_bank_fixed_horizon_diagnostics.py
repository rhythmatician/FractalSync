"""Fixed-horizon causal prediction diagnostics for the canonical CycleBank (#99).

Follow-up to PR #98 / issue #92. PR #98 measured whether the observed-ridge
CycleBank could predict each salient onset from the *latest* snapshot strictly
preceding that onset (~one canonical hop). Issue #99 asks the same question
at fixed horizons *before* the event:

    H ∈ {20, 50, 100, 200, 500} ms (default)

i.e. "how far ahead can a directly observed ridge still be useful?".

Canonical Rust authority is unchanged:

    native-rate PCM
        -> AnalysisTimebase
        -> AnalysisTick
        -> CycleBank.observe_tick(...)
        -> CycleMode.phase_at(...) / CycleMode.time_to_next(...)

Python remains an offline evaluator/orchestrator. It does not reimplement the
transform, ridge extraction/tracking, phase evolution, or event-time
prediction. The frozen candidate (mode id + reference event phase) is still
chosen by `_calibrate_observed_mode` from PR #98; this module only changes
*which* snapshot the prediction query is read from per horizon.

Strict causal rule for issue #99:

    For each oracle event at time t_event and horizon H:
        snapshot.time_seconds <= t_event - H

    The frozen mode's time_to_next(reference_phase) is then asked of THAT
    snapshot, and the resulting predicted crossing time is compared with
    t_event. No state at or after (t_event - H) is ever consulted for that
    prediction.

Cycle-ambiguity rule (issue #99 anticipated this explicitly):

    v1 prediction returns the *next* crossing of `reference_phase` after the
    snapshot. If the time horizon is at least one full mode period, that
    "next crossing" identifies only the first recurrence; it cannot tell us
    whether the oracle event is the first, second, third, ... recurrence
    after the snapshot. Such events are reported as
    `n_skipped_cycle_ambiguous` and never headline conditional accuracy.

    The skipping condition is:

        (t_event - t_snapshot) >= 1 / f_mode

    A 2 Hz mode at a 500 ms horizon is therefore an AMBIGUITY test (skipped),
    not a misleading timing-error test. A 2 Hz mode at a 50 ms horizon is
    fine.

The evaluator mechanically proves causality by using only TickSnapshot.time_seconds
values produced before the chosen horizon snapshot can be mutated for that
event. Synthetic tests verify the rule under explicit future mutation.

This module is offline-evaluation only. It does not change Player/model I/O,
CycleBank state shape, or any production path.

Usage:

    python scripts/cycle_bank_fixed_horizon_diagnostics.py
    python scripts/cycle_bank_fixed_horizon_diagnostics.py --songs ThirdEye.mp3
    python scripts/cycle_bank_fixed_horizon_diagnostics.py --horizons-ms 20 50 100 200 500
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cycle_bank_real_song_diagnostics as base  # noqa: E402

# Reuse the production analyzer (decoding, ticks, onset measurement, CycleBank
# replay) so this module never diverges from PR #98 on the canonical path.
AUDIO_DIR = base.AUDIO_DIR
DEFAULT_SONGS = base.SONGS
TickSnapshot = base.TickSnapshot
OnsetTimes = base.OnsetTimes


# Initial horizons from issue #99.  The 20 ms point is meant to bridge to PR
# #98's near-event B3 result; longer horizons reveal actual anticipatory
# usefulness.  Callers may override via --horizons-ms.
DEFAULT_HORIZONS_MS: tuple[int, ...] = (20, 50, 100, 200, 500)


@dataclass(frozen=True)
class HorizonSpec:
    """One requested prediction horizon H, measured in seconds."""

    horizon_ms: int

    def __post_init__(self) -> None:
        if self.horizon_ms <= 0:
            raise ValueError(f"horizon must be > 0 ms, got {self.horizon_ms}")

    @property
    def horizon_s(self) -> float:
        return self.horizon_ms / 1000.0


@dataclass
class HorizonResult:
    """Per-horizon, per-song evaluation outcome.

    Issue #99 wants coverage and conditional accuracy reported separately
    at every horizon. We decompose the "did not produce a prediction" set
    so the report can say *why* coverage fell, not just how far.
    """

    horizon_ms: int
    status: str
    reason: str | None
    n_oracle_events: int
    n_evaluation_opportunities: int
    n_predictions: int
    prediction_coverage: float
    # Skipped-event decomposition (issue #99 wants these reported distinctly):
    n_skipped_event_too_early: int  # calibration-eligible (PR #98 used these)
    n_skipped_no_pre_horizon_snapshot: int  # no snapshot <= cutoff
    n_skipped_candidate_missing: int  # mode not in selected snapshot
    n_skipped_cycle_ambiguous: (
        int  # horizon >= mode period; phase can't pick recurrence
    )
    signed_timing_error_s: dict[str, float | None]
    abs_timing_error_s: dict[str, float | None]
    fraction_within: dict[str, float | None]
    lead_time_s: dict[str, float | None]


@dataclass
class SongHorizonReport:
    """Bundle of one song's calibration, frozen candidate, and per-horizon metrics."""

    song: str
    duration_s: float
    n_oracle_events: int
    calibration: dict[str, Any]
    candidate_mode: dict[str, Any] | None
    horizons: list[HorizonResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Snapshot selection at fixed horizon
# ---------------------------------------------------------------------------


def _snapshot_at_or_before(
    snapshots: Sequence[TickSnapshot],
    snapshot_times: Sequence[float],
    cutoff_time: float,
) -> TickSnapshot | None:
    """Return the latest snapshot with snapshot.time_seconds <= cutoff_time.

    bisect_right - 1 is the standard "latest value <= key" idiom. The strict
    causal rule for issue #99 uses `snapshot.time_seconds <= t_event - H`; we
    still want the *latest* such snapshot (not the earliest), because that
    gives the most recent state available without violating causality.
    """

    index = bisect.bisect_right(snapshot_times, cutoff_time) - 1
    if index < 0:
        return None
    return snapshots[index]


def _mode_period_s(mode: Any) -> float | None:
    """Return 1 / frequency_hz for a Rust/Fake cycle mode, or None if invalid."""

    f = float(getattr(mode, "frequency_hz"))
    if not (math.isfinite(f) and f > 0.0):
        return None
    return 1.0 / f


def _predict_one_event_at_horizon(
    snapshot: TickSnapshot,
    *,
    event_time: float,
    horizon_s: float,
    mode_id: int,
    phi_event: float,
) -> tuple[float, float, float] | None:
    """Predict an oracle event from a snapshot selected at horizon `horizon_s`.

    Returns (predicted_time, signed_error_seconds, lead_seconds) or None if
    the frozen mode is unavailable. The snapshot is required to satisfy
    snapshot.time_seconds <= event_time - horizon_s. The lead time itself
    comes from Rust time_to_next(); Python never extrapolates phase.

    Note: this helper does NOT itself detect cycle ambiguity — the caller is
    responsible, because cycle ambiguity is a function of the candidate's
    mode frequency and the snapshot's mode identity.
    """

    if not snapshot.time_seconds <= event_time - horizon_s:
        raise ValueError(
            "snapshot must satisfy snapshot.time_seconds <= event_time - horizon_s"
        )

    mode = base._mode_in_snapshot(snapshot, mode_id)
    if mode is None:
        return None

    lead = mode.time_to_next(phi_event)
    if lead is None:
        return None

    predicted_time = snapshot.time_seconds + float(lead)
    return predicted_time, predicted_time - event_time, float(lead)


# ---------------------------------------------------------------------------
# Per-horizon scoring
# ---------------------------------------------------------------------------


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


def _empty_horizon_result(
    *, horizon: HorizonSpec, n_oracle_events: int, status: str, reason: str
) -> HorizonResult:
    return HorizonResult(
        horizon_ms=horizon.horizon_ms,
        status=status,
        reason=reason,
        n_oracle_events=n_oracle_events,
        n_evaluation_opportunities=0,
        n_predictions=0,
        prediction_coverage=0.0,
        n_skipped_event_too_early=0,
        n_skipped_no_pre_horizon_snapshot=0,
        n_skipped_candidate_missing=0,
        n_skipped_cycle_ambiguous=0,
        signed_timing_error_s=_distribution([]),
        abs_timing_error_s=_distribution([]),
        fraction_within={
            "within_20_ms": None,
            "within_30_ms": None,
            "within_40_ms": None,
        },
        lead_time_s=_distribution([]),
    )


def _evaluate_one_horizon(
    *,
    song_name: str,
    calibration: dict[str, Any],
    onsets: OnsetTimes,
    snapshots: Sequence[TickSnapshot],
    horizon: HorizonSpec,
) -> HorizonResult:
    """Score every evaluation-eligible oracle event at one fixed horizon.

    Contract (issue #99 production parity):
      The caller passes the FULL oracle onset sequence, exactly as production
      passes it to `_causal_timing_evaluation` in PR #98. This evaluator trims
      calibration-eligible events itself via
      `calibration["last_calibration_event_index"]`. Tests should not silently
      double-trim the onset array before passing it in.
    """

    if calibration.get("status") != "ok":
        return _empty_horizon_result(
            horizon=horizon,
            n_oracle_events=len(onsets),
            status="insufficient_data",
            reason=calibration.get("reason", "calibration_failed"),
        )

    mode_id = int(calibration["candidate_mode_id"])
    phi_event = float(calibration["phi_event"])
    last_calibration_event_index = int(calibration["last_calibration_event_index"])
    stream_epoch = int(calibration["stream_epoch"])

    snapshot_times = [s.time_seconds for s in snapshots]
    horizon_s = horizon.horizon_s

    # Calibration-eligible oracle events are events at indices <=
    # last_calibration_event_index. We only ever score the rest.
    n_evaluation_opportunities = max(
        0, len(onsets) - (last_calibration_event_index + 1)
    )
    n_skipped_event_too_early = len(onsets) - n_evaluation_opportunities

    errors: list[float] = []
    leads: list[float] = []
    skipped_no_snapshot = 0
    skipped_candidate_missing = 0
    skipped_cycle_ambiguous = 0

    for raw_event_time in onsets[last_calibration_event_index + 1 :]:
        event_time = float(raw_event_time)

        # Strict causal cutoff: snapshot must be at or before t_event - H.
        snapshot = _snapshot_at_or_before(
            snapshots, snapshot_times, event_time - horizon_s
        )
        if snapshot is None:
            # No snapshot has been observed by the cutoff; this is the
            # legitimate "pre-acquisition" case for short horizons on songs
            # where the bank has not warmed up yet. Report it explicitly.
            skipped_no_snapshot += 1
            continue
        if snapshot.stream_epoch != stream_epoch:
            # Mode identity is epoch-local; never carry it across resets.
            skipped_candidate_missing += 1
            continue

        mode = base._mode_in_snapshot(snapshot, mode_id)
        if mode is None:
            skipped_candidate_missing += 1
            continue

        # Cycle ambiguity: when the lead budget (t_event - t_snapshot) is at
        # least one full mode period, v1 prediction cannot identify which
        # recurrence is the oracle event. The snapshot does not know which
        # integer multiple of the period to add. Skip explicitly rather than
        # produce a misleading large-error prediction.
        period_s = _mode_period_s(mode)
        if period_s is None:
            skipped_candidate_missing += 1
            continue
        available_lead = event_time - snapshot.time_seconds
        if available_lead >= period_s:
            skipped_cycle_ambiguous += 1
            continue

        prediction = _predict_one_event_at_horizon(
            snapshot,
            event_time=event_time,
            horizon_s=horizon_s,
            mode_id=mode_id,
            phi_event=phi_event,
        )
        if prediction is None:
            skipped_candidate_missing += 1
            continue

        _, error, lead = prediction
        errors.append(error)
        leads.append(lead)

    abs_errors = [abs(e) for e in errors]
    n_predictions = len(errors)
    coverage = (
        n_predictions / n_evaluation_opportunities
        if n_evaluation_opportunities
        else 0.0
    )

    return HorizonResult(
        horizon_ms=horizon.horizon_ms,
        status="ok" if n_predictions else "insufficient_data",
        reason=None if n_predictions else "no_predictions_at_horizon",
        n_oracle_events=len(onsets),
        n_evaluation_opportunities=n_evaluation_opportunities,
        n_predictions=n_predictions,
        prediction_coverage=coverage,
        n_skipped_event_too_early=n_skipped_event_too_early,
        n_skipped_no_pre_horizon_snapshot=skipped_no_snapshot,
        n_skipped_candidate_missing=skipped_candidate_missing,
        n_skipped_cycle_ambiguous=skipped_cycle_ambiguous,
        signed_timing_error_s=_distribution(errors),
        abs_timing_error_s=_distribution(abs_errors),
        fraction_within={
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
        lead_time_s=_distribution(leads),
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _horizon_table_rows(
    report: SongHorizonReport,
) -> list[dict[str, Any]]:
    """Compact per-song horizon rows for human-readable tables."""

    rows: list[dict[str, Any]] = []
    for h in report.horizons:
        rows.append(
            {
                "song": report.song,
                "horizon_ms": h.horizon_ms,
                "coverage": h.prediction_coverage,
                "median_abs_ms": (
                    h.abs_timing_error_s["median"] * 1000.0
                    if h.abs_timing_error_s["median"] is not None
                    else None
                ),
                "p90_ms": (
                    h.abs_timing_error_s["p90"] * 1000.0
                    if h.abs_timing_error_s["p90"] is not None
                    else None
                ),
                "within_40ms": h.fraction_within["within_40_ms"],
                "n_predictions": h.n_predictions,
                "n_cycle_ambiguous": h.n_skipped_cycle_ambiguous,
                "n_no_pre_horizon_snapshot": h.n_skipped_no_pre_horizon_snapshot,
                "n_candidate_missing": h.n_skipped_candidate_missing,
                "n_evaluation_opportunities": h.n_evaluation_opportunities,
                "status": h.status,
            }
        )
    return rows


def analyze_song_fixed_horizon(
    path: Path,
    *,
    horizons_ms: Sequence[int] = DEFAULT_HORIZONS_MS,
    calibration_events: int = base.DEFAULT_CALIBRATION_EVENTS,
    min_candidate_hits: int = base.DEFAULT_MIN_CANDIDATE_HITS,
) -> SongHorizonReport:
    """Run canonical CycleBank + fixed-horizon causal evaluation on one song.

    Reuses `analyze_song_bundle` from PR #98 so we do not decode the song
    twice or replay the Rust CycleBank twice. The bundle exposes the same
    immutable snapshots and onset oracle PR #98 already built.
    """

    bundle = base.analyze_song_bundle(
        path,
        calibration_events=calibration_events,
        min_candidate_hits=min_candidate_hits,
    )
    result = bundle["result"]
    snapshots = bundle["snapshots"]
    onsets = bundle["onsets"]
    calibration = result["timing_evaluation"].get("calibration", {})

    horizons: list[HorizonResult] = []
    for raw_h in horizons_ms:
        horizon = HorizonSpec(horizon_ms=int(raw_h))
        horizons.append(
            _evaluate_one_horizon(
                song_name=path.name,
                calibration=calibration,
                onsets=onsets,
                snapshots=snapshots,
                horizon=horizon,
            )
        )

    return SongHorizonReport(
        song=result["song"],
        duration_s=result["duration_s"],
        n_oracle_events=int(result["n_onset_events"]),
        calibration=calibration,
        candidate_mode=result.get("candidate_mode"),
        horizons=horizons,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _render_horizon_table(report: SongHorizonReport) -> str:
    """Compact horizon curve for one song (issue #99 'compact horizon curve')."""

    header = (
        "horizon_ms | coverage | median_abs_ms | p90_ms | <=40ms | "
        "n_predictions | n_ambig | status"
    )
    lines = [report.song, header, "-" * len(header)]
    for row in _horizon_table_rows(report):
        lines.append(
            "  {h:>9d} | {cov:>4.2f}   | {med:>10.2f}   | {p90:>7.2f} | "
            "{w40:>5.2f} | {n:>13d} | {na:>7d} | {st}".format(
                h=row["horizon_ms"],
                cov=row["coverage"] if row["coverage"] is not None else 0.0,
                med=row["median_abs_ms"] if row["median_abs_ms"] is not None else 0.0,
                p90=row["p90_ms"] if row["p90_ms"] is not None else 0.0,
                w40=row["within_40ms"] if row["within_40ms"] is not None else 0.0,
                n=row["n_predictions"],
                na=row["n_cycle_ambiguous"],
                st=row["status"],
            )
        )
    return "\n".join(lines)


def _report_to_dict(report: SongHorizonReport) -> dict[str, Any]:
    return {
        "song": report.song,
        "duration_s": report.duration_s,
        "n_oracle_events": report.n_oracle_events,
        "candidate_mode": report.candidate_mode,
        "calibration": report.calibration,
        "horizons": [
            {
                "horizon_ms": h.horizon_ms,
                "status": h.status,
                "reason": h.reason,
                "n_oracle_events": h.n_oracle_events,
                "n_evaluation_opportunities": h.n_evaluation_opportunities,
                "n_predictions": h.n_predictions,
                "prediction_coverage": h.prediction_coverage,
                "n_skipped_event_too_early": h.n_skipped_event_too_early,
                "n_skipped_no_pre_horizon_snapshot": h.n_skipped_no_pre_horizon_snapshot,
                "n_skipped_candidate_missing": h.n_skipped_candidate_missing,
                "n_skipped_cycle_ambiguous": h.n_skipped_cycle_ambiguous,
                "signed_timing_error_s": h.signed_timing_error_s,
                "abs_timing_error_s": h.abs_timing_error_s,
                "fraction_within": h.fraction_within,
                "lead_time_s": h.lead_time_s,
            }
            for h in report.horizons
        ],
        "horizon_table": _horizon_table_rows(report),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "backend" / "logs" / "cycle_bank_fixed_horizon"),
    )
    parser.add_argument(
        "--songs",
        nargs="*",
        default=None,
        help="subset of song filenames",
    )
    parser.add_argument(
        "--horizons-ms",
        nargs="*",
        type=int,
        default=list(DEFAULT_HORIZONS_MS),
        help="prediction horizons in milliseconds (issue #99 default: 20 50 100 200 500)",
    )
    parser.add_argument(
        "--calibration-events",
        type=int,
        default=base.DEFAULT_CALIBRATION_EVENTS,
    )
    parser.add_argument(
        "--min-candidate-hits",
        type=int,
        default=base.DEFAULT_MIN_CANDIDATE_HITS,
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.horizons_ms:
        print("error: at least one horizon (ms) is required")
        return 2
    horizons_ms: list[int] = sorted({int(h) for h in args.horizons_ms if int(h) > 0})
    if not horizons_ms:
        print("error: all horizons must be > 0 ms")
        return 2

    songs = [s for s in DEFAULT_SONGS if args.songs is None or s in args.songs]
    reports: list[SongHorizonReport] = []

    for name in songs:
        path = AUDIO_DIR / name
        if not path.exists():
            print(f"skip missing {name}")
            continue
        print(f"analyzing {name} ...")
        report = analyze_song_fixed_horizon(
            path,
            horizons_ms=horizons_ms,
            calibration_events=args.calibration_events,
            min_candidate_hits=args.min_candidate_hits,
        )
        reports.append(report)
        print(_render_horizon_table(report))
        print()

    payload = {
        "horizons_ms": horizons_ms,
        "calibration_events": args.calibration_events,
        "min_candidate_hits": args.min_candidate_hits,
        "songs": [_report_to_dict(r) for r in reports],
    }
    out_path = out_dir / "cycle_bank_fixed_horizon_report.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
