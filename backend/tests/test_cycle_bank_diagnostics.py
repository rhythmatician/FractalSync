"""Deterministic tests for CycleBank real-song causal timing diagnostics (#92).

These tests pin the *offline evaluator* independently of audio decoding and
the Rust DSP implementation. Fake modes implement the same public prediction
surface (`phase_at`, `time_to_next`) as runtime_core.CycleMode so the tests can
verify:

- event-phase calibration uses Rust-style future-phase queries rather than
  Python-side phase dynamics;
- candidate selection uses calibration data only;
- scoring uses the most recent tick STRICTLY before each oracle event;
- an event-tick state or any later/future state cannot change that event's
  prediction;
- a frozen candidate is not silently replaced when another mode later becomes
  more persistent;
- missing frozen-mode state lowers prediction coverage rather than causing
  hindsight mode switching.

The real runtime_core binding is exercised separately by
test_cycle_bank_binding.py and the Rust CycleBank acceptance suite.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cycle_bank_real_song_diagnostics as diag  # noqa: E402


TAU = 2.0 * math.pi


def _wrap_phase(value: float) -> float:
    wrapped = value % TAU
    if wrapped > math.pi:
        wrapped -= TAU
    return wrapped


def _positive_phase(value: float) -> float:
    return value % TAU


@dataclass(frozen=True)
class FakeMode:
    """Minimal immutable stand-in for runtime_core.CycleMode."""

    id: int
    frequency_hz: float
    phase: float
    confidence: float = 1.0
    strength: float = 1.0

    def phase_at(self, delta_seconds: float) -> float:
        # This test-double behavior mirrors the public Rust contract. Production
        # evaluator code must call this method rather than reproduce this math.
        return _wrap_phase(self.phase + TAU * self.frequency_hz * delta_seconds)

    def time_to_next(self, reference_phase: float) -> float | None:
        if self.frequency_hz <= 0.0:
            return None
        diff = _positive_phase(reference_phase - self.phase)
        return diff / (TAU * self.frequency_hz)


def _mode_at(
    *,
    mode_id: int,
    frequency_hz: float,
    time_seconds: float,
    event_phase: float,
    phase_origin_time: float = 0.0,
    confidence: float = 1.0,
) -> FakeMode:
    """Construct a perfect positively advancing mode.

    If oracle events occur at phase_origin_time + k/f, the mode reaches
    event_phase exactly at every oracle event.
    """

    phase = _wrap_phase(
        event_phase + TAU * frequency_hz * (time_seconds - phase_origin_time)
    )
    return FakeMode(
        id=mode_id,
        frequency_hz=frequency_hz,
        phase=phase,
        confidence=confidence,
    )


def _snapshots(
    *,
    end_time: float,
    dt: float,
    builders,
) -> list[diag.TickSnapshot]:
    out: list[diag.TickSnapshot] = []
    n = int(round(end_time / dt))
    for i in range(n + 1):
        t = i * dt
        modes = tuple(builder(t) for builder in builders if builder(t) is not None)
        out.append(diag.TickSnapshot(time_seconds=t, stream_epoch=1, modes=modes))
    return out


def _circular_distance(a: float, b: float) -> float:
    return abs(_wrap_phase(a - b))


def test_calibration_recovers_clean_event_phase_and_high_concentration() -> None:
    f_hz = 2.0
    phi_event = -0.7
    onsets = np.array([(k + 1) / f_hz for k in range(16)], dtype=float)

    def builder(t: float):
        return _mode_at(
            mode_id=7,
            frequency_hz=f_hz,
            time_seconds=t,
            event_phase=phi_event,
        )

    snapshots = _snapshots(end_time=8.5, dt=0.025, builders=[builder])

    calibration = diag._calibrate_observed_mode(
        onsets,
        snapshots,
        calibration_events=8,
        min_candidate_hits=4,
    )

    assert calibration["status"] == "ok"
    assert calibration["candidate_mode_id"] == 7
    assert _circular_distance(calibration["phi_event"], phi_event) < 1.0e-10
    assert calibration["phase_concentration"] > 0.999999
    assert calibration["calibration_coverage"] == 1.0


def test_clean_tracker_predicts_post_calibration_events_at_zero_error() -> None:
    f_hz = 2.0
    phi_event = 0.42
    onsets = np.array([(k + 1) / f_hz for k in range(30)], dtype=float)

    # Onsets land exactly on the 25 ms tick grid. The evaluator must still use
    # the *preceding* 25 ms snapshot, not the snapshot that already ingested
    # the event.
    def builder(t: float):
        return _mode_at(
            mode_id=1,
            frequency_hz=f_hz,
            time_seconds=t,
            event_phase=phi_event,
        )

    snapshots = _snapshots(end_time=15.0, dt=0.025, builders=[builder])

    timing = diag._causal_timing_evaluation(
        onsets,
        snapshots,
        calibration_events=8,
        min_candidate_hits=4,
    )

    assert timing["status"] == "ok"
    evaluation = timing["evaluation"]
    assert evaluation["n_predictions"] == 22
    assert evaluation["prediction_coverage"] == 1.0
    assert evaluation["fraction_within"]["within_20_ms"] == 1.0
    assert evaluation["fraction_within"]["within_30_ms"] == 1.0
    assert evaluation["fraction_within"]["within_40_ms"] == 1.0
    assert evaluation["abs_timing_error_s"]["median"] < 1.0e-10
    assert evaluation["abs_timing_error_s"]["p95"] < 1.0e-10


def test_event_tick_and_future_mutation_cannot_change_prediction() -> None:
    """State at the event tick itself is causally forbidden for that event."""

    f_hz = 1.0
    phi_event = 0.3
    event_time = 2.0

    prior = diag.TickSnapshot(
        time_seconds=1.9,
        stream_epoch=1,
        modes=(
            _mode_at(
                mode_id=5,
                frequency_hz=f_hz,
                time_seconds=1.9,
                event_phase=phi_event,
            ),
        ),
    )

    clean_event_tick = diag.TickSnapshot(
        time_seconds=2.0,
        stream_epoch=1,
        modes=(
            _mode_at(
                mode_id=5,
                frequency_hz=f_hz,
                time_seconds=2.0,
                event_phase=phi_event,
            ),
        ),
    )
    corrupted_event_tick = diag.TickSnapshot(
        time_seconds=2.0,
        stream_epoch=1,
        modes=(FakeMode(id=5, frequency_hz=5.7, phase=-2.4, confidence=0.01),),
    )

    snapshots_a = [
        diag.TickSnapshot(1.8, 1, prior.modes),
        prior,
        clean_event_tick,
        diag.TickSnapshot(2.1, 1, clean_event_tick.modes),
    ]
    snapshots_b = [
        diag.TickSnapshot(1.8, 1, prior.modes),
        prior,
        corrupted_event_tick,
        diag.TickSnapshot(
            2.1,
            1,
            (FakeMode(id=999, frequency_hz=7.0, phase=2.0, confidence=1.0),),
        ),
    ]

    times_a = [s.time_seconds for s in snapshots_a]
    times_b = [s.time_seconds for s in snapshots_b]
    chosen_a = diag._strictly_previous_snapshot(snapshots_a, times_a, event_time)
    chosen_b = diag._strictly_previous_snapshot(snapshots_b, times_b, event_time)

    assert chosen_a is prior
    assert chosen_b is prior

    pred_a = diag._predict_one_event(
        chosen_a,
        event_time=event_time,
        mode_id=5,
        phi_event=phi_event,
    )
    pred_b = diag._predict_one_event(
        chosen_b,
        event_time=event_time,
        mode_id=5,
        phi_event=phi_event,
    )

    assert pred_a is not None
    assert pred_b is not None
    assert pred_a == pred_b
    assert abs(pred_a[1]) < 1.0e-12


def test_candidate_selection_is_frozen_before_future_mode_dominates() -> None:
    """Whole-song persistence must not influence calibration selection."""

    phi_event = -0.2
    onsets = np.array([float(k) for k in range(1, 13)], dtype=float)

    def mode_one(t: float):
        # Strong, coherent mode throughout the first four calibration events;
        # then disappears.
        if t >= 4.2:
            return None
        return _mode_at(
            mode_id=1,
            frequency_hz=1.0,
            time_seconds=t,
            event_phase=phi_event,
            confidence=0.95,
        )

    def mode_two(t: float):
        # Weak/noisy calibration candidate, then overwhelmingly persistent in
        # the future. A whole-song hindsight selector would choose this.
        if t < 0.5:
            return None
        if t < 4.2:
            return FakeMode(
                id=2,
                frequency_hz=1.6,
                phase=_wrap_phase(1.7 + 0.17 * t),
                confidence=0.25,
            )
        return _mode_at(
            mode_id=2,
            frequency_hz=1.0,
            time_seconds=t,
            event_phase=phi_event,
            confidence=1.0,
        )

    snapshots = _snapshots(
        end_time=12.0,
        dt=0.1,
        builders=[mode_one, mode_two],
    )

    calibration = diag._calibrate_observed_mode(
        onsets,
        snapshots,
        calibration_events=4,
        min_candidate_hits=3,
    )

    assert calibration["status"] == "ok"
    assert calibration["candidate_mode_id"] == 1

    evaluation = diag._score_frozen_candidate(onsets, snapshots, calibration)
    assert evaluation["n_evaluation_events"] > 0
    # Mode 1 disappears after calibration. The evaluator must not swap to mode
    # 2 simply because mode 2 looks excellent with hindsight.
    assert evaluation["prediction_coverage"] < 1.0
    assert evaluation["n_skipped_candidate_missing"] > 0


def test_missing_candidate_reduces_coverage_without_hindsight_switching() -> None:
    f_hz = 2.0
    phi_event = 0.1
    onsets = np.array([(k + 1) / f_hz for k in range(16)], dtype=float)

    def frozen_mode(t: float):
        # Present through calibration and for some evaluation events, then gone.
        if t >= 5.1:
            return None
        return _mode_at(
            mode_id=10,
            frequency_hz=f_hz,
            time_seconds=t,
            event_phase=phi_event,
        )

    def replacement_mode(t: float):
        if t < 5.0:
            return None
        return _mode_at(
            mode_id=11,
            frequency_hz=f_hz,
            time_seconds=t,
            event_phase=phi_event,
        )

    snapshots = _snapshots(
        end_time=8.0,
        dt=0.025,
        builders=[frozen_mode, replacement_mode],
    )

    timing = diag._causal_timing_evaluation(
        onsets,
        snapshots,
        calibration_events=6,
        min_candidate_hits=4,
    )

    assert timing["calibration"]["candidate_mode_id"] == 10
    evaluation = timing["evaluation"]
    assert 0.0 < evaluation["prediction_coverage"] < 1.0
    assert evaluation["n_skipped_candidate_missing"] > 0


def test_insufficient_data_is_explicit_and_well_formed() -> None:
    onsets = np.array([0.5, 1.0, 1.5], dtype=float)
    snapshots = [
        diag.TickSnapshot(
            time_seconds=0.1 * i,
            stream_epoch=1,
            modes=(),
        )
        for i in range(20)
    ]

    timing = diag._causal_timing_evaluation(
        onsets,
        snapshots,
        calibration_events=4,
        min_candidate_hits=2,
    )

    assert timing["status"] == "insufficient_data"
    assert timing["reason"] == "not_enough_calibration_events_with_modes"
    assert timing["evaluation"]["n_predictions"] == 0
