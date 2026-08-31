"""Deterministic synthetic tests for fixed-horizon CycleBank prediction (#99).

These tests pin the *offline evaluator* independently of audio decoding and
the Rust DSP implementation. They reuse the FakeMode test double from
test_cycle_bank_diagnostics.py and add the fixed-horizon layer from issue
#99.

What these tests verify:

- cycle ambiguity: when horizon >= mode period, the evaluator reports
  `n_skipped_cycle_ambiguous` and does NOT headline conditional accuracy;
- exact clean oscillator predicts correctly at several horizons that all
  stay strictly shorter than the mode period;
- horizon causality / future mutation: mutating *every* snapshot after the
  selected horizon snapshot does NOT change the prediction;
- no accidental latest-tick fallback: if no snapshot exists at or before
  t_event - H, the evaluator reports no prediction rather than falling
  forward;
- chirp / linear frequency drift: with v1 first-order phase prediction,
  timing error grows sensibly with horizon when the true frequency differs
  from the snapshot's measurement;
- missing frozen mode: coverage falls when the frozen mode is unavailable at
  the required horizon snapshot, with no hindsight mode switching;
- horizon monotonicity sanity: longer horizons select the same snapshot or an
  earlier one for the same event;
- API contract: `_evaluate_one_horizon` receives the FULL oracle onset
  sequence (production parity) and trims calibration-eligible events
  internally;
- PR #98 ↔ PR #99 / issue #99 bridge at the 20 ms horizon (one canonical
  hop).

The real runtime_core binding is exercised separately by
test_cycle_bank_binding.py.
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
import cycle_bank_fixed_horizon_diagnostics as fh  # noqa: E402


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
    """Minimal immutable stand-in for runtime_core.CycleMode.

    For a constant frequency mode, the phase at time t advances linearly.
    For a chirp / drifting mode, the phase at time t advances as the integral
    of the instantaneous frequency (see ChirpMode below). The calibration /
    horizon evaluators only consume ``phase_at`` and ``time_to_next``, which
    both classes implement.
    """

    id: int
    frequency_hz: float
    phase: float
    confidence: float = 1.0
    strength: float = 1.0

    def phase_at(self, delta_seconds: float) -> float:
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
    """Construct a positively advancing constant mode that hits `event_phase`.

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


@dataclass(frozen=True)
class ChirpMode:
    """Stand-in for runtime_core.CycleMode under a linear chirp.

    The instantaneous frequency is f(t) = f0 + a * (t - t0) and the
    integrated phase is

        phi(t) = phi0 + 2π (f0 (t - t0) + 0.5 * a * (t - t0)^2)

    For an oracle event at `phase_origin_time` with crossings of
    `event_phase` happening at times satisfying

        f0 (t - t0) + 0.5 * a * (t - t0)^2 = k

    the calibration and horizon evaluators receive only `phase_at` and
    `time_to_next`, so we mirror those methods faithfully here.
    """

    id: int
    f0_hz: float
    a_hz_per_s: float
    t0_seconds: float
    phase: float
    confidence: float = 1.0
    strength: float = 1.0

    @property
    def frequency_hz(self) -> float:
        """Current measured frequency at the snapshot's time.

        The fake mode represents the snapshot's instantaneous-frequency
        estimate, exactly as the Rust CycleMode emits. Calibration and
        cycle-ambiguity tests both rely on this single number.
        """
        return self.f0_hz

    def _phi(self, t: float) -> float:
        dt = t - self.t0_seconds
        return _wrap_phase(
            self.phase + TAU * (self.f0_hz * dt + 0.5 * self.a_hz_per_s * dt * dt)
        )

    def phase_at(self, delta_seconds: float) -> float:
        # In the chirp setting, Rust's first-order phase_at uses the
        # snapshot's current frequency, not the integral. We mirror that.
        return _wrap_phase(self.phase + TAU * self.f0_hz * delta_seconds)

    def time_to_next(self, reference_phase: float) -> float | None:
        if self.f0_hz <= 0.0:
            return None
        diff = _positive_phase(reference_phase - self.phase)
        return diff / (TAU * self.f0_hz)


def _snapshots(
    *,
    end_time: float,
    dt: float,
    builders,
    stream_epoch: int = 1,
    start_time: float = 0.0,
) -> list[diag.TickSnapshot]:
    out: list[diag.TickSnapshot] = []
    n = int(round(end_time / dt))
    for i in range(n + 1):
        t = start_time + i * dt
        modes = tuple(builder(t) for builder in builders if builder(t) is not None)
        out.append(
            diag.TickSnapshot(time_seconds=t, stream_epoch=stream_epoch, modes=modes)
        )
    return out


def _calibrate_with_full_onsets(
    *,
    f_hz: float,
    phi_event: float,
    end_time: float,
    dt: float,
    calibration_events: int = 8,
    min_candidate_hits: int = 4,
    builders: list | None = None,
    onsets: np.ndarray | None = None,
    extra_skipped_onsets: int = 0,
) -> tuple[dict, list[diag.TickSnapshot], np.ndarray]:
    """Build a deterministic calibrated scenario for a constant mode.

    Returns (calibration, snapshots, full_onsets).

    The returned `full_onsets` is the SAME array the production evaluator
    would receive (the complete oracle sequence including calibration +
    evaluation events). Tests should pass this into `_evaluate_one_horizon`
    unchanged; the evaluator trims calibration-eligible events itself.
    """

    if builders is None:

        def _b(t: float):  # noqa: ANN202
            return _mode_at(
                mode_id=1,
                frequency_hz=f_hz,
                time_seconds=t,
                event_phase=phi_event,
            )

        builders = [_b]
    snapshots = _snapshots(end_time=end_time, dt=dt, builders=builders)

    if onsets is None:
        onsets = np.array(
            [
                (k + 1) / f_hz
                for k in range(int(end_time * f_hz) + 4 + extra_skipped_onsets)
            ],
            dtype=float,
        )

    calibration = diag._calibrate_observed_mode(
        onsets,
        snapshots,
        calibration_events=calibration_events,
        min_candidate_hits=min_candidate_hits,
    )
    assert calibration["status"] == "ok"
    return calibration, snapshots, onsets


# ---------------------------------------------------------------------------
# Cycle ambiguity (issue #99's core conceptual finding)
# ---------------------------------------------------------------------------


def test_cycle_ambiguity_skipped_when_horizon_at_or_exceeds_mode_period() -> None:
    """At horizon >= 1 / f_mode, v1 prediction cannot identify which
    recurrence of `reference_phase` the oracle event is. The evaluator must
    report `n_skipped_cycle_ambiguous` and produce NO prediction rather than
    generating a misleading large-error score.

    Use a 2 Hz mode (period 0.500 s). At H = 500 ms, every evaluation event
    is ambiguous because the lead budget (event - snapshot) is at least one
    full period (modulo a one-hop slack of ~21.33 ms). At H = 50 ms, no
    evaluation event strictly inside the snapshot stream is ambiguous.
    """

    f_hz = 2.0
    phi_event = 0.0
    dt = 0.025

    calibration, snapshots, full_onsets = _calibrate_with_full_onsets(
        f_hz=f_hz,
        phi_event=phi_event,
        end_time=10.0,
        dt=dt,
        calibration_events=8,
        min_candidate_hits=4,
    )

    # H = 500 ms must skip EVERY evaluation event as cycle-ambiguous.
    h500 = fh.HorizonSpec(horizon_ms=500)
    r500 = fh._evaluate_one_horizon(
        song_name="synthetic",
        calibration=calibration,
        onsets=full_onsets,
        snapshots=snapshots,
        horizon=h500,
    )
    assert r500.n_predictions == 0
    assert r500.n_skipped_cycle_ambiguous == r500.n_evaluation_opportunities, r500
    assert r500.prediction_coverage == 0.0

    # H = 50 ms is comfortably inside the period. Events strictly inside
    # the snapshot stream are scored unambiguously; events past the stream
    # end are ambiguous (legitimate: the only available snapshot is now
    # one full period or more away).
    h50 = fh.HorizonSpec(horizon_ms=50)
    r50 = fh._evaluate_one_horizon(
        song_name="synthetic",
        calibration=calibration,
        onsets=full_onsets,
        snapshots=snapshots,
        horizon=h50,
    )
    # Predictions exist (and the unambiguous window has perfect coverage).
    assert r50.n_predictions > 0
    last_snap_t = snapshots[-1].time_seconds
    last_idx = int(calibration["last_calibration_event_index"])
    in_stream_onsets = full_onsets[
        last_idx + 1 : full_onsets.searchsorted(last_snap_t + 1e-9)
    ]
    # Each in-stream event has available_lead <= 0.075 s (50 ms + one-hop
    # slack) which is strictly less than the 0.5 s period.
    assert r50.n_skipped_cycle_ambiguous == len(full_onsets) - len(in_stream_onsets) - (
        last_idx + 1
    )


def test_cycle_ambiguity_boundary_is_strict_at_one_period() -> None:
    """The cycle-ambiguity rule fires when available_lead >= period_s.

    At horizon just below the mode period (e.g. 400 ms for a 2 Hz mode) the
    evaluator must NOT mark events as ambiguous; predictions remain scored.
    We verify this for events that lie strictly inside the snapshot stream.
    """

    f_hz = 2.0
    phi_event = 0.0
    dt = 0.025

    calibration, snapshots, full_onsets = _calibrate_with_full_onsets(
        f_hz=f_hz,
        phi_event=phi_event,
        end_time=10.0,
        dt=dt,
        calibration_events=8,
        min_candidate_hits=4,
    )

    # Crop onsets to events inside the snapshot stream so the boundary case
    # is exactly about the in-stream horizon-vs-period relation.
    last_snap_t = snapshots[-1].time_seconds
    last_idx = int(calibration["last_calibration_event_index"])
    in_stream_onsets = full_onsets[
        last_idx + 1 : full_onsets.searchsorted(last_snap_t - 0.400 + 1e-9)
    ]
    assert len(in_stream_onsets) > 0

    h = fh.HorizonSpec(horizon_ms=400)
    r = fh._evaluate_one_horizon(
        song_name="synthetic",
        calibration=calibration,
        onsets=in_stream_onsets,
        snapshots=snapshots,
        horizon=h,
    )
    # 0.4 s horizon on a 0.5 s period stays strictly inside one period;
    # nothing should be marked ambiguous here.
    assert r.n_skipped_cycle_ambiguous == 0
    assert r.n_predictions == r.n_evaluation_opportunities
    assert r.n_predictions > 0


# ---------------------------------------------------------------------------
# Exact clean oscillator
# ---------------------------------------------------------------------------


def test_exact_clean_oscillator_predicts_correctly_across_horizons() -> None:
    """A perfect constant-frequency mode predicts correctly at horizons that
    are strictly shorter than the mode period.

    Uses f = 1.7 Hz (period ≈ 588 ms) so all tested horizons (20/50/100/200
    ms) sit comfortably inside one period. The 500 ms horizon is now
    ambiguous and is verified separately in
    `test_cycle_ambiguity_skipped_when_horizon_at_or_exceeds_mode_period`.

    We restrict the oracle onsets to events strictly inside the snapshot
    stream so every event has a pre-horizon snapshot and no event crosses
    the mode period. The full onset sequence is what production would pass,
    but the test focuses on the in-stream prediction behavior.
    """

    f_hz = 1.7
    phi_event = 0.42
    dt = 0.025
    end_time = 30.0
    calibration, snapshots, full_onsets = _calibrate_with_full_onsets(
        f_hz=f_hz,
        phi_event=phi_event,
        end_time=end_time,
        dt=dt,
        calibration_events=12,
        min_candidate_hits=6,
    )

    # Crop to events strictly inside the snapshot stream with enough lead
    # room for every horizon.
    last_snap_t = snapshots[-1].time_seconds
    last_idx = int(calibration["last_calibration_event_index"])
    max_safe_horizon = 0.500  # keep < period / 2 for v1 exactness
    in_stream_onsets = full_onsets[
        last_idx + 1 : full_onsets.searchsorted(last_snap_t - max_safe_horizon + 1e-9)
    ]
    assert len(in_stream_onsets) > 0

    for h_ms in (20, 50, 100, 200):
        h = fh.HorizonSpec(horizon_ms=h_ms)
        r = fh._evaluate_one_horizon(
            song_name="synthetic",
            calibration=calibration,
            onsets=in_stream_onsets,
            snapshots=snapshots,
            horizon=h,
        )
        assert r.status == "ok", (h_ms, r)
        assert r.n_predictions > 0
        assert r.n_skipped_cycle_ambiguous == 0
        # The snapshot selected at-or-before t_event - H lands within one
        # hop of the ideal cut-off; for a perfect constant mode, v1
        # prediction therefore lands within one hop of the true event.
        assert r.abs_timing_error_s["max"] is not None
        assert r.abs_timing_error_s["max"] <= dt + 1e-9, (
            h_ms,
            r.abs_timing_error_s,
        )
        assert r.fraction_within["within_40_ms"] == 1.0


# ---------------------------------------------------------------------------
# Horizon causality under future mutation
# ---------------------------------------------------------------------------


def test_horizon_causality_under_complete_future_mutation() -> None:
    """Mutating *every* snapshot after the selected horizon snapshot,
    including the snapshots at and after t_event, must NOT change the
    prediction produced from the selected horizon snapshot.

    This is stronger than PR #98's near-event test: it populates the entire
    interval (snapshot.selected_time, t_event] with normal-hop snapshots,
    then mutates each of them to bogus state. The horizon evaluator must
    only ever consult the selected snapshot, so the predictions must agree.
    """

    f_hz = 2.0
    phi_event = -0.3
    dt = 0.025
    h_s = 0.300  # 300 ms horizon; period is 500 ms so cycle ambiguity is off
    event_time = 8.0  # k = 16, an exact phi_event crossing

    # Build snapshots over the full window [0, event_time + 2 * dt].
    end_time = event_time + 2 * dt
    pre_horizon_index = int(round((event_time - h_s) / dt))
    pre_horizon_t = pre_horizon_index * dt

    def builder(t: float):
        return _mode_at(
            mode_id=11,
            frequency_hz=f_hz,
            time_seconds=t,
            event_phase=phi_event,
        )

    snapshots_a: list[diag.TickSnapshot] = []
    snapshots_b: list[diag.TickSnapshot] = []
    n_steps = int(round(end_time / dt)) + 1
    for i in range(n_steps):
        t = i * dt
        modes_clean = (builder(t),) if builder(t) is not None else ()
        if t > pre_horizon_t + 1e-9:
            # Everything from the selected horizon snapshot onwards gets a
            # bogus mutation in world B. The selected snapshot itself is
            # identical between worlds (the rule is causal with respect to
            # state at-or-before t_event - H).
            modes_bogus = (
                FakeMode(id=999, frequency_hz=9.0, phase=2.0, confidence=0.01),
            )
        else:
            modes_bogus = modes_clean
        snapshots_a.append(
            diag.TickSnapshot(time_seconds=t, stream_epoch=1, modes=modes_clean)
        )
        snapshots_b.append(
            diag.TickSnapshot(time_seconds=t, stream_epoch=1, modes=modes_bogus)
        )

    times_a = [s.time_seconds for s in snapshots_a]
    times_b = [s.time_seconds for s in snapshots_b]

    selected_a = fh._snapshot_at_or_before(snapshots_a, times_a, event_time - h_s)
    selected_b = fh._snapshot_at_or_before(snapshots_b, times_b, event_time - h_s)
    assert selected_a is not None
    assert selected_b is not None
    # Both worlds share the same horizon snapshot; the bogus state is at/after.
    assert selected_a.time_seconds == selected_b.time_seconds

    pred_a = fh._predict_one_event_at_horizon(
        selected_a,
        event_time=event_time,
        horizon_s=h_s,
        mode_id=11,
        phi_event=phi_event,
    )
    pred_b = fh._predict_one_event_at_horizon(
        selected_b,
        event_time=event_time,
        horizon_s=h_s,
        mode_id=11,
        phi_event=phi_event,
    )
    assert pred_a is not None and pred_b is not None
    assert pred_a[0] == pred_b[0]
    assert pred_a[1] == pred_b[1]


# ---------------------------------------------------------------------------
# No-fallback when no pre-horizon snapshot exists
# ---------------------------------------------------------------------------


def test_no_accidental_latest_tick_fallback_when_no_pre_horizon_snapshot() -> None:
    """If no snapshot exists at or before t_event - H, the evaluator must
    report no prediction. It MUST NOT fall forward to a later snapshot.

    Construction: build the snapshot stream, calibrate on it normally, and
    then construct a snapshot list that deliberately has a GAP after the
    calibration cutoff. The first event past the cutoff (a) lands BEFORE
    the next snapshot in the gap and therefore has no pre-horizon snapshot;
    subsequent events (b, c, ...) land after the gap resumes and have a
    normal pre-horizon snapshot. The evaluator must:
      * skip event (a) as n_skipped_no_pre_horizon_snapshot;
      * score events (b, c, ...) as predictions;
      * never produce a "prediction" for event (a) by walking forward.
    """

    f_hz = 2.0
    phi_event = 0.0
    dt = 0.025

    calibration, full_snapshots, full_onsets = _calibrate_with_full_onsets(
        f_hz=f_hz,
        phi_event=phi_event,
        end_time=20.0,
        dt=dt,
        calibration_events=8,
        min_candidate_hits=4,
    )

    cutoff_calib = float(calibration["calibration_cutoff_s"])

    # Pick a gap location: a snapshot index drop point just past the
    # calibration cutoff. We keep snapshots strictly past `cutoff_calib +
    # 0.500` so the first evaluation event (at cutoff_calib + 0.500 s,
    # minus 30 ms = cutoff_calib + 0.470 s) has no pre-horizon snapshot.
    gap_start = cutoff_calib + 0.500
    truncated_snapshots = [
        s for s in full_snapshots if s.time_seconds > gap_start
    ]
    assert truncated_snapshots, "no evaluation snapshots remain after truncation"

    # The first evaluation event past calibration cutoff:
    last_idx = int(calibration["last_calibration_event_index"])
    first_eval_t = float(full_onsets[last_idx + 1])
    assert first_eval_t > cutoff_calib

    h_s = 0.030
    h = fh.HorizonSpec(horizon_ms=int(h_s * 1000))
    assert first_eval_t - h_s < truncated_snapshots[0].time_seconds, (
        first_eval_t,
        truncated_snapshots[0].time_seconds,
    )

    # Direct selector test: the snapshot selector itself refuses to fall
    # forward.
    chosen = fh._snapshot_at_or_before(
        truncated_snapshots,
        [s.time_seconds for s in truncated_snapshots],
        first_eval_t - h_s,
    )
    assert chosen is None

    # End-to-end test: feed the full production-shaped onset sequence.
    result = fh._evaluate_one_horizon(
        song_name="synthetic",
        calibration=calibration,
        onsets=full_onsets,
        snapshots=truncated_snapshots,
        horizon=h,
    )

    # The first evaluation event must NOT be predicted.
    assert result.n_skipped_no_pre_horizon_snapshot >= 1
    # Subsequent evaluation events DO have a pre-horizon snapshot and must
    # score as predictions.
    assert result.n_predictions > 0
    # And the bookkeeping is exact: every evaluation opportunity falls into
    # exactly one bucket.
    accounted = (
        result.n_predictions
        + result.n_skipped_no_pre_horizon_snapshot
        + result.n_skipped_candidate_missing
        + result.n_skipped_cycle_ambiguous
    )
    assert accounted == result.n_evaluation_opportunities


# ---------------------------------------------------------------------------
# Chirp / linear drift sensitivity
# ---------------------------------------------------------------------------


def test_chirp_sensitivity_grows_with_horizon() -> None:
    """A linearly drifting mode: v1 first-order phase prediction is exact at
    the snapshot time but degrades with horizon because the true frequency
    at the prediction time differs from the snapshot's measurement.

    For a chirp f(t) = f0 + a t, the integrated phase at time t is
    phi(t) = phi0 + 2π (f0 t + 0.5 a t^2).  The k-th crossing of
    `event_phase` (after t = 0) satisfies:

        f0 t_k + 0.5 a t_k^2 = k

    This test computes the event timestamps by solving the quadratic and
    places the snapshot's reported frequency at f0 (the mode's "birth"
    frequency), which is what v1 prediction uses.
    """

    f0 = 2.0
    a = 0.4  # Hz/s; over 500 ms horizon, f rises ~0.2 Hz, a meaningful shift
    phi_event = 0.0
    end_time = 30.0
    dt = 0.025

    # Build a chirp snapshot stream whose first snapshot is at t = 0 with
    # phase phi_event = 0 and instantaneous frequency f0.
    def chirp_builder(t: float):
        return ChirpMode(
            id=3,
            f0_hz=f0,
            a_hz_per_s=a,
            t0_seconds=0.0,
            phase=phi_event,
        )

    snapshots = _snapshots(end_time=end_time, dt=dt, builders=[chirp_builder])

    # Solve f0 t + 0.5 a t^2 = k for the first several crossings.
    crossings: list[float] = []
    k = 0
    while True:
        # f0 t + 0.5 a t^2 = k  -> 0.5 a t^2 + f0 t - k = 0
        A = 0.5 * a
        B = f0
        C = -k
        if A == 0.0:
            t_cross = C / (-B) if B != 0.0 else None
        else:
            disc = B * B - 4 * A * C
            if disc < 0:
                break
            sqrt_disc = math.sqrt(disc)
            t1 = (-B + sqrt_disc) / (2 * A)
            t2 = (-B - sqrt_disc) / (2 * A)
            t_cross = max(t1, t2)
        if t_cross is None or t_cross > end_time - 1.0:
            break
        if t_cross > 0.0:
            crossings.append(t_cross)
        k += 1
        if k > 200:
            break
    assert len(crossings) >= 20, "not enough chirp crossings generated"

    onsets = np.asarray(crossings, dtype=float)

    calibration = diag._calibrate_observed_mode(
        onsets,
        snapshots,
        calibration_events=8,
        min_candidate_hits=4,
    )
    assert calibration["status"] == "ok"

    # Each horizon measures how far the v1 (constant-frequency) prediction
    # drifts from the true chirp crossing. Error must grow with horizon.
    # H = 500 ms sits at exactly the mode period and so the test asserts the
    # cycle-ambiguity counter rather than headline accuracy there.
    max_err_by_h: dict[int, float] = {}
    n_ambiguous_by_h: dict[int, int] = {}
    for h_ms in (20, 100, 200, 500):
        h = fh.HorizonSpec(horizon_ms=h_ms)
        r = fh._evaluate_one_horizon(
            song_name="chirp",
            calibration=calibration,
            onsets=onsets,
            snapshots=snapshots,
            horizon=h,
        )
        n_ambiguous_by_h[h_ms] = r.n_skipped_cycle_ambiguous
        if r.n_predictions > 0:
            assert r.abs_timing_error_s["max"] is not None
            max_err_by_h[h_ms] = float(r.abs_timing_error_s["max"])

    # Strict monotonic: H = 200 > H = 100 > H = 20 in error budget.
    assert max_err_by_h[100] >= max_err_by_h[20]
    assert max_err_by_h[200] >= max_err_by_h[100]

    # H = 500 ms sits at the period boundary and must surface cycle
    # ambiguity rather than headline accuracy. f0 = 2 Hz gives period
    # 0.500 s, so at H = 500 ms virtually every in-stream event is
    # ambiguous.
    assert n_ambiguous_by_h[500] > 0
    assert 500 not in max_err_by_h or max_err_by_h[500] <= 0.001


# ---------------------------------------------------------------------------
# Missing frozen mode at the horizon snapshot
# ---------------------------------------------------------------------------


def test_missing_frozen_mode_lowers_coverage_without_hindsight_switching() -> None:
    """If the frozen mode is unavailable at the horizon snapshot, coverage
    falls. The evaluator MUST NOT hindsight-switch to a different mode.

    Uses horizons strictly shorter than the period so cycle ambiguity does
    not pollute this assertion (the missing-mode behavior is orthogonal).
    """

    f_hz = 2.0
    phi_event = 0.0
    end_time = 15.0
    dt = 0.025

    def frozen(t: float):
        if t >= 5.0:
            return None
        return _mode_at(
            mode_id=20,
            frequency_hz=f_hz,
            time_seconds=t,
            event_phase=phi_event,
        )

    def replacement(t: float):
        if t < 4.8:
            return None
        return _mode_at(
            mode_id=21,
            frequency_hz=f_hz,
            time_seconds=t,
            event_phase=phi_event,
        )

    snapshots = _snapshots(end_time=end_time, dt=dt, builders=[frozen, replacement])
    onsets = np.array(
        [(k + 1) / f_hz for k in range(int(end_time * f_hz) + 2)], dtype=float
    )

    calibration = diag._calibrate_observed_mode(
        onsets,
        snapshots,
        calibration_events=8,
        min_candidate_hits=4,
    )
    assert calibration["status"] == "ok"
    assert calibration["candidate_mode_id"] == 20

    for h_ms in (20, 100, 200):
        h = fh.HorizonSpec(horizon_ms=h_ms)
        r = fh._evaluate_one_horizon(
            song_name="synthetic",
            calibration=calibration,
            onsets=onsets,
            snapshots=snapshots,
            horizon=h,
        )
        # Frozen mode disappears after t = 5 s; predictions must come from
        # mode 20 alone, with no hindsight switching to mode 21.
        assert 0.0 < r.prediction_coverage < 1.0, (h_ms, r)
        assert r.n_skipped_candidate_missing > 0


# ---------------------------------------------------------------------------
# Horizon monotonicity
# ---------------------------------------------------------------------------


def test_horizon_monotonicity_snapshots_move_earlier_for_longer_horizons() -> None:
    """For the same event, longer horizons must select the same snapshot or
    one strictly earlier in time. We check by reading snapshot times
    selected for the same event at each requested horizon and verifying
    pairwise non-increasingness in the order of (h_ms ASCENDING).
    """

    f_hz = 2.0
    phi_event = 0.0
    end_time = 10.0
    dt = 0.025

    calibration, snapshots, full_onsets = _calibrate_with_full_onsets(
        f_hz=f_hz,
        phi_event=phi_event,
        end_time=end_time,
        dt=dt,
        calibration_events=8,
        min_candidate_hits=4,
    )

    snapshot_times = [s.time_seconds for s in snapshots]
    last_idx = int(calibration["last_calibration_event_index"])
    eval_onsets = full_onsets[last_idx + 1 :]

    for event_time in eval_onsets[:5]:
        selected: dict[int, float] = {}
        for h_ms in (20, 50, 100, 200, 500):
            snap = fh._snapshot_at_or_before(
                snapshots, snapshot_times, event_time - h_ms / 1000.0
            )
            assert snap is not None
            selected[h_ms] = snap.time_seconds
        # Pairwise: t(h_ms=larger) <= t(h_ms=smaller).
        h_sorted = sorted(selected)
        for h_smaller, h_larger in zip(h_sorted, h_sorted[1:]):
            assert selected[h_larger] <= selected[h_smaller] + 1e-12, (
                event_time,
                selected,
            )


# ---------------------------------------------------------------------------
# API contract: production parity on the onset array
# ---------------------------------------------------------------------------


def test_evaluate_one_horizon_receives_full_onset_sequence() -> None:
    """`_evaluate_one_horizon` must receive the FULL oracle onset sequence
    and trim calibration-eligible events itself, mirroring the production
    contract. We verify by passing the full sequence and reading back the
    `n_skipped_event_too_early` counter.
    """

    f_hz = 2.0
    phi_event = 0.0
    dt = 0.025

    calibration, snapshots, full_onsets = _calibrate_with_full_onsets(
        f_hz=f_hz,
        phi_event=phi_event,
        end_time=10.0,
        dt=dt,
        calibration_events=8,
        min_candidate_hits=4,
    )

    r = fh._evaluate_one_horizon(
        song_name="synthetic",
        calibration=calibration,
        onsets=full_onsets,
        snapshots=snapshots,
        horizon=fh.HorizonSpec(horizon_ms=50),
    )

    n_calibration_eligible = int(calibration["last_calibration_event_index"]) + 1
    assert r.n_skipped_event_too_early == n_calibration_eligible
    # And the evaluation opportunities match what PR #98's near-event
    # evaluator would have produced from the same onsets.
    expected_eval = len(full_onsets) - n_calibration_eligible
    assert r.n_evaluation_opportunities == expected_eval


# ---------------------------------------------------------------------------
# PR #98 ↔ issue #99 bridge at the one-hop horizon
# ---------------------------------------------------------------------------


def test_horizon_close_to_hop_bridges_to_pr98_near_event_result() -> None:
    """At a 20 ms horizon (close to one canonical hop), the fixed-horizon
    evaluator selects a snapshot at-or-before t_event - 20 ms, which on the
    canonical grid is one snapshot earlier than PR #98's strictly-previous
    evaluator. The predictions agree to within the same grid quantization
    because v1 prediction is exact for a constant mode.
    """

    f_hz = 1.7  # off-grid; period ≈ 588 ms, well above 20 ms
    phi_event = 0.5
    end_time = 12.0
    dt = 0.025

    calibration, snapshots, full_onsets = _calibrate_with_full_onsets(
        f_hz=f_hz,
        phi_event=phi_event,
        end_time=end_time,
        dt=dt,
        calibration_events=8,
        min_candidate_hits=4,
    )

    snapshot_times = [s.time_seconds for s in snapshots]
    last_idx = int(calibration["last_calibration_event_index"])
    eval_onsets = full_onsets[last_idx + 1 :][:3]

    for event_time in eval_onsets:
        pr98 = diag._strictly_previous_snapshot(
            snapshots, snapshot_times, float(event_time)
        )
        fh20 = fh._snapshot_at_or_before(
            snapshots, snapshot_times, float(event_time) - 0.020
        )
        assert pr98 is not None and fh20 is not None
        # fh20 must be the same as or earlier than pr98 (longer horizon =>
        # earlier or equal snapshot).
        assert fh20.time_seconds <= pr98.time_seconds + 1e-12
        # On the canonical grid the difference is at most one hop.
        assert abs(pr98.time_seconds - fh20.time_seconds) <= dt + 1e-9


# ---------------------------------------------------------------------------
# Empty / insufficient calibration
# ---------------------------------------------------------------------------


def test_evaluate_one_horizon_returns_insufficient_when_calibration_failed() -> None:
    """When calibration fails, per-horizon results report insufficient_data
    with the same reason, not a synthetic zero-coverage score.
    """

    bad_calibration = {
        "status": "insufficient_data",
        "reason": "no_candidate_met_min_hits",
    }
    result = fh._evaluate_one_horizon(
        song_name="synthetic",
        calibration=bad_calibration,
        onsets=np.array([0.1, 0.2, 0.3]),
        snapshots=[],
        horizon=fh.HorizonSpec(horizon_ms=100),
    )
    assert result.status == "insufficient_data"
    assert result.reason == "no_candidate_met_min_hits"
    assert result.n_predictions == 0
    assert result.prediction_coverage == 0.0
    assert result.n_skipped_cycle_ambiguous == 0
