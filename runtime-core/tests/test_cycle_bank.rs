//! Deterministic synthetic acceptance tests for the canonical observed-ridge
//! `CycleBank` (issue #92).
//!
//! This suite is written for the corrected `cycle_bank.rs` implementation:
//!
//! - one scalar value per named evidence channel per authoritative hop;
//! - accumulated causal demodulator phase;
//! - continuous off-grid instantaneous-frequency recovery;
//! - phase-calibrated observed ridges;
//! - prediction-before-correction temporal association;
//! - real pending-mode birth hysteresis;
//! - free-running through missing evidence;
//! - relation stability measured across time;
//! - explicit stream-epoch and channel-schema hygiene.
//!
//! These tests deliberately stop at directly observed ridges.  Latent temporal
//! undertones / missing fundamentals belong to issue #97.

use runtime_core::cycle_bank::{
    CycleBank, CycleBankConfig, CycleBankError, CycleEvidenceChannel, CycleMode,
    CycleObservation, CycleRelation, CYCLE_BANK_VERSION,
};
use runtime_core::controller::{HOP_LENGTH, SAMPLE_RATE};
use std::f64::consts::{PI, TAU};

const DT: f64 = HOP_LENGTH as f64 / SAMPLE_RATE as f64;

const DEFAULT_WARMUP_SECONDS: f64 = 18.0;

struct Harness {
    bank: CycleBank,
    epoch: u64,
    sample_index: u64,
}

impl Harness {
    fn new(config: CycleBankConfig) -> Self {
        Self {
            bank: CycleBank::new(config),
            epoch: 1,
            sample_index: 0,
        }
    }

    fn current_time(&self) -> f64 {
        self.sample_index as f64 / SAMPLE_RATE as f64
    }

    fn next_time(&self) -> f64 {
        (self.sample_index + HOP_LENGTH as u64) as f64 / SAMPLE_RATE as f64
    }

    fn observe(&mut self, channels: Vec<CycleEvidenceChannel>) -> Vec<CycleMode> {
        self.sample_index += HOP_LENGTH as u64;
        let obs = CycleObservation {
            sample_index: self.sample_index,
            dt_seconds: DT,
            stream_epoch: self.epoch,
            channels,
        };
        self.bank.observe(&obs).expect("CycleBank observation");
        self.bank.modes()
    }

    fn observe_one(&mut self, name: &str, value: f64) -> Vec<CycleMode> {
        self.observe(vec![CycleEvidenceChannel::new(name, value)])
    }

    fn change_epoch(&mut self, epoch: u64, restart_sample_clock: bool) {
        self.epoch = epoch;
        if restart_sample_clock {
            self.sample_index = 0;
        }
    }
}

fn hops(seconds: f64) -> usize {
    (seconds / DT).ceil() as usize
}

fn wrap_phase(value: f64) -> f64 {
    let mut wrapped = value.rem_euclid(TAU);
    if wrapped > PI {
        wrapped -= TAU;
    }
    wrapped
}

fn circular_distance(a: f64, b: f64) -> f64 {
    wrap_phase(a - b).abs()
}

fn closest_mode(modes: &[CycleMode], frequency_hz: f64) -> Option<CycleMode> {
    modes
        .iter()
        .min_by(|a, b| {
            (a.frequency_hz - frequency_hz)
                .abs()
                .partial_cmp(&(b.frequency_hz - frequency_hz).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
}

fn mode_by_id(modes: &[CycleMode], id: u64) -> Option<CycleMode> {
    modes.iter().find(|mode| mode.id == id).cloned()
}

fn nearest_nominal_scale(config: &CycleBankConfig, target_hz: f64) -> f64 {
    let step = 2.0_f64.powf(1.0 / config.scales_per_octave as f64);
    let mut f = config.f_min_hz;
    let mut nearest = f;
    let mut nearest_error = (f - target_hz).abs();

    while f <= config.f_max_hz * (1.0 + 1.0e-12) {
        let error = (f - target_hz).abs();
        if error < nearest_error {
            nearest = f;
            nearest_error = error;
        }
        f *= step;
    }
    nearest
}

fn acquire_sinusoid(
    harness: &mut Harness,
    channel: &str,
    frequency_hz: f64,
    amplitude: f64,
    phase0: f64,
    seconds: f64,
) -> Vec<Vec<CycleMode>> {
    let mut snapshots = Vec::new();
    for _ in 0..hops(seconds) {
        let t = harness.next_time();
        let value = amplitude * (TAU * frequency_hz * t + phase0).cos();
        snapshots.push(harness.observe_one(channel, value));
    }
    snapshots
}

fn deterministic_jitter_for_cycle(cycle: i64, max_abs_seconds: f64) -> f64 {
    // Deterministic integer hash -> [-1, 1].
    let mut x = cycle as u64 ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = (x as f64) / (u64::MAX as f64);
    (2.0 * unit - 1.0) * max_abs_seconds
}

fn jittered_pulse(t: f64, period: f64, max_jitter: f64, half_width: f64) -> f64 {
    let center = (t / period).round() as i64;
    for k in [center - 1, center, center + 1] {
        if k < 0 {
            continue;
        }
        let event_time =
            k as f64 * period + deterministic_jitter_for_cycle(k, max_jitter);
        if (t - event_time).abs() <= half_width {
            return 1.0;
        }
    }
    0.0
}

fn relation_for_pair<'a>(
    relations: &'a [CycleRelation],
    first: &CycleMode,
    second: &CycleMode,
    lower_multiplier: u32,
    upper_multiplier: u32,
) -> Option<&'a CycleRelation> {
    // If lower-frequency mode is relation.i, then
    // lower_multiplier * f_low ~= upper_multiplier * f_high.
    //
    // If ids happen to be ordered the other way around, the numerator and
    // denominator swap.  This helper checks both orientations rather than
    // coupling the scientific assertion to birth/id ordering.
    let (low, high) = if first.frequency_hz <= second.frequency_hz {
        (first, second)
    } else {
        (second, first)
    };

    relations.iter().find(|relation| {
        if relation.i_id == low.id && relation.j_id == high.id {
            relation.m == lower_multiplier && relation.n == upper_multiplier
        } else if relation.i_id == high.id && relation.j_id == low.id {
            relation.m == upper_multiplier && relation.n == lower_multiplier
        } else {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// Contract / invariant smoke tests
// ---------------------------------------------------------------------------

#[test]
fn cycle_bank_contract_version_is_explicit() {
    let bank = CycleBank::new(CycleBankConfig::default());
    assert_eq!(bank.version(), CYCLE_BANK_VERSION);
    assert_eq!(CYCLE_BANK_VERSION, "cycle-bank/2");
}

#[test]
fn invalid_configuration_is_rejected_without_constructing_state() {
    let mut config = CycleBankConfig::default();
    config.q_cycles = 0.0;
    assert!(matches!(
        CycleBank::try_new(config),
        Err(CycleBankError::InvalidConfig(_))
    ));

    let mut config = CycleBankConfig::default();
    config.scales_per_octave = 0;
    assert!(matches!(
        CycleBank::try_new(config),
        Err(CycleBankError::InvalidConfig(_))
    ));

    let mut config = CycleBankConfig::default();
    config.free_run_max_observations = 40;
    config.death_persistence = 20;
    assert!(matches!(
        CycleBank::try_new(config),
        Err(CycleBankError::InvalidConfig(_))
    ));
}

#[test]
fn nonfinite_evidence_and_invalid_dt_are_rejected() {
    let mut bank = CycleBank::new(CycleBankConfig::default());

    let bad_dt = CycleObservation {
        sample_index: HOP_LENGTH as u64,
        dt_seconds: 0.0,
        stream_epoch: 1,
        channels: vec![CycleEvidenceChannel::new("mono", 0.0)],
    };
    assert!(matches!(
        bank.observe(&bad_dt),
        Err(CycleBankError::InvalidDt(_))
    ));

    let nonfinite = CycleObservation {
        sample_index: HOP_LENGTH as u64,
        dt_seconds: DT,
        stream_epoch: 1,
        channels: vec![CycleEvidenceChannel::new("mono", f64::NAN)],
    };
    assert!(matches!(
        bank.observe(&nonfinite),
        Err(CycleBankError::NonFiniteEvidence { .. })
    ));
}

#[test]
fn channel_schema_identity_and_order_are_persistent_within_epoch() {
    let mut bank = CycleBank::new(CycleBankConfig::default());

    bank.observe(&CycleObservation {
        sample_index: HOP_LENGTH as u64,
        dt_seconds: DT,
        stream_epoch: 1,
        channels: vec![
            CycleEvidenceChannel::new("onset", 0.3),
            CycleEvidenceChannel::new("low", 0.2),
        ],
    })
    .unwrap();

    let swapped = CycleObservation {
        sample_index: 2 * HOP_LENGTH as u64,
        dt_seconds: DT,
        stream_epoch: 1,
        channels: vec![
            CycleEvidenceChannel::new("low", 0.2),
            CycleEvidenceChannel::new("onset", 0.3),
        ],
    };

    assert!(matches!(
        bank.observe(&swapped),
        Err(CycleBankError::ChannelSchemaMismatch { .. })
    ));
}

#[test]
fn duplicate_channel_names_are_rejected() {
    let mut bank = CycleBank::new(CycleBankConfig::default());
    let obs = CycleObservation {
        sample_index: HOP_LENGTH as u64,
        dt_seconds: DT,
        stream_epoch: 1,
        channels: vec![
            CycleEvidenceChannel::new("same", 0.1),
            CycleEvidenceChannel::new("same", 0.2),
        ],
    };
    assert!(matches!(
        bank.observe(&obs),
        Err(CycleBankError::InvalidChannelName(_))
    ));
}

#[test]
fn sample_index_must_increase_within_an_epoch() {
    let mut bank = CycleBank::new(CycleBankConfig::default());

    let first = CycleObservation {
        sample_index: HOP_LENGTH as u64,
        dt_seconds: DT,
        stream_epoch: 1,
        channels: vec![CycleEvidenceChannel::new("mono", 0.0)],
    };
    bank.observe(&first).unwrap();

    let repeated = CycleObservation {
        sample_index: HOP_LENGTH as u64,
        dt_seconds: DT,
        stream_epoch: 1,
        channels: vec![CycleEvidenceChannel::new("mono", 0.0)],
    };
    assert!(matches!(
        bank.observe(&repeated),
        Err(CycleBankError::NonMonotonicSampleIndex { .. })
    ));
}

// ---------------------------------------------------------------------------
// A. Off-grid continuous-frequency recovery
// ---------------------------------------------------------------------------

#[test]
fn off_grid_sinusoid_recovers_continuous_frequency_not_scale_center() {
    let target_hz = 2.1667;
    let phase0 = 0.37;
    let mut config = CycleBankConfig::default();
    config.f_min_hz = 0.5;
    config.f_max_hz = 4.0;
    config.birth_persistence = 2;

    let nearest_center = nearest_nominal_scale(&config, target_hz);
    assert!(
        (nearest_center - target_hz).abs() > 1.0e-3,
        "fixture accidentally landed on a nominal scale center"
    );

    let mut harness = Harness::new(config.clone());
    acquire_sinusoid(
        &mut harness,
        "mono",
        target_hz,
        0.8,
        phase0,
        DEFAULT_WARMUP_SECONDS,
    );

    let modes = harness.bank.modes();
    let mode = closest_mode(&modes, target_hz).expect("observed ridge near target");

    let recovered_error = (mode.frequency_hz - target_hz).abs();
    let center_error = (nearest_center - target_hz).abs();

    assert!(
        recovered_error < 0.02,
        "continuous frequency recovery inaccurate: target={target_hz}, recovered={}, modes={modes:?}",
        mode.frequency_hz
    );
    assert!(
        recovered_error < center_error,
        "recovered frequency did not beat the nearest nominal scale center: target={target_hz}, recovered={}, center={nearest_center}",
        mode.frequency_hz
    );
}

// ---------------------------------------------------------------------------
// B. Numerical scale-resolution convergence
// ---------------------------------------------------------------------------

#[test]
fn scale_resolution_converges_without_turning_scales_into_musical_buckets() {
    let target_hz = 1.6234;
    let phase0 = -0.42;
    let mut estimates = Vec::new();

    for scales_per_octave in [6usize, 12, 24, 48] {
        let mut config = CycleBankConfig::default();
        config.f_min_hz = 0.5;
        config.f_max_hz = 4.0;
        config.scales_per_octave = scales_per_octave;
        config.birth_persistence = 2;

        let mut harness = Harness::new(config);
        acquire_sinusoid(
            &mut harness,
            "mono",
            target_hz,
            0.8,
            phase0,
            DEFAULT_WARMUP_SECONDS,
        );

        let modes = harness.bank.modes();
        let mode =
            closest_mode(&modes, target_hz).expect("ridge in convergence experiment");
        let frequency_error = (mode.frequency_hz - target_hz).abs();

        println!(
            "scales/octave={scales_per_octave}: recovered={:.8} Hz, abs_err={:.8} Hz",
            mode.frequency_hz, frequency_error
        );
        estimates.push((scales_per_octave, mode.frequency_hz, frequency_error));
    }

    // All practically useful grids should recover a continuous frequency
    // close to the same answer.  We intentionally do NOT require strictly
    // monotone error: numerical ridge selection need not improve monotonically
    // at every intermediate grid density.
    for &(spo, recovered, error) in estimates.iter() {
        assert!(
            error / target_hz < 0.02,
            "{spo} scales/octave did not converge near the continuous target: recovered={recovered}"
        );
    }

    let f06 = estimates.iter().find(|(s, _, _)| *s == 6).unwrap().1;
    let f12 = estimates.iter().find(|(s, _, _)| *s == 12).unwrap().1;
    let f24 = estimates.iter().find(|(s, _, _)| *s == 24).unwrap().1;
    let f48 = estimates.iter().find(|(s, _, _)| *s == 48).unwrap().1;

    assert!(
        (f06 - f48).abs() / target_hz < 0.02,
        "6 scales/octave has not numerically converged to the dense reference: f06={f06}, f48={f48}"
    );
    assert!(
        (f24 - f48).abs() / target_hz < 0.01,
        "24 and 48 scales/octave have not numerically converged: f24={f24}, f48={f48}"
    );
    assert!(
        (f12 - f48).abs() / target_hz < 0.02,
        "12 scales/octave is materially different from the dense reference: f12={f12}, f48={f48}"
    );
}

// ---------------------------------------------------------------------------
// C. Phase calibration and free-running prediction
// ---------------------------------------------------------------------------

#[test]
fn calibrated_phase_matches_known_input_analytic_phase() {
    let phase0 = 0.73;

    // Phase calibration is the property we actually care about (the Player
    // must land on musical events within a small timing envelope). Express
    // the threshold as a per-cycle timing error `e_t = |e_phi| / (2π f)`
    // rather than a raw radian number, so the assertion stays meaningful
    // across the whole frequency band — a 0.45 rad budget is ~36 ms at 2 Hz
    // and ~90 ms at 0.8 Hz, but the convergence study reports phase errors
    // of ~0.03–0.05 rad ≈ 2–4 ms, so 20 ms leaves generous margin.
    const MAX_TIMING_ERROR_S: f64 = 20.0e-3;

    for target_hz in [0.8, 1.5, 2.1667, 2.7, 4.0] {
        let mut config = CycleBankConfig::default();
        config.f_min_hz = 0.5;
        config.f_max_hz = 5.0;
        config.birth_persistence = 2;

        let mut harness = Harness::new(config);
        acquire_sinusoid(
            &mut harness,
            "mono",
            target_hz,
            0.8,
            phase0,
            DEFAULT_WARMUP_SECONDS,
        );

        let modes = harness.bank.modes();
        let mode = closest_mode(&modes, target_hz).expect("calibrated ridge");
        let expected_phase =
            wrap_phase(TAU * target_hz * harness.current_time() + phase0);
        let phase_error_rad = circular_distance(mode.phase, expected_phase);
        let timing_error_s = phase_error_rad / (TAU * target_hz);

        println!(
            "phase calibration: f={target_hz:.4} Hz, phase_err={phase_error_rad:.5} rad, timing_err={timing_error_s:.6} s",
        );

        assert!(
            timing_error_s < MAX_TIMING_ERROR_S,
            "causal filter phase was not correctly calibrated at {target_hz} Hz: \
             phase_error={phase_error_rad} rad, timing_error={timing_error_s} s \
             (max {MAX_TIMING_ERROR_S} s)",
        );
    }
}

#[test]
fn phase_at_one_period_returns_to_current_phase() {
    let mut harness = Harness::new(CycleBankConfig::default());
    acquire_sinusoid(
        &mut harness,
        "mono",
        2.1667,
        0.8,
        0.2,
        DEFAULT_WARMUP_SECONDS,
    );

    let mode = closest_mode(&harness.bank.modes(), 2.1667).expect("ridge");
    let predicted = mode.phase_at(1.0 / mode.frequency_hz);

    assert!(
        circular_distance(predicted, mode.phase) < 1.0e-10,
        "one-period free-running phase prediction did not close"
    );
}

#[test]
fn time_to_next_reference_phase_matches_quarter_period() {
    let mut harness = Harness::new(CycleBankConfig::default());
    acquire_sinusoid(
        &mut harness,
        "mono",
        2.0,
        0.8,
        -0.1,
        DEFAULT_WARMUP_SECONDS,
    );

    let mode = closest_mode(&harness.bank.modes(), 2.0).expect("ridge");
    let reference = wrap_phase(mode.phase + PI / 2.0);
    let measured = mode.time_to_next(reference).expect("positive frequency");
    let expected = 0.25 / mode.frequency_hz;

    assert!(
        (measured - expected).abs() < 1.0e-10,
        "time_to_next mismatch: measured={measured}, expected={expected}"
    );
}

// ---------------------------------------------------------------------------
// D. Pulse timing and jitter
// ---------------------------------------------------------------------------

#[test]
fn clean_two_hz_pulse_train_has_observed_two_hz_mode_and_predictive_phase() {
    let period = 0.5;
    let mut config = CycleBankConfig::default();
    config.birth_persistence = 2;
    let mut harness = Harness::new(config);

    let mut snapshots = Vec::new();
    for _ in 0..hops(16.0) {
        let t = harness.next_time();
        let phase_in_period = (t / period).rem_euclid(1.0);
        let value = if phase_in_period < 0.06 { 1.0 } else { 0.0 };
        snapshots.push(harness.observe_one("onset", value));
    }

    let final_modes = snapshots.last().unwrap();
    let mode = closest_mode(final_modes, 2.0).expect("2 Hz pulse ridge");
    assert!(
        (mode.frequency_hz - 2.0).abs() / 2.0 < 0.05,
        "pulse ridge frequency inaccurate: {} Hz",
        mode.frequency_hz
    );

    let previous_modes = &snapshots[snapshots.len() - 2];
    let previous =
        mode_by_id(previous_modes, mode.id).expect("stable identity across adjacent hops");
    let predicted = previous.phase_at(DT);
    assert!(
        circular_distance(predicted, mode.phase) < 0.75,
        "pulse ridge next-hop phase prediction is too poor"
    );
}

#[test]
fn deterministically_jittered_pulses_keep_bounded_short_horizon_prediction() {
    let period = 0.5;
    let max_jitter = 0.025; // +/-25 ms
    let pulse_half_width = 0.018;

    let mut config = CycleBankConfig::default();
    config.birth_persistence = 2;
    let mut harness = Harness::new(config);

    let mut recent_errors = Vec::new();
    let mut previous: Option<CycleMode> = None;

    for _ in 0..hops(20.0) {
        let t = harness.next_time();
        let value = jittered_pulse(t, period, max_jitter, pulse_half_width);
        let modes = harness.observe_one("onset", value);

        if let Some(current) = closest_mode(&modes, 2.0) {
            if let Some(previous_mode) = previous.as_ref() {
                if previous_mode.id == current.id {
                    recent_errors.push(circular_distance(
                        previous_mode.phase_at(DT),
                        current.phase,
                    ));
                    if recent_errors.len() > 64 {
                        recent_errors.remove(0);
                    }
                }
            }
            previous = Some(current);
        }
    }

    assert!(
        recent_errors.len() >= 20,
        "jittered pulse mode did not remain observable long enough"
    );

    let max_error = recent_errors
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let mean_error = recent_errors.iter().sum::<f64>() / recent_errors.len() as f64;

    assert!(
        mean_error < 0.55,
        "mean next-hop phase error under jitter too large: {mean_error} rad"
    );
    assert!(
        max_error < 1.4,
        "jitter caused catastrophic next-hop phase error: {max_error} rad"
    );
}

// ---------------------------------------------------------------------------
// E. Missing evidence: free-run and confidence decay
// ---------------------------------------------------------------------------

#[test]
fn missing_evidence_free_runs_existing_mode_and_decays_confidence() {
    let mut config = CycleBankConfig::default();
    config.f_min_hz = 0.5;
    config.f_max_hz = 4.0;
    config.q_cycles = 1.0;
    config.birth_persistence = 2;
    config.free_run_max_observations = 80;
    config.death_persistence = 120;
    config.missing_strength_decay = 0.50;

    let mut harness = Harness::new(config);
    acquire_sinusoid(&mut harness, "mono", 2.0, 0.8, 0.3, 8.0);

    let established = closest_mode(&harness.bank.modes(), 2.0).expect("established ridge");
    let id = established.id;

    // The filter itself rings down causally; with q_cycles = 1 the one-pole
    // baseband at 2 Hz has tau ~= 0.5 s (~46 hops), during which the decaying
    // analytic tail is still legitimate observed support and the tracker keeps
    // matching it.  Wait long enough for the tail to fall below the ridge
    // threshold and the tracked mode to enter the bank's explicit
    // missing/free-run state.
    let mut first_missing: Option<CycleMode> = None;
    for _ in 0..400 {
        let modes = harness.observe_one("mono", 0.0);
        if let Some(mode) = mode_by_id(&modes, id) {
            if mode.missing_observations > 0 {
                first_missing = Some(mode);
                break;
            }
        }
    }

    let first_missing = first_missing.expect("mode never entered free-run state");
    let first_missing_count = first_missing.missing_observations;

    let additional_missing_hops = 10u64;
    let mut later = None;
    for _ in 0..additional_missing_hops {
        let modes = harness.observe_one("mono", 0.0);
        later = mode_by_id(&modes, id);
    }
    let later = later.expect("mode died before configured death hysteresis");

    assert!(
        later.missing_observations >= first_missing_count + additional_missing_hops,
        "missing streak did not advance"
    );

    let expected_phase =
        first_missing.phase_at(additional_missing_hops as f64 * DT);
    assert!(
        circular_distance(later.phase, expected_phase) < 0.15,
        "mode did not free-run predictively during missing evidence"
    );

    assert!(
        later.strength < first_missing.strength,
        "direct ridge strength did not decay during free-run"
    );
    assert!(
        later.confidence < first_missing.confidence,
        "confidence did not decay during unsupported free-run"
    );
}

// ---------------------------------------------------------------------------
// F. Spurious transient robustness
// ---------------------------------------------------------------------------

#[test]
fn short_off_cycle_burst_does_not_destroy_established_two_hz_identity() {
    let mut config = CycleBankConfig::default();
    config.birth_persistence = 2;
    config.death_persistence = 48;
    let mut harness = Harness::new(config);

    acquire_sinusoid(&mut harness, "mono", 2.0, 0.7, 0.1, 12.0);
    let baseline = closest_mode(&harness.bank.modes(), 2.0).expect("baseline ridge");
    let baseline_id = baseline.id;

    let burst_start = harness.current_time();
    for _ in 0..12 {
        let t = harness.next_time();
        let value =
            0.7 * (TAU * 2.0 * t + 0.1).cos()
                + 1.2 * (TAU * 6.5 * (t - burst_start)).cos();
        harness.observe_one("mono", value);
    }

    // Give the original ridge a short recovery interval.
    for _ in 0..20 {
        let t = harness.next_time();
        harness.observe_one("mono", 0.7 * (TAU * 2.0 * t + 0.1).cos());
    }

    let modes = harness.bank.modes();
    let recovered =
        mode_by_id(&modes, baseline_id).expect("established mode identity was destroyed");

    assert!(
        (recovered.frequency_hz - 2.0).abs() / 2.0 < 0.08,
        "established mode was dragged away by spurious burst: {} Hz",
        recovered.frequency_hz
    );
}

// ---------------------------------------------------------------------------
// G. Frequency drift / no-click behavior
// ---------------------------------------------------------------------------

#[test]
fn slow_chirp_tracks_continuously_without_discrete_tempo_state() {
    let f0 = 1.5;
    let f1 = 3.0;
    let duration = 20.0;

    let mut config = CycleBankConfig::default();
    config.f_min_hz = 0.75;
    config.f_max_hz = 4.0;
    config.birth_persistence = 2;
    let mut harness = Harness::new(config);

    // Follow individual mode identities rather than "whichever mode happens to
    // be closest to the instantaneous truth".  A chirp can legitimately cause
    // the tracker to hand off between tracked modes; the architectural
    // requirement is that each persistent mode evolves CONTINUOUSLY (no
    // discrete frequency jump within one mode) and that SOME observed ridge
    // stays near the true drifting frequency.
    let mut observations = Vec::new(); // (true_f, mode id, mode f)
    let n = hops(duration);

    for i in 0..n {
        let t = harness.next_time();
        let chirp_rate = (f1 - f0) / duration;
        let true_frequency = f0 + chirp_rate * t;
        let phase = TAU * (f0 * t + 0.5 * chirp_rate * t * t);
        let modes = harness.observe_one("mono", 0.8 * phase.cos());

        if i > n / 4 {
            if let Some(mode) = closest_mode(&modes, true_frequency) {
                observations.push((true_frequency, mode.id, mode.frequency_hz));
            }
        }
    }

    assert!(
        observations.len() > n / 3,
        "chirp ridge was not observable for enough of the sweep"
    );

    // Some observed ridge stays near the true (drifting) frequency.
    let (true_end, _, recovered_end) = *observations.last().unwrap();
    assert!(
        (recovered_end - true_end).abs() / true_end < 0.15,
        "chirp endpoint lag/error too large: true={true_end}, recovered={recovered_end}"
    );

    // Within each persistent mode identity, consecutive reported frequencies
    // must not jump discretely.  This is the "no discrete tempo jumps"
    // property: it is a per-mode continuity statement, not a statement about
    // whichever mode is currently closest to truth.
    let mut per_mode_max_jump: std::collections::HashMap<u64, f64> =
        std::collections::HashMap::new();
    let mut previous: Option<(u64, f64)> = None;
    for &(_, id, frequency) in &observations {
        if let Some((prev_id, prev_frequency)) = previous {
            if prev_id == id {
                let rel = ((frequency - prev_frequency) / prev_frequency.max(1.0e-9)).abs();
                let entry = per_mode_max_jump.entry(id).or_insert(0.0);
                if rel > *entry {
                    *entry = rel;
                }
            }
        }
        previous = Some((id, frequency));
    }

    let worst = per_mode_max_jump
        .values()
        .copied()
        .fold(0.0_f64, f64::max);
    assert!(
        worst < 0.10,
        "a persistent mode changed frequency discretely (max in-mode relative jump {worst}); \
         per-mode maxima: {per_mode_max_jump:?}"
    );
}

// ---------------------------------------------------------------------------
// H. Multiple simultaneous observed modes / independent channels
// ---------------------------------------------------------------------------

#[test]
fn independent_evidence_channels_can_support_simultaneous_modes() {
    let f_low = 1.4;
    let f_high = 3.6;

    let mut config = CycleBankConfig::default();
    config.f_min_hz = 0.75;
    config.f_max_hz = 5.0;
    config.birth_persistence = 2;
    let mut harness = Harness::new(config);

    for _ in 0..hops(DEFAULT_WARMUP_SECONDS) {
        let t = harness.next_time();
        harness.observe(vec![
            CycleEvidenceChannel::new("low", 0.8 * (TAU * f_low * t).cos()),
            CycleEvidenceChannel::new("high", 0.8 * (TAU * f_high * t).cos()),
        ]);
    }

    let modes = harness.bank.modes();
    let low = closest_mode(&modes, f_low).expect("low-frequency observed ridge");
    let high = closest_mode(&modes, f_high).expect("high-frequency observed ridge");

    assert_ne!(low.id, high.id, "two observed clocks collapsed into one mode");
    assert!(
        (low.frequency_hz - f_low).abs() / f_low < 0.06,
        "low observed mode inaccurate: {low:?}"
    );
    assert!(
        (high.frequency_hz - f_high).abs() / f_high < 0.06,
        "high observed mode inaccurate: {high:?}"
    );
}

// ---------------------------------------------------------------------------
// I. Rational / polymetric relation diagnostics (observed modes only)
// ---------------------------------------------------------------------------

#[test]
fn observed_five_four_relation_has_stable_generalized_phase() {
    let f_low = 1.25;
    let f_high = f_low * 5.0 / 4.0;

    let mut config = CycleBankConfig::default();
    config.f_min_hz = 0.75;
    config.f_max_hz = 3.0;
    config.birth_persistence = 2;
    config.rational_ratio_tolerance = 0.03;
    config.relation_history_len = 48;

    let mut harness = Harness::new(config);
    for _ in 0..hops(24.0) {
        let t = harness.next_time();
        harness.observe(vec![
            CycleEvidenceChannel::new("a", 0.8 * (TAU * f_low * t + 0.2).cos()),
            CycleEvidenceChannel::new("b", 0.8 * (TAU * f_high * t - 0.4).cos()),
        ]);
    }

    let modes = harness.bank.modes();
    let low = closest_mode(&modes, f_low).expect("low 5:4 ridge");
    let high = closest_mode(&modes, f_high).expect("high 5:4 ridge");
    assert_ne!(low.id, high.id);

    let relations = harness.bank.latest_relations();
    let relation =
        relation_for_pair(&relations, &low, &high, 5, 4).expect("5:4 relation");

    assert!(
        relation.freq_residual < 0.03,
        "5:4 frequency relation residual too large: {relation:?}"
    );
    assert!(
        relation.phase_stability < 0.20,
        "5:4 generalized phase is not stable across time: {relation:?}"
    );
}

#[test]
fn observed_seven_four_relation_is_representable_without_meter_specific_code() {
    let f_low = 1.0;
    let f_high = 1.75;

    let mut config = CycleBankConfig::default();
    config.f_min_hz = 0.5;
    config.f_max_hz = 3.0;
    config.birth_persistence = 2;
    config.rational_ratio_tolerance = 0.03;
    config.relation_history_len = 48;

    let mut harness = Harness::new(config);
    for _ in 0..hops(24.0) {
        let t = harness.next_time();
        harness.observe(vec![
            CycleEvidenceChannel::new("a", 0.8 * (TAU * f_low * t + 0.1).cos()),
            CycleEvidenceChannel::new("b", 0.8 * (TAU * f_high * t + 0.7).cos()),
        ]);
    }

    let modes = harness.bank.modes();
    let low = closest_mode(&modes, f_low).expect("low 7:4 ridge");
    let high = closest_mode(&modes, f_high).expect("high 7:4 ridge");
    let relations = harness.bank.latest_relations();

    let relation =
        relation_for_pair(&relations, &low, &high, 7, 4).expect("7:4 relation");
    assert!(
        relation.phase_stability < 0.25,
        "7:4 relation did not remain phase-stable: {relation:?}"
    );
}

// ---------------------------------------------------------------------------
// J. Causality
// ---------------------------------------------------------------------------

fn causal_run(prefix_hops: usize, total_hops: usize, future_kind: u8) -> Vec<Vec<CycleMode>> {
    let mut config = CycleBankConfig::default();
    config.birth_persistence = 2;
    let mut harness = Harness::new(config);

    let mut snapshots = Vec::with_capacity(total_hops);
    for i in 0..total_hops {
        let t = harness.next_time();
        let value = if i < prefix_hops {
            0.8 * (TAU * 2.0 * t + 0.3).cos()
        } else {
            match future_kind {
                0 => 0.8 * (TAU * 2.0 * t + 0.3).cos(),
                1 => 0.0,
                _ => 0.8 * (TAU * 3.7 * t - 0.9).cos(),
            }
        };
        snapshots.push(harness.observe_one("mono", value));
    }
    snapshots
}

#[test]
fn changing_future_evidence_cannot_change_already_emitted_cycle_state() {
    let prefix_hops = 180;
    let total_hops = 260;

    let a = causal_run(prefix_hops, total_hops, 0);
    let b = causal_run(prefix_hops, total_hops, 1);
    let c = causal_run(prefix_hops, total_hops, 2);

    for i in 0..prefix_hops {
        assert_eq!(a[i], b[i], "future silence changed past state at hop {i}");
        assert_eq!(
            a[i], c[i],
            "future different-frequency evidence changed past state at hop {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// K. Deterministic replay / caller chunk-group invariance
// ---------------------------------------------------------------------------

fn deterministic_stream_value(i: usize) -> f64 {
    let t = (i + 1) as f64 * DT;
    0.55 * (TAU * 1.8 * t + 0.2).cos()
        + 0.25 * (TAU * 3.1 * t - 0.4).cos()
}

#[test]
fn same_per_hop_observations_are_invariant_to_caller_chunk_grouping() {
    // CycleBank deliberately has a per-authoritative-hop API.  Transport PCM
    // chunk partitioning is already normalized by AnalysisTimebase.  The
    // relevant invariant here is that grouping those per-hop calls differently
    // in the caller cannot change bank state.
    let n_hops = 240usize;

    let mut one_by_one = Harness::new(CycleBankConfig::default());
    let mut states_a = Vec::new();
    for i in 0..n_hops {
        states_a.push(one_by_one.observe_one("mono", deterministic_stream_value(i)));
    }

    let mut grouped = Harness::new(CycleBankConfig::default());
    let mut states_b = Vec::new();
    let group = 7usize;
    for chunk_start in (0..n_hops).step_by(group) {
        let chunk_end = (chunk_start + group).min(n_hops);
        for i in chunk_start..chunk_end {
            states_b.push(grouped.observe_one("mono", deterministic_stream_value(i)));
        }
    }

    assert_eq!(states_a, states_b);
}

// ---------------------------------------------------------------------------
// L. Stream epoch reset hygiene
// ---------------------------------------------------------------------------

#[test]
fn stream_epoch_change_clears_old_modes_and_allows_new_channel_schema() {
    let mut config = CycleBankConfig::default();
    config.f_min_hz = 0.5;
    config.f_max_hz = 4.0;
    config.birth_persistence = 2;
    let mut harness = Harness::new(config);

    acquire_sinusoid(&mut harness, "old", 1.5, 0.8, 0.0, 14.0);
    let old_modes = harness.bank.modes();
    assert!(
        closest_mode(&old_modes, 1.5).is_some(),
        "epoch 1 never acquired its observed ridge"
    );

    harness.change_epoch(2, true);
    let first_new = harness.observe_one("new", 0.0);

    assert!(
        first_new.is_empty(),
        "old epoch modes leaked through the epoch boundary: {first_new:?}"
    );

    acquire_sinusoid(&mut harness, "new", 3.4, 0.8, 0.5, 14.0);
    let new_modes = harness.bank.modes();

    let has_old = new_modes
        .iter()
        .any(|mode| (mode.frequency_hz - 1.5).abs() / 1.5 < 0.05);
    let has_new = new_modes
        .iter()
        .any(|mode| (mode.frequency_hz - 3.4).abs() / 3.4 < 0.06);

    assert!(!has_old, "epoch-1 ridge leaked into epoch 2: {new_modes:?}");
    assert!(has_new, "epoch-2 ridge was not acquired: {new_modes:?}");
}

// ---------------------------------------------------------------------------
// M. Representation/state-management regressions from the previous draft
// ---------------------------------------------------------------------------

#[test]
fn confirmed_mode_age_increments_once_per_observation() {
    let mut config = CycleBankConfig::default();
    config.birth_persistence = 2;
    let mut harness = Harness::new(config);

    acquire_sinusoid(&mut harness, "mono", 2.0, 0.8, 0.0, 12.0);
    let before = closest_mode(&harness.bank.modes(), 2.0).expect("mode");
    let id = before.id;

    let t = harness.next_time();
    let modes = harness.observe_one("mono", 0.8 * (TAU * 2.0 * t).cos());
    let after = mode_by_id(&modes, id).expect("same mode id");

    assert_eq!(
        after.age,
        before.age + 1,
        "mode age should advance exactly once per authoritative observation"
    );
}

#[test]
fn public_mode_count_never_exceeds_configured_max_modes() {
    let mut config = CycleBankConfig::default();
    config.f_min_hz = 0.5;
    config.f_max_hz = 7.5;
    config.birth_persistence = 2;
    config.max_modes = 3;
    let max_modes = config.max_modes;

    let mut harness = Harness::new(config);
    let frequencies = [0.8, 1.2, 1.8, 2.7, 4.1, 6.2];

    for _ in 0..hops(20.0) {
        let t = harness.next_time();
        let channels = frequencies
            .iter()
            .enumerate()
            .map(|(i, &frequency)| {
                CycleEvidenceChannel::new(
                    format!("ch{i}"),
                    0.7 * (TAU * frequency * t + i as f64 * 0.13).cos(),
                )
            })
            .collect();
        harness.observe(channels);
        assert!(
            harness.bank.num_modes() <= max_modes,
            "mode budget exceeded: {} > {}",
            harness.bank.num_modes(),
            max_modes
        );
    }
}

#[test]
fn confidence_and_support_are_bounded() {
    let mut harness = Harness::new(CycleBankConfig::default());
    let snapshots =
        acquire_sinusoid(&mut harness, "mono", 2.0, 0.8, 0.2, 10.0);

    for snapshot in snapshots {
        for mode in snapshot {
            assert!(
                (0.0..=1.0).contains(&mode.confidence),
                "confidence out of bounds: {mode:?}"
            );
            assert!(
                (0.0..=1.0).contains(&mode.channel_support),
                "channel support out of bounds: {mode:?}"
            );
            assert!(
                mode.frequency_hz.is_finite() && mode.frequency_hz > 0.0,
                "mode frequency is not finite/positive: {mode:?}"
            );
            assert!(
                mode.phase.is_finite() && (-PI..=PI).contains(&mode.phase),
                "mode phase not wrapped to [-pi, pi]: {mode:?}"
            );
        }
    }
}

#[test]
fn explicit_reset_clears_public_state() {
    let mut harness = Harness::new(CycleBankConfig::default());
    acquire_sinusoid(&mut harness, "mono", 2.0, 0.8, 0.0, 12.0);
    assert!(harness.bank.num_modes() > 0);

    harness.bank.reset();

    assert_eq!(harness.bank.num_modes(), 0);
    assert!(harness.bank.latest_relations().is_empty());
    assert!(harness.bank.relations_snapshot().is_empty());
}
