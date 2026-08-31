//! Canonical predictive musical-timing primitive (issue #92).
//!
//! `CycleBank` is the Rust authority for directly observed predictive temporal
//! modes.  It implements the architecture from ADR 0002:
//!
//! causal scalar evidence at the authoritative hop clock
//!     -> causal constant-Q analytic field
//!     -> continuous instantaneous-frequency estimates
//!     -> observed ridge candidates
//!     -> persistent `CycleMode`s
//!
//! Important invariants:
//! - transform scales are numerical sampling locations, not musical buckets;
//! - mode frequency is continuous and is never snapped to a scale center;
//! - the runtime path is strictly causal;
//! - each evidence channel contributes exactly one new scalar per observation;
//! - there is no global tempo / beat / bar / phrase tracker here;
//! - latent missing-fundamental / undertone inference belongs to issue #97;
//! - Python and TypeScript consume bindings and must not mirror this math.
//!
//! ## Analytic transform used by v1
//!
//! At each logarithmic center frequency `f_c`, the real input is mixed to
//! baseband with an accumulated complex oscillator and passed through a short
//! cascade of one-pole low-pass sections.  For one section:
//!
//! ```text
//! theta[n] = theta[n-1] + 2*pi*f_c*dt
//! u[n]     = x[n] * exp(-i*theta[n])
//! b[n]     = rho*b[n-1] + (1-rho)*u[n]
//! rho      = exp(-dt*f_c / q_cycles)
//! ```
//!
//! Two cascaded sections strongly suppress the negative-frequency image from
//! real input while preserving a cheap recursive implementation.  The final
//! baseband phase increment gives a continuous detuning estimate:
//!
//! ```text
//! f_hat = f_c + arg(b[n] * conj(b[n-1])) / (2*pi*dt)
//! ```
//!
//! Thus a 2.1667 Hz mode can be reported near 2.1667 Hz even when no scale is
//! centered there.
//!
//! The baseband low-pass contributes a deterministic off-center phase shift.
//! `scale_measurement()` removes the exact discrete transfer phase of the
//! cascade before exposing a ridge phase, so `CycleMode.phase` is the estimated
//! input analytic phase at the current observation, not the filter's internal
//! phase.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::{PI, TAU};

/// Rust-owned public contract version for observed-ridge temporal state.
pub const CYCLE_BANK_VERSION: &str = "cycle-bank/2";

/// Number of causal baseband low-pass sections per scale.
///
/// Two sections are enough to strongly reject the negative-frequency image of
/// real scalar evidence while keeping the filter cheap at the ~46.875 Hz hop
/// cadence.  Changing this changes phase/strength semantics and therefore
/// requires a `CYCLE_BANK_VERSION` bump.
const ANALYTIC_STAGES: usize = 2;

const MODE_HISTORY_LEN: usize = 32;
const CONFIDENCE_STRENGTH_SCALE: f64 = 0.10;

/// One new scalar value from one causal evidence channel at one analysis hop.
///
/// The scalar shape intentionally makes the invalid "60 rolling feature values
/// are 60 new temporal samples" state unrepresentable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CycleEvidenceChannel {
    pub name: String,
    pub value: f64,
}

impl CycleEvidenceChannel {
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// One causal observation at the authoritative analysis-hop clock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CycleObservation {
    /// Canonical sample-clock identity of this observation.
    pub sample_index: u64,
    /// Elapsed authoritative audio time since the previous observation.
    pub dt_seconds: f64,
    /// Stream epoch from `AnalysisTimebase`; a change resets temporal state.
    pub stream_epoch: u64,
    /// Exactly one new scalar per named evidence channel.
    pub channels: Vec<CycleEvidenceChannel>,
}

/// Configuration for the observed-ridge CycleBank.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CycleBankConfig {
    /// Inclusive analysis range in Hz.  These are numerical limits, not
    /// semantic beat/phrase limits.
    pub f_min_hz: f64,
    pub f_max_hz: f64,

    /// Effective memory measured in cycles: tau = q_cycles / f_center.
    pub q_cycles: f64,

    /// Numerical logarithmic scale density.  This must be selected by
    /// convergence tests, not treated as "musical buckets per octave".
    pub scales_per_octave: usize,

    /// Minimum baseband coefficient magnitude used for IF estimation and ridge
    /// detection.
    pub weak_threshold: f64,

    /// Maximum number of confirmed simultaneous observed modes.
    pub max_modes: usize,

    /// Temporal association gate in natural-log frequency distance.
    pub association_log_freq_tolerance: f64,

    /// Temporal association gate for phase prediction residual.
    pub association_phase_tolerance_rad: f64,

    /// Gain applied to candidate phase residual when correcting a predicted
    /// tracked phase.  1.0 follows the measurement; 0.0 free-runs only.
    pub phase_correction_gain: f64,

    /// Exponential update weight for measured continuous frequency.
    pub frequency_smoothing: f64,

    /// Exponential update weight for direct ridge strength/support.
    pub strength_smoothing: f64,

    /// Exponential update weight for the diagnostic frequency slope.
    pub slope_smoothing: f64,

    /// Consecutive matched observations required before a pending candidate is
    /// promoted to a public tracked mode.
    pub birth_persistence: usize,

    /// Confidence reaches zero after this many consecutive missing
    /// observations (the mode may remain alive a little longer).
    pub free_run_max_observations: usize,

    /// Confirmed mode is removed after this many consecutive missing
    /// observations.
    pub death_persistence: usize,

    /// Per-hop direct-strength decay while a mode is free-running without a
    /// current observed ridge.
    pub missing_strength_decay: f64,

    /// Near-duplicate confirmed modes closer than both merge tolerances are
    /// collapsed deterministically, preserving the older id.
    pub merge_log_freq_tolerance: f64,
    pub merge_phase_tolerance_rad: f64,

    /// Small-integer rational-relation search bounds.
    pub max_numer: u32,
    pub max_denom: u32,
    pub rational_ratio_tolerance: f64,

    /// Number of generalized-phase observations retained per relation when
    /// estimating phase-lock stability.
    pub relation_history_len: usize,
}

impl Default for CycleBankConfig {
    fn default() -> Self {
        Self {
            f_min_hz: 0.0625,
            f_max_hz: 8.0,
            q_cycles: 4.0,
            // Numerical convergence choice, not a musical constant.  Measured
            // (docs/cycle_bank_scale_convergence.md): continuous frequency
            // recovery converges by 6 scales/octave (~0.3% rel err); 12 keeps
            // comfortable margin at negligible cost.  Enforced by the
            // scale-resolution convergence test.
            scales_per_octave: 12,
            weak_threshold: 1.0e-4,
            max_modes: 8,
            association_log_freq_tolerance: 0.18,
            association_phase_tolerance_rad: PI * 0.75,
            phase_correction_gain: 0.65,
            frequency_smoothing: 0.25,
            strength_smoothing: 0.40,
            slope_smoothing: 0.20,
            birth_persistence: 3,
            free_run_max_observations: 24,
            death_persistence: 32,
            missing_strength_decay: 0.90,
            merge_log_freq_tolerance: 0.025,
            merge_phase_tolerance_rad: PI / 4.0,
            max_numer: 8,
            max_denom: 8,
            rational_ratio_tolerance: 0.02,
            relation_history_len: 32,
        }
    }
}

impl CycleBankConfig {
    fn validate(&self) -> Result<(), CycleBankError> {
        if !self.f_min_hz.is_finite() || self.f_min_hz <= 0.0 {
            return Err(CycleBankError::InvalidConfig(
                "f_min_hz must be finite and > 0".into(),
            ));
        }
        if !self.f_max_hz.is_finite() || self.f_max_hz <= self.f_min_hz {
            return Err(CycleBankError::InvalidConfig(
                "f_max_hz must be finite and > f_min_hz".into(),
            ));
        }
        if !self.q_cycles.is_finite() || self.q_cycles <= 0.0 {
            return Err(CycleBankError::InvalidConfig(
                "q_cycles must be finite and > 0".into(),
            ));
        }
        if self.scales_per_octave == 0 {
            return Err(CycleBankError::InvalidConfig(
                "scales_per_octave must be > 0".into(),
            ));
        }
        if self.max_modes == 0 {
            return Err(CycleBankError::InvalidConfig(
                "max_modes must be > 0".into(),
            ));
        }
        if self.birth_persistence == 0 {
            return Err(CycleBankError::InvalidConfig(
                "birth_persistence must be > 0".into(),
            ));
        }
        if self.free_run_max_observations == 0 {
            return Err(CycleBankError::InvalidConfig(
                "free_run_max_observations must be > 0".into(),
            ));
        }
        if self.death_persistence < self.free_run_max_observations {
            return Err(CycleBankError::InvalidConfig(
                "death_persistence must be >= free_run_max_observations".into(),
            ));
        }
        for (name, value) in [
            ("phase_correction_gain", self.phase_correction_gain),
            ("frequency_smoothing", self.frequency_smoothing),
            ("strength_smoothing", self.strength_smoothing),
            ("slope_smoothing", self.slope_smoothing),
            ("missing_strength_decay", self.missing_strength_decay),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(CycleBankError::InvalidConfig(format!(
                    "{name} must be finite and in [0, 1]"
                )));
            }
        }
        if !self.association_log_freq_tolerance.is_finite()
            || self.association_log_freq_tolerance <= 0.0
            || !self.association_phase_tolerance_rad.is_finite()
            || self.association_phase_tolerance_rad <= 0.0
        {
            return Err(CycleBankError::InvalidConfig(
                "association tolerances must be finite and > 0".into(),
            ));
        }
        if self.relation_history_len == 0 {
            return Err(CycleBankError::InvalidConfig(
                "relation_history_len must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// One confirmed directly-observed temporal ridge.
///
/// Serialized with **camelCase** keys so the wasm binding and the PyO3
/// binding emit the same wire shape (cross-surface parity, issue #93).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CycleMode {
    /// Stable within one stream epoch; ids restart after reset.
    pub id: u64,
    /// Continuous estimated frequency in Hz.
    pub frequency_hz: f64,
    /// Estimated input analytic phase at the current observation, wrapped to
    /// (-pi, pi].  The causal filter transfer phase has already been removed.
    pub phase: f64,
    /// Direct current ridge support.  This is not confidence.
    pub strength: f64,
    /// Deterministic trust score in [0, 1].
    pub confidence: f64,
    /// Fraction of evidence channels currently contributing to this mode.
    pub channel_support: f64,
    pub age: u64,
    pub missing_observations: u64,
    /// Smoothed diagnostic derivative in Hz/s.  v1 prediction remains first
    /// order in frequency; this field is exposed for diagnostics only.
    pub frequency_slope: f64,
    /// Variance of recent continuous frequency measurements (Hz^2).
    pub frequency_uncertainty: f64,
}

impl CycleMode {
    /// First-order causal free-running phase prediction.
    pub fn phase_at(&self, delta_seconds: f64) -> f64 {
        wrap_phase(self.phase + TAU * self.frequency_hz * delta_seconds)
    }

    /// Time until the next occurrence of `reference_phase`, assuming constant
    /// current frequency.
    pub fn time_to_next(&self, reference_phase: f64) -> Option<f64> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return None;
        }
        let diff = positive_phase(reference_phase - self.phase);
        Some(diff / (TAU * self.frequency_hz))
    }
}

/// Diagnostic rational relationship between two *observed* modes.
///
/// Serialized with **camelCase** keys to match the wasm/PyO3 wire shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CycleRelation {
    pub i_id: u64,
    pub j_id: u64,
    pub m: u32,
    pub n: u32,
    pub freq_residual: f64,
    pub generalized_phase: f64,
    /// Circular variance of the generalized phase over recent observations.
    /// 0 is stable phase lock; 1 is maximally incoherent.
    pub phase_stability: f64,
}

#[derive(Clone, Debug)]
struct ScaleState {
    f_center: f64,
    omega_center: f64,
    /// Accumulated demodulator phase.  This is the piece the old implementation
    /// was missing: each observation advances the oscillator, rather than
    /// multiplying every input by the same complex number.
    demod_phase: f64,
    lowpass: [Complex64; ANALYTIC_STAGES],
    prev_output: Complex64,
    rho: f64,
    has_previous_output: bool,
}

impl ScaleState {
    fn output(&self) -> Complex64 {
        self.lowpass[ANALYTIC_STAGES - 1]
    }
}

#[derive(Clone, Debug)]
struct ScaleMeasurement {
    frequency_hz: f64,
    phase: f64,
    strength: f64,
}

#[derive(Clone, Debug)]
struct ChannelRidge {
    channel_index: usize,
    frequency_hz: f64,
    phase: f64,
    strength: f64,
}

#[derive(Clone, Debug)]
struct RidgeCandidate {
    frequency_hz: f64,
    phase: f64,
    strength: f64,
    channel_support: f64,
    support_by_channel: Vec<f64>,
}

#[derive(Clone, Debug)]
struct TrackedMode {
    id: u64,
    frequency_hz: f64,
    phase: f64,
    frequency_slope: f64,
    strength: f64,
    support_by_channel: Vec<f64>,
    age: u64,
    missing_streak: usize,
    frequency_history: VecDeque<f64>,
    phase_error_history: VecDeque<f64>,
}

impl TrackedMode {
    fn predicted_phase(&self, dt_seconds: f64) -> f64 {
        wrap_phase(self.phase + TAU * self.frequency_hz * dt_seconds)
    }

    fn channel_support(&self) -> f64 {
        if self.support_by_channel.is_empty() || self.missing_streak > 0 {
            return 0.0;
        }
        let active = self
            .support_by_channel
            .iter()
            .filter(|&&x| x > 0.0)
            .count();
        active as f64 / self.support_by_channel.len() as f64
    }

    fn frequency_variance(&self) -> f64 {
        variance(self.frequency_history.iter().copied())
    }

    fn confidence(&self, cfg: &CycleBankConfig) -> f64 {
        let age_score = 1.0 - (-(self.age as f64) / 16.0).exp();
        let strength_score = if self.strength <= 0.0 {
            0.0
        } else {
            self.strength / (self.strength + CONFIDENCE_STRENGTH_SCALE)
        };
        let support_score = self.channel_support();

        // Coherence is measured on prediction residuals, not raw rotating
        // phase.  A perfect oscillator visits every phase; raw-phase circular
        // variance would therefore incorrectly call it unstable.
        let phase_residual_rms = circular_rms(self.phase_error_history.iter().copied());
        let phase_scale = cfg.association_phase_tolerance_rad.max(1.0e-9);
        let phase_score = (-(phase_residual_rms / phase_scale).powi(2)).exp();

        let freq_score = if self.frequency_history.len() < 2 || self.frequency_hz <= 1.0e-9 {
            0.5
        } else {
            let rel_std = self.frequency_variance().sqrt() / self.frequency_hz.abs();
            (-25.0 * rel_std * rel_std).exp()
        };

        let missing_score = (1.0
            - self.missing_streak as f64 / cfg.free_run_max_observations as f64)
            .clamp(0.0, 1.0);

        (0.20 * age_score
            + 0.20 * strength_score
            + 0.15 * support_score
            + 0.20 * phase_score
            + 0.15 * freq_score
            + 0.10 * missing_score)
            .clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug)]
struct PendingMode {
    frequency_hz: f64,
    phase: f64,
    strength: f64,
    support_by_channel: Vec<f64>,
    consecutive_hits: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RelationKey {
    i_id: u64,
    j_id: u64,
    m: u32,
    n: u32,
}

/// Canonical Rust observed-ridge cycle bank.
pub struct CycleBank {
    cfg: CycleBankConfig,
    channel_names: Vec<String>,
    channel_banks: Vec<Vec<ScaleState>>,
    modes: Vec<TrackedMode>,
    pending: Vec<PendingMode>,
    relation_phase_history: HashMap<RelationKey, VecDeque<f64>>,
    relation_snapshots: VecDeque<Vec<CycleRelation>>,
    last_epoch: Option<u64>,
    last_sample_index: Option<u64>,
    next_id: u64,
}

impl CycleBank {
    /// Construct a bank, panicking only for programmer-invalid configuration.
    /// Bindings that need recoverable config errors can call `try_new`.
    pub fn new(config: CycleBankConfig) -> Self {
        Self::try_new(config).expect("invalid CycleBankConfig")
    }

    pub fn try_new(config: CycleBankConfig) -> Result<Self, CycleBankError> {
        config.validate()?;
        Ok(Self {
            cfg: config,
            channel_names: Vec::new(),
            channel_banks: Vec::new(),
            modes: Vec::new(),
            pending: Vec::new(),
            relation_phase_history: HashMap::new(),
            relation_snapshots: VecDeque::new(),
            last_epoch: None,
            last_sample_index: None,
            next_id: 1,
        })
    }

    pub fn config(&self) -> &CycleBankConfig {
        &self.cfg
    }

    pub fn version(&self) -> &'static str {
        CYCLE_BANK_VERSION
    }

    pub fn num_modes(&self) -> usize {
        self.modes.len()
    }

    pub fn modes(&self) -> Vec<CycleMode> {
        let mut out: Vec<CycleMode> = self
            .modes
            .iter()
            .map(|mode| CycleMode {
                id: mode.id,
                frequency_hz: mode.frequency_hz,
                phase: mode.phase,
                strength: mode.strength,
                confidence: mode.confidence(&self.cfg),
                channel_support: mode.channel_support(),
                age: mode.age,
                missing_observations: mode.missing_streak as u64,
                frequency_slope: mode.frequency_slope,
                frequency_uncertainty: mode.frequency_variance(),
            })
            .collect();
        out.sort_by_key(|mode| mode.id);
        out
    }

    pub fn latest_relations(&self) -> Vec<CycleRelation> {
        self.relation_snapshots
            .back()
            .cloned()
            .unwrap_or_default()
    }

    /// Flatten recent relation snapshots for diagnostics.  Stability values in
    /// each relation were computed from per-relation history across time, not
    /// from the single current batch.
    pub fn relations_snapshot(&self) -> Vec<CycleRelation> {
        self.relation_snapshots.iter().flatten().cloned().collect()
    }

    /// Process exactly one authoritative temporal observation.
    pub fn observe(&mut self, obs: &CycleObservation) -> Result<(), CycleBankError> {
        validate_observation(obs)?;

        let epoch_changed = self
            .last_epoch
            .is_some_and(|previous| previous != obs.stream_epoch);
        if epoch_changed {
            self.reset_internal();
        }

        if let (Some(previous_epoch), Some(previous_sample)) =
            (self.last_epoch, self.last_sample_index)
        {
            if previous_epoch == obs.stream_epoch && obs.sample_index <= previous_sample {
                return Err(CycleBankError::NonMonotonicSampleIndex {
                    previous: previous_sample,
                    current: obs.sample_index,
                });
            }
        }

        if self.channel_banks.is_empty() {
            self.initialize_channels(obs)?;
        } else {
            self.validate_channel_schema(obs)?;
        }

        self.last_epoch = Some(obs.stream_epoch);
        self.last_sample_index = Some(obs.sample_index);

        for (bank, channel) in self.channel_banks.iter_mut().zip(&obs.channels) {
            step_grid(bank, channel.value, obs.dt_seconds, self.cfg.q_cycles);
        }

        let candidates = self.collect_ridge_candidates(obs.dt_seconds);
        self.update_modes(candidates, obs.dt_seconds);
        self.merge_near_duplicates();
        self.update_relations();

        Ok(())
    }

    /// Explicit discontinuity reset.  The next observation may establish a new
    /// channel schema and begins a new local mode-id sequence.
    pub fn reset(&mut self) {
        self.reset_internal();
    }

    fn initialize_channels(&mut self, obs: &CycleObservation) -> Result<(), CycleBankError> {
        let mut seen = HashSet::new();
        for channel in &obs.channels {
            if channel.name.trim().is_empty() {
                return Err(CycleBankError::InvalidChannelName(
                    "channel names must be non-empty".into(),
                ));
            }
            if !seen.insert(channel.name.clone()) {
                return Err(CycleBankError::InvalidChannelName(format!(
                    "duplicate channel name: {}",
                    channel.name
                )));
            }
        }

        self.channel_names = obs.channels.iter().map(|c| c.name.clone()).collect();
        self.channel_banks = (0..obs.channels.len())
            .map(|_| build_grid(&self.cfg, obs.dt_seconds))
            .collect();
        Ok(())
    }

    fn validate_channel_schema(&self, obs: &CycleObservation) -> Result<(), CycleBankError> {
        let current: Vec<&str> = obs.channels.iter().map(|c| c.name.as_str()).collect();
        let expected: Vec<&str> = self.channel_names.iter().map(String::as_str).collect();
        if current != expected {
            return Err(CycleBankError::ChannelSchemaMismatch {
                expected: self.channel_names.clone(),
                got: obs.channels.iter().map(|c| c.name.clone()).collect(),
            });
        }
        Ok(())
    }

    fn reset_internal(&mut self) {
        self.channel_names.clear();
        self.channel_banks.clear();
        self.modes.clear();
        self.pending.clear();
        self.relation_phase_history.clear();
        self.relation_snapshots.clear();
        self.last_epoch = None;
        self.last_sample_index = None;
        self.next_id = 1;
    }

    fn collect_ridge_candidates(&self, dt_seconds: f64) -> Vec<RidgeCandidate> {
        let n_channels = self.channel_banks.len();
        if n_channels == 0 {
            return Vec::new();
        }

        let mut channel_ridges = Vec::new();
        for (channel_index, bank) in self.channel_banks.iter().enumerate() {
            let measurements: Vec<Option<ScaleMeasurement>> = bank
                .iter()
                .map(|scale| scale_measurement(scale, dt_seconds, &self.cfg))
                .collect();

            for i in 0..bank.len() {
                let Some(measurement) = measurements[i].as_ref() else {
                    continue;
                };
                if !measurement_is_local_max(&measurements, i) {
                    continue;
                }
                channel_ridges.push(ChannelRidge {
                    channel_index,
                    frequency_hz: measurement.frequency_hz,
                    phase: measurement.phase,
                    strength: measurement.strength,
                });
            }
        }

        cluster_channel_ridges(
            channel_ridges,
            n_channels,
            self.cfg.scales_per_octave,
            self.cfg.max_modes,
        )
    }

    fn update_modes(&mut self, candidates: Vec<RidgeCandidate>, dt_seconds: f64) {
        // Prediction first.  We use the predicted phase in association and only
        // then correct it from a matched observed ridge.
        let predicted_phases: Vec<f64> = self
            .modes
            .iter()
            .map(|mode| mode.predicted_phase(dt_seconds))
            .collect();

        let mut pair_costs = Vec::new();
        for (mode_index, mode) in self.modes.iter().enumerate() {
            for (candidate_index, candidate) in candidates.iter().enumerate() {
                if let Some(cost) = association_cost(
                    mode,
                    predicted_phases[mode_index],
                    candidate,
                    &self.cfg,
                ) {
                    pair_costs.push((cost, mode_index, candidate_index));
                }
            }
        }
        pair_costs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut mode_matched = vec![false; self.modes.len()];
        let mut candidate_matched = vec![false; candidates.len()];

        for (_, mode_index, candidate_index) in pair_costs {
            if mode_matched[mode_index] || candidate_matched[candidate_index] {
                continue;
            }
            let candidate = &candidates[candidate_index];
            update_matched_mode(
                &mut self.modes[mode_index],
                predicted_phases[mode_index],
                candidate,
                dt_seconds,
                &self.cfg,
            );
            mode_matched[mode_index] = true;
            candidate_matched[candidate_index] = true;
        }

        // Confirmed modes that were not directly observed free-run instead of
        // phase-resetting.  Age is incremented exactly once per observation.
        for (index, mode) in self.modes.iter_mut().enumerate() {
            if mode_matched[index] {
                continue;
            }
            mode.phase = predicted_phases[index];
            mode.age = mode.age.saturating_add(1);
            mode.missing_streak = mode.missing_streak.saturating_add(1);
            mode.strength *= self.cfg.missing_strength_decay;
            mode.frequency_slope *= 0.9;
            for support in &mut mode.support_by_channel {
                *support *= self.cfg.missing_strength_decay;
            }
        }
        self.modes
            .retain(|mode| mode.missing_streak < self.cfg.death_persistence);

        let unmatched_candidates: Vec<RidgeCandidate> = candidates
            .into_iter()
            .enumerate()
            .filter_map(|(index, candidate)| (!candidate_matched[index]).then_some(candidate))
            .collect();
        self.update_pending(unmatched_candidates, dt_seconds);
    }

    fn update_pending(&mut self, candidates: Vec<RidgeCandidate>, dt_seconds: f64) {
        let mut pending_used = vec![false; self.pending.len()];
        let mut candidate_used = vec![false; candidates.len()];

        let mut costs = Vec::new();
        for (pending_index, pending) in self.pending.iter().enumerate() {
            for (candidate_index, candidate) in candidates.iter().enumerate() {
                let predicted = wrap_phase(
                    pending.phase + TAU * pending.frequency_hz * dt_seconds,
                );
                let freq_distance = log_frequency_distance(
                    pending.frequency_hz,
                    candidate.frequency_hz,
                );
                let phase_distance = circular_distance(predicted, candidate.phase);
                if freq_distance <= self.cfg.association_log_freq_tolerance
                    && phase_distance <= self.cfg.association_phase_tolerance_rad
                {
                    costs.push((
                        freq_distance
                            / self.cfg.association_log_freq_tolerance
                            + phase_distance / self.cfg.association_phase_tolerance_rad,
                        pending_index,
                        candidate_index,
                    ));
                }
            }
        }
        costs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        for (_, pending_index, candidate_index) in costs {
            if pending_used[pending_index] || candidate_used[candidate_index] {
                continue;
            }
            let candidate = &candidates[candidate_index];
            let pending = &mut self.pending[pending_index];
            pending.frequency_hz = lerp(
                pending.frequency_hz,
                candidate.frequency_hz,
                self.cfg.frequency_smoothing,
            );
            pending.phase = candidate.phase;
            pending.strength = lerp(
                pending.strength,
                candidate.strength,
                self.cfg.strength_smoothing,
            );
            blend_support(
                &mut pending.support_by_channel,
                &candidate.support_by_channel,
                self.cfg.strength_smoothing,
            );
            pending.consecutive_hits += 1;
            pending_used[pending_index] = true;
            candidate_used[candidate_index] = true;
        }

        // Birth persistence means consecutive evidence.  An unmatched pending
        // candidate is discarded rather than being publicly visible early.
        let mut survivors = Vec::new();
        let mut promotions = Vec::new();
        for (index, pending) in self.pending.drain(..).enumerate() {
            if !pending_used[index] {
                continue;
            }
            if pending.consecutive_hits >= self.cfg.birth_persistence {
                promotions.push(pending);
            } else {
                survivors.push(pending);
            }
        }
        self.pending = survivors;

        for pending in promotions {
            if self.modes.len() >= self.cfg.max_modes {
                break;
            }
            let id = self.next_id;
            self.next_id += 1;
            self.modes.push(TrackedMode {
                id,
                frequency_hz: pending.frequency_hz,
                phase: pending.phase,
                frequency_slope: 0.0,
                strength: pending.strength,
                support_by_channel: pending.support_by_channel,
                age: pending.consecutive_hits as u64,
                missing_streak: 0,
                frequency_history: singleton_deque(pending.frequency_hz),
                phase_error_history: singleton_deque(0.0),
            });
        }

        // New unmatched candidates begin a tentative track but remain private
        // until they satisfy `birth_persistence`.
        for (index, candidate) in candidates.into_iter().enumerate() {
            if candidate_used[index] {
                continue;
            }
            if self.modes.len() + self.pending.len() >= self.cfg.max_modes * 2 {
                break;
            }
            let pending = PendingMode {
                frequency_hz: candidate.frequency_hz,
                phase: candidate.phase,
                strength: candidate.strength,
                support_by_channel: candidate.support_by_channel,
                consecutive_hits: 1,
            };
            if self.cfg.birth_persistence <= 1 && self.modes.len() < self.cfg.max_modes {
                let id = self.next_id;
                self.next_id += 1;
                self.modes.push(TrackedMode {
                    id,
                    frequency_hz: pending.frequency_hz,
                    phase: pending.phase,
                    frequency_slope: 0.0,
                    strength: pending.strength,
                    support_by_channel: pending.support_by_channel,
                    age: 1,
                    missing_streak: 0,
                    frequency_history: singleton_deque(pending.frequency_hz),
                    phase_error_history: singleton_deque(0.0),
                });
            } else {
                self.pending.push(pending);
            }
        }
    }

    fn merge_near_duplicates(&mut self) {
        self.modes.sort_by_key(|mode| mode.id);
        let mut i = 0;
        while i < self.modes.len() {
            let mut j = i + 1;
            while j < self.modes.len() {
                let near_frequency = log_frequency_distance(
                    self.modes[i].frequency_hz,
                    self.modes[j].frequency_hz,
                ) <= self.cfg.merge_log_freq_tolerance;
                let near_phase = circular_distance(
                    self.modes[i].phase,
                    self.modes[j].phase,
                ) <= self.cfg.merge_phase_tolerance_rad;

                if near_frequency && near_phase {
                    let other = self.modes.remove(j);
                    merge_mode_into(&mut self.modes[i], other);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    fn update_relations(&mut self) {
        let mut latest = Vec::new();
        let mut seen_keys = HashSet::new();

        for i in 0..self.modes.len() {
            for j in (i + 1)..self.modes.len() {
                let a = &self.modes[i];
                let b = &self.modes[j];
                if a.frequency_hz <= 0.0 || b.frequency_hz <= 0.0 {
                    continue;
                }

                for m in 1..=self.cfg.max_numer {
                    for n in 1..=self.cfg.max_denom {
                        if gcd(m, n) != 1 {
                            continue;
                        }
                        let lhs = m as f64 * a.frequency_hz;
                        let rhs = n as f64 * b.frequency_hz;
                        let residual = (lhs - rhs).abs() / rhs.abs().max(1.0e-12);
                        if residual > self.cfg.rational_ratio_tolerance {
                            continue;
                        }

                        let key = RelationKey {
                            i_id: a.id,
                            j_id: b.id,
                            m,
                            n,
                        };
                        seen_keys.insert(key);
                        let generalized = wrap_phase(
                            m as f64 * a.phase - n as f64 * b.phase,
                        );
                        let history = self
                            .relation_phase_history
                            .entry(key)
                            .or_default();
                        push_bounded(
                            history,
                            generalized,
                            self.cfg.relation_history_len,
                        );
                        let stability = circular_variance(history.iter().copied());
                        latest.push(CycleRelation {
                            i_id: a.id,
                            j_id: b.id,
                            m,
                            n,
                            freq_residual: residual,
                            generalized_phase: generalized,
                            phase_stability: stability,
                        });
                    }
                }
            }
        }

        self.relation_phase_history
            .retain(|key, _| seen_keys.contains(key));
        latest.sort_by(|a, b| {
            a.i_id
                .cmp(&b.i_id)
                .then_with(|| a.j_id.cmp(&b.j_id))
                .then_with(|| a.m.cmp(&b.m))
                .then_with(|| a.n.cmp(&b.n))
        });

        if self.relation_snapshots.len() >= self.cfg.relation_history_len {
            self.relation_snapshots.pop_front();
        }
        self.relation_snapshots.push_back(latest);
    }
}

fn validate_observation(obs: &CycleObservation) -> Result<(), CycleBankError> {
    if obs.channels.is_empty() {
        return Err(CycleBankError::NoChannels);
    }
    if !obs.dt_seconds.is_finite() || obs.dt_seconds <= 0.0 {
        return Err(CycleBankError::InvalidDt(obs.dt_seconds));
    }
    for channel in &obs.channels {
        if !channel.value.is_finite() {
            return Err(CycleBankError::NonFiniteEvidence {
                channel: channel.name.clone(),
                value: channel.value,
            });
        }
    }
    Ok(())
}

fn build_grid(cfg: &CycleBankConfig, dt_seconds: f64) -> Vec<ScaleState> {
    let step = 2.0_f64.powf(1.0 / cfg.scales_per_octave as f64);
    let mut f = cfg.f_min_hz;
    let mut grid = Vec::new();
    while f <= cfg.f_max_hz * (1.0 + 1.0e-12) {
        let rho = (-dt_seconds * f / cfg.q_cycles).exp();
        grid.push(ScaleState {
            f_center: f,
            omega_center: TAU * f,
            demod_phase: 0.0,
            lowpass: [Complex64::new(0.0, 0.0); ANALYTIC_STAGES],
            prev_output: Complex64::new(0.0, 0.0),
            rho,
            has_previous_output: false,
        });
        f *= step;
    }
    grid
}

/// Advance every causal analytic scale by one scalar observation.
fn step_grid(grid: &mut [ScaleState], x: f64, dt_seconds: f64, q_cycles: f64) {
    for scale in grid {
        scale.rho = (-dt_seconds * scale.f_center / q_cycles).exp();
        scale.demod_phase = positive_phase(
            scale.demod_phase + scale.omega_center * dt_seconds,
        );

        scale.prev_output = scale.output();
        let oscillator = Complex64::from_polar(1.0, -scale.demod_phase);
        let mut stage_input = oscillator * x;
        for stage in &mut scale.lowpass {
            *stage = scale.rho * *stage + (1.0 - scale.rho) * stage_input;
            stage_input = *stage;
        }
        scale.has_previous_output = true;
    }
}

/// Convert one scale's internal baseband state into a continuous-frequency,
/// phase-calibrated analytic measurement.
fn scale_measurement(
    scale: &ScaleState,
    dt_seconds: f64,
    cfg: &CycleBankConfig,
) -> Option<ScaleMeasurement> {
    if !scale.has_previous_output {
        return None;
    }
    let current = scale.output();
    let previous = scale.prev_output;
    if current.norm() < cfg.weak_threshold || previous.norm() < cfg.weak_threshold {
        return None;
    }

    let baseband_dphi = (current * previous.conj()).arg();
    let frequency_hz = scale.f_center + baseband_dphi / (TAU * dt_seconds);
    if !frequency_hz.is_finite()
        || frequency_hz < cfg.f_min_hz
        || frequency_hz > cfg.f_max_hz
    {
        return None;
    }

    // Exact discrete transfer function of one baseband low-pass section for
    // detuning delta = (omega_hat - omega_center) * dt:
    //
    // H(delta) = (1-rho) / (1 - rho*exp(-i*delta))
    //
    // The cascade contributes H^ANALYTIC_STAGES.  Removing its phase yields
    // the estimated input analytic phase after remodulation.
    let delta = TAU * (frequency_hz - scale.f_center) * dt_seconds;
    let one_stage = Complex64::new(1.0 - scale.rho, 0.0)
        / (Complex64::new(1.0, 0.0)
            - scale.rho * Complex64::from_polar(1.0, -delta));
    let mut transfer = Complex64::new(1.0, 0.0);
    for _ in 0..ANALYTIC_STAGES {
        transfer *= one_stage;
    }

    let phase = wrap_phase(current.arg() + scale.demod_phase - transfer.arg());

    // A real sinusoid contributes half its amplitude to the positive-frequency
    // analytic branch.  The factor of two makes strength approximately input
    // amplitude at an on-ridge scale.  It remains a direct-support diagnostic,
    // not a normalized probability.
    let strength = 2.0 * current.norm();

    Some(ScaleMeasurement {
        frequency_hz,
        phase,
        strength,
    })
}

fn measurement_is_local_max(measurements: &[Option<ScaleMeasurement>], i: usize) -> bool {
    let Some(current) = measurements[i].as_ref() else {
        return false;
    };
    let left = i
        .checked_sub(1)
        .and_then(|index| measurements[index].as_ref())
        .map(|m| m.strength);
    let right = measurements
        .get(i + 1)
        .and_then(Option::as_ref)
        .map(|m| m.strength);

    left.is_none_or(|value| current.strength >= value)
        && right.is_none_or(|value| current.strength >= value)
        && (left.is_none_or(|value| (current.strength - value).abs() > 1.0e-12)
            || right.is_none_or(|value| (current.strength - value).abs() > 1.0e-12))
}

fn cluster_channel_ridges(
    mut ridges: Vec<ChannelRidge>,
    n_channels: usize,
    scales_per_octave: usize,
    max_modes: usize,
) -> Vec<RidgeCandidate> {
    ridges.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.channel_index.cmp(&b.channel_index))
    });

    #[derive(Clone)]
    struct Cluster {
        frequency_weighted_sum: f64,
        phase_vector: Complex64,
        total_weight: f64,
        strength_sum: f64,
        supporters: usize,
        support_by_channel: Vec<f64>,
    }

    impl Cluster {
        fn frequency_hz(&self) -> f64 {
            self.frequency_weighted_sum / self.total_weight.max(1.0e-12)
        }
    }

    // Ridges from different channels that represent the same physical mode
    // should generally fall within about one numerical scale step.  This is
    // aggregation tolerance only; it does not quantize the final frequency.
    let cluster_tolerance = (2.0_f64.ln() / scales_per_octave.max(1) as f64) * 1.25;
    let mut clusters: Vec<Cluster> = Vec::new();

    for ridge in ridges {
        let mut best: Option<(usize, f64)> = None;
        for (index, cluster) in clusters.iter().enumerate() {
            if cluster.support_by_channel[ridge.channel_index] > 0.0 {
                continue;
            }
            let distance = log_frequency_distance(
                cluster.frequency_hz(),
                ridge.frequency_hz,
            );
            if distance <= cluster_tolerance
                && best.is_none_or(|(_, best_distance)| distance < best_distance)
            {
                best = Some((index, distance));
            }
        }

        let weight = ridge.strength.max(1.0e-12);
        if let Some((index, _)) = best {
            let cluster = &mut clusters[index];
            cluster.frequency_weighted_sum += ridge.frequency_hz * weight;
            cluster.phase_vector += Complex64::from_polar(weight, ridge.phase);
            cluster.total_weight += weight;
            cluster.strength_sum += ridge.strength;
            cluster.supporters += 1;
            cluster.support_by_channel[ridge.channel_index] = ridge.strength;
        } else {
            let mut support_by_channel = vec![0.0; n_channels];
            support_by_channel[ridge.channel_index] = ridge.strength;
            clusters.push(Cluster {
                frequency_weighted_sum: ridge.frequency_hz * weight,
                phase_vector: Complex64::from_polar(weight, ridge.phase),
                total_weight: weight,
                strength_sum: ridge.strength,
                supporters: 1,
                support_by_channel,
            });
        }
    }

    let mut candidates: Vec<RidgeCandidate> = clusters
        .into_iter()
        .map(|cluster| RidgeCandidate {
            frequency_hz: cluster.frequency_hz(),
            phase: cluster.phase_vector.arg(),
            strength: cluster.strength_sum / cluster.supporters.max(1) as f64,
            channel_support: cluster.supporters as f64 / n_channels.max(1) as f64,
            support_by_channel: cluster.support_by_channel,
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.frequency_hz
                    .partial_cmp(&b.frequency_hz)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    candidates.truncate(max_modes.saturating_mul(2).max(max_modes));
    candidates
}

fn association_cost(
    mode: &TrackedMode,
    predicted_phase: f64,
    candidate: &RidgeCandidate,
    cfg: &CycleBankConfig,
) -> Option<f64> {
    let frequency_distance = log_frequency_distance(mode.frequency_hz, candidate.frequency_hz);
    if frequency_distance > cfg.association_log_freq_tolerance {
        return None;
    }
    let phase_distance = circular_distance(predicted_phase, candidate.phase);
    if phase_distance > cfg.association_phase_tolerance_rad {
        return None;
    }

    let channel_distance = 1.0
        - cosine_similarity(&mode.support_by_channel, &candidate.support_by_channel);
    // Prefer candidates whose cross-channel support is corroborated: a ridge
    // seen by more channels is a more trustworthy association target.
    let support_bonus = 1.0 - candidate.channel_support.clamp(0.0, 1.0);
    Some(
        0.50 * frequency_distance / cfg.association_log_freq_tolerance
            + 0.32 * phase_distance / cfg.association_phase_tolerance_rad
            + 0.08 * channel_distance.clamp(0.0, 1.0)
            + 0.10 * support_bonus,
    )
}

fn update_matched_mode(
    mode: &mut TrackedMode,
    predicted_phase: f64,
    candidate: &RidgeCandidate,
    dt_seconds: f64,
    cfg: &CycleBankConfig,
) {
    let phase_error = wrap_phase(candidate.phase - predicted_phase);
    mode.phase = wrap_phase(predicted_phase + cfg.phase_correction_gain * phase_error);

    let old_frequency = mode.frequency_hz;
    mode.frequency_hz = lerp(
        mode.frequency_hz,
        candidate.frequency_hz,
        cfg.frequency_smoothing,
    );
    let instantaneous_slope = (mode.frequency_hz - old_frequency) / dt_seconds;
    mode.frequency_slope = lerp(
        mode.frequency_slope,
        instantaneous_slope,
        cfg.slope_smoothing,
    );

    mode.strength = lerp(
        mode.strength,
        candidate.strength,
        cfg.strength_smoothing,
    );
    blend_support(
        &mut mode.support_by_channel,
        &candidate.support_by_channel,
        cfg.strength_smoothing,
    );
    mode.age = mode.age.saturating_add(1);
    mode.missing_streak = 0;
    push_bounded(
        &mut mode.frequency_history,
        mode.frequency_hz,
        MODE_HISTORY_LEN,
    );
    push_bounded(
        &mut mode.phase_error_history,
        phase_error,
        MODE_HISTORY_LEN,
    );
}

fn merge_mode_into(keep: &mut TrackedMode, other: TrackedMode) {
    let keep_weight = keep.strength.max(1.0e-12);
    let other_weight = other.strength.max(1.0e-12);
    let total = keep_weight + other_weight;

    keep.frequency_hz =
        (keep.frequency_hz * keep_weight + other.frequency_hz * other_weight) / total;
    keep.phase = circular_weighted_mean(
        keep.phase,
        keep_weight,
        other.phase,
        other_weight,
    );
    keep.frequency_slope =
        (keep.frequency_slope * keep_weight + other.frequency_slope * other_weight) / total;
    keep.strength = keep.strength.max(other.strength);
    keep.age = keep.age.max(other.age);
    keep.missing_streak = keep.missing_streak.min(other.missing_streak);

    if keep.support_by_channel.len() == other.support_by_channel.len() {
        for (left, right) in keep
            .support_by_channel
            .iter_mut()
            .zip(other.support_by_channel.iter())
        {
            *left = (*left).max(*right);
        }
    }

    for value in other.frequency_history {
        push_bounded(&mut keep.frequency_history, value, MODE_HISTORY_LEN);
    }
    for value in other.phase_error_history {
        push_bounded(&mut keep.phase_error_history, value, MODE_HISTORY_LEN);
    }
}

fn blend_support(target: &mut [f64], source: &[f64], alpha: f64) {
    for (target_value, source_value) in target.iter_mut().zip(source.iter()) {
        *target_value = lerp(*target_value, *source_value, alpha);
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let a_norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let b_norm = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if a_norm <= 1.0e-12 || b_norm <= 1.0e-12 {
        0.0
    } else {
        (dot / (a_norm * b_norm)).clamp(0.0, 1.0)
    }
}

fn log_frequency_distance(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return f64::INFINITY;
    }
    (a.ln() - b.ln()).abs()
}

fn circular_distance(a: f64, b: f64) -> f64 {
    wrap_phase(a - b).abs()
}

fn circular_weighted_mean(a: f64, a_weight: f64, b: f64, b_weight: f64) -> f64 {
    let vector = Complex64::from_polar(a_weight, a)
        + Complex64::from_polar(b_weight, b);
    if vector.norm() <= 1.0e-12 {
        a
    } else {
        vector.arg()
    }
}

fn circular_variance<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = Complex64::new(0.0, 0.0);
    let mut count = 0usize;
    for value in values {
        sum += Complex64::from_polar(1.0, value);
        count += 1;
    }
    if count == 0 {
        return 1.0;
    }
    (1.0 - sum.norm() / count as f64).clamp(0.0, 1.0)
}

fn circular_rms<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for value in values {
        let wrapped = wrap_phase(value);
        sum_sq += wrapped * wrapped;
        count += 1;
    }
    if count == 0 {
        PI
    } else {
        (sum_sq / count as f64).sqrt()
    }
}

fn variance<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let values: Vec<f64> = values.into_iter().collect();
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

fn singleton_deque<T>(value: T) -> VecDeque<T> {
    let mut queue = VecDeque::new();
    queue.push_back(value);
    queue
}

fn push_bounded<T>(queue: &mut VecDeque<T>, value: T, capacity: usize) {
    if queue.len() >= capacity.max(1) {
        queue.pop_front();
    }
    queue.push_back(value);
}

fn lerp(a: f64, b: f64, alpha: f64) -> f64 {
    a + alpha * (b - a)
}

fn wrap_phase(value: f64) -> f64 {
    let mut wrapped = value.rem_euclid(TAU);
    if wrapped > PI {
        wrapped -= TAU;
    }
    wrapped
}

fn positive_phase(value: f64) -> f64 {
    value.rem_euclid(TAU)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

#[derive(Debug, Clone, PartialEq)]
pub enum CycleBankError {
    InvalidConfig(String),
    NoChannels,
    InvalidDt(f64),
    InvalidChannelName(String),
    NonFiniteEvidence { channel: String, value: f64 },
    ChannelSchemaMismatch { expected: Vec<String>, got: Vec<String> },
    NonMonotonicSampleIndex { previous: u64, current: u64 },
}

impl std::fmt::Display for CycleBankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid CycleBankConfig: {message}"),
            Self::NoChannels => write!(f, "observation has no evidence channels"),
            Self::InvalidDt(dt) => write!(f, "dt_seconds must be finite and > 0, got {dt}"),
            Self::InvalidChannelName(message) => write!(f, "invalid evidence channel: {message}"),
            Self::NonFiniteEvidence { channel, value } => {
                write!(f, "channel {channel} supplied non-finite evidence value {value}")
            }
            Self::ChannelSchemaMismatch { expected, got } => write!(
                f,
                "channel schema changed mid-epoch: expected {expected:?}, got {got:?}"
            ),
            Self::NonMonotonicSampleIndex { previous, current } => write!(
                f,
                "sample_index must increase within an epoch: previous={previous}, current={current}"
            ),
        }
    }
}

impl std::error::Error for CycleBankError {}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1024.0 / 48_000.0;

    fn observation(sample_index: u64, value: f64) -> CycleObservation {
        CycleObservation {
            sample_index,
            dt_seconds: DT,
            stream_epoch: 1,
            channels: vec![CycleEvidenceChannel::new("synthetic", value)],
        }
    }

    #[test]
    fn phase_prediction_returns_after_one_period() {
        let mode = CycleMode {
            id: 1,
            frequency_hz: 2.1667,
            phase: 0.7,
            strength: 1.0,
            confidence: 1.0,
            channel_support: 1.0,
            age: 10,
            missing_observations: 0,
            frequency_slope: 0.0,
            frequency_uncertainty: 0.0,
        };
        let predicted = mode.phase_at(1.0 / mode.frequency_hz);
        assert!(circular_distance(predicted, mode.phase) < 1.0e-10);
    }

    #[test]
    fn channel_schema_cannot_silently_reorder_state() {
        let mut bank = CycleBank::new(CycleBankConfig::default());
        let first = CycleObservation {
            sample_index: 1024,
            dt_seconds: DT,
            stream_epoch: 1,
            channels: vec![
                CycleEvidenceChannel::new("onset", 0.2),
                CycleEvidenceChannel::new("low", 0.1),
            ],
        };
        bank.observe(&first).unwrap();

        let swapped = CycleObservation {
            sample_index: 2048,
            dt_seconds: DT,
            stream_epoch: 1,
            channels: vec![
                CycleEvidenceChannel::new("low", 0.1),
                CycleEvidenceChannel::new("onset", 0.2),
            ],
        };
        assert!(matches!(
            bank.observe(&swapped),
            Err(CycleBankError::ChannelSchemaMismatch { .. })
        ));
    }

    #[test]
    fn configured_q_is_the_runtime_q_authority() {
        let mut cfg = CycleBankConfig::default();
        cfg.q_cycles = 7.0;
        let mut grid = build_grid(&cfg, DT);
        let expected = (-DT * grid[0].f_center / cfg.q_cycles).exp();
        step_grid(&mut grid, 1.0, DT, cfg.q_cycles);
        assert!((grid[0].rho - expected).abs() < 1.0e-14);
    }

    #[test]
    fn off_grid_frequency_is_recovered_continuously() {
        let mut cfg = CycleBankConfig::default();
        cfg.f_min_hz = 0.5;
        cfg.f_max_hz = 4.0;
        cfg.birth_persistence = 2;
        cfg.max_modes = 6;
        let mut bank = CycleBank::new(cfg.clone());

        let target_hz = 2.1667;
        let input_phase = 0.7;
        let steps = (18.0 / DT) as usize;
        for n in 1..=steps {
            let t = n as f64 * DT;
            let value = (TAU * target_hz * t + input_phase).cos();
            let obs = observation(n as u64 * 1024, value);
            bank.observe(&obs).unwrap();
        }

        let modes = bank.modes();
        let closest = modes
            .iter()
            .min_by(|a, b| {
                (a.frequency_hz - target_hz)
                    .abs()
                    .partial_cmp(&(b.frequency_hz - target_hz).abs())
                    .unwrap()
            })
            .expect("expected at least one observed ridge");

        assert!(
            (closest.frequency_hz - target_hz).abs() < 0.02,
            "target={target_hz}, recovered={} modes={modes:?}",
            closest.frequency_hz
        );

        // Prove this is not merely the nearest numerical scale center.
        let numerical_centers: Vec<f64> = build_grid(&cfg, DT)
            .into_iter()
            .map(|scale| scale.f_center)
            .collect();
        let nearest_center = numerical_centers
            .iter()
            .copied()
            .min_by(|a, b| {
                (a - target_hz)
                    .abs()
                    .partial_cmp(&(b - target_hz).abs())
                    .unwrap()
            })
            .unwrap();
        assert!(
            (closest.frequency_hz - target_hz).abs()
                < (nearest_center - target_hz).abs()
        );
    }

    #[test]
    fn epoch_change_clears_modes_and_allows_new_schema() {
        let mut cfg = CycleBankConfig::default();
        cfg.f_min_hz = 1.0;
        cfg.f_max_hz = 3.0;
        cfg.birth_persistence = 1;
        let mut bank = CycleBank::new(cfg);

        for n in 1..=300 {
            let t = n as f64 * DT;
            bank.observe(&CycleObservation {
                sample_index: n as u64 * 1024,
                dt_seconds: DT,
                stream_epoch: 1,
                channels: vec![CycleEvidenceChannel::new(
                    "old",
                    (TAU * 2.0 * t).cos(),
                )],
            })
            .unwrap();
        }
        assert!(bank.num_modes() > 0);

        bank.observe(&CycleObservation {
            sample_index: 1024,
            dt_seconds: DT,
            stream_epoch: 2,
            channels: vec![CycleEvidenceChannel::new("new", 0.0)],
        })
        .unwrap();
        assert_eq!(bank.num_modes(), 0);
        assert_eq!(bank.channel_names, vec!["new".to_string()]);
    }

    #[test]
    fn relation_stability_uses_time_history() {
        let cfg = CycleBankConfig::default();
        let mut bank = CycleBank::new(cfg);
        bank.modes = vec![
            TrackedMode {
                id: 1,
                frequency_hz: 2.0,
                phase: 0.0,
                frequency_slope: 0.0,
                strength: 1.0,
                support_by_channel: vec![1.0],
                age: 10,
                missing_streak: 0,
                frequency_history: singleton_deque(2.0),
                phase_error_history: singleton_deque(0.0),
            },
            TrackedMode {
                id: 2,
                frequency_hz: 1.0,
                phase: 0.0,
                frequency_slope: 0.0,
                strength: 1.0,
                support_by_channel: vec![1.0],
                age: 10,
                missing_streak: 0,
                frequency_history: singleton_deque(1.0),
                phase_error_history: singleton_deque(0.0),
            },
        ];

        for k in 0..8 {
            bank.modes[0].phase = wrap_phase(0.4 * k as f64);
            bank.modes[1].phase = wrap_phase(0.2 * k as f64);
            bank.update_relations();
        }

        let relation = bank
            .latest_relations()
            .into_iter()
            .find(|r| r.m == 1 && r.n == 2)
            .expect("expected 1:2 relation");
        assert!(relation.phase_stability < 1.0e-9);
    }
}
