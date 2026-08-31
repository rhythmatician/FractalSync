//! Orbit synthesizer for Julia set parameter generation
//! 
//! WebAssembly bindings to runtime_core for browser use.

use wasm_bindgen::prelude::*;
use serde::Serialize;
use num_complex::Complex64 as RustComplex;
use runtime_core::controller::{
    OrbitState as RustOrbitState,
    PlayerState as RustPlayerState,
    OrbitController as RustOrbitController,
    ResidualParams as RustResidualParams,
    synthesize as rust_synthesize,
    DEFAULT_K_RESIDUALS,
    DEFAULT_RESIDUAL_CAP,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    CONTROLLER_VERSION,
    DEFAULT_BASE_OMEGA,
    DEFAULT_ORBIT_SEED,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    WINDOW_FRAMES,
};
use runtime_core::features::{FEATURE_VERSION, NORM_EPS};
use runtime_core::features::FeatureExtractor as RustFeatureExtractor;
use runtime_core::timebase::{
    cycle_observation_from_tick, AnalysisTimebase as RustAnalysisTimebase,
    AnalysisTick as RustAnalysisTick, ResetReason as RustResetReason,
    ANALYSIS_PIPELINE_VERSION,
};
use runtime_core::cycle_bank::{
    CycleBank as RustCycleBank, CycleBankConfig as RustCycleBankConfig,
    CycleEvidenceChannel as RustCycleEvidenceChannel,
    CycleObservation as RustCycleObservation, CYCLE_BANK_VERSION,
};
use serde::Deserialize;

/// Shared constants exposed to JavaScript
#[wasm_bindgen]
pub fn constants() -> JsValue {
    #[derive(Serialize)]
    struct Constants {
        sample_rate: usize,
        hop_length: usize,
        n_fft: usize,
        window_frames: usize,
        default_k_residuals: usize,
        default_residual_cap: f64,
        default_residual_omega_scale: f64,
        default_base_omega: f64,
        default_orbit_seed: u64,
        controller_version: String,
        feature_version: String,
        analysis_pipeline_version: String,
        cycle_bank_version: String,
        norm_eps: f64,
    }

    let c = Constants {
        sample_rate: SAMPLE_RATE,
        hop_length: HOP_LENGTH,
        n_fft: N_FFT,
        window_frames: WINDOW_FRAMES,
        default_k_residuals: DEFAULT_K_RESIDUALS,
        default_residual_cap: DEFAULT_RESIDUAL_CAP,
        default_residual_omega_scale: DEFAULT_RESIDUAL_OMEGA_SCALE,
        default_base_omega: DEFAULT_BASE_OMEGA,
        controller_version: CONTROLLER_VERSION.to_string(),
        feature_version: FEATURE_VERSION.to_string(),
        analysis_pipeline_version: ANALYSIS_PIPELINE_VERSION.to_string(),
        cycle_bank_version: CYCLE_BANK_VERSION.to_string(),
        norm_eps: NORM_EPS,
        default_orbit_seed: DEFAULT_ORBIT_SEED,
    };

    serde_wasm_bindgen::to_value(&c).unwrap_or_else(|_| JsValue::NULL)
}

/// Audio feature extractor — the SAME Rust implementation the trainer uses.
///
/// The browser feeds a rolling window of PCM samples (from
/// AnalyserNode.getFloatTimeDomainData) and receives feature windows laid
/// out identically to training inputs. This eliminates the entire class of
/// browser-vs-trainer extraction drift (FFT size, smoothing, dB conversion,
/// per-file normalization, window layout) by construction: there is only
/// one implementation, executed in two places.
#[wasm_bindgen]
pub struct FeatureExtractor {
    inner: RustFeatureExtractor,
}

#[wasm_bindgen]
impl FeatureExtractor {
    /// Create an extractor with the shared runtime defaults (48 kHz,
    /// hop 1024, n_fft 4096). Callers must resample browser audio to the
    /// runtime sample rate before feeding PCM here.
    #[wasm_bindgen(constructor)]
    pub fn new() -> FeatureExtractor {
        FeatureExtractor {
            inner: RustFeatureExtractor::default(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn num_features_per_frame(&self) -> usize {
        self.inner.num_features_per_frame()
    }

    /// Extract the MOST RECENT flattened feature window from `audio`
    /// (frame-major).
    ///
    /// `audio` is the rolling PCM history in chronological order; the
    /// returned window covers the latest `window_frames` STFT frames,
    /// matching what live inference needs. Short input is padded by
    /// repeating the last frame, matching training behavior for short
    /// files.
    #[wasm_bindgen]
    pub fn extract_window(&self, audio: Vec<f32>, window_frames: usize) -> Vec<f64> {
        let windows = self.inner.extract_windowed_features(&audio, window_frames);
        windows.into_iter().last().unwrap_or_default()
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Authoritative sample-clock audio timebase (issue #91).
///
/// The browser's AudioWorklet transport feeds non-overlapping PCM blocks
/// here; the Rust timebase validates monotonicity, resamples statefully to
/// the canonical 48 kHz timeline, schedules exact 1024-sample hops, and runs
/// the canonical FeatureExtractor — all in Rust, so there is no TypeScript
/// mirror of the timing/scheduling math (ADR 0001).
#[wasm_bindgen]
pub struct AnalysisTimebase {
    inner: RustAnalysisTimebase,
}

// ---------------------------------------------------------------------------
// TypeScript shape of the canonical analysis-tick and timebase-diagnostics
// records. Emitted as a custom section so wasm-pack includes them verbatim
// in the generated ``orbit_synth_wasm.d.ts``. Frontend consumers import
// ``AnalysisTick`` and ``TimebaseDiagnostics`` from ``orbit-synth-wasm`` and
// do NOT redeclare them — Rust is the single source of truth for the wire
// format (ADR 0001, issue #93 strict-version review).
//
// Field names match the camelCase wire format produced by
// ``#[serde(rename_all = "camelCase")]` on the Rust structs that
// ``ingest``/``flush``/``diagnostics`` serialize. Python mirrors the same
// keys (see ``runtime_core::pybindings::tick_to_pydict``) so the trainer
// and the browser see an identical record shape — cross-surface parity.
// ---------------------------------------------------------------------------
#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
/** Canonical analysis tick — the seam CycleBank will consume (issue #91). */
export interface AnalysisTick {
    features: number[];
    sampleIndex: number;
    timeSeconds: number;
    dtSeconds: number;
    streamEpoch: number;
}

/** Diagnostic snapshot for manual verification of the canonical clock. */
export interface TimebaseDiagnostics {
    sourceSampleRate: number;
    sourceFramesIngested: number;
    canonicalSampleIndex: number;
    analysisHopCount: number;
    timeSeconds: number;
    streamEpoch: number;
    detectedGaps: number;
    detectedOverlaps: number;
    lastSourceStartFrame: number;
    lastSourceEndFrame: number;
}
"#;

/// A single emitted analysis tick, serialized to JS.
///
/// Wire format is **camelCase** to match the generated TypeScript shape
/// emitted by the ``ts_types`` custom section below. The matching
/// Python serializer (see ``runtime_core::pybindings::tick_to_pydict``)
/// uses the same keys so a tick arriving via either binding is keyed
/// identically across surfaces (cross-surface parity, issue #93).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisTickJs {
    features: Vec<f64>,
    sample_index: u64,
    time_seconds: f64,
    dt_seconds: f64,
    stream_epoch: u64,
}

#[wasm_bindgen]
impl AnalysisTimebase {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AnalysisTimebase {
        AnalysisTimebase {
            inner: RustAnalysisTimebase::new(),
        }
    }

    /// Ingest one non-overlapping PCM block. Returns a JS array of ticks
    /// (possibly empty). Throws on overlap / mid-stream rate change.
    ///
    /// Type note: the generated .d.ts types this as `any` because the
    /// function returns a `JsValue` (it serializes via
    /// ``serde_wasm_bindgen``). The TS adapter in
    /// ``frontend/src/lib/analysisTimebase.ts`` re-types this signature
    /// as ``AnalysisTick[]`` and the ``AnalysisTick`` interface itself
    /// is provided by the ``TS_TYPES`` custom section above — so the
    /// Rust source remains the single authority for the wire shape.
    #[wasm_bindgen]
    pub fn ingest(
        &mut self,
        samples: Vec<f32>,
        source_sample_rate: usize,
        source_start_frame: u64,
    ) -> Result<JsValue, JsValue> {
        let ticks = self
            .inner
            .ingest(&samples, source_sample_rate, source_start_frame)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let js: Vec<AnalysisTickJs> = ticks
            .into_iter()
            .map(|t| AnalysisTickJs {
                features: t.features,
                sample_index: t.sample_index,
                time_seconds: t.time_seconds,
                dt_seconds: t.dt_seconds,
                stream_epoch: t.stream_epoch,
            })
            .collect();
        serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Flush end-of-stream (recovers the deferred final sample/tick).
    /// See ``ingest`` for the typing note about the .d.ts return type.
    #[wasm_bindgen]
    pub fn flush(&mut self) -> JsValue {
        let js: Vec<AnalysisTickJs> = self
            .inner
            .flush()
            .into_iter()
            .map(|t| AnalysisTickJs {
                features: t.features,
                sample_index: t.sample_index,
                time_seconds: t.time_seconds,
                dt_seconds: t.dt_seconds,
                stream_epoch: t.stream_epoch,
            })
            .collect();
        serde_wasm_bindgen::to_value(&js).unwrap_or(JsValue::NULL)
    }

    /// Declare a stream discontinuity. `reason` is informational; the epoch
    /// always bumps and the schedule resets.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.inner.reset(RustResetReason::SourceReplacement);
    }

    /// Diagnostic snapshot for verifying the clock manually.
    /// See ``ingest`` for the typing note about the .d.ts return type.
    #[wasm_bindgen]
    pub fn diagnostics(&self) -> JsValue {
        let d = self.inner.diagnostics();
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Diag {
            source_sample_rate: usize,
            source_frames_ingested: u64,
            canonical_sample_index: u64,
            analysis_hop_count: u64,
            time_seconds: f64,
            stream_epoch: u64,
            detected_gaps: u64,
            detected_overlaps: u64,
            last_source_start_frame: u64,
            last_source_end_frame: u64,
        }
        serde_wasm_bindgen::to_value(&Diag {
            source_sample_rate: d.source_sample_rate,
            source_frames_ingested: d.source_frames_ingested,
            canonical_sample_index: d.canonical_sample_index,
            analysis_hop_count: d.analysis_hop_count,
            time_seconds: d.time_seconds,
            stream_epoch: d.stream_epoch,
            detected_gaps: d.detected_gaps,
            detected_overlaps: d.detected_overlaps,
            last_source_start_frame: d.last_source_start_frame,
            last_source_end_frame: d.last_source_end_frame,
        })
        .unwrap_or(JsValue::NULL)
    }
}

impl Default for AnalysisTimebase {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for complex number to/from JavaScript
#[wasm_bindgen]
#[derive(Clone)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl From<RustComplex> for Complex {
    fn from(c: RustComplex) -> Self {
        Self {
            real: c.re,
            imag: c.im,
        }
    }
}

/// Orbit state wrapper for WASM
#[wasm_bindgen]
#[derive(Clone)]
pub struct OrbitState {
    inner: RustOrbitState,
}

#[wasm_bindgen]
impl OrbitState {
    /// Create new orbit state with optional seed
    #[wasm_bindgen(constructor)]
    pub fn new(
        lobe: u32,
        sub_lobe: u32,
        theta: f64,
        omega: f64,
        s: f64,
        alpha: f64,
        k_residuals: usize,
        residual_omega_scale: f64,
        seed: Option<u64>,
    ) -> OrbitState {
        let inner = match seed {
            Some(seed_val) => RustOrbitState::new_with_seed(
                lobe,
                sub_lobe,
                theta,
                omega,
                s,
                alpha,
                k_residuals,
                residual_omega_scale,
                seed_val,
            ),
            None => RustOrbitState::new(
                lobe,
                sub_lobe,
                theta,
                omega,
                s,
                alpha,
                k_residuals,
                residual_omega_scale,
            ),
        };

        OrbitState { inner }
    }

    /// Create deterministic orbit with default parameters and seed
    #[wasm_bindgen(js_name = "newDefault")]
    pub fn new_default(seed: u64) -> OrbitState {
        let inner = RustOrbitState::new_with_seed(
            1,
            0,
            0.0,
            DEFAULT_BASE_OMEGA,
            1.02,
            0.3,
            DEFAULT_K_RESIDUALS,
            DEFAULT_RESIDUAL_OMEGA_SCALE,
            seed,
        );
        OrbitState { inner }
    }

    /// Get lobe
    #[wasm_bindgen(getter)]
    pub fn lobe(&self) -> u32 {
        self.inner.lobe
    }

    /// Set lobe
    #[wasm_bindgen(setter)]
    pub fn set_lobe(&mut self, lobe: u32) {
        self.inner.lobe = lobe;
    }

    /// Set s (radius scaling)
    #[wasm_bindgen(setter)]
    pub fn set_s(&mut self, s: f64) {
        self.inner.s = s;
    }

    /// Set alpha (residual amplitude)
    #[wasm_bindgen(setter)]
    pub fn set_alpha(&mut self, alpha: f64) {
        self.inner.alpha = alpha;
    }

    /// Set omega (base angular velocity)
    #[wasm_bindgen(setter)]
    pub fn set_omega(&mut self, omega: f64) {
        self.inner.omega = omega;
    }

    /// Get sub_lobe
    #[wasm_bindgen(getter)]
    pub fn sub_lobe(&self) -> u32 {
        self.inner.sub_lobe
    }

    /// Set sub_lobe
    #[wasm_bindgen(setter)]
    pub fn set_sub_lobe(&mut self, sub_lobe: u32) {
        self.inner.sub_lobe = sub_lobe;
    }

    /// Get theta
    #[wasm_bindgen(getter)]
    pub fn theta(&self) -> f64 {
        self.inner.theta
    }

    /// Get s (radius scaling)
    #[wasm_bindgen(getter)]
    pub fn s(&self) -> f64 {
        self.inner.s
    }

    /// Get alpha (residual amplitude)
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// Advance state by dt seconds
    pub fn advance(&mut self, dt: f64) {
        self.inner.advance(dt);
    }
}

/// Player c-space integrator wrapper for WASM (issue #88, Q2).
///
/// Holds `c` as persistent state and moves it toward a model-driven target
/// point on the Mandelbrot boundary, biased along the Shore's contours via
/// the minimap. This replaces the closed-loop carrier for audio-driven
/// wandering.
#[wasm_bindgen]
#[derive(Clone)]
pub struct PlayerState {
    inner: RustPlayerState,
}

#[wasm_bindgen]
impl PlayerState {
    /// Create a PlayerState starting on the boundary at (s, alpha).
    #[wasm_bindgen(constructor)]
    pub fn new(lobe: u32, sub_lobe: u32, s: f64, alpha: f64) -> PlayerState {
        PlayerState {
            inner: RustPlayerState::new(lobe, sub_lobe, s, alpha),
        }
    }

    /// Current c (real part).
    #[wasm_bindgen(getter)]
    pub fn c_re(&self) -> f64 {
        self.inner.c.re
    }

    /// Current c (imaginary part).
    #[wasm_bindgen(getter)]
    pub fn c_im(&self) -> f64 {
        self.inner.c.im
    }

    /// Current c-space velocity magnitude (Momentum diagnostic).
    #[wasm_bindgen(getter)]
    pub fn speed(&self) -> f64 {
        self.inner.velocity.norm()
    }

    /// Set the mip level used for the contour step.
    #[wasm_bindgen(setter)]
    pub fn set_level(&mut self, level: usize) {
        self.inner.level = level;
    }

    /// Set the target shore-proximity distance the servo pulls toward.
    #[wasm_bindgen(setter)]
    pub fn set_d_star(&mut self, d_star: f64) {
        self.inner.d_star = d_star;
    }

    /// Set the maximum world-space step per frame.
    #[wasm_bindgen(setter)]
    pub fn set_max_step(&mut self, max_step: f64) {
        self.inner.max_step = max_step;
    }

    /// Set the audio energy in [0, 1] (loudness). Raises the servo's
    /// target shore-proximity: loud audio pulls c toward the Shore.
    #[wasm_bindgen(setter)]
    pub fn set_energy(&mut self, energy: f64) {
        self.inner.energy = energy.clamp(0.0, 1.0);
    }

    /// Apply model-predicted control signals.
    pub fn apply_controls(&mut self, s: f64, alpha: f64, omega_scale: f64) {
        self.inner.apply_controls(s, alpha, omega_scale);
    }

    /// Switch the active Mandelbrot lobe.
    pub fn set_lobe(&mut self, lobe: u32, sub_lobe: u32) {
        self.inner.lobe = lobe;
        self.inner.sub_lobe = sub_lobe;
    }

    /// Advance the Player by dt, moving c toward the model-driven target,
    /// biased along the Shore's contours. Returns the new c.
    pub fn step(&mut self, dt: f64, h: f64, band_gates: Option<Vec<f64>>) -> Complex {
        let c = self
            .inner
            .step(dt, h, band_gates.as_deref())
            .into();
        c
    }
}

/// Residual parameters
#[wasm_bindgen]
#[derive(Clone)]
pub struct ResidualParams {
    inner: RustResidualParams,
}

#[wasm_bindgen]
impl ResidualParams {
    /// Create default residual parameters
    #[wasm_bindgen(constructor)]
    pub fn new(
        k_residuals: usize,
        residual_cap: f64,
        radius_scale: f64,
    ) -> ResidualParams {
        ResidualParams {
            inner: RustResidualParams {
                k_residuals,
                residual_cap,
                radius_scale,
            },
        }
    }
}

/// Synthesize Julia parameter from orbit state
#[wasm_bindgen(js_name = "synthesize")]
pub fn synthesize(
    state: &OrbitState,
    residual_params: &ResidualParams,
    band_gates: Option<Vec<f64>>,
) -> Complex {
    let c = rust_synthesize(
        &state.inner,
        residual_params.inner,
        band_gates.as_deref(),
    );
    c.into()
}

/// Step the orbit forward and synthesize
#[wasm_bindgen]
pub fn step(
    state: &mut OrbitState,
    dt: f64,
    residual_params: &ResidualParams,
    band_gates: Option<Vec<f64>>,
) -> Complex {
    state.inner.advance(dt);
    synthesize(state, residual_params, band_gates)
}


/// --- Minimap / mip pyramid bindings (issue #88) ---

use num_complex::Complex64 as Rc64;

/// Set the mip pyramid from host-provided flat planes (row-major, per level).
#[wasm_bindgen]
pub fn set_mip_pyramid(
    f_flat: Vec<f32>,
    s_flat: Vec<f32>,
    widths: Vec<u32>,
    heights: Vec<u32>,
    re_min: f64,
    re_max: f64,
    im_min: f64,
    im_max: f64,
) -> Result<(), JsValue> {
    if f_flat.len() != s_flat.len() || widths.len() != heights.len() {
        return Err(JsValue::from_str("mip pyramid buffer mismatch"));
    }
    let total: usize = widths
        .iter()
        .zip(heights.iter())
        .map(|(&w, &h)| w as usize * h as usize)
        .sum();
    if f_flat.len() != total {
        return Err(JsValue::from_str(
            "mip pyramid plane sizes do not match data length",
        ));
    }
    let split = |flat: &[f32]| -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(widths.len());
        let mut pos = 0usize;
        for i in 0..widths.len() {
            let n = widths[i] as usize * heights[i] as usize;
            out.push(flat[pos..pos + n].to_vec());
            pos += n;
        }
        out
    };
    let mut pyr = runtime_core::minimap::MipPyramid::from_levels(
        split(&s_flat),
        widths.iter().map(|&w| w as usize).collect(),
        heights.iter().map(|&h| h as usize).collect(),
        re_min,
        re_max,
        im_min,
        im_max,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    pyr.set_escape_field(split(&f_flat));
    runtime_core::minimap::set_pyramid(pyr).map_err(|e| JsValue::from_str(&e))
}

/// The Player's full observation at c: 4x81 greys + 8 slope values = 332.
#[wasm_bindgen]
pub fn player_observation(real: f64, imag: f64) -> Result<Vec<f32>, JsValue> {
    runtime_core::minimap::with_pyramid(|pyr| {
        let pyr = pyr.ok_or_else(|| JsValue::from_str("mip pyramid not loaded"))?;
        pyr.player_observation(Rc64::new(real, imag))
            .ok_or_else(|| JsValue::from_str("c outside map extent"))
    })
}

/// Slope of the shore-proximity field at c on a mip level. Returns [gx, gy].
#[wasm_bindgen]
pub fn minimap_slope(real: f64, imag: f64, level: usize) -> Result<Vec<f64>, JsValue> {
    runtime_core::minimap::with_pyramid(|pyr| {
        let pyr = pyr.ok_or_else(|| JsValue::from_str("mip pyramid not loaded"))?;
        let (gx, gy) = pyr
            .slope(Rc64::new(real, imag), level)
            .ok_or_else(|| JsValue::from_str("c outside map extent"))?;
        Ok(vec![gx, gy])
    })
}

/// Contour-biased integrator step for Physics. Returns [new_real, new_imag].
#[wasm_bindgen]
pub fn contour_biased_step(
    real: f64,
    imag: f64,
    u_real: f64,
    u_imag: f64,
    h: f64,
    d_star: f64,
    max_step: f64,
    level: usize,
    energy: f64,
) -> Result<Vec<f64>, JsValue> {
    let (nr, ni) = runtime_core::minimap::contour_biased_step(
        real, imag, u_real, u_imag, h, d_star, max_step, level, energy,
    )?;
    Ok(vec![nr, ni])
}

/// --- May-proven OrbitController bindings (restored baseline) ---

#[wasm_bindgen]
#[derive(Clone)]
pub struct OrbitController {
    inner: RustOrbitController,
}

#[wasm_bindgen]
impl OrbitController {
    #[wasm_bindgen(constructor)]
    pub fn new(s: f64, alpha: f64, omega: f64) -> OrbitController {
        OrbitController {
            inner: RustOrbitController::new(s, alpha, omega),
        }
    }

    /// Wobble phase (diagnostic).
    #[wasm_bindgen(getter)]
    pub fn theta(&self) -> f64 {
        self.inner.theta
    }

    /// Apply model-predicted control signals (s, alpha).
    pub fn apply_controls(&mut self, s: f64, alpha: f64) {
        self.inner.apply_controls(s, alpha);
    }

    /// Refinement 1 toggle: momentum (persistent velocity + drag).
    #[wasm_bindgen(setter)]
    pub fn set_momentum(&mut self, on: bool) {
        self.inner.momentum = on;
    }

    /// Friction for momentum refinement (default 0.90).
    #[wasm_bindgen(setter)]
    pub fn set_drag(&mut self, drag: f64) {
        self.inner.drag = drag;
    }

    /// Audio thrust for momentum: sustained energy builds inertia.
    #[wasm_bindgen(setter)]
    pub fn set_thrust(&mut self, thrust: f64) {
        self.inner.thrust = thrust;
    }

    /// Audio energy in [0, 1]: raises the servo's target shore-proximity
    /// (loud audio pulls c toward the Shore).
    #[wasm_bindgen(setter)]
    pub fn set_energy(&mut self, energy: f64) {
        self.inner.energy = energy.clamp(0.0, 1.0);
    }

    /// Refinement 2 toggle: shore bias via minimap contour stepping.
    #[wasm_bindgen(setter)]
    pub fn set_shore_bias(&mut self, on: bool) {
        self.inner.shore_bias = on;
    }

    /// Target shore proximity for the shore-bias servo.
    #[wasm_bindgen(setter)]
    pub fn set_d_star(&mut self, d_star: f64) {
        self.inner.d_star = d_star;
    }

    /// Max world-space step per frame for shore bias.
    #[wasm_bindgen(setter)]
    pub fn set_max_step(&mut self, max_step: f64) {
        self.inner.max_step = max_step;
    }

    /// Advance one frame; returns the new c. `h` is the transient signal
    /// in [0, 1] — near 1 opens the Shore wall for boundary crossing.
    pub fn step(&mut self, dt: f64, h: f64, band_gates: Option<Vec<f64>>) -> Complex {
        self.inner.step(dt, band_gates.as_deref(), h).into()
    }
}

// ---------------------------------------------------------------------------
// CycleBank (issue #92) — BROWSER surface.
//
// This is the SAME Rust CycleBank the trainer uses via PyO3. TypeScript only
// passes canonical ticks in and reads observed modes / relations out. ALL
// transform, ridge, tracking, frequency, phase, confidence, relation, and
// prediction math stays in Rust (ADR 0001, ADR 0003); there is no TypeScript
// mirror. The TS wire shapes below are emitted into the generated .d.ts via
// the custom section so the frontend imports them rather than redeclaring.
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_CYCLE_TYPES: &'static str = r#"
/** One directly observed temporal ridge (issue #92). All math is Rust-owned. */
export interface CycleMode {
    id: number;
    frequencyHz: number;
    phase: number;
    strength: number;
    confidence: number;
    channelSupport: number;
    age: number;
    missingObservations: number;
    frequencySlope: number;
    frequencyUncertainty: number;
}

/** Diagnostic rational relationship between two observed modes. */
export interface CycleRelation {
    iId: number;
    jId: number;
    m: number;
    n: number;
    freqResidual: number;
    generalizedPhase: number;
    phaseStability: number;
}

/** One named scalar evidence channel value for an explicit observation. */
export interface CycleEvidenceChannelInput {
    name: string;
    value: number;
}
"#;

/// Deserialization mirror of the canonical camelCase tick wire format. The
/// newest-frame -> observation mapping is NOT done here; the tick is passed
/// straight to the canonical Rust seam (`cycle_observation_from_tick`).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisTickJsIn {
    features: Vec<f64>,
    sample_index: u64,
    time_seconds: f64,
    dt_seconds: f64,
    stream_epoch: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CycleEvidenceChannelJsIn {
    name: String,
    value: f64,
}

/// Canonical observed-ridge CycleBank (issue #92), browser surface.
///
/// The browser feeds one canonical `AnalysisTick` per authoritative hop and
/// reads the currently observed modes / relations. It never interprets the
/// rolling feature window's offsets itself.
#[wasm_bindgen]
pub struct CycleBank {
    inner: RustCycleBank,
}

#[wasm_bindgen]
impl CycleBank {
    /// Construct with the canonical defaults (no config). Config overrides
    /// are a Rust-side concern; the browser runs the canonical pipeline.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<CycleBank, JsValue> {
        let inner = RustCycleBank::try_new(RustCycleBankConfig::default())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(CycleBank { inner })
    }

    /// The Rust-owned contract version (`CYCLE_BANK_VERSION`).
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        CYCLE_BANK_VERSION.to_string()
    }

    /// Feed one canonical analysis tick (the `AnalysisTick` produced by the
    /// wasm `AnalysisTimebase.ingest`/`flush`). Returns the current observed
    /// `CycleMode[]`. The newest-frame extraction is done in Rust.
    #[wasm_bindgen]
    pub fn observe_tick(&mut self, tick: JsValue) -> Result<JsValue, JsValue> {
        let js_in: AnalysisTickJsIn = serde_wasm_bindgen::from_value(tick)
            .map_err(|e| JsValue::from_str(&format!("invalid AnalysisTick: {e}")))?;
        let tick = RustAnalysisTick {
            features: js_in.features,
            sample_index: js_in.sample_index,
            time_seconds: js_in.time_seconds,
            dt_seconds: js_in.dt_seconds,
            stream_epoch: js_in.stream_epoch,
        };
        let obs = cycle_observation_from_tick(&tick).ok_or_else(|| {
            JsValue::from_str("tick feature window is not the expected frame-major shape")
        })?;
        self.inner
            .observe(&obs)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.modes_js()
    }

    /// Feed one explicit observation of named scalar evidence channels
    /// (diagnostic entry point; the production path is `observe_tick`).
    #[wasm_bindgen]
    pub fn observe(
        &mut self,
        sample_index: u64,
        dt_seconds: f64,
        stream_epoch: u64,
        channels: JsValue,
    ) -> Result<JsValue, JsValue> {
        let channels: Vec<CycleEvidenceChannelJsIn> =
            serde_wasm_bindgen::from_value(channels)
                .map_err(|e| JsValue::from_str(&format!("invalid channels: {e}")))?;
        let obs = RustCycleObservation {
            sample_index,
            dt_seconds,
            stream_epoch,
            channels: channels
                .into_iter()
                .map(|c| RustCycleEvidenceChannel::new(c.name, c.value))
                .collect(),
        };
        self.inner
            .observe(&obs)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.modes_js()
    }

    /// Current confirmed observed modes (`CycleMode[]`).
    #[wasm_bindgen]
    pub fn modes(&self) -> JsValue {
        self.modes_js().unwrap_or(JsValue::NULL)
    }

    /// Rational relations among the currently observed modes (latest batch).
    #[wasm_bindgen]
    pub fn latest_relations(&self) -> JsValue {
        let relations = self.inner.latest_relations();
        serde_wasm_bindgen::to_value(&relations).unwrap_or(JsValue::NULL)
    }

    /// Number of currently confirmed modes.
    #[wasm_bindgen]
    pub fn num_modes(&self) -> usize {
        self.inner.num_modes()
    }

    /// Deterministic discontinuity reset.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    fn modes_js(&self) -> Result<JsValue, JsValue> {
        let modes = self.inner.modes();
        serde_wasm_bindgen::to_value(&modes).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for CycleBank {
    fn default() -> Self {
        Self::new().expect("canonical CycleBankConfig is valid")
    }
}
