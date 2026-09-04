//! Orbit synthesizer for Julia set parameter generation
//! 
//! WebAssembly bindings to runtime_core for browser use.

use wasm_bindgen::prelude::*;
use js_sys::Array;
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
use runtime_core::controls::{
    ControlsV2 as RustControlsV2, MotionControls as RustMotionControls,
    JuliaViewControls as RustJuliaViewControls, JuliaViewState as RustJuliaViewState,
    ColorIntent as RustColorIntent, Harmony as RustHarmony, CONTROLS_VERSION,
};
use runtime_core::manifold::{
    ManifoldConfig as RustManifoldConfig, embedding as rust_embedding,
    jacobian as rust_jacobian, q_dot as rust_q_dot, sigma_dot as rust_sigma_dot,
    signed_distance as rust_signed_distance,
    regularized_distance as rust_regularized_distance,
    mandelbrot_scale as rust_mandelbrot_scale,
    scale_gradient as rust_scale_gradient,
    scale_hessian as rust_scale_hessian,
    induced_metric as rust_induced_metric,
    kinetic_energy as rust_kinetic_energy,
    potential_energy as rust_potential_energy,
    total_energy as rust_total_energy,
    christoffel_symbols as rust_christoffel_symbols,
    geodesic_acceleration as rust_geodesic_acceleration,
    potential_force as rust_potential_force,
    apply_generalized_force as rust_apply_generalized_force,
    drag_force as rust_drag_force,
    integrate_step as rust_integrate_step, unsigned_distance as rust_unsigned_distance,
};
use runtime_core::debug::{
    TerrainPatch as RustTerrainPatch,
    DEBUG_SNAPSHOT_VERSION, CANONICAL_DT as DEBUG_CANONICAL_DT,
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
        controls_version: String,
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
        controls_version: CONTROLS_VERSION.to_string(),
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

    // ---- Manifold physics (issue #106) ----

    /// Enable or disable manifold physics. When on, step() routes through a
    /// LEGACY ADAPTER that translates the old (s, alpha, energy) servo into a
    /// generalized force covector for the musically-ignorant manifold kernel.
    /// Transitional; not destination Controls v2 (issue #107).
    #[wasm_bindgen(setter)]
    pub fn set_manifold_physics(&mut self, on: bool) {
        self.inner.manifold_physics = on;
    }

    /// Whether manifold physics is currently enabled.
    #[wasm_bindgen(getter)]
    pub fn manifold_physics(&self) -> bool {
        self.inner.manifold_physics
    }

    /// The most recent manifold-physics failure, if any. When manifold mode is
    /// selected and the integrator fails, the controller fails closed (holds
    /// the last valid state) and records the error here.
    #[wasm_bindgen(getter)]
    pub fn manifold_error(&self) -> Option<String> {
        self.inner.manifold_error.clone()
    }

    /// Set the manifold configuration (used only when manifold_physics is on).
    pub fn set_manifold_config(&mut self, config: &ManifoldConfig) {
        self.inner.manifold_config = config.into();
    }

    /// Get the current manifold configuration.
    pub fn manifold_config(&self) -> ManifoldConfig {
        ManifoldConfig {
            d_ref: self.inner.manifold_config.d_ref,
            epsilon: self.inner.manifold_config.epsilon,
            lambda_sq: self.inner.manifold_config.lambda_sq,
            kappa: self.inner.manifold_config.kappa,
        }
    }

    /// Set the drag coefficient for manifold physics (beta in Q_drag = -beta*G*v).
    #[wasm_bindgen(setter)]
    pub fn set_manifold_drag(&mut self, drag: f64) {
        self.inner.manifold_drag = drag;
    }

    /// Get the drag coefficient for manifold physics.
    #[wasm_bindgen(getter)]
    pub fn manifold_drag(&self) -> f64 {
        self.inner.manifold_drag
    }

    /// Authoritative player position c in the complex plane.
    /// Read/write so test harnesses and the debug cockpit can seed a
    /// non-default starting point (e.g. "approach from outside M" trajectories
    /// that begin at a seahorse-basin c without paying the launch cost of
    /// crossing the cardioid ridge).
    #[wasm_bindgen(getter)]
    pub fn c(&self) -> Complex {
        Complex { real: self.inner.c.re, imag: self.inner.c.im }
    }

    /// Seed the authoritative player position from (re, im) parts. The next
    /// step_with_controls call advances from this point. Parts (not a
    /// Complex instance) so callers never need to construct wasm objects.
    #[wasm_bindgen(js_name = "setC")]
    pub fn set_c(&mut self, re: f64, im: f64) {
        self.inner.c = RustComplex::new(re, im);
    }

    /// Authoritative planar velocity (vx, vy) used by the destination
    /// manifold integrator.
    #[wasm_bindgen(getter)]
    pub fn velocity(&self) -> Complex {
        Complex { real: self.inner.velocity.re, imag: self.inner.velocity.im }
    }

    /// Seed the planar velocity from (vx, vy) parts. The next
    /// step_with_controls call applies Q_drive and drag from this velocity.
    #[wasm_bindgen(js_name = "setVelocity")]
    pub fn set_velocity(&mut self, vx: f64, vy: f64) {
        self.inner.velocity = RustComplex::new(vx, vy);
    }

    /// Destination manifold step driven by Controls v2 (issue #107/#106).
    #[wasm_bindgen(js_name = "stepWithControls")]
    pub fn step_with_controls(&mut self, dt: f64, motion: &MotionControls) -> Complex {
        let c = self.inner.step_with_controls(dt, &motion.clone().into());
        c.into()
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

// ---------------------------------------------------------------------------
// Manifold physics (issue #106) — BROWSER surface.
//
// Mirrors the Python bindings so the browser can run the same differential
// geometry as the trainer. Rust remains canonical under ADR 0001.
// ---------------------------------------------------------------------------

/// Manifold configuration for the browser (issue #106).
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ManifoldConfig {
    d_ref: f64,
    epsilon: f64,
    lambda_sq: f64,
    kappa: f64,
}

impl From<&ManifoldConfig> for RustManifoldConfig {
    fn from(c: &ManifoldConfig) -> RustManifoldConfig {
        RustManifoldConfig {
            d_ref: c.d_ref,
            epsilon: c.epsilon,
            lambda_sq: c.lambda_sq,
            kappa: c.kappa,
        }
    }
}

#[wasm_bindgen]
impl ManifoldConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(d_ref: f64, epsilon: f64, lambda_sq: f64, kappa: f64) -> ManifoldConfig {
        ManifoldConfig {
            d_ref,
            epsilon,
            lambda_sq,
            kappa,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn d_ref(&self) -> f64 {
        self.d_ref
    }
    #[wasm_bindgen(getter)]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
    #[wasm_bindgen(getter)]
    pub fn lambda_sq(&self) -> f64 {
        self.lambda_sq
    }
    #[wasm_bindgen(getter)]
    pub fn kappa(&self) -> f64 {
        self.kappa
    }
}

/// Signed distance to the Mandelbrot boundary. Positive outside, negative inside.
#[wasm_bindgen]
pub fn manifold_signed_distance(real: f64, imag: f64) -> Result<f64, JsValue> {
    let c = RustComplex::new(real, imag);
    rust_signed_distance(c).map_err(|e| JsValue::from_str(&e))
}

/// Regularized distance rho(c) = sqrt(D^2 + epsilon^2).
#[wasm_bindgen]
pub fn manifold_regularized_distance(real: f64, imag: f64, epsilon: f64) -> Result<f64, JsValue> {
    let c = RustComplex::new(real, imag);
    rust_regularized_distance(c, epsilon).map_err(|e| JsValue::from_str(&e))
}

/// Mandelbrot scale sigma(c) = log2(d_ref / rho(c)).
#[wasm_bindgen]
pub fn manifold_mandelbrot_scale(real: f64, imag: f64, config: &ManifoldConfig) -> Result<f64, JsValue> {
    let c = RustComplex::new(real, imag);
    rust_mandelbrot_scale(c, &config.into()).map_err(|e| JsValue::from_str(&e))
}

/// Scale gradient ∇sigma(c) = (∂sigma/∂x, ∂sigma/∂y). Returns [gx, gy].
#[wasm_bindgen]
pub fn manifold_scale_gradient(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Vec<f64>, JsValue> {
    let c = RustComplex::new(real, imag);
    let (gx, gy) = rust_scale_gradient(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![gx, gy])
}

/// Scale Hessian [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]].
/// Returns a flat JS array [xx, xy, xy, yy].
#[wasm_bindgen]
pub fn manifold_scale_hessian(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let h = rust_scale_hessian(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(h[0][0]));
    arr.push(&JsValue::from_f64(h[0][1]));
    arr.push(&JsValue::from_f64(h[1][0]));
    arr.push(&JsValue::from_f64(h[1][1]));
    Ok(arr)
}

/// Induced metric G(c) = I + lambda^2 * grad_sigma * grad_sigma^T.
/// Returns a flat JS array [g11, g12, g12, g22].
#[wasm_bindgen]
pub fn manifold_induced_metric(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let g = rust_induced_metric(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(g[0][0]));
    arr.push(&JsValue::from_f64(g[0][1]));
    arr.push(&JsValue::from_f64(g[1][0]));
    arr.push(&JsValue::from_f64(g[1][1]));
    Ok(arr)
}

/// Kinetic energy K = 1/2 v^T G v.
#[wasm_bindgen]
pub fn manifold_kinetic_energy(
    vx: f64,
    vy: f64,
    real: f64,
    imag: f64,
    config: &ManifoldConfig,
) -> Result<f64, JsValue> {
    let c = RustComplex::new(real, imag);
    rust_kinetic_energy((vx, vy), c, &config.into()).map_err(|e| JsValue::from_str(&e))
}

/// Native potential U = kappa * sigma(c).
#[wasm_bindgen]
pub fn manifold_potential_energy(real: f64, imag: f64, config: &ManifoldConfig) -> Result<f64, JsValue> {
    let c = RustComplex::new(real, imag);
    rust_potential_energy(c, &config.into()).map_err(|e| JsValue::from_str(&e))
}

/// Total mechanical energy E = K + U.
#[wasm_bindgen]
pub fn manifold_total_energy(
    vx: f64,
    vy: f64,
    real: f64,
    imag: f64,
    config: &ManifoldConfig,
) -> Result<f64, JsValue> {
    let c = RustComplex::new(real, imag);
    rust_total_energy((vx, vy), c, &config.into()).map_err(|e| JsValue::from_str(&e))
}

/// Christoffel symbols Gamma^i_jk. Returns a flat JS array of 8 values:
/// [Gamma^0_00, Gamma^0_01, Gamma^0_10, Gamma^0_11, Gamma^1_00, Gamma^1_01, Gamma^1_10, Gamma^1_11].
#[wasm_bindgen]
pub fn manifold_christoffel_symbols(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let g = rust_christoffel_symbols(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(g[0][0][0]));
    arr.push(&JsValue::from_f64(g[0][0][1]));
    arr.push(&JsValue::from_f64(g[0][1][0]));
    arr.push(&JsValue::from_f64(g[0][1][1]));
    arr.push(&JsValue::from_f64(g[1][0][0]));
    arr.push(&JsValue::from_f64(g[1][0][1]));
    arr.push(&JsValue::from_f64(g[1][1][0]));
    arr.push(&JsValue::from_f64(g[1][1][1]));
    Ok(arr)
}

/// Geodesic acceleration term: Gamma^i_jk v^j v^k. Returns [ax, ay].
#[wasm_bindgen]
pub fn manifold_geodesic_acceleration(
    vx: f64,
    vy: f64,
    real: f64,
    imag: f64,
    config: &ManifoldConfig,
) -> Result<Vec<f64>, JsValue> {
    let c = RustComplex::new(real, imag);
    let (ax, ay) = rust_geodesic_acceleration((vx, vy), c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![ax, ay])
}

/// Generalized potential force covector: Q_potential = -grad U = -kappa grad sigma.
/// Returns [Qx, Qy]. This is a covector, not a coordinate acceleration; convert
/// with `manifold_apply_generalized_force`.
#[wasm_bindgen]
pub fn manifold_potential_force(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Vec<f64>, JsValue> {
    let c = RustComplex::new(real, imag);
    let (fx, fy) = rust_potential_force(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![fx, fy])
}

/// Convert a generalized force covector to coordinate acceleration: a = G^{-1} Q.
/// Returns [ax, ay]. This is the single place G^{-1} maps a covector to acceleration.
#[wasm_bindgen]
pub fn manifold_apply_generalized_force(
    qx: f64,
    qy: f64,
    real: f64,
    imag: f64,
    config: &ManifoldConfig,
) -> Result<Vec<f64>, JsValue> {
    let c = RustComplex::new(real, imag);
    let (ax, ay) = rust_apply_generalized_force((qx, qy), c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![ax, ay])
}

/// Metric-consistent isotropic drag covector: Q_drag = -beta G v. Returns [Qx, Qy].
/// This is a covector, not a coordinate acceleration; its power P = v^T Q_drag <= 0.
#[wasm_bindgen]
pub fn manifold_drag_force(
    vx: f64,
    vy: f64,
    real: f64,
    imag: f64,
    beta: f64,
    config: &ManifoldConfig,
) -> Result<Vec<f64>, JsValue> {
    let c = RustComplex::new(real, imag);
    let (qx, qy) = rust_drag_force((vx, vy), c, beta, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![qx, qy])
}

/// Embedding q(c) = (x, y, sigma(c)). Returns [x, y, sigma].
#[wasm_bindgen]
pub fn manifold_embedding(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let (x, y, s) = rust_embedding(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(x));
    arr.push(&JsValue::from_f64(y));
    arr.push(&JsValue::from_f64(s));
    Ok(arr)
}

/// Jacobian J_q(c) = ∂q/∂(x,y) as a 3×2 matrix. Returns flat array [1,0,0,1,sigma_x,sigma_y].
#[wasm_bindgen]
pub fn manifold_jacobian(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let j = rust_jacobian(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    for row in j { for v in row { arr.push(&JsValue::from_f64(v)); } }
    Ok(arr)
}

/// Embedded velocity q_dot = J_q(c) v. Returns [vx, vy, sigma_dot].
#[wasm_bindgen]
pub fn manifold_q_dot(vx: f64, vy: f64, real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let (qx, qy, qz) = rust_q_dot(c, (vx, vy), &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(qx));
    arr.push(&JsValue::from_f64(qy));
    arr.push(&JsValue::from_f64(qz));
    Ok(arr)
}

/// Time derivative of Mandelbrot scale: sigma_dot = ∇sigma·v. No independent v_sigma.
#[wasm_bindgen]
pub fn manifold_sigma_dot(vx: f64, vy: f64, real: f64, imag: f64, config: &ManifoldConfig) -> Result<f64, JsValue> {
    let c = RustComplex::new(real, imag);
    rust_sigma_dot(c, (vx, vy), &config.into()).map_err(|e| JsValue::from_str(&e))
}

/// Unsigned geometric distance d(c) = |D(c)|. Distinct from S sensitivity.
#[wasm_bindgen]
pub fn manifold_unsigned_distance(real: f64, imag: f64) -> Result<f64, JsValue> {
    let c = RustComplex::new(real, imag);
    rust_unsigned_distance(c).map_err(|e| JsValue::from_str(&e))
}

/// Semi-implicit Euler integration step for manifold dynamics.
///
/// Integrates: r_ddot + Gamma(r_dot, r_dot) = -G^{-1}∇U + G^{-1}Q
///
/// Returns [new_re, new_im, new_vx, new_vy, kinetic, potential, total, delta_total, delta_kinetic].
#[wasm_bindgen]
pub fn manifold_integrate_step(
    c_re: f64,
    c_im: f64,
    vx: f64,
    vy: f64,
    qx: f64,
    qy: f64,
    beta: f64,
    dt: f64,
    config: &ManifoldConfig,
) -> Result<Vec<f64>, JsValue> {
    let c = RustComplex::new(c_re, c_im);
    let (c_new, v_new, info) = rust_integrate_step(
        c,
        (vx, vy),
        (qx, qy),
        beta,
        dt,
        &config.into(),
    )
    .map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![
        c_new.re,
        c_new.im,
        v_new.0,
        v_new.1,
        info.kinetic,
        info.potential,
        info.total,
        info.delta_total,
        info.delta_kinetic,
    ])
}

// ---------------------------------------------------------------------------
// Controls v2 (issue #107) — BROWSER surface.
//
// The same Rust ControlsV2 the trainer uses via PyO3. This fulfills the
// binding-symmetry gate (see shared/canonical_surfaces.json) and makes the
// 13-channel contract reachable from both surfaces via their real public
// seams (ADR 0001).
// ---------------------------------------------------------------------------

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct MotionControls {
    direction_x: f64,
    direction_y: f64,
    throttle: f64,
    brake: f64,
    grip: f64,
    impulse: f64,
}

#[wasm_bindgen]
impl MotionControls {
    #[wasm_bindgen(constructor)]
    pub fn new(
        direction_x: f64,
        direction_y: f64,
        throttle: f64,
        brake: f64,
        grip: f64,
        impulse: f64,
    ) -> MotionControls {
        MotionControls { direction_x, direction_y, throttle, brake, grip, impulse }
    }

    #[wasm_bindgen(getter)]
    pub fn direction_x(&self) -> f64 { self.direction_x }
    #[wasm_bindgen(setter)]
    pub fn set_direction_x(&mut self, v: f64) { self.direction_x = v; }
    #[wasm_bindgen(getter)]
    pub fn direction_y(&self) -> f64 { self.direction_y }
    #[wasm_bindgen(setter)]
    pub fn set_direction_y(&mut self, v: f64) { self.direction_y = v; }
    #[wasm_bindgen(getter)]
    pub fn throttle(&self) -> f64 { self.throttle }
    #[wasm_bindgen(setter)]
    pub fn set_throttle(&mut self, v: f64) { self.throttle = v; }
    #[wasm_bindgen(getter)]
    pub fn brake(&self) -> f64 { self.brake }
    #[wasm_bindgen(setter)]
    pub fn set_brake(&mut self, v: f64) { self.brake = v; }
    #[wasm_bindgen(getter)]
    pub fn grip(&self) -> f64 { self.grip }
    #[wasm_bindgen(setter)]
    pub fn set_grip(&mut self, v: f64) { self.grip = v; }
    #[wasm_bindgen(getter)]
    pub fn impulse(&self) -> f64 { self.impulse }
    #[wasm_bindgen(setter)]
    pub fn set_impulse(&mut self, v: f64) { self.impulse = v; }

    pub fn clamped(&self) -> MotionControls {
        let inner: RustMotionControls = self.clone().into();
        inner.clamped().into()
    }

    pub fn drive_magnitude(&self) -> f64 {
        let inner: RustMotionControls = self.clone().into();
        inner.drive_magnitude()
    }

    pub fn friction_beta(&self) -> f64 {
        let inner: RustMotionControls = self.clone().into();
        inner.friction_beta()
    }
}

impl From<RustMotionControls> for MotionControls {
    fn from(m: RustMotionControls) -> Self {
        Self { direction_x: m.direction[0], direction_y: m.direction[1], throttle: m.throttle, brake: m.brake, grip: m.grip, impulse: m.impulse }
    }
}
impl From<MotionControls> for RustMotionControls {
    fn from(m: MotionControls) -> RustMotionControls {
        RustMotionControls { direction: [m.direction_x, m.direction_y], throttle: m.throttle, brake: m.brake, grip: m.grip, impulse: m.impulse }.clamped()
    }
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct JuliaViewControls {
    zoom_delta: f64,
    rotation_delta: f64,
    hue_delta: f64,
    chroma_delta: f64,
    lightness_delta: f64,
    accent_delta: f64,
    harmony_shift: f64,
}

#[wasm_bindgen]
impl JuliaViewControls {
    #[wasm_bindgen(constructor)]
    pub fn new(
        zoom_delta: f64,
        rotation_delta: f64,
        hue_delta: f64,
        chroma_delta: f64,
        lightness_delta: f64,
        accent_delta: f64,
        harmony_shift: f64,
    ) -> JuliaViewControls {
        JuliaViewControls { zoom_delta, rotation_delta, hue_delta, chroma_delta, lightness_delta, accent_delta, harmony_shift }
    }

    #[wasm_bindgen(getter)]
    pub fn zoom_delta(&self) -> f64 { self.zoom_delta }
    #[wasm_bindgen(setter)]
    pub fn set_zoom_delta(&mut self, v: f64) { self.zoom_delta = v; }
    #[wasm_bindgen(getter)]
    pub fn rotation_delta(&self) -> f64 { self.rotation_delta }
    #[wasm_bindgen(setter)]
    pub fn set_rotation_delta(&mut self, v: f64) { self.rotation_delta = v; }
    #[wasm_bindgen(getter)]
    pub fn hue_delta(&self) -> f64 { self.hue_delta }
    #[wasm_bindgen(setter)]
    pub fn set_hue_delta(&mut self, v: f64) { self.hue_delta = v; }
    #[wasm_bindgen(getter)]
    pub fn chroma_delta(&self) -> f64 { self.chroma_delta }
    #[wasm_bindgen(setter)]
    pub fn set_chroma_delta(&mut self, v: f64) { self.chroma_delta = v; }
    #[wasm_bindgen(getter)]
    pub fn lightness_delta(&self) -> f64 { self.lightness_delta }
    #[wasm_bindgen(setter)]
    pub fn set_lightness_delta(&mut self, v: f64) { self.lightness_delta = v; }
    #[wasm_bindgen(getter)]
    pub fn accent_delta(&self) -> f64 { self.accent_delta }
    #[wasm_bindgen(setter)]
    pub fn set_accent_delta(&mut self, v: f64) { self.accent_delta = v; }
    #[wasm_bindgen(getter)]
    pub fn harmony_shift(&self) -> f64 { self.harmony_shift }
    #[wasm_bindgen(setter)]
    pub fn set_harmony_shift(&mut self, v: f64) { self.harmony_shift = v; }

    pub fn clamped(&self) -> JuliaViewControls {
        let inner: RustJuliaViewControls = self.clone().into();
        inner.clamped().into()
    }
}

impl From<RustJuliaViewControls> for JuliaViewControls {
    fn from(v: RustJuliaViewControls) -> Self {
        Self { zoom_delta: v.zoom_delta, rotation_delta: v.rotation_delta, hue_delta: v.hue_delta, chroma_delta: v.chroma_delta, lightness_delta: v.lightness_delta, accent_delta: v.accent_delta, harmony_shift: v.harmony_shift }
    }
}
impl From<JuliaViewControls> for RustJuliaViewControls {
    fn from(v: JuliaViewControls) -> RustJuliaViewControls {
        RustJuliaViewControls { zoom_delta: v.zoom_delta, rotation_delta: v.rotation_delta, hue_delta: v.hue_delta, chroma_delta: v.chroma_delta, lightness_delta: v.lightness_delta, accent_delta: v.accent_delta, harmony_shift: v.harmony_shift }.clamped()
    }
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ControlsV2 {
    motion: MotionControls,
    view: JuliaViewControls,
}

#[wasm_bindgen]
impl ControlsV2 {
    #[wasm_bindgen(constructor)]
    pub fn new(motion: MotionControls, view: JuliaViewControls) -> ControlsV2 {
        ControlsV2 { motion, view }
    }

    #[wasm_bindgen(getter)]
    pub fn motion(&self) -> MotionControls { self.motion.clone() }
    #[wasm_bindgen(setter)]
    pub fn set_motion(&mut self, v: MotionControls) { self.motion = v; }
    #[wasm_bindgen(getter)]
    pub fn view(&self) -> JuliaViewControls { self.view.clone() }
    #[wasm_bindgen(setter)]
    pub fn set_view(&mut self, v: JuliaViewControls) { self.view = v; }

    pub fn clamped(&self) -> ControlsV2 {
        let inner: RustControlsV2 = self.clone().into();
        inner.clamped().into()
    }

    pub fn to_model_output(&self) -> Vec<f64> {
        let inner: RustControlsV2 = self.clone().into();
        inner.to_model_output()
    }

    #[wasm_bindgen(js_name = "fromModelOutput")]
    pub fn from_model_output(output: Vec<f64>) -> Result<ControlsV2, JsValue> {
        RustControlsV2::from_model_output(&output).map(|c| c.into()).map_err(|e| JsValue::from_str(&e))
    }

    #[wasm_bindgen(js_name = "modelOutputOrder")]
    pub fn model_output_order() -> Vec<JsValue> {
        RustControlsV2::model_output_order().into_iter().map(|s| JsValue::from_str(s)).collect()
    }
}

impl From<RustControlsV2> for ControlsV2 {
    fn from(c: RustControlsV2) -> Self {
        Self { motion: c.motion.into(), view: c.view.into() }
    }
}
impl From<ControlsV2> for RustControlsV2 {
    fn from(c: ControlsV2) -> RustControlsV2 {
        RustControlsV2 { motion: c.motion.into(), view: c.view.into() }.clamped()
    }
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ColorIntent {
    anchor_hue: f64,
    chroma: f64,
    lightness: f64,
    harmony: String,
    accent_weight: f64,
}

#[wasm_bindgen]
impl ColorIntent {
    #[wasm_bindgen(constructor)]
    pub fn new(anchor_hue: f64, chroma: f64, lightness: f64, harmony: String, accent_weight: f64) -> ColorIntent {
        ColorIntent { anchor_hue, chroma, lightness, harmony, accent_weight }
    }

    #[wasm_bindgen(getter)]
    pub fn anchor_hue(&self) -> f64 { self.anchor_hue }
    #[wasm_bindgen(setter)]
    pub fn set_anchor_hue(&mut self, v: f64) { self.anchor_hue = v; }
    #[wasm_bindgen(getter)]
    pub fn chroma(&self) -> f64 { self.chroma }
    #[wasm_bindgen(setter)]
    pub fn set_chroma(&mut self, v: f64) { self.chroma = v; }
    #[wasm_bindgen(getter)]
    pub fn lightness(&self) -> f64 { self.lightness }
    #[wasm_bindgen(setter)]
    pub fn set_lightness(&mut self, v: f64) { self.lightness = v; }
    #[wasm_bindgen(getter)]
    pub fn harmony(&self) -> String { self.harmony.clone() }
    #[wasm_bindgen(setter)]
    pub fn set_harmony(&mut self, v: String) { self.harmony = v; }
    #[wasm_bindgen(getter)]
    pub fn accent_weight(&self) -> f64 { self.accent_weight }
    #[wasm_bindgen(setter)]
    pub fn set_accent_weight(&mut self, v: f64) { self.accent_weight = v; }
}

impl From<RustColorIntent> for ColorIntent {
    fn from(c: RustColorIntent) -> Self {
        Self { anchor_hue: c.anchor_hue, chroma: c.chroma, lightness: c.lightness, harmony: c.harmony.name().to_string(), accent_weight: c.accent_weight }
    }
}
impl From<ColorIntent> for RustColorIntent {
    fn from(c: ColorIntent) -> RustColorIntent {
        let harmony = match c.harmony.as_str() {
            "monochrome" => RustHarmony::Monochrome,
            "opponent" => RustHarmony::Opponent,
            _ => RustHarmony::Analogous,
        };
        RustColorIntent { anchor_hue: c.anchor_hue, chroma: c.chroma, lightness: c.lightness, harmony, accent_weight: c.accent_weight }.clamped()
    }
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct JuliaViewState {
    zoom: f64,
    rotation: f64,
    color: ColorIntent,
    harmony_cooldown: u32,
    harmony_armed: bool,
}

#[wasm_bindgen]
impl JuliaViewState {
    #[wasm_bindgen(constructor)]
    pub fn new(zoom: f64, rotation: f64, color: Option<ColorIntent>, harmony_cooldown: u32, harmony_armed: bool) -> JuliaViewState {
        JuliaViewState { zoom, rotation, color: color.unwrap_or_else(|| RustColorIntent::default().into()), harmony_cooldown, harmony_armed }
    }

    #[wasm_bindgen(getter)]
    pub fn zoom(&self) -> f64 { self.zoom }
    #[wasm_bindgen(setter)]
    pub fn set_zoom(&mut self, v: f64) { self.zoom = v; }
    #[wasm_bindgen(getter)]
    pub fn rotation(&self) -> f64 { self.rotation }
    #[wasm_bindgen(setter)]
    pub fn set_rotation(&mut self, v: f64) { self.rotation = v; }
    #[wasm_bindgen(getter)]
    pub fn color(&self) -> ColorIntent { self.color.clone() }
    #[wasm_bindgen(setter)]
    pub fn set_color(&mut self, v: ColorIntent) { self.color = v; }
    #[wasm_bindgen(getter)]
    pub fn harmony_cooldown(&self) -> u32 { self.harmony_cooldown }
    #[wasm_bindgen(setter)]
    pub fn set_harmony_cooldown(&mut self, v: u32) { self.harmony_cooldown = v; }
    #[wasm_bindgen(getter)]
    pub fn harmony_armed(&self) -> bool { self.harmony_armed }
    #[wasm_bindgen(setter)]
    pub fn set_harmony_armed(&mut self, v: bool) { self.harmony_armed = v; }

    #[wasm_bindgen(js_name = "applyControls")]
    pub fn apply_controls(&mut self, controls: &JuliaViewControls) {
        let mut inner: RustJuliaViewState = self.clone().into();
        inner.apply_controls(controls.clone().into());
        *self = inner.into();
    }

    pub fn clamped(&self) -> JuliaViewState {
        let inner: RustJuliaViewState = self.clone().into();
        inner.clamped().into()
    }
}

impl From<RustJuliaViewState> for JuliaViewState {
    fn from(s: RustJuliaViewState) -> Self {
        Self { zoom: s.zoom, rotation: s.rotation, color: s.color.into(), harmony_cooldown: s.harmony_cooldown, harmony_armed: s.harmony_armed }
    }
}
impl From<JuliaViewState> for RustJuliaViewState {
    fn from(s: JuliaViewState) -> RustJuliaViewState {
        RustJuliaViewState { zoom: s.zoom, rotation: s.rotation, color: s.color.into(), harmony_cooldown: s.harmony_cooldown, harmony_armed: s.harmony_armed }.clamped()
    }
}

// Controls-driven manifold step (destination physics seam for #107) exposed to JS for parity and
// visualization. Returns [new_re, new_im, new_vx, new_vy, kinetic, potential, total, delta_total, delta_kinetic].
#[wasm_bindgen(js_name = "controlsIntegrateStep")]
pub fn controls_integrate_step(
    c_re: f64,
    c_im: f64,
    vx: f64,
    vy: f64,
    motion: &MotionControls,
    dt: f64,
    config: &ManifoldConfig,
) -> Result<Vec<f64>, JsValue> {
    let c = RustComplex::new(c_re, c_im);
    let (c_new, v_new, info) = runtime_core::controls::integrate_motion_controls(c, (vx, vy), &motion.clone().into(), dt, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![c_new.re, c_new.im, v_new.0, v_new.1, info.kinetic, info.potential, info.total, info.delta_total, info.delta_kinetic])
}

#[wasm_bindgen(js_name = "motionDriveCovector")]
pub fn motion_drive_covector(
    c_re: f64,
    c_im: f64,
    motion: &MotionControls,
    config: &ManifoldConfig,
) -> Result<Vec<f64>, JsValue> {
    let c = RustComplex::new(c_re, c_im);
    let cov = RustMotionControls::from(motion.clone()).drive_covector(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![cov.0, cov.1])
}

// ---------------------------------------------------------------------------
// DebugSnapshot (issue #111 Phase A) — BROWSER surface.
//
// Read-only diagnostic seam: Rust owns all semantics; the browser only
// renders. Snapshot creation never mutates runtime state. Wire format is
// camelCase serde, matching the PyO3 surface and the AnalysisTick parity
// convention.
// ---------------------------------------------------------------------------

#[wasm_bindgen(typescript_custom_section)]
const TS_DEBUG_TYPES: &'static str = r#"
/** Version of the read-only DebugSnapshot contract (issue #111). */
export interface DebugSnapshotMeta {
    version: string;
    canonicalDt: number;
}
"#;

/// The DebugSnapshot contract version and canonical step cadence.
#[wasm_bindgen(js_name = "debugSnapshotMeta")]
pub fn debug_snapshot_meta() -> JsValue {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Meta {
        version: &'static str,
        canonical_dt: f64,
    }
    serde_wasm_bindgen::to_value(&Meta {
        version: DEBUG_SNAPSHOT_VERSION,
        canonical_dt: DEBUG_CANONICAL_DT,
    })
    .unwrap_or(JsValue::NULL)
}

/// Build a read-only DebugSnapshot from explicit authoritative state.
///
/// `motion_raw` is the last raw (pre-clamp) MotionControls, or null before
/// the first step. `last_delta_total` is the last step's total-energy change
/// (NaN = none). Never mutates runtime state.
#[wasm_bindgen(js_name = "debugSnapshotFromState")]
#[allow(clippy::too_many_arguments)]
pub fn debug_snapshot_from_state(
    c_re: f64,
    c_im: f64,
    vx: f64,
    vy: f64,
    motion_raw: Option<MotionControls>,
    friction_beta: f64,
    friction_power: f64,
    manifold_drag: f64,
    config: &ManifoldConfig,
    last_delta_total: f64,
    time_seconds: f64,
) -> Result<JsValue, JsValue> {
    let last_action = motion_raw.map(|m| {
        // Build the RAW struct directly: the From impl clamps, which would
        // destroy the raw-vs-effective provenance this seam exists to expose
        // (same discipline as the PyO3 surface).
        let raw = RustMotionControls {
            direction: [m.direction_x, m.direction_y],
            throttle: m.throttle,
            brake: m.brake,
            grip: m.grip,
            impulse: m.impulse,
        };
        runtime_core::debug::LastAction {
            raw,
            friction_beta,
            friction_power,
        }
    });
    let delta = if last_delta_total.is_nan() {
        None
    } else {
        Some(last_delta_total)
    };
    let mut snap = runtime_core::debug::snapshot_from_state(
        RustComplex::new(c_re, c_im),
        (vx, vy),
        last_action,
        Some(manifold_drag),
        &config.into(),
        delta,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    snap.time_seconds = time_seconds;
    serde_wasm_bindgen::to_value(&snap).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Sample an n x n terrain patch of the canonical embedding
/// Q(c) = (x, y, lambda*sigma(c)) centered at (cx, cy) with half-extent
/// `half` in c-space. Returns a camelCase JSON object:
/// { n, center, half, positions, signed, realm }.
#[wasm_bindgen(js_name = "debugTerrainPatch")]
pub fn debug_terrain_patch(
    cx: f64,
    cy: f64,
    half: f64,
    n: usize,
    config: &ManifoldConfig,
) -> Result<JsValue, JsValue> {
    let patch: RustTerrainPatch = runtime_core::debug::terrain_patch(cx, cy, half, n, &config.into())
        .map_err(|e| JsValue::from_str(&e))?;
    serde_wasm_bindgen::to_value(&patch).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Deep-zoom unsigned distance field for the minimap (issue #111 feedback:
/// the minimap is a Mandelbrot deep zoom whose zoom level follows the
/// player). Resolution-unlimited escape-iteration estimator — resolves
/// structure where the baked mip pyramid runs out of texels. Returns one
/// unsigned distance per input point (0 inside the set).
#[wasm_bindgen(js_name = "deepZoomField")]
pub fn deep_zoom_field(re: Vec<f64>, im: Vec<f64>) -> Result<Vec<f32>, JsValue> {
    runtime_core::minimap::deep_zoom_field(&re, &im).map_err(|e| JsValue::from_str(&e))
}

/// Convenience: DebugSnapshot for an OrbitController's current state.
#[wasm_bindgen]
impl OrbitController {
    /// Read-only DebugSnapshot of the current authoritative state.
    #[wasm_bindgen(js_name = "debugSnapshot")]
    pub fn debug_snapshot(&self) -> Result<JsValue, JsValue> {
        let snap = self.inner.debug_snapshot().map_err(|e| JsValue::from_str(&e))?;
        serde_wasm_bindgen::to_value(&snap).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Batch shore-proximity (S field) sampling over the canonical mip pyramid
/// (issue #111 minimap panel). Same field/level/rounding as the single-point
/// sampler; one lock for the whole batch. Returns a flat Float32Array.
#[wasm_bindgen(js_name = "minimapShoreProximityBatch")]
pub fn minimap_shore_proximity_batch(
    re: Vec<f64>,
    im: Vec<f64>,
    level: usize,
) -> Result<Vec<f32>, JsValue> {
    runtime_core::minimap::shore_proximity_batch(&re, &im, level)
        .map_err(|e| JsValue::from_str(&e))
}
