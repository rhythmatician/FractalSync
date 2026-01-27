//! WebAssembly bindings for runtime-core
//!
//! This module exposes the shared runtime to JavaScript via
//! `wasm-bindgen`. The API is intentionally kept close to the Python
//! bindings so both front-end and back-end can call the same logic.

use wasm_bindgen::prelude::*;
use js_sys::Array;
use serde::{Deserialize, Serialize};

use crate::controller::{
    step as rust_step,
    synthesize as rust_synthesize,
    OrbitState as RustOrbitState,
    ResidualParams as RustResidualParams,
    DEFAULT_BASE_OMEGA,
    DEFAULT_K_RESIDUALS,
    DEFAULT_ORBIT_SEED,
    DEFAULT_RESIDUAL_CAP,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    WINDOW_FRAMES,
};
use crate::height_controller::{
    contour_correct_delta as rust_contour_correct_delta,
    sample_height_field as rust_sample_height_field,
    ContourControllerParams as RustContourControllerParams,
    ContourState as RustContourState,
    ContourStep as RustContourStep,
    HeightFieldParams as RustHeightFieldParams,
    HeightFieldSample as RustHeightFieldSample,
    DEFAULT_CONTOUR_CORRECTION_GAIN,
    DEFAULT_CONTOUR_PROJECTION_EPSILON,
    DEFAULT_HEIGHT_ITERATIONS,
    DEFAULT_HEIGHT_MIN_MAGNITUDE,
};
use crate::features::FeatureExtractor as RustFeatureExtractor;
use crate::geometry::{lobe_point_at_angle as rust_lobe_point_at_angle, Complex as RustComplex};

/// A complex number (Julia parameter c = a + bi).
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Complex {
    real: f64,
    imag: f64,
}

impl From<RustComplex> for Complex {
    fn from(c: RustComplex) -> Self {
        Self { real: c.real, imag: c.imag }
    }
}

impl From<&Complex> for RustComplex {
    fn from(c: &Complex) -> Self {
        RustComplex::new(c.real, c.imag)
    }
}

#[wasm_bindgen]
impl Complex {
    #[wasm_bindgen(constructor)]
    pub fn new(real: f64, imag: f64) -> Complex {
        Complex { real, imag }
    }

    #[wasm_bindgen(getter)]
    pub fn real(&self) -> f64 {
        self.real
    }

    #[wasm_bindgen(getter)]
    pub fn imag(&self) -> f64 {
        self.imag
    }
}

/// Parameters controlling the residual epicycle sum.
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidualParams {
    k_residuals: usize,
    residual_cap: f64,
    radius_scale: f64,
}

impl From<&ResidualParams> for RustResidualParams {
    fn from(p: &ResidualParams) -> RustResidualParams {
        RustResidualParams {
            k_residuals: p.k_residuals,
            residual_cap: p.residual_cap,
            radius_scale: p.radius_scale,
        }
    }
}

/// Parameters for height-field sampling.
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeightFieldParams {
    iterations: usize,
    min_magnitude: f64,
}

impl From<&HeightFieldParams> for RustHeightFieldParams {
    fn from(p: &HeightFieldParams) -> RustHeightFieldParams {
        RustHeightFieldParams {
            iterations: p.iterations,
            min_magnitude: p.min_magnitude,
        }
    }
}

#[wasm_bindgen]
impl HeightFieldParams {
    #[wasm_bindgen(constructor)]
    pub fn new(iterations: usize, min_magnitude: f64) -> HeightFieldParams {
        HeightFieldParams {
            iterations,
            min_magnitude,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    #[wasm_bindgen(getter)]
    pub fn min_magnitude(&self) -> f64 {
        self.min_magnitude
    }
}

/// Height-field sample for the Mandelbrot parameter plane.
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeightFieldSample {
    height: f64,
    gradient: Complex,
    z: Complex,
    w: Complex,
    magnitude: f64,
}

impl From<RustHeightFieldSample> for HeightFieldSample {
    fn from(sample: RustHeightFieldSample) -> Self {
        Self {
            height: sample.height,
            gradient: sample.gradient.into(),
            z: sample.z.into(),
            w: sample.w.into(),
            magnitude: sample.magnitude,
        }
    }
}

#[wasm_bindgen]
impl HeightFieldSample {
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f64 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn gradient(&self) -> Complex {
        self.gradient.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn z(&self) -> Complex {
        self.z.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn w(&self) -> Complex {
        self.w.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn magnitude(&self) -> f64 {
        self.magnitude
    }
}

/// Parameters for contour correction.
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContourControllerParams {
    correction_gain: f64,
    projection_epsilon: f64,
}

impl From<&ContourControllerParams> for RustContourControllerParams {
    fn from(p: &ContourControllerParams) -> RustContourControllerParams {
        RustContourControllerParams {
            correction_gain: p.correction_gain,
            projection_epsilon: p.projection_epsilon,
        }
    }
}

#[wasm_bindgen]
impl ContourControllerParams {
    #[wasm_bindgen(constructor)]
    pub fn new(correction_gain: f64, projection_epsilon: f64) -> ContourControllerParams {
        ContourControllerParams {
            correction_gain,
            projection_epsilon,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn correction_gain(&self) -> f64 {
        self.correction_gain
    }

    #[wasm_bindgen(getter)]
    pub fn projection_epsilon(&self) -> f64 {
        self.projection_epsilon
    }
}

/// Contour controller state.
#[wasm_bindgen]
pub struct ContourState {
    inner: RustContourState,
}

/// Output from contour controller step.
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContourStep {
    c: Complex,
    height: f64,
    height_error: f64,
    gradient: Complex,
    corrected_delta: Complex,
}

impl From<RustContourStep> for ContourStep {
    fn from(step: RustContourStep) -> Self {
        Self {
            c: step.c.into(),
            height: step.height,
            height_error: step.height_error,
            gradient: step.gradient.into(),
            corrected_delta: step.corrected_delta.into(),
        }
    }
}

#[wasm_bindgen]
impl ContourStep {
    #[wasm_bindgen(getter)]
    pub fn c(&self) -> Complex {
        self.c.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f64 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn height_error(&self) -> f64 {
        self.height_error
    }

    #[wasm_bindgen(getter)]
    pub fn gradient(&self) -> Complex {
        self.gradient.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn corrected_delta(&self) -> Complex {
        self.corrected_delta.clone()
    }
}

#[wasm_bindgen]
impl ResidualParams {
    #[wasm_bindgen(constructor)]
    pub fn new(k_residuals: usize, residual_cap: f64, radius_scale: f64) -> ResidualParams {
        ResidualParams {
            k_residuals,
            residual_cap,
            radius_scale,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn k_residuals(&self) -> usize {
        self.k_residuals
    }

    #[wasm_bindgen(getter)]
    pub fn residual_cap(&self) -> f64 {
        self.residual_cap
    }

    #[wasm_bindgen(getter)]
    pub fn radius_scale(&self) -> f64 {
        self.radius_scale
    }
}

/// Orbit state (carrier + residual phases).
#[wasm_bindgen]
pub struct OrbitState {
    inner: RustOrbitState,
}

#[wasm_bindgen]
impl OrbitState {
    /// Create a new orbit state.
    ///
    /// If you want deterministic residual phases, use `new_with_seed`.
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
    ) -> OrbitState {
        OrbitState {
            inner: RustOrbitState::new(
                lobe,
                sub_lobe,
                theta,
                omega,
                s,
                alpha,
                k_residuals,
                residual_omega_scale,
            ),
        }
    }

    /// Create a new orbit state with a fixed RNG seed (deterministic).
    #[wasm_bindgen]
    pub fn new_with_seed(
        lobe: u32,
        sub_lobe: u32,
        theta: f64,
        omega: f64,
        s: f64,
        alpha: f64,
        k_residuals: usize,
        residual_omega_scale: f64,
        seed: u64,
    ) -> OrbitState {
        OrbitState {
            inner: RustOrbitState::new_with_seed(
                lobe,
                sub_lobe,
                theta,
                omega,
                s,
                alpha,
                k_residuals,
                residual_omega_scale,
                seed,
            ),
        }
    }

    /// Advance phases by dt (seconds).
    #[wasm_bindgen]
    pub fn advance(&mut self, dt: f64) {
        self.inner.advance(dt);
    }

    /// Compute c(t) without advancing time.
    #[wasm_bindgen]
    pub fn synthesize(&self, residual_params: &ResidualParams, band_gates: Option<Vec<f64>>) -> Complex {
        rust_synthesize(&self.inner, RustResidualParams::from(residual_params), band_gates.as_deref()).into()
    }

    /// Advance by dt and return c(t). Mutates this OrbitState.
    #[wasm_bindgen]
    pub fn step(&mut self, dt: f64, residual_params: &ResidualParams, band_gates: Option<Vec<f64>>) -> Complex {
        rust_step(&mut self.inner, dt, RustResidualParams::from(residual_params), band_gates.as_deref()).into()
    }
}

/// Audio feature extractor.
#[wasm_bindgen]
pub struct FeatureExtractor {
    inner: RustFeatureExtractor,
}

#[wasm_bindgen]
impl ContourState {
    #[wasm_bindgen(constructor)]
    pub fn new(real: f64, imag: f64, field_params: Option<HeightFieldParams>) -> ContourState {
        let params = field_params.unwrap_or(HeightFieldParams {
            iterations: DEFAULT_HEIGHT_ITERATIONS,
            min_magnitude: DEFAULT_HEIGHT_MIN_MAGNITUDE,
        });
        ContourState {
            inner: RustContourState::new(RustComplex::new(real, imag), RustHeightFieldParams::from(&params)),
        }
    }

    #[wasm_bindgen]
    pub fn c(&self) -> Complex {
        self.inner.c.into()
    }

    #[wasm_bindgen]
    pub fn target_height(&self) -> f64 {
        self.inner.target_height
    }

    #[wasm_bindgen]
    pub fn set_target_height(&mut self, target_height: f64) {
        self.inner.set_target_height(target_height);
    }

    #[wasm_bindgen]
    pub fn step(
        &mut self,
        model_delta: &Complex,
        field_params: Option<HeightFieldParams>,
        controller_params: Option<ContourControllerParams>,
    ) -> ContourStep {
        let field_params = field_params.unwrap_or(HeightFieldParams {
            iterations: DEFAULT_HEIGHT_ITERATIONS,
            min_magnitude: DEFAULT_HEIGHT_MIN_MAGNITUDE,
        });
        let controller_params = controller_params.unwrap_or(ContourControllerParams {
            correction_gain: DEFAULT_CONTOUR_CORRECTION_GAIN,
            projection_epsilon: DEFAULT_CONTOUR_PROJECTION_EPSILON,
        });
        let step = self.inner.step(
            RustComplex::from(model_delta),
            RustHeightFieldParams::from(&field_params),
            RustContourControllerParams::from(&controller_params),
        );
        step.into()
    }
}

#[wasm_bindgen]
impl FeatureExtractor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        sr: usize,
        hop_length: usize,
        n_fft: usize,
        include_delta: bool,
        include_delta_delta: bool,
    ) -> FeatureExtractor {
        FeatureExtractor {
            inner: RustFeatureExtractor::new(sr, hop_length, n_fft, include_delta, include_delta_delta),
        }
    }

    #[wasm_bindgen]
    pub fn num_features_per_frame(&self) -> usize {
        self.inner.num_features_per_frame()
    }

    /// Extract windowed feature vectors.
    ///
    /// Returns a nested JS array: Vec<Vec<f64>>.
    #[wasm_bindgen]
    pub fn extract_windowed_features(&self, audio: Vec<f32>, window_frames: usize) -> Array {
        let windows = self.inner.extract_windowed_features(&audio[..], window_frames);
        let outer = Array::new();

        for w in windows {
            let inner = Array::new();
            for v in w {
                inner.push(&JsValue::from_f64(v));
            }
            outer.push(&inner);
        }

        outer
    }
}

/// Point on a lobe boundary.
#[wasm_bindgen]
pub fn lobe_point_at_angle(lobe: u32, sub_lobe: u32, theta: f64, s: f64) -> Complex {
    rust_lobe_point_at_angle(lobe, sub_lobe, theta, s).into()
}

/// Sample the height field at a point in the c-plane.
#[wasm_bindgen]
pub fn sample_height_field(real: f64, imag: f64, params: Option<HeightFieldParams>) -> HeightFieldSample {
    let params = params.unwrap_or(HeightFieldParams {
        iterations: DEFAULT_HEIGHT_ITERATIONS,
        min_magnitude: DEFAULT_HEIGHT_MIN_MAGNITUDE,
    });
    rust_sample_height_field(RustComplex::new(real, imag), RustHeightFieldParams::from(&params)).into()
}

/// Apply contour correction to a proposed delta.
#[wasm_bindgen]
pub fn contour_correct_delta(
    model_delta: &Complex,
    gradient: &Complex,
    height_error: f64,
    params: Option<ContourControllerParams>,
) -> Complex {
    let params = params.unwrap_or(ContourControllerParams {
        correction_gain: DEFAULT_CONTOUR_CORRECTION_GAIN,
        projection_epsilon: DEFAULT_CONTOUR_PROJECTION_EPSILON,
    });
    rust_contour_correct_delta(
        RustComplex::from(model_delta),
        RustComplex::from(gradient),
        height_error,
        RustContourControllerParams::from(&params),
    )
    .into()
}

/// Shared runtime constants for parity checks between backend and frontend.
#[wasm_bindgen]
pub fn sample_rate() -> usize {
    SAMPLE_RATE
}

#[wasm_bindgen]
pub fn hop_length() -> usize {
    HOP_LENGTH
}

#[wasm_bindgen]
pub fn n_fft() -> usize {
    N_FFT
}

#[wasm_bindgen]
pub fn window_frames() -> usize {
    WINDOW_FRAMES
}

#[wasm_bindgen]
pub fn default_k_residuals() -> usize {
    DEFAULT_K_RESIDUALS
}

#[wasm_bindgen]
pub fn default_residual_cap() -> f64 {
    DEFAULT_RESIDUAL_CAP
}

#[wasm_bindgen]
pub fn default_residual_omega_scale() -> f64 {
    DEFAULT_RESIDUAL_OMEGA_SCALE
}

#[wasm_bindgen]
pub fn default_base_omega() -> f64 {
    DEFAULT_BASE_OMEGA
}

#[wasm_bindgen]
pub fn default_orbit_seed() -> u64 {
    DEFAULT_ORBIT_SEED
}

#[wasm_bindgen]
pub fn default_height_iterations() -> usize {
    DEFAULT_HEIGHT_ITERATIONS
}

#[wasm_bindgen]
pub fn default_height_min_magnitude() -> f64 {
    DEFAULT_HEIGHT_MIN_MAGNITUDE
}

#[wasm_bindgen]
pub fn default_contour_correction_gain() -> f64 {
    DEFAULT_CONTOUR_CORRECTION_GAIN
}

#[wasm_bindgen]
pub fn default_contour_projection_epsilon() -> f64 {
    DEFAULT_CONTOUR_PROJECTION_EPSILON
}
