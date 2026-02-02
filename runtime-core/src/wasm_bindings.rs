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
use crate::features::FeatureExtractor as RustFeatureExtractor;
use crate::geometry::{lobe_point_at_angle as rust_lobe_point_at_angle, Complex as RustComplex};
use crate::visual_metrics::{compute_runtime_metrics, RuntimeVisualMetrics as RustRuntimeVisualMetrics};

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

/// Runtime visual metrics computed in Rust.
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeVisualMetrics {
    edge_density: f64,
    color_uniformity: f64,
    brightness_mean: f64,
    brightness_std: f64,
    brightness_range: f64,
    mandelbrot_membership: bool,
}

impl From<RustRuntimeVisualMetrics> for RuntimeVisualMetrics {
    fn from(metrics: RustRuntimeVisualMetrics) -> Self {
        Self {
            edge_density: metrics.edge_density,
            color_uniformity: metrics.color_uniformity,
            brightness_mean: metrics.brightness_mean,
            brightness_std: metrics.brightness_std,
            brightness_range: metrics.brightness_range,
            mandelbrot_membership: metrics.mandelbrot_membership,
        }
    }
}

#[wasm_bindgen]
impl RuntimeVisualMetrics {
    #[wasm_bindgen(getter)]
    pub fn edge_density(&self) -> f64 {
        self.edge_density
    }

    #[wasm_bindgen(getter)]
    pub fn color_uniformity(&self) -> f64 {
        self.color_uniformity
    }

    #[wasm_bindgen(getter)]
    pub fn brightness_mean(&self) -> f64 {
        self.brightness_mean
    }

    #[wasm_bindgen(getter)]
    pub fn brightness_std(&self) -> f64 {
        self.brightness_std
    }

    #[wasm_bindgen(getter)]
    pub fn brightness_range(&self) -> f64 {
        self.brightness_range
    }

    #[wasm_bindgen(getter)]
    pub fn mandelbrot_membership(&self) -> bool {
        self.mandelbrot_membership
    }
}

#[wasm_bindgen]
pub fn compute_runtime_visual_metrics(
    image: Vec<f64>,
    width: usize,
    height: usize,
    channels: usize,
    c_real: f64,
    c_imag: f64,
    max_iter: usize,
) -> Result<RuntimeVisualMetrics, JsValue> {
    let metrics = compute_runtime_metrics(
        &image,
        width,
        height,
        channels,
        RustComplex::new(c_real, c_imag),
        max_iter,
    )
    .map_err(|message| JsValue::from_str(message))?;
    Ok(metrics.into())
}

/// Point on a lobe boundary.
#[wasm_bindgen]
pub fn lobe_point_at_angle(lobe: u32, sub_lobe: u32, theta: f64, s: f64) -> Complex {
    rust_lobe_point_at_angle(lobe, sub_lobe, theta, s).into()
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

/// Load a precomputed distance field (.npy) and optional .json metadata from
/// the host file system (when running under a host that provides a file API).
#[wasm_bindgen]
pub fn load_distance_field(path: &str) -> Result<(), JsValue> {
    crate::distance_field::load_distance_field(path).map_err(|e| JsValue::from_str(&e))
}

/// Set an in-memory distance field from a flat buffer with explicit
/// dimensions and bounding box. The flat buffer is row-major.
#[wasm_bindgen]
pub fn set_distance_field(flat: Vec<f32>, rows: usize, cols: usize, xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Result<(), JsValue> {
    crate::distance_field::set_distance_field_from_vec(flat, rows, cols, xmin, xmax, ymin, ymax)
        .map_err(|e| JsValue::from_str(&e))
}

/// Sample the currently-loaded distance field at arrays of x and y coords.
#[wasm_bindgen]
pub fn sample_distance_field(x_coords: Vec<f64>, y_coords: Vec<f64>) -> Result<Vec<f32>, JsValue> {
    crate::distance_field::sample_distance_field(&x_coords, &y_coords).map_err(|e| JsValue::from_str(&e))
}

/// Load a built-in distance field (embedded at compile time) and return its
/// metadata as a JS array [rows, cols, xmin, xmax, ymin, ymax].
#[wasm_bindgen]
pub fn get_builtin_distance_field(name: &str) -> Result<Array, JsValue> {
    match crate::distance_field::load_builtin_distance_field(name) {
        Ok((rows, cols, xmin, xmax, ymin, ymax)) => {
            let arr = Array::new();
            arr.push(&JsValue::from_f64(rows as f64));
            arr.push(&JsValue::from_f64(cols as f64));
            arr.push(&JsValue::from_f64(xmin));
            arr.push(&JsValue::from_f64(xmax));
            arr.push(&JsValue::from_f64(ymin));
            arr.push(&JsValue::from_f64(ymax));
            Ok(arr)
        }
        Err(e) => Err(JsValue::from_str(&e)),
    }
}
