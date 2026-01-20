//! WebAssembly bindings for runtime‑core
//!
//! This module exposes Rust structs and functions to JavaScript via
//! `wasm‑bindgen`.  It mirrors the API of the Python bindings so
//! that browser code can interact with the same orbit and feature
//! logic as the backend.

use wasm_bindgen::prelude::*;

use crate::controller::{OrbitState as RustOrbitState, ResidualParams as RustResidualParams, step as rust_step, synthesize as rust_synthesize};
use crate::features::FeatureExtractor as RustFeatureExtractor;
use crate::geometry::{lobe_point_at_angle as rust_lobe_point_at_angle, Complex as RustComplex};

/// Simple JavaScript representation of a complex number.  This
/// mirrors the structure exposed in Python.  When returned to
/// JavaScript it becomes an object with `real` and `imag` fields.
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl From<RustComplex> for Complex {
    fn from(c: RustComplex) -> Self {
        Self { real: c.real, imag: c.imag }
    }
}

/// Residual parameters exposed to JavaScript.
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidualParams {
    pub k_residuals: usize,
    pub residual_cap: f64,
    pub radius_scale: f64,
}

impl From<RustResidualParams> for ResidualParams {
    fn from(p: RustResidualParams) -> Self {
        Self {
            k_residuals: p.k_residuals,
            residual_cap: p.residual_cap,
            radius_scale: p.radius_scale,
        }
    }
}

impl From<ResidualParams> for RustResidualParams {
    fn from(p: ResidualParams) -> RustResidualParams {
        RustResidualParams {
            k_residuals: p.k_residuals,
            residual_cap: p.residual_cap,
            radius_scale: p.radius_scale,
        }
    }
}

/// Orbit state accessible from JavaScript.  This struct wraps the
/// internal Rust state and exposes methods to advance and synthesise
/// the orbit.
#[wasm_bindgen]
pub struct OrbitState {
    inner: RustOrbitState,
}

#[wasm_bindgen]
impl OrbitState {
    /// Construct a new orbit state with random residual phases.
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

    /// Advance time by dt.  Mutates the internal state but does not
    /// return c(t).  Call `synthesize` or `step` to compute the next
    /// complex value.
    pub fn advance(&mut self, dt: f64) {
        self.inner.advance(dt);
    }

    /// Synthesize c(t) without advancing time.  Band gates are
    /// provided as an array of floats or omitted.  Returns an object
    /// with `real` and `imag` fields.
    #[wasm_bindgen]
    pub fn synthesize(&self, residual_params: &ResidualParams, band_gates: Option<Vec<f64>>) -> Complex {
        rust_synthesize(&self.inner, residual_params.clone().into(), band_gates.as_deref()).into()
    }

    /// Advance time by dt and synthesise c(t) in one call.  The band
    /// gates are applied to the residuals.  Returns a tuple
    /// `(c, new_state)` where `c` has `real` and `imag` fields and
    /// `new_state` is a new `OrbitState` reflecting the updated
    /// phases.
    #[wasm_bindgen]
    pub fn step(&self, dt: f64, residual_params: &ResidualParams, band_gates: Option<Vec<f64>>) -> JsValue {
        // Clone the state so that we do not mutate the original
        let mut new_state = self.inner.clone();
        let c = rust_step(&mut new_state, dt, residual_params.clone().into(), band_gates.as_deref());
        // Wrap into JS objects
        serde_wasm_bindgen::to_value(&serde_json::json!({
            "c": Complex::from(c),
            "newState": OrbitState { inner: new_state }
        })).unwrap()
    }
}

/// Feature extractor exposed to JavaScript.  Note that processing
/// large audio arrays in WebAssembly can be expensive; consider
/// downsampling on the JavaScript side before calling into Rust.
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

    /// Extract a sequence of windowed features from a Float32Array.  The
    /// result is returned as an array of arrays (nested JS arrays).  If
    /// you need high performance consider streaming audio to a Web
    /// Worker and processing it there.
    #[wasm_bindgen]
    pub fn extract_windowed_features(&self, audio: Vec<f32>, window_frames: usize) -> JsValue {
        let windows = self.inner.extract_windowed_features(&audio[..], window_frames);
        serde_wasm_bindgen::to_value(&windows).unwrap()
    }
}

/// Compute a point on the Mandelbrot lobe boundary.  Returns an
/// object with `real` and `imag` fields.  This is a free function
/// analogous to the Python binding.
#[wasm_bindgen]
pub fn lobe_point_at_angle(lobe: u32, sub_lobe: u32, theta: f64, s: f64) -> Complex {
    rust_lobe_point_at_angle(lobe, sub_lobe, theta, s).into()
}