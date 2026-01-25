//! WebAssembly bindings for runtime-core
//!
//! This module exposes the shared runtime to JavaScript via
//! `wasm-bindgen`. The API is intentionally kept close to the Python
//! bindings so both front-end and back-end can call the same logic.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use js_sys::{Array, Function, Reflect};
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
        rust_step(&mut self.inner, dt, RustResidualParams::from(residual_params), band_gates.as_deref(), None, 0.0, None, None).into()
    }

    /// Advance with transient and integrator options: h in [0,1], optional d_star, max_step.
    #[wasm_bindgen]
    pub fn step_advanced(&mut self, dt: f64, residual_params: &ResidualParams, band_gates: Option<Vec<f64>>, h: f64, d_star: Option<f64>, max_step: Option<f64>, distance_field: Option<JsValue>) -> Complex {
        // If JS provides a DistanceField wrapper (JsValue), use a JS-aware codepath that
        // calls the object's methods for sampling/gradient/velocity instead of attempting an
        // ownership transfer. This avoids runtime ownership/borrow panics that were observed
        // when the JS wrapper was moved while it was borrowed elsewhere.
        if let Some(js_df) = distance_field.as_ref() {
            // JS-backed execution path
            return self.step_advanced_with_js_df(dt, RustResidualParams::from(residual_params), band_gates.as_deref(), h, d_star, max_step, js_df).into();
        }

        // Fallback: no DF provided at all
        rust_step(&mut self.inner, dt, RustResidualParams::from(residual_params), band_gates.as_deref(), None, h, d_star, max_step).into()
    }

    // Internal helper: run the stepping path using a JS distance field object (calls into its
    // `get_velocity_scale`, `sample_bilinear`, and `gradient` methods via `Reflect`).
    fn step_advanced_with_js_df(&mut self, dt: f64, residual_params: RustResidualParams, band_gates: Option<&[f64]>, h: f64, d_star: Option<f64>, max_step: Option<f64>, js_df: &JsValue) -> RustComplex {
        // Helper closures to call into the JS DF safely with fallbacks
        let call_get_velocity_scale = |c: RustComplex| -> f64 {
            // Default velocity scale is 1.0
            if let Ok(func) = Reflect::get(js_df, &JsValue::from_str("get_velocity_scale")) {
                if func.is_function() {
                    let f: Function = func.unchecked_into();
                    let r = f.call2(js_df, &JsValue::from_f64(c.real), &JsValue::from_f64(c.imag));
                    if let Ok(rv) = r {
                        return rv.as_f64().unwrap_or(1.0);
                    }
                }
            }
            1.0
        };

        let call_sample_bilinear = |c: RustComplex| -> f64 {
            if let Ok(func) = Reflect::get(js_df, &JsValue::from_str("sample_bilinear")) {
                if func.is_function() {
                    let f: Function = func.unchecked_into();
                    let r = f.call2(js_df, &JsValue::from_f64(c.real), &JsValue::from_f64(c.imag));
                    if let Ok(rv) = r {
                        return rv.as_f64().unwrap_or(1.0);
                    }
                }
            }
            1.0
        };

        let call_gradient = |c: RustComplex| -> (f64, f64) {
            if let Ok(func) = Reflect::get(js_df, &JsValue::from_str("gradient")) {
                if func.is_function() {
                    let f: Function = func.unchecked_into();
                    if let Ok(rv) = f.call2(js_df, &JsValue::from_f64(c.real), &JsValue::from_f64(c.imag)) {
                        let arr = Array::from(&rv);
                        let gx = arr.get(0).as_f64().unwrap_or(0.0);
                        let gy = arr.get(1).as_f64().unwrap_or(0.0);
                        return (gx, gy);
                    }
                }
            }
            (0.0, 0.0)
        };

        // Get effective dt via velocity scale
        let c_current = crate::controller::synthesize(&self.inner, residual_params, band_gates);
        let velocity_scale = call_get_velocity_scale(c_current) as f64;
        let effective_dt = dt * velocity_scale;

        // Sample distance and gradient via JS DF BEFORE advancing to avoid reentrant
        // calls into JS that might trigger Rust-side borrows while we hold a mutable
        // borrow during `advance()`.
        let d = call_sample_bilinear(c_current);
        let (gx, gy) = call_gradient(c_current);
        let grad_norm = (gx * gx + gy * gy).sqrt();

        // Advance and synthesize proposal
        self.inner.advance(effective_dt);
        let c_proposed = crate::controller::synthesize(&self.inner, residual_params, band_gates);

        // If max_step > 0, apply clamping similar to original step_logic
        let max_s = max_step.unwrap_or(0.5_f64);

        // If JS DF provided, perform contour-biased integrator using JS calls
        let u_real = c_proposed.real - c_current.real;
        let u_imag = c_proposed.imag - c_current.imag;

        // d_star or fallback
        let d_s = d_star.unwrap_or(0.0);

        if grad_norm <= 1e-12 {
            let u_mag = (u_real * u_real + u_imag * u_imag).sqrt();
            let scale = if u_mag > max_s { max_s / u_mag } else { 1.0 };
            return RustComplex::new(c_current.real + u_real * scale, c_current.imag + u_imag * scale);
        }

        let nx = gx / grad_norm;
        let ny = gy / grad_norm;
        let tx = -gy / grad_norm;
        let ty = gx / grad_norm;

        let proj_t = u_real * tx + u_imag * ty;
        let proj_n = u_real * nx + u_imag * ny;

        let normal_scale_no_hit = 0.05_f64;
        let normal_scale_hit = 1.0_f64;
        let tangential_scale = 1.0_f64;
        let normal_scale = normal_scale_no_hit + (normal_scale_hit - normal_scale_no_hit) * (h.clamp(0.0, 1.0) as f64);
        let servo_gain = 0.2_f64;
        let servo = servo_gain * (d_s - d);

        let mut dx = tx * (proj_t * tangential_scale) + nx * (proj_n * normal_scale + servo);
        let mut dy = ty * (proj_t * tangential_scale) + ny * (proj_n * normal_scale + servo);

        let mag = (dx * dx + dy * dy).sqrt();
        if mag > max_s && mag > 0.0 {
            let s = max_s / mag;
            dx *= s;
            dy *= s;
        }

        RustComplex::new(c_current.real + dx, c_current.imag + dy)
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
/// Distance field wrapper and helpers for WASM
#[wasm_bindgen]
pub struct DistanceField {
    inner: crate::distance_field::DistanceField,
}

#[wasm_bindgen]
impl DistanceField {
    #[wasm_bindgen(constructor)]
    pub fn new(field: Vec<f32>, resolution: usize, real_min: f64, real_max: f64, imag_min: f64, imag_max: f64, max_distance: f64, slowdown_threshold: f64) -> DistanceField {
        let real_range = (real_min, real_max);
        let imag_range = (imag_min, imag_max);
        DistanceField { inner: crate::distance_field::DistanceField::new(field, resolution, real_range, imag_range, max_distance, slowdown_threshold) }
    }

    #[wasm_bindgen]
    pub fn lookup(&self, real: f64, imag: f64) -> f32 {
        self.inner.lookup(RustComplex::new(real, imag))
    }

    #[wasm_bindgen]
    pub fn sample_bilinear(&self, real: f64, imag: f64) -> f32 {
        self.inner.sample_bilinear(RustComplex::new(real, imag))
    }

    #[wasm_bindgen]
    pub fn gradient(&self, real: f64, imag: f64) -> js_sys::Array {
        let (gx, gy) = self.inner.gradient(RustComplex::new(real, imag));
        let arr = js_sys::Array::new();
        arr.push(&JsValue::from_f64(gx));
        arr.push(&JsValue::from_f64(gy));
        arr
    }
}

/// Contour-biased integrator exposed to JS
#[wasm_bindgen]
pub fn contour_biased_step(real: f64, imag: f64, u_real: f64, u_imag: f64, h: f64, d_star: f64, max_step: f64, distance_field: Option<JsValue>) -> Complex {
    if let Some(js_df) = distance_field.as_ref() {
        return contour_biased_step_with_js_df(RustComplex::new(real, imag), u_real, u_imag, h, js_df, d_star, max_step).into();
    }
    crate::controller::contour_biased_step(RustComplex::new(real, imag), u_real, u_imag, h, None, d_star, max_step).into()
}

// JS-aware variant that calls the DF methods via Reflect/Function.
fn contour_biased_step_with_js_df(c: RustComplex, u_real: f64, u_imag: f64, h: f64, js_df: &JsValue, d_star: f64, max_step: f64) -> RustComplex {
    // helpers mirroring the ones in step_advanced_with_js_df
    let call_sample_bilinear = |c: RustComplex| -> f64 {
        if let Ok(func) = Reflect::get(js_df, &JsValue::from_str("sample_bilinear")) {
            if func.is_function() {
                let f: Function = func.unchecked_into();
                let r = f.call2(js_df, &JsValue::from_f64(c.real), &JsValue::from_f64(c.imag));
                if let Ok(rv) = r {
                    return rv.as_f64().unwrap_or(1.0);
                }
            }
        }
        1.0
    };

    let call_gradient = |c: RustComplex| -> (f64, f64) {
        if let Ok(func) = Reflect::get(js_df, &JsValue::from_str("gradient")) {
            if func.is_function() {
                let f: Function = func.unchecked_into();
                if let Ok(rv) = f.call2(js_df, &JsValue::from_f64(c.real), &JsValue::from_f64(c.imag)) {
                    let arr = Array::from(&rv);
                    let gx = arr.get(0).as_f64().unwrap_or(0.0);
                    let gy = arr.get(1).as_f64().unwrap_or(0.0);
                    return (gx, gy);
                }
            }
        }
        (0.0, 0.0)
    };

    let d = call_sample_bilinear(c);
    let (gx, gy) = call_gradient(c);
    let grad_norm = (gx * gx + gy * gy).sqrt();

    let u_mag = (u_real * u_real + u_imag * u_imag).sqrt();
    if grad_norm <= 1e-12 {
        let scale = if u_mag > max_step { max_step / u_mag } else { 1.0 };
        return RustComplex::new(c.real + u_real * scale, c.imag + u_imag * scale);
    }

    let nx = gx / grad_norm;
    let ny = gy / grad_norm;
    let tx = -gy / grad_norm;
    let ty = gx / grad_norm;

    let proj_t = u_real * tx + u_imag * ty;
    let proj_n = u_real * nx + u_imag * ny;

    let normal_scale_no_hit = 0.05_f64;
    let normal_scale_hit = 1.0_f64;
    let tangential_scale = 1.0_f64;
    let normal_scale = normal_scale_no_hit + (normal_scale_hit - normal_scale_no_hit) * (h.clamp(0.0, 1.0) as f64);
    let servo_gain = 0.2_f64;
    let servo = servo_gain * (d_star - d);

    let mut dx = tx * (proj_t * tangential_scale) + nx * (proj_n * normal_scale + servo);
    let mut dy = ty * (proj_t * tangential_scale) + ny * (proj_n * normal_scale + servo);

    let mag = (dx * dx + dy * dy).sqrt();
    if mag > max_step && mag > 0.0 {
        let s = max_step / mag;
        dx *= s;
        dy *= s;
    }

    RustComplex::new(c.real + dx, c.imag + dy)
}