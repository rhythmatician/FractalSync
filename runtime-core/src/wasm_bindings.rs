//! WebAssembly bindings for runtime-core
//!
//! This module exposes the shared runtime to JavaScript via
//! `wasm-bindgen`. The API is intentionally kept close to the Python
//! bindings so both front-end and back-end can call the same logic.

use wasm_bindgen::prelude::*;
use js_sys::Array;
use serde::{Deserialize, Serialize};
use num_complex::Complex64 as RustComplex;

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
use crate::geometry::{lobe_point_at_angle as rust_lobe_point_at_angle};
use crate::visual_metrics::{compute_runtime_metrics, RuntimeVisualMetrics as RustRuntimeVisualMetrics};
use crate::controls::{
    ControlsV2 as RustControlsV2,
    MotionControls as RustMotionControls,
    JuliaViewControls as RustJuliaViewControls,
    JuliaViewState as RustJuliaViewState,
    ColorIntent as RustColorIntent,
    Harmony as RustHarmony,
    CONTROLS_VERSION,
};
use crate::manifold::{
    ManifoldConfig as RustManifoldConfig,
    EnergyInfo as RustEnergyInfo,
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
    integrate_step as rust_integrate_step,
};

/// Manifold configuration for WASM (issue #106).
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ManifoldConfig {
    d_ref: f64,
    epsilon: f64,
    lambda_sq: f64,
    kappa: f64,
    mu: f64,
}

impl From<&ManifoldConfig> for RustManifoldConfig {
    fn from(c: &ManifoldConfig) -> RustManifoldConfig {
        RustManifoldConfig {
            d_ref: c.d_ref,
            epsilon: c.epsilon,
            lambda_sq: c.lambda_sq,
            kappa: c.kappa,
            mu: c.mu,
        }
    }
}

#[wasm_bindgen]
impl ManifoldConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(d_ref: f64, epsilon: f64, lambda_sq: f64, kappa: f64, mu: f64) -> ManifoldConfig {
        ManifoldConfig {
            d_ref,
            epsilon,
            lambda_sq,
            kappa,
            mu,
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
    #[wasm_bindgen(getter)]
    pub fn mu(&self) -> f64 {
        self.mu
    }
}

/// A complex number (Julia parameter c = a + bi).
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Complex {
    real: f64,
    imag: f64,
}

impl From<RustComplex> for Complex {
    fn from(c: RustComplex) -> Self {
        Self { real: c.re, imag: c.im }
    }
}

impl From<Complex> for RustComplex {
    fn from(c: Complex) -> Self {
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
    c: Complex,
    max_iter: usize,
) -> Result<RuntimeVisualMetrics, JsValue> {
    let metrics = compute_runtime_metrics(
        &image,
        width,
        height,
        channels,
        RustComplex::from(c),
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

/// Feature-extraction contract version (ADR 0001).
#[wasm_bindgen]
pub fn feature_version() -> &'static str {
    crate::features::FEATURE_VERSION
}

/// Analysis-pipeline contract version (issue #93): versions HOW audio
/// reaches the extractor (resampling ownership, hop scheduling, epoch
/// semantics) — distinct from the feature FORMULA version.
#[wasm_bindgen]
pub fn analysis_pipeline_version() -> &'static str {
    crate::timebase::ANALYSIS_PIPELINE_VERSION
}

/// Pinned normalization epsilon shared by trainer and browser.
#[wasm_bindgen]
pub fn norm_eps() -> f64 {
    crate::features::NORM_EPS
}

/// Load a precomputed distance field (.npy) and optional .json metadata from
/// the host file system (when running under a host that provides a file API).
///
/// Note: This operation is not currently supported in the WASM runtime.
/// Use `set_distance_field` to provide an in-memory distance field, or
/// `get_builtin_distance_field` to use an embedded distance field instead.
#[wasm_bindgen]
pub fn load_distance_field(_: &str) -> Result<(), JsValue> {
    Err(JsValue::from_str(
        "load_distance_field is not supported in the WASM runtime; use set_distance_field or get_builtin_distance_field instead.",
    ))
}

/// Set an in-memory distance field from a flat buffer with explicit
/// dimensions and bounding box. The flat buffer is row-major.
#[wasm_bindgen]
pub fn set_distance_field(flat: Vec<f32>, rows: usize, cols: usize, xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Result<(), JsValue> {
    crate::distance_field::set_distance_field_from_vec(flat, rows, cols, xmin, xmax, ymin, ymax)
        .map_err(|e| JsValue::from_str(&e))
}

/// Sample the currently-loaded distance field at arrays of complex coords.
#[wasm_bindgen]
pub fn sample_distance_field(coords: Vec<Complex>) -> Result<Vec<f32>, JsValue> {
    let points: Vec<RustComplex> = coords.into_iter().map(RustComplex::from).collect();
    crate::distance_field::sample_distance_field(&points).map_err(|e| JsValue::from_str(&e))
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

/// Set the mip pyramid (the Map's multi-scale minimaps) from host-provided
/// flat planes. `f_flat` and `s_flat` are concatenated per-level row-major
/// f32 planes; widths/heights give each level's dimensions.
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
        return Err(JsValue::from_str(format!(
            "expected {} values, got {}",
            total,
            f_flat.len()
        ).as_str()));
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
    let mut pyr = crate::minimap::MipPyramid::from_levels(
        split(&s_flat),
        widths.iter().map(|&w| w as usize).collect(),
        heights.iter().map(|&h| h as usize).collect(),
        re_min,
        re_max,
        im_min,
        im_max,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    pyr.fields = [split(&f_flat), split(&s_flat)];
    crate::minimap::set_pyramid(pyr).map_err(|e| JsValue::from_str(&e))
}

/// The Player's full observation at c: 4×81 greys + 8 slope values = 332.
#[wasm_bindgen]
pub fn player_observation(real: f64, imag: f64) -> Result<Vec<f32>, JsValue> {
    crate::minimap::with_pyramid(|pyr| {
        let pyr = pyr.ok_or_else(|| JsValue::from_str("mip pyramid not loaded"))?;
        let c = RustComplex::new(real, imag);
        pyr.player_observation(c)
            .ok_or_else(|| JsValue::from_str("c outside map extent"))
    })
}

/// Slope of the shore-proximity field at c on a mip level.
#[wasm_bindgen]
pub fn minimap_slope(real: f64, imag: f64, level: usize) -> Result<Array, JsValue> {
    crate::minimap::with_pyramid(|pyr| {
        let pyr = pyr.ok_or_else(|| JsValue::from_str("mip pyramid not loaded"))?;
        let c = RustComplex::new(real, imag);
        let (gx, gy) = pyr
            .slope(c, level)
            .ok_or_else(|| JsValue::from_str("c outside map extent"))?;
        let arr = Array::new();
        arr.push(&JsValue::from_f64(gx));
        arr.push(&JsValue::from_f64(gy));
        Ok(arr)
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
) -> Result<Array, JsValue> {
    let (nr, ni) = crate::minimap::contour_biased_step(
        real, imag, u_real, u_imag, h, d_star, max_step, level,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(nr));
    arr.push(&JsValue::from_f64(ni));
    Ok(arr)
}

// ---------------------------------------------------------------------------
// Manifold physics (issue #106) — WASM surface.
//
// Mirrors the Python bindings so the browser can run the same differential
// geometry as the trainer. Rust remains canonical under ADR 0001.
// ---------------------------------------------------------------------------

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
pub fn manifold_scale_gradient(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let (gx, gy) = rust_scale_gradient(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(gx));
    arr.push(&JsValue::from_f64(gy));
    Ok(arr)
}

/// Scale Hessian [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]].
#[wasm_bindgen]
pub fn manifold_scale_hessian(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let h = rust_scale_hessian(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let outer = Array::new();
    let row0 = Array::new();
    row0.push(&JsValue::from_f64(h[0][0]));
    row0.push(&JsValue::from_f64(h[0][1]));
    let row1 = Array::new();
    row1.push(&JsValue::from_f64(h[1][0]));
    row1.push(&JsValue::from_f64(h[1][1]));
    outer.push(&row0);
    outer.push(&row1);
    Ok(outer)
}

/// Scale-relative induced metric
/// G(c) = rho^-2 I + lambda^2 * grad_sigma * grad_sigma^T.
/// Returns [[g11, g12], [g12, g22]].
#[wasm_bindgen]
pub fn manifold_induced_metric(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let g = rust_induced_metric(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let outer = Array::new();
    let row0 = Array::new();
    row0.push(&JsValue::from_f64(g[0][0]));
    row0.push(&JsValue::from_f64(g[0][1]));
    let row1 = Array::new();
    row1.push(&JsValue::from_f64(g[1][0]));
    row1.push(&JsValue::from_f64(g[1][1]));
    outer.push(&row0);
    outer.push(&row1);
    Ok(outer)
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

/// Total mechanical energy E = K + U_sigma + U_wall.
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

/// Christoffel symbols Gamma^i_jk. Returns [[[g00,g01],[g10,g11]],[[g20,g21],[g30,g31]]].
#[wasm_bindgen]
pub fn manifold_christoffel_symbols(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let g = rust_christoffel_symbols(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let outer = Array::new();
    for i in 0..2 {
        let mid = Array::new();
        for j in 0..2 {
            let inner = Array::new();
            inner.push(&JsValue::from_f64(g[i][j][0]));
            inner.push(&JsValue::from_f64(g[i][j][1]));
            mid.push(&inner);
        }
        outer.push(&mid);
    }
    Ok(outer)
}

/// Geodesic acceleration term: Gamma^i_jk v^j v^k. Returns [ax, ay].
#[wasm_bindgen]
pub fn manifold_geodesic_acceleration(
    vx: f64,
    vy: f64,
    real: f64,
    imag: f64,
    config: &ManifoldConfig,
) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let (ax, ay) = rust_geodesic_acceleration((vx, vy), c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(ax));
    arr.push(&JsValue::from_f64(ay));
    Ok(arr)
}

/// Generalized potential force covector: Q_potential = -grad U = -kappa grad sigma.
/// Returns [Qx, Qy]. This is a covector, not a coordinate acceleration; convert
/// with `manifold_apply_generalized_force`.
#[wasm_bindgen]
pub fn manifold_potential_force(real: f64, imag: f64, config: &ManifoldConfig) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let (fx, fy) = rust_potential_force(c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(fx));
    arr.push(&JsValue::from_f64(fy));
    Ok(arr)
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
) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let (ax, ay) = rust_apply_generalized_force((qx, qy), c, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(ax));
    arr.push(&JsValue::from_f64(ay));
    Ok(arr)
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
) -> Result<Array, JsValue> {
    let c = RustComplex::new(real, imag);
    let (qx, qy) = rust_drag_force((vx, vy), c, beta, &config.into()).map_err(|e| JsValue::from_str(&e))?;
    let arr = Array::new();
    arr.push(&JsValue::from_f64(qx));
    arr.push(&JsValue::from_f64(qy));
    Ok(arr)
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
) -> Result<Array, JsValue> {
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
    let arr = Array::new();
    arr.push(&JsValue::from_f64(c_new.re));
    arr.push(&JsValue::from_f64(c_new.im));
    arr.push(&JsValue::from_f64(v_new.0));
    arr.push(&JsValue::from_f64(v_new.1));
    arr.push(&JsValue::from_f64(info.kinetic));
    arr.push(&JsValue::from_f64(info.potential));
    arr.push(&JsValue::from_f64(info.total));
    arr.push(&JsValue::from_f64(info.delta_total));
    arr.push(&JsValue::from_f64(info.delta_kinetic));
    Ok(arr)
}
