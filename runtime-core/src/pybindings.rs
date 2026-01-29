//! Python bindings for runtime‑core
//!
//! This module exposes the Rust runtime via PyO3.  It defines
//! Python classes and functions that mirror the structs and free
//! functions in `geometry`, `controller` and `features`.  These
//! bindings allow the Python backend to import a compiled
//! `runtime_core` module and call the shared logic directly.

use pyo3::prelude::*;

use crate::controller::{
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
    step as rust_step,
    synthesize as rust_synthesize,
};
use crate::features::FeatureExtractor as RustFeatureExtractor;
use crate::geometry::{lobe_point_at_angle as rust_lobe_point_at_angle, Complex as RustComplex};
use crate::minimap::{Minimap as RustMinimap, MINIMAP_MIP_LEVELS, MINIMAP_PATCH_K};
use crate::step_controller::{
    ControllerContext as RustControllerContext,
    StepController as RustStepController,
    StepResult as RustStepResult,
    CONTEXT_LEN,
    mip_for_delta,
};
use crate::visual_metrics::{compute_runtime_metrics, RuntimeVisualMetrics as RustRuntimeVisualMetrics};

/// Helper struct to expose a complex number to Python as a tuple.
#[pyclass]
#[derive(Clone, Debug)]
pub struct Complex {
    #[pyo3(get)]
    pub real: f64,
    #[pyo3(get)]
    pub imag: f64,
}

impl From<RustComplex> for Complex {
    fn from(c: RustComplex) -> Self {
        Self { real: c.real, imag: c.imag }
    }
}

/// Python wrapper for `ResidualParams`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ResidualParams {
    #[pyo3(get, set)]
    pub k_residuals: usize,
    #[pyo3(get, set)]
    pub residual_cap: f64,
    #[pyo3(get, set)]
    pub radius_scale: f64,
}

#[pymethods]
impl ResidualParams {
    #[new]
    #[pyo3(signature = (k_residuals=DEFAULT_K_RESIDUALS, residual_cap=DEFAULT_RESIDUAL_CAP, radius_scale=1.0))]
    fn py_new(
        k_residuals: usize,
        residual_cap: f64,
        radius_scale: f64,
    ) -> Self {
        Self {
            k_residuals,
            residual_cap,
            radius_scale,
        }
    }
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

/// Python wrapper for the orbit state.
#[pyclass]
#[derive(Clone, Debug)]
pub struct OrbitState {
    inner: RustOrbitState,
}

#[pymethods]
impl OrbitState {
    #[new]
    #[pyo3(signature = (lobe, sub_lobe, theta, omega, s, alpha, k_residuals, residual_omega_scale, seed=None))]
    fn py_new(
        lobe: u32,
        sub_lobe: u32,
        theta: f64,
        omega: f64,
        s: f64,
        alpha: f64,
        k_residuals: usize,
        residual_omega_scale: f64,
        seed: Option<u64>,
    ) -> Self {
        Self {
            inner: match seed {
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
            },
        }
    }

    /// Create with deterministic seed (no-arg convenience using shared defaults).
    #[staticmethod]
    #[pyo3(signature = (seed=DEFAULT_ORBIT_SEED))]
    fn new_default_seeded(seed: u64) -> Self {
        Self {
            inner: RustOrbitState::new_with_seed(
                1,
                0,
                0.0,
                DEFAULT_BASE_OMEGA,
                1.02,
                0.3,
                DEFAULT_K_RESIDUALS,
                DEFAULT_RESIDUAL_OMEGA_SCALE,
                seed,
            ),
        }
    }

    /// Create fully specified state with deterministic seed.
    #[staticmethod]
    #[pyo3(signature = (lobe, sub_lobe, theta, omega, s, alpha, k_residuals, residual_omega_scale, seed))]
    fn new_with_seed(
        lobe: u32,
        sub_lobe: u32,
        theta: f64,
        omega: f64,
        s: f64,
        alpha: f64,
        k_residuals: usize,
        residual_omega_scale: f64,
        seed: u64,
    ) -> Self {
        Self {
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

    /// Advance the state by dt without synthesising c.  This mutates
    /// the internal phases.
    fn advance(&mut self, dt: f64) {
        self.inner.advance(dt);
    }

    /// Return the current carrier point (no residuals).  This calls
    /// `lobe_point_at_angle` with the current theta and radial scale.
    fn carrier(&self) -> Complex {
        rust_lobe_point_at_angle(self.inner.lobe, self.inner.sub_lobe, self.inner.theta, self.inner.s).into()
    }

    /// Get a copy of the residual phases.  This can be used by
    /// training code to seed deterministic initial phases.
    fn residual_phases(&self) -> Vec<f64> {
        self.inner.residual_phases.clone()
    }

    /// Get a copy of the residual angular velocities.
    fn residual_omegas(&self) -> Vec<f64> {
        self.inner.residual_omegas.clone()
    }

    /// Synthesize c(t) without advancing time.  Band gates may be
    /// provided as a list of floats with length `k_residuals`.  If
    /// omitted each residual is fully enabled.
    #[pyo3(signature = (residual_params, band_gates=None))]
    fn synthesize(&self, residual_params: ResidualParams, band_gates: Option<Vec<f64>>) -> Complex {
        let gates_ref = band_gates.as_deref();
        rust_synthesize(&self.inner, residual_params.into(), gates_ref).into()
    }

    /// Advance time by dt and return the next c(t).  The band gates
    /// are applied to each residual.
    #[pyo3(signature = (dt, residual_params, band_gates=None))]
    fn step(&mut self, dt: f64, residual_params: ResidualParams, band_gates: Option<Vec<f64>>) -> Complex {
        rust_step(&mut self.inner, dt, residual_params.into(), band_gates.as_deref()).into()
    }
}

/// Python wrapper for the feature extractor
#[pyclass]
#[derive(Clone)]
pub struct FeatureExtractor {
    inner: RustFeatureExtractor,
}

#[pymethods]
impl FeatureExtractor {
    #[new]
    #[pyo3(signature = (sr=48_000, hop_length=1024, n_fft=4096, include_delta=false, include_delta_delta=false))]
    fn py_new(
        sr: usize,
        hop_length: usize,
        n_fft: usize,
        include_delta: bool,
        include_delta_delta: bool,
    ) -> Self {
        Self {
            inner: RustFeatureExtractor::new(sr, hop_length, n_fft, include_delta, include_delta_delta),
        }
    }

    /// Return the number of features per frame (including deltas).
    fn num_features_per_frame(&self) -> usize {
        self.inner.num_features_per_frame()
    }
    
    /// Simple test function to verify Rust execution
    fn test_simple(&self) -> Vec<f32> {
        eprintln!("[DEBUG] test_simple called");
        vec![1.0, 2.0, 3.0]
    }

    /// Extract windowed features from audio samples as a Python list.
    #[pyo3(signature = (audio, window_frames))]
    fn extract_windowed_features(&self, audio: Vec<f32>, window_frames: usize) -> PyResult<Vec<Vec<f64>>> {
        eprintln!("[PYBIND] extract_windowed_features called with {} samples", audio.len());
        let features = self.inner.extract_windowed_features(&audio, window_frames);
        eprintln!("[PYBIND] Returned {} windows", features.len());
        Ok(features)
    }
}

/// Free function: compute a point on the Mandelbrot lobe in Python.
#[pyfunction]
#[pyo3(signature = (lobe, sub_lobe, theta, s))]
fn lobe_point_at_angle(lobe: u32, sub_lobe: u32, theta: f64, s: f64) -> Complex {
    rust_lobe_point_at_angle(lobe, sub_lobe, theta, s).into()
}

/// Runtime visual metrics computed in Rust.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RuntimeVisualMetrics {
    #[pyo3(get)]
    pub edge_density: f64,
    #[pyo3(get)]
    pub color_uniformity: f64,
    #[pyo3(get)]
    pub brightness_mean: f64,
    #[pyo3(get)]
    pub brightness_std: f64,
    #[pyo3(get)]
    pub brightness_range: f64,
    #[pyo3(get)]
    pub mandelbrot_membership: bool,
}

/// Python wrapper for minimap sampling.
#[pyclass]
#[derive(Clone)]
pub struct Minimap {
    inner: RustMinimap,
}

#[pymethods]
impl Minimap {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: RustMinimap::new(),
        }
    }

    /// Sample F(c) and gradient at a mip level.
    #[pyo3(signature = (c_real, c_imag, mip_level))]
    fn sample(&self, c_real: f64, c_imag: f64, mip_level: usize) -> (f32, f32, f32) {
        let sample = self.inner.sample(RustComplex::new(c_real, c_imag), mip_level);
        (sample.f_value, sample.grad_re, sample.grad_im)
    }

    /// Sample a k×k patch (row-major) at a mip level.
    #[pyo3(signature = (c_real, c_imag, mip_level, k=MINIMAP_PATCH_K))]
    fn sample_patch(&self, c_real: f64, c_imag: f64, mip_level: usize, k: usize) -> Vec<f32> {
        self.inner
            .sample_patch(RustComplex::new(c_real, c_imag), mip_level, k)
    }

    /// Build the controller context vector in the canonical order.
    #[pyo3(signature = (c_real, c_imag, prev_delta_real=0.0, prev_delta_imag=0.0, mip_level=0))]
    fn context_features(
        &self,
        c_real: f64,
        c_imag: f64,
        prev_delta_real: f64,
        prev_delta_imag: f64,
        mip_level: usize,
    ) -> Vec<f32> {
        let context = RustControllerContext {
            c: RustComplex::new(c_real, c_imag),
            prev_delta: RustComplex::new(prev_delta_real, prev_delta_imag),
            nu_norm: self.inner.sample(RustComplex::new(c_real, c_imag), mip_level).f_value,
            membership: RustMinimap::membership(RustComplex::new(c_real, c_imag)),
            grad_re: self.inner.sample(RustComplex::new(c_real, c_imag), mip_level).grad_re,
            grad_im: self.inner.sample(RustComplex::new(c_real, c_imag), mip_level).grad_im,
            sensitivity: 0.0,
            patch: self
                .inner
                .sample_patch(RustComplex::new(c_real, c_imag), mip_level, MINIMAP_PATCH_K),
        };
        let mut context = context;
        let sample = self.inner.sample(RustComplex::new(c_real, c_imag), mip_level);
        context.sensitivity = crate::step_controller::sensitivity_from_grad(sample);
        context.to_feature_vector()
    }
}

/// Debug info for the step controller.
#[pyclass]
#[derive(Clone, Debug)]
pub struct StepDebug {
    #[pyo3(get)]
    pub mip_level: usize,
    #[pyo3(get)]
    pub scale_g: f64,
    #[pyo3(get)]
    pub scale_df: f64,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub wall_applied: bool,
}

impl From<crate::step_controller::StepDebug> for StepDebug {
    fn from(debug: crate::step_controller::StepDebug) -> Self {
        Self {
            mip_level: debug.mip_level,
            scale_g: debug.scale_g,
            scale_df: debug.scale_df,
            scale: debug.scale,
            wall_applied: debug.wall_applied,
        }
    }
}

/// Controller context values.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ControllerContext {
    inner: RustControllerContext,
}

#[pymethods]
impl ControllerContext {
    #[getter]
    fn c_real(&self) -> f64 {
        self.inner.c.real
    }

    #[getter]
    fn c_imag(&self) -> f64 {
        self.inner.c.imag
    }

    #[getter]
    fn prev_delta_real(&self) -> f64 {
        self.inner.prev_delta.real
    }

    #[getter]
    fn prev_delta_imag(&self) -> f64 {
        self.inner.prev_delta.imag
    }

    #[getter]
    fn nu_norm(&self) -> f32 {
        self.inner.nu_norm
    }

    #[getter]
    fn membership(&self) -> bool {
        self.inner.membership
    }

    #[getter]
    fn grad_re(&self) -> f32 {
        self.inner.grad_re
    }

    #[getter]
    fn grad_im(&self) -> f32 {
        self.inner.grad_im
    }

    #[getter]
    fn sensitivity(&self) -> f32 {
        self.inner.sensitivity
    }

    #[getter]
    fn patch(&self) -> Vec<f32> {
        self.inner.patch.clone()
    }

    fn feature_vector(&self) -> Vec<f32> {
        self.inner.to_feature_vector()
    }
}

impl From<RustControllerContext> for ControllerContext {
    fn from(inner: RustControllerContext) -> Self {
        Self { inner }
    }
}

/// Result of applying a step.
#[pyclass]
#[derive(Clone, Debug)]
pub struct StepResult {
    #[pyo3(get)]
    pub delta_real: f64,
    #[pyo3(get)]
    pub delta_imag: f64,
    #[pyo3(get)]
    pub c_next_real: f64,
    #[pyo3(get)]
    pub c_next_imag: f64,
    #[pyo3(get)]
    pub debug: StepDebug,
    #[pyo3(get)]
    pub context: ControllerContext,
}

impl From<RustStepResult> for StepResult {
    fn from(result: RustStepResult) -> Self {
        Self {
            delta_real: result.delta_applied.real,
            delta_imag: result.delta_applied.imag,
            c_next_real: result.c_next.real,
            c_next_imag: result.c_next.imag,
            debug: result.debug.into(),
            context: result.context.into(),
        }
    }
}

/// Step controller wrapper.
#[pyclass]
#[derive(Clone)]
pub struct StepController {
    inner: RustStepController,
}

#[pymethods]
impl StepController {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: RustStepController::new(),
        }
    }

    #[pyo3(signature = (c_real, c_imag, delta_real, delta_imag, prev_delta_real=0.0, prev_delta_imag=0.0))]
    fn apply_step(
        &self,
        c_real: f64,
        c_imag: f64,
        delta_real: f64,
        delta_imag: f64,
        prev_delta_real: f64,
        prev_delta_imag: f64,
    ) -> StepResult {
        let result = self.inner.apply_step(
            RustComplex::new(c_real, c_imag),
            RustComplex::new(delta_real, delta_imag),
            Some(RustComplex::new(prev_delta_real, prev_delta_imag)),
        );
        result.into()
    }

    #[pyo3(signature = (c_real, c_imag, prev_delta_real=0.0, prev_delta_imag=0.0, mip_level=0))]
    fn context_features(
        &self,
        c_real: f64,
        c_imag: f64,
        prev_delta_real: f64,
        prev_delta_imag: f64,
        mip_level: usize,
    ) -> ControllerContext {
        self.inner
            .context_for(
                RustComplex::new(c_real, c_imag),
                Some(RustComplex::new(prev_delta_real, prev_delta_imag)),
                mip_level,
            )
            .into()
    }
}

#[pyfunction]
#[pyo3(signature = (delta_real, delta_imag))]
fn step_mip_for_delta(delta_real: f64, delta_imag: f64) -> usize {
    mip_for_delta(RustComplex::new(delta_real, delta_imag))
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

/// Compute runtime visual metrics from an image buffer and Julia seed.
#[pyfunction]
#[pyo3(signature = (image, width, height, channels, c_real, c_imag, max_iter=100))]
fn compute_runtime_visual_metrics(
    image: Vec<f64>,
    width: usize,
    height: usize,
    channels: usize,
    c_real: f64,
    c_imag: f64,
    max_iter: usize,
) -> PyResult<RuntimeVisualMetrics> {
    let metrics = compute_runtime_metrics(
        &image,
        width,
        height,
        channels,
        RustComplex::new(c_real, c_imag),
        max_iter,
    )
    .map_err(|message| pyo3::exceptions::PyValueError::new_err(message))?;
    Ok(metrics.into())
}

#[pymodule]
#[allow(deprecated)]
fn runtime_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Shared constants
    m.add("SAMPLE_RATE", SAMPLE_RATE)?;
    m.add("HOP_LENGTH", HOP_LENGTH)?;
    m.add("N_FFT", N_FFT)?;
    m.add("WINDOW_FRAMES", WINDOW_FRAMES)?;
    m.add("DEFAULT_K_RESIDUALS", DEFAULT_K_RESIDUALS)?;
    m.add("DEFAULT_RESIDUAL_CAP", DEFAULT_RESIDUAL_CAP)?;
    m.add("DEFAULT_RESIDUAL_OMEGA_SCALE", DEFAULT_RESIDUAL_OMEGA_SCALE)?;
    m.add("DEFAULT_BASE_OMEGA", DEFAULT_BASE_OMEGA)?;
    m.add("DEFAULT_ORBIT_SEED", DEFAULT_ORBIT_SEED)?;
    m.add("MINIMAP_MIP_LEVELS", MINIMAP_MIP_LEVELS)?;
    m.add("MINIMAP_PATCH_K", MINIMAP_PATCH_K)?;
    m.add("STEP_CONTEXT_LEN", CONTEXT_LEN)?;

    m.add_class::<Complex>()?;
    m.add_class::<ResidualParams>()?;
    m.add_class::<OrbitState>()?;
    m.add_class::<FeatureExtractor>()?;
    m.add_class::<Minimap>()?;
    m.add_class::<StepController>()?;
    m.add_class::<ControllerContext>()?;
    m.add_class::<StepResult>()?;
    m.add_class::<StepDebug>()?;
    m.add_class::<RuntimeVisualMetrics>()?;
    m.add_function(wrap_pyfunction!(lobe_point_at_angle, m)?)?;
    m.add_function(wrap_pyfunction!(compute_runtime_visual_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(step_mip_for_delta, m)?)?;
    Ok(())
}
