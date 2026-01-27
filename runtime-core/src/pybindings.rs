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

impl From<Complex> for RustComplex {
    fn from(c: Complex) -> Self {
        Self::new(c.real, c.imag)
    }
}

impl From<&Complex> for RustComplex {
    fn from(c: &Complex) -> Self {
        Self::new(c.real, c.imag)
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

/// Python wrapper for the height-field parameters.
#[pyclass]
#[derive(Clone, Debug)]
pub struct HeightFieldParams {
    #[pyo3(get, set)]
    pub iterations: usize,
    #[pyo3(get, set)]
    pub min_magnitude: f64,
}

#[pymethods]
impl HeightFieldParams {
    #[new]
    #[pyo3(signature = (iterations=DEFAULT_HEIGHT_ITERATIONS, min_magnitude=DEFAULT_HEIGHT_MIN_MAGNITUDE))]
    fn py_new(iterations: usize, min_magnitude: f64) -> Self {
        Self {
            iterations,
            min_magnitude,
        }
    }
}

impl From<HeightFieldParams> for RustHeightFieldParams {
    fn from(p: HeightFieldParams) -> RustHeightFieldParams {
        RustHeightFieldParams {
            iterations: p.iterations,
            min_magnitude: p.min_magnitude,
        }
    }
}

/// Python wrapper for a height-field sample.
#[pyclass]
#[derive(Clone, Debug)]
pub struct HeightFieldSample {
    #[pyo3(get)]
    pub height: f64,
    #[pyo3(get)]
    pub gradient: Complex,
    #[pyo3(get)]
    pub z: Complex,
    #[pyo3(get)]
    pub w: Complex,
    #[pyo3(get)]
    pub magnitude: f64,
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

/// Python wrapper for contour controller parameters.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ContourControllerParams {
    #[pyo3(get, set)]
    pub correction_gain: f64,
    #[pyo3(get, set)]
    pub projection_epsilon: f64,
}

#[pymethods]
impl ContourControllerParams {
    #[new]
    #[pyo3(signature = (correction_gain=DEFAULT_CONTOUR_CORRECTION_GAIN, projection_epsilon=DEFAULT_CONTOUR_PROJECTION_EPSILON))]
    fn py_new(correction_gain: f64, projection_epsilon: f64) -> Self {
        Self {
            correction_gain,
            projection_epsilon,
        }
    }
}

impl From<ContourControllerParams> for RustContourControllerParams {
    fn from(p: ContourControllerParams) -> RustContourControllerParams {
        RustContourControllerParams {
            correction_gain: p.correction_gain,
            projection_epsilon: p.projection_epsilon,
        }
    }
}

/// Python wrapper for contour controller state.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ContourState {
    inner: RustContourState,
}

#[pymethods]
impl ContourState {
    #[new]
    #[pyo3(signature = (c, field_params=None))]
    fn py_new(c: Complex, field_params: Option<HeightFieldParams>) -> Self {
        let params = field_params.unwrap_or_else(|| HeightFieldParams {
            iterations: DEFAULT_HEIGHT_ITERATIONS,
            min_magnitude: DEFAULT_HEIGHT_MIN_MAGNITUDE,
        });
        Self {
            inner: RustContourState::new(c.into(), params.into()),
        }
    }

    #[pyo3(signature = (target_height))]
    fn set_target_height(&mut self, target_height: f64) {
        self.inner.set_target_height(target_height);
    }

    fn target_height(&self) -> f64 {
        self.inner.target_height
    }

    fn c(&self) -> Complex {
        self.inner.c.into()
    }

    fn step(
        &mut self,
        model_delta: Complex,
        field_params: HeightFieldParams,
        controller_params: ContourControllerParams,
    ) -> ContourStep {
        let step = self
            .inner
            .step(model_delta.into(), field_params.into(), controller_params.into());
        step.into()
    }
}

/// Python wrapper for contour controller step output.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ContourStep {
    #[pyo3(get)]
    pub c: Complex,
    #[pyo3(get)]
    pub height: f64,
    #[pyo3(get)]
    pub height_error: f64,
    #[pyo3(get)]
    pub gradient: Complex,
    #[pyo3(get)]
    pub corrected_delta: Complex,
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

/// Free function: sample height field at a point.
#[pyfunction]
#[pyo3(signature = (c, params=None))]
fn sample_height_field(c: Complex, params: Option<HeightFieldParams>) -> HeightFieldSample {
    let params = params.unwrap_or_else(|| HeightFieldParams {
        iterations: DEFAULT_HEIGHT_ITERATIONS,
        min_magnitude: DEFAULT_HEIGHT_MIN_MAGNITUDE,
    });
    rust_sample_height_field(c.into(), params.into()).into()
}

/// Free function: apply contour correction to a proposed delta.
#[pyfunction]
#[pyo3(signature = (model_delta, gradient, height_error, params=None))]
fn contour_correct_delta(
    model_delta: Complex,
    gradient: Complex,
    height_error: f64,
    params: Option<ContourControllerParams>,
) -> Complex {
    let params = params.unwrap_or_else(|| ContourControllerParams {
        correction_gain: DEFAULT_CONTOUR_CORRECTION_GAIN,
        projection_epsilon: DEFAULT_CONTOUR_PROJECTION_EPSILON,
    });
    rust_contour_correct_delta(
        model_delta.into(),
        gradient.into(),
        height_error,
        params.into(),
    )
    .into()
}

#[pymodule]
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
    m.add("DEFAULT_HEIGHT_ITERATIONS", DEFAULT_HEIGHT_ITERATIONS)?;
    m.add("DEFAULT_HEIGHT_MIN_MAGNITUDE", DEFAULT_HEIGHT_MIN_MAGNITUDE)?;
    m.add("DEFAULT_CONTOUR_CORRECTION_GAIN", DEFAULT_CONTOUR_CORRECTION_GAIN)?;
    m.add("DEFAULT_CONTOUR_PROJECTION_EPSILON", DEFAULT_CONTOUR_PROJECTION_EPSILON)?;

    m.add_class::<Complex>()?;
    m.add_class::<ResidualParams>()?;
    m.add_class::<OrbitState>()?;
    m.add_class::<HeightFieldParams>()?;
    m.add_class::<HeightFieldSample>()?;
    m.add_class::<ContourControllerParams>()?;
    m.add_class::<ContourState>()?;
    m.add_class::<ContourStep>()?;
    m.add_class::<FeatureExtractor>()?;
    m.add_function(wrap_pyfunction!(lobe_point_at_angle, m)?)?;
    m.add_function(wrap_pyfunction!(sample_height_field, m)?)?;
    m.add_function(wrap_pyfunction!(contour_correct_delta, m)?)?;
    Ok(())
}
