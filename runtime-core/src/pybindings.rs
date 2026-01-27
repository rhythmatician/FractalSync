//! Python bindings for runtime-core
//!
//! Exposes the height-field controller and feature extractor.

use pyo3::prelude::*;

use crate::controller::{
    evaluate_height_field,
    step_height_controller,
    HeightControllerParams,
    DEFAULT_HEIGHT_EPSILON,
    DEFAULT_HEIGHT_GAIN,
    DEFAULT_HEIGHT_ITERATIONS,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    WINDOW_FRAMES,
};
use crate::features::FeatureExtractor as RustFeatureExtractor;
use crate::geometry::Complex as RustComplex;

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

impl From<&Complex> for RustComplex {
    fn from(c: &Complex) -> RustComplex {
        RustComplex::new(c.real, c.imag)
    }
}

/// Python wrapper for height-field samples.
#[pyclass]
#[derive(Clone, Debug)]
pub struct HeightFieldSample {
    #[pyo3(get)]
    pub height: f64,
    #[pyo3(get)]
    pub gradient: Complex,
}

/// Python wrapper for controller step results.
#[pyclass]
#[derive(Clone, Debug)]
pub struct HeightControllerStep {
    #[pyo3(get)]
    pub new_c: Complex,
    #[pyo3(get)]
    pub delta: Complex,
    #[pyo3(get)]
    pub height: f64,
    #[pyo3(get)]
    pub gradient: Complex,
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

/// Evaluate f(c) = log|z_N(c)| and its gradient.
#[pyfunction]
#[pyo3(signature = (c, iterations=DEFAULT_HEIGHT_ITERATIONS, epsilon=DEFAULT_HEIGHT_EPSILON))]
fn height_field(c: Complex, iterations: usize, epsilon: f64) -> HeightFieldSample {
    let sample = evaluate_height_field(RustComplex::from(&c), iterations, epsilon);
    HeightFieldSample {
        height: sample.height,
        gradient: sample.gradient.into(),
    }
}

/// Project a model step onto the height contour and apply correction.
#[pyfunction]
#[pyo3(signature = (c, delta_model, target_height, normal_risk, height_gain=DEFAULT_HEIGHT_GAIN, iterations=DEFAULT_HEIGHT_ITERATIONS, epsilon=DEFAULT_HEIGHT_EPSILON))]
fn height_controller_step(
    c: Complex,
    delta_model: Complex,
    target_height: f64,
    normal_risk: f64,
    height_gain: f64,
    iterations: usize,
    epsilon: f64,
) -> HeightControllerStep {
    let params = HeightControllerParams {
        target_height,
        normal_risk,
        height_gain,
    };
    let step = step_height_controller(
        RustComplex::from(&c),
        RustComplex::from(&delta_model),
        params,
        iterations,
        epsilon,
    );
    HeightControllerStep {
        new_c: step.new_c.into(),
        delta: step.delta.into(),
        height: step.height,
        gradient: step.gradient.into(),
    }
}

#[pymodule]
fn runtime_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Shared constants
    m.add("SAMPLE_RATE", SAMPLE_RATE)?;
    m.add("HOP_LENGTH", HOP_LENGTH)?;
    m.add("N_FFT", N_FFT)?;
    m.add("WINDOW_FRAMES", WINDOW_FRAMES)?;
    m.add("DEFAULT_HEIGHT_ITERATIONS", DEFAULT_HEIGHT_ITERATIONS)?;
    m.add("DEFAULT_HEIGHT_EPSILON", DEFAULT_HEIGHT_EPSILON)?;
    m.add("DEFAULT_HEIGHT_GAIN", DEFAULT_HEIGHT_GAIN)?;

    m.add_class::<Complex>()?;
    m.add_class::<HeightFieldSample>()?;
    m.add_class::<HeightControllerStep>()?;
    m.add_class::<FeatureExtractor>()?;
    m.add_function(wrap_pyfunction!(height_field, m)?)?;
    m.add_function(wrap_pyfunction!(height_controller_step, m)?)?;
    Ok(())
}
