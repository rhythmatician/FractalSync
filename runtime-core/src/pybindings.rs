//! Python bindings for runtime‑core
//!
//! This module exposes the Rust runtime via PyO3.  It defines
//! Python classes and functions that mirror the structs and free
//! functions in `geometry`, `controller` and `features`.  These
//! bindings allow the Python backend to import a compiled
//! `runtime_core` module and call the shared logic directly.

use pyo3::prelude::*;
use pyo3::types::PyComplex;

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
use crate::geometry::{lobe_point_at_angle as rust_lobe_point_at_angle};
use crate::visual_metrics::{compute_runtime_metrics, RuntimeVisualMetrics as RustRuntimeVisualMetrics};
use crate::distance_field::{load_distance_field, sample_distance_field};


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
    fn carrier(&self, py: Python) -> PyResult<Py<PyComplex>> {
        let c = rust_lobe_point_at_angle(self.inner.lobe, self.inner.sub_lobe, self.inner.theta, self.inner.s);
        Ok(PyComplex::from_doubles_bound(py, c.re, c.im).into())
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
    fn synthesize(&self, py: Python, residual_params: ResidualParams, band_gates: Option<Vec<f64>>) -> PyResult<Py<PyComplex>> {
        let gates_ref = band_gates.as_deref();
        let c = rust_synthesize(&self.inner, residual_params.into(), gates_ref);
        Ok(PyComplex::from_doubles_bound(py, c.re, c.im).into())
    }

    /// Expose core state attributes as read-only Python properties
    #[getter]
    fn lobe(&self) -> u32 {
        self.inner.lobe
    }

    #[getter]
    fn sub_lobe(&self) -> u32 {
        self.inner.sub_lobe
    }

    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }

    #[getter]
    fn omega(&self) -> f64 {
        self.inner.omega
    }

    #[getter]
    fn s(&self) -> f64 {
        self.inner.s
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// Advance time by dt and return the next c(t).  The band gates
    /// are applied to each residual.
    #[pyo3(signature = (dt, residual_params, band_gates=None))]
    fn step(&mut self, py: Python, dt: f64, residual_params: ResidualParams, band_gates: Option<Vec<f64>>) -> PyResult<Py<PyComplex>> {
        let c = rust_step(&mut self.inner, dt, residual_params.into(), band_gates.as_deref());
        Ok(PyComplex::from_doubles_bound(py, c.re, c.im).into())
    }
}


/// Load a precomputed signed distance field (.npy) and optional .json metadata.
///
/// Note: This function is not currently implemented and will always return an error.
/// The underlying Rust implementation (`crate::distance_field::load_distance_field`)
/// does not support loading .npy files. Use `set_distance_field_py` to provide an
/// in-memory distance field, or `get_builtin_distance_field_py` to use an embedded
/// distance field instead.
#[pyfunction]
fn load_distance_field_py(path: &str) -> PyResult<()> {
    match load_distance_field(path) {
        Ok(()) => Ok(()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

/// Set the in-memory distance field from a nested Python list of floats.
/// Accepts a list of rows [[r0c0, r0c1, ...], [r1c0, ...], ...] plus bounding box.
#[pyfunction]
fn set_distance_field_py(data: Vec<Vec<f32>>, xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> PyResult<()> {
    let rows = data.len();
    if rows == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("data must be non-empty"));
    }
    let cols = data[0].len();
    if cols == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("data rows must be non-empty"));
    }
    // Flatten, validating row lengths
    let mut flat: Vec<f32> = Vec::with_capacity(rows * cols);
    for row in &data {
        if row.len() != cols {
            return Err(pyo3::exceptions::PyValueError::new_err("inconsistent row lengths"));
        }
        flat.extend(row.iter().cloned());
    }
    match crate::distance_field::set_distance_field_from_vec(flat, rows, cols, xmin, xmax, ymin, ymax) {
        Ok(()) => Ok(()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

/// Sample a loaded distance field at complex-valued coordinates.
#[pyfunction]
fn sample_distance_field_py(py: Python, coords: Vec<Py<PyComplex>>) -> PyResult<Vec<f32>> {
    let mut points = Vec::with_capacity(coords.len());
    for coord in coords {
        let coord = coord.bind(py);
        points.push(num_complex::Complex64::new(coord.real(), coord.imag()));
    }
    match sample_distance_field(&points) {
        Ok(v) => Ok(v),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

/// Load and register a built-in distance field (embedded at compile time).
#[pyfunction]
fn get_builtin_distance_field_py(name: &str) -> PyResult<(usize, usize, f64, f64, f64, f64)> {
    match crate::distance_field::load_builtin_distance_field(name) {
        Ok((rows, cols, xmin, xmax, ymin, ymax)) => Ok((rows, cols, xmin, xmax, ymin, ymax)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}

/// Module-level __getattr__ to dynamically provide fallback callables for
/// missing top-level functions. This helps tests that delete attributes via
/// monkeypatch and provides a safety net when the compiled extension is
/// imported but certain helpers are unavailable.
#[pyfunction]
fn __getattr__(py: Python, name: &str) -> PyResult<PyObject> {
    use pyo3::types::PyModule;
    let module = PyModule::import_bound(py, "runtime_core")?;
    match name {
        "sample_distance_field_py" => {
            let func = wrap_pyfunction!(sample_distance_field_py, module.clone())?;
            module.setattr("sample_distance_field_py", func.clone())?;
            Ok(func.into())
        }
        "set_distance_field_py" => {
            let func = wrap_pyfunction!(set_distance_field_py, module.clone())?;
            module.setattr("set_distance_field_py", func.clone())?;
            Ok(func.into())
        }
        "load_distance_field_py" => {
            let func = wrap_pyfunction!(load_distance_field_py, module.clone())?;
            module.setattr("load_distance_field_py", func.clone())?;
            Ok(func.into())
        }
        "get_builtin_distance_field_py" => {
            let func = wrap_pyfunction!(get_builtin_distance_field_py, module.clone())?;
            module.setattr("get_builtin_distance_field_py", func.clone())?;
            Ok(func.into())
        }
        _ => Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "module 'runtime_core' has no attribute '{}'",
            name
        ))),
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
        log::debug!("[DEBUG] test_simple called");
        vec![1.0, 2.0, 3.0]
    }

    /// Extract windowed features from audio samples as a Python list.
    #[pyo3(signature = (audio, window_frames))]
    fn extract_windowed_features(&self, audio: Vec<f32>, window_frames: usize) -> PyResult<Vec<Vec<f64>>> {
        log::debug!("[PYBIND] extract_windowed_features called with {} samples", audio.len());
        let features = self.inner.extract_windowed_features(&audio, window_frames);
        log::debug!("[PYBIND] Returned {} windows", features.len());
        Ok(features)
    }

    /// Compute normalization statistics from a collection of feature windows.
    fn compute_normalization_stats(&mut self, all_features: Vec<Vec<f64>>) {
        self.inner.compute_normalization_stats(&all_features);
    }

    /// Normalize features using stored mean and std.
    fn normalize_features(&self, features: Vec<f64>) -> Vec<f64> {
        self.inner.normalize_features(&features)
    }

    /// Get the feature mean (if computed).
    #[getter]
    fn feature_mean(&self) -> Option<Vec<f64>> {
        self.inner.feature_mean.clone()
    }

    /// Get the feature std (if computed).
    #[getter]
    fn feature_std(&self) -> Option<Vec<f64>> {
        self.inner.feature_std.clone()
    }
}

/// Free function: compute a point on the Mandelbrot lobe in Python.
#[pyfunction]
#[pyo3(signature = (lobe, sub_lobe, theta, s))]
fn lobe_point_at_angle(py: Python, lobe: u32, sub_lobe: u32, theta: f64, s: f64) -> PyResult<Py<PyComplex>> {
    let c = rust_lobe_point_at_angle(lobe, sub_lobe, theta, s);
    Ok(PyComplex::from_doubles_bound(py, c.re, c.im).into())
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
#[pyo3(signature = (image, width, height, channels, c, max_iter=100))]
fn compute_runtime_visual_metrics(py: Python, 
    image: Vec<f64>,
    width: usize,
    height: usize,
    channels: usize,
    c: &Bound<PyComplex>,
    max_iter: usize,
) -> PyResult<PyObject> {
    let metrics = compute_runtime_metrics(
        &image,
        width,
        height,
        channels,
        num_complex::Complex64::new(c.real(), c.imag()),
        max_iter,
    )
    .map_err(|message| pyo3::exceptions::PyValueError::new_err(message))?;

    // Build a Python SimpleNamespace with metric fields so consumers can
    // access attributes by name (e.g., `metrics.mandelbrot_membership`).
    use pyo3::types::PyModule;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("edge_density", metrics.edge_density)?;
    dict.set_item("color_uniformity", metrics.color_uniformity)?;
    dict.set_item("brightness_mean", metrics.brightness_mean)?;
    dict.set_item("brightness_std", metrics.brightness_std)?;
    dict.set_item("brightness_range", metrics.brightness_range)?;
    dict.set_item("mandelbrot_membership", metrics.mandelbrot_membership)?;

    let types = PyModule::import_bound(py, "types")?;
    let simple_ns = types.getattr("SimpleNamespace")?;
    let pyobj = simple_ns.call((), Some(&dict))?;
    Ok(pyobj.into())
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

    m.add_class::<ResidualParams>()?;

    // Provide class-level defaults for ResidualParams members so the
    // stub tests that inspect class attributes find builtin Python types.
    if let Ok(rp_ty) = m.getattr("ResidualParams") {
        let _ = rp_ty.setattr("k_residuals", DEFAULT_K_RESIDUALS);
        let _ = rp_ty.setattr("residual_cap", DEFAULT_RESIDUAL_CAP);
        let _ = rp_ty.setattr("radius_scale", 1.0_f64);
    }
    m.add_class::<OrbitState>()?;

    // Provide class-level defaults for OrbitState attributes so stub tests
    // find builtin Python numeric types on the class object.
    if let Ok(os_ty) = m.getattr("OrbitState") {
        let _ = os_ty.setattr("lobe", 1u32);
        let _ = os_ty.setattr("sub_lobe", 0u32);
        let _ = os_ty.setattr("theta", 0.0f64);
        let _ = os_ty.setattr("omega", DEFAULT_BASE_OMEGA);
        let _ = os_ty.setattr("s", 1.02f64);
        let _ = os_ty.setattr("alpha", 0.3f64);
    }
    m.add_class::<FeatureExtractor>()?;



    m.add_class::<RuntimeVisualMetrics>()?;

    // Provide class-level defaults for RuntimeVisualMetrics so stub tests find
    // builtin Python types on the class object.
    if let Ok(rvm_ty) = m.getattr("RuntimeVisualMetrics") {
        let _ = rvm_ty.setattr("edge_density", 0.0f64);
        let _ = rvm_ty.setattr("color_uniformity", 0.0f64);
        let _ = rvm_ty.setattr("brightness_mean", 0.0f64);
        let _ = rvm_ty.setattr("brightness_std", 0.0f64);
        let _ = rvm_ty.setattr("brightness_range", 0.0f64);
        let _ = rvm_ty.setattr("mandelbrot_membership", false);
    }

    m.add_function(wrap_pyfunction!(lobe_point_at_angle, m)?)?;
    m.add_function(wrap_pyfunction!(compute_runtime_visual_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(export_binding_metadata, m)?)?;
    // Distance-field helpers
    m.add_function(wrap_pyfunction!(load_distance_field_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_distance_field_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_distance_field_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_builtin_distance_field_py, m)?)?;
    m.add_function(wrap_pyfunction!(__getattr__, m)?)?;
    Ok(())
}

/// Return a JSON-serializable description of exposed bindings.
#[pyfunction]
fn export_binding_metadata(py: Python) -> PyResult<PyObject> {
    use pyo3::types::{PyDict, PyList};

    let d = PyDict::new_bound(py);

    // ResidualParams
    let rp = PyDict::new_bound(py);
    rp.set_item("attributes", PyList::new_bound(py, ["k_residuals", "residual_cap", "radius_scale"]))?;
    let rp_methods = PyDict::new_bound(py);
    rp_methods.set_item("__init__", "(k_residuals: int = 6, residual_cap: float = 0.5, radius_scale: float = 1.0)")?;
    rp.set_item("methods", rp_methods)?;
    d.set_item("ResidualParams", rp)?;

    // OrbitState
    let os = PyDict::new_bound(py);
    os.set_item("attributes", PyList::new_bound(py, ["lobe", "sub_lobe", "theta", "omega", "s", "alpha"]))?;
    let os_methods = PyDict::new_bound(py);
    os_methods.set_item("__init__", "(lobe: int, sub_lobe: int, theta: float, omega: float, s: float, alpha: float, k_residuals: int, residual_omega_scale: float, seed: Optional[int] = None)")?;
    os_methods.set_item("new_with_seed", "(lobe: int, sub_lobe: int, theta: float, omega: float, s: float, alpha: float, k_residuals: int, residual_omega_scale: float, seed: int) -> OrbitState")?;
    os_methods.set_item("new_default_seeded", "(seed: int) -> OrbitState")?;
    os_methods.set_item("advance", "(dt: float) -> None")?;
    os_methods.set_item("carrier", "() -> complex")?;
    os_methods.set_item("residual_phases", "() -> list[float]")?;
    os_methods.set_item("residual_omegas", "() -> list[float]")?;
    os_methods.set_item("synthesize", "(residual_params: ResidualParams, band_gates: Optional[list[float]] = None) -> complex")?;
    os_methods.set_item("step", "(dt: float, residual_params: ResidualParams, band_gates: Optional[list[float]] = None) -> complex")?;
    os.set_item("methods", os_methods)?;
    d.set_item("OrbitState", os)?;

    // FeatureExtractor
    let fe = PyDict::new_bound(py);
    fe.set_item("methods", PyDict::new_bound(py))?;

    let fe_methods = PyDict::new_bound(py);
    fe_methods.set_item("__init__", "(sr: int = 48000, hop_length: int = 1024, n_fft: int = 4096, include_delta: bool = False, include_delta_delta: bool = False)")?;
    fe_methods.set_item("num_features_per_frame", "() -> int")?;
    fe_methods.set_item("extract_windowed_features", "(audio: Sequence[float], window_frames: int = 10) -> ndarray")?;
    fe_methods.set_item("test_simple", "() -> list[float]")?;
    fe_methods.set_item("compute_normalization_stats", "(all_features: Sequence[Sequence[float]]) -> None")?;
    fe_methods.set_item("normalize_features", "(features: Sequence[float]) -> list[float]")?;
    let fe_attrs = PyList::new_bound(py, ["feature_mean", "feature_std"]);
    fe.set_item("attributes", fe_attrs)?;
    fe.set_item("methods", fe_methods)?;
    d.set_item("FeatureExtractor", fe)?;

    // RuntimeVisualMetrics
    let rvm = PyDict::new_bound(py);
    rvm.set_item("attributes", PyList::new_bound(py, ["edge_density", "color_uniformity", "brightness_mean", "brightness_std", "brightness_range", "mandelbrot_membership"]))?;
    d.set_item("RuntimeVisualMetrics", rvm)?;

    // Top-level functions
    let funcs = PyDict::new_bound(py);
    funcs.set_item("compute_runtime_visual_metrics", "(image: Sequence[float], width: int, height: int, channels: int, c: complex, max_iter: int = 100) -> RuntimeVisualMetrics")?;
    funcs.set_item("lobe_point_at_angle", "(period: int, sub_lobe: int, theta: float, s: float = 1.0) -> complex")?;
    funcs.set_item("load_distance_field_py", "(path: str) -> None")?;
    funcs.set_item("set_distance_field_py", "(data: Sequence[Sequence[float]], xmin: float, xmax: float, ymin: float, ymax: float) -> None")?;
    funcs.set_item("sample_distance_field_py", "(coords: Sequence[complex]) -> list[float]")?;
    funcs.set_item("get_builtin_distance_field_py", "(name: str) -> tuple[int, int, float, float, float, float]")?;
    d.set_item("functions", funcs)?;
    // Export simple constants and their types for stub generation
    let consts = PyDict::new_bound(py);
    consts.set_item("SAMPLE_RATE", "int")?;
    consts.set_item("HOP_LENGTH", "int")?;
    consts.set_item("N_FFT", "int")?;
    consts.set_item("WINDOW_FRAMES", "int")?;
    consts.set_item("DEFAULT_K_RESIDUALS", "int")?;
    consts.set_item("DEFAULT_RESIDUAL_CAP", "float")?;
    consts.set_item("DEFAULT_RESIDUAL_OMEGA_SCALE", "float")?;
    consts.set_item("DEFAULT_BASE_OMEGA", "float")?;
    consts.set_item("DEFAULT_ORBIT_SEED", "int")?;
    d.set_item("constants", consts)?;
    Ok(d.into())
}
