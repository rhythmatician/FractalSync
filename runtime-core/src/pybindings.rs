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
    PlayerState as RustPlayerState,
    OrbitController as RustOrbitController,
    DEFAULT_BASE_OMEGA,
    DEFAULT_K_RESIDUALS,
    DEFAULT_ORBIT_SEED,
    CONTROLLER_VERSION,
    DEFAULT_RESIDUAL_CAP,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    WINDOW_FRAMES,
    step as rust_step,
    synthesize as rust_synthesize,
    residual_phases_for_seed as rust_residual_phases_for_seed,
};
use crate::features::FeatureExtractor as RustFeatureExtractor;
use crate::geometry::{lobe_point_at_angle as rust_lobe_point_at_angle};
use crate::proxies as rust_proxies;
use crate::visual_metrics::{compute_runtime_metrics, RuntimeVisualMetrics as RustRuntimeVisualMetrics};
use crate::distance_field::{sample_distance_field};


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

/// Python wrapper for the Player c-space integrator (momentum + drag).
#[pyclass]
#[derive(Clone, Debug)]
pub struct PlayerState {
    inner: RustPlayerState,
}

#[pymethods]
impl PlayerState {
    #[new]
    fn py_new(lobe: u32, sub_lobe: u32, s: f64, alpha: f64) -> Self {
        Self {
            inner: RustPlayerState::new(lobe, sub_lobe, s, alpha),
        }
    }

    /// Current c (real part).
    #[getter]
    fn c_re(&self) -> f64 {
        self.inner.c.re
    }

    /// Current c (imaginary part).
    #[getter]
    fn c_im(&self) -> f64 {
        self.inner.c.im
    }

    /// Current c-space speed (Momentum diagnostic).
    #[getter]
    fn speed(&self) -> f64 {
        self.inner.velocity.norm()
    }

    /// Apply model-predicted control signals.
    fn apply_controls(&mut self, s: f64, alpha: f64, omega_scale: f64) {
        self.inner.apply_controls(s, alpha, omega_scale);
    }

    /// Switch the active Mandelbrot lobe.
    fn set_lobe(&mut self, lobe: u32, sub_lobe: u32) {
        self.inner.lobe = lobe;
        self.inner.sub_lobe = sub_lobe;
    }

    /// Set the mip level for the contour step.
    fn set_level(&mut self, level: usize) {
        self.inner.level = level;
    }

    /// Set the target shore-proximity distance.
    fn set_d_star(&mut self, d_star: f64) {
        self.inner.d_star = d_star;
    }

    /// Set the maximum world-space step per frame.
    fn set_max_step(&mut self, max_step: f64) {
        self.inner.max_step = max_step;
    }

    /// Set the audio energy in [0, 1] (loudness). Raises the servo's
    /// target shore-proximity: loud audio pulls c toward the Shore.
    fn set_energy(&mut self, energy: f64) {
        self.inner.energy = energy.clamp(0.0, 1.0);
    }

    /// Advance by dt; returns (re, im). `h` in [0,1] allows contour crossing
    /// during transients. Band gates optional.
    #[pyo3(signature = (dt, h, band_gates=None))]
    fn step(
        &mut self,
        dt: f64,
        h: f64,
        band_gates: Option<Vec<f64>>,
    ) -> (f64, f64) {
        let c = self.inner.step(dt, h, band_gates.as_deref());
        (c.re, c.im)
    }
}

/// Python wrapper for the May-proven OrbitController (runtime baseline).
#[pyclass]
#[derive(Clone, Debug)]
pub struct OrbitController {
    inner: RustOrbitController,
}

#[pymethods]
impl OrbitController {
    #[new]
    fn py_new(s: f64, alpha: f64, omega: f64) -> Self {
        Self {
            inner: RustOrbitController::new(s, alpha, omega),
        }
    }

    /// Wobble phase (diagnostic).
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }

    /// Apply model-predicted control signals (s, alpha).
    fn apply_controls(&mut self, s: f64, alpha: f64) {
        self.inner.apply_controls(s, alpha);
    }

    /// Refinement toggles (all default off = May baseline).
    fn set_momentum(&mut self, on: bool) {
        self.inner.momentum = on;
    }

    /// Per-frame velocity retention for momentum (default 0.90).
    fn set_drag(&mut self, drag: f64) {
        self.inner.drag = drag;
    }

    /// Audio thrust magnitude for momentum: sustained energy builds inertia.
    fn set_thrust(&mut self, thrust: f64) {
        self.inner.thrust = thrust;
    }

    /// Audio energy in [0, 1]: raises the servo's target shore-proximity
    /// (loud audio pulls c toward the Shore).
    fn set_energy(&mut self, energy: f64) {
        self.inner.energy = energy.clamp(0.0, 1.0);
    }

    fn set_shore_bias(&mut self, on: bool) {
        self.inner.shore_bias = on;
    }

    /// Target shore-proximity for the shore-bias servo.
    fn set_d_star(&mut self, d_star: f64) {
        self.inner.d_star = d_star;
    }

    /// Max world-space step per frame for shore bias.
    fn set_max_step(&mut self, max_step: f64) {
        self.inner.max_step = max_step;
    }

    /// Mip level for the contour step.
    fn set_level(&mut self, level: usize) {
        self.inner.level = level;
    }

    /// Set the persistent c position (momentum/shore-bias paths).
    fn set_c(&mut self, re: f64, im: f64) {
        self.inner.c = num_complex::Complex64::new(re, im);
    }

    /// Advance one frame; returns (re, im). `h` is the transient signal
    /// in [0, 1] — near 1 opens the Shore wall for boundary crossing.
    #[pyo3(signature = (dt, band_gates=None, h=0.0))]
    fn step(&mut self, dt: f64, band_gates: Option<Vec<f64>>, h: f64) -> (f64, f64) {
        let c = self.inner.step(dt, band_gates.as_deref(), h);
        (c.re, c.im)
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

    // Setters for the fields the live controller mutates at runtime
    // (lobe transitions and smoothed s / residual alpha).  theta and
    // omega remain read-only: they are owned by the state machine.
    #[setter]
    fn set_lobe(&mut self, value: u32) {
        self.inner.lobe = value;
    }

    #[setter]
    fn set_sub_lobe(&mut self, value: u32) {
        self.inner.sub_lobe = value;
    }

    #[setter]
    fn set_s(&mut self, value: f64) {
        self.inner.s = value;
    }

    #[setter]
    fn set_alpha(&mut self, value: f64) {
        self.inner.alpha = value;
    }

    /// Advance time by dt and return the next c(t).  The band gates
    /// are applied to each residual.
    #[pyo3(signature = (dt, residual_params, band_gates=None))]
    fn step(&mut self, py: Python, dt: f64, residual_params: ResidualParams, band_gates: Option<Vec<f64>>) -> PyResult<Py<PyComplex>> {
        let c = rust_step(&mut self.inner, dt, residual_params.into(), band_gates.as_deref());
        Ok(PyComplex::from_doubles_bound(py, c.re, c.im).into())
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

/// Generate the deterministic residual phases for a given seed.
///
/// Single source of truth for residual phase generation, shared by the
/// runtime controller and the training-time differentiable mirror so both
/// use identical phase statistics.
#[pyfunction]
fn residual_phases_for_seed_py(seed: u64, k_residuals: usize) -> Vec<f64> {
    rust_residual_phases_for_seed(seed, k_residuals)
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

/// Load the baked mip pyramid (the Map's multi-scale minimaps) from files and
/// register it as the process-wide pyramid.
#[pyfunction]
fn load_mip_pyramid_py(
    f_bin_path: &str,
    s_bin_path: &str,
    meta_path: &str,
) -> PyResult<(usize, f64, f64, f64, f64)> {
    let pyr = crate::minimap::load_pyramid_from_files(f_bin_path, s_bin_path, meta_path)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let n = pyr.num_levels();
    let (re_min, re_max, im_min, im_max) = (pyr.re_min, pyr.re_max, pyr.im_min, pyr.im_max);
    crate::minimap::set_pyramid(pyr).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok((n, re_min, re_max, im_min, im_max))
}

/// Install a synthetic mip pyramid for parity tests and local debugging.
///
/// Each level plane is `widths[i] * heights[i]` floats (row-major). The same
/// planes populate both the F (escape) and S (shore-proximity) fields of the
/// pyramid; tests that need a separable field should call
/// [`runtime_core.minimap_set_escape_field`] afterwards. Returns the
/// number of levels installed.
#[pyfunction]
fn install_pyramid_py(
    levels_data: Vec<Vec<f32>>,
    widths: Vec<usize>,
    heights: Vec<usize>,
    re_min: f64,
    re_max: f64,
    im_min: f64,
    im_max: f64,
) -> PyResult<usize> {
    let pyr = crate::minimap::MipPyramid::from_levels(
        levels_data,
        widths,
        heights,
        re_min,
        re_max,
        im_min,
        im_max,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let n = pyr.num_levels();
    crate::minimap::set_pyramid(pyr).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(n)
}

/// Clear the process-wide pyramid (test helper).
#[pyfunction]
fn clear_pyramid_py() {
    crate::minimap::clear_pyramid();
}

/// The Player's full observation at c: 4×81 greys + 8 slope values = 332.
#[pyfunction]
fn player_observation_py(py: Python, c_re: f64, c_im: f64) -> PyResult<Vec<f32>> {
    crate::minimap::with_pyramid(|pyr| {
        let pyr = pyr.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("mip pyramid not loaded; call load_mip_pyramid_py first")
        })?;
        let c = num_complex::Complex64::new(c_re, c_im);
        pyr.player_observation(c)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("c outside map extent"))
            .map_err(|e| e.into())
            .map(|obs| {
                let _ = py;
                obs
            })
    })
}

/// Slope of the shore-proximity field at c on a level.
#[pyfunction]
fn minimap_slope_py(c_re: f64, c_im: f64, level: usize) -> PyResult<(f64, f64)> {
    crate::minimap::with_pyramid(|pyr| {
        let pyr = pyr.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("mip pyramid not loaded")
        })?;
        let c = num_complex::Complex64::new(c_re, c_im);
        pyr.slope(c, level)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("c outside map extent"))
    })
}

/// Batch shore proximity (S field) sampled at points on a mip level.
#[pyfunction]
fn minimap_shore_proximity_batch_py(
    re: Vec<f64>,
    im: Vec<f64>,
    level: usize,
) -> PyResult<Vec<f32>> {
    if re.len() != im.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "re/im length mismatch",
        ));
    }
    crate::minimap::with_pyramid(|pyr| {
        let pyr = pyr.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("mip pyramid not loaded")
        })?;
        let mut out = Vec::with_capacity(re.len());
        for (&r, &i) in re.iter().zip(im.iter()) {
            let c = num_complex::Complex64::new(r, i);
            let (fx, fy) = pyr
                .world_to_texel_pub(level, c)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad level"))?;
            let cx = fx.round() as isize;
            let cy = fy.round() as isize;
            out.push(pyr.sample_field_pub(level, cx, cy));
        }
        Ok(out)
    })
}

/// Contour-biased integrator step for Physics. Returns (new_re, new_im).
#[pyfunction]
#[pyo3(signature = (c_re, c_im, u_re, u_im, h, d_star, max_step, level, energy=0.0))]
fn contour_biased_step_py(
    c_re: f64,
    c_im: f64,
    u_re: f64,
    u_im: f64,
    h: f64,
    d_star: f64,
    max_step: f64,
    level: usize,
    energy: f64,
) -> PyResult<(f64, f64)> {
    crate::minimap::contour_biased_step(c_re, c_im, u_re, u_im, h, d_star, max_step, level, energy)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Mandelbrot distance estimate. Accepts:
/// 1) `mandelbrot_distance_estimate(coords: Sequence[complex])` -> list[float]
/// 2) `mandelbrot_distance_estimate((x_seq, y_seq))` -> list[float]
/// 3) `mandelbrot_distance_estimate(xs, ys)` (two equal-length sequences)
///
/// Signed distances: positive outside the set, non-positive inside.
#[pyfunction]
#[pyo3(signature = (coords, ys=None))]
fn mandelbrot_distance_estimate_py(
    coords: &Bound<'_, PyAny>,
    ys: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<f32>> {
    // Two-sequence form: (xs, ys)
    if let Some(ys) = ys {
        let xv: Vec<f64> = coords.extract()?;
        let yv: Vec<f64> = ys.extract()?;
        if xv.len() != yv.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "x and y sequences must have equal length",
            ));
        }
        let cs: Vec<num_complex::Complex64> = xv
            .iter()
            .zip(yv.iter())
            .map(|(&xr, &yr)| num_complex::Complex64::new(xr, yr))
            .collect();
        return crate::distance_field::mandelbrot_distance_estimate(&cs)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err);
    }

    // Tuple-of-sequences form: (xs, ys)
    if let Ok((xv, yv)) = coords.extract::<(Vec<f64>, Vec<f64>)>() {
        if xv.len() != yv.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "x and y sequences must have equal length",
            ));
        }
        let cs: Vec<num_complex::Complex64> = xv
            .iter()
            .zip(yv.iter())
            .map(|(&xr, &yr)| num_complex::Complex64::new(xr, yr))
            .collect();
        return crate::distance_field::mandelbrot_distance_estimate(&cs)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err);
    }

    // Sequence of complex-like values
    let mut cs: Vec<num_complex::Complex64> = Vec::new();
    match coords.iter() {
        Ok(seq) => {
            for item in seq {
                let el = item?;
                if let Ok((r, i)) = el.extract::<(f64, f64)>() {
                    cs.push(num_complex::Complex64::new(r, i));
                    continue;
                }
                if let (Ok(rp), Ok(ip)) = (el.getattr("real"), el.getattr("imag")) {
                    let r: f64 = rp.extract()?;
                    let i: f64 = ip.extract()?;
                    cs.push(num_complex::Complex64::new(r, i));
                    continue;
                }
                if let Ok(r) = el.extract::<f64>() {
                    cs.push(num_complex::Complex64::new(r, 0.0));
                    continue;
                }
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "coords must be complex-like values or (real, imag) tuples",
                ));
            }
        }
        Err(_) => {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "coords must be a sequence of complex-like values or an (xs, ys) pair",
            ))
        }
    }

    crate::distance_field::mandelbrot_distance_estimate(&cs)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
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

/// Python wrapper for the canonical sample-clock timebase (issue #91).
///
/// This is the TRAINING surface of the AnalysisTimebase: the backend must
/// ingest audio through the same Rust timebase the browser uses (via wasm),
/// so training and runtime execute the same resampling / hop-scheduling /
/// epoch pipeline rather than merely containing equivalent components.
#[pyclass]
pub struct AnalysisTimebase {
    inner: crate::timebase::AnalysisTimebase,
}

// ---------------------------------------------------------------------------
// CycleBank (issue #92) — TRAINING / offline-diagnostics surface.
//
// This is the SAME Rust CycleBank the browser runs via wasm-orbit. Python
// only orchestrates: it feeds canonical ticks in and reads observed modes /
// relations / predictions out. ALL transform, ridge, tracking, frequency,
// phase, confidence, relation, and prediction math stays in Rust (ADR 0001,
// ADR 0002); nothing here recomputes any of it.
// ---------------------------------------------------------------------------

/// Python wrapper for the canonical observed-ridge `CycleBank`.
///
/// Construct with no arguments for the canonical defaults, or pass a config
/// dict to override selected `CycleBankConfig` fields. Then drive it either
/// from canonical ticks (`observe_tick`) or from explicit observations
/// (`observe`). Predictive queries (`phase_at`, `time_to_next`) are read off
/// the returned mode dicts.
#[pyclass]
pub struct CycleBank {
    inner: crate::cycle_bank::CycleBank,
}

/// Serialize one observed `CycleMode` as a Python dict.
///
/// Keys are **camelCase** to match the wasm binding's `CycleMode` interface,
/// so a mode read by the trainer and by the browser has the same shape
/// (cross-surface parity, issue #93).
fn cycle_mode_to_pydict(
    py: Python,
    m: &crate::cycle_bank::CycleMode,
) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    let d = PyDict::new_bound(py);
    d.set_item("id", m.id)?;
    d.set_item("frequencyHz", m.frequency_hz)?;
    d.set_item("phase", m.phase)?;
    d.set_item("strength", m.strength)?;
    d.set_item("confidence", m.confidence)?;
    d.set_item("channelSupport", m.channel_support)?;
    d.set_item("age", m.age)?;
    d.set_item("missingObservations", m.missing_observations)?;
    d.set_item("frequencySlope", m.frequency_slope)?;
    d.set_item("frequencyUncertainty", m.frequency_uncertainty)?;
    Ok(d.into())
}

/// Serialize one observed-mode rational `CycleRelation` as a Python dict.
fn cycle_relation_to_pydict(
    py: Python,
    r: &crate::cycle_bank::CycleRelation,
) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    let d = PyDict::new_bound(py);
    d.set_item("iId", r.i_id)?;
    d.set_item("jId", r.j_id)?;
    d.set_item("m", r.m)?;
    d.set_item("n", r.n)?;
    d.set_item("freqResidual", r.freq_residual)?;
    d.set_item("generalizedPhase", r.generalized_phase)?;
    d.set_item("phaseStability", r.phase_stability)?;
    Ok(d.into())
}

/// Rebuild the canonical `CycleBankConfig`, applying any overrides from a
/// Python dict. Unknown keys are rejected loudly so a typo cannot silently
/// fall back to a default (the same strictness the ONNX metadata uses).
fn cycle_bank_config_from_dict(
    overrides: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<crate::cycle_bank::CycleBankConfig> {
    let mut cfg = crate::cycle_bank::CycleBankConfig::default();
    let Some(dict) = overrides else {
        return Ok(cfg);
    };
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "f_min_hz" => cfg.f_min_hz = value.extract()?,
            "f_max_hz" => cfg.f_max_hz = value.extract()?,
            "q_cycles" => cfg.q_cycles = value.extract()?,
            "scales_per_octave" => cfg.scales_per_octave = value.extract()?,
            "weak_threshold" => cfg.weak_threshold = value.extract()?,
            "max_modes" => cfg.max_modes = value.extract()?,
            "association_log_freq_tolerance" => {
                cfg.association_log_freq_tolerance = value.extract()?
            }
            "association_phase_tolerance_rad" => {
                cfg.association_phase_tolerance_rad = value.extract()?
            }
            "phase_correction_gain" => cfg.phase_correction_gain = value.extract()?,
            "frequency_smoothing" => cfg.frequency_smoothing = value.extract()?,
            "strength_smoothing" => cfg.strength_smoothing = value.extract()?,
            "slope_smoothing" => cfg.slope_smoothing = value.extract()?,
            "birth_persistence" => cfg.birth_persistence = value.extract()?,
            "free_run_max_observations" => {
                cfg.free_run_max_observations = value.extract()?
            }
            "death_persistence" => cfg.death_persistence = value.extract()?,
            "missing_strength_decay" => cfg.missing_strength_decay = value.extract()?,
            "merge_log_freq_tolerance" => cfg.merge_log_freq_tolerance = value.extract()?,
            "merge_phase_tolerance_rad" => {
                cfg.merge_phase_tolerance_rad = value.extract()?
            }
            "max_numer" => cfg.max_numer = value.extract()?,
            "max_denom" => cfg.max_denom = value.extract()?,
            "rational_ratio_tolerance" => cfg.rational_ratio_tolerance = value.extract()?,
            "relation_history_len" => cfg.relation_history_len = value.extract()?,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown CycleBankConfig key: {other}"
                )))
            }
        }
    }
    Ok(cfg)
}

/// Reconstruct a `crate::timebase::AnalysisTick` from the camelCase dict the
/// `AnalysisTimebase` binding emits, then route it through the canonical
/// Rust seam (`cycle_observation_from_tick`).  Python never computes the
/// newest-frame offset itself.
fn tick_from_pydict(
    dict: &pyo3::Bound<'_, pyo3::types::PyDict>,
) -> PyResult<crate::timebase::AnalysisTick> {
    let features: Vec<f64> = dict
        .get_item("features")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("features"))?
        .extract()?;
    let sample_index: u64 = dict
        .get_item("sampleIndex")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("sampleIndex"))?
        .extract()?;
    let time_seconds: f64 = dict
        .get_item("timeSeconds")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("timeSeconds"))?
        .extract()?;
    let dt_seconds: f64 = dict
        .get_item("dtSeconds")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("dtSeconds"))?
        .extract()?;
    let stream_epoch: u64 = dict
        .get_item("streamEpoch")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("streamEpoch"))?
        .extract()?;
    Ok(crate::timebase::AnalysisTick {
        features,
        sample_index,
        time_seconds,
        dt_seconds,
        stream_epoch,
    })
}

#[pymethods]
impl CycleBank {
    #[new]
    #[pyo3(signature = (config=None))]
    fn py_new(
        config: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let cfg = cycle_bank_config_from_dict(config)?;
        let inner = crate::cycle_bank::CycleBank::try_new(cfg)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// The Rust-owned contract version (`CYCLE_BANK_VERSION`). Python reads
    /// it; it never restates it.
    #[getter]
    fn version(&self) -> &'static str {
        self.inner.version()
    }

    /// Feed one canonical analysis tick (the dict shape returned by
    /// `AnalysisTimebase.ingest` / `flush`). The newest-frame -> observation
    /// mapping is done in Rust by the canonical seam; Python passes the tick
    /// through unchanged. Returns the current observed modes (list of dicts).
    fn observe_tick<'py>(
        &mut self,
        py: Python<'py>,
        tick: &pyo3::Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let tick = tick_from_pydict(tick)?;
        let obs = crate::timebase::cycle_observation_from_tick(&tick).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "tick feature window is not the expected frame-major shape",
            )
        })?;
        self.inner
            .observe(&obs)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.modes(py)
    }

    /// Feed one explicit observation of named scalar evidence channels.
    ///
    /// `sample_index` / `dt_seconds` / `stream_epoch` carry the #91 sample
    /// clock; `channels` is a sequence of `(name, value)` pairs. This entry
    /// point exists for synthetic diagnostics; the production path is
    /// `observe_tick`. Returns the current observed modes.
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
        sample_index: u64,
        dt_seconds: f64,
        stream_epoch: u64,
        channels: Vec<(String, f64)>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let obs = crate::cycle_bank::CycleObservation {
            sample_index,
            dt_seconds,
            stream_epoch,
            channels: channels
                .into_iter()
                .map(|(name, value)| crate::cycle_bank::CycleEvidenceChannel::new(name, value))
                .collect(),
        };
        self.inner
            .observe(&obs)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.modes(py)
    }

    /// Current confirmed observed modes (list of camelCase dicts).
    fn modes<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.inner
            .modes()
            .iter()
            .map(|m| cycle_mode_to_pydict(py, m).map(|o| o.into_bound(py)))
            .collect()
    }

    /// Rational relations among the currently observed modes (latest batch).
    fn latest_relations<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.inner
            .latest_relations()
            .iter()
            .map(|r| cycle_relation_to_pydict(py, r).map(|o| o.into_bound(py)))
            .collect()
    }

    /// Number of currently confirmed modes.
    fn num_modes(&self) -> usize {
        self.inner.num_modes()
    }

    /// Deterministic discontinuity reset (also triggered automatically by a
    /// `streamEpoch` change in the incoming tick).
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Causal free-running phase prediction for one mode, `delta_seconds`
    /// into the future, using that mode's continuous phase/frequency.
    /// Pure Rust math on already-emitted state.
    #[staticmethod]
    fn phase_at(phase: f64, frequency_hz: f64, delta_seconds: f64) -> f64 {
        let mode = crate::cycle_bank::CycleMode {
            id: 0,
            frequency_hz,
            phase,
            strength: 0.0,
            confidence: 0.0,
            channel_support: 0.0,
            age: 0,
            missing_observations: 0,
            frequency_slope: 0.0,
            frequency_uncertainty: 0.0,
        };
        mode.phase_at(delta_seconds)
    }
}

/// A single emitted analysis tick, materialized as a Python dict.
///
/// Wire format keys are **camelCase** so a tick read by the trainer
/// (via this binding) is keyed identically to a tick received by the
/// browser (via the wasm binding's ``AnalysisTick`` interface).
/// Cross-surface parity contract, issue #93 strict-version review.
fn tick_to_pydict(py: Python, t: crate::timebase::AnalysisTick) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    let dict = PyDict::new_bound(py);
    dict.set_item("features", t.features)?;
    dict.set_item("sampleIndex", t.sample_index)?;
    dict.set_item("timeSeconds", t.time_seconds)?;
    dict.set_item("dtSeconds", t.dt_seconds)?;
    dict.set_item("streamEpoch", t.stream_epoch)?;
    Ok(dict.into())
}

#[pymethods]
impl AnalysisTimebase {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: crate::timebase::AnalysisTimebase::new(),
        }
    }

    /// Ingest one non-overlapping PCM block. Returns a list of tick dicts
    /// (possibly empty). Raises ValueError on overlap / mid-stream rate
    /// change (transport bugs, not discontinuities).
    fn ingest<'py>(
        &mut self,
        py: Python<'py>,
        samples: Vec<f32>,
        source_sample_rate: usize,
        source_start_frame: u64,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let ticks = self
            .inner
            .ingest(&samples, source_sample_rate, source_start_frame)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        ticks
            .into_iter()
            .map(|t| {
                tick_to_pydict(py, t)
                    .map(|obj| obj.into_bound(py))
            })
            .collect()
    }

    /// Flush end-of-stream (recovers the deferred final sample/tick).
    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.inner
            .flush()
            .into_iter()
            .map(|t| {
                tick_to_pydict(py, t)
                    .map(|obj| obj.into_bound(py))
            })
            .collect()
    }

    /// Declare a stream discontinuity (start/stop/source replacement).
    /// Bumps the epoch and resets the hop schedule.
    fn reset(&mut self) {
        self.inner.reset(crate::timebase::ResetReason::SourceReplacement);
    }

    /// Diagnostic snapshot as a Python dict.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let d = self.inner.diagnostics();
        let dict = PyDict::new_bound(py);
        // camelCase to match the wasm binding's ``TimebaseDiagnostics``
        // interface (issue #93 strict-version review).
        dict.set_item("sourceSampleRate", d.source_sample_rate)?;
        dict.set_item("sourceFramesIngested", d.source_frames_ingested)?;
        dict.set_item("canonicalSampleIndex", d.canonical_sample_index)?;
        dict.set_item("analysisHopCount", d.analysis_hop_count)?;
        dict.set_item("timeSeconds", d.time_seconds)?;
        dict.set_item("streamEpoch", d.stream_epoch)?;
        dict.set_item("detectedGaps", d.detected_gaps)?;
        dict.set_item("detectedOverlaps", d.detected_overlaps)?;
        dict.set_item("lastSourceStartFrame", d.last_source_start_frame)?;
        dict.set_item("lastSourceEndFrame", d.last_source_end_frame)?;
        Ok(dict.into())
    }
}

/// Free function: compute a point on the Mandelbrot lobe in Python.
#[pyfunction]
#[pyo3(signature = (lobe, sub_lobe, theta, s))]
fn lobe_point_at_angle(py: Python, lobe: u32, sub_lobe: u32, theta: f64, s: f64) -> PyResult<Py<PyComplex>> {
    let c = rust_lobe_point_at_angle(lobe, sub_lobe, theta, s);
    Ok(PyComplex::from_doubles_bound(py, c.re, c.im).into())
}

/// Free function: compute cardioid-boundary proximity for a batch of points.
#[pyfunction]
#[pyo3(signature = (coords))]
fn mandelbrot_cardioid_proximity_batch(
    coords: Vec<Bound<'_, PyComplex>>,
) -> PyResult<Vec<f64>> {
    let cs: Vec<num_complex::Complex64> = coords
        .iter()
        .map(|c| num_complex::Complex64::new(c.real(), c.imag()))
        .collect();
    Ok(rust_proxies::mandelbrot_cardioid_proximity_batch(&cs))
}

/// Free function: compute orbit path metrics over a c(t) trajectory.
#[pyfunction]
#[pyo3(signature = (coords))]
fn orbit_path_metrics_py(coords: Vec<Bound<'_, PyComplex>>) -> PyResult<(f64, f64, f64)> {
    let pts: Vec<num_complex::Complex64> = coords
        .iter()
        .map(|c| num_complex::Complex64::new(c.real(), c.imag()))
        .collect();
    let m = rust_proxies::orbit_path_metrics(&pts);
    Ok((m.mean_speed, m.max_speed, m.spread))
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
    m.add("CONTROLLER_VERSION", CONTROLLER_VERSION)?;
    // Feature-extraction contract (ADR 0001): version + pinned epsilon.
    m.add("FEATURE_VERSION", crate::features::FEATURE_VERSION)?;
    m.add("NORM_EPS", crate::features::NORM_EPS)?;
    // Analysis-pipeline contract (issue #93): how audio reaches the
    // extractor (resampling ownership, hop scheduling, epoch semantics).
    // Distinct from FEATURE_VERSION (the formulas). Stamped into ONNX
    // metadata; the browser refuses mismatches.
    m.add(
        "ANALYSIS_PIPELINE_VERSION",
        crate::timebase::ANALYSIS_PIPELINE_VERSION,
    )?;
    // Observed-ridge CycleBank contract (issue #92). Rust-owned; Python and
    // the browser read it and never restate it.
    m.add("CYCLE_BANK_VERSION", crate::cycle_bank::CYCLE_BANK_VERSION)?;

    m.add_class::<ResidualParams>()?;
    m.add_class::<OrbitState>()?;
    m.add_class::<PlayerState>()?;
    m.add_class::<OrbitController>()?;

    m.add_class::<FeatureExtractor>()?;

    m.add_class::<AnalysisTimebase>()?;

    m.add_class::<CycleBank>()?;

    m.add_class::<RuntimeVisualMetrics>()?;

    m.add_function(wrap_pyfunction!(lobe_point_at_angle, m)?)?;
    m.add_function(wrap_pyfunction!(compute_runtime_visual_metrics, m)?)?;
    // Controller phase generation (shared with training for parity)
    m.add_function(wrap_pyfunction!(residual_phases_for_seed_py, m)?)?;
    // Distance-field helpers
    m.add_function(wrap_pyfunction!(set_distance_field_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_distance_field_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_builtin_distance_field_py, m)?)?;
    // Minimap / mip pyramid (the Player's windows onto the Map)
    m.add_function(wrap_pyfunction!(load_mip_pyramid_py, m)?)?;
    m.add_function(wrap_pyfunction!(install_pyramid_py, m)?)?;
    m.add_function(wrap_pyfunction!(clear_pyramid_py, m)?)?;
    m.add_function(wrap_pyfunction!(player_observation_py, m)?)?;
    m.add_function(wrap_pyfunction!(minimap_slope_py, m)?)?;
    m.add_function(wrap_pyfunction!(minimap_shore_proximity_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(contour_biased_step_py, m)?)?;
    m.add_function(wrap_pyfunction!(mandelbrot_distance_estimate_py, m)?)?;
    // Alias without the _py suffix (tests and distance_utils use this name)
    m.add("mandelbrot_distance_estimate", wrap_pyfunction!(mandelbrot_distance_estimate_py, m)?)?;
    // Differentiable-proxy reference implementations (training supervision)
    m.add_function(wrap_pyfunction!(mandelbrot_cardioid_proximity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(orbit_path_metrics_py, m)?)?;
    m.add_function(wrap_pyfunction!(__getattr__, m)?)?;
    m.add_function(wrap_pyfunction!(__getattr__, m)?)?;
    Ok(())
}
