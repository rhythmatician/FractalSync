//! Python bindings for runtime‑core
//!
//! This module exposes the Rust runtime via PyO3.  It defines
//! Python classes and functions that mirror the structs and free
//! functions in `geometry`, `controller` and `features`.  These
//! bindings allow the Python backend to import a compiled
//! `runtime_core` module and call the shared logic directly.

use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict};

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
use crate::controls::{
    ControlsV2 as RustControlsV2,
    MotionControls as RustMotionControls,
    JuliaViewControls as RustJuliaViewControls,
    JuliaViewState as RustJuliaViewState,
    ColorIntent as RustColorIntent,
    Harmony as RustHarmony,
    CONTROLS_VERSION,
};


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

/// Python wrapper for the manifold configuration (issue #106).
///
/// Controls the induced metric, native potential, and integration on the
/// Mandelbrot configuration manifold. All fields are get/set so the
/// trainer can tune them and the runtime can read them back.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ManifoldConfig {
    #[pyo3(get, set)]
    pub d_ref: f64,
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub lambda_sq: f64,
    #[pyo3(get, set)]
    pub kappa: f64,
}

#[pymethods]
impl ManifoldConfig {
    #[new]
    #[pyo3(signature = (d_ref=0.1, epsilon=1e-4, lambda_sq=1.0, kappa=1.0))]
    fn py_new(d_ref: f64, epsilon: f64, lambda_sq: f64, kappa: f64) -> Self {
        Self {
            d_ref,
            epsilon,
            lambda_sq,
            kappa,
        }
    }
}

impl From<RustManifoldConfig> for ManifoldConfig {
    fn from(c: RustManifoldConfig) -> Self {
        Self {
            d_ref: c.d_ref,
            epsilon: c.epsilon,
            lambda_sq: c.lambda_sq,
            kappa: c.kappa,
        }
    }
}

impl From<ManifoldConfig> for RustManifoldConfig {
    fn from(c: ManifoldConfig) -> RustManifoldConfig {
        RustManifoldConfig {
            d_ref: c.d_ref,
            epsilon: c.epsilon,
            lambda_sq: c.lambda_sq,
            kappa: c.kappa,
        }
    }
}

/// Python wrapper for the energy diagnostic returned by integrate_step.
#[pyclass]
#[derive(Clone, Debug)]
pub struct EnergyInfo {
    #[pyo3(get)]
    pub kinetic: f64,
    #[pyo3(get)]
    pub potential: f64,
    #[pyo3(get)]
    pub total: f64,
    #[pyo3(get)]
    pub delta_total: f64,
    #[pyo3(get)]
    pub delta_kinetic: f64,
}

impl From<RustEnergyInfo> for EnergyInfo {
    fn from(e: RustEnergyInfo) -> Self {
        Self {
            kinetic: e.kinetic,
            potential: e.potential,
            total: e.total,
            delta_total: e.delta_total,
            delta_kinetic: e.delta_kinetic,
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

    // ---- Manifold physics (issue #106) ----

    /// Enable or disable manifold physics. When on, step() routes through a
    /// LEGACY ADAPTER that translates the old (s, alpha, energy) servo into a
    /// generalized force covector for the musically-ignorant manifold kernel.
    /// Transitional; not destination Controls v2 (issue #107).
    fn set_manifold_physics(&mut self, on: bool) {
        self.inner.manifold_physics = on;
    }

    /// Whether manifold physics is currently enabled.
    #[getter]
    fn manifold_physics(&self) -> bool {
        self.inner.manifold_physics
    }

    /// The most recent manifold-physics failure, if any. When manifold mode is
    /// selected and the integrator fails, the controller fails closed (holds
    /// the last valid state) and records the error here.
    #[getter]
    fn manifold_error(&self) -> Option<String> {
        self.inner.manifold_error.clone()
    }

    /// Set the manifold configuration (used only when manifold_physics is on).
    fn set_manifold_config(&mut self, config: ManifoldConfig) {
        self.inner.manifold_config = config.into();
    }

    /// Get the current manifold configuration.
    #[getter]
    fn manifold_config(&self) -> ManifoldConfig {
        self.inner.manifold_config.clone().into()
    }

    /// Set the drag coefficient for manifold physics (beta in Q_drag = -beta*G*v).
    fn set_manifold_drag(&mut self, drag: f64) {
        self.inner.manifold_drag = drag;
    }

    /// Get the drag coefficient for manifold physics.
    #[getter]
    fn manifold_drag(&self) -> f64 {
        self.inner.manifold_drag
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

// ---------------------------------------------------------------------------
// Manifold physics (issue #106) — Python surface.
//
// These are the TRAINING surface of the manifold math: the trainer's
// differentiable mirror (see backend/src/cspace_proxies.py) must reproduce
// these values within tolerance, enforced by preflight parity checks.
// Rust remains canonical under ADR 0001.
// ---------------------------------------------------------------------------

/// Signed distance to the Mandelbrot boundary. Positive outside, negative inside.
#[pyfunction]
fn manifold_signed_distance(c: &Bound<'_, PyComplex>) -> PyResult<f64> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_signed_distance(cc).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Regularized distance rho(c) = sqrt(D^2 + epsilon^2).
#[pyfunction]
#[pyo3(signature = (c, epsilon))]
fn manifold_regularized_distance(c: &Bound<'_, PyComplex>, epsilon: f64) -> PyResult<f64> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_regularized_distance(cc, epsilon).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Mandelbrot scale sigma(c) = log2(d_ref / rho(c)).
#[pyfunction]
fn manifold_mandelbrot_scale(c: &Bound<'_, PyComplex>, config: ManifoldConfig) -> PyResult<f64> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_mandelbrot_scale(cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Scale gradient ∇sigma(c) = (∂sigma/∂x, ∂sigma/∂y).
#[pyfunction]
fn manifold_scale_gradient(c: &Bound<'_, PyComplex>, config: ManifoldConfig) -> PyResult<(f64, f64)> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_scale_gradient(cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Scale Hessian [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]].
#[pyfunction]
fn manifold_scale_hessian(c: &Bound<'_, PyComplex>, config: ManifoldConfig) -> PyResult<Vec<Vec<f64>>> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    let h = rust_scale_hessian(cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(vec![vec![h[0][0], h[0][1]], vec![h[1][0], h[1][1]]])
}

/// Induced metric G(c) = I + lambda^2 * grad_sigma * grad_sigma^T.
/// Returns [[g11, g12], [g12, g22]].
#[pyfunction]
fn manifold_induced_metric(c: &Bound<'_, PyComplex>, config: ManifoldConfig) -> PyResult<Vec<Vec<f64>>> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    let g = rust_induced_metric(cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(vec![vec![g[0][0], g[0][1]], vec![g[1][0], g[1][1]]])
}

/// Kinetic energy K = 1/2 v^T G v.
#[pyfunction]
#[pyo3(signature = (vx, vy, c, config))]
fn manifold_kinetic_energy(
    vx: f64,
    vy: f64,
    c: &Bound<'_, PyComplex>,
    config: ManifoldConfig,
) -> PyResult<f64> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_kinetic_energy((vx, vy), cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Native potential U = kappa * sigma(c).
#[pyfunction]
fn manifold_potential_energy(c: &Bound<'_, PyComplex>, config: ManifoldConfig) -> PyResult<f64> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_potential_energy(cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Total mechanical energy E = K + U.
#[pyfunction]
#[pyo3(signature = (vx, vy, c, config))]
fn manifold_total_energy(
    vx: f64,
    vy: f64,
    c: &Bound<'_, PyComplex>,
    config: ManifoldConfig,
) -> PyResult<f64> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_total_energy((vx, vy), cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Christoffel symbols Gamma^i_jk of the Levi-Civita connection.
/// Returns [[[g00, g01], [g10, g11]], [[g20, g21], [g30, g31]]]
#[pyfunction]
fn manifold_christoffel_symbols(c: &Bound<'_, PyComplex>, config: ManifoldConfig) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    let g = rust_christoffel_symbols(cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(vec![
        vec![vec![g[0][0][0], g[0][0][1]], vec![g[0][1][0], g[0][1][1]]],
        vec![vec![g[1][0][0], g[1][0][1]], vec![g[1][1][0], g[1][1][1]]],
    ])
}

/// Geodesic acceleration term: Gamma^i_jk v^j v^k.
#[pyfunction]
#[pyo3(signature = (vx, vy, c, config))]
fn manifold_geodesic_acceleration(
    vx: f64,
    vy: f64,
    c: &Bound<'_, PyComplex>,
    config: ManifoldConfig,
) -> PyResult<(f64, f64)> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_geodesic_acceleration((vx, vy), cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Generalized potential force covector: Q_potential = -grad U = -kappa grad sigma.
///
/// This is a generalized force COVECTOR (lower index), not a coordinate
/// acceleration. Convert to acceleration with `manifold_apply_generalized_force`.
#[pyfunction]
fn manifold_potential_force(c: &Bound<'_, PyComplex>, config: ManifoldConfig) -> PyResult<(f64, f64)> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_potential_force(cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Convert a generalized force covector to coordinate acceleration: a = G^{-1} Q.
///
/// This is the single place where the metric inverse maps a generalized force
/// covector into coordinate acceleration.
#[pyfunction]
#[pyo3(signature = (qx, qy, c, config))]
fn manifold_apply_generalized_force(
    qx: f64,
    qy: f64,
    c: &Bound<'_, PyComplex>,
    config: ManifoldConfig,
) -> PyResult<(f64, f64)> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_apply_generalized_force((qx, qy), cc, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Metric-consistent isotropic drag covector: Q_drag = -beta G v.
///
/// This is a generalized force COVECTOR (lower index), not a coordinate
/// acceleration. Its power P = v^T Q_drag <= 0, so drag never injects energy.
#[pyfunction]
#[pyo3(signature = (vx, vy, c, beta, config))]
fn manifold_drag_force(
    vx: f64,
    vy: f64,
    c: &Bound<'_, PyComplex>,
    beta: f64,
    config: ManifoldConfig,
) -> PyResult<(f64, f64)> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    rust_drag_force((vx, vy), cc, beta, &config.into()).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Semi-implicit Euler integration step for manifold dynamics.
///
/// Integrates: r_ddot + Gamma(r_dot, r_dot) = -G^{-1}∇U + G^{-1}Q
///
/// Returns (new_re, new_im, new_vx, new_vy, energy_info).
#[pyfunction]
#[pyo3(signature = (c_re, c_im, vx, vy, qx, qy, beta, dt, config))]
fn manifold_integrate_step(
    c_re: f64,
    c_im: f64,
    vx: f64,
    vy: f64,
    qx: f64,
    qy: f64,
    beta: f64,
    dt: f64,
    config: ManifoldConfig,
) -> PyResult<(f64, f64, f64, f64, EnergyInfo)> {
    let c = num_complex::Complex64::new(c_re, c_im);
    let (c_new, v_new, info) = rust_integrate_step(
        c,
        (vx, vy),
        (qx, qy),
        beta,
        dt,
        &config.into(),
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok((
        c_new.re,
        c_new.im,
        v_new.0,
        v_new.1,
        info.into(),
    ))
}

// ---------------------------------------------------------------------------
// Controls v2 (issue #107) — Python surface.
// ---------------------------------------------------------------------------

/// Python wrapper for MotionControls (2D drive + brake/grip + impulse).
#[pyclass]
#[derive(Clone, Debug)]
pub struct MotionControls {
    #[pyo3(get, set)]
    pub drive_x: f64,
    #[pyo3(get, set)]
    pub drive_y: f64,
    #[pyo3(get, set)]
    pub brake: f64,
    #[pyo3(get, set)]
    pub grip: f64,
    #[pyo3(get, set)]
    pub impulse_x: f64,
    #[pyo3(get, set)]
    pub impulse_y: f64,
}

#[pymethods]
impl MotionControls {
    #[new]
    #[pyo3(signature = (drive_x=0.0, drive_y=0.0, brake=0.0, grip=0.5, impulse_x=0.0, impulse_y=0.0))]
    fn py_new(drive_x: f64, drive_y: f64, brake: f64, grip: f64, impulse_x: f64, impulse_y: f64) -> Self {
        Self { drive_x, drive_y, brake, grip, impulse_x, impulse_y }
    }
    fn drive_magnitude(&self) -> f64 {
        let inner: RustMotionControls = self.clone().into();
        inner.drive_magnitude()
    }
    fn friction_beta(&self) -> f64 {
        let inner: RustMotionControls = self.clone().into();
        inner.friction_beta()
    }
    fn clamped(&self) -> Self {
        let inner: RustMotionControls = self.clone().into();
        inner.clamped().into()
    }
}

impl From<RustMotionControls> for MotionControls {
    fn from(m: RustMotionControls) -> Self {
        Self { drive_x: m.drive[0], drive_y: m.drive[1], brake: m.brake, grip: m.grip, impulse_x: m.impulse[0], impulse_y: m.impulse[1] }
    }
}
impl From<MotionControls> for RustMotionControls {
    fn from(m: MotionControls) -> RustMotionControls {
        RustMotionControls { drive: [m.drive_x, m.drive_y], brake: m.brake, grip: m.grip, impulse: [m.impulse_x, m.impulse_y] }.clamped()
    }
}

/// Python wrapper for JuliaViewControls (bounded deltas).
#[pyclass]
#[derive(Clone, Debug)]
pub struct JuliaViewControls {
    #[pyo3(get, set)]
    pub zoom_delta: f64,
    #[pyo3(get, set)]
    pub rotation_delta: f64,
    #[pyo3(get, set)]
    pub hue_delta: f64,
    #[pyo3(get, set)]
    pub chroma_delta: f64,
    #[pyo3(get, set)]
    pub lightness_delta: f64,
    #[pyo3(get, set)]
    pub accent_delta: f64,
    #[pyo3(get, set)]
    pub harmony_shift: f64,
}

#[pymethods]
impl JuliaViewControls {
    #[new]
    #[pyo3(signature = (zoom_delta=0.0, rotation_delta=0.0, hue_delta=0.0, chroma_delta=0.0, lightness_delta=0.0, accent_delta=0.0, harmony_shift=0.0))]
    fn py_new(zoom_delta: f64, rotation_delta: f64, hue_delta: f64, chroma_delta: f64, lightness_delta: f64, accent_delta: f64, harmony_shift: f64) -> Self {
        Self { zoom_delta, rotation_delta, hue_delta, chroma_delta, lightness_delta, accent_delta, harmony_shift }
    }
    fn clamped(&self) -> Self {
        let inner: RustJuliaViewControls = self.clone().into();
        inner.clamped().into()
    }
}

impl From<RustJuliaViewControls> for JuliaViewControls {
    fn from(v: RustJuliaViewControls) -> Self {
        Self { zoom_delta: v.zoom_delta, rotation_delta: v.rotation_delta, hue_delta: v.hue_delta, chroma_delta: v.chroma_delta, lightness_delta: v.lightness_delta, accent_delta: v.accent_delta, harmony_shift: v.harmony_shift }
    }
}
impl From<JuliaViewControls> for RustJuliaViewControls {
    fn from(v: JuliaViewControls) -> RustJuliaViewControls {
        RustJuliaViewControls { zoom_delta: v.zoom_delta, rotation_delta: v.rotation_delta, hue_delta: v.hue_delta, chroma_delta: v.chroma_delta, lightness_delta: v.lightness_delta, accent_delta: v.accent_delta, harmony_shift: v.harmony_shift }.clamped()
    }
}

/// Python wrapper for ControlsV2 (unified action surface).
#[pyclass]
#[derive(Clone, Debug)]
pub struct ControlsV2 {
    #[pyo3(get, set)]
    pub motion: MotionControls,
    #[pyo3(get, set)]
    pub view: JuliaViewControls,
}

#[pymethods]
impl ControlsV2 {
    #[new]
    fn py_new(motion: MotionControls, view: JuliaViewControls) -> Self {
        Self { motion, view }
    }
    #[staticmethod]
    fn from_model_output(output: Vec<f64>) -> PyResult<Self> {
        RustControlsV2::from_model_output(&output).map(|c| c.into()).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }
    fn to_model_output(&self) -> Vec<f64> {
        let inner: RustControlsV2 = self.clone().into();
        inner.to_model_output()
    }
    fn clamped(&self) -> Self {
        let inner: RustControlsV2 = self.clone().into();
        inner.clamped().into()
    }
    #[staticmethod]
    fn model_output_order() -> Vec<String> {
        RustControlsV2::model_output_order().into_iter().map(|s| s.to_string()).collect()
    }
}

impl From<RustControlsV2> for ControlsV2 {
    fn from(c: RustControlsV2) -> Self {
        Self { motion: c.motion.into(), view: c.view.into() }
    }
}
impl From<ControlsV2> for RustControlsV2 {
    fn from(c: ControlsV2) -> RustControlsV2 {
        RustControlsV2 { motion: c.motion.into(), view: c.view.into() }.clamped()
    }
}

/// Python wrapper for ColorIntent.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ColorIntent {
    #[pyo3(get, set)]
    pub anchor_hue: f64,
    #[pyo3(get, set)]
    pub chroma: f64,
    #[pyo3(get, set)]
    pub lightness: f64,
    #[pyo3(get, set)]
    pub harmony: String,
    #[pyo3(get, set)]
    pub accent_weight: f64,
}

#[pymethods]
impl ColorIntent {
    #[new]
    #[pyo3(signature = (anchor_hue=0.0, chroma=0.18, lightness=0.55, harmony="analogous".to_string(), accent_weight=0.35))]
    fn py_new(anchor_hue: f64, chroma: f64, lightness: f64, harmony: String, accent_weight: f64) -> Self {
        Self { anchor_hue, chroma, lightness, harmony, accent_weight }
    }
}

impl From<RustColorIntent> for ColorIntent {
    fn from(c: RustColorIntent) -> Self {
        Self { anchor_hue: c.anchor_hue, chroma: c.chroma, lightness: c.lightness, harmony: c.harmony.name().to_string(), accent_weight: c.accent_weight }
    }
}
impl From<ColorIntent> for RustColorIntent {
    fn from(c: ColorIntent) -> RustColorIntent {
        let harmony = match c.harmony.as_str() {
            "monochrome" => RustHarmony::Monochrome,
            "opponent" => RustHarmony::Opponent,
            _ => RustHarmony::Analogous,
        };
        RustColorIntent { anchor_hue: c.anchor_hue, chroma: c.chroma, lightness: c.lightness, harmony, accent_weight: c.accent_weight }.clamped()
    }
}

/// Python wrapper for JuliaViewState (persistent view state).
#[pyclass]
#[derive(Clone, Debug)]
pub struct JuliaViewState {
    #[pyo3(get, set)]
    pub zoom: f64,
    #[pyo3(get, set)]
    pub rotation: f64,
    #[pyo3(get, set)]
    pub color: ColorIntent,
}

#[pymethods]
impl JuliaViewState {
    #[new]
    #[pyo3(signature = (zoom=1.0, rotation=0.0, color=None))]
    fn py_new(zoom: f64, rotation: f64, color: Option<ColorIntent>) -> Self {
        Self { zoom, rotation, color: color.unwrap_or_else(|| RustColorIntent::default().into()) }
    }
    fn apply_controls(&mut self, controls: JuliaViewControls) {
        let mut inner: RustJuliaViewState = self.clone().into();
        inner.apply_controls(controls.into());
        *self = inner.into();
    }
    fn clamped(&self) -> Self {
        let inner: RustJuliaViewState = self.clone().into();
        inner.clamped().into()
    }
}

impl From<RustJuliaViewState> for JuliaViewState {
    fn from(s: RustJuliaViewState) -> Self {
        Self { zoom: s.zoom, rotation: s.rotation, color: s.color.into() }
    }
}
impl From<JuliaViewState> for RustJuliaViewState {
    fn from(s: JuliaViewState) -> RustJuliaViewState {
        RustJuliaViewState { zoom: s.zoom, rotation: s.rotation, color: s.color.into() }.clamped()
    }
}

/// Integrate one manifold step driven by MotionControls (destination physics seam).
/// Returns (new_re, new_im, new_vx, new_vy, energy_info).
#[pyfunction]
#[pyo3(signature = (c_re, c_im, vx, vy, motion, dt, config))]
fn controls_integrate_step(
    c_re: f64,
    c_im: f64,
    vx: f64,
    vy: f64,
    motion: MotionControls,
    dt: f64,
    config: ManifoldConfig,
) -> PyResult<(f64, f64, f64, f64, EnergyInfo)> {
    let c = num_complex::Complex64::new(c_re, c_im);
    let (c_new, v_new, info) = crate::controls::integrate_motion_controls(c, (vx, vy), &motion.into(), dt, &config.into()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok((c_new.re, c_new.im, v_new.0, v_new.1, info.into()))
}

/// Compute drive covector for inspection/diagnostics.
#[pyfunction]
#[pyo3(signature = (c, motion, config))]
fn motion_drive_covector(c: &Bound<'_, PyComplex>, motion: MotionControls, config: ManifoldConfig) -> PyResult<(f64, f64)> {
    let cc = num_complex::Complex64::new(c.real(), c.imag());
    crate::controls::MotionControls::from(motion).drive_covector(cc, &config.into()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
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
// ADR 0003); nothing here recomputes any of it.
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

/// One directly observed temporal ridge (issue #92), Python surface.
///
/// Read-only view of the Rust `CycleMode` state. Causal predictive queries
/// (`phase_at`, `time_to_next`) are methods on the mode so the trainer reads
/// predictions from the same Rust state the browser sees via the wasm
/// `CycleMode` interface. `to_dict()` produces the camelCase wire shape that
/// matches the browser's `CycleMode` interface (cross-surface parity).
#[pyclass]
#[derive(Clone, Debug)]
pub struct CycleMode {
    inner: crate::cycle_bank::CycleMode,
}

impl From<crate::cycle_bank::CycleMode> for CycleMode {
    fn from(inner: crate::cycle_bank::CycleMode) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl CycleMode {
    /// Canonical attribute schema for the `CycleMode` pyclass.
    ///
    /// Single Rust-owned source of truth that the Python stub generator
    /// (`scripts/generate_runtime_core_stubs.py`) reads instead of a
    /// hand-maintained `ATTR_TYPES["CycleMode"]` dict. Adding a new field to
    /// the Rust struct now updates the .pyi in lockstep — no scattered
    /// maintenance across the seven sites the issue #92 review flagged.
    ///
    /// Returns a dict `{attr_name: python_type_annotation}`.
    #[staticmethod]
    fn __fields__<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("id", "int")?;
        d.set_item("frequency_hz", "float")?;
        d.set_item("phase", "float")?;
        d.set_item("strength", "float")?;
        d.set_item("confidence", "float")?;
        d.set_item("channel_support", "float")?;
        d.set_item("age", "int")?;
        d.set_item("missing_observations", "int")?;
        d.set_item("frequency_slope", "float")?;
        d.set_item("frequency_uncertainty", "float")?;
        Ok(d)
    }

    /// Canonical method/function schema for the `CycleMode` pyclass.
    /// Mirrors `__fields__` but for callable members; returns
    /// `{method_name: {param_name: annotation, "__return__": annotation}}`.
    #[staticmethod]
    fn __methods__<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        let phase_at = PyDict::new_bound(py);
        phase_at.set_item("delta_seconds", "float")?;
        phase_at.set_item("__return__", "float")?;
        d.set_item("phase_at", phase_at)?;

        let time_to_next = PyDict::new_bound(py);
        time_to_next.set_item("reference_phase", "float")?;
        time_to_next.set_item("__return__", "Optional[float]")?;
        d.set_item("time_to_next", time_to_next)?;

        let to_dict = PyDict::new_bound(py);
        to_dict.set_item("__return__", "dict")?;
        d.set_item("to_dict", to_dict)?;
        Ok(d)
    }

    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }
    #[getter]
    fn frequency_hz(&self) -> f64 {
        self.inner.frequency_hz
    }
    #[getter]
    fn phase(&self) -> f64 {
        self.inner.phase
    }
    #[getter]
    fn strength(&self) -> f64 {
        self.inner.strength
    }
    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }
    #[getter]
    fn channel_support(&self) -> f64 {
        self.inner.channel_support
    }
    #[getter]
    fn age(&self) -> u64 {
        self.inner.age
    }
    #[getter]
    fn missing_observations(&self) -> u64 {
        self.inner.missing_observations
    }
    #[getter]
    fn frequency_slope(&self) -> f64 {
        self.inner.frequency_slope
    }
    #[getter]
    fn frequency_uncertainty(&self) -> f64 {
        self.inner.frequency_uncertainty
    }

    /// Causal free-running phase prediction `delta_seconds` into the future.
    fn phase_at(&self, delta_seconds: f64) -> f64 {
        self.inner.phase_at(delta_seconds)
    }

    /// Time until the mode next reaches `reference_phase`, assuming constant
    /// current frequency. `None` if the frequency is not positive/finite.
    fn time_to_next(&self, reference_phase: f64) -> Option<f64> {
        self.inner.time_to_next(reference_phase)
    }

    /// CamelCase dict matching the wasm `CycleMode` interface, so the
    /// trainer and the browser see the same wire shape (issue #93 parity).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let m = &self.inner;
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
        Ok(d.into_any())
    }
}

/// Serialize one observed `CycleMode` as a Python dict (camelCase wire shape).
///
/// Keys are **camelCase** to match the wasm binding's `CycleMode` interface,
/// so a mode read by the trainer and by the browser has the same shape
/// (cross-surface parity, issue #93).
fn cycle_mode_to_pydict(
    py: Python,
    m: &crate::cycle_bank::CycleMode,
) -> PyResult<PyObject> {
    CycleMode::from(m.clone()).to_dict(py).map(|b| b.unbind())
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
///
/// The conversion goes through `serde_json::Value` so the rules live in
/// exactly one place: `CycleBankConfig`'s `#[serde(deny_unknown_fields)]`
/// derive. Adding a new field to the struct now updates the wire shape for
/// every binding layer automatically — no per-key maintenance here.
fn cycle_bank_config_from_dict(
    overrides: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<crate::cycle_bank::CycleBankConfig> {
    let cfg = crate::cycle_bank::CycleBankConfig::default();
    let Some(dict) = overrides else {
        return Ok(cfg);
    };
    let value: serde_json::Value = python_dict_to_json_value(dict)?;
    // Start from the serde default (matches `Default::default()` because
    // `Default` is implemented and `serde_json::from_value` accepts a
    // partial by default). Apply the user overrides on top.
    let mut merged = serde_json::to_value(&cfg).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "could not serialize CycleBankConfig defaults: {e}"
        ))
    })?;
    if let serde_json::Value::Object(ref mut map) = merged {
        if let serde_json::Value::Object(overrides_map) = value {
            for (k, v) in overrides_map {
                map.insert(k, v);
            }
        }
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "CycleBankConfig did not serialize to a JSON object",
        ));
    }
    serde_json::from_value(merged).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "invalid CycleBankConfig: {e}"
        ))
    })
}

/// Convert a flat Python dict whose values are JSON-compatible scalars
/// (`int` / `float` / `str` / `bool` / nested `dict` / nested `list`) into a
/// `serde_json::Value`. This is a deliberately small helper: the cycle-bank
/// config only contains primitives.
fn python_dict_to_json_value(
    dict: &pyo3::Bound<'_, pyo3::types::PyDict>,
) -> PyResult<serde_json::Value> {
    use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
    let mut map = serde_json::Map::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let v = if value.is_instance_of::<PyBool>() {
            serde_json::Value::Bool(value.extract::<bool>()?)
        } else if value.is_instance_of::<PyInt>() {
            // PyO3 extracts the int directly; serde_json will accept it as u64/i64.
            let n: i64 = value.extract()?;
            serde_json::Value::Number(serde_json::Number::from(n))
        } else if value.is_instance_of::<PyFloat>() {
            let n: f64 = value.extract()?;
            serde_json::Value::Number(
                serde_json::Number::from_f64(n).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "CycleBankConfig key {key_str}: non-finite float"
                    ))
                })?,
            )
        } else if value.is_instance_of::<PyString>() {
            serde_json::Value::String(value.extract()?)
        } else if value.is_instance_of::<PyDict>() {
            let bound = value.downcast::<PyDict>()?;
            python_dict_to_json_value(bound)?
        } else if value.is_instance_of::<PyList>() {
            let list = value.downcast::<PyList>()?;
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                if item.is_instance_of::<PyDict>() {
                    let bound = item.downcast::<PyDict>()?;
                    out.push(python_dict_to_json_value(bound)?);
                } else if item.is_instance_of::<PyInt>() {
                    let n: i64 = item.extract()?;
                    out.push(serde_json::Value::Number(serde_json::Number::from(n)));
                } else if item.is_instance_of::<PyFloat>() {
                    let n: f64 = item.extract()?;
                    out.push(serde_json::Value::Number(
                        serde_json::Number::from_f64(n).ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(
                                "non-finite float in list",
                            )
                        })?,
                    ));
                } else if item.is_instance_of::<PyBool>() {
                    out.push(serde_json::Value::Bool(item.extract::<bool>()?));
                } else if item.is_instance_of::<PyString>() {
                    out.push(serde_json::Value::String(item.extract()?));
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "unsupported CycleBankConfig value type for key {key_str}"
                    )));
                }
            }
            serde_json::Value::Array(out)
        } else if value.is_none() {
            serde_json::Value::Null
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "unsupported CycleBankConfig value type for key {key_str}"
            )));
        };
        map.insert(key_str, v);
    }
    Ok(serde_json::Value::Object(map))
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
    /// through unchanged. Returns the current observed `CycleMode` objects.
    fn observe_tick(
        &mut self,
        tick: &pyo3::Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<Vec<CycleMode>> {
        let tick = tick_from_pydict(tick)?;
        let obs = crate::timebase::cycle_observation_from_tick(&tick).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "tick feature window is not the expected frame-major shape",
            )
        })?;
        self.inner
            .observe(&obs)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(self.modes())
    }

    /// Feed one explicit observation of named scalar evidence channels.
    ///
    /// `sample_index` / `dt_seconds` / `stream_epoch` carry the #91 sample
    /// clock; `channels` is a sequence of `(name, value)` pairs. This entry
    /// point exists for synthetic diagnostics; the production path is
    /// `observe_tick`. Returns the current observed `CycleMode` objects.
    fn observe(
        &mut self,
        sample_index: u64,
        dt_seconds: f64,
        stream_epoch: u64,
        channels: Vec<(String, f64)>,
    ) -> PyResult<Vec<CycleMode>> {
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
        Ok(self.modes())
    }

    /// Current confirmed observed modes as `CycleMode` objects.
    fn modes(&self) -> Vec<CycleMode> {
        self.inner
            .modes()
            .into_iter()
            .map(CycleMode::from)
            .collect()
    }

    /// Current observed modes as camelCase dicts (the same wire shape the
    /// browser's wasm `CycleMode` interface uses). Convenience for code that
    /// wants plain records instead of objects.
    fn modes_as_dicts<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
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
}

/// Read the per-channel scalar evidence the canonical Rust seam extracts from
/// one analysis tick, WITHOUT advancing any bank state.
///
/// This is the ONLY supported way for Python to see the newest-frame evidence
/// values (e.g. to measure onset events for diagnostics): the frame-major
/// offset arithmetic lives in Rust (`cycle_observation_from_tick`), never in
/// Python. Returns a list of `(name, value)` pairs in the canonical channel
/// schema order.
#[pyfunction]
#[pyo3(signature = (tick))]
fn cycle_observation_channels_from_tick(
    tick: &pyo3::Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Vec<(String, f64)>> {
    let tick = tick_from_pydict(tick)?;
    let obs = crate::timebase::cycle_observation_from_tick(&tick).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "tick feature window is not the expected frame-major shape",
        )
    })?;
    Ok(obs
        .channels
        .into_iter()
        .map(|c| (c.name, c.value))
        .collect())
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
    // Controls v2 contract (issue #107)
    m.add("CONTROLS_VERSION", crate::controls::CONTROLS_VERSION)?;
    m.add("CONTROLS_MAX_DRIVE_FORCE", crate::controls::MAX_DRIVE_FORCE)?;
    m.add("CONTROLS_MAX_IMPULSE", crate::controls::MAX_IMPULSE)?;
    m.add("CONTROLS_BRAKE_COEFF", crate::controls::BRAKE_COEFF)?;

    m.add_class::<ResidualParams>()?;
    m.add_class::<ManifoldConfig>()?;
    m.add_class::<EnergyInfo>()?;
    m.add_class::<OrbitState>()?;
    m.add_class::<PlayerState>()?;
    m.add_class::<OrbitController>()?;

    m.add_class::<FeatureExtractor>()?;

    m.add_class::<AnalysisTimebase>()?;

    m.add_class::<CycleMode>()?;
    m.add_class::<CycleBank>()?;
    m.add_class::<MotionControls>()?;
    m.add_class::<JuliaViewControls>()?;
    m.add_class::<ControlsV2>()?;
    m.add_class::<ColorIntent>()?;
    m.add_class::<JuliaViewState>()?;

    m.add_class::<RuntimeVisualMetrics>()?;

    m.add_function(wrap_pyfunction!(lobe_point_at_angle, m)?)?;
    m.add_function(wrap_pyfunction!(compute_runtime_visual_metrics, m)?)?;
    // CycleBank tick-seam evidence accessor (issue #92): newest-frame channel
    // values from a canonical tick, computed in Rust.
    m.add_function(wrap_pyfunction!(cycle_observation_channels_from_tick, m)?)?;
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
    // Manifold physics (issue #106) — training surface
    m.add_function(wrap_pyfunction!(manifold_signed_distance, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_regularized_distance, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_mandelbrot_scale, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_scale_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_scale_hessian, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_induced_metric, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_kinetic_energy, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_potential_energy, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_total_energy, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_christoffel_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_geodesic_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_potential_force, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_apply_generalized_force, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_drag_force, m)?)?;
    m.add_function(wrap_pyfunction!(manifold_integrate_step, m)?)?;
    m.add_function(wrap_pyfunction!(controls_integrate_step, m)?)?;
    m.add_function(wrap_pyfunction!(motion_drive_covector, m)?)?;
    m.add_function(wrap_pyfunction!(__getattr__, m)?)?;
    m.add_function(wrap_pyfunction!(__getattr__, m)?)?;
    Ok(())
}
