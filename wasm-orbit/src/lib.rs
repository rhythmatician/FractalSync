//! Orbit synthesizer for Julia set parameter generation
//! 
//! WebAssembly bindings to runtime_core for browser use.

use wasm_bindgen::prelude::*;
use serde::Serialize;
use runtime_core::controller::{
    OrbitState as RustOrbitState,
    ResidualParams as RustResidualParams,
    synthesize as rust_synthesize,
    DEFAULT_K_RESIDUALS,
    DEFAULT_RESIDUAL_CAP,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    DEFAULT_BASE_OMEGA,
    DEFAULT_ORBIT_SEED,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    WINDOW_FRAMES,
};
use runtime_core::geometry::Complex as RustComplex;

/// Shared constants exposed to JavaScript
#[wasm_bindgen]
pub fn constants() -> JsValue {
    #[derive(Serialize)]
    struct Constants {
        sample_rate: usize,
        hop_length: usize,
        n_fft: usize,
        window_frames: usize,
        default_k_residuals: usize,
        default_residual_cap: f64,
        default_residual_omega_scale: f64,
        default_base_omega: f64,
        default_orbit_seed: u64,
    }

    let c = Constants {
        sample_rate: SAMPLE_RATE,
        hop_length: HOP_LENGTH,
        n_fft: N_FFT,
        window_frames: WINDOW_FRAMES,
        default_k_residuals: DEFAULT_K_RESIDUALS,
        default_residual_cap: DEFAULT_RESIDUAL_CAP,
        default_residual_omega_scale: DEFAULT_RESIDUAL_OMEGA_SCALE,
        default_base_omega: DEFAULT_BASE_OMEGA,
        default_orbit_seed: DEFAULT_ORBIT_SEED,
    };

    serde_wasm_bindgen::to_value(&c).unwrap_or_else(|_| JsValue::NULL)
}

/// Wrapper for complex number to/from JavaScript
#[wasm_bindgen]
#[derive(Clone)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl From<RustComplex> for Complex {
    fn from(c: RustComplex) -> Self {
        Self {
            real: c.real,
            imag: c.imag,
        }
    }
}

/// Orbit state wrapper for WASM
#[wasm_bindgen]
#[derive(Clone)]
pub struct OrbitState {
    inner: RustOrbitState,
}

#[wasm_bindgen]
impl OrbitState {
    /// Create new orbit state with optional seed
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
        seed: Option<u64>,
    ) -> OrbitState {
        let inner = match seed {
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
        };

        OrbitState { inner }
    }

    /// Create deterministic orbit with default parameters and seed
    #[wasm_bindgen(js_name = "newDefault")]
    pub fn new_default(seed: u64) -> OrbitState {
        let inner = RustOrbitState::new_with_seed(
            1,
            0,
            0.0,
            DEFAULT_BASE_OMEGA,
            1.02,
            0.3,
            DEFAULT_K_RESIDUALS,
            DEFAULT_RESIDUAL_OMEGA_SCALE,
            seed,
        );
        OrbitState { inner }
    }

    /// Get lobe
    #[wasm_bindgen(getter)]
    pub fn lobe(&self) -> u32 {
        self.inner.lobe
    }

    /// Set lobe
    #[wasm_bindgen(setter)]
    pub fn set_lobe(&mut self, lobe: u32) {
        self.inner.lobe = lobe;
    }

    /// Get theta
    #[wasm_bindgen(getter)]
    pub fn theta(&self) -> f64 {
        self.inner.theta
    }

    /// Get s (radius scaling)
    #[wasm_bindgen(getter)]
    pub fn s(&self) -> f64 {
        self.inner.s
    }

    /// Get alpha (residual amplitude)
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// Advance state by dt seconds
    pub fn advance(&mut self, dt: f64) {
        self.inner.advance(dt);
    }

    /// Compute c(t) without advancing time.
    pub fn synthesize(&self, residual_params: &ResidualParams, band_gates: Option<Vec<f64>>) -> Complex {
        rust_synthesize(
            &self.inner,
            residual_params.inner,
            band_gates.as_deref(),
        )
        .into()
    }

    /// Advance by dt and return c(t). Mutates this OrbitState.
    pub fn step(&mut self, dt: f64, residual_params: &ResidualParams, band_gates: Option<Vec<f64>>) -> Complex {
        rust_step(
            &mut self.inner,
            dt,
            residual_params.inner,
            band_gates.as_deref(),
            None,
            0.0,
            None,
            None,
        )
        .into()
    }

    /// Advance with transient and integrator options: h in [0,1], optional d_star, max_step.
    pub fn step_advanced(
        &mut self,
        dt: f64,
        residual_params: &ResidualParams,
        band_gates: Option<Vec<f64>>,
        h: f64,
        d_star: Option<f64>,
        max_step: Option<f64>,
        distance_field: Option<&DistanceField>,
    ) -> Complex {
        let rust_field = distance_field.map(|df| &df.inner);
        rust_step(
            &mut self.inner,
            dt,
            residual_params.inner,
            band_gates.as_deref(),
            rust_field,
            h,
            d_star,
            max_step,
        )
        .into()
    }
}

/// Residual parameters
#[wasm_bindgen]
#[derive(Clone)]
pub struct ResidualParams {
    inner: RustResidualParams,
}

#[wasm_bindgen]
impl ResidualParams {
    /// Create default residual parameters
    #[wasm_bindgen(constructor)]
    pub fn new(
        k_residuals: usize,
        residual_cap: f64,
        radius_scale: f64,
    ) -> ResidualParams {
        ResidualParams {
            inner: RustResidualParams {
                k_residuals,
                residual_cap,
                radius_scale,
            },
        }
    }
}

/// Synthesize Julia parameter from orbit state
#[wasm_bindgen(js_name = "synthesize")]
pub fn synthesize(
    state: &OrbitState,
    residual_params: &ResidualParams,
    band_gates: Option<Vec<f64>>,
) -> Complex {
    let c = rust_synthesize(
        &state.inner,
        residual_params.inner,
        band_gates.as_deref(),
    );
    c.into()
}

/// Step the orbit forward and synthesize
#[wasm_bindgen]
pub fn step(
    state: &mut OrbitState,
    dt: f64,
    residual_params: &ResidualParams,
    band_gates: Option<Vec<f64>>,
) -> Complex {
    state.inner.advance(dt);
    synthesize(state, residual_params, band_gates)
}

/// Distance field wrapper for WASM
#[wasm_bindgen]
pub struct DistanceField {
    inner: RustDistanceField,
}

#[wasm_bindgen]
impl DistanceField {
    #[wasm_bindgen(constructor)]
    pub fn new(
        field: Vec<f32>,
        resolution: usize,
        real_range: (f64, f64),
        imag_range: (f64, f64),
        max_distance: f64,
        slowdown_threshold: f64,
    ) -> DistanceField {
        DistanceField {
            inner: RustDistanceField::new(
                field,
                resolution,
                real_range,
                imag_range,
                max_distance,
                slowdown_threshold,
            ),
        }
    }

    pub fn lookup(&self, real: f64, imag: f64) -> f32 {
        self.inner.lookup(RustComplex::new(real, imag))
    }

    pub fn sample_bilinear(&self, real: f64, imag: f64) -> f32 {
        self.inner.sample_bilinear(RustComplex::new(real, imag))
    }

    pub fn gradient(&self, real: f64, imag: f64) -> js_sys::Array {
        let (gx, gy) = self.inner.gradient(RustComplex::new(real, imag));
        let arr = js_sys::Array::new();
        arr.push(&JsValue::from_f64(gx));
        arr.push(&JsValue::from_f64(gy));
        arr
    }

    pub fn get_velocity_scale(&self, real: f64, imag: f64) -> f32 {
        self.inner.get_velocity_scale(RustComplex::new(real, imag))
    }
}
