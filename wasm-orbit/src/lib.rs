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
use runtime_core::step_controller::{
    StepController as RustStepController,
    StepState as RustStepState,
};

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

#[derive(Serialize)]
struct StepContextJs {
    c_real: f64,
    c_imag: f64,
    prev_delta_real: f64,
    prev_delta_imag: f64,
    nu_norm: f32,
    membership: bool,
    grad_re: f32,
    grad_im: f32,
    sensitivity: f32,
    patch: Vec<f32>,
    mip_level: usize,
    feature_vec: Vec<f32>,
}

#[derive(Serialize)]
struct StepDebugJs {
    mip_level: usize,
    scale_g: f64,
    scale_df: f64,
    scale: f64,
    delta_f_pred: f64,
    wall_applied: bool,
}

#[derive(Serialize)]
struct StepResultJs {
    c_real: f64,
    c_imag: f64,
    delta_real: f64,
    delta_imag: f64,
    debug: StepDebugJs,
    context: StepContextJs,
}

fn context_to_js(context: runtime_core::step_controller::StepContext) -> StepContextJs {
    StepContextJs {
        c_real: context.c.real,
        c_imag: context.c.imag,
        prev_delta_real: context.prev_delta.real,
        prev_delta_imag: context.prev_delta.imag,
        nu_norm: context.nu_norm,
        membership: context.membership,
        grad_re: context.grad_re,
        grad_im: context.grad_im,
        sensitivity: context.sensitivity,
        patch: context.patch.clone(),
        mip_level: context.mip_level,
        feature_vec: context.as_feature_vec(),
    }
}

/// Step controller state for JS.
#[wasm_bindgen]
#[derive(Clone)]
pub struct StepState {
    inner: RustStepState,
}

#[wasm_bindgen]
impl StepState {
    #[wasm_bindgen(constructor)]
    pub fn new(
        c_real: f64,
        c_imag: f64,
        prev_delta_real: f64,
        prev_delta_imag: f64,
    ) -> StepState {
        StepState {
            inner: RustStepState {
                c: RustComplex::new(c_real, c_imag),
                prev_delta: RustComplex::new(prev_delta_real, prev_delta_imag),
            },
        }
    }

    #[wasm_bindgen(getter)]
    pub fn c_real(&self) -> f64 {
        self.inner.c.real
    }

    #[wasm_bindgen(setter)]
    pub fn set_c_real(&mut self, value: f64) {
        self.inner.c.real = value;
    }

    #[wasm_bindgen(getter)]
    pub fn c_imag(&self) -> f64 {
        self.inner.c.imag
    }

    #[wasm_bindgen(setter)]
    pub fn set_c_imag(&mut self, value: f64) {
        self.inner.c.imag = value;
    }

    #[wasm_bindgen(getter)]
    pub fn prev_delta_real(&self) -> f64 {
        self.inner.prev_delta.real
    }

    #[wasm_bindgen(setter)]
    pub fn set_prev_delta_real(&mut self, value: f64) {
        self.inner.prev_delta.real = value;
    }

    #[wasm_bindgen(getter)]
    pub fn prev_delta_imag(&self) -> f64 {
        self.inner.prev_delta.imag
    }

    #[wasm_bindgen(setter)]
    pub fn set_prev_delta_imag(&mut self, value: f64) {
        self.inner.prev_delta.imag = value;
    }
}

/// Step-based controller for JS.
#[wasm_bindgen]
pub struct StepController {
    inner: RustStepController,
}

#[wasm_bindgen]
impl StepController {
    #[wasm_bindgen(constructor)]
    pub fn new() -> StepController {
        StepController {
            inner: RustStepController::new(),
        }
    }

    pub fn context(&self, state: &StepState) -> JsValue {
        let ctx = self.inner.context_for_state(&state.inner);
        let js = context_to_js(ctx);
        serde_wasm_bindgen::to_value(&js).unwrap_or_else(|_| JsValue::NULL)
    }

    pub fn step(&self, state: &mut StepState, delta_real: f64, delta_imag: f64) -> JsValue {
        let result = self
            .inner
            .step(&mut state.inner, RustComplex::new(delta_real, delta_imag));
        let debug = StepDebugJs {
            mip_level: result.debug.mip_level,
            scale_g: result.debug.scale_g,
            scale_df: result.debug.scale_df,
            scale: result.debug.scale,
            delta_f_pred: result.debug.delta_f_pred,
            wall_applied: result.debug.wall_applied,
        };
        let ctx = context_to_js(result.context);
        let js = StepResultJs {
            c_real: result.c_next.real,
            c_imag: result.c_next.imag,
            delta_real: result.delta_applied.real,
            delta_imag: result.delta_applied.imag,
            debug,
            context: ctx,
        };
        serde_wasm_bindgen::to_value(&js).unwrap_or_else(|_| JsValue::NULL)
    }
}
