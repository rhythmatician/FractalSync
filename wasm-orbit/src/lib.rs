//! Orbit synthesizer for Julia set parameter generation
//! 
//! WebAssembly bindings to runtime_core for browser use.

use wasm_bindgen::prelude::*;
use serde::Serialize;
use num_complex::Complex64 as RustComplex;
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
            real: c.re,
            imag: c.im,
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

    /// Set s (radius scaling)
    #[wasm_bindgen(setter)]
    pub fn set_s(&mut self, s: f64) {
        self.inner.s = s;
    }

    /// Set alpha (residual amplitude)
    #[wasm_bindgen(setter)]
    pub fn set_alpha(&mut self, alpha: f64) {
        self.inner.alpha = alpha;
    }

    /// Set omega (base angular velocity)
    #[wasm_bindgen(setter)]
    pub fn set_omega(&mut self, omega: f64) {
        self.inner.omega = omega;
    }

    /// Get sub_lobe
    #[wasm_bindgen(getter)]
    pub fn sub_lobe(&self) -> u32 {
        self.inner.sub_lobe
    }

    /// Set sub_lobe
    #[wasm_bindgen(setter)]
    pub fn set_sub_lobe(&mut self, sub_lobe: u32) {
        self.inner.sub_lobe = sub_lobe;
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


/// --- Minimap / mip pyramid bindings (issue #88) ---

use num_complex::Complex64 as Rc64;

/// Set the mip pyramid from host-provided flat planes (row-major, per level).
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
        return Err(JsValue::from_str(
            "mip pyramid plane sizes do not match data length",
        ));
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
    let mut pyr = runtime_core::minimap::MipPyramid::from_levels(
        split(&s_flat),
        widths.iter().map(|&w| w as usize).collect(),
        heights.iter().map(|&h| h as usize).collect(),
        re_min,
        re_max,
        im_min,
        im_max,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    pyr.set_escape_field(split(&f_flat));
    runtime_core::minimap::set_pyramid(pyr).map_err(|e| JsValue::from_str(&e))
}

/// The Player's full observation at c: 4x81 greys + 8 slope values = 332.
#[wasm_bindgen]
pub fn player_observation(real: f64, imag: f64) -> Result<Vec<f32>, JsValue> {
    runtime_core::minimap::with_pyramid(|pyr| {
        let pyr = pyr.ok_or_else(|| JsValue::from_str("mip pyramid not loaded"))?;
        pyr.player_observation(Rc64::new(real, imag))
            .ok_or_else(|| JsValue::from_str("c outside map extent"))
    })
}

/// Slope of the shore-proximity field at c on a mip level. Returns [gx, gy].
#[wasm_bindgen]
pub fn minimap_slope(real: f64, imag: f64, level: usize) -> Result<Vec<f64>, JsValue> {
    runtime_core::minimap::with_pyramid(|pyr| {
        let pyr = pyr.ok_or_else(|| JsValue::from_str("mip pyramid not loaded"))?;
        let (gx, gy) = pyr
            .slope(Rc64::new(real, imag), level)
            .ok_or_else(|| JsValue::from_str("c outside map extent"))?;
        Ok(vec![gx, gy])
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
) -> Result<Vec<f64>, JsValue> {
    let (nr, ni) = runtime_core::minimap::contour_biased_step(
        real, imag, u_real, u_imag, h, d_star, max_step, level,
    )?;
    Ok(vec![nr, ni])
}
