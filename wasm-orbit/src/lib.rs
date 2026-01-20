//! Orbit synthesizer for Julia set parameter generation
//! 
//! Single source of truth for orbit synthesis logic.
//! Compiled to WebAssembly for use in browser.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct OrbitState {
    pub lobe: u32,
    pub sub_lobe: u32,
    pub theta: f64,
    pub omega: f64,
    pub s: f64,
    pub alpha: f64,
    residual_phases: Vec<f64>,
    residual_omegas: Vec<f64>,
}

#[wasm_bindgen]
impl OrbitState {
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
        let residual_phases: Vec<f64> = (0..k_residuals)
            .map(|_| js_sys::Math::random() * 2.0 * std::f64::consts::PI)
            .collect();
        
        let residual_omegas: Vec<f64> = (0..k_residuals)
            .map(|k| residual_omega_scale * omega * (k as f64 + 1.0))
            .collect();

        OrbitState {
            lobe,
            sub_lobe,
            theta,
            omega,
            s,
            alpha,
            residual_phases,
            residual_omegas,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn residual_phases(&self) -> Vec<f64> {
        self.residual_phases.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn residual_omegas(&self) -> Vec<f64> {
        self.residual_omegas.clone()
    }
}

#[wasm_bindgen]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

#[wasm_bindgen]
pub struct OrbitSynthesizer {
    k_residuals: usize,
    residual_cap: f64,
}

#[wasm_bindgen]
impl OrbitSynthesizer {
    #[wasm_bindgen(constructor)]
    pub fn new(k_residuals: usize, residual_cap: f64) -> OrbitSynthesizer {
        OrbitSynthesizer {
            k_residuals,
            residual_cap,
        }
    }

    /// Synthesize Julia parameter c from orbit state
    /// 
    /// FORMULA: amplitude_k = α × (s × radius) / (k + 1)²
    pub fn synthesize(&self, state: &OrbitState, band_gates: Option<Vec<f64>>) -> Complex {
        // Carrier: deterministic orbit point
        let carrier = lobe_point_at_angle(state.lobe, state.theta, state.s, state.sub_lobe);

        if state.alpha == 0.0 {
            return carrier;
        }

        let radius = get_lobe_radius(state.lobe, state.sub_lobe);

        // Sum residual circles with 1/k² amplitude decay
        let mut residual_real = 0.0;
        let mut residual_imag = 0.0;

        for k in 0..self.k_residuals {
            // CORE FORMULA: amplitude decreases as 1/k²
            let amplitude = (state.alpha * (state.s * radius)) / ((k as f64 + 1.0).powi(2));
            let g_k = band_gates.as_ref().map(|g| g[k]).unwrap_or(1.0);

            let phase = state.residual_phases[k];
            residual_real += amplitude * g_k * phase.cos();
            residual_imag += amplitude * g_k * phase.sin();
        }

        // Cap residual magnitude
        let mag = (residual_real * residual_real + residual_imag * residual_imag).sqrt();
        let cap = self.residual_cap * radius;
        if mag > cap {
            let scale = cap / mag;
            residual_real *= scale;
            residual_imag *= scale;
        }

        Complex {
            real: carrier.real + residual_real,
            imag: carrier.imag + residual_imag,
        }
    }

    /// Step forward in time and synthesize c(t)
    pub fn step(&self, state: &OrbitState, dt: f64, band_gates: Option<Vec<f64>>) -> JsValue {
        let c = self.synthesize(state, band_gates);

        let new_theta = (state.theta + state.omega * dt) % (2.0 * std::f64::consts::PI);
        let new_residual_phases: Vec<f64> = state.residual_phases
            .iter()
            .zip(&state.residual_omegas)
            .map(|(phase, omega)| (phase + omega * dt) % (2.0 * std::f64::consts::PI))
            .collect();

        let new_state = OrbitState {
            lobe: state.lobe,
            sub_lobe: state.sub_lobe,
            theta: new_theta,
            omega: state.omega,
            s: state.s,
            alpha: state.alpha,
            residual_phases: new_residual_phases,
            residual_omegas: state.residual_omegas.clone(),
        };

        // Return as JavaScript object
        serde_wasm_bindgen::to_value(&serde_json::json!({
            "c": { "real": c.real, "imag": c.imag },
            "newState": new_state
        })).unwrap()
    }
}

/// Compute position on lobe boundary at given angle
fn lobe_point_at_angle(lobe: u32, theta: f64, s: f64, sub_lobe: u32) -> Complex {
    if lobe == 1 {
        // Cardioid parametrization
        let r = 0.25 * (1.0 - theta.cos());
        let phi = theta;
        Complex {
            real: s * r * phi.cos(),
            imag: s * r * phi.sin(),
        }
    } else {
        // Period-n bulb
        let center = period_n_bulb_center(lobe, sub_lobe);
        let radius = period_n_bulb_radius(lobe, sub_lobe);
        Complex {
            real: center.real + s * radius * theta.cos(),
            imag: center.imag + s * radius * theta.sin(),
        }
    }
}

/// Get center of period-n bulb
fn period_n_bulb_center(n: u32, k: u32) -> Complex {
    if n == 1 {
        return Complex { real: 0.0, imag: 0.0 };
    }

    // Hardcoded centers for common periods
    match (n, k) {
        (2, 0) => Complex { real: -1.0, imag: 0.0 },
        (3, 0) => Complex { real: -0.125, imag: 0.649519 },
        (3, 1) => Complex { real: -0.125, imag: -0.649519 },
        _ => {
            // Approximate formula
            let angle = (2.0 * std::f64::consts::PI * k as f64) / n as f64;
            let r = 0.25 * (1.0 - angle.cos());
            Complex {
                real: r * angle.cos(),
                imag: r * angle.sin(),
            }
        }
    }
}

/// Get radius of period-n bulb
fn period_n_bulb_radius(n: u32, _k: u32) -> f64 {
    match n {
        1 => 0.25,
        2 => 0.25,
        3 => 0.0943,
        4 => 0.04,
        _ => 0.25 / (n as f64 * n as f64),
    }
}

/// Get lobe radius for residual scaling
fn get_lobe_radius(lobe: u32, sub_lobe: u32) -> f64 {
    if lobe == 1 {
        0.25
    } else {
        period_n_bulb_radius(lobe, sub_lobe)
    }
}
