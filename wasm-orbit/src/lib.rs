//! WebAssembly wrapper that re-exports the runtime-core wasm bindings to avoid duplicate wasm exports.

use wasm_bindgen::prelude::*;
use serde::Serialize;

// Re-export items from runtime-core's wasm bindings so the frontend imports a single module.
pub use runtime_core::wasm_bindings::{
    Complex,
    OrbitState,
    ResidualParams,
    DistanceField,
    FeatureExtractor,
    lobe_point_at_angle,
    sample_rate,
    hop_length,
    n_fft,
    window_frames,
    default_k_residuals,
    default_residual_cap,
    default_residual_omega_scale,
    default_base_omega,
    default_orbit_seed,
    contour_biased_step,
};

/// Aggregate constants as a convenience for JS (keeps parity checks single place).
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
        sample_rate: sample_rate(),
        hop_length: hop_length(),
        n_fft: n_fft(),
        window_frames: window_frames(),
        default_k_residuals: default_k_residuals(),
        default_residual_cap: default_residual_cap(),
        default_residual_omega_scale: default_residual_omega_scale(),
        default_base_omega: default_base_omega(),
        default_orbit_seed: default_orbit_seed(),
    };

    serde_wasm_bindgen::to_value(&c).unwrap_or_else(|_| JsValue::NULL)
}

