//! WebAssembly bindings for runtime-core
//!
//! Exposes the height-field controller and audio feature extractor.

use js_sys::Array;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::controller::{
    evaluate_height_field,
    step_height_controller,
    HeightControllerParams as RustHeightControllerParams,
    HeightControllerStep as RustHeightControllerStep,
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

/// A complex number (c = a + bi).
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Complex {
    real: f64,
    imag: f64,
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

#[wasm_bindgen]
impl Complex {
    #[wasm_bindgen(constructor)]
    pub fn new(real: f64, imag: f64) -> Complex {
        Complex { real, imag }
    }

    #[wasm_bindgen(getter)]
    pub fn real(&self) -> f64 {
        self.real
    }

    #[wasm_bindgen(getter)]
    pub fn imag(&self) -> f64 {
        self.imag
    }
}

/// Height-field sample returned by evaluating f(c).
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeightFieldSample {
    height: f64,
    gradient: Complex,
}

#[wasm_bindgen]
impl HeightFieldSample {
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f64 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn gradient(&self) -> Complex {
        self.gradient.clone()
    }
}

/// Controller step result.
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeightControllerStep {
    new_c: Complex,
    delta: Complex,
    height: f64,
    gradient: Complex,
}

impl From<RustHeightControllerStep> for HeightControllerStep {
    fn from(step: RustHeightControllerStep) -> Self {
        Self {
            new_c: step.new_c.into(),
            delta: step.delta.into(),
            height: step.height,
            gradient: step.gradient.into(),
        }
    }
}

#[wasm_bindgen]
impl HeightControllerStep {
    #[wasm_bindgen(getter)]
    pub fn new_c(&self) -> Complex {
        self.new_c.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn delta(&self) -> Complex {
        self.delta.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f64 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn gradient(&self) -> Complex {
        self.gradient.clone()
    }
}

/// Evaluate the height field at c.
#[wasm_bindgen]
pub fn height_field(c: &Complex, iterations: Option<usize>, epsilon: Option<f64>) -> HeightFieldSample {
    let sample = evaluate_height_field(
        RustComplex::from(c),
        iterations.unwrap_or(DEFAULT_HEIGHT_ITERATIONS),
        epsilon.unwrap_or(DEFAULT_HEIGHT_EPSILON),
    );
    HeightFieldSample {
        height: sample.height,
        gradient: sample.gradient.into(),
    }
}

/// Project a model step onto the height contour and apply correction.
#[wasm_bindgen]
pub fn height_controller_step(
    c: &Complex,
    delta_model: &Complex,
    target_height: f64,
    normal_risk: f64,
    height_gain: Option<f64>,
    iterations: Option<usize>,
    epsilon: Option<f64>,
) -> HeightControllerStep {
    let params = RustHeightControllerParams {
        target_height,
        normal_risk,
        height_gain: height_gain.unwrap_or(DEFAULT_HEIGHT_GAIN),
    };
    step_height_controller(
        RustComplex::from(c),
        RustComplex::from(delta_model),
        params,
        iterations.unwrap_or(DEFAULT_HEIGHT_ITERATIONS),
        epsilon.unwrap_or(DEFAULT_HEIGHT_EPSILON),
    )
    .into()
}

/// Audio feature extractor.
#[wasm_bindgen]
pub struct FeatureExtractor {
    inner: RustFeatureExtractor,
}

#[wasm_bindgen]
impl FeatureExtractor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        sr: usize,
        hop_length: usize,
        n_fft: usize,
        include_delta: bool,
        include_delta_delta: bool,
    ) -> FeatureExtractor {
        FeatureExtractor {
            inner: RustFeatureExtractor::new(sr, hop_length, n_fft, include_delta, include_delta_delta),
        }
    }

    #[wasm_bindgen]
    pub fn num_features_per_frame(&self) -> usize {
        self.inner.num_features_per_frame()
    }

    /// Extract windowed feature vectors.
    ///
    /// Returns a nested JS array: Vec<Vec<f64>>.
    #[wasm_bindgen]
    pub fn extract_windowed_features(&self, audio: Vec<f32>, window_frames: usize) -> Array {
        let windows = self.inner.extract_windowed_features(&audio[..], window_frames);
        let outer = Array::new();

        for w in windows {
            let inner = Array::new();
            for v in w {
                inner.push(&JsValue::from_f64(v));
            }
            outer.push(&inner);
        }

        outer
    }
}

/// Shared runtime constants for parity checks between backend and frontend.
#[wasm_bindgen]
pub fn sample_rate() -> usize {
    SAMPLE_RATE
}

#[wasm_bindgen]
pub fn hop_length() -> usize {
    HOP_LENGTH
}

#[wasm_bindgen]
pub fn n_fft() -> usize {
    N_FFT
}

#[wasm_bindgen]
pub fn window_frames() -> usize {
    WINDOW_FRAMES
}

#[wasm_bindgen]
pub fn default_height_iterations() -> usize {
    DEFAULT_HEIGHT_ITERATIONS
}

#[wasm_bindgen]
pub fn default_height_epsilon() -> f64 {
    DEFAULT_HEIGHT_EPSILON
}

#[wasm_bindgen]
pub fn default_height_gain() -> f64 {
    DEFAULT_HEIGHT_GAIN
}
