//! Height-field sampling and contour-following controller.
//!
//! This module provides a lightweight "height map" over the
//! Mandelbrot c-plane using a fixed iteration depth. It also
//! implements a controller that projects a model-proposed step onto
//! the tangent of a level set while applying a corrective term to
//! stay near a target height.

use crate::geometry::Complex;

/// Default iteration count for the height field sampler.
pub const DEFAULT_HEIGHT_ITERATIONS: usize = 64;
/// Minimum magnitude used to avoid log(0) and divide-by-zero.
pub const DEFAULT_HEIGHT_MIN_MAGNITUDE: f64 = 1.0e-6;
/// Default corrective gain for contour control.
pub const DEFAULT_CONTOUR_CORRECTION_GAIN: f64 = 0.8;
/// Default epsilon for projection stability.
pub const DEFAULT_CONTOUR_PROJECTION_EPSILON: f64 = 1.0e-6;

/// Parameters for sampling the Mandelbrot height field at fixed depth.
#[derive(Clone, Copy, Debug)]
pub struct HeightFieldParams {
    pub iterations: usize,
    pub min_magnitude: f64,
}

impl Default for HeightFieldParams {
    fn default() -> Self {
        Self {
            iterations: DEFAULT_HEIGHT_ITERATIONS,
            min_magnitude: DEFAULT_HEIGHT_MIN_MAGNITUDE,
        }
    }
}

/// Result of sampling the height field at a point.
#[derive(Clone, Copy, Debug)]
pub struct HeightFieldSample {
    pub height: f64,
    pub gradient: Complex,
    pub z: Complex,
    pub w: Complex,
    pub magnitude: f64,
}

/// Parameters controlling the contour-following projection.
#[derive(Clone, Copy, Debug)]
pub struct ContourControllerParams {
    pub correction_gain: f64,
    pub projection_epsilon: f64,
}

impl Default for ContourControllerParams {
    fn default() -> Self {
        Self {
            correction_gain: DEFAULT_CONTOUR_CORRECTION_GAIN,
            projection_epsilon: DEFAULT_CONTOUR_PROJECTION_EPSILON,
        }
    }
}

/// State for contour-following motion in the c-plane.
#[derive(Clone, Debug)]
pub struct ContourState {
    pub c: Complex,
    pub target_height: f64,
    pub last_delta: Complex,
}

/// Output for a single contour controller step.
#[derive(Clone, Copy, Debug)]
pub struct ContourStep {
    pub c: Complex,
    pub height: f64,
    pub height_error: f64,
    pub gradient: Complex,
    pub corrected_delta: Complex,
}

#[inline]
fn dot(a: Complex, b: Complex) -> f64 {
    a.real * b.real + a.imag * b.imag
}

#[inline]
fn sub(a: Complex, b: Complex) -> Complex {
    Complex::new(a.real - b.real, a.imag - b.imag)
}

/// Sample the fixed-depth Mandelbrot height field at `c`.
///
/// The height is defined as f(c) = log(|z_N|) with a floor on |z_N|
/// for numerical stability.
pub fn sample_height_field(c: Complex, params: HeightFieldParams) -> HeightFieldSample {
    let mut z = Complex::new(0.0, 0.0);
    let mut w = Complex::new(0.0, 0.0);

    for _ in 0..params.iterations {
        w = z.mul(w).scale(2.0).add(Complex::new(1.0, 0.0));
        z = z.mul(z).add(c);
    }

    let magnitude = z.mag();
    let safe_mag = magnitude.max(params.min_magnitude);
    let height = safe_mag.ln();

    let denom = (safe_mag * safe_mag).max(params.min_magnitude * params.min_magnitude);
    let conj_z = Complex::new(z.real, -z.imag);
    let g_complex = conj_z.mul(w).scale(1.0 / denom);
    let gradient = Complex::new(g_complex.real, -g_complex.imag);

    HeightFieldSample {
        height,
        gradient,
        z,
        w,
        magnitude,
    }
}

/// Project a model-proposed delta onto the level set of the height
/// field and apply a corrective term to reduce height error.
pub fn contour_correct_delta(
    model_delta: Complex,
    gradient: Complex,
    height_error: f64,
    params: ContourControllerParams,
) -> Complex {
    let grad_norm_sq = gradient.real * gradient.real + gradient.imag * gradient.imag;
    let denom = grad_norm_sq + params.projection_epsilon;
    let normal_scale = dot(gradient, model_delta) / denom;
    let delta_tangent = sub(model_delta, gradient.scale(normal_scale));
    let correction_scale = params.correction_gain * height_error / denom;
    let correction = gradient.scale(correction_scale);
    sub(delta_tangent, correction)
}

impl ContourState {
    /// Create a new contour state with target height initialized
    /// from the given starting point.
    pub fn new(c: Complex, field_params: HeightFieldParams) -> Self {
        let sample = sample_height_field(c, field_params);
        Self {
            c,
            target_height: sample.height,
            last_delta: Complex::new(0.0, 0.0),
        }
    }

    /// Update the target height explicitly.
    pub fn set_target_height(&mut self, target_height: f64) {
        self.target_height = target_height;
    }

    /// Step the contour controller: sample field, correct the model
    /// delta, and update the current c.
    pub fn step(
        &mut self,
        model_delta: Complex,
        field_params: HeightFieldParams,
        controller_params: ContourControllerParams,
    ) -> ContourStep {
        let sample = sample_height_field(self.c, field_params);
        let height_error = sample.height - self.target_height;
        let corrected_delta = contour_correct_delta(
            model_delta,
            sample.gradient,
            height_error,
            controller_params,
        );

        self.c = self.c.add(corrected_delta);
        self.last_delta = corrected_delta;

        ContourStep {
            c: self.c,
            height: sample.height,
            height_error,
            gradient: sample.gradient,
            corrected_delta,
        }
    }
}
