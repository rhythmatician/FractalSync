//! Height-field controller and shared runtime constants.
//!
//! The controller treats the Mandelbrot map as a scalar field
//! f(c) = log|z_N(c)| and projects model-proposed steps onto the
//! level set of f. A small servo term pulls back to the target height.

use crate::geometry::{iterate_with_derivative, Complex};

/// Shared runtime constants to keep backend and frontend in lockstep.
pub const SAMPLE_RATE: usize = 48_000;
pub const HOP_LENGTH: usize = 1_024;
pub const N_FFT: usize = 4_096;
pub const WINDOW_FRAMES: usize = 10;

/// Default height-field parameters.
pub const DEFAULT_HEIGHT_ITERATIONS: usize = 32;
pub const DEFAULT_HEIGHT_EPSILON: f64 = 1e-8;
pub const DEFAULT_HEIGHT_GAIN: f64 = 0.15;

/// Result of evaluating the height field at a point.
#[derive(Clone, Copy, Debug)]
pub struct HeightFieldSample {
    pub height: f64,
    /// Gradient in the c-plane: (real = ∂f/∂x, imag = ∂f/∂y).
    pub gradient: Complex,
    pub z: Complex,
    pub w: Complex,
}

/// Controller parameters for projecting a model step.
#[derive(Clone, Copy, Debug)]
pub struct HeightControllerParams {
    pub target_height: f64,
    /// 0 = fully tangent, 1 = allow full normal motion.
    pub normal_risk: f64,
    pub height_gain: f64,
}

/// Result of a controller step.
#[derive(Clone, Copy, Debug)]
pub struct HeightControllerStep {
    pub new_c: Complex,
    pub delta: Complex,
    pub height: f64,
    pub gradient: Complex,
}

/// Evaluate f(c) = log|z_N(c)| and its gradient.
pub fn evaluate_height_field(
    c: Complex,
    iterations: usize,
    epsilon: f64,
) -> HeightFieldSample {
    let (z, w) = iterate_with_derivative(c, iterations);
    let z_mag_sq = z.mag_sq().max(epsilon);
    let z_mag = z_mag_sq.sqrt();
    let height = (z_mag + epsilon).ln();

    let denom = z_mag_sq + epsilon;
    let a = z.conj().scale(1.0 / denom).mul(w);
    let gradient = Complex::new(a.real, -a.imag);

    HeightFieldSample {
        height,
        gradient,
        z,
        w,
    }
}

/// Project a model step onto the height-field level set and apply a servo term.
pub fn step_height_controller(
    c: Complex,
    delta_model: Complex,
    params: HeightControllerParams,
    iterations: usize,
    epsilon: f64,
) -> HeightControllerStep {
    let sample = evaluate_height_field(c, iterations, epsilon);
    let g = sample.gradient;
    let g2 = (g.real * g.real + g.imag * g.imag).max(epsilon);

    let normal_component = (g.real * delta_model.real + g.imag * delta_model.imag) / g2;
    let projection_scale = (1.0 - params.normal_risk).clamp(0.0, 1.0) * normal_component;
    let projected = Complex::new(
        delta_model.real - g.real * projection_scale,
        delta_model.imag - g.imag * projection_scale,
    );

    let height_error = sample.height - params.target_height;
    let servo_scale = -params.height_gain * height_error / g2;
    let servo = Complex::new(g.real * servo_scale, g.imag * servo_scale);

    let delta = Complex::new(projected.real + servo.real, projected.imag + servo.imag);
    let new_c = Complex::new(c.real + delta.real, c.imag + delta.imag);

    HeightControllerStep {
        new_c,
        delta,
        height: sample.height,
        gradient: sample.gradient,
    }
}
