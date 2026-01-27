use runtime_core::geometry::Complex;
use runtime_core::height_controller::{
    contour_correct_delta,
    sample_height_field,
    ContourControllerParams,
    HeightFieldParams,
};

#[test]
fn height_field_sample_is_finite() {
    let params = HeightFieldParams {
        iterations: 32,
        min_magnitude: 1.0e-6,
    };
    let c = Complex::new(-0.75, 0.1);
    let sample = sample_height_field(c, params);
    assert!(sample.height.is_finite());
    assert!(sample.gradient.real.is_finite());
    assert!(sample.gradient.imag.is_finite());
}

#[test]
fn contour_projection_removes_normal_component() {
    let params = ContourControllerParams {
        correction_gain: 0.0,
        projection_epsilon: 1.0e-6,
    };
    let gradient = Complex::new(1.0, 0.0);
    let model_delta = Complex::new(0.5, 0.0);
    let corrected = contour_correct_delta(model_delta, gradient, 0.0, params);
    assert!(corrected.real.abs() < 1.0e-6);
    assert!(corrected.imag.abs() < 1.0e-6);
}

#[test]
fn contour_projection_preserves_tangent_component() {
    let params = ContourControllerParams {
        correction_gain: 0.0,
        projection_epsilon: 1.0e-6,
    };
    let gradient = Complex::new(0.0, 1.0);
    let model_delta = Complex::new(0.3, -0.2);
    let corrected = contour_correct_delta(model_delta, gradient, 0.0, params);
    assert!((corrected.real - model_delta.real).abs() < 1.0e-6);
}
