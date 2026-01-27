use runtime_core::controller::{
    evaluate_height_field, step_height_controller, HeightControllerParams, DEFAULT_HEIGHT_EPSILON,
    DEFAULT_HEIGHT_ITERATIONS,
};
use runtime_core::geometry::Complex;

#[test]
fn test_height_field_is_finite() {
    let c = Complex::new(-0.75, 0.1);
    let sample = evaluate_height_field(c, DEFAULT_HEIGHT_ITERATIONS, DEFAULT_HEIGHT_EPSILON);
    assert!(sample.height.is_finite());
    assert!(sample.gradient.real.is_finite());
    assert!(sample.gradient.imag.is_finite());
}

#[test]
fn test_controller_step_changes_c() {
    let c = Complex::new(-0.5, 0.2);
    let delta_model = Complex::new(0.01, -0.015);
    let params = HeightControllerParams {
        target_height: -0.5,
        normal_risk: 0.0,
        height_gain: 0.2,
    };
    let step = step_height_controller(
        c,
        delta_model,
        params,
        DEFAULT_HEIGHT_ITERATIONS,
        DEFAULT_HEIGHT_EPSILON,
    );
    assert!(step.new_c.real != c.real || step.new_c.imag != c.imag);
}
