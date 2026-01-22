use runtime_core::controller::{OrbitState, ResidualParams, synthesize, step};
use runtime_core::geometry::{lobe_point_at_angle, period_n_bulb_radius};
use runtime_core::features::FeatureExtractor;

#[test]
fn geometry_main_cardioid_known_points() {
    // theta = 0 => mu = 1 => c = 1/4
    let c0 = lobe_point_at_angle(1, 0, 0.0, 1.0);
    assert!((c0.real - 0.25).abs() < 1e-12);
    assert!(c0.imag.abs() < 1e-12);

    // theta = pi => mu = -1 => c = -3/4
    let cpi = lobe_point_at_angle(1, 0, std::f64::consts::PI, 1.0);
    assert!((cpi.real + 0.75).abs() < 1e-12);
    assert!(cpi.imag.abs() < 1e-12);
}

#[test]
fn controller_alpha_zero_returns_carrier() {
    let mut state = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.0, 6, 1.0, 123);
    let params = ResidualParams::default();

    let carrier = lobe_point_at_angle(1, 0, state.theta, state.s);
    let c = synthesize(&state, params, None);
    assert!((c.real - carrier.real).abs() < 1e-12);
    assert!((c.imag - carrier.imag).abs() < 1e-12);

    // stepping should still return carrier because alpha=0
    let c2 = step(&mut state, 0.1, params, None);
    let carrier2 = lobe_point_at_angle(1, 0, state.theta, state.s);
    assert!((c2.real - carrier2.real).abs() < 1e-12);
    assert!((c2.imag - carrier2.imag).abs() < 1e-12);
}

#[test]
fn controller_seed_is_deterministic() {
    let p = ResidualParams::default();
    let mut a = OrbitState::new_with_seed(2, 0, 0.123, 0.5, 1.0, 0.9, 6, 1.0, 999);
    let mut b = OrbitState::new_with_seed(2, 0, 0.123, 0.5, 1.0, 0.9, 6, 1.0, 999);

    for _ in 0..10 {
        let ca = step(&mut a, 0.01, p, None);
        let cb = step(&mut b, 0.01, p, None);
        assert!((ca.real - cb.real).abs() < 1e-12);
        assert!((ca.imag - cb.imag).abs() < 1e-12);
    }
}

#[test]
fn features_short_audio_no_panic_and_correct_shape() {
    let fx = FeatureExtractor::default();
    // shorter than n_fft
    let audio = vec![0.0_f32; 1024];
    let windows = fx.extract_windowed_features(&audio[..], 8);

    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].len(), fx.num_features_per_frame() * 8);
}

#[test]
fn period_radius_is_positive_for_known_lobes() {
    assert!(period_n_bulb_radius(2, 0) > 0.0);
    assert!(period_n_bulb_radius(3, 0) > 0.0);
}
