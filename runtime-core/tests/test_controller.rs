use runtime_core::controller::{OrbitState, ResidualParams, synthesize, step};

#[test]
fn test_orbit_state_initialization() {
    let state = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.02, 0.3, 6, 1.0, 12345);
    
    assert_eq!(state.lobe, 1);
    assert_eq!(state.sub_lobe, 0);
    assert_eq!(state.theta, 0.0);
    assert_eq!(state.omega, 1.0);
    assert_eq!(state.s, 1.02);
    assert_eq!(state.alpha, 0.3);
}

#[test]
fn test_deterministic_synthesis() {
    let params = ResidualParams::default();
    
    let state1 = OrbitState::new_with_seed(1, 0, 0.5, 1.0, 1.0, 0.5, 6, 1.0, 999);
    let state2 = OrbitState::new_with_seed(1, 0, 0.5, 1.0, 1.0, 0.5, 6, 1.0, 999);
    
    let c1 = synthesize(&state1, params, None);
    let c2 = synthesize(&state2, params, None);
    
    assert_eq!(c1.re, c2.re);
    assert_eq!(c1.im, c2.im);
}

#[test]
fn test_deterministic_stepping() {
    let params = ResidualParams::default();
    
    let mut state1 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 42);
    let mut state2 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 42);
    
    for _ in 0..20 {
        let c1 = step(&mut state1, 0.01, params, None);
        let c2 = step(&mut state2, 0.01, params, None);
        
        assert_eq!(c1.re, c2.re);
        assert_eq!(c1.im, c2.im);
    }
}

#[test]
fn test_alpha_zero_returns_carrier() {
    let params = ResidualParams::default();
    let state = OrbitState::new_with_seed(1, 0, 0.123, 1.0, 1.0, 0.0, 6, 1.0, 123);
    
    let c = synthesize(&state, params, None);
    
    // With alpha=0, should return just the carrier (lobe point)
    let carrier = runtime_core::geometry::lobe_point_at_angle(1, 0, 0.123, 1.0);
    
    assert!((c.re - carrier.re).abs() < 1e-10);
    assert!((c.im - carrier.im).abs() < 1e-10);
}

#[test]
fn test_alpha_one_includes_residuals() {
    let params = ResidualParams::default();
    let state = OrbitState::new_with_seed(1, 0, 0.123, 1.0, 1.0, 1.0, 6, 1.0, 123);
    
    let c = synthesize(&state, params, None);
    let carrier = runtime_core::geometry::lobe_point_at_angle(1, 0, 0.123, 1.0);
    
    // With alpha=1.0, result should differ from pure carrier
    let diff = ((c.re - carrier.re).powi(2) + (c.im - carrier.im).powi(2)).sqrt();
    assert!(diff > 1e-6, "Alpha=1.0 should produce residuals");
}

#[test]
fn test_k_residuals_effect() {
    let params = ResidualParams::default();
    
    let state_k3 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 3, 1.0, 999);
    let state_k6 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 999);
    
    let c3 = synthesize(&state_k3, params, None);
    let c6 = synthesize(&state_k6, params, None);
    
    // Different k_residuals should produce different results
    let diff = ((c3.re - c6.re).powi(2) + (c3.im - c6.im).powi(2)).sqrt();
    assert!(diff > 1e-6, "Different k_residuals should affect output");
}

#[test]
fn test_omega_scale_sensitivity() {
    let params = ResidualParams::default();
    
    // Create two states with different omega_scale values
    let mut state1 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 999);
    let mut state2 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 2.0, 999);
    
    // Advance time so that omega_scale difference has an effect
    let dt = 0.1;
    state1.advance(dt);
    state2.advance(dt);
    
    let c1 = synthesize(&state1, params, None);
    let c2 = synthesize(&state2, params, None);
    
    // Different omega_scale should cause phases to diverge after time advances
    let diff = ((c1.re - c2.re).powi(2) + (c1.im - c2.im).powi(2)).sqrt();
    assert!(diff > 1e-6, "Omega scale should affect output after time advances");
}

#[test]
fn test_seed_changes_residuals() {
    let params = ResidualParams::default();
    
    let state1 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 111);
    let state2 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 222);
    
    let c1 = synthesize(&state1, params, None);
    let c2 = synthesize(&state2, params, None);
    
    // Different seeds should produce different residuals
    let diff = ((c1.re - c2.re).powi(2) + (c1.im - c2.im).powi(2)).sqrt();
    assert!(diff > 1e-6, "Different seeds should affect output");
}

#[test]
fn test_step_advances_theta() {
    let params = ResidualParams::default();
    let mut state = OrbitState::new_with_seed(1, 0, 0.0, 2.0, 1.0, 0.5, 6, 1.0, 123);
    
    let initial_theta = state.theta;
    step(&mut state, 0.1, params, None);
    
    assert!(
        (state.theta - initial_theta).abs() > 1e-6,
        "Step should advance theta"
    );
}

#[test]
fn test_step_with_dt_consistency() {
    let params = ResidualParams::default();
    
    // Step with large dt
    let mut state1 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 42);
    step(&mut state1, 1.0, params, None);
    
    // Step with small dt multiple times
    let mut state2 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 42);
    for _ in 0..10 {
        step(&mut state2, 0.1, params, None);
    }
    
    // Should end up at approximately the same theta
    assert!(
        (state1.theta - state2.theta).abs() < 0.01,
        "Multiple small steps should approximate one large step"
    );
}

#[test]
fn test_band_gates_effect() {
    let params = ResidualParams::default();
    let state = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 123);
    
    // All gates open (1.0)
    let gates_open = vec![1.0; 6];
    let c_open = synthesize(&state, params, Some(&gates_open));
    
    // All gates closed (0.0)
    let gates_closed = vec![0.0; 6];
    let c_closed = synthesize(&state, params, Some(&gates_closed));
    
    // With gates closed, should be closer to pure carrier
    let carrier = runtime_core::geometry::lobe_point_at_angle(1, 0, state.theta, state.s);
    let dist_open = ((c_open.re - carrier.re).powi(2) + (c_open.im - carrier.im).powi(2)).sqrt();
    let dist_closed = ((c_closed.re - carrier.re).powi(2) + (c_closed.im - carrier.im).powi(2)).sqrt();
    
    assert!(
        dist_closed < dist_open,
        "Closed gates should produce output closer to carrier"
    );
}

#[test]
fn test_residual_params_effect() {
    let params_default = ResidualParams::default();
    let params_custom = ResidualParams {
        k_residuals: 6,
        residual_cap: 0.5,
        radius_scale: 2.0,
    };
    
    let state = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 123);
    
    let c_default = synthesize(&state, params_default, None);
    let c_custom = synthesize(&state, params_custom, None);
    
    // Different residual params should affect output
    let diff = ((c_default.re - c_custom.re).powi(2) + (c_default.im - c_custom.im).powi(2)).sqrt();
    assert!(diff > 1e-6, "Residual params should affect output");
}

use runtime_core::controller::PlayerState;

#[test]
fn test_player_state_starts_on_boundary() {
    // PlayerState::new should place c on the boundary point for (s, alpha).
    let p = PlayerState::new(1, 0, 1.0, 0.25);
    let expected = runtime_core::geometry::lobe_point_at_angle(
        1,
        0,
        0.25 * 2.0 * std::f64::consts::PI,
        1.0,
    );
    assert!((p.c.re - expected.re).abs() < 1e-9);
    assert!((p.c.im - expected.im).abs() < 1e-9);
}

#[test]
fn test_player_state_moves_toward_target() {
    // Without a pyramid, PlayerState falls back to clamped motion toward the
    // model-driven target. Changing alpha should pull c toward the new point.
    runtime_core::minimap::clear_pyramid();
    let mut p = PlayerState::new(1, 0, 1.0, 0.0);
    let start = p.c;
    // Move the target to alpha=0.5 (opposite side of the cardioid).
    p.apply_controls(1.0, 0.5, 1.0);
    let c = p.step(1.0 / 60.0, 0.0, None);
    let moved = ((c.re - start.re).powi(2) + (c.im - start.im).powi(2)).sqrt();
    assert!(
        moved > 1e-6,
        "PlayerState should move toward the model-driven target, moved={}",
        moved
    );
}

#[test]
fn test_player_state_does_not_trace_closed_loop() {
    // With a fixed target and no gates, the momentum integrator still
    // converges: friction bleeds velocity, c settles near the target.
    runtime_core::minimap::clear_pyramid();
    let mut p = PlayerState::new(1, 0, 1.0, 0.5);
    for _ in 0..600 {
        p.step(1.0 / 60.0, 0.0, None);
    }
    // After settling, stepping more should not move it much (no loop).
    let before = p.c;
    for _ in 0..60 {
        p.step(1.0 / 60.0, 0.0, None);
    }
    let drift = ((p.c.re - before.re).powi(2) + (p.c.im - before.im).powi(2)).sqrt();
    assert!(
        drift < 1e-3,
        "PlayerState should settle at the target, not loop (drift={})",
        drift
    );
}

#[test]
fn test_player_state_has_momentum() {
    // The frozen-c regression: with a barely-moving target, a pure servo
    // parks at its fixed point. The momentum integrator must keep moving
    // when the model's controls vary frame-to-frame (as real output does).
    runtime_core::minimap::clear_pyramid();
    let mut p = PlayerState::new(1, 0, 2.7, 0.95);
    let mut total_travel = 0.0f64;
    let mut prev = p.c;
    for i in 0..600 {
        let t = i as f64;
        let s_t = 2.7 + 0.03 * (t * 0.05).sin();
        let a_t = (0.95 + 0.002 * (t * 0.03).cos()).clamp(0.0, 1.0);
        p.apply_controls(s_t, a_t, 4.0);
        let c = p.step(1.0 / 60.0, 0.0, Some(&[0.95; 6]));
        total_travel += ((c.re - prev.re).powi(2) + (c.im - prev.im).powi(2)).sqrt();
        prev = c;
    }
    assert!(
        total_travel > 0.05,
        "PlayerState must keep wandering when controls vary (path={})",
        total_travel
    );
}

#[test]
fn test_synthesize_returns_finite_values() {
    let params = ResidualParams::default();
    
    // Test with various parameter combinations
    for seed in [1, 42, 123, 999] {
        for alpha in [0.0, 0.3, 0.7, 1.0] {
            let state = OrbitState::new_with_seed(1, 0, 0.5, 1.0, 1.0, alpha, 6, 1.0, seed);
            let c = synthesize(&state, params, None);
            
            assert!(c.re.is_finite(), "Real part must be finite");
            assert!(c.im.is_finite(), "Imag part must be finite");
        }
    }
}

#[test]
fn test_different_lobes() {
    let params = ResidualParams::default();
    
    let state_lobe1 = OrbitState::new_with_seed(1, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 123);
    let state_lobe2 = OrbitState::new_with_seed(2, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 123);
    let state_lobe3 = OrbitState::new_with_seed(3, 0, 0.0, 1.0, 1.0, 0.5, 6, 1.0, 123);
    
    let c1 = synthesize(&state_lobe1, params, None);
    let c2 = synthesize(&state_lobe2, params, None);
    let c3 = synthesize(&state_lobe3, params, None);
    
    // Different lobes should produce different outputs
    let diff12 = ((c1.re - c2.re).powi(2) + (c1.im - c2.im).powi(2)).sqrt();
    let diff23 = ((c2.re - c3.re).powi(2) + (c2.im - c3.im).powi(2)).sqrt();
    
    assert!(diff12 > 0.01, "Different lobes should produce different outputs");
    assert!(diff23 > 0.01, "Different lobes should produce different outputs");
}
