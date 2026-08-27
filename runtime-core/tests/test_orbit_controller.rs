//! Parity tests: Rust OrbitController vs the original TypeScript implementation
//! (pre-fe1087b). The TS code is the proven baseline; these tests pin the Rust
//! port to it exactly.

use runtime_core::controller::OrbitController;

/// Reference implementation of the May TS `mandelbrotBoundary(s, alpha)`.
fn ts_mandelbrot_boundary(s: f64, alpha: f64) -> (f64, f64) {
    let s = s.clamp(0.01, 3.0);
    let alpha = alpha.clamp(0.0, 1.0);
    let theta = 2.0 * std::f64::consts::PI * alpha;
    let r = 0.25 * (1.0 - theta.cos());
    let scale = s.min(1.5);
    (
        r * (theta / 2.0).cos() * scale,
        r * (theta / 2.0).sin() * scale,
    )
}

#[test]
fn boundary_matches_ts_reference() {
    for i in 0..16 {
        for j in 0..8 {
            let alpha = i as f64 / 16.0;
            let s = 0.2 + 2.6 * j as f64 / 7.0;
            let mut ctrl = OrbitController::new(s, alpha, 1.0);
            let c = ctrl.mandelbrot_boundary();
            let (re, im) = ts_mandelbrot_boundary(s, alpha);
            assert!(
                (c.re - re).abs() < 1e-12 && (c.im - im).abs() < 1e-12,
                "boundary mismatch at alpha={} s={}: rust=({}, {}) ts=({}, {})",
                alpha, s, c.re, c.im, re, im
            );
        }
    }
}

#[test]
fn step_matches_ts_semantics() {
    // TS step: newTheta = (theta + omega*dt) % 2pi; base + sum gate*0.05*e^{i*(k+2)*newTheta}
    let mut ctrl = OrbitController::new(1.2, 0.4, 1.5);
    let gates = [0.9, 0.3, 1.0, 0.0, 0.7, 0.5];
    let mut ts_theta = 0.0f64;
    for _ in 0..120 {
        let dt = 1.0 / 60.0;
        ts_theta = (ts_theta + 1.5 * dt) % (2.0 * std::f64::consts::PI);
        let (bre, bim) = ts_mandelbrot_boundary(1.2, 0.4);
        let mut tre = bre;
        let mut tim = bim;
        for (k, &g) in gates.iter().enumerate() {
            let phase = (k as f64 + 2.0) * ts_theta;
            tre += g * 0.05 * phase.cos();
            tim += g * 0.05 * phase.sin();
        }
        ctrl.apply_controls(1.2, 0.4);
        let c = ctrl.step(dt, Some(&gates));
        assert!((c.re - tre).abs() < 1e-12 && (c.im - tim).abs() < 1e-12);
    }
}

#[test]
fn audio_drives_position_not_time() {
    // The whole point of the May controller: changing (s, alpha) moves c,
    // regardless of time. Freeze theta by omega=0 and vary controls.
    let mut ctrl = OrbitController::new(1.0, 0.0, 0.0);
    let c1 = ctrl.step(1.0 / 60.0, None);
    ctrl.apply_controls(1.0, 0.5); // opposite side of cardioid
    let c2 = ctrl.step(1.0 / 60.0, None);
    let moved = ((c2.re - c1.re).powi(2) + (c2.im - c1.im).powi(2)).sqrt();
    assert!(moved > 0.1, "controls must move c directly, moved={}", moved);
}
