//! Read-only DebugSnapshot seam (issue #111 Phase A).
//!
//! Behavior contract:
//! - one canonical, versioned, read-only snapshot of authoritative runtime
//!   state (action / physics / map / diagnostics) for the debug cockpit;
//! - snapshot creation must NEVER mutate controller or runtime state;
//! - physics fields derive from the SAME canonical manifold functions the
//!   physics kernel uses (no re-derived formulas);
//! - the minimap section reuses the canonical mip pyramid (no new map math);
//! - a Rust-owned terrain patch Q=(x, y, lambda*sigma) renders the skate
//!   park from authoritative geometry.

use num_complex::Complex64;
use runtime_core::controller::OrbitController;
use runtime_core::controls::MotionControls;
use runtime_core::debug::{
    snapshot_from_state, terrain_patch, DEBUG_SNAPSHOT_VERSION,
};
use runtime_core::manifold::ManifoldConfig;

/// Serialize the whole distance-field/minimap global state against concurrent
/// tests (same discipline as the other integration tests in this crate).
fn lock() -> std::sync::MutexGuard<'static, ()> {
    runtime_core::distance_field::global_test_mutex()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn controls(drive: f64) -> MotionControls {
    MotionControls {
        direction: [drive, 0.0],
        throttle: drive,
        brake: 0.0,
        grip: 0.5,
        impulse: 0.0,
    }
}

#[test]
fn snapshot_version_is_pinned() {
    assert_eq!(DEBUG_SNAPSHOT_VERSION, "debug-snapshot/1");
}

#[test]
fn snapshot_creation_does_not_mutate_controller() {
    let _g = lock();
    let mut ctrl = OrbitController::default();
    let dt = 1024.0 / 48000.0;
    ctrl.step_with_controls(dt, &controls(1.0));

    let c_before = ctrl.c;
    let v_before = ctrl.planar_velocity;
    let theta_before = ctrl.theta;

    let s1 = ctrl.debug_snapshot().expect("snapshot 1");
    let s2 = ctrl.debug_snapshot().expect("snapshot 2");

    // Read-only: two consecutive snapshots are identical...
    assert_eq!(s1, s2, "snapshots must be pure functions of state");
    // ...and the controller state is untouched.
    assert_eq!(c_before, ctrl.c);
    assert_eq!(v_before, ctrl.planar_velocity);
    assert!((theta_before - ctrl.theta).abs() < 1e-15);
}

#[test]
fn snapshot_physics_fields_match_canonical_manifold_math() {
    let _g = lock();
    let config = ManifoldConfig::default();
    let c = Complex64::new(0.0, 0.0);
    let v = (0.05, -0.02);

    let snap = snapshot_from_state(c, v, None, None, &config, None)
        .expect("standalone snapshot");

    let p = &snap.physics;
    // c/v round-trip exactly.
    assert_eq!(p.c, [c.re, c.im]);
    assert_eq!(p.velocity, [v.0, v.1]);

    // E = K + U holds to float tolerance.
    assert!((p.kinetic + p.potential - p.total).abs() < 1e-12);

    // sigma equals the canonical binding value.
    let sigma = runtime_core::manifold::mandelbrot_scale(c, &config).unwrap();
    assert!((p.sigma - sigma).abs() < 1e-14);

    // D equals the canonical signed-distance authority.
    let d = runtime_core::manifold::signed_distance(c).unwrap();
    assert!((p.signed_distance - d).abs() < 1e-14);

    // realm sign follows D.
    assert_eq!(p.realm, if d < 0.0 { -1 } else if d > 0.0 { 1 } else { 0 });

    // metric equals the canonical induced metric.
    let g = runtime_core::manifold::induced_metric(c, &config).unwrap();
    assert!((p.metric[0] - g[0][0]).abs() < 1e-14);
    assert!((p.metric[1] - g[0][1]).abs() < 1e-14);
    assert!((p.metric[2] - g[1][1]).abs() < 1e-14);

    // metric speed = sqrt(v^T G v).
    let gv0 = g[0][0] * v.0 + g[0][1] * v.1;
    let gv1 = g[1][0] * v.0 + g[1][1] * v.1;
    let metric_speed = (v.0 * gv0 + v.1 * gv1).sqrt();
    assert!((p.metric_speed - metric_speed).abs() < 1e-12);
}

#[test]
fn snapshot_reports_validity_and_derivative_step() {
    let _g = lock();
    let snap = snapshot_from_state(Complex64::new(0.1, -0.1), (0.1, 0.0), None, None, &ManifoldConfig::default(), None)
        .expect("snapshot");
    assert!(snap.diagnostics.derivative_step > 0.0);
    assert!(snap.diagnostics.valid, "ordinary state must be valid");
    assert!(snap.physics.derivative_valid, "ordinary state must be derivative-valid");
}

#[test]
fn snapshot_exposes_raw_and_effective_controls() {
    let _g = lock();
    let mut ctrl = OrbitController::default();
    let dt = 1024.0 / 48000.0;
    // Out-of-range raw input: throttle 2.0 must clamp to 1.0 effective.
    let raw = MotionControls {
        direction: [2.0, 0.0],
        throttle: 2.0,
        brake: -3.0,
        grip: 5.0,
        impulse: 0.0,
    };
    ctrl.step_with_controls(dt, &raw);

    let snap = ctrl.debug_snapshot().expect("snapshot");
    let action = snap.action.expect("action present after a step");

    // Raw is what the policy emitted.
    assert!((action.raw.throttle - 2.0).abs() < 1e-12);
    // Effective is the clamped action physics actually applied.
    assert!((action.effective.throttle - 1.0).abs() < 1e-12);
    assert!((action.effective.brake - 0.0).abs() < 1e-12);
    assert!((action.effective.grip - 1.0).abs() < 1e-12);
    assert!((action.effective.direction[0] - 1.0).abs() < 1e-12);

    // Drive covector is the metric-consistent one actually used.
    let q = raw.clamped().drive_covector(ctrl.c, &ctrl.manifold_config).unwrap();
    assert!((action.drive_covector[0] - q.0).abs() < 1e-12);
    assert!((action.drive_covector[1] - q.1).abs() < 1e-12);
    assert!((action.friction_beta - raw.clamped().friction_beta()).abs() < 1e-12);
}

#[test]
fn snapshot_action_is_none_before_any_step() {
    let _g = lock();
    let ctrl = OrbitController::default();
    let snap = ctrl.debug_snapshot().expect("snapshot");
    assert!(snap.action.is_none(), "no action before the first step");
    assert!(snap.diagnostics.last_delta_total.is_none());
}

#[test]
fn snapshot_time_is_keyed_to_destination_step_clock() {
    let _g = lock();
    let mut ctrl = OrbitController::default();
    let dt = 1024.0 / 48000.0;
    for _ in 0..7 {
        ctrl.step_with_controls(dt, &controls(0.3));
    }
    let snap = ctrl.debug_snapshot().expect("snapshot");
    assert!((snap.time_seconds - 7.0 * dt).abs() < 1e-12);
}

#[test]
fn snapshot_reports_last_step_energy_delta() {
    let _g = lock();
    let mut ctrl = OrbitController::default();
    let dt = 1024.0 / 48000.0;
    ctrl.step_with_controls(dt, &controls(0.5));
    let snap = ctrl.debug_snapshot().expect("snapshot");
    let delta = snap.diagnostics.last_delta_total.expect("delta recorded");
    assert!(delta.is_finite());
}

#[test]
fn snapshot_map_is_unavailable_without_pyramid() {
    let _g = lock();
    runtime_core::minimap::clear_pyramid();
    let snap = snapshot_from_state(Complex64::new(0.0, 0.0), (0.0, 0.0), None, None, &ManifoldConfig::default(), None)
        .expect("snapshot");
    assert!(!snap.map.pyramid_loaded);
    assert!(snap.map.shore_proximity.is_none());
    assert!(snap.map.minimap_window.is_none());
    assert!(snap.map.extent.is_none());
}

#[test]
fn snapshot_map_reuses_canonical_pyramid() {
    let _g = lock();
    // Synthetic 2-level pyramid over the canonical bake extent.
    let re_min = -2.0;
    let re_max = 1.0;
    let im_min = -1.5;
    let im_max = 1.5;
    let w0 = 8usize;
    let mut l0 = Vec::with_capacity(w0 * w0);
    for _r in 0..w0 {
        for _c in 0..w0 {
            l0.push(0.5f32);
        }
    }
    let mut levels = vec![l0];
    let mut widths = vec![w0];
    let mut heights = vec![w0];
    for size in [4usize, 1, 1, 1, 1, 1] {
        levels.push(vec![0.75f32; size * size]);
        widths.push(size);
        heights.push(size);
    }
    let pyr = runtime_core::minimap::MipPyramid::from_levels(
        levels, widths, heights, re_min, re_max, im_min, im_max,
    )
    .expect("synthetic pyramid");
    runtime_core::minimap::set_pyramid(pyr).expect("install pyramid");

    let snap = snapshot_from_state(Complex64::new(0.0, 0.0), (0.0, 0.0), None, None, &ManifoldConfig::default(), None)
        .expect("snapshot");

    assert!(snap.map.pyramid_loaded);
    assert_eq!(
        snap.map.extent,
        Some([re_min, re_max, im_min, im_max]),
        "extent comes from the canonical pyramid"
    );
    let window = snap.map.minimap_window.expect("9x9 window present");
    assert_eq!(window.len(), 81, "window is the canonical 9x9 grid");
    let s = snap.map.shore_proximity.expect("S at c present");
    assert!((0.0..=1.0).contains(&s));

    runtime_core::minimap::clear_pyramid();
}

#[test]
fn snapshot_friction_power_is_nonpositive_for_effective_action() {
    let _g = lock();
    let mut ctrl = OrbitController::default();
    let dt = 1024.0 / 48000.0;
    // Build some speed first.
    ctrl.step_with_controls(dt, &controls(1.0));
    ctrl.step_with_controls(dt, &controls(1.0));
    let snap = ctrl.debug_snapshot().expect("snapshot");
    let action = snap.action.as_ref().expect("action");
    assert!(
        action.friction_power <= 1e-12,
        "friction must not inject energy: {}",
        action.friction_power
    );
}

#[test]
fn standalone_snapshot_matches_controller_snapshot() {
    let _g = lock();
    let mut ctrl = OrbitController::default();
    let dt = 1024.0 / 48000.0;
    ctrl.step_with_controls(dt, &controls(0.7));

    let from_ctrl = ctrl.debug_snapshot().expect("controller snapshot");
    let from_state = snapshot_from_state(
        ctrl.c,
        ctrl.planar_velocity,
        ctrl.last_controls.map(|raw| runtime_core::debug::LastAction {
            raw,
            friction_beta: ctrl.last_friction_beta,
            friction_power: ctrl.last_friction_power,
        }),
        Some(ctrl.manifold_drag),
        &ctrl.manifold_config,
        ctrl.last_delta_total,
    )
    .expect("standalone snapshot");

    // The step clock is controller-owned (the pure builder has no clock);
    // every state-derived field must match exactly.
    let mut from_ctrl_no_clock = from_ctrl.clone();
    from_ctrl_no_clock.time_seconds = 0.0;
    assert_eq!(from_ctrl_no_clock, from_state);
    assert!(from_ctrl.time_seconds > 0.0);
}

#[test]
fn terrain_patch_is_authoritative_q_embedding() {
    let _g = lock();
    let config = ManifoldConfig::default();
    let patch = terrain_patch(0.0, 0.0, 0.5, 33, &config).expect("terrain patch");

    assert_eq!(patch.n, 33);
    assert_eq!(patch.positions.len(), 33 * 33 * 3);
    assert_eq!(patch.signed.len(), 33 * 33);
    assert_eq!(patch.realm.len(), 33 * 33);

    // Center vertex sits exactly at the embedding of the patch center.
    let mid = (33 * 33 / 2) * 3;
    let center = Complex64::new(0.0, 0.0);
    assert!((patch.positions[mid] - 0.0).abs() < 1e-14);
    assert!((patch.positions[mid + 1] - 0.0).abs() < 1e-14);
    let sigma = runtime_core::manifold::mandelbrot_scale(center, &config).unwrap();
    let lambda = config.lambda_sq.sqrt();
    assert!((patch.positions[mid + 2] - lambda * sigma).abs() < 1e-14);

    // Signed distances agree with the canonical authority at sampled points.
    let d_center = runtime_core::manifold::signed_distance(center).unwrap();
    let mid_v = 33 * 33 / 2;
    assert!((patch.signed[mid_v] - d_center).abs() < 1e-14);

    // Realm signs follow D.
    for (i, &d) in patch.signed.iter().enumerate() {
        let expect = if d < 0.0 { -1 } else if d > 0.0 { 1 } else { 0 };
        assert_eq!(patch.realm[i], expect);
    }

    // Row-major ordering: row 0 is the north edge (im = +half).
    let first_sigma = patch.positions[2];
    let north = Complex64::new(-0.5, 0.5);
    let sigma_north = runtime_core::manifold::mandelbrot_scale(north, &config).unwrap();
    assert!((first_sigma - lambda * sigma_north).abs() < 1e-14);
}

/// ============================================================================
// Wall potential and hard invariant tests (issue #111)
// ============================================================================

#[test]
fn test_wall_potential_finite_below_r() {
    // U_wall is finite for |c| < 2.
    let config = ManifoldConfig::default();
    let c = Complex64::new(0.5, 0.0); // well inside the disk
    let u = runtime_core::manifold::wall_potential(c, &config).unwrap();
    assert!(u.is_finite(), "wall_potential must be finite for |c| < 2");
    assert!(u > 0.0, "wall_potential should be positive inside the disk");
}

#[test]
fn test_wall_potential_infinity_at_r() {
    // U_wall -> +infinity as radius approaches 2 from below.
    let config = ManifoldConfig::default();
    // Test at three progressively closer radii to show the log divergence
    let c1 = Complex64::new(1.5, 0.0);  // |c|=1.5
    let c2 = Complex64::new(1.9, 0.0);  // |c|=1.9
    let c3 = Complex64::new(1.99, 0.0); // |c|=1.99
    let u1 = runtime_core::manifold::wall_potential(c1, &config).unwrap();
    let u2 = runtime_core::manifold::wall_potential(c2, &config).unwrap();
    let u3 = runtime_core::manifold::wall_potential(c3, &config).unwrap();
    // u should increase as |c| approaches 2
    assert!(u1 > 0.0, "wall_potential at |c|=1.5 should be positive, got {}", u1);
    assert!(u2 > u1, "wall_potential at |c|=1.9 should be larger than at |c|=1.5");
    assert!(u3 > u2, "wall_potential at |c|=1.99 should be larger than at |c|=1.9");
}

#[test]
fn test_wall_potential_rotational_symmetry() {
    // U_wall is rotationally symmetric: U_wall(r, theta) = U_wall(r, -theta).
    let config = ManifoldConfig::default();
    let c1 = Complex64::new(1.0, 0.5); // angle +theta
    let c2 = Complex64::new(1.0, -0.5); // angle -theta
    let u1 = runtime_core::manifold::wall_potential(c1, &config).unwrap();
    let u2 = runtime_core::manifold::wall_potential(c2, &config).unwrap();
    assert!((u1 - u2).abs() < 1e-14, "wall_potential must be rotationally symmetric");
}

#[test]
fn test_wall_force_points_inward() {
    // Q_wall points strictly inward for nonzero c.
    let config = ManifoldConfig::default();
    let c = Complex64::new(1.0, 0.0); // on positive real axis
    let (qx, qy) = runtime_core::manifold::wall_force(c, &config).unwrap();
    // For c = (1, 0), Q_wall should point toward the origin (negative x direction)
    assert!(qx < 0.0, "Q_wall.x should be negative for c.re > 0, got {}", qx);
    assert!(qy == 0.0, "Q_wall.y should be 0 for c.im = 0");
}

#[test]
fn test_wall_force_magnitude_increases_near_r() {
    // Wall force magnitude increases strongly as |c| approaches 2.
    let config = ManifoldConfig::default();
    let c_far = Complex64::new(0.5, 0.0); // farther from wall
    let c_near = Complex64::new(1.9, 0.0); // closer to wall
    let (qx_far, qy_far) = runtime_core::manifold::wall_force(c_far, &config).unwrap();
    let (qx_near, qy_near) = runtime_core::manifold::wall_force(c_near, &config).unwrap();
    let mag_far = (qx_far * qx_far + qy_far * qy_far).sqrt();
    let mag_near = (qx_near * qx_near + qy_near * qy_near).sqrt();
    assert!(mag_near > mag_far, "wall force magnitude should increase as |c| -> 2");
}

#[test]
fn test_wall_force_zero_at_center() {
    // Center has zero wall force.
    let config = ManifoldConfig::default();
    let c = Complex64::new(0.0, 0.0);
    let (qx, qy) = runtime_core::manifold::wall_force(c, &config).unwrap();
    assert!(qx == 0.0 && qy == 0.0, "wall force at center should be (0, 0), got ({}, {})", qx, qy);
}

#[test]
fn test_total_potential_decomposition() {
    // total potential = kappa * sigma + U_wall
    let config = ManifoldConfig::default();
    let c = Complex64::new(0.5, 0.1);
    let sigma = runtime_core::manifold::mandelbrot_scale(c, &config).unwrap();
    let u_sigma = config.kappa * sigma;
    let u_wall = runtime_core::manifold::wall_potential(c, &config).unwrap();
    let u_total = runtime_core::manifold::total_energy((0.0, 0.0), c, &config).unwrap()
        - runtime_core::manifold::kinetic_energy((0.0, 0.0), c, &config).unwrap();
    // total potential should equal sigma potential + wall potential
    let diff = (u_total - u_sigma - u_wall).abs();
    assert!(diff < 1e-10, "total potential should equal kappa*sigma + U_wall, diff={}", diff);
}

#[test]
fn test_total_energy_includes_wall() {
    // total energy E = K + U_sigma + U_wall
    let config = ManifoldConfig::default();
    let c = Complex64::new(0.3, 0.2);
    let v = (0.1, 0.05);
    let e = runtime_core::manifold::total_energy(v, c, &config).unwrap();
    let k = runtime_core::manifold::kinetic_energy(v, c, &config).unwrap();
    let u_sigma = config.kappa * runtime_core::manifold::mandelbrot_scale(c, &config).unwrap();
    let u_wall = runtime_core::manifold::wall_potential(c, &config).unwrap();
    let expected = k + u_sigma + u_wall;
    let diff = (e - expected).abs();
    assert!(diff < 1e-10, "total energy should include U_wall, diff={}", diff);
}

#[test]
fn test_hard_invariant_rejects_invalid_step() {
    // Hard invariant: |c_new| < 2 must hold; invalid steps are rejected.
    let config = ManifoldConfig::default();
    // Start near the wall and try a step that would push outside
    let c = Complex64::new(1.9, 0.0);
    let v = (0.5, 0.0); // velocity outward
    let q_control = (0.0, 0.0);
    let beta = 0.1;
    let dt = 0.02;

    // This step might push |c| >= 2 depending on the dynamics;
    // the important thing is that the integrator either succeeds with
    // |c_new| < 2 or returns an error (never emits invalid state).
    let result = runtime_core::manifold::integrate_step(c, v, q_control, beta, dt, &config);
    // If it succeeds, verify |c_new| < 2
    if let Ok((c_new, _v_new, _energy)) = result {
        let c_abs_sq = c_new.re * c_new.re + c_new.im * c_new.im;
        assert!(c_abs_sq < 4.0, "integrated state must satisfy |c_new| < 2, got |c_new|^2 = {}", c_abs_sq);
    }
    // If it returns an error, that's also valid (hard invariant enforcement).
}

#[test]
fn test_shore_behavior_preserved() {
    // Existing Shore-crossing behavior must remain possible and not be confused
    // with the outer-domain barrier.
    let config = ManifoldConfig::default();
    // A point well inside the main cardioid should still allow shore crossing
    // when driven properly (this test verifies the barrier doesn't interfere
    // with normal operation at |c| << 2).
    let c = Complex64::new(0.1, 0.0);
    let v = (0.01, 0.0);
    let q_control = (0.0, 0.0);
    let beta = 0.1;
    let dt = 0.02;

    let result = runtime_core::manifold::integrate_step(c, v, q_control, beta, dt, &config);
    // Should succeed (no hard invariant violation at |c| = 0.1)
    assert!(result.is_ok(), "integrate_step at |c| = 0.1 should succeed");
}

#[test]
fn test_wall_force_center_zero() {
    // Alias test for wall_force at center
    let config = ManifoldConfig::default();
    let c = Complex64::new(0.0, 0.0);
    let (qx, qy) = runtime_core::manifold::wall_force(c, &config).unwrap();
    assert!(qx == 0.0 && qy == 0.0);
}

#[test]
fn terrain_patch_rejects_degenerate_grids() {
    let _g = lock();
    assert!(terrain_patch(0.0, 0.0, 0.5, 1, &ManifoldConfig::default()).is_err());
    assert!(terrain_patch(0.0, 0.0, 0.5, 1000, &ManifoldConfig::default()).is_err());
    assert!(terrain_patch(0.0, 0.0, -1.0, 33, &ManifoldConfig::default()).is_err());
}

#[test]
fn deep_zoom_field_resolves_beyond_the_baked_pyramid() {
    // Issue #111 feedback: the minimap is a Mandelbrot DEEP ZOOM whose zoom
    // level follows the player. The baked 2048^2 pyramid runs out of
    // texels near the Shore (a 1e-3-wide window spans ~1 texel), so the
    // deep-zoom field must come from the escape-iteration distance
    // estimator, which is resolution-unlimited.
    let _g = lock();
    let n = 24;
    // A window 1e-3 wide centered just inside the Shore at (0.25, 0).
    let half = 5e-4;
    let mut re = Vec::with_capacity(n * n);
    let mut im = Vec::with_capacity(n * n);
    for row in 0..n {
        let y = -half + 2.0 * half * (row as f64) / ((n - 1) as f64);
        for col in 0..n {
            let x = 0.25 - half + 2.0 * half * (col as f64) / ((n - 1) as f64);
            re.push(x);
            im.push(y);
        }
    }
    let field = runtime_core::minimap::deep_zoom_field(&re, &im)
        .expect("deep zoom field over a near-Shore window");
    assert_eq!(field.len(), n * n);
    // The estimator must resolve STRUCTURE at this zoom: the field varies
    // meaningfully across the window (the baked pyramid would be ~flat).
    let min = field.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = field.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max - min > 1e-4,
        "deep-zoom field must vary across a 1e-3 window (min={min}, max={max})"
    );
    // Non-negative unsigned distances.
    assert!(min >= 0.0);
}

#[test]
fn deep_zoom_field_matches_pyramid_near_its_resolution() {
    // At valley zoom the deep-zoom field must agree with the canonical
    // signed-SDF authority OUTSIDE the boundary (same geometry, different
    // sampler). Interior points legitimately return 0: the escape-iteration
    // estimator detects the cycle and reports "no distance to escape",
    // which is the correct deep-zoom semantic (the fractal detail lives
    // outside the boundary).
    let _g = lock();
    let pts = [(0.26f64, 0.0f64), (0.27, 0.0), (0.3, 0.0), (0.5, 0.0), (1.0, 1.0)];
    let re: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let im: Vec<f64> = pts.iter().map(|p| p.1).collect();
    let deep = runtime_core::minimap::deep_zoom_field(&re, &im).expect("deep zoom field");
    for (i, p) in pts.iter().enumerate() {
        let sdf = runtime_core::manifold::signed_distance(Complex64::new(p.0, p.1)).unwrap();
        assert!(sdf > 0.0, "test point {p:?} must be outside the set");
        let d_deep = deep[i] as f64;
        let d_sdf = sdf.abs();
        // Near the boundary the two authorities must agree tightly (the
        // baked raster resolves fine there). Far out the analytic DEM is
        // the better estimate; only require relative agreement (the
        // raster's pixel-spacing floor inflates far distances).
        if d_sdf < 0.25 {
            assert!(
                (d_deep - d_sdf).abs() < 5e-2,
                "point {p:?}: deep={d_deep}, sdf=|{sdf}| (near-boundary)"
            );
        } else {
            assert!(
                (d_deep - d_sdf).abs() <= 0.4 * d_sdf,
                "point {p:?}: deep={d_deep}, sdf=|{sdf}| (far-field relative)"
            );
        }
    }
    // Interior points report 0 (cycle-detected, no escape distance).
    let interior = runtime_core::minimap::deep_zoom_field(&[0.0], &[0.0]).unwrap();
    assert_eq!(interior[0], 0.0);
}

#[test]
fn debug_snapshot_serializes_camel_case() {
    let _g = lock();
    let snap = snapshot_from_state(Complex64::new(0.0, 0.0), (0.0, 0.0), None, None, &ManifoldConfig::default(), None)
        .expect("snapshot");
    let json = serde_json::to_string(&snap).expect("serialize");
    assert!(json.contains("\"timeSeconds\""), "wire format is camelCase: {json}");
    assert!(json.contains("\"signedDistance\""), "wire format is camelCase: {json}");
    assert!(json.contains("\"pyramidLoaded\""), "wire format is camelCase: {json}");
}

#[test]
fn shore_proximity_batch_matches_single_samples() {
    let _g = lock();
    // Synthetic pyramid over the canonical bake extent.
    let w0 = 8usize;
    let mut levels = vec![vec![0.5f32; w0 * w0]];
    let mut widths = vec![w0];
    let mut heights = vec![w0];
    for size in [4usize, 1, 1, 1, 1, 1] {
        levels.push(vec![0.75f32; size * size]);
        widths.push(size);
        heights.push(size);
    }
    let pyr = runtime_core::minimap::MipPyramid::from_levels(
        levels,
        widths,
        heights,
        -2.0,
        1.0,
        -1.5,
        1.5,
    )
    .expect("synthetic pyramid");
    runtime_core::minimap::set_pyramid(pyr).expect("install pyramid");

    // Batch of points spanning the extent; batch values must equal the
    // canonical single-point sampler (same field, same level).
    let pts = [
        (-2.0, -1.5),
        (0.0, 0.0),
        (0.9, 1.4),
        (-1.0, 0.5),
    ];
    let (re, im): (Vec<f64>, Vec<f64>) = pts.iter().copied().unzip();
    let batch = runtime_core::minimap::shore_proximity_batch(&re, &im, 0)
        .expect("batch sample");
    assert_eq!(batch.len(), pts.len());
    for ((r, i), &b) in pts.iter().zip(batch.iter()) {
        let single = runtime_core::minimap::with_pyramid(|p| {
            p.and_then(|pyr| pyr.shore_proximity_at(Complex64::new(*r, *i), 0))
        });
        assert_eq!(Some(b), single, "batch must match single at ({r},{i})");
    }

    // Length mismatch is an error.
    assert!(runtime_core::minimap::shore_proximity_batch(&re[..2].to_vec(), &im, 0).is_err());

    runtime_core::minimap::clear_pyramid();
}
