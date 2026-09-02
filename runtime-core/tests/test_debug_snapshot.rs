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

use num_complex::Complex64 as C;
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
    let c = C::new(0.0, 0.0);
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
    let snap = snapshot_from_state(C::new(0.1, -0.1), (0.1, 0.0), None, None, &ManifoldConfig::default(), None)
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
    let snap = snapshot_from_state(C::new(0.0, 0.0), (0.0, 0.0), None, None, &ManifoldConfig::default(), None)
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

    let snap = snapshot_from_state(C::new(0.0, 0.0), (0.0, 0.0), None, None, &ManifoldConfig::default(), None)
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
    let center = C::new(0.0, 0.0);
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
    let north = C::new(-0.5, 0.5);
    let sigma_north = runtime_core::manifold::mandelbrot_scale(north, &config).unwrap();
    assert!((first_sigma - lambda * sigma_north).abs() < 1e-14);
}

#[test]
fn terrain_patch_rejects_degenerate_grids() {
    let _g = lock();
    assert!(terrain_patch(0.0, 0.0, 0.5, 1, &ManifoldConfig::default()).is_err());
    assert!(terrain_patch(0.0, 0.0, 0.5, 1000, &ManifoldConfig::default()).is_err());
    assert!(terrain_patch(0.0, 0.0, -1.0, 33, &ManifoldConfig::default()).is_err());
}

#[test]
fn debug_snapshot_serializes_camel_case() {
    let _g = lock();
    let snap = snapshot_from_state(C::new(0.0, 0.0), (0.0, 0.0), None, None, &ManifoldConfig::default(), None)
        .expect("snapshot");
    let json = serde_json::to_string(&snap).expect("serialize");
    assert!(json.contains("\"timeSeconds\""), "wire format is camelCase: {json}");
    assert!(json.contains("\"signedDistance\""), "wire format is camelCase: {json}");
    assert!(json.contains("\"pyramidLoaded\""), "wire format is camelCase: {json}");
}
