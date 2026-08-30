//! Tests for the contour-biased stepper (Physics reading the slope).

use std::sync::{Mutex, MutexGuard, OnceLock};

use runtime_core::minimap::{contour_biased_step, MipPyramid};

/// The pyramid is process-global state (`minimap::set_pyramid`), so tests
/// that install/clear it must not run concurrently — a parallel test's
/// `clear_pyramid()` makes another test's step take the no-map fallback
/// path and produce zero motion, failing its assertions intermittently.
/// Every test in this file takes this lock for its entire body.
fn pyramid_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Pyramid with a linear shore-proximity field increasing with column
/// (toward +Re). Gradient points toward +Re; contours run along Im.
fn linear_pyramid() -> MipPyramid {
    let mut levels = Vec::new();
    let mut widths = Vec::new();
    let mut heights = Vec::new();
    for &size in &[64usize, 32, 16, 8, 4, 2, 1] {
        // value = normalized column index in [0, 1]
        let plane: Vec<f32> = (0..size * size)
            .map(|i| {
                let col = i % size;
                col as f32 / (size - 1).max(1) as f32
            })
            .collect();
        levels.push(plane);
        widths.push(size);
        heights.push(size);
    }
    MipPyramid::from_levels(levels, widths, heights, -2.0, 1.0, -1.5, 1.5)
        .expect("linear pyramid")
}

#[test]
fn no_pyramid_falls_back_to_clamped_motion() {
    let _guard = pyramid_lock();
    runtime_core::minimap::clear_pyramid();
    // Huge proposed delta must be clamped to max_step.
    let (x, _y) = contour_biased_step(0.0, 0.0, 10.0, 0.0, 0.0, 0.5, 0.01, 2, 0.0)
        .expect("step with no pyramid");
    let dx = x - 0.0;
    assert!((dx - 0.01).abs() < 1e-9, "expected clamped step, got {}", dx);
}

#[test]
fn tangential_motion_passes_through() {
    let _guard = pyramid_lock();
    runtime_core::minimap::set_pyramid(linear_pyramid()).unwrap();
    // Contours run along Im (gradient is along Re). Moving purely along Im
    // is tangential: it should pass through nearly unmodified.
    // d_star = current proximity (~0.1667 at re=-0.5) so the servo is neutral.
    let (x, y) = contour_biased_step(-0.5, 0.0, 0.0, 0.01, 0.0, 32.0 / 63.0, 1.0, 0, 0.0)
        .expect("tangential step");
    assert!((x - (-0.5)).abs() < 1e-3, "re should not change much");
    assert!((y - 0.01).abs() < 1e-3, "im should pass through, got {}", y);
    runtime_core::minimap::clear_pyramid();
}

#[test]
fn normal_motion_suppressed_between_hits() {
    let _guard = pyramid_lock();
    runtime_core::minimap::set_pyramid(linear_pyramid()).unwrap();
    // Motion purely along the gradient (+Re) is normal to the contour.
    // With h=0 (no transient), normal motion is suppressed to 2% (the wall).
    // d_star = current proximity so the servo is neutral.
    let (x, _y) = contour_biased_step(-0.5, 0.0, 0.01, 0.0, 0.0, 32.0 / 63.0, 1.0, 0, 0.0)
        .expect("normal step");
    let dx = x - (-0.5);
    assert!(
        dx < 0.01 * 0.2,
        "normal motion should be strongly suppressed between hits, got {}",
        dx
    );
    runtime_core::minimap::clear_pyramid();
}

#[test]
fn normal_motion_allowed_during_hits() {
    let _guard = pyramid_lock();
    runtime_core::minimap::set_pyramid(linear_pyramid()).unwrap();
    // With h=1 (full transient), the wall opens: normal motion passes through.
    // d_star = current proximity so the servo is neutral.
    let (x, _y) = contour_biased_step(-0.5, 0.0, 0.01, 0.0, 1.0, 32.0 / 63.0, 1.0, 0, 0.0)
        .expect("hit step");
    let dx = x - (-0.5);
    assert!(
        (dx - 0.01).abs() < 1e-3,
        "normal motion should pass during hits, got {}",
        dx
    );
    runtime_core::minimap::clear_pyramid();
}

#[test]
fn music_push_moves_uphill() {
    let _guard = pyramid_lock();
    runtime_core::minimap::set_pyramid(linear_pyramid()).unwrap();
    // Zero proposed motion, zero energy: no push, c does not move.
    let (x_quiet, _) = contour_biased_step(-0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0, 0.0)
        .expect("quiet step");
    // Full energy: the music push drives c UP the slope (toward higher
    // proximity = +Re in this fixture).
    let (x_loud, _) = contour_biased_step(-0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0, 1.0)
        .expect("loud step");
    assert!(
        (x_quiet - (-0.5)).abs() < 1e-9,
        "no energy = no push, got {}",
        x_quiet
    );
    assert!(
        x_loud > -0.5,
        "energy must push c uphill (toward the Shore), got {}",
        x_loud
    );
    runtime_core::minimap::clear_pyramid();
}

#[test]
fn max_step_clamps_total_motion() {
    let _guard = pyramid_lock();
    runtime_core::minimap::set_pyramid(linear_pyramid()).unwrap();
    let (x, y) = contour_biased_step(0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.02, 0, 0.0)
        .expect("clamped step");
    let dist = (x * x + y * y).sqrt();
    assert!(
        dist <= 0.02 + 1e-9,
        "total step {} exceeds max_step 0.02",
        dist
    );
    runtime_core::minimap::clear_pyramid();
}

/// Flat-S pyramid: every level plane is all zeros. This forces the analytic
/// cardioid fallback because the mip S gradient is exactly zero everywhere.
fn flat_pyramid() -> MipPyramid {
    let mut levels = Vec::new();
    let mut widths = Vec::new();
    let mut heights = Vec::new();
    for &size in &[64usize, 32, 16, 8, 4, 2, 1] {
        levels.push(vec![0.0f32; size * size]);
        widths.push(size);
        heights.push(size);
    }
    MipPyramid::from_levels(levels, widths, heights, -2.0, 1.0, -1.5, 1.5)
        .expect("flat pyramid")
}

/// Cardioid proximity p(c) = ||mu|-1| with mu = 1 - sqrt(1-4c). This is the
/// same closed form the cardioid_fallback_step uses internally.
fn cardioid_proximity(c_re: f64, c_im: f64) -> f64 {
    use num_complex::Complex64;
    let cc = Complex64::new(c_re, c_im);
    let inner = (1.0 - 4.0 * cc).sqrt();
    let mu = Complex64::new(1.0, 0.0) - inner;
    (mu.norm() - 1.0).abs()
}

#[test]
fn cardioid_fallback_sign_regression() {
    let _guard = pyramid_lock();
    // Regression test for the music-push sign bug in cardioid_fallback_step.
    //
    // p(c) = ||mu|-1| DECREASES toward the cardioid boundary, so the
    // shoreward direction is -grad p. Energy > 0 must therefore move c
    // IN THE DIRECTION of -grad p — descending p, even if a large push
    // magnitude overshoots past the boundary. An earlier commit had the
    // sign inverted (used +grad p), which shoved c in the direction of
    // steepest p ASCENT — i.e., away from the Shore, into the interior
    // valley precisely when the music went loud. That was the exact
    // opposite of the intended physics.
    //
    // The test uses a *directional* assertion: the displacement produced
    // by the energy push must have NEGATIVE projection on grad p. The
    // broken (pre-fix) sign would have had POSITIVE projection — moving
    // c in the direction of p increase, which is what made the model
    // park in the valley. A magnitude-overshoot does not break this
    // directional test; the bug was about the sign of the push, not
    // its size.
    runtime_core::minimap::set_pyramid(flat_pyramid()).unwrap();

    // Starting c: well inside the cardioid on the real axis, where the
    // cardioid proximity gradient is purely along Re and easily verified.
    let c0 = (-0.4_f64, 0.0_f64);

    // Compute grad p at c0 with the same finite-difference eps the
    // production code uses. The reference gradient must point toward
    // INCREASING p (i.e., away from the boundary).
    let eps = 1e-4_f64;
    let gx_ref = (cardioid_proximity(c0.0 + eps, c0.1)
        - cardioid_proximity(c0.0 - eps, c0.1)) / (2.0 * eps);
    let gy_ref = (cardioid_proximity(c0.0, c0.1 + eps)
        - cardioid_proximity(c0.0, c0.1 - eps)) / (2.0 * eps);
    assert!(
        gx_ref.abs() > 1e-6,
        "test precondition: grad p should be measurably non-zero at c0, got ({}, {})",
        gx_ref, gy_ref
    );

    // Loud frame, no proposed motion: the energy push must displace c
    // opposite to grad p (toward the shore).
    let (x_loud, y_loud) = contour_biased_step(
        c0.0, c0.1, 0.0, 0.0, 0.0, 0.5, 1.0, 0, 1.0
    ).expect("loud fallback step");
    let dx = x_loud - c0.0;
    let dy = y_loud - c0.1;
    // The push direction must have negative projection on grad p
    // (i.e., the push moves c in the direction of decreasing p).
    let projection = dx * gx_ref + dy * gy_ref;
    assert!(
        projection < -1e-6,
        "energy push must descend p (negative projection on grad p); \
         got c0=({}, {}) -> ({}, {}), dx,dy=({}, {}), grad p=({}, {}), \
         projection={:.6e}. A positive projection means the push is moving c \
         TOWARD increasing p (away from the Shore) — the sign-bug behaviour.",
        c0.0, c0.1, x_loud, y_loud, dx, dy, gx_ref, gy_ref, projection,
    );

    // Quiet frame, same starting c: no push, c unchanged.
    let (x_quiet, y_quiet) = contour_biased_step(
        c0.0, c0.1, 0.0, 0.0, 0.0, 0.5, 1.0, 0, 0.0
    ).expect("quiet fallback step");
    assert!(
        (x_quiet - c0.0).abs() < 1e-12 && (y_quiet - c0.1).abs() < 1e-12,
        "energy == 0 must not move c; got ({}, {}) from ({}, {})",
        x_quiet, y_quiet, c0.0, c0.1,
    );
    runtime_core::minimap::clear_pyramid();
}
