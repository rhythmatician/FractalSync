//! Tests for the contour-biased stepper (Physics reading the slope).

use runtime_core::minimap::{contour_biased_step, MipPyramid};

/// Pyramid with a linear shore-proximity field increasing with column
/// (toward +Re). Gradient points toward +Re; contours run along Im.
fn linear_pyramid() -> MipPyramid {
    let w = 64usize;
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
    runtime_core::minimap::clear_pyramid();
    // Huge proposed delta must be clamped to max_step.
    let (x, y) = contour_biased_step(0.0, 0.0, 10.0, 0.0, 0.0, 0.5, 0.01, 2, 0.0)
        .expect("step with no pyramid");
    let dx = x - 0.0;
    assert!((dx - 0.01).abs() < 1e-9, "expected clamped step, got {}", dx);
}

#[test]
fn tangential_motion_passes_through() {
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
    runtime_core::minimap::set_pyramid(linear_pyramid()).unwrap();
    // Motion purely along the gradient (+Re) is normal to the contour.
    // With h=0 (no transient), normal motion is suppressed to 2% (the wall).
    // d_star = current proximity so the servo is neutral.
    let (x, y) = contour_biased_step(-0.5, 0.0, 0.01, 0.0, 0.0, 32.0 / 63.0, 1.0, 0, 0.0)
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
