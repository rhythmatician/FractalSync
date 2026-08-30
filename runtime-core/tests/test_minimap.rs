//! Tests for the minimap reader (mip pyramid of the Map).
//!
//! Domain vocabulary (issue #88):
//! - The Map: Mandelbrot set; The Shore: its boundary
//! - The minimaps: the Player's 9x9 windows at c, one per selected mip level
//! - The mip pyramid: pre-rendered multi-scale maps, base 2048^2
//! - Slope: gradient vector of the grey field at a point

use runtime_core::minimap::{
    MINIMAP_LEVELS, MINIMAP_WINDOW, MipPyramid, PLAYER_OBSERVATION_LEN,
};

/// Build a tiny synthetic pyramid: 2 levels, 8x8 and 4x4, covering the same
/// extent as the real bake. Grey value encodes position so tests can verify
/// window placement and orientation.
fn synthetic_pyramid() -> MipPyramid {
    let re_min = -2.0;
    let re_max = 1.0;
    let im_min = -1.5;
    let im_max = 1.5;

    // Level 0: 8x8, value = row + col (row-major, row 0 = im_max)
    let w0 = 8usize;
    let mut l0 = Vec::with_capacity(w0 * w0);
    for r in 0..w0 {
        for c in 0..w0 {
            l0.push((r * w0 + c) as f32);
        }
    }
    // Levels 1..6: shrinking constant planes (player_observation needs
    // levels {0,2,4,6} to exist).
    let mut levels = vec![l0];
    let mut widths = vec![w0];
    let mut heights = vec![w0];
    for size in [4usize, 2, 1, 1, 1, 1] {
        levels.push(vec![0.5f32; size * size]);
        widths.push(size);
        heights.push(size);
    }

    MipPyramid::from_levels(levels, widths, heights, re_min, re_max, im_min, im_max)
        .expect("synthetic pyramid should build")
}

#[test]
fn selected_levels_are_even() {
    assert_eq!(MINIMAP_LEVELS, [0usize, 2, 4, 6]);
}

#[test]
fn minimap_window_is_9x9() {
    assert_eq!(MINIMAP_WINDOW, 9usize);
}

#[test]
fn player_observation_is_332_inputs() {
    // 4 levels x 9x9 greys = 324, plus slope at c per level: 2 x 4 = 8
    assert_eq!(PLAYER_OBSERVATION_LEN, 332);
}

#[test]
fn pyramid_reports_level_dimensions() {
    let pyr = synthetic_pyramid();
    assert_eq!(pyr.level_size(0), Some((8, 8)));
    assert_eq!(pyr.level_size(1), Some((4, 4)));
    assert_eq!(pyr.level_size(6), Some((1, 1)));
    assert!(pyr.level_size(7).is_none());
}

#[test]
fn minimap_window_is_centered_on_c() {
    let pyr = synthetic_pyramid();
    // c at exact center of the level-0 grid: col 3.5 -> sample around center.
    // Use a point that maps to texel (4, 4): re spans -2..1 over 8 cols,
    // so col i center is at re = -2 + (i+0.5)*3/8. For i=4: -2 + 4.5*0.375 = -0.3125
    // im spans -1.5..1.5 over 8 rows with row 0 = im_max:
    // row j center is at im = 1.5 - (j+0.5)*3/8. For j=4: 1.5 - 4.5*0.375 = -0.1875
    let c = num_complex::Complex64::new(-0.3125, -0.1875);
    let win = pyr.minimap(c, 0, 4).expect("window within bounds");
    assert_eq!(win.len(), 81);
    // Center pixel (index 40 in the flattened 9x9) should be the value at
    // texel (4,4) = 4*8+4 = 36.
    assert!((win[40] - 36.0).abs() < 1e-4);
}

#[test]
fn minimap_rows_run_from_high_im_to_low_im() {
    let pyr = synthetic_pyramid();
    // Value = row*8 + col, and row index grows as Im decreases. So the top
    // row of the window (higher Im, smaller row index) must have SMALLER
    // values than the bottom row.
    let c = num_complex::Complex64::new(-0.3125, -0.1875);
    let win = pyr.minimap(c, 0, 4).expect("window within bounds");
    let top_mean: f32 = win[0..9].iter().sum::<f32>() / 9.0;
    let bottom_mean: f32 = win[72..81].iter().sum::<f32>() / 9.0;
    assert!(
        top_mean < bottom_mean,
        "top row ({}) should be below bottom row ({})",
        top_mean,
        bottom_mean
    );
}

#[test]
fn minimap_window_clamps_at_extent_edges() {
    let pyr = synthetic_pyramid();
    // A corner of the plane: window must clamp rather than fail.
    let c = num_complex::Complex64::new(re_edge(&pyr, true), im_edge(&pyr, true));
    let win = pyr.minimap(c, 0, 4).expect("clamped window");
    assert_eq!(win.len(), 81);
    assert!(win.iter().all(|v| v.is_finite()));
}

fn re_edge(pyr: &MipPyramid, min: bool) -> f64 {
    if min {
        pyr.re_min
    } else {
        pyr.re_max
    }
}

fn im_edge(pyr: &MipPyramid, max: bool) -> f64 {
    if max {
        pyr.im_max
    } else {
        pyr.im_min
    }
}

#[test]
fn slope_is_zero_on_constant_field() {
    let pyr = synthetic_pyramid();
    // Level 1 is constant 0.5 -> gradient must be ~zero everywhere on it.
    let c = num_complex::Complex64::new(-0.5, 0.25);
    let (gx, gy) = pyr.slope(c, 1).expect("slope available");
    assert!(gx.abs() < 1e-6 && gy.abs() < 1e-6);
}

#[test]
fn slope_points_uphill_on_linear_field() {
    let pyr = synthetic_pyramid();
    // Level 0 increases with column index (re direction), so dF/dRe > 0.
    // Field step per texel is 1 unit; texel spacing is 3/8 world units,
    // so expected gx ~= 1 / (3/8) = 8/3.
    let c = num_complex::Complex64::new(-0.3125, -0.1875);
    let (gx, _gy) = pyr.slope(c, 0).expect("slope available");
    let expected = 8.0f64 / 3.0;
    assert!(
        (gx - expected).abs() < 1e-3,
        "gx {} vs expected {}",
        gx,
        expected
    );
}

#[test]
fn observation_has_exact_shape_and_raw_values() {
    let pyr = synthetic_pyramid();
    let c = num_complex::Complex64::new(-0.3125, -0.1875);
    let obs = pyr.player_observation(c).expect("observation built");
    assert_eq!(obs.len(), PLAYER_OBSERVATION_LEN);
    // First 324 entries are greys from level 0 first: they must be finite.
    assert!(obs[..324].iter().all(|v| v.is_finite()));
    // Last 8 are slopes per level (gx, gy) x 4 levels.
    assert!(obs[324..].iter().all(|v| v.is_finite()));
}
