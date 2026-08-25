//! Differentiable-proxy reference implementations for training supervision.
//!
//! These functions compute audio→visual alignment proxies directly from the
//! synthesized Julia parameter trajectory `c(t)` in c-space, without rendering
//! images. They are the canonical (Rust-first) definitions; the PyTorch
//! trainer mirrors them with tensor ops so gradients can flow. A parity test
//! asserts the two implementations agree.
//!
//! Proxies:
//! - [`mandelbrot_cardioid_proximity`]: how close a point is to the main
//!   cardioid boundary (loudness ↔ boundary proximity supervision).
//! - [`orbit_path_metrics`]: mean/max step speed and spatial spread
//!   (coverage) of a c(t) trajectory (transient impact & diversity).

use num_complex::Complex64;

/// Distance proxy from a complex point to the main cardioid boundary.
///
/// The main cardioid is parameterized by the multiplier map
/// `c(μ) = μ/2 − μ²/4` with `|μ| = 1` on the boundary. Inverting the map
/// gives `μ = 1 ± sqrt(1 − 4c)`; we take the `μ = 1 − w` branch with
/// `w = sqrt(1 − 4c)` (principal square root), which maps interior points
/// near the cusp correctly for our operating region.
///
/// The returned value is `||μ| − 1|`: zero exactly on the boundary,
/// growing roughly linearly with distance from it. Fully smooth almost
/// everywhere (non-smooth only on the branch cut of the square root,
/// which lies outside the training region for typical orbits).
pub fn mandelbrot_cardioid_proximity(c: Complex64) -> f64 {
    // w = sqrt(1 - 4c)
    let inner = Complex64::new(1.0, 0.0) - c.scale(4.0);
    let w = inner.sqrt();
    // mu = 1 - w (branch chosen so mu ≈ e^{iθ} on the boundary)
    let mu = Complex64::new(1.0, 0.0) - w;
    (mu.norm() - 1.0).abs()
}

/// Batch form of [`mandelbrot_cardioid_proximity`].
pub fn mandelbrot_cardioid_proximity_batch(cs: &[Complex64]) -> Vec<f64> {
    cs.iter().map(|&c| mandelbrot_cardioid_proximity(c)).collect()
}

/// Aggregate geometric metrics over a c(t) trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitPathMetrics {
    /// Mean Euclidean step size between consecutive points.
    pub mean_speed: f64,
    /// Maximum Euclidean step size between consecutive points.
    pub max_speed: f64,
    /// Spatial spread: mean pairwise distance between points (coverage proxy).
    pub spread: f64,
}

impl OrbitPathMetrics {
    /// Metrics for an empty or single-point trajectory (all zeros).
    pub fn zeroed() -> Self {
        Self {
            mean_speed: 0.0,
            max_speed: 0.0,
            spread: 0.0,
        }
    }
}

/// Compute path metrics over a sequence of c-space points.
///
/// Trajectories with fewer than two points yield zeros. Pairwise spread is
/// O(n²); intended for short horizons (≤ a few hundred points).
pub fn orbit_path_metrics(points: &[Complex64]) -> OrbitPathMetrics {
    if points.len() < 2 {
        return OrbitPathMetrics::zeroed();
    }

    let mut speed_sum = 0.0;
    let mut max_speed = 0.0f64;
    for pair in points.windows(2) {
        let d = (pair[1] - pair[0]).norm();
        speed_sum += d;
        if d > max_speed {
            max_speed = d;
        }
    }
    let mean_speed = speed_sum / (points.len() - 1) as f64;

    let mut spread_sum = 0.0;
    let mut count = 0usize;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            spread_sum += (points[j] - points[i]).norm();
            count += 1;
        }
    }
    let spread = if count > 0 {
        spread_sum / count as f64
    } else {
        0.0
    };

    OrbitPathMetrics {
        mean_speed,
        max_speed,
        spread,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::lobe_point_at_angle;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn proximity_zero_on_cardioid_boundary() {
        // Points on the main cardioid boundary (s = 1) must have ~zero proximity.
        for k in 0..16 {
            let theta = 2.0 * std::f64::consts::PI * (k as f64) / 16.0;
            let c = lobe_point_at_angle(1, 0, theta, 1.0);
            let p = mandelbrot_cardioid_proximity(c);
            assert!(
                approx_eq(p, 0.0, 1e-9),
                "boundary point {c} gave proximity {p}"
            );
        }
    }

    #[test]
    fn proximity_grows_with_radial_scale() {
        let theta = 0.7;
        let inner = lobe_point_at_angle(1, 0, theta, 0.8);
        let outer = lobe_point_at_angle(1, 0, theta, 1.3);
        assert!(mandelbrot_cardioid_proximity(inner) > 1e-3);
        assert!(mandelbrot_cardioid_proximity(outer) > 1e-3);
        assert!(
            mandelbrot_cardioid_proximity(outer) > mandelbrot_cardioid_proximity(inner),
            "farther-from-boundary point should have larger proximity"
        );
    }

    #[test]
    fn path_metrics_basic() {
        let pts = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 1.0),
        ];
        let m = orbit_path_metrics(&pts);
        assert!(approx_eq(m.mean_speed, 1.0, 1e-12));
        assert!(approx_eq(m.max_speed, 1.0, 1e-12));
        // pairwise distances: 1, 1, sqrt(2)
        let expected_spread = (1.0 + 1.0 + std::f64::consts::SQRT_2) / 3.0;
        assert!(approx_eq(m.spread, expected_spread, 1e-12));
    }

    #[test]
    fn path_metrics_degenerate() {
        assert_eq!(orbit_path_metrics(&[]), OrbitPathMetrics::zeroed());
        assert_eq!(
            orbit_path_metrics(&[Complex64::new(1.0, 2.0)]),
            OrbitPathMetrics::zeroed()
        );
    }

    #[test]
    fn static_trajectory_has_zero_speed_nonzero_spread() {
        let p = Complex64::new(0.3, 0.2);
        let m = orbit_path_metrics(&[p, p, p]);
        assert!(approx_eq(m.mean_speed, 0.0, 1e-12));
        assert!(approx_eq(m.spread, 0.0, 1e-12));
    }
}
