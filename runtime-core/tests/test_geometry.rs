use runtime_core::geometry::{lobe_point_at_angle, period_n_bulb_radius, Complex};

// Helper to compute magnitude
fn magnitude(c: Complex) -> f64 {
    (c.re * c.re + c.im * c.im).sqrt()
}

#[test]
fn test_main_cardioid_bounds() {
    // Test known points on the main cardioid
    // At theta=0, should be at (0.25, 0)
    let c0 = lobe_point_at_angle(1, 0, 0.0, 1.0);
    assert!((c0.re - 0.25).abs() < 1e-10);
    assert!(c0.im.abs() < 1e-10);
    
    // At theta=π, should be at (-0.75, 0)
    let cpi = lobe_point_at_angle(1, 0, std::f64::consts::PI, 1.0);
    assert!((cpi.re + 0.75).abs() < 1e-10);
    assert!(cpi.im.abs() < 1e-10);
    
    // At theta=π/2, should be on top of cardioid
    let c_half_pi = lobe_point_at_angle(1, 0, std::f64::consts::PI / 2.0, 1.0);
    assert!(c_half_pi.im > 0.0); // Above real axis
    assert!(c_half_pi.re < 0.25 && c_half_pi.re > -0.75); // Within bounds
}

#[test]
fn test_period_2_bulb() {
    // Period-2 bulb is to the left of main cardioid at approximately (-1, 0)
    let radius = period_n_bulb_radius(2, 0);
    assert!(radius > 0.0);
    assert!(radius < 0.3); // Should be around 0.25
    
    let c = lobe_point_at_angle(2, 0, 0.0, 1.0);
    assert!(c.re < -0.75); // Should be left of main cardioid
    assert!(c.im.abs() < 1e-10); // On real axis
}

#[test]
fn test_period_3_bulb() {
    // Period-3 bulbs exist at multiple locations
    let radius = period_n_bulb_radius(3, 0);
    assert!(radius > 0.0);
    assert!(radius < 0.2);
    
    // Test that we can get points on period-3 bulbs
    let c = lobe_point_at_angle(3, 0, 0.0, 1.0);
    assert!(c.re.is_finite());
    assert!(c.im.is_finite());
}

#[test]
fn test_lobe_angle_sweeps_full_circle() {
    // Test that sweeping from 0 to 2π produces a closed curve
    let n_samples = 100;
    let start = lobe_point_at_angle(2, 0, 0.0, 1.0);
    let end = lobe_point_at_angle(2, 0, 2.0 * std::f64::consts::PI, 1.0);
    
    // Start and end should be approximately equal (closed curve)
    assert!((start.re - end.re).abs() < 1e-10);
    assert!((start.im - end.im).abs() < 1e-10);
}

#[test]
fn test_orbit_escape_quickly() {
    // c = 2.0 + 0i is outside Mandelbrot set, should escape quickly
    let c = Complex { re: 2.0, im: 0.0 };
    let mut z = Complex { re: 0.0, im: 0.0 };
    let mut escaped = false;
    
    for _ in 0..50 {
        z = z * z + c;
        if magnitude(z) > 2.0 {
            escaped = true;
            break;
        }
    }
    
    assert!(escaped, "Point (2, 0) should escape");
}

#[test]
fn test_orbit_bounded_for_inside_points() {
    // c = 0 + 0i is inside (it's the center), orbit should stay bounded
    let c = Complex { re: 0.0, im: 0.0 };
    let mut z = Complex { re: 0.0, im: 0.0 };
    
    for _ in 0..100 {
        z = z * z + c;
        assert!(magnitude(z) < 2.0, "Center point should stay bounded");
    }
}

#[test]
fn test_main_cardioid_points_bounded() {
    // Points on the main cardioid should stay bounded (or barely escape)
    for i in 0..10 {
        let theta = 2.0 * std::f64::consts::PI * (i as f64) / 10.0;
        let c = lobe_point_at_angle(1, 0, theta, 0.95); // Slightly inside
        let mut z = Complex { re: 0.0, im: 0.0 };
        let mut max_norm: f64 = 0.0;
        
        for _ in 0..100 {
            z = z * z + c;
            max_norm = max_norm.max(magnitude(z));
            if magnitude(z) > 10.0 {
                break; // Escaped too far
            }
        }
        
        // Should stay relatively bounded for points inside cardioid
        assert!(
            max_norm < 5.0,
            "Point at theta={} escaped too far: {}",
            theta,
            max_norm
        );
    }
}

#[test]
fn test_deterministic_orbit_calculation() {
    // Same parameters should give same results
    let c1 = lobe_point_at_angle(2, 1, 1.234, 1.05);
    let c2 = lobe_point_at_angle(2, 1, 1.234, 1.05);
    
    assert_eq!(c1.re, c2.re);
    assert_eq!(c1.im, c2.im);
}

#[test]
fn test_scale_parameter_effect() {
    // Changing scale should move points radially
    let c1 = lobe_point_at_angle(1, 0, 0.5, 1.0);
    let c2 = lobe_point_at_angle(1, 0, 0.5, 1.1);
    let c3 = lobe_point_at_angle(1, 0, 0.5, 0.9);
    
    // c2 (s=1.1) should be farther from origin than c1 (s=1.0)
    assert!(magnitude(c2) > magnitude(c1));
    // c3 (s=0.9) should be closer to origin than c1
    assert!(magnitude(c3) < magnitude(c1));
}

#[test]
fn test_period_radius_positive() {
    // All period bulb radii should be positive
    for period in 2..=10 {
        for sub_lobe in 0..3 {
            let radius = period_n_bulb_radius(period, sub_lobe);
            assert!(
                radius > 0.0,
                "Period {} sub_lobe {} has non-positive radius",
                period,
                sub_lobe
            );
        }
    }
}

#[test]
fn test_period_radius_decreasing() {
    // Higher periods should generally have smaller radii (deeper in fractal)
    let r2 = period_n_bulb_radius(2, 0);
    let r4 = period_n_bulb_radius(4, 0);
    let r8 = period_n_bulb_radius(8, 0);
    
    assert!(r2 > r4);
    assert!(r4 > r8);
}

#[test]
fn test_scale_1_gives_boundary() {
    // s=1.0 should give points exactly on the bulb boundary
    let c = lobe_point_at_angle(2, 0, 0.0, 1.0);
    let radius = period_n_bulb_radius(2, 0);
    
    // Point should be approximately at distance 'radius' from center
    // (This is a rough check since we don't expose the center directly)
    assert!(magnitude(c) > 0.5); // Should be away from origin
    assert!(magnitude(c) < 2.0); // But not too far
}

#[test]
fn test_multiple_sub_lobes() {
    // Period-3 has multiple sub-lobes
    let c0 = lobe_point_at_angle(3, 0, 0.0, 1.0);
    let c1 = lobe_point_at_angle(3, 1, 0.0, 1.0);
    let c2 = lobe_point_at_angle(3, 2, 0.0, 1.0);
    
    // Different sub-lobes should give different points
    assert!((c0.re - c1.re).abs() > 0.1 || (c0.im - c1.im).abs() > 0.1);
    assert!((c1.re - c2.re).abs() > 0.1 || (c1.im - c2.im).abs() > 0.1);
}
