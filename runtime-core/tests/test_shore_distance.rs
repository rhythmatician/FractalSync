use runtime_core::geometry_provider::{
    GeometryJet, GeometryValidity, SingularityKind, GEOMETRY_PROVIDER_VERSION,
};
use runtime_core::manifold::ManifoldConfig;
use runtime_core::shore_distance::{RadialProfile, ShoreDistanceError, ShoreDistanceEstimate};

fn simpson(end: f64, f: impl Fn(f64) -> f64) -> f64 {
    let n = 4096;
    let step = end / n as f64;
    let mut sum = f(0.0) + f(end);
    for i in 1..n {
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * f(i as f64 * step);
    }
    sum * step / 3.0
}

fn assert_near(actual: f64, expected: f64, relative: f64) {
    let tolerance = (relative * expected.abs()).max(4.0 * f64::from_bits(1));
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.17e} vs {expected:.17e}, tolerance {tolerance:.3e}"
    );
}

#[test]
fn scalar_matches_independent_quadrature_and_analytic_bounds() {
    let epsilon = 0.125;
    for a in [0.0_f64, 0.2, 1.0 - 1e-10, 1.0, 1.0 + 1e-10, 2.0, 100.0] {
        let profile = RadialProfile::new(epsilon, a).unwrap();
        assert_eq!(profile.evaluate(0.0).unwrap(), 0.0);
        let mut previous = 0.0;
        let mut previous_q = 0.0;
        for q in [1e-14_f64, 1e-8, 0.001, 0.1, 1.0, 2.0, 20.0, 1e6] {
            let value = profile.evaluate(epsilon * q).unwrap();
            let lower = q.asinh();
            let upper = a.hypot(1.0) * lower;
            // Substitution t = sinh(x) removes the near-crest length scale
            // from the integral and gives a smooth quadrature integrand.
            let integral = simpson(lower, |x| (a * x.tanh()).hypot(1.0));
            assert_near(value, integral, 2e-9);
            assert!(value >= lower * (1.0 - 1e-15));
            assert!(value <= upper * (1.0 + 1e-15));
            assert!(value > previous);
            let bound = profile
                .propagate_radius_error(epsilon * (q - previous_q))
                .unwrap();
            assert!(value - previous <= bound * (1.0 + 1e-14));
            assert_eq!(
                value,
                RadialProfile::new(1.0, a).unwrap().evaluate(q).unwrap()
            );
            if a == 0.0 {
                assert_near(value, lower, 2e-15);
            }
            previous = value;
            previous_q = q;
        }
    }

    // Sharpness for a > 1 is attained at q = sqrt((a² - 1)/(a² + 1)).
    let profile = RadialProfile::new(epsilon, 2.0).unwrap();
    let peak = (3.0_f64 / 5.0).sqrt() * epsilon;
    let delta = 1e-4 * epsilon;
    let increment = profile.evaluate(peak + delta / 2.0).unwrap()
        - profile.evaluate(peak - delta / 2.0).unwrap();
    assert_near(
        profile.propagate_radius_error(delta).unwrap(),
        increment,
        2e-9,
    );
}

#[test]
fn binary64_extremes_match_high_precision_reference() {
    let tiny = f64::from_bits(1);
    // Rounded from the subtractive primitive at 780 decimal digits using
    // formal/checks/check_stable_radial_profile.py::reference.
    for (radius, epsilon, a, expected) in [
        (1e-300, 1.0, 1e300, 1.147_793_574_696_319_1e-300),
        (1e300, 1e-100, 1.0, 1_303.127_840_440_300_8),
        (2.0, 1.0, f64::MAX, 1.446_637_743_084_959_6e308),
        (1e-308, 1.0, f64::MAX, 1.403_846_933_750_871_5e-308),
        (1.01, 1.0, f64::MAX, 6.320_212_809_565_063e307),
        (f64::MAX, 1.0, 0.0, 710.475_860_073_944),
        (tiny, 1.0, f64::MAX, tiny),
        (1e300, tiny, 0.0, 1_435.908_747_000_155),
        (1e-308, 8.0, 1e308, 1.253_247_621_193_04e-309),
    ] {
        let value = RadialProfile::new(epsilon, a)
            .unwrap()
            .evaluate(radius)
            .unwrap();
        assert_near(value, expected, 5e-14);
    }
    assert_eq!(
        RadialProfile::new(1.0, f64::MAX).unwrap().evaluate(tiny),
        Ok(tiny)
    );
    assert_eq!(
        RadialProfile::new(f64::MAX, 1.0).unwrap().evaluate(tiny),
        Ok(0.0)
    );
    assert_eq!(
        RadialProfile::new(1.0, f64::MAX).unwrap().evaluate(4.0),
        Err(ShoreDistanceError::Overflow)
    );
    let profile = RadialProfile::new(1e300, 1e300).unwrap();
    assert_near(
        profile.propagate_radius_error(1e-300).unwrap(),
        5e-301,
        5e-13,
    );
}

#[test]
fn closed_shore_path_lengths_match_the_profile_and_bound_detours() {
    let epsilon = 0.23;
    let a = 1.7;
    let profile = RadialProfile::new(epsilon, a).unwrap();
    let speed = |d: f64, distance_rate: f64, planar_speed: f64| {
        let rho = d.hypot(epsilon);
        (planar_speed / rho).hypot(a * d * distance_rate / rho.powi(2))
    };

    // Plane x=0, with the start on either side of the Shore. Slanting
    // toward a different endpoint costs more than the nearest segment.
    for d0 in [-0.6_f64, 0.6] {
        let expected = profile.evaluate(d0.abs()).unwrap();
        let nearest = simpson(1.0, |t| speed(d0 * (1.0 - t), -d0, d0.abs()));
        let slanted = simpson(1.0, |t| speed(d0 * (1.0 - t), -d0, d0.hypot(1.0)));
        assert_near(nearest, expected, 2e-13);
        assert!(slanted > expected);
    }

    // The unit circle is a curved nonconvex closed set. Every endpoint on
    // the circle is a nearest point to the center. The radial distance is
    // well defined there despite the nonunique normal.
    for start in [0.0_f64, 0.5, 1.5] {
        let dr = 1.0 - start;
        let expected = profile.evaluate(dr.abs()).unwrap();
        let nearest = simpson(1.0, |t| speed(start + t * dr - 1.0, dr, dr.abs()));
        let spiral = simpson(1.0, |t| {
            let r = start + t * dr;
            speed(r - 1.0, dr, dr.hypot(r * std::f64::consts::PI))
        });
        assert_near(nearest, expected, 2e-13);
        assert!(spiral > expected);
    }
}

#[test]
fn provider_estimate_preserves_signed_input_and_rejects_unknown_correspondence() {
    let mut jet = GeometryJet {
        d: -0.3,
        grad_d: [1.0, 0.0],
        hessian_d: [[0.0; 2]; 2],
        requested_scale: 0.000_1,
        resolved_scale: 0.004,
        validity: GeometryValidity::Unresolved,
        singularity: SingularityKind::None,
        provider_version: GEOMETRY_PROVIDER_VERSION.to_owned(),
        tile_id: "test".to_owned(),
        is_bridge: true,
    };
    let config = ManifoldConfig {
        epsilon: 0.02,
        lambda_sq: 4.0,
        ..ManifoldConfig::default()
    };
    let estimate = ShoreDistanceEstimate::from_jet(&jet, &config).unwrap();
    assert_eq!(jet.d, -0.3);
    assert_eq!(estimate.epsilon, config.epsilon);
    assert_eq!(estimate.anisotropy, 2.0 / std::f64::consts::LN_2);
    jet.d = 0.3;
    jet.validity = GeometryValidity::Singular;
    assert_eq!(ShoreDistanceEstimate::from_jet(&jet, &config), Ok(estimate));
    for validity in [
        GeometryValidity::OutsideDomain,
        GeometryValidity::ProviderFailure,
    ] {
        jet.validity = validity;
        assert_eq!(
            ShoreDistanceEstimate::from_jet(&jet, &config),
            Err(ShoreDistanceError::UnavailableProviderDistance)
        );
    }
    jet.validity = GeometryValidity::Regular;
    jet.d = f64::NAN;
    assert_eq!(
        ShoreDistanceEstimate::from_jet(&jet, &config),
        Err(ShoreDistanceError::UnavailableProviderDistance)
    );
    jet.d = 0.3;
    jet.is_bridge = false;
    jet.provider_version = "experimental-operational-field/1;reference-distance-unknown".to_owned();
    assert_eq!(
        ShoreDistanceEstimate::from_jet(&jet, &config),
        Err(ShoreDistanceError::UnavailableProviderDistance)
    );
}

#[test]
fn invalid_scalar_inputs_are_reported_separately() {
    for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            RadialProfile::new(epsilon, 0.0).unwrap_err(),
            ShoreDistanceError::InvalidEpsilon
        );
    }
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            RadialProfile::new(1.0, invalid).unwrap_err(),
            ShoreDistanceError::InvalidAnisotropy
        );
        let profile = RadialProfile::new(1.0, 1.0).unwrap();
        assert_eq!(
            profile.evaluate(invalid),
            Err(ShoreDistanceError::InvalidRadius)
        );
        assert_eq!(
            profile.propagate_radius_error(invalid),
            Err(ShoreDistanceError::InvalidRadiusError)
        );
    }
}
