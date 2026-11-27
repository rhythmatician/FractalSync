use super::*;
use crate::crossing_feasibility::{
    evaluate_feasibility, FeasibilityLimits, FeasibilityStatus, LengthLowerBound,
};
use crate::distance_field::{
    clear_distance_field, global_test_mutex, load_builtin_distance_field,
    sample_signed_distance_field, set_distance_field_from_vec,
};
use crate::geometry_provider::{query_geometry, rho_from_jet, with_test_plane};
use crate::manifold::induced_metric;
use crate::timebase::CANONICAL_SAMPLE_RATE;
use num_complex::Complex64;

#[test]
fn default_raster_point_ball_and_exit_action() {
    let _guard = global_test_mutex()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let bounds = [[-1.5, -0.45], [-0.4, 0.45]];
    clear_distance_field();
    assert!(RasterNavigationRegion::for_default_profile(bounds).is_none());
    load_builtin_distance_field("mandelbrot_default").unwrap();
    let region = RasterNavigationRegion::for_default_profile(bounds).unwrap();
    let config = ManifoldConfig::default();
    let start = [-1.0, 0.0];
    let center = [-0.7, 0.1];
    let point = region.target_lower_bound(start, center, 0.0).unwrap();
    let ball = region.target_lower_bound(start, center, 0.025).unwrap();
    let exit = region.target_lower_bound(start, [1.4, 0.0], 0.0).unwrap();
    assert!(0.0 < ball && ball < point && point < exit);
    let exit_distance = 0.45;
    let outside_at_boundary = region.target_lower_bound(start, [-1.0, 0.45], 0.0);
    assert_eq!(outside_at_boundary, Some(exit));
    let metric_ruler = mul_up(std::f64::consts::SQRT_2.next_up(), region.rho_upper());
    assert!(exit <= exit_distance / metric_ruler);
    assert_eq!(region.target_lower_bound(start, center, 0.4), Some(0.0));
    assert_eq!(
        region.target_lower_bound([-1.5, 0.0], center, 0.0),
        Some(0.0)
    );
    assert_eq!(region.target_lower_bound([-1.51, 0.0], center, 0.0), None);
    assert_eq!(region.target_lower_bound(start, center, -1.0), None);

    for [x, y] in [start, center, bounds[0], bounds[1]] {
        let c = Complex64::new(x, y);
        let jet = query_geometry(c, config.epsilon).unwrap();
        assert!(jet.is_bridge);
        let rho = rho_from_jet(&jet, config.epsilon);
        assert!(rho <= region.rho_upper());
        let metric = induced_metric(c, &config).unwrap();
        for [vx, vy] in [[1.0, 0.0], [0.0, 1.0], [0.6, -0.8]] {
            let kinetic_twice = vx * (metric[0][0] * vx + metric[0][1] * vy)
                + vy * (metric[1][0] * vx + metric[1][1] * vy);
            assert!(kinetic_twice > 0.5 / (rho * rho));
        }
    }

    let limits = FeasibilityLimits {
        kinetic_action_budget: Some(0.02),
        ..FeasibilityLimits::default()
    };
    let one_second = CANONICAL_SAMPLE_RATE as u64;
    let fast = evaluate_feasibility(LengthLowerBound::Certified(ball), one_second, limits);
    let slow = evaluate_feasibility(LengthLowerBound::Certified(ball), 100 * one_second, limits);
    assert_eq!(fast.status, FeasibilityStatus::RuledOut);
    assert_eq!(
        fast.deadline_status,
        FeasibilityStatus::InsufficientEvidence
    );
    assert_eq!(
        slow.kinetic_action_status,
        Some(FeasibilityStatus::NotRuledOut)
    );
    eprintln!(
        "mandelbrot_default, raster generation {}, rho upper {:.12}, point lower {:.12}, ball lower {:.12}, exit lower {:.12}, action lower {:.12} at 1s and {:.12} at 100s",
        region.raster_generation(),
        region.rho_upper(),
        point,
        ball,
        exit,
        fast.minimum_kinetic_action.unwrap(),
        slow.minimum_kinetic_action.unwrap()
    );

    let unsupported = [[0.465, -0.05], [0.470, 0.05]];
    let bound = RasterNavigationRegion::for_default_profile(unsupported)
        .and_then(|region| region.target_lower_bound([0.467, 0.0], [0.47, 0.0], 0.0));
    assert_eq!(bound, None);
    let control = evaluate_feasibility(LengthLowerBound::Unavailable, one_second, limits);
    assert_eq!(control.status, FeasibilityStatus::InsufficientEvidence);
    with_test_plane(|| {
        assert!(!region.is_current());
        assert!(RasterNavigationRegion::for_default_profile(bounds).is_none());
    });
    load_builtin_distance_field("mandelbrot_default").unwrap();
    assert!(!region.is_current());
    assert_eq!(region.target_lower_bound(start, center, 0.0), None);
}

#[test]
fn overshoot_and_incomplete_support_are_visible() {
    let _guard = global_test_mutex()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let bounds = [[-0.1, -0.1], [0.1, 0.1]];
    let tile = [0.0, -1.0, 1.0, 1.0, -1.0, 0.0];
    set_distance_field_from_vec(tile.repeat(6), 6, 6, -1.5, 1.5, -1.5, 1.5).unwrap();
    let region = RasterNavigationRegion::for_default_profile(bounds).unwrap();
    let center = Complex64::new(0.0, 0.0);
    let sampled = sample_signed_distance_field(&[center]).unwrap()[0];
    assert_eq!(sampled, 1.25);
    let epsilon = ManifoldConfig::default().epsilon;
    let naive = (1.0 + epsilon * epsilon).sqrt();
    let actual = rho_from_jet(&query_geometry(center, epsilon).unwrap(), epsilon);
    assert!(naive < actual && actual <= region.rho_upper());
    let stencil_crosses_support = [[0.85, -0.1], [0.88, 0.1]];
    {
        let slot = DIST_FIELD.read().unwrap();
        assert!(
            sample_magnitude_upper(slot.field.as_ref().unwrap(), stencil_crosses_support).is_some()
        );
    }
    assert!(RasterNavigationRegion::for_default_profile(stencil_crosses_support).is_none());
    assert!(RasterNavigationRegion::for_default_profile([[1.19, -0.1], [1.22, 0.1]]).is_none());
    assert!(RasterNavigationRegion::for_default_profile([[1.49, -0.1], [1.51, 0.1]]).is_none());

    let mut missing = tile.repeat(6);
    missing[7] = f32::NAN;
    set_distance_field_from_vec(missing, 6, 6, -1.5, 1.5, -1.5, 1.5).unwrap();
    assert!(RasterNavigationRegion::for_default_profile(bounds).is_none());
    set_distance_field_from_vec(vec![0.0; 4], 2, 2, -1.5, 1.5, -1.5, 1.5).unwrap();
    assert!(RasterNavigationRegion::for_default_profile(bounds).is_none());

    let subnormal = f32::from_bits(1);
    set_distance_field_from_vec(vec![subnormal; 36], 6, 6, -1.5, 1.5, -1.5, 1.5).unwrap();
    let tiny = RasterNavigationRegion::for_default_profile(bounds).unwrap();
    let jet = query_geometry(center, epsilon).unwrap();
    assert_eq!(jet.d, f64::from(subnormal));
    assert!(rho_from_jet(&jet, epsilon) <= tiny.rho_upper());
    set_distance_field_from_vec(vec![f32::MAX; 36], 6, 6, -1.5, 1.5, -1.5, 1.5).unwrap();
    assert!(RasterNavigationRegion::for_default_profile(bounds).is_none());
    load_builtin_distance_field("mandelbrot_default").unwrap();
}

#[cfg(feature = "experimental-operational-field")]
#[test]
fn prepared_field_override_disables_raster_certificate() {
    use crate::operational_field::{PreparedField, ACTIVE};
    use std::sync::Arc;

    let _guard = global_test_mutex()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    load_builtin_distance_field("mandelbrot_default").unwrap();
    let bounds = [[-1.5, -0.45], [-0.4, 0.45]];
    let region = RasterNavigationRegion::for_default_profile(bounds).unwrap();
    let field = PreparedField::new([-2.0, -2.0], 0.25, [8, 8], vec![0.25; 64]).unwrap();
    let previous = ACTIVE.with(|active| active.replace(Some(Arc::new(field))));
    let active_certificate = RasterNavigationRegion::for_default_profile(bounds);
    let raster_current = region.is_current();
    ACTIVE.with(|active| active.replace(previous));
    assert!(active_certificate.is_none());
    assert!(!raster_current);
    assert!(region.is_current());
}
