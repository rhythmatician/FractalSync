use runtime_core::map_budget::{
    assess_map_error, field_error_target, residual_error_bound, Assurance, MapErrorInputs,
    MapStatus,
};

#[test]
fn fixed_scale_budget_scales_with_window_size() {
    let inputs = MapErrorInputs {
        estimated_scale: 0.01,
        scale_error: 0.0,
        center_error: 0.000_001,
        query_error: 0.000_001,
        offset_norm: 1.0,
        numerical_error: 0.0,
        assurance: Assurance::Certified,
    };
    let result = assess_map_error(inputs, 0.001).unwrap();
    assert_eq!(result.status, MapStatus::Certified);
    assert!((result.error_bound.unwrap() - 0.0002).abs() < 1e-15);
    assert_eq!(
        assess_map_error(inputs, 0.0001).unwrap().status,
        MapStatus::Unresolved
    );
}

#[test]
fn computed_ruler_and_numerical_errors_are_separate_from_field_errors() {
    let result = assess_map_error(
        MapErrorInputs {
            estimated_scale: 2.0,
            scale_error: 0.2,
            center_error: 0.1,
            query_error: 0.3,
            offset_norm: 0.5,
            numerical_error: 0.03,
            assurance: Assurance::Certified,
        },
        0.4,
    )
    .unwrap();
    assert!((result.error_bound.unwrap() - 0.33).abs() < 1e-15);
    assert_eq!(result.status, MapStatus::Certified);
}

#[test]
fn heuristic_evidence_never_becomes_a_certificate() {
    let inputs = MapErrorInputs {
        estimated_scale: 1.0,
        scale_error: 0.0,
        center_error: 0.0,
        query_error: 0.0,
        offset_norm: 0.0,
        numerical_error: 0.0,
        assurance: Assurance::Heuristic,
    };
    assert_eq!(
        assess_map_error(inputs, 1.0).unwrap().status,
        MapStatus::Uncertified
    );
    let unavailable = assess_map_error(
        MapErrorInputs {
            assurance: Assurance::Unavailable,
            ..inputs
        },
        1.0,
    )
    .unwrap();
    assert_eq!(unavailable.status, MapStatus::Uncertified);
    assert_eq!(unavailable.error_bound, None);
    assert!(assess_map_error(
        MapErrorInputs {
            estimated_scale: 0.0,
            ..inputs
        },
        1.0
    )
    .is_none());
}

#[test]
fn absolute_tolerance_allocates_pointwise_error_at_fixed_scale() {
    assert!((field_error_target(0.01, 0.001).unwrap() - 0.000005).abs() < 1e-18);
    assert!(field_error_target(-1.0, 0.01).is_none());
}

#[test]
fn residual_budget_accounts_for_gradient_error_without_claiming_segment_validity() {
    // The supplied Hessian bound is applicable along the entire physical ray.
    let bound = residual_error_bound(4.0, 0.1, 2.0, 0.03).unwrap();
    assert!((bound - 0.86).abs() < 1e-14);
    assert!(residual_error_bound(f64::NAN, 0.1, 2.0, 0.03).is_none());
}
