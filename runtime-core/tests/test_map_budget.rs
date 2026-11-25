use num_bigint::BigInt;
use num_rational::BigRational;
use runtime_core::map_budget::{
    assess_map_error, field_error_target, residual_error_bound, Assurance, MapErrorInputs,
    MapStatus,
};

// Decode the input bits into an exact rational, independently of floating-point arithmetic.
fn exact(value: f64) -> BigRational {
    assert!(value.is_finite() && value >= 0.0);
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, shift) = if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1_u64 << 52), exponent - 1075)
    };
    if shift >= 0 {
        BigRational::from_integer(BigInt::from(significand) << shift as usize)
    } else {
        BigRational::new(
            BigInt::from(significand),
            BigInt::from(1) << (-shift) as usize,
        )
    }
}

fn exact_map_bound(input: MapErrorInputs) -> BigRational {
    (exact(input.center_error)
        + exact(input.query_error)
        + exact(input.offset_norm) * exact(input.scale_error) * BigInt::from(2))
        / exact(input.estimated_scale)
        + exact(input.numerical_error)
}

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
    assert!(exact(result.error_bound.unwrap()) >= exact_map_bound(inputs));
    assert_eq!(
        assess_map_error(inputs, 0.0001).unwrap().status,
        MapStatus::Unresolved
    );
}

#[test]
fn computed_ruler_and_numerical_errors_are_separate_from_field_errors() {
    let inputs = MapErrorInputs {
        estimated_scale: 2.0,
        scale_error: 0.2,
        center_error: 0.1,
        query_error: 0.3,
        offset_norm: 0.5,
        numerical_error: 0.03,
        assurance: Assurance::Certified,
    };
    let result = assess_map_error(inputs, 0.4).unwrap();
    assert!(exact(result.error_bound.unwrap()) >= exact_map_bound(inputs));
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
    assert!(
        exact(field_error_target(0.01, 0.001).unwrap())
            <= exact(0.01) * exact(0.001) / BigInt::from(2)
    );
    assert!(field_error_target(-1.0, 0.01).is_none());
}

#[test]
fn residual_budget_accounts_for_gradient_error_without_claiming_segment_validity() {
    // The supplied Hessian bound is applicable along the entire physical ray.
    let bound = residual_error_bound(4.0, 0.1, 2.0, 0.03).unwrap();
    let reference = exact(4.0) * exact(0.1) * exact(2.0) * exact(2.0) / BigInt::from(2)
        + exact(0.03) * exact(2.0);
    assert!(exact(bound) >= reference);
    assert!(residual_error_bound(f64::NAN, 0.1, 2.0, 0.03).is_none());
}

#[test]
fn certificate_and_rounding_hold_at_adjacent_thresholds_and_subnormal_boundaries() {
    let tiny = f64::from_bits(1);
    let cases = [
        MapErrorInputs {
            estimated_scale: 3.0,
            scale_error: 0.2,
            center_error: 0.1,
            query_error: 0.3,
            offset_norm: 0.5,
            numerical_error: 0.03,
            assurance: Assurance::Certified,
        },
        MapErrorInputs {
            estimated_scale: 1.0,
            scale_error: tiny,
            center_error: tiny,
            query_error: tiny,
            offset_norm: 0.5,
            numerical_error: tiny,
            assurance: Assurance::Certified,
        },
        MapErrorInputs {
            estimated_scale: 2.0,
            scale_error: 0.0,
            center_error: tiny,
            query_error: 0.0,
            offset_norm: 0.0,
            numerical_error: 0.0,
            assurance: Assurance::Certified,
        },
    ];
    for input in cases {
        let bound = assess_map_error(input, f64::MAX)
            .unwrap()
            .error_bound
            .unwrap();
        assert!(
            exact(bound) >= exact_map_bound(input),
            "underestimated: {input:?}"
        );
        assert_eq!(
            assess_map_error(input, bound).unwrap().status,
            MapStatus::Certified
        );
        let below = f64::from_bits(bound.to_bits() - 1);
        assert_eq!(
            assess_map_error(input, below).unwrap().status,
            MapStatus::Unresolved
        );
    }
    for (scale, tolerance) in [
        (1.0, tiny),
        (3.0, tiny),
        (0.01, 0.001),
        (f64::MIN_POSITIVE, 0.5),
    ] {
        let target = field_error_target(scale, tolerance).unwrap();
        assert!(exact(target) <= exact(scale) * exact(tolerance) / BigInt::from(2));
    }
    for (hessian, scale, offset, gradient) in [
        (4.0, 0.1, 2.0, 0.03),
        (tiny, 1.0, 0.5, tiny),
        (1.0, tiny, 1.0, 0.0),
    ] {
        let bound = residual_error_bound(hessian, scale, offset, gradient).unwrap();
        let reference = exact(hessian) * exact(scale) * exact(offset) * exact(offset)
            / BigInt::from(2)
            + exact(gradient) * exact(offset);
        assert!(exact(bound) >= reference);
    }
}
