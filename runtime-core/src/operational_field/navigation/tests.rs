use crate::manifold::ManifoldConfig;
use crate::operational_field::PreparedField;

#[test]
fn constant_field_encloses_exact_diagonal_and_target_tolerance() {
    let field = PreparedField::new([0.0; 2], 0.125, [20; 2], vec![0.375; 400]).unwrap();
    let config = ManifoldConfig {
        epsilon: 0.5,
        ..Default::default()
    };
    let region = field
        .navigation_region([[0.5, 0.5], [1.5, 1.5]], &config)
        .unwrap();
    let start = [0.5, 0.75];
    let end = [1.25, 1.25];
    let ruler = 0.625;
    let exact = (0.75_f64.powi(2) + 0.5_f64.powi(2)).sqrt() / ruler;
    let length = region.segment_bounds(start, end).unwrap();
    assert!(length.lower <= exact && exact <= length.upper);
    assert!(length.upper - length.lower < 1e-12);
    assert_eq!(length, region.segment_bounds(end, start).unwrap());
    assert_eq!(region.segment_bounds(start, start).unwrap().upper, 0.0);
    let lower = region.target_lower_bound(start, end, 0.25).unwrap();
    assert!(lower <= exact - 0.25 / ruler && (exact - 0.25 / ruler) - lower < 1e-12);
    assert_eq!(region.target_lower_bound(start, end, 2.0), Some(0.0));
    assert!(region.budget_radius_upper(2.0).unwrap() >= 2.0 * ruler);
}

#[test]
fn scalar_variation_on_a_join_is_enclosed_and_missing_support_refuses() {
    let mut values = vec![Some(0.0); 24 * 24];
    for (row, coefficients) in values.chunks_exact_mut(24).enumerate() {
        for (column, coefficient) in coefficients.iter_mut().enumerate() {
            *coefficient = Some(column as f64 / 8.0 - 1.0);
            if (column, row) == (12, 12) {
                *coefficient = None;
            }
        }
    }
    let field = PreparedField::new_sparse([0.0; 2], 0.125, [24; 2], values).unwrap();
    let config = ManifoldConfig {
        epsilon: 0.25,
        lambda_sq: 0.0,
        ..Default::default()
    };
    let region = field
        .navigation_region([[0.5, 0.5], [2.0, 2.0]], &config)
        .unwrap();
    // Spline first-moment reproduction gives F(x,y) = x - 1. The conformal
    // horizontal integral across the joins has the analytic asinh primitive.
    let exact = 2.0 * 2.0_f64.asinh();
    let cross = region.segment_bounds([0.5, 0.5], [1.5, 0.5]).unwrap();
    assert!(cross.lower <= exact && exact <= cross.upper);
    assert!(region.segment_bounds([0.5, 1.5], [2.0, 1.5]).is_none());
    assert!(region.segment_bounds([0.5, 0.5], [2.125, 0.5]).is_none());
    assert!(
        field
            .navigation_region([[0.0, 0.0], [1.0, 1.0]], &config)
            .is_none()
    );
}
