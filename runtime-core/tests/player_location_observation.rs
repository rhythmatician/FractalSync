use num_complex::Complex64;
use runtime_core::model_io::player_location_observation;

#[test]
fn arrows_isolate_positive_scale_and_positive_turn() {
    use runtime_core::model_io::{fixed_point_effect_headings, fixed_point_inverse_response};
    let c = Complex64::new(-0.7269, 0.1889);
    let rate = fixed_point_inverse_response(c).unwrap();
    let [sx, sy, rx, ry] = fixed_point_effect_headings(c).unwrap();
    let scale = rate * Complex64::new(sx, sy);
    let rotation = rate * Complex64::new(rx, ry);
    assert!(scale.re > 0.0 && scale.im.abs() < 1e-12);
    assert!(rotation.im > 0.0 && rotation.re.abs() < 1e-12);
    assert!(fixed_point_effect_headings(Complex64::new(0.0, 0.0)).is_none());
}

#[test]
fn pre_action_position_and_velocity_are_preserved_when_geometry_is_unavailable() {
    let observation = player_location_observation(
        Complex64::new(0.5, 0.0),
        (0.25, -0.125),
        &Default::default(),
    );
    assert_eq!(observation, [0.5, 0.0, 0.25, -0.125, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn resolved_repelling_branch_supplies_independent_scale_and_turn_rates() {
    let row = player_location_observation(
        Complex64::new(-0.7569, 0.1439), (0.0, 0.0), &Default::default(),
    );
    assert_eq!(row[8], 1.0);
    assert!(row[4] * row[7] - row[5] * row[6] > 0.0);
}
