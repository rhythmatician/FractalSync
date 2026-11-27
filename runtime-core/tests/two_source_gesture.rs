use runtime_core::two_source_gesture::known_phase;
use runtime_core::two_source_gesture::known_phase_misaligned;
use runtime_core::two_source_gesture::{Geometry, Gesture, Source};

fn source(phase: f64, gain: f64) -> Source {
    Source { phase, gain }
}

#[test]
fn known_passage_departs_from_and_returns_to_three_to_four_lock() {
    let (a, b) = known_phase(0.0).unwrap();
    let reference = 4.0 * a - 3.0 * b;
    let (a, b) = known_phase(16.0).unwrap();
    assert!((4.0 * a - 3.0 * b - reference).abs() > 1.0);
    let (a, b) = known_phase(24.0).unwrap();
    assert!((4.0 * a - 3.0 * b - reference).abs() < 1e-10);
    assert!(known_phase(-1.0).is_err());
}

#[test]
fn mismatched_comparison_changes_visual_relationship_at_different_times() {
    let (a, b) = known_phase(1.0).unwrap();
    let (other_a, other_b) = known_phase_misaligned(1.0).unwrap();
    assert_eq!(a, other_a);
    assert!((b - other_b).sin().abs() > 0.5);
}

#[test]
fn rotating_sources_reinforce_and_cancel_at_equal_gain() {
    let gesture = Gesture {
        geometry: Geometry::Rotating,
        excursion: 1.0,
    };
    let reinforce = gesture
        .displacement(source(0.0, 0.5), source(0.0, 0.5))
        .unwrap();
    let cancel = gesture
        .displacement(source(0.0, 0.5), source(std::f64::consts::PI, 0.5))
        .unwrap();
    assert!((reinforce[0] - 1.0).abs() < 1e-12);
    assert!(cancel[0].hypot(cancel[1]) < 1e-12);
}

#[test]
fn distinct_sources_move_independently() {
    let gesture = Gesture {
        geometry: Geometry::Distinct,
        excursion: 0.25,
    };
    let result = gesture
        .displacement(source(0.0, 1.0), source(std::f64::consts::PI, 0.5))
        .unwrap();
    assert!((result[0] - 0.25).abs() < 1e-12);
    assert!((result[1] + 0.125).abs() < 1e-12);
}

#[test]
fn refuses_nonfinite_or_negative_inputs() {
    let gesture = Gesture {
        geometry: Geometry::Rotating,
        excursion: 1.0,
    };
    assert!(gesture
        .displacement(source(f64::NAN, 1.0), source(0.0, 1.0))
        .is_err());
    assert!(gesture
        .displacement(source(0.0, -1.0), source(0.0, 1.0))
        .is_err());
    assert!(gesture
        .displacement(source(0.0, f64::MAX), source(0.0, f64::MAX))
        .is_err());
}

#[test]
fn locked_three_to_four_sources_still_change_excursion() {
    let gesture = Gesture {
        geometry: Geometry::Rotating,
        excursion: 1.0,
    };
    let start = gesture
        .displacement(source(0.0, 0.5), source(0.0, 0.5))
        .unwrap();
    let theta = std::f64::consts::PI;
    let opposite = gesture
        .displacement(source(3.0 * theta, 0.5), source(4.0 * theta, 0.5))
        .unwrap();
    assert!(start[0].hypot(start[1]) > 0.99);
    assert!(opposite[0].hypot(opposite[1]) < 1e-12);
}

#[test]
fn local_mapping_preserves_intrinsic_length_at_anchor() {
    let gesture = Gesture {
        geometry: Geometry::Distinct,
        excursion: 1.0,
    };
    let point = gesture
        .local_target(
            [0.1, 0.2],
            [[4.0, 1.0], [1.0, 2.0]],
            source(0.0, 0.6),
            source(0.0, 0.8),
        )
        .unwrap();
    let x = point[0] - 0.1;
    let y = point[1] - 0.2;
    assert!((4.0 * x * x + 2.0 * x * y + 2.0 * y * y - 1.0).abs() < 1e-12);
    assert!(gesture
        .local_target(
            [0.0, 0.0],
            [[1.0, 2.0], [2.0, 1.0]],
            source(0.0, 1.0),
            source(0.0, 0.0)
        )
        .is_err());
}
