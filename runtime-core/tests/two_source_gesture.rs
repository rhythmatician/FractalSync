use runtime_core::two_source_gesture::known_phase;
use runtime_core::two_source_gesture::known_phase_misaligned;
use runtime_core::two_source_gesture::{Geometry, Gesture, Source};

#[test]
fn decoded_pcm_can_supply_an_explicit_gesture_binding() {
    use runtime_core::{timebase::AnalysisTimebase, two_source_gesture::PassageCycles};
    let mut timebase = AnalysisTimebase::new();
    let mut cycles = PassageCycles::default();
    let rate = 44_100;
    let block_size = 441;
    let tau = std::f64::consts::TAU;
    for start in (0..rate * 18).step_by(block_size) {
        let pcm: Vec<f32> = (start..start + block_size)
            .map(|index| {
                let time = index as f64 / rate as f64;
                ((0.55 + 0.4 * (tau * 1.5 * time).cos()) * (tau * 330.0 * time).sin()) as f32
            })
            .collect();
        for tick in timebase.ingest(&pcm, rate, start as u64).unwrap() {
            cycles.observe_tick(&tick).unwrap();
        }
    }
    let frame = cycles.frame_at(18.0);
    assert_eq!(frame.phases, [None; 2]);
    let id = frame
        .modes
        .iter()
        .find(|mode| (mode.frequency_hz - 1.5).abs() < 0.15 && mode.missing_observations == 0)
        .expect("observed PCM modulation")
        .id;
    cycles.bind(0, Some(id)).unwrap();
    assert!(cycles.frame_at(18.0).phases[0].is_some());
}

#[test]
fn passage_cycle_roles_require_an_explicit_observed_identity() {
    use runtime_core::two_source_gesture::PassageCycles;
    let mut cycles = PassageCycles::default();
    assert!(cycles.bind(0, Some(1)).is_err());
    for second in 1..=18 {
        cycles.advance(second as f64).unwrap();
    }
    let frame = cycles.advance(18.0).unwrap();
    assert_eq!(frame.phases, [None, None]);
    let id = frame
        .modes
        .iter()
        .find(|mode| mode.missing_observations == 0)
        .unwrap()
        .id;
    cycles.bind(0, Some(id)).unwrap();
    assert_eq!(cycles.advance(18.0).unwrap().roles, [Some(id), None]);
    assert!(cycles.advance(18.1).unwrap().phases[0].is_some());
    assert!(cycles.bind(0, Some(u64::MAX)).is_err());
    assert_eq!(cycles.advance(18.1).unwrap().roles, [Some(id), None]);
    cycles.bind(0, None).unwrap();
    assert_eq!(cycles.advance(18.1).unwrap().phases, [None, None]);
}

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

#[test]
fn tracking_a_fixed_reference_uses_bounded_controls_and_physics() {
    use runtime_core::two_source_gesture::GestureTracker;
    let _guard = runtime_core::distance_field::global_test_mutex()
        .lock()
        .unwrap();
    runtime_core::distance_field::load_builtin_distance_field("mandelbrot_default").unwrap();
    let mut tracker = GestureTracker::new([-0.5, 0.1], [0.0, 0.0]).unwrap();
    let target = [-0.4, 0.1];
    let initial = (target[0] + 0.5_f64).abs();
    let mut sample = tracker.step(target, 0.01).unwrap();
    for _ in 0..199 {
        sample = tracker.step(target, 0.01).unwrap();
    }
    assert!(
        (sample.position[0] - target[0]).abs() < initial,
        "{sample:?}"
    );
    assert!((0.0..=1.0).contains(&sample.throttle));
    assert!(sample.energy.is_finite() && sample.control_work.is_finite());
    assert_ne!(
        sample.position, target,
        "Physics must realize the target, not assign it"
    );
}

#[test]
fn rejected_reference_does_not_advance_tracker() {
    use runtime_core::two_source_gesture::GestureTracker;
    let _guard = runtime_core::distance_field::global_test_mutex()
        .lock()
        .unwrap();
    runtime_core::distance_field::load_builtin_distance_field("mandelbrot_default").unwrap();
    let mut tracker = GestureTracker::new([-0.5, 0.1], [0.0, 0.0]).unwrap();
    let mut control = GestureTracker::new([-0.5, 0.1], [0.0, 0.0]).unwrap();
    assert!(tracker.step([f64::NAN, 0.0], 0.01).is_err());
    let after = tracker.step([-0.4, 0.1], 0.01).unwrap();
    let expected = control.step([-0.4, 0.1], 0.01).unwrap();
    assert_eq!(after.position, expected.position);
    assert_eq!(after.velocity, expected.velocity);
}

#[test]
fn free_navigation_and_tracking_share_position_and_momentum() {
    use runtime_core::two_source_gesture::GestureTracker;
    let _guard = runtime_core::distance_field::global_test_mutex()
        .lock()
        .unwrap();
    runtime_core::distance_field::load_builtin_distance_field("mandelbrot_default").unwrap();
    let mut tracker = GestureTracker::new([-0.5, 0.1], [0.0, 0.0]).unwrap();
    let before = tracker.step([-0.4, 0.1], 0.02).unwrap();
    let free = tracker.navigate([0.0, 1.0], 0.02).unwrap();
    assert_eq!(free.start_position, before.position);
    assert_eq!(free.start_velocity, before.velocity);
    assert!(free.reference.is_none());
    let after = tracker.step([-0.4, 0.1], 0.02).unwrap();
    assert_eq!(after.start_position, free.position);
    assert_eq!(after.start_velocity, free.velocity);
}
