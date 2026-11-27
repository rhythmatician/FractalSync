//! Exercise the navigation-to-feasibility handoff on analytic flat-Shore paths.

use runtime_core::controller::SAMPLE_RATE;
use runtime_core::crossing_feasibility::{
    evaluate_feasibility, FeasibilityAssessment, FeasibilityLimits, FeasibilityStatus,
    LengthLowerBound, SpeedCap,
};
use runtime_core::debug::snapshot_from_state;
use runtime_core::intrinsic_bounds::{
    intrinsic_target_lower_bound, point_lower_bound, CrossingEvidence, EndpointBounds,
    IntrinsicMetric, PointBound,
};
use serde_json::json;

const FLOOR: f64 = 0.25;
const METRIC: IntrinsicMetric = IntrinsicMetric {
    epsilon: FLOOR,
    a: 0.0,
};
const LEFT: [f64; 2] = [-0.5, 0.0];
const RIGHT: [f64; 2] = [0.5, 0.0];

fn endpoint(point: [f64; 2]) -> EndpointBounds {
    let d = point[0].abs();
    EndpointBounds {
        shore_distance: [d; 2],
        ruler_upper: FLOOR + d,
    }
}

fn bound(start: [f64; 2], goal: [f64; 2], crossing: CrossingEvidence) -> PointBound {
    // The selected endpoints have exact dyadic offsets. The maximum coordinate
    // displacement is an exact lower bound on the Euclidean displacement.
    let displacement = (goal[0] - start[0]).abs().max((goal[1] - start[1]).abs());
    point_lower_bound(
        METRIC,
        endpoint(start),
        endpoint(goal),
        displacement,
        crossing,
    )
    .expect("analytic endpoints and profile")
}

fn assessment(lower: f64, duration_samples: u64) -> FeasibilityAssessment {
    evaluate_feasibility(
        LengthLowerBound::Certified(lower),
        duration_samples,
        FeasibilityLimits {
            speed_cap: Some(SpeedCap::Certified(1.0)),
            ..FeasibilityLimits::default()
        },
    )
}

#[derive(Clone, Copy, serde::Serialize)]
struct ObservedPath {
    length: f64,
    kinetic_action: f64,
}

fn integrate_normal_path(
    start: f64,
    end: f64,
    duration: f64,
    variable_speed: bool,
    panels: u32,
) -> ObservedPath {
    // Reconstruct physical coordinate velocity first and measure its norm in
    // the flat conformal graph. The chart u=asinh(r/epsilon) sizes the motion.
    let [a, b] = [start, end].map(|r| (r / FLOOR).asinh());
    let speed = |t: f64| {
        let q = t / duration;
        let (fraction, derivative) = if variable_speed {
            (q * q, 2.0 * q / duration)
        } else {
            (q, 1.0 / duration)
        };
        let u = a + (b - a) * fraction;
        let r = FLOOR * u.sinh();
        let dr = FLOOR * u.cosh() * (b - a) * derivative;
        dr.abs() / FLOOR.hypot(r)
    };
    let h = duration / f64::from(panels);
    let (mut length, mut action) = (0.0, 0.0);
    for j in 0..=panels {
        let s = speed(f64::from(j) * h);
        let weight = if j == 0 || j == panels {
            1.0
        } else if j % 2 == 0 {
            2.0
        } else {
            4.0
        };
        length += weight * s;
        action += weight * s * s / 2.0;
    }
    ObservedPath {
        length: length * h / 3.0,
        kinetic_action: action * h / 3.0,
    }
}

fn main() -> Result<(), String> {
    let sample_rate = SAMPLE_RATE as u64;
    let crossing = bound(LEFT, RIGHT, CrossingEvidence::Certified);
    let reverse = bound(RIGHT, LEFT, CrossingEvidence::Certified);
    let without_crossing = bound(LEFT, RIGHT, CrossingEvidence::Unknown);
    assert_eq!(crossing.lower, reverse.lower);
    assert!(crossing.lower > without_crossing.lower);

    // A cap applies only to the scenario for which the caller has evidence.
    // For the 4-second constant-speed normal witness, |asinh(2)| < 2 suffices.
    let too_early = assessment(crossing.lower, 2 * sample_rate);
    let normal_window = assessment(crossing.lower, 4 * sample_rate);
    let near_enough_lower =
        intrinsic_target_lower_bound(crossing.lower, 0.25).expect("valid tolerance");
    let near_enough = assessment(near_enough_lower, 2 * sample_rate);
    assert_eq!(too_early.status, FeasibilityStatus::RuledOut);
    assert_eq!(normal_window.status, FeasibilityStatus::NotRuledOut);
    assert_eq!(near_enough.status, FeasibilityStatus::NotRuledOut);

    let ideal = 2.0 * 2.0_f64.asinh();
    let constant_speed = integrate_normal_path(LEFT[0], RIGHT[0], 4.0, false, 128);
    let reverse_speed = integrate_normal_path(RIGHT[0], LEFT[0], 4.0, false, 128);
    let variable_speed = integrate_normal_path(LEFT[0], RIGHT[0], 4.0, true, 128);
    let variable_refined = integrate_normal_path(LEFT[0], RIGHT[0], 4.0, true, 256);
    assert!((constant_speed.length - ideal).abs() < 1e-12);
    assert!((constant_speed.kinetic_action - ideal.powi(2) / 8.0).abs() < 1e-12);
    assert!((reverse_speed.length - constant_speed.length).abs() < 1e-12);
    assert!((variable_speed.length - variable_refined.length).abs() < 1e-12);
    assert!((variable_speed.kinetic_action - variable_refined.kinetic_action).abs() < 1e-12);
    // This path is used only for action evidence. Its peak speed exceeds the
    // certified speed cap of the separate constant-speed scenario.
    assert!(variable_refined.kinetic_action > constant_speed.kinetic_action);
    assert!(constant_speed.length >= crossing.lower);
    assert!(constant_speed.kinetic_action >= normal_window.minimum_kinetic_action.unwrap());

    // Travel down to Shore, along it, then away. Normal legs take one second
    // each, and the 0.5-unit tangential leg takes two seconds at ruler 0.25.
    let left_leg = integrate_normal_path(-0.5, 0.0, 1.0, false, 128);
    let right_leg = integrate_normal_path(0.0, 0.5, 1.0, false, 128);
    let detour = ObservedPath {
        length: left_leg.length + 2.0 + right_leg.length,
        kinetic_action: left_leg.kinetic_action + 1.0 + right_leg.kinetic_action,
    };
    let detour_bound = bound(LEFT, [0.5, 0.5], CrossingEvidence::Certified);
    assert!(detour.length > ideal && detour.length >= detour_bound.lower);

    let on_shore = bound([0.0, 0.0], [0.0, 0.5], CrossingEvidence::Unknown);
    let same_side = bound(RIGHT, [1.0, 0.0], CrossingEvidence::Unknown);
    assert!(0.0 < on_shore.lower && on_shore.lower <= 2.0);
    assert!(0.0 < same_side.lower && same_side.lower <= 4.0_f64.asinh() - 2.0_f64.asinh());

    let uncertain = evaluate_feasibility(
        LengthLowerBound::Estimate(crossing.lower),
        sample_rate,
        FeasibilityLimits {
            speed_cap: Some(SpeedCap::Certified(1.0)),
            ..FeasibilityLimits::default()
        },
    );
    let missing_speed = evaluate_feasibility(
        LengthLowerBound::Certified(crossing.lower),
        sample_rate,
        FeasibilityLimits::default(),
    );
    assert_eq!(uncertain.status, FeasibilityStatus::InsufficientEvidence);
    assert_eq!(
        missing_speed.status,
        FeasibilityStatus::InsufficientEvidence
    );
    assert!(missing_speed.minimum_kinetic_action.is_some());

    // Read the same optional raster diagnostic that the cockpit records.
    // It supplies an estimate to the Shore, never evidence for exclusion.
    let config = runtime_core::manifold::ManifoldConfig::default();
    let snapshot = snapshot_from_state(
        num_complex::Complex64::new(-0.45, 0.1),
        (0.0, 0.0),
        None,
        None,
        &config,
        None,
    )?;
    let geometry = snapshot.diagnostics.geometry;
    let raster_length = geometry
        .shore_distance_estimate
        .map_or(LengthLowerBound::Unavailable, |radial| {
            LengthLowerBound::Estimate(radial.value)
        });
    assert!(matches!(raster_length, LengthLowerBound::Estimate(_)));
    let raster = evaluate_feasibility(raster_length, sample_rate, FeasibilityLimits::default());
    assert_eq!(raster.status, FeasibilityStatus::InsufficientEvidence);

    println!(
        "{}",
        json!({
            "schema": "navigation-feasibility-fixture/1",
            "reference": "analytic straight Shore on the plane",
            "sample_rate": sample_rate,
            "epsilon": FLOOR,
            "anisotropy": 0.0,
            "start": LEFT,
            "goal": RIGHT,
            "point_length_lower": crossing.lower,
            "length_lower_without_certified_crossing": without_crossing.lower,
            "target_tolerance_intrinsic": 0.25,
            "too_early": too_early,
            "normal_window": normal_window,
            "near_enough": near_enough,
            "constant_speed": constant_speed,
            "variable_speed": variable_refined,
            "detour": detour,
            "detour_length_lower": detour_bound.lower,
            "on_shore_length_lower": on_shore.lower,
            "same_side_length_lower": same_side.lower,
            "estimated_length": uncertain,
            "missing_speed_cap": missing_speed,
            "raster_shore_arrival": {
                "start": snapshot.physics.c,
                "geometry": geometry,
                "assessment": raster,
            },
        })
    );
    Ok(())
}
