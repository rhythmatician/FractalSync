use super::*;
use crate::manifold::ManifoldConfig;
use crate::operational_engine::{Integrator, PhysicsProfile, ReferenceCut};
use crate::operational_field::PreparedField;
use std::sync::Arc;

fn engine(cuts: Vec<ReferenceCut>) -> FrozenRollout {
    let field = PreparedField::new([-1.0; 2], 0.0625, [33; 2], vec![0.375; 33 * 33]).unwrap();
    let profile = PhysicsProfile::new(
        ManifoldConfig {
            epsilon: 0.5,
            ..ManifoldConfig::default()
        },
        0.01,
        0.0,
        Integrator::Rk4FourSubsteps,
    )
    .unwrap();
    FrozenRollout::new(Arc::new(field), profile, cuts).unwrap()
}

fn request(center: [f64; 2]) -> OperationalRouteRequest {
    OperationalRouteRequest {
        window: [[-0.5, 0.125], [0.5, 0.625]],
        grid_steps: 16,
        target: AmbientTarget {
            center,
            radius: 0.0,
        },
        length_budget: None,
        nominal_timing: None,
    }
}

#[test]
fn guidance_from_a_committed_position_supports_off_grid_points_and_arrival() {
    let engine = engine(vec![]);
    let state = engine.initial_state([-0.484375, 0.375], [0.0; 2]).unwrap();
    let target = [0.484375, 0.375];
    let saved_work = engine.last_evaluation_work();
    let report = engine.route_guidance(&state, request(target)).unwrap();
    assert_eq!(report.status, OperationalRouteStatus::RouteFound);
    assert_eq!(report.path.first(), Some(&state.position));
    assert_eq!(report.path.last(), Some(&target));
    assert!(report.graph_length_lower.unwrap() <= 1.55);
    assert!((report.route_length_upper.unwrap() - 1.55).abs() < 2e-6);
    let bearing = report.next_direction.unwrap();
    assert!((bearing[0] - 0.625).abs() < 1e-12 && bearing[1] == 0.0);

    let uncertain = engine
        .route_guidance(
            &state,
            OperationalRouteRequest {
                length_budget: Some(report.nominal_window_length_lower),
                ..request(target)
            },
        )
        .unwrap();
    assert_eq!(uncertain.status, OperationalRouteStatus::NoCertifiedRoute);
    assert!(uncertain.graph_length_lower.is_some());
    assert!(uncertain.path.is_empty());

    let mut ball = request(state.position);
    ball.target.center[0] += 0.01;
    ball.target.radius = 0.02;
    let reached = engine.route_guidance(&state, ball).unwrap();
    assert_eq!(reached.status, OperationalRouteStatus::Reached);
    assert!(reached.next_direction.is_none());
    assert_eq!(reached.route_length_upper, Some(0.0));
    assert_eq!(engine.last_evaluation_work(), saved_work);
}

#[test]
fn refusal_boxes_and_length_budget_change_guidance_without_mutating_state() {
    let cut = ReferenceCut {
        bounds: [-0.0625, 0.25, 0.0625, 0.5],
        authority: "synthetic route cut".into(),
    };
    let engine = engine(vec![cut]);
    let state = engine.initial_state([-0.4375, 0.375], [0.0; 2]).unwrap();
    let target = [0.4375, 0.375];
    let query = request(target);
    let route = engine.route_guidance(&state, query).unwrap();
    assert_eq!(route.status, OperationalRouteStatus::RouteFound);
    assert!(route.refused_edges > 0);
    assert!(route.path.iter().any(|p| p[1] < 0.25 || p[1] > 0.5));
    assert!(route.route_length_upper.unwrap() > 1.4);
    let exclusion = engine
        .route_guidance(
            &state,
            OperationalRouteRequest {
                length_budget: Some(1.0),
                ..query
            },
        )
        .unwrap();
    assert_eq!(
        exclusion.status,
        OperationalRouteStatus::GraphBudgetExcluded
    );
    assert!(exclusion.path.is_empty());
    assert_eq!(state.position, [-0.4375, 0.375]);
}

#[test]
fn a_full_refusal_barrier_leaves_navigation_without_a_graph_witness() {
    let engine = engine(vec![ReferenceCut {
        bounds: [-0.0625, 0.125, 0.0625, 0.625],
        authority: "synthetic disconnected window".into(),
    }]);
    let state = engine.initial_state([-0.4375, 0.375], [0.0; 2]).unwrap();
    let report = engine
        .route_guidance(&state, request([0.4375, 0.375]))
        .unwrap();
    assert_eq!(report.status, OperationalRouteStatus::NoCertifiedRoute);
    assert!(report.graph_length_lower.is_none());
    assert!(report.next_direction.is_none());
    assert!(report.refused_edges > 0);
}

#[test]
fn budget_prunes_spline_queries_before_search_and_preserves_a_route_at_the_budget() {
    let engine = engine(vec![]);
    let state = engine.initial_state([-0.4375, 0.375], [0.0; 2]).unwrap();
    let query = request([-0.3125, 0.375]);
    let full = engine.route_guidance(&state, query).unwrap();
    let budget = full.route_length_upper.unwrap();
    let bounded = engine
        .route_guidance(
            &state,
            OperationalRouteRequest {
                length_budget: Some(budget),
                ..query
            },
        )
        .unwrap();
    assert_eq!(bounded.status, OperationalRouteStatus::RouteFound);
    assert_eq!(bounded.route_length_upper, Some(budget));
    assert_eq!(bounded.path.first(), Some(&state.position));
    assert_eq!(bounded.path.last(), Some(&query.target.center));
    assert!(bounded.graph_length_lower.unwrap() <= 0.2);
    assert!(bounded.budget_skipped_edges > 0);
    assert!(bounded.edge_certificate_queries < full.edge_certificate_queries);
    assert!(bounded.cell_certificate_queries < full.cell_certificate_queries);
}

#[test]
fn partial_graph_without_a_witness_keeps_the_budget_as_a_lower_bound() {
    let engine = engine(vec![ReferenceCut {
        bounds: [-0.0625, 0.25, 0.0625, 0.5],
        authority: "synthetic route cut".into(),
    }]);
    let state = engine.initial_state([-0.4375, 0.375], [0.0; 2]).unwrap();
    let query = request([0.4375, 0.375]);
    let budget = 1.41;
    let report = engine
        .route_guidance(
            &state,
            OperationalRouteRequest {
                length_budget: Some(budget),
                ..query
            },
        )
        .unwrap();
    assert_eq!(report.status, OperationalRouteStatus::NoCertifiedRoute);
    assert!(report.nominal_window_length_lower < budget);
    assert!(report.budget_skipped_edges > 0);
    assert_eq!(report.graph_length_lower, Some(budget));
    assert!(report.route_length_upper.is_none());
}

#[test]
fn timed_guidance_assesses_action_under_the_nominal_window_premise() {
    use crate::crossing_feasibility::FeasibilityStatus;
    use crate::timebase::CANONICAL_SAMPLE_RATE;

    let engine = engine(vec![]);
    let state = engine.initial_state([-0.4375, 0.375], [0.0; 2]).unwrap();
    let target = [0.4375, 0.375];
    let timing = NominalRouteTiming {
        start_sample: 20,
        deadline_sample: 20 + CANONICAL_SAMPLE_RATE as u64,
        kinetic_action_budget: Some(0.1),
    };
    let report = engine
        .route_guidance(
            &state,
            OperationalRouteRequest {
                nominal_timing: Some(timing),
                ..request(target)
            },
        )
        .unwrap();
    let check = report.nominal_feasibility.unwrap();
    assert_eq!(report.status, OperationalRouteStatus::RouteFound);
    assert_eq!(check.status, FeasibilityStatus::RuledOut);
    assert_eq!(
        check.deadline_status,
        FeasibilityStatus::InsufficientEvidence
    );
    let invalid = engine.route_guidance(
        &state,
        OperationalRouteRequest {
            nominal_timing: Some(NominalRouteTiming {
                deadline_sample: 19,
                ..timing
            }),
            ..request(target)
        },
    );
    assert!(invalid.is_err());
}

#[test]
fn nominal_chord_lengths_enclose_a_committed_constant_metric_trace() {
    let rollout = engine(vec![]);
    let mut state = rollout.initial_state([0.0, 0.375], [0.25, 0.0]).unwrap();
    let (_, trace) = rollout
        .attempt_step_with_trace(&mut state, [0.0; 2])
        .unwrap();
    let bounds = request([0.0, 0.375]).window;
    let before = rollout.last_evaluation_work();
    let length = rollout
        .nominal_accepted_polyline_length(&trace, bounds)
        .unwrap();
    let action = rollout
        .nominal_accepted_chord_action(&trace, bounds)
        .unwrap();
    assert_eq!(action.length, length);
    let mut expected_action = 0.0;
    let mut missed = 0.0;
    for step in trace.rk_substeps() {
        let speed = step.stages[0][2].abs() / 0.625;
        expected_action += 0.5 * step.duration * speed * speed;
        let chord = (step.end[0] - step.stages[0][0]).abs() / 0.625;
        missed += (chord - step.duration * speed).max(0.0);
    }
    assert!(action.sampled_action_lower > 0.0);
    assert!(action.sampled_action_lower <= expected_action);
    assert!(action.missed_length_upper >= missed);
    assert!(
        engine(vec![])
            .nominal_accepted_chord_action(&trace, bounds)
            .is_none()
    );
    let expected = state.position[0] / 0.625;
    assert!(length.lower <= expected && expected <= length.upper);
    assert!(length.upper - length.lower < 1e-10);
    assert!(
        engine(vec![])
            .nominal_accepted_polyline_length(&trace, bounds)
            .is_none()
    );
    assert!(
        rollout
            .nominal_accepted_polyline_length(&trace, [[0.1, 0.125], bounds[1]])
            .is_none()
    );
    assert_eq!(rollout.last_evaluation_work(), before);
}
