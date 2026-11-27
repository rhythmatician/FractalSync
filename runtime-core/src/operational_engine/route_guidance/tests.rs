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
    }
}

#[test]
fn guidance_from_a_committed_position_supports_off_grid_points_and_arrival() {
    let engine = engine(vec![]);
    let state = engine.initial_state([-0.484375, 0.375], [0.0; 2]).unwrap();
    let target = [0.484375, 0.375];
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
                length_budget: Some(
                    (report.graph_length_lower.unwrap() + report.route_length_upper.unwrap()) / 2.0,
                ),
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
