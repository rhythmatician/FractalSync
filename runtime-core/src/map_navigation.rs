//! Target diagnostics for existing Controls v2 replays on the Map.
//!
//! Ideal Mandelbrot bounds can order a menu of recorded routes. Arrival is
//! observed at a sampled Controls state within the target ball. The ideal
//! bounds have no established correspondence to the operational metric or
//! its edge prices, so every route remains eligible and every scan runs to
//! the first observed arrival or the end of its valid prefix.

use std::f64::consts::LN_2;

use serde::Serialize;

use crate::intrinsic_bounds::{
    ambient_target_lower_bound, point_lower_bound, CrossingEvidence, IntrinsicMetric,
};
use crate::manifold::ManifoldConfig;
use crate::map_certificates::{
    certify_mandelbrot_endpoint, displacement_lower, MandelbrotEndpoint,
};

/// A Controls replay with its first recorded velocity. Supply 1 for a
/// successful step and 0 for a failed step. Only the uninterrupted successful
/// prefix can witness arrival.
pub struct MapRoute<'a> {
    pub positions: &'a [[f64; 2]],
    pub valid_steps: &'a [u8],
    pub first_velocity: [f64; 2],
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MapRouteSearch {
    /// Candidate indices in scan order.
    pub order: Vec<usize>,
    /// First arrival frame for each candidate, indexed in input order.
    pub arrivals: Vec<Option<usize>>,
    /// First arriving candidate encountered in this scan order.
    pub first_arrival_route: Option<usize>,
    /// Membership comparisons against the closed ambient target ball.
    pub target_queries: usize,
    pub queries_to_first_arrival: Option<usize>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MapNavigationReport {
    pub metric: &'static str,
    pub epsilon: f64,
    pub anisotropy: f64,
    pub start_lower_bound: Option<f64>,
    pub start_crossing_certified: bool,
    /// Original indices for the matching launch. The scans index into this list.
    pub candidate_indices: Vec<usize>,
    pub candidate_lower_bounds: Vec<Option<f64>>,
    pub valid_prefix_lengths: Vec<usize>,
    pub ranking_frame: usize,
    pub certificate_queries: usize,
    pub baseline: MapRouteSearch,
    pub bounded: MapRouteSearch,
}

/// Compare input order with a stable ordering by the remaining ideal distance
/// bound at the same recorded frame of each candidate. An unavailable bound
/// has priority zero. The completed coverage scan in both orders measures
/// the same witnessed arrivals. It also records how many comparisons each
/// order took to surface its first witness.
///
/// Compare replays matching the selected route's first recorded position and
/// velocity. All inputs must share the supplied physics profile.
pub fn evaluate_map_navigation(
    selected_route: usize,
    target_center: [f64; 2],
    target_radius: f64,
    ranking_frame: usize,
    config: &ManifoldConfig,
    candidates: &[MapRoute<'_>],
) -> Result<MapNavigationReport, String> {
    let selected = candidates
        .get(selected_route)
        .ok_or("Select a Map replay with recorded frames")?;
    let &start = selected
        .positions
        .first()
        .ok_or("Select a Map replay with recorded frames")?;
    if !finite_point(start)
        || !finite_point(selected.first_velocity)
        || !finite_point(target_center)
        || !target_radius.is_finite()
        || target_radius < 0.0
    {
        return Err(
            "Map navigation requires finite coordinates and a nonnegative target radius".into(),
        );
    }
    let metric = IntrinsicMetric {
        epsilon: config.epsilon,
        a: config.lambda_sq.sqrt() / LN_2,
    };
    if !metric.epsilon.is_finite() || metric.epsilon <= 0.0 || !metric.a.is_finite() {
        return Err(
            "Map navigation requires a finite positive ruler floor and finite anisotropy".into(),
        );
    }
    let mut candidate_indices = Vec::new();
    let mut prefixes = Vec::new();
    for (index, route) in candidates.iter().enumerate() {
        if route.positions.first() == Some(&start)
            && route.first_velocity == selected.first_velocity
        {
            candidate_indices.push(index);
            prefixes.push(valid_prefix(route)?);
        }
    }

    let mut certificate_queries = 0;
    let mut certify = |point| {
        certificate_queries += 1;
        certify_mandelbrot_endpoint(point)
    };
    let start_certificate = certify(start);
    let target_certificate = if target_center == start {
        start_certificate
    } else {
        certify(target_center)
    };
    let target_bound = |point, endpoint| {
        let start: MandelbrotEndpoint = endpoint?;
        let target = target_certificate?;
        let start_bounds = start.bounds(metric)?;
        let target_bounds = target.bounds(metric)?;
        let displacement = displacement_lower(point, target_center)?;
        let result = if target_radius == 0.0 {
            point_lower_bound(
                metric,
                start_bounds,
                target_bounds,
                displacement,
                start.crossing_to(target),
            )
        } else {
            ambient_target_lower_bound(
                metric,
                start_bounds,
                target_bounds,
                displacement,
                target_radius,
            )
        };
        result.map(|bound| bound.lower)
    };
    let start_lower_bound = target_bound(start, start_certificate);
    let start_crossing_certified = target_radius == 0.0
        && start_lower_bound.is_some()
        && start_certificate
            .zip(target_certificate)
            .is_some_and(|(from, to)| from.crossing_to(to) == CrossingEvidence::Certified);
    let candidate_lower_bounds: Vec<_> = prefixes
        .iter()
        .map(|positions| {
            let point = *positions.get(ranking_frame)?;
            if point == start {
                return start_lower_bound;
            }
            if target_certificate.is_none() {
                return None;
            }
            let endpoint = if point == target_center {
                target_certificate
            } else {
                certify(point)
            };
            target_bound(point, endpoint)
        })
        .collect();

    let baseline_order: Vec<_> = (0..prefixes.len()).collect();
    let mut bounded_order = baseline_order.clone();
    bounded_order.sort_by(|&left, &right| {
        candidate_lower_bounds[left]
            .unwrap_or(0.0)
            .total_cmp(&candidate_lower_bounds[right].unwrap_or(0.0))
    });
    Ok(MapNavigationReport {
        metric: "ideal_mandelbrot_distance_graph",
        epsilon: metric.epsilon,
        anisotropy: metric.a,
        start_lower_bound,
        start_crossing_certified,
        candidate_indices,
        candidate_lower_bounds,
        valid_prefix_lengths: prefixes.iter().map(|path| path.len()).collect(),
        ranking_frame,
        certificate_queries,
        baseline: scan_routes(&prefixes, baseline_order, target_center, target_radius),
        bounded: scan_routes(&prefixes, bounded_order, target_center, target_radius),
    })
}

fn valid_prefix<'a>(route: &MapRoute<'a>) -> Result<&'a [[f64; 2]], String> {
    if route.positions.len() != route.valid_steps.len()
        || route.valid_steps.iter().any(|&flag| flag > 1)
    {
        return Err("Map routes need one binary success flag per frame".into());
    }
    let length = route
        .valid_steps
        .iter()
        .take_while(|&&flag| flag == 1)
        .count();
    let prefix = &route.positions[..length];
    if !prefix.iter().all(|&point| finite_point(point)) {
        return Err("A successful Map frame has a nonfinite position".into());
    }
    Ok(prefix)
}

fn finite_point(point: [f64; 2]) -> bool {
    point.into_iter().all(f64::is_finite)
}

fn scan_routes(
    prefixes: &[&[[f64; 2]]],
    order: Vec<usize>,
    center: [f64; 2],
    radius: f64,
) -> MapRouteSearch {
    let mut arrivals = vec![None; prefixes.len()];
    let mut target_queries = 0;
    let mut first_arrival_route = None;
    let mut queries_to_first_arrival = None;
    for &route in &order {
        arrivals[route] = prefixes[route].iter().position(|&[x, y]| {
            target_queries += 1;
            (x - center[0]).hypot(y - center[1]) <= radius
        });
        if first_arrival_route.is_none() && arrivals[route].is_some() {
            first_arrival_route = Some(route);
            queries_to_first_arrival = Some(target_queries);
        }
    }
    MapRouteSearch {
        order,
        arrivals,
        first_arrival_route,
        target_queries,
        queries_to_first_arrival,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: [f64; 2] = [0.1875, 0.0];
    const TARGET: [f64; 2] = [0.375, 0.0];

    #[test]
    fn coverage_requires_an_observed_arrival_in_a_successful_prefix() {
        let stuck = [START, [0.20, 0.0], [0.23, 0.0]];
        let hit = [START, [0.25, 0.0], TARGET];
        let aborted = [START, [0.25, 0.0], TARGET, TARGET];
        let initial = [START];
        let report = evaluate_map_navigation(
            0,
            TARGET,
            0.0,
            2,
            &ManifoldConfig::default(),
            &[
                MapRoute {
                    positions: &stuck,
                    valid_steps: &[1, 1, 1],
                    first_velocity: [0.0, 0.0],
                },
                MapRoute {
                    positions: &hit,
                    valid_steps: &[1, 1, 1],
                    first_velocity: [1.0, 0.0],
                },
                MapRoute {
                    positions: &hit,
                    valid_steps: &[1, 1, 1],
                    first_velocity: [0.0, 0.0],
                },
                MapRoute {
                    positions: &aborted,
                    valid_steps: &[1, 1, 0, 1],
                    first_velocity: [0.0, 0.0],
                },
                MapRoute {
                    positions: &[TARGET],
                    valid_steps: &[1],
                    first_velocity: [0.0, 0.0],
                },
                MapRoute {
                    positions: &initial,
                    valid_steps: &[1],
                    first_velocity: [0.0, 0.0],
                },
            ],
        )
        .unwrap();
        assert!(report.start_crossing_certified);
        assert!(report.start_lower_bound.unwrap() > 0.0);
        assert_eq!(report.candidate_indices, [0, 2, 3, 5]);
        assert_eq!(report.valid_prefix_lengths, [3, 3, 2, 1]);
        assert_eq!(report.baseline.arrivals, [None, Some(2), None, None]);
        assert_eq!(report.bounded.arrivals, report.baseline.arrivals);
        assert_eq!(report.baseline.target_queries, 9);
        assert_eq!(report.bounded.target_queries, 9);
        assert_eq!(report.baseline.queries_to_first_arrival, Some(6));
        assert_eq!(report.bounded.queries_to_first_arrival, Some(3));
    }

    #[test]
    fn unavailable_bounds_leave_candidate_order_and_arrivals_intact() {
        // Inflating the target enclosure to this radius overflows the
        // upper ruler, so a valid ambient target can have no usable bound.
        let path = [START, TARGET];
        let report = evaluate_map_navigation(
            0,
            TARGET,
            f64::MAX,
            1,
            &ManifoldConfig::default(),
            &[
                MapRoute {
                    positions: &path,
                    valid_steps: &[1, 1],
                    first_velocity: [0.0, 0.0],
                },
                MapRoute {
                    positions: &path,
                    valid_steps: &[1, 1],
                    first_velocity: [0.0, 0.0],
                },
            ],
        )
        .unwrap();
        assert!(!report.start_crossing_certified);
        assert_eq!(report.start_lower_bound, None);
        assert_eq!(report.candidate_lower_bounds, [None, None]);
        assert_eq!(report.bounded, report.baseline);
        assert_eq!(report.bounded.arrivals, [Some(0), Some(0)]);
    }

    #[test]
    fn selected_route_requires_recorded_frames() {
        assert!(evaluate_map_navigation(
            0,
            TARGET,
            0.0,
            0,
            &ManifoldConfig::default(),
            &[MapRoute {
                positions: &[],
                valid_steps: &[],
                first_velocity: [0.0, 0.0],
            }],
        )
        .is_err());
    }
}
