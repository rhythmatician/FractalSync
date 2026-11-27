//! Compare searches on exactly the same prepared-field edge enclosures.

use runtime_core::crossing_feasibility::{
    FeasibilityLimits, FeasibilityStatus, LengthLowerBound, evaluate_feasibility,
};
use runtime_core::manifold::ManifoldConfig;
use runtime_core::map_certificates::displacement_lower;
use runtime_core::operational_field::{OperationalNavigationRegion, PreparedField};
use runtime_core::timebase::CANONICAL_SAMPLE_RATE;
use serde_json::{Value, json};

#[path = "support/map_search.rs"]
mod map_search;
use map_search::{Counts, SearchLimits, search};

const WIDTH: usize = 101;
const HEIGHT: usize = 81;
const NODES: usize = WIDTH * HEIGHT;
const STEP: f64 = 1.0 / 2048.0;
const QUANTA: f64 = (1 << 24) as f64;
const DIRECTIONS: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];

#[derive(Clone, Copy)]
struct Edge {
    to: usize,
    lower: f64,
    upper: f64,
    diagonal: bool,
}

fn point(node: usize) -> [f64; 2] {
    [
        (450.0 + (node % WIDTH) as f64) * STEP,
        ((node / WIDTH) as f64 - 41.0) * STEP,
    ]
}

fn estimate_length(
    field: &PreparedField,
    config: &ManifoldConfig,
    start: [f64; 2],
    end: [f64; 2],
    subdivisions: usize,
) -> Result<f64, String> {
    let [x, y] = [end[0] - start[0], end[1] - start[1]];
    let mut total = 0.0;
    for i in 0..subdivisions {
        let t = (i as f64 + 0.5) / subdivisions as f64;
        let jet = field
            .query([start[0] + x * t, start[1] + y * t])
            .map_err(|_| "compiled query refused interior quadrature point")?;
        let rho = jet.value.hypot(config.epsilon);
        let rho_rate = jet.value / rho * (jet.gradient[0] * x + jet.gradient[1] * y);
        let squared_speed =
            x * x + y * y + config.lambda_sq * (rho_rate / std::f64::consts::LN_2).powi(2);
        total += squared_speed.sqrt() / rho;
    }
    Ok(total / subdivisions as f64)
}

fn graph(
    region: &OperationalNavigationRegion<'_>,
    field: &PreparedField,
    config: &ManifoldConfig,
) -> Result<(Vec<Vec<Edge>>, Value), String> {
    let mut adjacency = vec![Vec::new(); NODES];
    let mut certified = 0;
    let mut refused = 0;
    let mut checked = 0;
    let mut max_change = 0.0_f64;
    let mut max_upper = 0.0_f64;
    let mut max_width = 0.0_f64;
    for from in 0..NODES {
        for (dx, dy) in DIRECTIONS {
            let Some(x) = (from % WIDTH).checked_add_signed(dx).filter(|&x| x < WIDTH) else {
                continue;
            };
            let Some(y) = (from / WIDTH)
                .checked_add_signed(dy)
                .filter(|&y| y < HEIGHT)
            else {
                continue;
            };
            let to = y * WIDTH + x;
            let Some(length) = region.segment_bounds(point(from), point(to)) else {
                refused += 1;
                continue;
            };
            if certified % 97 == 0 {
                let coarse = estimate_length(field, config, point(from), point(to), 32)?;
                let fine = estimate_length(field, config, point(from), point(to), 128)?;
                assert!(length.lower - 1e-10 <= fine && fine <= length.upper + 1e-10);
                max_change = max_change.max((fine - coarse).abs());
                checked += 1;
            }
            certified += 1;
            let diagonal = dx != 0 && dy != 0;
            let lower = (length.lower * QUANTA).floor() / QUANTA;
            let upper = (length.upper * QUANTA).ceil() / QUANTA;
            max_upper = max_upper.max(upper);
            max_width = max_width.max(upper - lower);
            adjacency[from].push(Edge {
                to,
                lower,
                upper,
                diagonal,
            });
            adjacency[to].push(Edge {
                to: from,
                lower,
                upper,
                diagonal,
            });
        }
    }
    // Simple paths have fewer than NODES edges. The priority search can
    // compare all represented path sums exactly in binary64.
    assert!(max_upper * QUANTA * ((NODES + 1) as f64) < (1_u64 << 53) as f64);
    Ok((
        adjacency,
        json!({
            "certifiedEdges": certified, "refusedEdges": refused,
            "edgesCheckedByCompiledQuadrature": checked, "maxQuadratureChange": max_change,
            "maxEnclosureWidth": max_width
        }),
    ))
}

fn compare(
    region: &OperationalNavigationRegion<'_>,
    adjacency: &[Vec<Edge>],
    diagonal: bool,
) -> Value {
    let cases = [
        (40 * WIDTH + 2, 40 * WIDTH + 87, 0),
        (18 * WIDTH + 20, 53 * WIDTH + 90, 0),
        (55 * WIDTH + 20, 22 * WIDTH + 88, 4),
        (37 * WIDTH + 85, 70 * WIDTH + 93, 3),
    ];
    let mut totals = [Counts::default(); 2];
    let mut outcomes = Vec::new();
    for (start, center, radius_steps) in cases {
        let radius = f64::from(radius_steps) * STEP;
        let accepts = |node: usize| {
            let x = node as i32 % WIDTH as i32 - center as i32 % WIDTH as i32;
            let y = node as i32 / WIDTH as i32 - center as i32 / WIDTH as i32;
            x * x + y * y <= radius_steps * radius_steps
        };
        let heuristic = |node| region.target_lower_bound(point(node), point(center), radius);
        let lower = search(
            NODES,
            start,
            accepts,
            |node| {
                adjacency[node].iter().filter_map(move |edge| {
                    (!edge.diagonal || diagonal).then_some((edge.to, edge.lower))
                })
            },
            |_| None,
            |_| true,
            SearchLimits {
                budget: f64::INFINITY,
                prune_with_bound: false,
            },
        )
        .cost
        .expect("the frozen lower graph connects each target");
        let edges = |node: usize| {
            adjacency[node].iter().filter_map(move |edge| {
                (!edge.diagonal || diagonal).then_some((edge.to, edge.upper))
            })
        };
        let unlimited = search(
            NODES,
            start,
            accepts,
            edges,
            |_| None,
            |_| true,
            SearchLimits {
                budget: f64::INFINITY,
                prune_with_bound: false,
            },
        );
        let upper = unlimited
            .cost
            .expect("the frozen fixture connects each target");
        assert!(lower <= upper);
        let mut caps = Vec::new();
        for budget in [f64::INFINITY, upper, upper - 1.0 / QUANTA, 0.0] {
            let radius_upper = region.budget_radius_upper(budget);
            let envelope = |node| {
                radius_upper
                    .is_none_or(|r| displacement_lower(point(start), point(node)).unwrap() <= r)
            };
            let baseline = search(
                NODES,
                start,
                accepts,
                edges,
                |_| None,
                |_| true,
                SearchLimits {
                    budget,
                    prune_with_bound: false,
                },
            );
            let bounded = search(
                NODES,
                start,
                accepts,
                edges,
                heuristic,
                envelope,
                SearchLimits {
                    budget,
                    prune_with_bound: true,
                },
            );
            assert_eq!(baseline.cost, bounded.cost);
            totals[0].record(baseline.counts);
            totals[1].record(bounded.counts);
            let upper_witness = bounded.cost.is_some();
            let excluded_by_lower = lower > budget;
            assert!(!(upper_witness && excluded_by_lower));
            caps.push(json!({"budget": budget.is_finite().then_some(budget),
                "upperCostWitness": upper_witness, "lowerCostExcludes": excluded_by_lower}));
        }
        outcomes.push(json!({"start": point(start), "targetCenter": point(center), "radiusSteps": radius_steps,
            "lower": lower, "upper": upper, "budgets": caps}));
    }
    json!({"diagonals": diagonal, "cases": outcomes,
        "dijkstraExpansions": totals[0].expanded, "boundedExpansions": totals[1].expanded,
        "boundedPrunes": totals[1].pruned})
}

fn main() -> Result<(), String> {
    let asset = std::env::args_os()
        .nth(1)
        .ok_or("supply the retained fine FSOP0001 asset")?;
    let bytes = std::fs::read(asset).map_err(|error| error.to_string())?;
    let field = PreparedField::from_asset_bytes(&bytes)?;
    if field.identity() != 0x9721bb7647a0b8e6 {
        return Err("the fixture expects the published retained fine field identity".into());
    }
    let config = ManifoldConfig::default();
    let region = field
        .navigation_region([point(0), point(NODES - 1)], &config)
        .ok_or("window is outside supported retained coefficients")?;
    let (adjacency, edges) = graph(&region, &field, &config)?;
    let start = point(40 * WIDTH + 2);
    let target = point(40 * WIDTH + 87);
    let length = region.target_lower_bound(start, target, 0.0).unwrap();
    let budgets = FeasibilityLimits {
        kinetic_action_budget: Some(0.05),
        ..Default::default()
    };
    let second = CANONICAL_SAMPLE_RATE as u64;
    let short = evaluate_feasibility(LengthLowerBound::Certified(length), second, budgets);
    let long = evaluate_feasibility(LengthLowerBound::Certified(length), 100 * second, budgets);
    assert_eq!(short.status, FeasibilityStatus::RuledOut);
    assert_eq!(long.status, FeasibilityStatus::InsufficientEvidence);
    println!(
        "{}",
        json!({"fieldIdentity": format!("{:016x}", field.identity()),
            "metricScope": "exactPreparedSplineInWindow",
            "window": [point(0), point(NODES - 1)], "shape": [WIDTH, HEIGHT], "edges": edges,
            "nominalWindowFeasibility": {"start": start, "target": target, "actionBudget": 0.05,
                "oneSecond": short, "hundredSeconds": long},
        "searches": [compare(&region, &adjacency, false), compare(&region, &adjacency, true)]})
    );
    Ok(())
}
