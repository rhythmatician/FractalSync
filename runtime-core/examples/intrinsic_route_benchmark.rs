use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;

use runtime_core::intrinsic_bounds::{
    CrossingEvidence, EndpointBounds, IntrinsicMetric, ambient_target_lower_bound,
    point_lower_bound, search_envelope_radius,
};

use runtime_core::map_navigation::search::{Counts, SearchLimits, SearchResult, search};

const WIDTH: usize = 65;
const HEIGHT: usize = 33;
const NODES: usize = WIDTH * HEIGHT;
const COST_UNITS: f64 = 16_777_216.0;

#[derive(Clone, Copy, PartialEq, Eq)]
struct GridPoint {
    x: i32,
    y: i32,
}

impl GridPoint {
    const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn node(self) -> usize {
        ((self.y + 16) * WIDTH as i32 + self.x + 32) as usize
    }

    fn at(node: usize) -> Self {
        Self::new((node % WIDTH) as i32 - 32, (node / WIDTH) as i32 - 16)
    }

    fn steps_squared(self, other: Self) -> i32 {
        (self.x - other.x).pow(2) + (self.y - other.y).pow(2)
    }

    fn displacement_lower(self, other: Self, step: f64) -> f64 {
        // Squared displacements and the power-of-two rescaling are exact on this grid.
        let square = f64::from(self.steps_squared(other));
        (square.sqrt().next_down() * step).max(0.0)
    }

    fn bounds(self, metric: IntrinsicMetric) -> EndpointBounds {
        let distance = f64::from(self.y.abs()) * metric.epsilon / 4.0;
        EndpointBounds {
            shore_distance: [distance, distance],
            ruler_upper: (metric.epsilon * metric.epsilon + distance * distance)
                .sqrt()
                .next_up(),
        }
    }
}

#[derive(Clone, Copy)]
struct Target {
    center: GridPoint,
    radius_steps: i32,
}

impl Target {
    fn contains(self, point: GridPoint) -> bool {
        self.center.steps_squared(point) <= self.radius_steps.pow(2)
    }

    fn lower(self, metric: IntrinsicMetric, point: GridPoint) -> Option<f64> {
        let step = metric.epsilon / 4.0;
        let displacement = point.displacement_lower(self.center, step);
        let start = point.bounds(metric);
        let end = self.center.bounds(metric);
        if self.radius_steps != 0 {
            return ambient_target_lower_bound(
                metric,
                start,
                end,
                displacement,
                f64::from(self.radius_steps) * step,
            )
            .map(|bound| bound.lower);
        }
        let crossing = if point.y * self.center.y < 0 {
            CrossingEvidence::Certified
        } else {
            CrossingEvidence::Unknown
        };
        point_lower_bound(metric, start, end, displacement, crossing).map(|bound| bound.lower)
    }
}

#[derive(Clone, Copy)]
enum Mask {
    Clear,
    Detour,
    Cut,
}

impl Mask {
    fn permits(self, from: GridPoint, to: GridPoint) -> bool {
        if from.x.min(to.x) != 0 || from.x.max(to.x) != 1 {
            return true;
        }
        match self {
            Self::Clear => true,
            Self::Detour => from.y.abs() >= 13,
            Self::Cut => false,
        }
    }
}

struct FlatGraph {
    horizontal: [f64; HEIGHT],
    vertical: [f64; HEIGHT - 1],
    mask: Mask,
}

fn quantize_up(upper: f64) -> f64 {
    (upper * COST_UNITS).ceil() / COST_UNITS
}

impl FlatGraph {
    fn new(metric: IntrinsicMetric, mask: Mask) -> Self {
        let step = metric.epsilon / 4.0;
        let floor_squared = metric.epsilon * metric.epsilon;
        let horizontal = std::array::from_fn(|row| {
            let y = (row as i32 - 16) as f64 * step;
            let ruler_lower = (floor_squared + y * y).sqrt().next_down();
            quantize_up((step / ruler_lower).next_up())
        });
        let vertical = std::array::from_fn(|row| {
            let y = f64::from(row as i32 - 16) * step;
            let rectangle_width = step / 32.0;
            let mut upper = 0.0_f64;
            for substep in 0..32 {
                let left = (y + f64::from(substep) * rectangle_width).abs();
                let right = (y + f64::from(substep + 1) * rectangle_width).abs();
                let min = left.min(right);
                let max = left.max(right);
                // The floor, coefficient, y values and their products/sums are exact
                // for the two power-of-two floors and the small integer weights below.
                let numerator = (floor_squared + (1.0 + metric.a * metric.a) * (max * max))
                    .sqrt()
                    .next_up();
                let density_upper = (numerator / (floor_squared + min * min)).next_up();
                upper = (upper + (rectangle_width * density_upper).next_up()).next_up();
            }
            quantize_up(upper)
        });
        Self {
            horizontal,
            vertical,
            mask,
        }
    }

    fn edges(&self, node: usize) -> impl Iterator<Item = (usize, f64)> + use<'_> {
        let from = GridPoint::at(node);
        let row = (from.y + 16) as usize;
        let neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        neighbours.into_iter().filter_map(move |(dx, dy)| {
            let to = GridPoint::new(from.x + dx, from.y + dy);
            if to.x.abs() > 32 || to.y.abs() > 16 || !self.mask.permits(from, to) {
                return None;
            }
            let weight = if dy == 0 {
                self.horizontal[row]
            } else {
                self.vertical[row.min((to.y + 16) as usize)]
            };
            Some((to.node(), weight))
        })
    }
}

#[derive(Clone, Copy, PartialEq)]
enum SearchMode {
    Dijkstra,
    AStar,
    BoundedAStar,
}

const MODES: [(SearchMode, &str); 3] = [
    (SearchMode::Dijkstra, "Dijkstra"),
    (SearchMode::AStar, "A*"),
    (SearchMode::BoundedAStar, "bounded A*"),
];

#[derive(Clone, Copy)]
struct Query {
    group: usize,
    start: GridPoint,
    target: Target,
    mask: Mask,
    metric_available: bool,
}

fn solve(
    graph: &FlatGraph,
    metric: IntrinsicMetric,
    query: Query,
    mode: SearchMode,
    budget: f64,
) -> SearchResult {
    let certified = query.metric_available && mode != SearchMode::Dijkstra;
    let bounded = certified && mode == SearchMode::BoundedAStar;
    let radius = if bounded {
        search_envelope_radius(query.start.bounds(metric).ruler_upper, budget)
    } else {
        None
    };
    let queries = Cell::new(0);
    let mut result = search(
        NODES,
        query.start.node(),
        |node| query.target.contains(GridPoint::at(node)),
        |node| graph.edges(node),
        |node| {
            if !certified {
                return None;
            }
            queries.set(queries.get() + 1);
            query.target.lower(metric, GridPoint::at(node))
        },
        |node| {
            radius.is_none_or(|radius| {
                GridPoint::at(node).displacement_lower(query.start, metric.epsilon / 4.0) <= radius
            })
        },
        SearchLimits {
            budget,
            prune_with_bound: bounded,
        },
    );
    // The search fills its cache only once at each inspected vertex. The counter below is
    // recorded by measuring calls to the provided geometry function rather than cache hits.
    result.counts.bound_queries = queries.get();
    result
}

const GROUPS: [&str; 7] = [
    "same side",
    "cross Shore",
    "on Shore",
    "arrival ball",
    "masked detour",
    "disconnected",
    "missing certificate",
];

fn query(group: usize, start: [i32; 2], center: [i32; 2], radius_steps: i32) -> Query {
    Query {
        group,
        start: GridPoint::new(start[0], start[1]),
        target: Target {
            center: GridPoint::new(center[0], center[1]),
            radius_steps,
        },
        mask: Mask::Clear,
        metric_available: true,
    }
}

fn held_out_queries() -> [Query; 21] {
    [
        query(0, [-30, 3], [24, 5], 0),
        query(0, [27, 15], [-28, 9], 0),
        query(0, [15, -3], [-4, -2], 0),
        query(0, [29, 1], [30, 13], 0),
        query(1, [-24, -3], [25, 4], 0),
        query(1, [18, 12], [-15, -11], 0),
        query(1, [-9, -15], [-9, 15], 0),
        query(1, [7, 1], [8, -1], 0),
        query(2, [2, 0], [2, 0], 0),
        query(2, [-9, 0], [3, 0], 0),
        query(2, [19, 0], [6, 0], 0),
        query(2, [-31, 0], [28, 0], 0),
        query(2, [-5, 0], [1, 0], 0),
        query(3, [-3, -5], [1, -3], 1),
        query(3, [28, -14], [-17, 6], 6),
        query(3, [21, 3], [-6, 0], 3),
        query(3, [4, 2], [3, 0], 4),
        Query {
            mask: Mask::Detour,
            ..query(4, [0, 0], [1, 0], 0)
        },
        Query {
            mask: Mask::Detour,
            ..query(4, [-16, -2], [16, 2], 3)
        },
        Query {
            mask: Mask::Cut,
            ..query(5, [-3, 1], [7, 2], 2)
        },
        Query {
            mask: Mask::Detour,
            metric_available: false,
            ..query(6, [-2, 2], [4, -1], 0)
        },
    ]
}

fn validate_reopening() {
    // The heuristic is admissible, but h(2) > w(2, 1) + h(1).
    let edges = [
        vec![(1, 2.0), (2, 1.0)],
        vec![(3, 2.0)],
        vec![(1, 0.5)],
        vec![],
    ];
    let result = search(
        edges.len(),
        0,
        |node| node == 3,
        |node| edges[node].iter().copied(),
        |node| Some([0.0, 0.0, 2.0, 0.0][node]),
        |_| true,
        SearchLimits {
            budget: f64::INFINITY,
            prune_with_bound: false,
        },
    );
    assert_eq!(result.cost, Some(3.5));
    assert_eq!(result.counts.reopened, 1);
}

fn validate_underestimated_edge() {
    let metric = IntrinsicMetric {
        epsilon: 1.0,
        a: 0.0,
    };
    let query = query(2, [0, 0], [2, 0], 0);
    let start = query.start.node();
    let goal = query.target.center.node();
    let graph = FlatGraph::new(metric, Mask::Clear);
    let shortcut = 0.25;
    let run = |bounded| {
        search(
            NODES,
            start,
            |node| node == goal,
            |node| {
                let invalid_edge = (node == start).then_some((goal, shortcut));
                graph.edges(node).chain(invalid_edge)
            },
            |node| {
                bounded
                    .then(|| query.target.lower(metric, GridPoint::at(node)))
                    .flatten()
            },
            |_| true,
            SearchLimits {
                budget: shortcut,
                prune_with_bound: bounded,
            },
        )
    };
    // The edge directly on the Shore costs 0.5 along the represented segment.
    // A 0.25 edge also violates the ideal point bound. This input must never be
    // presented to the planner as a certified graph weight.
    assert_eq!(run(false).cost, Some(shortcut));
    assert_eq!(run(true).cost, None);
}

fn validate_envelope(graph: &FlatGraph, metric: IntrinsicMetric) -> usize {
    let start = GridPoint::new(-3, -1);
    let step = metric.epsilon / 4.0;
    let mut excluded_cells = 0;
    for budget in [0.0, 1.0 / 1024.0, 0.5, 3.0] {
        let radius = search_envelope_radius(start.bounds(metric).ruler_upper, budget)
            .expect("small fixture budgets have finite envelopes");
        for node in 0..NODES {
            let point = GridPoint::at(node);
            let (left_y, right_y) = (start.y.min(point.y), start.y.max(point.y));
            let vertical = ((left_y + 16)..(right_y + 16))
                .map(|row| graph.vertical[row as usize])
                .sum::<f64>();
            let horizontal =
                f64::from((point.x - start.x).abs()) * graph.horizontal[(start.y + 16) as usize];
            // A constructive prefix follows the start row, then the destination
            // column. Its exact dyadic cost upper-bounds that allowed path.
            let prefix_cost = horizontal + vertical;
            // Minimum displacement to the closed square cell, including its footprint.
            let dx = (2 * (point.x - start.x).abs() - 1).max(0);
            let dy = (2 * (point.y - start.y).abs() - 1).max(0);
            let footprint_lower =
                (f64::from(dx * dx + dy * dy).sqrt().next_down() * (step / 2.0)).max(0.0);
            if prefix_cost <= budget {
                assert!(point.displacement_lower(start, step) <= radius);
                assert!(footprint_lower <= radius);
            }
            excluded_cells += usize::from(footprint_lower > radius);
        }
    }
    excluded_cells
}

fn validate_radial_clearance_limit() {
    // With a=0, a fixed vertical segment from y=1 to y=2 attains the
    // radial lower bound for every positive floor. At epsilon=0,
    // its length is ln(2) and its minimum clearance is 1.
    let zero_floor_length = 2.0_f64.ln();
    assert!(1.0 >= (-zero_floor_length).exp());
    let mut previous = 0.0;
    for epsilon in [1.0, 0.5, 0.25, 0.125] {
        let metric = IntrinsicMetric { epsilon, a: 0.0 };
        let start = EndpointBounds::from_shore_interval(metric, [1.0; 2]).unwrap();
        let end = EndpointBounds::from_shore_interval(metric, [2.0; 2]).unwrap();
        let bound = point_lower_bound(metric, start, end, 1.0, CrossingEvidence::Unknown).unwrap();
        let reference = (2.0_f64 / epsilon).asinh() - (1.0_f64 / epsilon).asinh();
        assert!(previous < reference && reference < zero_floor_length);
        assert!(bound.lower <= reference);
        previous = reference;
    }
}

fn main() {
    validate_reopening();
    validate_radial_clearance_limit();
    validate_underestimated_edge();
    let mut counts = [[Counts::default(); MODES.len()]; GROUPS.len()];
    let mut timings = [std::time::Duration::ZERO; MODES.len()];
    let mut accepted = [0; GROUPS.len()];
    let mut searches = [0; GROUPS.len()];
    let mut prunes = 0;
    let mut excluded_cells = 0;
    for epsilon in [0.5, 1.0] {
        for a in [0.0, 1.0, 3.0] {
            let metric = IntrinsicMetric { epsilon, a };
            let graphs =
                [Mask::Clear, Mask::Detour, Mask::Cut].map(|mask| FlatGraph::new(metric, mask));
            excluded_cells += validate_envelope(&graphs[0], metric);
            for query in held_out_queries() {
                let graph = &graphs[query.mask as usize];
                let baseline = solve(graph, metric, query, SearchMode::Dijkstra, f64::INFINITY);
                assert_eq!(baseline.cost.is_none(), query.group == 5);
                if query.group == 4 {
                    // The short direct segment is masked in the first detour, and every
                    // route through the wall must go above/below the local window.
                    assert!(baseline.cost.unwrap() > 1.0);
                }
                let mut budgets = vec![f64::INFINITY, 0.0, 3.0];
                if let Some(cost) = baseline.cost.filter(|&cost| cost > 0.0) {
                    budgets.extend([cost - 1.0 / COST_UNITS, cost]);
                }
                for budget in budgets {
                    let expected = baseline.cost.filter(|cost| *cost <= budget);
                    let mut expansion_count = 0;
                    searches[query.group] += 1;
                    accepted[query.group] += usize::from(expected.is_some());
                    for (index, (mode, _)) in MODES.into_iter().enumerate() {
                        let start = Instant::now();
                        let result = black_box(solve(graph, metric, query, mode, budget));
                        timings[index] += start.elapsed();
                        assert_eq!(
                            result.cost,
                            expected,
                            "group {}, ({},{}) to ({},{}), epsilon={epsilon}, a={a}, budget={budget}",
                            GROUPS[query.group],
                            query.start.x,
                            query.start.y,
                            query.target.center.x,
                            query.target.center.y,
                        );
                        if mode == SearchMode::Dijkstra {
                            expansion_count = result.counts.expanded;
                        }
                        if !query.metric_available {
                            assert_eq!(result.counts.expanded, expansion_count);
                            assert_eq!(result.counts.pruned, 0);
                            assert_eq!(result.counts.bound_queries, 0);
                        }
                        prunes += result.counts.pruned;
                        counts[query.group][index].record(result.counts);
                    }
                }
            }
        }
    }
    assert!(prunes > 0, "benchmark must exercise certified pruning");
    println!("Envelope excluded {excluded_cells} cell footprints across 24 budget checks");
    println!("case,searches,accepted,algorithm,expansions,reopenings,bound queries,pruned");
    for (group, label) in GROUPS.into_iter().enumerate() {
        for (index, (_, mode)) in MODES.into_iter().enumerate() {
            let row = counts[group][index];
            println!(
                "{label},{},{},{mode},{},{},{},{}",
                searches[group],
                accepted[group],
                row.expanded,
                row.reopened,
                row.bound_queries,
                row.pruned
            );
        }
    }
    for (index, (_, mode)) in MODES.into_iter().enumerate() {
        println!(
            "{mode} search time {:.3} ms",
            timings[index].as_secs_f64() * 1_000.0
        );
    }
}
