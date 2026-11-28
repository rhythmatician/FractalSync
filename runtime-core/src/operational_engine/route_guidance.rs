//! Read-only route guidance in the nominal geometry of a frozen prepared field.

use super::frame_transport::MetricFactor;
use super::{AcceptedStepTrace, CANONICAL_REFERENCE_POINTS, FrozenRollout, MechanicalState};
use crate::crossing_feasibility::{
    FeasibilityAssessment, FeasibilityLimits, LengthLowerBound, evaluate_feasibility,
};
use crate::intrinsic_bounds::{add_down, add_up, mul_down, mul_up};
use crate::map_certificates::displacement_lower;
use crate::map_navigation::search::{SearchLimits, search};
use crate::operational_field::{
    OperationalLengthInterval, OperationalNavigationRegion, SegmentEvaluator,
};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

const MAX_GRID_STEPS: usize = 64;
const COST_UNITS: f64 = (1_u32 << 24) as f64;
const NEIGHBORS: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AmbientTarget {
    pub center: [f64; 2],
    pub radius: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OperationalRouteRequest {
    /// Lower and upper coordinate corners of the route window.
    pub window: [[f64; 2]; 2],
    pub grid_steps: usize,
    pub target: AmbientTarget,
    pub length_budget: Option<f64>,
    /// Necessary conditions for a timed task confined to the nominal window.
    pub nominal_timing: Option<NominalRouteTiming>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NominalRouteTiming {
    pub start_sample: u64,
    pub deadline_sample: u64,
    pub kinetic_action_budget: Option<f64>,
}

impl NominalRouteTiming {
    fn assess(self, length: f64) -> Result<FeasibilityAssessment, &'static str> {
        let duration = self
            .deadline_sample
            .checked_sub(self.start_sample)
            .ok_or("deadline sample precedes route start sample")?;
        if self
            .kinetic_action_budget
            .is_some_and(|b| !b.is_finite() || b < 0.0)
        {
            return Err("route kinetic action budget must be finite and nonnegative");
        }
        Ok(evaluate_feasibility(
            LengthLowerBound::Certified(length),
            duration,
            FeasibilityLimits {
                kinetic_action_budget: self.kinetic_action_budget,
                ..Default::default()
            },
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum OperationalRouteStatus {
    Reached,
    RouteFound,
    NoCertifiedRoute,
    GraphBudgetExcluded,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationalRouteGuidance {
    pub metric_scope: &'static str,
    pub field_id: String,
    pub physics_profile_id: String,
    pub target: AmbientTarget,
    pub window: [[f64; 2]; 2],
    pub status: OperationalRouteStatus,
    pub nominal_window_length_lower: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nominal_feasibility: Option<FeasibilityAssessment>,
    /// Lower bound on the full admitted graph, including paths outside the
    /// budgeted subset. A budgeted query may report a weaker lower bound.
    pub graph_length_lower: Option<f64>,
    pub route_length_upper: Option<f64>,
    pub path: Vec<[f64; 2]>,
    pub next_direction: Option<[f64; 2]>,
    pub certified_edges: usize,
    pub refused_edges: usize,
    /// Candidate edges skipped before checking reference support or spline bounds.
    pub budget_skipped_edges: usize,
    /// Edge enclosure attempts after the budget and reference checks.
    pub edge_certificate_queries: usize,
    /// Distinct spline cells inspected by the per-request evaluator.
    pub cell_certificate_queries: usize,
    pub expanded_upper: usize,
}

/// Bounds for reconstructed chords and nominal left-endpoint kinetic action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NominalChordAction {
    pub length: OperationalLengthInterval,
    pub missed_length_upper: f64,
    pub sampled_action_lower: f64,
}

impl FrozenRollout {
    /// Bound the nominal left-endpoint sampled action on an accepted RK trace.
    /// Time weights are the represented RK substep durations. The length
    /// allowance covers the reconstructed polygon in the nominal spline
    /// metric. The compiled energy ledger and continuous ODE path have
    /// separate correspondence requirements.
    pub fn nominal_accepted_chord_action(
        &self,
        trace: &AcceptedStepTrace,
        window: [[f64; 2]; 2],
    ) -> Option<NominalChordAction> {
        self.check_trace_identity(trace).ok()?;
        let steps = trace.rk_substeps();
        if steps.is_empty() || steps.len() + 1 != trace.endpoints().len() {
            return None;
        }
        let region = self
            .field
            .navigation_region(window, self.profile.config())?;
        let mut evaluator = region.segment_evaluator();
        let mut length = OperationalLengthInterval {
            lower: 0.0,
            upper: 0.0,
        };
        let mut missed = 0.0;
        let mut action = 0.0;
        for (pair, step) in trace.endpoints().windows(2).zip(steps) {
            let [a, b] = [pair[0].position, pair[1].position];
            let h = step.duration;
            if !h.is_finite() || h <= 0.0 || !self.chart_edge_is_clear(chord_bounds(a, b)) {
                return None;
            }
            let chord = evaluator.bounds(a, b)?;
            let speed = region.nominal_speed_lower(pair[0].velocity)?;
            let drift = mul_down(h, speed);
            let excess = if chord.upper > drift {
                (chord.upper - drift).next_up()
            } else {
                0.0
            };
            length.lower = add_down(length.lower, chord.lower);
            length.upper = add_up(length.upper, chord.upper);
            missed = add_up(missed, excess);
            action = add_down(action, mul_down(drift, mul_down(0.5, speed)));
        }
        if ![length.upper, missed, action]
            .into_iter()
            .all(f64::is_finite)
        {
            return None;
        }
        Some(NominalChordAction {
            length,
            missed_length_upper: missed,
            sampled_action_lower: action,
        })
    }

    /// Enclose the reconstructed polygon through accepted endpoints in the
    /// nominal exact-real spline metric on the supplied window. Each chord
    /// must pass the same chart and reference checks as a route edge.
    pub fn nominal_accepted_polyline_length(
        &self,
        trace: &AcceptedStepTrace,
        window: [[f64; 2]; 2],
    ) -> Option<OperationalLengthInterval> {
        self.check_trace_identity(trace).ok()?;
        let region = self
            .field
            .navigation_region(window, self.profile.config())?;
        let mut evaluator = region.segment_evaluator();
        trace.endpoints().windows(2).try_fold(
            OperationalLengthInterval {
                lower: 0.0,
                upper: 0.0,
            },
            |total, endpoints| {
                let [a, b] = [endpoints[0].position, endpoints[1].position];
                if !self.chart_edge_is_clear(chord_bounds(a, b)) {
                    return None;
                }
                let length = evaluator.bounds(a, b)?;
                let lower = add_down(total.lower, length.lower);
                let upper = add_up(total.upper, length.upper);
                upper
                    .is_finite()
                    .then_some(OperationalLengthInterval { lower, upper })
            },
        )
    }

    /// Give geometric guidance from the current state. The length bracket and
    /// budget status concern the admitted window graph in the exact prepared
    /// spline metric. The bearing uses the compiled metric at the start.
    pub fn route_guidance(
        &self,
        state: &MechanicalState,
        request: OperationalRouteRequest,
    ) -> Result<OperationalRouteGuidance, String> {
        if state.field_id != self.field_id() || state.physics_profile_id != self.profile().id() {
            return Err("FrozenIdentity: state belongs to another field/profile".into());
        }
        let start = state.position;
        let size = request.grid_steps;
        let target = request.target;
        if !(1..=MAX_GRID_STEPS).contains(&size)
            || !in_box(start, request.window)
            || !in_box(target.center, request.window)
            || !target.radius.is_finite()
            || target.radius < 0.0
            || request
                .length_budget
                .is_some_and(|b| !b.is_finite() || b < 0.0)
        {
            return Err("invalid Map route window, target, grid or budget".into());
        }
        let [lo, hi] = request.window;
        let absolute = std::array::from_fn::<_, 2, _>(|i| lo[i].abs().max(hi[i].abs()));
        if add_up(
            mul_up(absolute[0], absolute[0]),
            mul_up(absolute[1], absolute[1]),
        ) >= 4.0
        {
            return Err("Map route window exceeds the mechanical disk".into());
        }
        let x = axis(lo[0], hi[0], size, self.field.spacing())?;
        let y = axis(lo[1], hi[1], size, self.field.spacing())?;
        let region = self
            .field
            .navigation_region(request.window, self.profile.config())
            .ok_or("Map route window has no supported spline range")?;
        let ambient_lower = region
            .target_lower_bound(start, target.center, target.radius)
            .ok_or("invalid Map route target length bound")?;
        let nominal_feasibility = request
            .nominal_timing
            .map(|timing| timing.assess(ambient_lower))
            .transpose()?;
        let previous_work = self.last_work.get();
        let metric_query = self.evaluate(|config| {
            let c = Complex64::new(start[0], start[1]);
            MetricFactor::new(crate::manifold::induced_metric(c, config)?)
        });
        self.last_work.set(previous_work);
        let (metric, _) = metric_query?;
        let mut report = OperationalRouteGuidance {
            metric_scope: "exactPreparedSplineInWindow",
            field_id: format!("{:016x}", self.field_id()),
            physics_profile_id: format!("{:016x}", self.profile().id()),
            target,
            window: request.window,
            status: OperationalRouteStatus::NoCertifiedRoute,
            nominal_window_length_lower: ambient_lower,
            nominal_feasibility,
            graph_length_lower: None,
            route_length_upper: None,
            path: Vec::new(),
            next_direction: None,
            certified_edges: 0,
            refused_edges: 0,
            budget_skipped_edges: 0,
            edge_certificate_queries: 0,
            cell_certificate_queries: 0,
            expanded_upper: 0,
        };
        if target_relation(start, target) == TargetRelation::Inside {
            report.status = OperationalRouteStatus::Reached;
            report.graph_length_lower = Some(0.0);
            report.route_length_upper = Some(0.0);
            report.path.push(start);
            return Ok(report);
        }

        let graph = RouteGraph::new(self, &region, &x, &y, start, target, request.length_budget)?;
        report.certified_edges = graph.certified;
        report.refused_edges = graph.refused;
        report.budget_skipped_edges = graph.budget_skipped;
        report.edge_certificate_queries = graph.edge_queries;
        report.cell_certificate_queries = graph.cell_queries;
        let outcomes: Vec<_> = graph
            .points
            .iter()
            .map(|&p| target_relation(p, target))
            .collect();
        report.graph_length_lower = search(
            graph.points.len(),
            graph.start,
            |i| outcomes[i] != TargetRelation::Outside,
            |i| graph.edges[i].iter().map(|&(j, b)| (j, b.lower)),
            |_| None,
            |_| true,
            SearchLimits {
                budget: f64::INFINITY,
                prune_with_bound: false,
            },
        )
        .cost;

        let budget = request.length_budget.unwrap_or(f64::INFINITY);
        let budget_excluded = ambient_lower > budget
            || report
                .graph_length_lower
                .is_some_and(|lower| lower > budget);
        // A skipped path costs more than the budget, but can still cost less
        // than an admitted detour. Keep the lower bound valid for both.
        if graph.budget_skipped > 0 {
            report.graph_length_lower =
                Some(report.graph_length_lower.unwrap_or(budget).min(budget));
        }
        if budget_excluded {
            report.status = OperationalRouteStatus::GraphBudgetExcluded;
            return Ok(report);
        }

        let radius_upper = region.budget_radius_upper(budget);
        let upper = search(
            graph.points.len(),
            graph.start,
            |i| outcomes[i] == TargetRelation::Inside,
            |i| graph.edges[i].iter().map(|&(j, b)| (j, b.upper)),
            |i| region.target_lower_bound(graph.points[i], target.center, target.radius),
            |i| {
                radius_upper.is_none_or(|r| {
                    displacement_lower(start, graph.points[i]).is_some_and(|d| d <= r)
                })
            },
            SearchLimits {
                budget,
                prune_with_bound: true,
            },
        );
        report.expanded_upper = upper.counts.expanded;
        if let Some(upper_cost) = upper.cost {
            report.status = OperationalRouteStatus::RouteFound;
            report.route_length_upper = Some(upper_cost);
            report.path = upper.path.into_iter().map(|i| graph.points[i]).collect();
            report.path.dedup();
            let next = report
                .path
                .get(1)
                .ok_or("Map route has no departure point")?;
            report.next_direction = Some(metric.unit([next[0] - start[0], next[1] - start[1]])?);
        }
        Ok(report)
    }

    pub(super) fn chart_edge_is_clear(&self, [lo, hi]: [[f64; 2]; 2]) -> bool {
        let avoids_canonical = !CANONICAL_REFERENCE_POINTS
            .iter()
            .any(|&p| in_box(p, [lo, hi]));
        avoids_canonical
            && self.cuts.iter().all(|cut| {
                let [xmin, ymin, xmax, ymax] = cut.bounds;
                hi[0] < xmin || xmax < lo[0] || hi[1] < ymin || ymax < lo[1]
            })
    }
}

struct RouteGraph {
    points: Vec<[f64; 2]>,
    edges: Vec<Vec<(usize, OperationalLengthInterval)>>,
    start: usize,
    certified: usize,
    refused: usize,
    budget_skipped: usize,
    edge_queries: usize,
    cell_queries: usize,
    allowed: Vec<bool>,
}

impl RouteGraph {
    fn new(
        engine: &FrozenRollout,
        region: &OperationalNavigationRegion<'_>,
        x: &[f64],
        y: &[f64],
        start: [f64; 2],
        target: AmbientTarget,
        budget: Option<f64>,
    ) -> Result<Self, String> {
        let index = |i, j| j * x.len() + i;
        let mut points: Vec<_> = y
            .iter()
            .flat_map(|&py| x.iter().map(move |&px| [px, py]))
            .collect();
        let start_index = points.len();
        points.extend([start, target.center]);
        let allowed = points
            .iter()
            .map(|&p| {
                budget.is_none_or(|cap| {
                    let to_vertex = region.target_lower_bound(start, p, 0.0).unwrap_or(0.0);
                    let from_vertex = region
                        .target_lower_bound(p, target.center, target.radius)
                        .unwrap_or(0.0);
                    add_down(to_vertex, from_vertex) <= cap
                })
            })
            .collect();
        let limit = (((1_u64 << 53) - 1) / (points.len() as u64 + 1)) as f64;
        let mut graph = Self {
            edges: vec![Vec::new(); points.len()],
            points,
            start: start_index,
            certified: 0,
            refused: 0,
            budget_skipped: 0,
            edge_queries: 0,
            cell_queries: 0,
            allowed,
        };
        let mut segment = region.segment_evaluator();
        for j in 0..y.len() {
            for i in 0..x.len() {
                for (dx, dy) in NEIGHBORS {
                    let Some(ix) = i.checked_add_signed(dx).filter(|&v| v < x.len()) else {
                        continue;
                    };
                    let Some(iy) = j.checked_add_signed(dy).filter(|&v| v < y.len()) else {
                        continue;
                    };
                    graph.add_edge(engine, &mut segment, index(i, j), index(ix, iy), limit)?;
                }
            }
        }
        for (offset, p) in [start, target.center].into_iter().enumerate() {
            let ix = x.partition_point(|&v| v <= p[0]).clamp(1, x.len() - 1);
            let iy = y.partition_point(|&v| v <= p[1]).clamp(1, y.len() - 1);
            for j in (iy - 1)..=iy {
                for i in (ix - 1)..=ix {
                    graph.add_edge(
                        engine,
                        &mut segment,
                        start_index + offset,
                        index(i, j),
                        limit,
                    )?;
                }
            }
        }
        graph.cell_queries = segment.cell_queries();
        Ok(graph)
    }

    fn add_edge(
        &mut self,
        engine: &FrozenRollout,
        segment: &mut SegmentEvaluator<'_, '_>,
        from: usize,
        to: usize,
        limit: f64,
    ) -> Result<(), String> {
        if !self.allowed[from] || !self.allowed[to] {
            self.budget_skipped += 1;
            return Ok(());
        }
        let a = self.points[from];
        let b = self.points[to];
        let bbox = chord_bounds(a, b);
        let length = engine
            .chart_edge_is_clear(bbox)
            .then(|| {
                self.edge_queries += 1;
                segment.bounds(a, b)
            })
            .flatten();
        let Some(length) = length else {
            self.refused += 1;
            return Ok(());
        };
        let upper_units = (length.upper * COST_UNITS).ceil();
        if upper_units > limit {
            return Err("Map route cost exceeds the exact graph sum range".into());
        }
        let length = OperationalLengthInterval {
            lower: (length.lower * COST_UNITS).floor() / COST_UNITS,
            upper: upper_units / COST_UNITS,
        };
        self.edges[from].push((to, length));
        self.edges[to].push((from, length));
        self.certified += 1;
        Ok(())
    }
}

fn axis(lo: f64, hi: f64, steps: usize, spacing: f64) -> Result<Vec<f64>, String> {
    let mut nodes: Vec<_> = (0..steps)
        .map(|i| lo + (hi - lo) * (i as f64 / steps as f64))
        .collect();
    nodes.push(hi);
    if nodes
        .windows(2)
        .any(|pair| pair[1] <= pair[0] || pair[1] - pair[0] > 2.0 * spacing)
    {
        return Err(
            "Map route grid has indistinct nodes or edges wider than two spline cells".into(),
        );
    }
    Ok(nodes)
}

fn in_box([x, y]: [f64; 2], [[xmin, ymin], [xmax, ymax]]: [[f64; 2]; 2]) -> bool {
    x >= xmin && x <= xmax && y >= ymin && y <= ymax
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TargetRelation {
    Inside,
    BoundaryUncertain,
    Outside,
}

fn target_relation(point: [f64; 2], target: AmbientTarget) -> TargetRelation {
    if point == target.center {
        return TargetRelation::Inside;
    }
    if target.radius == 0.0
        || displacement_lower(point, target.center).is_some_and(|d| d > target.radius)
    {
        return TargetRelation::Outside;
    }
    let [x, y] = std::array::from_fn::<_, 2, _>(|i| {
        if point[i] == target.center[i] {
            0.0
        } else {
            (point[i] - target.center[i]).abs().next_up()
        }
    });
    let distance_upper = add_up(mul_up(x, x), mul_up(y, y)).sqrt().next_up();
    if distance_upper <= target.radius {
        TargetRelation::Inside
    } else {
        TargetRelation::BoundaryUncertain
    }
}

fn chord_bounds(a: [f64; 2], b: [f64; 2]) -> [[f64; 2]; 2] {
    [
        [a[0].min(b[0]), a[1].min(b[1])],
        [a[0].max(b[0]), a[1].max(b[1])],
    ]
}

#[cfg(test)]
mod tests;
