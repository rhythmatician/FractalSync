//! Displacement bounds for the default raster-bridge metric on a fixed window.

use super::{raster_pixel, raster_stamp, DistanceField, DIST_FIELD};
use crate::geometry_provider::{
    default_query_uses_raster_bridge, GEOMETRY_MAX_STEP, GEOMETRY_MIN_STEP, GEOMETRY_SCALE_ALPHA,
};
use crate::intrinsic_bounds::{add_up, div_down, div_up, mul_up, sub_down};
use crate::manifold::ManifoldConfig;
use crate::map_certificates::displacement_lower;
use crate::map_navigation::search::{search_with_sum, SearchLimits, SearchResult};

/// A vertex in a directed graph of straight Map segments.
pub struct RasterWaypoint<'a> {
    pub position: [f64; 2],
    pub neighbors: &'a [usize],
}

/// One closed ambient ball in a raster route's accepted target set.
#[derive(Clone, Copy, Debug)]
pub struct RasterTargetBall {
    pub center: [f64; 2],
    pub radius: f64,
}

#[derive(Debug, PartialEq)]
pub struct RasterRouteWitness {
    /// Waypoint indices in traversal order, including start and arrival.
    pub path: Vec<usize>,
    pub length_upper: f64,
    pub raster_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RasterRouteError {
    InvalidInput,
    StaleRegion,
}

/// A full-support window for the default profile's raster-bridge metric.
///
/// The metric uses the pointwise binary64 matrix entries returned by Physics,
/// interpreted as real coefficients. A trajectory using this evidence must
/// use the same loaded generation and default profile throughout the prefix
/// being bounded.
#[derive(Clone, Copy, Debug)]
pub struct RasterNavigationRegion {
    bounds: [[f64; 2]; 2],
    raster_generation: u64,
    rho_upper: f64,
    length_per_displacement_upper: f64,
}

impl RasterNavigationRegion {
    /// Certify a closed Map rectangle `[lower, upper]` in the loaded raster.
    /// The provider's center and adaptive stencil must have complete bicubic
    /// support. This read does not install or switch a provider.
    pub fn for_default_profile(bounds: [[f64; 2]; 2]) -> Option<Self> {
        if !default_query_uses_raster_bridge() || !in_hard_disk(bounds) {
            return None;
        }
        let config = ManifoldConfig::default();
        if !(1e-8..=1.0).contains(&config.epsilon) || config.lambda_sq != 1.0 {
            return None;
        }
        let slot = DIST_FIELD.read().ok()?;
        let field = slot.field.as_ref()?;
        let d_upper = sample_magnitude_upper(field, bounds)?;
        let rho_upper = add_up(
            mul_up(d_upper, d_upper),
            mul_up(config.epsilon, config.epsilon),
        )
        .sqrt()
        .next_up();
        let step_lower =
            (GEOMETRY_SCALE_ALPHA * config.epsilon).clamp(GEOMETRY_MIN_STEP, GEOMETRY_MAX_STEP);
        let step_upper =
            (GEOMETRY_SCALE_ALPHA * rho_upper).clamp(GEOMETRY_MIN_STEP, GEOMETRY_MAX_STEP);
        let [lo, hi] = bounds;
        let stencil_bounds = [
            lo.map(|v| (v - step_upper).next_down()),
            hi.map(|v| (v + step_upper).next_up()),
        ];
        let stencil_upper = sample_magnitude_upper(field, stencil_bounds)?;
        // The two-point difference is at most 2 * stencil_upper. The
        // gradient limit leaves room for rounding in the metric's rank-one term.
        let gradient_upper = div_up(stencil_upper, step_lower);
        if gradient_upper > f64::from(1 << 18) {
            return None;
        }
        Some(Self {
            bounds,
            raster_generation: slot.generation,
            rho_upper,
            length_per_displacement_upper: metric_length_factor(
                (config.epsilon * config.epsilon).sqrt(),
                d_upper,
                gradient_upper,
            ),
        })
    }

    pub fn raster_generation(&self) -> u64 {
        self.raster_generation
    }

    pub fn rho_upper(&self) -> f64 {
        self.rho_upper
    }

    /// Whether the raster is still installed and selected for queries on
    /// this thread. A caller using a trajectory must establish uninterrupted
    /// provider and profile identity for the relevant prefix.
    pub fn is_current(&self) -> bool {
        default_query_uses_raster_bridge()
            && raster_stamp()
                .is_some_and(|stamp| stamp.loaded && stamp.generation == self.raster_generation)
    }

    /// Outward upper bound on the straight segment's length in this region's
    /// default raster metric. Both endpoints must lie in the certified window.
    /// The caller must keep the provider and profile fixed during use.
    pub fn segment_length_upper(&self, start: [f64; 2], end: [f64; 2]) -> Option<f64> {
        if !default_query_uses_raster_bridge() || !self.contains(start) || !self.contains(end) {
            return None;
        }
        let slot = DIST_FIELD.read().ok()?;
        let field = slot
            .field
            .as_ref()
            .filter(|_| slot.generation == self.raster_generation)?;
        Some(self.segment_upper(field, start, end))
    }

    fn segment_upper(&self, field: &DistanceField, start: [f64; 2], end: [f64; 2]) -> f64 {
        let displacement = displacement_upper(start, end);
        if displacement == 0.0 {
            return 0.0;
        }
        let bounds = [
            std::array::from_fn(|i| start[i].min(end[i])),
            std::array::from_fn(|i| start[i].max(end[i])),
        ];
        let regional = self.length_per_displacement_upper;
        let factor =
            local_length_factor(field, bounds).map_or(regional, |local| local.min(regional));
        mul_up(displacement, factor)
    }

    /// Find a waypoint path whose certified length upper bound fits the budget.
    /// All waypoints must lie in this region and name valid neighbors.
    /// A waypoint is accepted when its outward displacement bound fits the
    /// ambient ball. Rounding can leave near-boundary waypoints unaccepted.
    /// A zero-radius ball accepts exact coordinate equality.
    /// An empty result rules out a witness under these graph tests only.
    pub fn search_route(
        &self,
        waypoints: &[RasterWaypoint<'_>],
        start: usize,
        center: [f64; 2],
        radius: f64,
        length_budget: f64,
    ) -> Result<Option<RasterRouteWitness>, RasterRouteError> {
        self.search_target_set(
            waypoints,
            start,
            &[RasterTargetBall { center, radius }],
            length_budget,
        )
    }

    /// Search for a waypoint in a nonempty union of closed ambient balls.
    /// Each component uses the same conservative arrival check as a single ball.
    pub fn search_target_set(
        &self,
        waypoints: &[RasterWaypoint<'_>],
        start: usize,
        targets: &[RasterTargetBall],
        length_budget: f64,
    ) -> Result<Option<RasterRouteWitness>, RasterRouteError> {
        let result = self.search_waypoints(waypoints, start, targets, (length_budget, true))?;
        Ok(result.cost.map(|length_upper| RasterRouteWitness {
            path: result.path,
            length_upper,
            raster_generation: self.raster_generation,
        }))
    }

    fn search_waypoints(
        &self,
        waypoints: &[RasterWaypoint<'_>],
        start: usize,
        targets: &[RasterTargetBall],
        (length_budget, use_bound): (f64, bool),
    ) -> Result<SearchResult, RasterRouteError> {
        if start >= waypoints.len()
            || targets.is_empty()
            || targets.iter().any(|target| {
                !target.center.into_iter().all(f64::is_finite)
                    || !target.radius.is_finite()
                    || target.radius < 0.0
            })
            || length_budget.is_nan()
            || length_budget < 0.0
            || waypoints.iter().any(|vertex| {
                !self.contains(vertex.position)
                    || vertex
                        .neighbors
                        .iter()
                        .any(|&index| index >= waypoints.len())
            })
        {
            return Err(RasterRouteError::InvalidInput);
        }
        if !default_query_uses_raster_bridge() {
            return Err(RasterRouteError::StaleRegion);
        }
        let slot = DIST_FIELD
            .read()
            .map_err(|_| RasterRouteError::StaleRegion)?;
        let field = slot
            .field
            .as_ref()
            .filter(|_| slot.generation == self.raster_generation)
            .ok_or(RasterRouteError::StaleRegion)?;
        let result = search_with_sum(
            waypoints.len(),
            start,
            |node| {
                targets.iter().any(|target| {
                    displacement_upper(waypoints[node].position, target.center) <= target.radius
                })
            },
            |node| {
                let vertex = &waypoints[node];
                vertex.neighbors.iter().map(move |&neighbor| {
                    (
                        neighbor,
                        self.segment_upper(field, vertex.position, waypoints[neighbor].position),
                    )
                })
            },
            |node| {
                // Both endpoints of each edge lie in the convex window. Graph
                // routes never need the first-exit allowance for continuous paths.
                use_bound.then(|| {
                    targets
                        .iter()
                        .map(|target| {
                            self.displacement_length_lower(
                                waypoints[node].position,
                                target.center,
                                target.radius,
                            )
                            .unwrap_or(0.0)
                        })
                        .fold(f64::INFINITY, f64::min)
                })
            },
            |_| true,
            (
                SearchLimits {
                    budget: length_budget,
                    prune_with_bound: use_bound,
                },
                add_up,
            ),
        );
        Ok(result)
    }

    pub(crate) fn contains(&self, point: [f64; 2]) -> bool {
        let [lo, hi] = self.bounds;
        (0..2).all(|axis| lo[axis] <= point[axis] && point[axis] <= hi[axis])
    }

    /// Lower bound for arrival anywhere in a closed ambient target ball,
    /// including paths that first leave this window. For an exiting path,
    /// the length already accumulated at the boundary supplies the bound.
    pub fn target_lower_bound(
        &self,
        start: [f64; 2],
        center: [f64; 2],
        radius: f64,
    ) -> Option<f64> {
        if !self.is_current() || !radius.is_finite() || radius < 0.0 || !self.contains(start) {
            return None;
        }
        self.target_lower(start, center, radius)
    }

    fn target_lower(&self, start: [f64; 2], center: [f64; 2], radius: f64) -> Option<f64> {
        let [lo, hi] = self.bounds;
        let exit = (0..2).fold(f64::INFINITY, |distance, axis| {
            distance
                .min((start[axis] - lo[axis]).next_down().max(0.0))
                .min((hi[axis] - start[axis]).next_down().max(0.0))
        });
        Some(
            self.displacement_length_lower(start, center, radius)?
                .min(div_down(exit, self.metric_ruler_upper())),
        )
    }

    fn displacement_length_lower(
        &self,
        start: [f64; 2],
        center: [f64; 2],
        radius: f64,
    ) -> Option<f64> {
        let target = sub_down(displacement_lower(start, center)?, radius);
        Some(div_down(target, self.metric_ruler_upper()))
    }

    fn metric_ruler_upper(&self) -> f64 {
        // The rounded matrix still dominates I/(2 * rho^2) on this window.
        // See docs/research/174-raster-route-bounds.md for the rounding argument.
        mul_up(std::f64::consts::SQRT_2.next_up(), self.rho_upper)
    }
}

fn displacement_upper(start: [f64; 2], end: [f64; 2]) -> f64 {
    let [dx, dy] = std::array::from_fn(|axis| {
        if start[axis] == end[axis] {
            0.0
        } else {
            (end[axis] - start[axis]).abs().next_up()
        }
    });
    let square = add_up(mul_up(dx, dx), mul_up(dy, dy));
    if square == 0.0 {
        0.0
    } else {
        square.sqrt().next_up()
    }
}

fn in_hard_disk([lo, hi]: [[f64; 2]; 2]) -> bool {
    if !(0..2).all(|axis| lo[axis].is_finite() && hi[axis].is_finite() && lo[axis] < hi[axis]) {
        return false;
    }
    let [x, y] = std::array::from_fn(|i| lo[i].abs().max(hi[i].abs()));
    add_up(mul_up(x, x), mul_up(y, y)) < 4.0
}

fn coefficient_range(field: &DistanceField, [lo, hi]: [[f64; 2]; 2]) -> Option<[f64; 2]> {
    let min = [field.xmin, field.ymin];
    let max = [field.xmax, field.ymax];
    let shape = [field.data.ncols(), field.data.nrows()];
    let mut first = [0; 2];
    let mut last = [0; 2];
    for axis in 0..2 {
        let span = max[axis] - min[axis];
        if !span.is_finite()
            || span <= 0.0
            || lo[axis] < min[axis]
            || hi[axis] > max[axis]
            || !(4..=(1u64 << 52)).contains(&(shape[axis] as u64))
        {
            return None;
        }
        // The provider uses this monotone chart with the same rounded operations.
        let count = shape[axis] as f64;
        let lower = raster_pixel(lo[axis], min[axis], span, count);
        let upper = raster_pixel(hi[axis], min[axis], span, count);
        if lower < 1.0 || upper >= count - 2.0 {
            return None;
        }
        first[axis] = lower.floor() as usize - 1;
        last[axis] = upper.floor() as usize + 2;
    }
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for row in first[1]..=last[1] {
        for column in first[0]..=last[0] {
            let value = field.data[[row, column]];
            if !value.is_finite() {
                return None;
            }
            lower = lower.min(f64::from(value));
            upper = upper.max(f64::from(value));
        }
    }
    Some([lower, upper])
}

fn sample_magnitude_upper(field: &DistanceField, bounds: [[f64; 2]; 2]) -> Option<f64> {
    let [lower, upper] = coefficient_range(field, bounds)?;
    let magnitude = lower.abs().max(upper.abs());
    // Exact Catmull–Rom tensor weight L1 is at most 25/16. The extra
    // 1/64 bounds the binary64 kernel, sum and division before the binary32
    // cast. A monotone cast bounds the sampler's own final cast.
    let upper = mul_up(101.0 / 64.0, magnitude) as f32;
    upper.is_finite().then_some(f64::from(upper))
}

fn sample_range(field: &DistanceField, bounds: [[f64; 2]; 2]) -> Option<[f64; 2]> {
    let [lower, upper] = coefficient_range(field, bounds)?;
    let spread = range_width([lower, upper]);
    let magnitude = lower.abs().max(upper.abs());
    // The tensor's negative weight mass is at most 9/32. The additional
    // allowance covers the compiled kernel and normalization before casting.
    let allowance = add_up(
        mul_up(9.0 / 32.0, spread),
        mul_up(2.0_f64.powi(-26), magnitude),
    );
    let lo = (lower - allowance).next_down() as f32;
    let hi = (upper + allowance).next_up() as f32;
    (lo.is_finite() && hi.is_finite()).then_some([f64::from(lo), f64::from(hi)])
}

fn range_width([lo, hi]: [f64; 2]) -> f64 {
    if lo == hi {
        0.0
    } else {
        (hi - lo).next_up()
    }
}

fn local_length_factor(field: &DistanceField, bounds: [[f64; 2]; 2]) -> Option<f64> {
    let epsilon = ManifoldConfig::default().epsilon;
    let [lo, hi] = sample_range(field, bounds)?;
    let d_upper = lo.abs().max(hi.abs());
    let rho_upper = add_up(mul_up(d_upper, d_upper), mul_up(epsilon, epsilon))
        .sqrt()
        .next_up();
    let d_lower = lo.max(-hi).max(0.0);
    let step_lower =
        (GEOMETRY_SCALE_ALPHA * epsilon.max(d_lower)).clamp(GEOMETRY_MIN_STEP, GEOMETRY_MAX_STEP);
    let step_upper = (GEOMETRY_SCALE_ALPHA * rho_upper).clamp(GEOMETRY_MIN_STEP, GEOMETRY_MAX_STEP);
    let stencil_bounds = [
        bounds[0].map(|v| (v - step_upper).next_down()),
        bounds[1].map(|v| (v + step_upper).next_up()),
    ];
    let gradient_upper = div_up(
        range_width(sample_range(field, stencil_bounds)?),
        2.0 * step_lower,
    );
    if gradient_upper > f64::from(1 << 18) {
        return None;
    }
    // Use the same monotone rounded operations as rho_from_jet.
    let rho_lower = (d_lower * d_lower + epsilon * epsilon).sqrt();
    Some(metric_length_factor(rho_lower, d_upper, gradient_upper))
}

/// Enclose the compiled default matrix by its maximum absolute row sum.
fn metric_length_factor(rho_lower: f64, d_upper: f64, gradient_upper: f64) -> f64 {
    // Every binary32 center sample widens exactly. Its square is exact in
    // binary64, so the rounded ruler is also at least the sample magnitude.
    let ratio_upper = (d_upper / rho_lower).min(1.0);
    let sigma_upper = (ratio_upper * gradient_upper) / (rho_lower * std::f64::consts::LN_2);
    let base_upper = 1.0 / (rho_lower * rho_lower);
    let rank_upper = sigma_upper * sigma_upper;
    let diagonal_upper = base_upper + rank_upper;
    add_up(diagonal_upper, rank_upper).sqrt().next_up()
}

#[cfg(test)]
mod tests;
