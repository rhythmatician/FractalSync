//! Displacement bounds for the default raster-bridge metric on a fixed window.

use super::{raster_pixel, raster_stamp, DistanceField, DIST_FIELD};
use crate::geometry_provider::{
    default_query_uses_raster_bridge, GEOMETRY_MAX_STEP, GEOMETRY_MIN_STEP, GEOMETRY_SCALE_ALPHA,
};
use crate::intrinsic_bounds::{add_up, div_down, div_up, mul_up, sub_down};
use crate::manifold::ManifoldConfig;
use crate::map_certificates::displacement_lower;

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
            // The compiled metric is bounded above by (2 + 6 * gradient_upper)^2 / rho^2.
            // The rounded ruler is at least epsilon / 2.
            length_per_displacement_upper: div_up(
                add_up(4.0, mul_up(12.0, gradient_upper)),
                config.epsilon,
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
        if !self.is_current() || !self.contains(start) || !self.contains(end) {
            return None;
        }
        let [dx, dy] = std::array::from_fn(|axis| {
            if start[axis] == end[axis] {
                0.0
            } else {
                (end[axis] - start[axis]).abs().next_up()
            }
        });
        let square = add_up(mul_up(dx, dx), mul_up(dy, dy));
        let displacement = if square == 0.0 {
            0.0
        } else {
            square.sqrt().next_up()
        };
        Some(mul_up(displacement, self.length_per_displacement_upper))
    }

    fn contains(&self, point: [f64; 2]) -> bool {
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
        let [lo, hi] = self.bounds;
        if !self.is_current() || !radius.is_finite() || radius < 0.0 || !self.contains(start) {
            return None;
        }
        let exit = (0..2).fold(f64::INFINITY, |distance, axis| {
            distance
                .min((start[axis] - lo[axis]).next_down().max(0.0))
                .min((hi[axis] - start[axis]).next_down().max(0.0))
        });
        let target = sub_down(displacement_lower(start, center)?, radius);
        // The rounded matrix still dominates I/(2 * rho^2) on this window.
        // See docs/research/174-raster-route-bounds.md for the rounding argument.
        let metric_ruler_upper = mul_up(std::f64::consts::SQRT_2.next_up(), self.rho_upper);
        Some(div_down(target.min(exit), metric_ruler_upper))
    }
}

fn in_hard_disk([lo, hi]: [[f64; 2]; 2]) -> bool {
    if !(0..2).all(|axis| lo[axis].is_finite() && hi[axis].is_finite() && lo[axis] < hi[axis]) {
        return false;
    }
    let [x, y] = std::array::from_fn(|i| lo[i].abs().max(hi[i].abs()));
    add_up(mul_up(x, x), mul_up(y, y)) < 4.0
}

fn sample_magnitude_upper(field: &DistanceField, [lo, hi]: [[f64; 2]; 2]) -> Option<f64> {
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
    let mut magnitude = 0.0_f64;
    for row in first[1]..=last[1] {
        for column in first[0]..=last[0] {
            let value = field.data[[row, column]];
            if !value.is_finite() {
                return None;
            }
            magnitude = magnitude.max(f64::from(value.abs()));
        }
    }
    // Exact Catmull–Rom tensor weight L1 is at most 25/16. The extra
    // 1/64 bounds the binary64 kernel, sum and division before the binary32
    // cast. A monotone cast bounds the sampler's own final cast.
    let upper = mul_up(101.0 / 64.0, magnitude) as f32;
    upper.is_finite().then_some(f64::from(upper))
}

#[cfg(test)]
mod tests;
