//! Length enclosures for the exact real metric of a finite prepared spline.
//!
//! The stored coefficients and chart are interpreted as real numbers. These
//! bounds do not enclose every rounded Physics evaluation or compare the
//! spline to the true Shore.

use super::PreparedField;
use crate::intrinsic_bounds::{add_down, add_up, div_down, div_up, mul_down, mul_up, sub_down};
use crate::manifold::ManifoldConfig;
use crate::map_certificates::displacement_lower;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OperationalLengthInterval {
    pub lower: f64,
    pub upper: f64,
}

pub(crate) struct SegmentEvaluator<'r, 'f> {
    region: &'r OperationalNavigationRegion<'f>,
    cells: BTreeMap<[usize; 2], Option<([f64; 2], [f64; 2])>>,
}

impl SegmentEvaluator<'_, '_> {
    pub(crate) fn bounds(
        &mut self,
        start: [f64; 2],
        end: [f64; 2],
    ) -> Option<OperationalLengthInterval> {
        self.region.segment_bounds_using(start, end, |cell| {
            *self
                .cells
                .entry(cell)
                .or_insert_with(|| self.region.field.bernstein_cell_bounds(cell))
        })
    }

    pub(crate) fn cell_queries(&self) -> usize {
        self.cells.len()
    }
}

/// A rectangular search window with a fixed prepared field and metric profile.
/// Each segment still needs complete coefficient support.
pub struct OperationalNavigationRegion<'a> {
    field: &'a PreparedField,
    bounds: [[f64; 2]; 2],
    epsilon: f64,
    anisotropy_squared_upper: f64,
    ruler_upper: f64,
}

impl PreparedField {
    /// Bind bounds to this asset and profile. A sparse window may contain
    /// unavailable segments. The regional lower bound applies to paths
    /// contained in the window, including the arrival point.
    pub fn navigation_region(
        &self,
        bounds: [[f64; 2]; 2],
        config: &ManifoldConfig,
    ) -> Option<OperationalNavigationRegion<'_>> {
        let epsilon = config.epsilon;
        if !epsilon.is_finite()
            || epsilon <= 0.0
            || !config.lambda_sq.is_finite()
            || config.lambda_sq < 0.0
        {
            return None;
        }
        let [first, last] = self.navigation_cells(bounds)?;
        let ln_2_lower = std::f64::consts::LN_2.next_down();
        let anisotropy_squared_upper = if config.lambda_sq == 0.0 {
            0.0
        } else {
            div_up(config.lambda_sq, mul_down(ln_2_lower, ln_2_lower))
        };
        let ruler_upper = ruler_upper(self.navigation_coefficient_limit(first, last)?, epsilon);
        (ruler_upper.is_finite() && anisotropy_squared_upper.is_finite()).then_some(
            OperationalNavigationRegion {
                field: self,
                bounds,
                epsilon,
                anisotropy_squared_upper,
                ruler_upper,
            },
        )
    }

    pub(super) fn navigation_cells(&self, [lo, hi]: [[f64; 2]; 2]) -> Option<[[usize; 2]; 2]> {
        let mut first = [0; 2];
        let mut last = [0; 2];
        for axis in 0..2 {
            if !lo[axis].is_finite() || !hi[axis].is_finite() || lo[axis] > hi[axis] {
                return None;
            }
            // The affine chart is monotone. Round out at both steps to cover
            // every floor-selected patch, including both sides at a join.
            let q_lo = ((lo[axis] - self.origin[axis]).next_down() / self.spacing).next_down();
            let q_hi = ((hi[axis] - self.origin[axis]).next_up() / self.spacing).next_up();
            if q_lo < 2.0 || q_hi >= (self.shape[axis] - 3) as f64 {
                return None;
            }
            first[axis] = q_lo.floor() as usize;
            last[axis] = q_hi.floor() as usize;
        }
        Some([first, last])
    }

    fn navigation_coefficient_limit(&self, first: [usize; 2], last: [usize; 2]) -> Option<f64> {
        let mut magnitude = f64::NEG_INFINITY;
        for row in (first[1] - 2)..=(last[1] + 3) {
            for column in (first[0] - 2)..=(last[0] + 3) {
                let index = row * self.shape[0] + column;
                if self.available.as_ref().is_none_or(|mask| mask[index]) {
                    magnitude = magnitude.max(self.coefficients[index].abs());
                }
            }
        }
        magnitude.is_finite().then_some(magnitude)
    }
}

impl OperationalNavigationRegion<'_> {
    /// The nominal matrix dominates the identity divided by the squared ruler.
    pub(crate) fn nominal_speed_lower(&self, velocity: [f64; 2]) -> Option<f64> {
        let [x, y] = velocity;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        // The maximum component is a lower bound on the Euclidean norm.
        Some(div_down(x.abs().max(y.abs()), self.ruler_upper))
    }

    /// Enclose the length of a straight segment. The certificate uses the
    /// support of its coordinate rectangle, so even an off-segment support
    /// hole can make a segment unavailable.
    pub fn segment_bounds(
        &self,
        start: [f64; 2],
        end: [f64; 2],
    ) -> Option<OperationalLengthInterval> {
        self.segment_bounds_using(start, end, |cell| self.field.bernstein_cell_bounds(cell))
    }

    /// Reuse cell enclosures while building one graph over this region.
    pub(crate) fn segment_evaluator(&self) -> SegmentEvaluator<'_, '_> {
        SegmentEvaluator {
            region: self,
            cells: BTreeMap::new(),
        }
    }

    fn segment_bounds_using(
        &self,
        start: [f64; 2],
        end: [f64; 2],
        cell_bounds: impl FnMut([usize; 2]) -> Option<([f64; 2], [f64; 2])>,
    ) -> Option<OperationalLengthInterval> {
        if !self.contains(start) || !self.contains(end) {
            return None;
        }
        let bounds = [
            [start[0].min(end[0]), start[1].min(end[1])],
            [start[0].max(end[0]), start[1].max(end[1])],
        ];
        let delta = std::array::from_fn(|i| {
            if bounds[0][i] == bounds[1][i] {
                0.0
            } else {
                (bounds[1][i] - bounds[0][i]).next_up()
            }
        });
        let (upper, rho_upper) = self.enclose_speed(bounds, delta, cell_bounds)?;
        let lower = div_down(displacement_lower(start, end)?, rho_upper);
        Some(OperationalLengthInterval { lower, upper })
    }

    /// Bound speed in a caller-certified tube using absolute component caps
    /// on the ambient velocity. The full position box needs spline support.
    pub(crate) fn tube_speed_upper(
        &self,
        bounds: [[f64; 2]; 2],
        velocity_abs_upper: [f64; 2],
    ) -> Option<f64> {
        self.enclose_speed(bounds, velocity_abs_upper, |cell| {
            self.field.bernstein_cell_bounds(cell)
        })
        .map(|(speed, _)| speed)
    }

    fn enclose_speed(
        &self,
        bounds: [[f64; 2]; 2],
        delta: [f64; 2],
        mut cell_bounds: impl FnMut([usize; 2]) -> Option<([f64; 2], [f64; 2])>,
    ) -> Option<(f64, f64)> {
        if !bounds.into_iter().all(|corner| self.contains(corner))
            || !delta.into_iter().all(|cap| cap.is_finite() && cap >= 0.0)
        {
            return None;
        }
        let [first, last] = self.field.navigation_cells(bounds)?;
        let mut value_lower = f64::INFINITY;
        let mut value_upper = f64::NEG_INFINITY;
        let mut gradient = [0.0_f64; 2];
        for y in first[1]..=last[1] {
            for x in first[0]..=last[0] {
                let ([lo, hi], upper) = cell_bounds([x, y])?;
                value_lower = value_lower.min(lo);
                value_upper = value_upper.max(hi);
                gradient[0] = gradient[0].max(upper[0]);
                gradient[1] = gradient[1].max(upper[1]);
            }
        }

        let d_max = value_lower.abs().max(value_upper.abs());
        let d_min = value_lower.max(-value_upper).max(0.0);
        let rho_lower = add_down(mul_down(d_min, d_min), mul_down(self.epsilon, self.epsilon))
            .sqrt()
            .next_down()
            .max(self.epsilon)
            .max(d_min);
        let rho_upper = ruler_upper(d_max, self.epsilon);
        if delta == [0.0; 2] {
            return Some((0.0, rho_upper));
        }
        let displacement_squared = add_up(mul_up(delta[0], delta[0]), mul_up(delta[1], delta[1]));
        let field_rate = add_up(mul_up(gradient[0], delta[0]), mul_up(gradient[1], delta[1]));
        let ruler_rate = mul_up(field_rate, div_up(d_max, rho_lower).min(1.0));
        let weighted_rate_squared = mul_up(
            self.anisotropy_squared_upper,
            mul_up(ruler_rate, ruler_rate),
        );
        let numerator = add_up(displacement_squared, weighted_rate_squared)
            .sqrt()
            .next_up();
        let upper = div_up(numerator, rho_lower);
        upper.is_finite().then_some((upper, rho_upper))
    }

    /// Lower bound over paths that stay in this window to any point of the
    /// accepted ambient ball, intersected with the window.
    pub fn target_lower_bound(
        &self,
        start: [f64; 2],
        center: [f64; 2],
        radius: f64,
    ) -> Option<f64> {
        if !self.contains(start) || !radius.is_finite() || radius < 0.0 {
            return None;
        }
        let distance = sub_down(displacement_lower(start, center)?, radius);
        Some(div_down(distance, self.ruler_upper))
    }

    /// Outer ambient radius for a path prefix with the stated length budget.
    pub fn budget_radius_upper(&self, length: f64) -> Option<f64> {
        if !length.is_finite() || length < 0.0 {
            return None;
        }
        let upper = mul_up(length, self.ruler_upper);
        upper.is_finite().then_some(upper)
    }

    fn contains(&self, point: [f64; 2]) -> bool {
        (0..2).all(|i| self.bounds[0][i] <= point[i] && point[i] <= self.bounds[1][i])
    }
}

pub(super) fn ruler_upper(distance: f64, epsilon: f64) -> f64 {
    add_up(mul_up(distance, distance), mul_up(epsilon, epsilon))
        .sqrt()
        .next_up()
}

#[cfg(test)]
mod tests;
