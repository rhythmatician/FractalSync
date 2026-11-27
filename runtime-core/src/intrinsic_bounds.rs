//! Intrinsic navigation bounds for the positive-floor ideal distance graph.
//!
//! The checked real inequalities are in `TwoEndedDisplacement.lean` and
//! `IntrinsicTargetBounds.lean`. Inputs used for pruning must enclose the true
//! quantities for the same nonempty closed Shore and the same ideal metric,
//! or for a metric whose path lengths are known to dominate it. An operational
//! GeometryJet or an unresolved or smoothed distance field supplies no such
//! certificate on its own.
//!
//! Graph search additionally needs edge costs that dominate the lengths of
//! admissible paths. Allow node reopening when using bounds with independent
//! interval inputs, which need not give a consistent heuristic.

use crate::shore_distance::RadialProfile;

/// Version of the input, evidence and numerical conventions for these bounds.
pub const INTRINSIC_BOUNDS_VERSION: &str = "intrinsic-bounds/1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntrinsicMetric {
    pub epsilon: f64,
    /// Anisotropy parameter lambda / ln(2).
    pub a: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointBounds {
    /// Closed interval enclosing the true nonnegative Euclidean distance to Shore.
    pub shore_distance: [f64; 2],
    /// Upper bound on sqrt(d² + epsilon²) at the endpoint.
    pub ruler_upper: f64,
}

impl EndpointBounds {
    /// Derive a ruler upper bound from a certified Shore-distance interval.
    pub fn from_shore_interval(metric: IntrinsicMetric, shore_distance: [f64; 2]) -> Option<Self> {
        RadialProfile::new(metric.epsilon, metric.a).ok()?;
        if !valid_interval(shore_distance) {
            return None;
        }
        let radius_upper = shore_distance[1];
        let larger = metric.epsilon.max(radius_upper);
        let smaller = metric.epsilon.min(radius_upper);
        let ratio = div_up(smaller, larger);
        // IEEE 754 square root is correctly rounded. Scale before squaring.
        let root = add_up(1.0, mul_up(ratio, ratio)).sqrt().next_up();
        let ruler_upper = if smaller == 0.0 {
            larger
        } else {
            mul_up(larger, root)
        };
        ruler_upper.is_finite().then_some(Self {
            shore_distance,
            ruler_upper,
        })
    }

    fn is_valid(self, epsilon: f64) -> bool {
        valid_interval(self.shore_distance)
            && self.ruler_upper.is_finite()
            && self.ruler_upper >= epsilon
            && self.ruler_upper >= self.shore_distance[0]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossingEvidence {
    Unknown,
    /// Every admissible connecting path intersects the actual Shore. This can
    /// follow from opposite signs under a common, certified signed-distance
    /// convention or membership in different actual complement components.
    Certified,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointBound {
    /// Maximum of the one-ended displacement lower bounds.
    pub one_ended: f64,
    /// Improvement available when the two upper rulers are compatible with
    /// the displacement lower bound.
    pub two_ended: Option<f64>,
    /// Certified radial lower bound, using a conservative density bound.
    pub radial: f64,
    /// Canonical radial profile evaluated at the same interval endpoints.
    /// Its floating-point evaluation is diagnostic only.
    pub radial_estimate: Option<f64>,
    /// Maximum of the certified components.
    pub lower: f64,
}

/// Lower bound to a point with certified endpoint intervals, upper rulers and
/// a nonnegative lower bound on Euclidean displacement. The metric parameters
/// are taken as exact binary64 values. Unknown crossing evidence uses only
/// interval separation.
pub fn point_lower_bound(
    metric: IntrinsicMetric,
    start: EndpointBounds,
    end: EndpointBounds,
    displacement_lower: f64,
    crossing: CrossingEvidence,
) -> Option<PointBound> {
    let profile = RadialProfile::new(metric.epsilon, metric.a).ok()?;
    if !start.is_valid(metric.epsilon)
        || !end.is_valid(metric.epsilon)
        || !nonnegative_finite(displacement_lower)
    {
        return None;
    }
    let small = start.ruler_upper.min(end.ruler_upper);
    let large = start.ruler_upper.max(end.ruler_upper);
    let one_ended = log1p_lower(div_down(displacement_lower, small));

    // For r <= s, split distance at (d+s-r)/2. The second term uses the
    // remaining (d+r-s)/2. Incompatible inputs still give the one-ended bound.
    let delta_upper = sub_up(large, small);
    let two_ended = (displacement_lower >= delta_upper).then(|| {
        let first = add_down(
            mul_down(displacement_lower, 0.5),
            mul_down(sub_down(large, small), 0.5),
        );
        let second = mul_down(sub_down(displacement_lower, delta_upper), 0.5);
        add_down(
            log1p_lower(div_down(first, small)),
            log1p_lower(div_down(second, large)),
        )
    });

    let [start_lo, start_hi] = start.shore_distance;
    let [end_lo, end_hi] = end.shore_distance;
    let (radial, radial_estimate) = match crossing {
        CrossingEvidence::Certified => (
            add_down(
                radial_gap_lower(metric, 0.0, start_lo),
                radial_gap_lower(metric, 0.0, end_lo),
            ),
            profile.evaluate(start_lo).ok().and_then(|start_value| {
                let sum = start_value + profile.evaluate(end_lo).ok()?;
                sum.is_finite().then_some(sum)
            }),
        ),
        CrossingEvidence::Unknown => {
            let lower_radius = start_hi.min(end_hi);
            let upper_radius = start_lo.max(end_lo);
            if lower_radius >= upper_radius {
                (0.0, Some(0.0))
            } else {
                (
                    radial_gap_lower(metric, lower_radius, upper_radius),
                    profile.evaluate(upper_radius).ok().and_then(|upper| {
                        Some((upper - profile.evaluate(lower_radius).ok()?).max(0.0))
                    }),
                )
            }
        }
    };
    Some(PointBound {
        one_ended,
        two_ended,
        radial,
        radial_estimate,
        lower: one_ended.max(two_ended.unwrap_or(0.0)).max(radial),
    })
}

/// Lower bound to a closed intrinsic ball. `point_lower` must be a certified
/// lower bound to the ball center in the same metric.
pub fn intrinsic_target_lower_bound(point_lower: f64, tolerance: f64) -> Option<f64> {
    (nonnegative_finite(point_lower) && nonnegative_finite(tolerance))
        .then(|| sub_down(point_lower, tolerance))
}

/// Lower bound to the full closed Euclidean ball around `center`. Lipschitz
/// continuity inflates the center's distance interval and ruler to cover every
/// candidate in the ball. A center crossing certificate cannot be extended to
/// the whole target without separate evidence.
pub fn ambient_target_lower_bound(
    metric: IntrinsicMetric,
    start: EndpointBounds,
    center: EndpointBounds,
    displacement_lower: f64,
    radius: f64,
) -> Option<PointBound> {
    if !center.is_valid(metric.epsilon)
        || !nonnegative_finite(radius)
        || !nonnegative_finite(displacement_lower)
    {
        return None;
    }
    let [lo, hi] = center.shore_distance;
    let target = EndpointBounds {
        shore_distance: [sub_down(lo, radius), add_up(hi, radius)],
        ruler_upper: add_up(center.ruler_upper, radius),
    };
    point_lower_bound(
        metric,
        start,
        target,
        sub_down(displacement_lower, radius),
        CrossingEvidence::Unknown,
    )
}

/// Outer Euclidean radius from the path start. `ruler_upper` must enclose
/// the ideal ruler there. `budget` bounds accumulated intrinsic path length,
/// and callers must account for query footprints and domain intersections.
/// A missing or overflowing radius leaves search unbounded by this result.
pub fn search_envelope_radius(ruler_upper: f64, budget: f64) -> Option<f64> {
    if !ruler_upper.is_finite() || ruler_upper <= 0.0 || !nonnegative_finite(budget) {
        return None;
    }
    if budget == 0.0 {
        return Some(0.0);
    }
    // Even the smallest positive binary64 ruler has an unrepresentable
    // radius at this budget. Bound the range-reduction work on invalid scales.
    if budget >= 2048.0 {
        return None;
    }
    let radius = if budget >= 512.0 {
        // At this scale exp(R) is a close upper bound on expm1(R).
        // Multiply by exp(R/4) four times so a small ruler can still
        // give a finite radius when expm1(R) alone would overflow.
        let factor = add_up(1.0, expm1_upper(budget * 0.25));
        (0..4).fold(ruler_upper, |radius, _| mul_up(radius, factor))
    } else {
        mul_up(ruler_upper, expm1_upper(budget))
    };
    radius.is_finite().then_some(radius)
}

fn valid_interval([lo, hi]: [f64; 2]) -> bool {
    nonnegative_finite(lo) && hi.is_finite() && hi >= lo
}

fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

// At all a >= 0 the radial density is at least 1/sqrt(t² + epsilon²),
// hence at least 1/(epsilon + t). The anisotropy factor is nondecreasing
// and is at least a*t/(epsilon+t). When useful, bound it at the lower
// endpoint of each of two intervals to retain a contribution from a.
fn radial_gap_lower(metric: IntrinsicMetric, lo: f64, hi: f64) -> f64 {
    let midpoint = lo + (hi - lo) * 0.5;
    let interval = |left: f64, right: f64| {
        let ruler = add_up(metric.epsilon, left);
        let scale = log1p_lower(div_down(sub_down(right, left), ruler));
        let factor = mul_down(metric.a, div_down(left, ruler)).max(1.0);
        mul_down(scale, factor)
    };
    let whole = interval(lo, hi);
    if metric.a <= 1.0 || midpoint <= lo || midpoint >= hi {
        return whole;
    }
    whole.max(add_down(interval(lo, midpoint), interval(midpoint, hi)))
}

// Directed basic operations on nonnegative f64 values. One adjacent
// representable value encloses the exact result of an IEEE 754 operation
// rounded to nearest, including subnormal results and overflow. The
// transcendental bounds below use arithmetic series instead of libm's
// target-dependent rounding. Some exact cases preserve zero and equality.
pub(crate) fn add_down(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        left + right
    } else {
        (left + right).next_down().max(0.0)
    }
}

pub(crate) fn add_up(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        left + right
    } else {
        (left + right).next_up()
    }
}

pub(crate) fn sub_down(left: f64, right: f64) -> f64 {
    if left <= right {
        0.0
    } else if right == 0.0 || left <= 2.0 * right {
        // Sterbenz's lemma gives exact subtraction for these positive values.
        left - right
    } else {
        (left - right).next_down().max(0.0)
    }
}

fn sub_up(left: f64, right: f64) -> f64 {
    if left <= right || right == 0.0 || left <= 2.0 * right {
        (left - right).max(0.0)
    } else {
        (left - right).next_up()
    }
}

pub(crate) fn mul_down(left: f64, right: f64) -> f64 {
    (left * right).next_down().max(0.0)
}

pub(crate) fn mul_up(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        0.0
    } else {
        (left * right).next_up()
    }
}

pub(crate) fn div_down(numerator: f64, denominator: f64) -> f64 {
    (numerator / denominator).next_down().max(0.0)
}

pub(crate) fn div_up(numerator: f64, denominator: f64) -> f64 {
    (numerator / denominator).next_up()
}

// For 0 <= x <= 1, log1p(x) = 2 atanh(x/(2+x)). Truncating the
// positive power series yields a lower bound. A binary exponent reduces
// larger arguments to the same interval.
fn log1p_lower(x: f64) -> f64 {
    if x <= 1.0 {
        return log1p_unit_lower(x);
    }
    let y = add_down(1.0, x);
    let bits = y.to_bits();
    let exponent = ((bits >> 52) as i32) - 1023;
    let fraction = f64::from_bits((bits & ((1 << 52) - 1)) | 1.0_f64.to_bits());
    // The first 16 terms of 2 atanh(1/3) already exceed this value.
    let ln_2_lower = std::f64::consts::LN_2.next_down();
    add_down(
        mul_down(f64::from(exponent), ln_2_lower),
        log1p_unit_lower(fraction - 1.0),
    )
}

fn log1p_unit_lower(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let ratio = div_down(x, add_up(2.0, x));
    let square = mul_down(ratio, ratio);
    let mut series = 0.0;
    for index in (0..10).rev() {
        series = add_down(
            div_down(1.0, f64::from(2 * index + 1)),
            mul_down(square, series),
        );
    }
    mul_down(mul_down(2.0, ratio), series)
}

// Reduce to x <= 1/2. From degree 13 on, successive Taylor terms have
// ratio less than 1/2, so twice the degree-13 term bounds the tail.
// Then expm1(2x) = expm1(x) * (2 + expm1(x)).
fn expm1_upper(mut x: f64) -> f64 {
    let mut doublings = 0;
    while x > 0.5 {
        x *= 0.5;
        doublings += 1;
    }
    let mut series = 2.0;
    for degree in (2..=13).rev() {
        series = add_up(1.0, mul_up(div_up(x, f64::from(degree)), series));
    }
    let mut value = mul_up(x, series);
    for _ in 0..doublings {
        value = mul_up(value, add_up(2.0, value));
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(epsilon: f64, a: f64) -> IntrinsicMetric {
        IntrinsicMetric { epsilon, a }
    }

    fn endpoint(metric: IntrinsicMetric, lo: f64, hi: f64) -> EndpointBounds {
        EndpointBounds::from_shore_interval(metric, [lo, hi]).unwrap()
    }

    #[test]
    fn distinct_shore_points_use_tangential_displacement() {
        let metric = metric(0.25, 2.0);
        let shore = endpoint(metric, 0.0, 0.0);
        let d = 0.5;
        let bound = point_lower_bound(metric, shore, shore, d, CrossingEvidence::Unknown).unwrap();
        let two_ended = bound.two_ended.unwrap();
        let reference = 2.0 * (d / (2.0 * metric.epsilon)).ln_1p();
        assert_eq!(bound.radial, 0.0);
        assert_eq!(bound.lower, two_ended);
        assert!(two_ended > bound.one_ended);
        assert!(two_ended <= reference && reference - two_ended < 1e-9);
        assert!(bound.lower <= d / metric.epsilon);
    }

    #[test]
    fn uncertain_rulers_use_one_ended_fallback() {
        let metric = metric(1.0, 0.0);
        let start = endpoint(metric, 0.0, 1.0);
        let end = endpoint(metric, 0.5, 10.0);
        let bound = point_lower_bound(metric, start, end, 0.25, CrossingEvidence::Unknown).unwrap();
        assert_eq!(bound.radial, 0.0);
        assert_eq!(bound.radial_estimate, Some(0.0));
        assert_eq!(bound.two_ended, None);
        assert_eq!(bound.lower, bound.one_ended);
        let reference = (0.25 / start.ruler_upper).ln_1p();
        assert!(bound.lower <= reference && reference - bound.lower < 1e-9);
    }

    #[test]
    fn compatible_at_the_split_boundary() {
        let metric = metric(1.0, 0.0);
        let start = EndpointBounds {
            shore_distance: [0.0, 0.0],
            ruler_upper: 1.0,
        };
        let end = EndpointBounds {
            shore_distance: [0.0, 0.0],
            ruler_upper: 2.0,
        };
        let bound = point_lower_bound(metric, start, end, 1.0, CrossingEvidence::Unknown).unwrap();
        assert!(bound.two_ended.is_some());
        assert!(bound.lower <= std::f64::consts::LN_2);
    }

    #[test]
    fn radial_intervals_and_crossing_use_the_appropriate_endpoints() {
        for a in [0.0, 8.0] {
            let metric = metric(1.0, a);
            let start = endpoint(metric, 0.9, 1.1);
            let end = endpoint(metric, 1.0, 1.2);
            let unknown =
                point_lower_bound(metric, start, end, 1.9, CrossingEvidence::Unknown).unwrap();
            assert_eq!(unknown.radial, 0.0);
            let crossing =
                point_lower_bound(metric, start, end, 1.9, CrossingEvidence::Certified).unwrap();
            assert!(crossing.radial > unknown.lower);
            assert!(crossing.radial <= crossing.radial_estimate.unwrap());

            let separated = endpoint(metric, 2.0, 2.1);
            let separation =
                point_lower_bound(metric, start, separated, 0.0, CrossingEvidence::Unknown)
                    .unwrap();
            assert!(separation.radial > 0.0);
            assert!(separation.radial <= separation.radial_estimate.unwrap());
        }
    }

    #[test]
    fn ideal_flat_normal_paths_bound_analytic_fixtures_at_each_scale() {
        let mut reference_bound: Option<f64> = None;
        for scale in [1e-8, 0.05, 1.0, 200.0, 1e120] {
            let metric = metric(scale, 0.0);
            let start = endpoint(metric, 3.0 * scale, 3.0 * scale);
            let end = endpoint(metric, 7.0 * scale, 7.0 * scale);
            let same_side =
                point_lower_bound(metric, start, end, 4.0 * scale, CrossingEvidence::Unknown)
                    .unwrap();
            let opposite = point_lower_bound(
                metric,
                start,
                end,
                10.0 * scale,
                CrossingEvidence::Certified,
            )
            .unwrap();
            let exact_start = 3.0_f64.asinh();
            let exact_end = 7.0_f64.asinh();
            assert!(same_side.lower <= exact_end - exact_start);
            assert!(opposite.lower <= exact_end + exact_start);
            if let Some(reference_bound) = reference_bound {
                assert!((opposite.lower - reference_bound).abs() < 1e-12);
            } else {
                reference_bound = Some(opposite.lower);
            }
        }
    }

    #[test]
    fn closed_target_balls_use_the_whole_acceptance_set() {
        let metric = metric(1.0, 0.0);
        let start = endpoint(metric, 1.0, 1.0);
        let center = endpoint(metric, 8.0, 8.0);
        let point =
            point_lower_bound(metric, start, center, 7.0, CrossingEvidence::Unknown).unwrap();
        let ambient = ambient_target_lower_bound(metric, start, center, 7.0, 2.0).unwrap();
        assert!(ambient.lower > 0.0 && ambient.lower < point.lower);
        assert!(ambient.lower <= 6.0_f64.asinh() - 1.0_f64.asinh());
        assert_eq!(
            ambient_target_lower_bound(metric, start, center, 7.0, 7.0)
                .unwrap()
                .lower,
            0.0
        );
        let tolerance = 0.25;
        let intrinsic = intrinsic_target_lower_bound(point.lower, tolerance).unwrap();
        assert!(intrinsic > 0.0 && intrinsic <= point.lower - tolerance);
        assert_eq!(intrinsic_target_lower_bound(point.lower, 4.0), Some(0.0));
    }

    #[test]
    fn envelope_contains_known_path_prefixes_and_handles_overflow() {
        assert_eq!(search_envelope_radius(1.0, 0.0), Some(0.0));
        for budget in [f64::MIN_POSITIVE, 1e-20, 0.125, 1.0, 8.0, 500.0, 702.0] {
            let radius = search_envelope_radius(0.25, budget).unwrap();
            let flat_shore_normal_path_prefix = 0.25 * budget.sinh();
            assert!(radius >= flat_shore_normal_path_prefix);
            let exact_envelope = 0.25 * budget.exp_m1();
            assert!(radius >= exact_envelope);
            assert!(radius / exact_envelope < 1.0 + 1e-9);
        }
        assert!(search_envelope_radius(f64::MIN_POSITIVE, 800.0).unwrap() > 1e39);
        assert_eq!(search_envelope_radius(1.0, 1024.0), None);
    }

    #[test]
    fn invalid_evidence_and_parameters_return_no_bound() {
        let metric = metric(1.0, 0.0);
        let shore = endpoint(metric, 0.0, 0.0);
        assert!(EndpointBounds::from_shore_interval(metric, [1.0, 0.0]).is_none());
        let invalid_ruler = EndpointBounds {
            ruler_upper: 0.5,
            ..shore
        };
        assert!(
            point_lower_bound(metric, shore, invalid_ruler, 1.0, CrossingEvidence::Unknown)
                .is_none()
        );
        assert!(
            point_lower_bound(metric, shore, shore, f64::NAN, CrossingEvidence::Unknown).is_none()
        );
        assert!(intrinsic_target_lower_bound(1.0, -1.0).is_none());
        assert!(ambient_target_lower_bound(metric, shore, shore, 1.0, f64::INFINITY).is_none());
        assert_eq!(search_envelope_radius(0.0, 1.0), None);
        assert_eq!(search_envelope_radius(1.0, f64::INFINITY), None);
    }
}
