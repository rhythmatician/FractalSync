//! Endpoint certificates for the boundary of the mathematical Mandelbrot set.
//!
//! These intervals enclose distance to the ideal Shore. An operational field
//! requires an independent metric comparison before using these certificates.

use serde::Serialize;

use crate::intrinsic_bounds::{
    add_up, mul_down, CrossingEvidence, EndpointBounds, IntrinsicMetric,
};

pub const MANDELBROT_CERTIFICATE_VERSION: &str = "mandelbrot-endpoint/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum MandelbrotSide {
    Inside,
    Outside,
    Unknown,
}

/// Source of the lower bound and side. Every upper bound uses boundary witnesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
enum LowerBoundProof {
    InvariantDisk,
    EscapeDisk,
    IntervalEscape,
    Nonnegativity,
}

/// A distance enclosure and any proved side of the ideal Mandelbrot boundary.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MandelbrotEndpoint {
    version: &'static str,
    point: [f64; 2],
    shore_distance: [f64; 2],
    side: MandelbrotSide,
    lower_bound_proof: LowerBoundProof,
}

impl MandelbrotEndpoint {
    pub fn shore_distance(self) -> [f64; 2] {
        self.shore_distance
    }

    pub fn side(self) -> MandelbrotSide {
        self.side
    }

    pub fn bounds(self, metric: IntrinsicMetric) -> Option<EndpointBounds> {
        EndpointBounds::from_shore_interval(metric, self.shore_distance)
    }

    /// Opposite sides imply that every continuous connecting path hits Shore.
    pub fn crossing_to(self, other: Self) -> CrossingEvidence {
        use MandelbrotSide::{Inside, Outside};
        match (self.side, other.side) {
            (Inside, Outside) | (Outside, Inside) => CrossingEvidence::Certified,
            _ => CrossingEvidence::Unknown,
        }
    }
}

/// Enclose the Euclidean distance to the ideal Mandelbrot boundary.
///
/// The closed disk `|c| <= 1/4` is inside the Map because `|z| <= 1/2`
/// is forward invariant there. The Map lies in `|c| <= 2` by the escape
/// criterion. The cusp `c = 1/4` is a boundary point since larger real
/// parameters escape, and `c = -2` is a boundary point by the bounded orbit
/// `0, -2, 2, 2, ...` and the outer disk. Distances to these two points give
/// upper bounds everywhere. A local escape test can improve the exterior
/// lower bound. An unresolved side still gives a valid distance enclosure.
pub fn certify_mandelbrot_endpoint([x, y]: [f64; 2]) -> Option<MandelbrotEndpoint> {
    if !x.is_finite() || !y.is_finite() {
        return None;
    }

    let upper = add_up(abs_difference_upper(x, 0.25), y.abs())
        .min(add_up(abs_difference_upper(x, -2.0), y.abs()));
    if !upper.is_finite() {
        return None;
    }

    use MandelbrotSide::{Inside, Outside, Unknown};
    let norm_upper = add_up(x.abs(), y.abs());
    let norm_lower = norm_lower([x.abs(), y.abs()]);
    let (lower, side, lower_bound_proof) = if norm_upper < 0.25 {
        (
            (0.25 - norm_upper).next_down(),
            Inside,
            LowerBoundProof::InvariantDisk,
        )
    } else if norm_lower > 2.0 {
        (
            (norm_lower - 2.0).next_down(),
            Outside,
            LowerBoundProof::EscapeDisk,
        )
    } else if let Some(radius) = escape_radius([x, y], upper) {
        (radius, Outside, LowerBoundProof::IntervalEscape)
    } else {
        (0.0, Unknown, LowerBoundProof::Nonnegativity)
    };

    Some(MandelbrotEndpoint {
        version: MANDELBROT_CERTIFICATE_VERSION,
        point: [x, y],
        shore_distance: [lower, upper],
        side,
        lower_bound_proof,
    })
}

/// A directed-rounding lower bound for Euclidean displacement in Map units.
pub fn displacement_lower(start: [f64; 2], end: [f64; 2]) -> Option<f64> {
    if !start.into_iter().chain(end).all(f64::is_finite) {
        return None;
    }
    Some(norm_lower(std::array::from_fn(|axis| {
        (start[axis] - end[axis]).abs().next_down().max(0.0)
    })))
}

fn abs_difference_upper(left: f64, right: f64) -> f64 {
    if left == right {
        0.0
    } else {
        (left - right).abs().next_up()
    }
}

fn norm_lower([x, y]: [f64; 2]) -> f64 {
    let squared = (mul_down(x, x) + mul_down(y, y)).next_down().max(0.0);
    squared.sqrt().next_down().max(x).max(y)
}

// A square with half-side r contains the entire Euclidean closed r-ball.
// When every parameter in the square escapes, that ball misses the Shore.
fn escape_radius(point: [f64; 2], shore_upper: f64) -> Option<f64> {
    let mut radius = shore_upper.min(0.25) * 0.125;
    for _ in 0..3 {
        if radius == 0.0 {
            break;
        }
        let parameter = point.map(|axis| Interval {
            lo: (axis - radius).next_down(),
            hi: (axis + radius).next_up(),
        });
        if box_escapes(parameter) {
            return Some(radius);
        }
        radius *= 0.25;
    }
    None
}

fn box_escapes([x, y]: [Interval; 2]) -> bool {
    if add_up(x.abs_upper(), y.abs_upper()) > 2.0 {
        return false;
    }
    let zero = Interval { lo: 0.0, hi: 0.0 };
    let (mut real, mut imag) = (zero, zero);
    for _ in 0..32 {
        let product = real.mul(imag);
        real = real.square().sub(imag.square()).add(x);
        imag = product.add(product).add(y);
        if !real.is_finite() || !imag.is_finite() {
            return false;
        }
        // With |c| <= 2, once |z| > 2 the orbit diverges. Checking one
        // coordinate is sufficient here and avoids evaluating a norm.
        if real.abs_lower().max(imag.abs_lower()) > 2.0 {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn is_finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    fn abs_lower(self) -> f64 {
        self.lo.max(-self.hi).max(0.0)
    }

    fn abs_upper(self) -> f64 {
        self.hi.max(-self.lo)
    }

    fn add(self, other: Self) -> Self {
        Self {
            lo: (self.lo + other.lo).next_down(),
            hi: (self.hi + other.hi).next_up(),
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            lo: (self.lo - other.hi).next_down(),
            hi: (self.hi - other.lo).next_up(),
        }
    }

    fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        Self {
            lo: products
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            hi: products
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }

    fn square(self) -> Self {
        let lo = self.abs_lower();
        let hi = self.abs_upper();
        Self {
            lo: mul_down(lo, lo),
            hi: (hi * hi).next_up(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_shore_witnesses_have_zero_distance() {
        for x in [-2.0, 0.25] {
            let endpoint = certify_mandelbrot_endpoint([x, 0.0]).unwrap();
            assert_eq!(endpoint.shore_distance(), [0.0, 0.0]);
        }
    }

    #[test]
    fn certified_crossing_near_cusp_has_positive_lower_distances() {
        let start = certify_mandelbrot_endpoint([0.1875, 0.0]).unwrap();
        let end = certify_mandelbrot_endpoint([0.375, 0.0]).unwrap();
        assert_eq!(start.side(), MandelbrotSide::Inside);
        assert_eq!(end.side(), MandelbrotSide::Outside);
        assert_eq!(start.crossing_to(end), CrossingEvidence::Certified);
        assert_eq!(end.crossing_to(start), CrossingEvidence::Certified);
        assert!(start.shore_distance()[0] > 0.06);
        assert!(end.shore_distance()[0] > 0.0);
        assert!(end.shore_distance()[0] <= 0.125);
        assert!(end.shore_distance()[1] >= 0.125);
        let evidence = serde_json::to_value(end).unwrap();
        assert_eq!(evidence["version"], MANDELBROT_CERTIFICATE_VERSION);
        assert_eq!(evidence["side"], "outside");
        assert_eq!(evidence["lowerBoundProof"], "intervalEscape");
    }

    #[test]
    fn outer_disk_and_unresolved_endpoint_have_conservative_enclosures() {
        let outside = certify_mandelbrot_endpoint([-2.125, 0.0]).unwrap();
        let unknown = certify_mandelbrot_endpoint([-1.0, 0.0]).unwrap();
        assert_eq!(outside.side(), MandelbrotSide::Outside);
        assert!(outside.shore_distance()[0] > 0.12);
        assert!(outside.shore_distance()[0] <= 0.125);
        assert!(outside.shore_distance()[1] >= 0.125);
        assert_eq!(unknown.side(), MandelbrotSide::Unknown);
        assert_eq!(unknown.shore_distance()[0], 0.0);
        assert_eq!(unknown.crossing_to(outside), CrossingEvidence::Unknown);
    }

    #[test]
    fn bounds_are_conservative_under_invalid_and_extreme_inputs() {
        assert_eq!(certify_mandelbrot_endpoint([f64::NAN, 0.0]), None);
        assert_eq!(certify_mandelbrot_endpoint([f64::MAX, f64::MAX]), None);
        assert_eq!(displacement_lower([0.0; 2], [f64::INFINITY, 0.0]), None);
        assert_eq!(displacement_lower([0.0; 2], [0.0; 2]), Some(0.0));
        let diagonal = displacement_lower([0.0; 2], [3.0, 4.0]).unwrap();
        assert!(diagonal <= 5.0 && diagonal > 4.99);
        assert_eq!(
            displacement_lower([-f64::MAX, 0.0], [f64::MAX, 0.0]),
            Some(f64::MAX)
        );
    }
}
