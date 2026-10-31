use super::interval::Interval;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
pub(super) struct DistanceBounds {
    pub(super) lower: f64,
    pub(super) upper: f64,
    pub(super) visits: usize,
}

pub(super) fn curve_box(a: f64, b: f64, cotangent: bool) -> [Interval; 2] {
    if !a.is_finite() || !b.is_finite() || a < -1. || b > 1. || a > b {
        return [Interval { lo: f64::NEG_INFINITY, hi: f64::INFINITY }; 2];
    }
    let one = Interval::point(1.);
    let parameter = Interval { lo: a, hi: b };
    let square = parameter.square();
    let denominator = one.add(square);
    let numerator = if cotangent { square.sub(one) } else { one.sub(square) };
    let u = numerator.div_positive(denominator);
    let v = Interval::point(2.).mul(parameter).div_positive(denominator);
    let half = Interval::point(0.5);
    let quarter = Interval::point(0.25);
    [half.mul(u).sub(quarter.mul(u.square().sub(v.square()))),
     half.mul(v).sub(half.mul(u.mul(v)))]
}

fn norm_bounds(vector: [Interval; 2]) -> Interval {
    let square = vector[0].square().add(vector[1].square());
    Interval { lo: square.lo.max(0.).sqrt().next_down().max(0.),
               hi: square.hi.sqrt().next_up() }
}

fn point_distance(c: [f64; 2], point: [Interval; 2]) -> Interval {
    norm_bounds([Interval::point(c[0]).sub(point[0]), Interval::point(c[1]).sub(point[1])])
}

fn midpoint_and_radius(point: [Interval; 2]) -> ([f64; 2], f64) {
    let middle = [point[0].lo + (point[0].hi - point[0].lo) * 0.5,
                  point[1].lo + (point[1].hi - point[1].lo) * 0.5];
    let radius = norm_bounds([point[0].sub(Interval::point(middle[0])),
                              point[1].sub(Interval::point(middle[1]))]).hi;
    (middle, radius)
}

fn chord_distance(c: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Interval {
    let delta = [Interval::point(b[0]).sub(Interval::point(a[0])),
                 Interval::point(b[1]).sub(Interval::point(a[1]))];
    let relative = [Interval::point(c[0]).sub(Interval::point(a[0])),
                    Interval::point(c[1]).sub(Interval::point(a[1]))];
    let denominator = delta[0].square().add(delta[1].square());
    if denominator.lo <= 0. {
        // A nonpositive denominator enclosure need not prove equal endpoints.
        // Distance to this segment is at least endpoint distance minus its
        // full length and at most endpoint distance.
        let endpoint = norm_bounds(relative);
        let length = norm_bounds(delta).hi;
        return Interval {lo:(endpoint.lo-length).next_down().max(0.),hi:endpoint.hi};
    }
    let numerator = relative[0].mul(delta[0]).add(relative[1].mul(delta[1]));
    let t = numerator.div_positive(denominator);
    let clamped = Interval { lo: t.lo.clamp(0., 1.), hi: t.hi.clamp(0., 1.) };
    norm_bounds([relative[0].sub(clamped.mul(delta[0])),
                 relative[1].sub(clamped.mul(delta[1]))])
}

#[derive(Clone, Copy)]
struct Region { a: f64, b: f64, cotangent: bool, lower: f64, upper: f64 }
impl PartialEq for Region { fn eq(&self, other: &Self) -> bool { self.cmp(other) == Ordering::Equal } }
impl Eq for Region {}
impl PartialOrd for Region { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }
impl Ord for Region {
    fn cmp(&self, other: &Self) -> Ordering {
        other.lower.total_cmp(&self.lower)
            .then_with(|| self.a.total_cmp(&other.a))
            .then_with(|| self.b.total_cmp(&other.b))
            .then_with(|| self.cotangent.cmp(&other.cotangent))
    }
}

fn region_bounds(c: [f64; 2], a: f64, b: f64, cotangent: bool) -> Region {
    let (first, first_error) = midpoint_and_radius(curve_box(a, a, cotangent));
    let (last, last_error) = midpoint_and_radius(curve_box(b, b, cotangent));
    let chord = chord_distance(c, first, last);
    // p(theta)=e^(i theta)/2-e^(2i theta)/4 has |p'|<=1, |p''|<=3/2.
    // Both rational charts have |theta'|<=2 and |theta''|<=4.
    // Thus |d²p/ds²|<=10. Linear interpolation has Hausdorff error
    // at most 10*(b-a)^2/8 in both directions, by its integral remainder.
    let width = (b - a).next_up();
    let interpolation = (1.25 * (width * width).next_up()).next_up();
    let error = (interpolation + first_error.max(last_error)).next_up();
    Region { a, b, cotangent,
             lower: (chord.lo - error).next_down().max(0.),
             upper: (chord.hi + error).next_up() }
}

pub(super) fn distance_bounds(c: [f64; 2], tolerance: f64, budget: usize) -> Option<DistanceBounds> {
    if !c.iter().all(|x| x.is_finite()) || !tolerance.is_finite() || tolerance <= 0. { return None; }
    let mut upper = point_distance(c, curve_box(0., 0., false)).hi
        .min(point_distance(c, curve_box(0., 0., true)).hi);
    let mut heap = BinaryHeap::new();
    for cotangent in [false, true] {
        let region = region_bounds(c, -1., 1., cotangent);
        upper = upper.min(region.upper);
        heap.push(region);
    }
    for visits in 1..=budget {
        let region = heap.pop()?;
        if region.lower > upper { continue; }
        if (upper - region.lower).next_up() <= tolerance {
            return Some(DistanceBounds { lower: region.lower, upper, visits });
        }
        let middle = region.a + (region.b - region.a) * 0.5;
        if middle == region.a || middle == region.b { return None; }
        for (a,b) in [(region.a,middle),(middle,region.b)] {
            let child = region_bounds(c,a,b,region.cotangent);
            if !child.lower.is_finite() || !child.upper.is_finite() { return None; }
            upper = upper.min(child.upper);
            if child.lower <= upper { heap.push(child); }
        }
    }
    None
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn encloses_exact_nearest_cusp_and_left_tip_distances() {
        for c in [[0.,0.],[-0.5,0.]] {
            let bounds=distance_bounds(c,1e-13,8192).unwrap();
            assert!(bounds.lower<=0.25 && bounds.upper>=0.25);
            assert!((bounds.upper-bounds.lower).next_up()<=1e-13);
        }
    }
    #[test] fn contains_regular_reference_distances() {
        for (c,expected) in [([-0.5,0.1],0.23815848417896568),
            ([0.30007240651412576,0.05348910097156354],0.019521804826773806),
            ([0.21474166530729011,0.12774018997553147],0.13251681516052227),
            ([0.21398109574495830,0.12976651381062876],0.13453475128696687)] {
            let bounds=distance_bounds(c,1e-13,8192).unwrap();
            assert!(bounds.lower<=expected+1e-15 && bounds.upper>=expected-1e-15,"{c:?} {bounds:?}");
        }
    }
    #[test] fn exhausted_or_invalid_queries_cannot_supply_a_certificate() {
        assert!(distance_bounds([-0.5,0.1],1e-13,0).is_none());
        assert!(distance_bounds([f64::NAN,0.],1e-13,8192).is_none());
        assert!(distance_bounds([0.,0.],0.,8192).is_none());
    }
    #[test] fn underflowing_nonzero_chord_still_encloses_its_endpoint() {
        let bounds=chord_distance([1e-200,0.],[0.,0.],[1e-200,0.]);
        assert!(bounds.lo<=0. && bounds.hi>=0.);
    }
}
