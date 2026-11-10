mod scalar {
    include!("geometry-145-cardioid-probe.rs");
    pub fn parameters(c: [f64; 2]) -> Vec<(f64, bool)> {
        let [x, y] = c;
        let a = [0.5 - 2. * x, -6. * y, 6. * x + 0.5, 2. * y];
        let mut values = vec![(0., false), (0., true)];
        values.extend(roots(a).into_iter().map(|s| (s, false)));
        values.extend(
            roots([a[3], a[2], a[1], a[0]])
                .into_iter()
                .map(|r| (r, true)),
        );
        values
    }
}
#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn add(self, b: Self) -> Self {
        Self {
            lo: (self.lo + b.lo).next_down(),
            hi: (self.hi + b.hi).next_up(),
        }
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn sub(self, b: Self) -> Self {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Self {
        let p = [
            self.lo * b.lo,
            self.lo * b.hi,
            self.hi * b.lo,
            self.hi * b.hi,
        ];
        Self {
            lo: p.into_iter().fold(f64::INFINITY, f64::min).next_down(),
            hi: p.into_iter().fold(f64::NEG_INFINITY, f64::max).next_up(),
        }
    }
    fn div_positive(self, b: Self) -> Self {
        assert!(b.lo > 0.);
        self.mul(Self {
            lo: (1. / b.hi).next_down(),
            hi: (1. / b.lo).next_up(),
        })
    }
    fn abs_upper(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
}
fn boundary_box(parameter: f64, cotangent: bool) -> [Interval; 2] {
    let one = Interval::point(1.);
    let s = Interval::point(parameter);
    let q = s.mul(s);
    let denominator = one.add(q);
    let numerator = if cotangent { q.sub(one) } else { one.sub(q) };
    let u = numerator.div_positive(denominator);
    let v = Interval::point(2.).mul(s).div_positive(denominator);
    let half = Interval::point(0.5);
    let quarter = Interval::point(0.25);
    [
        half.mul(u).sub(quarter.mul(u.mul(u).sub(v.mul(v)))),
        half.mul(v).sub(half.mul(u.mul(v))),
    ]
}
fn distance_upper(c: [f64; 2]) -> f64 {
    scalar::parameters(c)
        .into_iter()
        .map(|(s, cot)| {
            let p = boundary_box(s, cot);
            let x = Interval::point(c[0]).sub(p[0]).abs_upper();
            let y = Interval::point(c[1]).sub(p[1]).abs_upper();
            ((x * x).next_up() + (y * y).next_up())
                .next_up()
                .sqrt()
                .next_up()
        })
        .fold(f64::INFINITY, f64::min)
}
fn main() {
    let c = [0.32046083605181780, 0.04736134021833255];
    for k in [12, 13, 14, 18, 20, 21, 22, 23] {
        let h = 2.0_f64.powi(-k);
        let mut bound: f64 = 0.;
        for y in -3..=4 {
            for x in -3..=4 {
                let p = [
                    ((c[0] / h).floor() + f64::from(x)) * h,
                    ((c[1] / h).floor() + f64::from(y)) * h,
                ];
                bound = bound.max(distance_upper(p));
            }
        }
        let ideal_field_bound = (4. * bound).next_up();
        let denominator = ((ideal_field_bound * ideal_field_bound).next_up()
            + (1e-4_f64 * 1e-4).next_up())
        .next_up()
        .sqrt()
        .next_up();
        let numerator = (h / (0.1 / 128.)).next_down();
        let ratio = (numerator / denominator).next_down();
        println!("k={k} node_boundary_upper={bound:.17e} ideal_field_abs_bound={ideal_field_bound:.17e} size_ratio_lower={ratio:.17e}");
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_cusp_and_left_tip_are_enclosed() {
        for (s, cot, expected) in [(0., false, [0.25, 0.]), (0., true, [-0.75, 0.])] {
            let b = boundary_box(s, cot);
            for i in 0..2 {
                assert!(b[i].lo <= expected[i] && b[i].hi >= expected[i]);
            }
            assert!(distance_upper(expected) < 1e-13);
        }
    }
    #[test]
    fn rational_parameter_needs_no_trigonometric_rounding_assumption() {
        // s=1 is the exact unit-circle point i, whose cardioid point is (1/4,1/2).
        let b = boundary_box(1., false);
        for (i, x) in [0.25, 0.5].into_iter().enumerate() {
            assert!(b[i].lo <= x && b[i].hi >= x);
        }
    }
}
