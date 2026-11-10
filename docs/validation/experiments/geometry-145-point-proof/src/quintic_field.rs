//! One scalar polynomial and its differentiated jet, with nodal uncertainty.
//! Enclosures refer to the reconstruction of true nodal distances; they do not
//! bound the interpolation remainder against the true distance between nodes.
use crate::validated_branch::cover::bracket::DistanceBracket;
use crate::{number, C, P};
use astro_float::{BigFloat as B, RoundingMode as R};
use std::collections::BTreeMap;

#[derive(Clone)]
struct Interval {
    lo: B,
    hi: B,
}
impl Interval {
    fn exact(x: B) -> Self {
        Self {
            lo: x.clone(),
            hi: x,
        }
    }
    fn zero() -> Self {
        Self::exact(B::new(P))
    }
    fn add(&self, b: &Self) -> Self {
        Self {
            lo: self.lo.add(&b.lo, P, R::Down),
            hi: self.hi.add(&b.hi, P, R::Up),
        }
    }
    fn scale(&self, k: f64) -> Self {
        let x = B::from_f64(k, P);
        let (lo, hi) = if k >= 0. {
            (&self.lo, &self.hi)
        } else {
            (&self.hi, &self.lo)
        };
        Self {
            lo: lo.mul(&x, P, R::Down),
            hi: hi.mul(&x, P, R::Up),
        }
    }
    fn mul(&self, b: &Self) -> Self {
        let pairs = [
            (&self.lo, &b.lo),
            (&self.lo, &b.hi),
            (&self.hi, &b.lo),
            (&self.hi, &b.hi),
        ];
        let mut lo = pairs[0].0.mul(pairs[0].1, P, R::Down);
        let mut hi = pairs[0].0.mul(pairs[0].1, P, R::Up);
        for (a, b) in pairs.into_iter().skip(1) {
            let l = a.mul(b, P, R::Down);
            let u = a.mul(b, P, R::Up);
            if l.cmp(&lo).is_some_and(|v| v < 0) {
                lo = l;
            }
            if u.cmp(&hi).is_some_and(|v| v > 0) {
                hi = u;
            }
        }
        Self { lo, hi }
    }
    fn divide_positive(&self, b: &B) -> Self {
        Self {
            lo: self.lo.div(b, P, R::Down),
            hi: self.hi.div(b, P, R::Up),
        }
    }
    fn finite(&self) -> bool {
        [&self.lo, &self.hi]
            .iter()
            .all(|x| !x.is_nan() && !x.is_inf())
            && self.lo.cmp(&self.hi).is_some_and(|v| v <= 0)
    }
    fn midpoint(&self) -> f64 {
        number(
            &self
                .lo
                .add(&self.hi, P, R::ToEven)
                .mul(&B::from_f64(0.5, P), P, R::ToEven),
        )
    }
    fn deviation(&self, value: f64) -> B {
        let v = B::from_f64(value, P);
        let a = v.sub(&self.lo, P, R::Up);
        let b = self.hi.sub(&v, P, R::Up);
        if a.cmp(&b).is_some_and(|v| v >= 0) {
            a
        } else {
            b
        }
    }
}
#[derive(Clone)]
pub(crate) struct Sample {
    coordinate: [f64; 2],
    value: f64,
    bounds: Interval,
}
impl Sample {
    pub(crate) fn certify(
        node: &C,
        value: f64,
        error: f64,
        bracket: &DistanceBracket,
    ) -> Option<Self> {
        if !bracket.certifies_value(value, error)
            || bracket.query().re.cmp(&node.re) != Some(0)
            || bracket.query().im.cmp(&node.im) != Some(0)
        {
            return None;
        }
        let coordinate = node.f64();
        if !coordinate.iter().all(|x| x.is_finite())
            || B::from_f64(coordinate[0], P).cmp(&node.re) != Some(0)
            || B::from_f64(coordinate[1], P).cmp(&node.im) != Some(0)
        {
            return None;
        }
        let candidate = B::from_f64(value, P);
        let error = B::from_f64(error, P);
        Some(Self {
            coordinate,
            value,
            bounds: Interval {
                lo: candidate.sub(&error, P, R::Down),
                hi: candidate.add(&error, P, R::Up),
            },
        })
    }
}
pub(crate) struct NodalField {
    step: f64,
    samples: BTreeMap<(i64, i64), Sample>,
}
impl NodalField {
    pub(crate) fn new(step: f64, samples: Vec<Sample>) -> Option<Self> {
        if !step.is_normal() || step <= 0. || step.to_bits() & ((1u64 << 52) - 1) != 0 {
            return None;
        }
        let mut entries: BTreeMap<(i64, i64), Sample> = BTreeMap::new();
        for sample in samples {
            if !sample.value.is_finite() || !sample.bounds.finite() {
                return None;
            }
            let indices = sample.coordinate.map(|x| x / step);
            if indices
                .iter()
                .any(|x| !x.is_finite() || x.fract() != 0. || x.abs() > (1u64 << 50) as f64)
            {
                return None;
            }
            let key = (indices[0] as i64, indices[1] as i64);
            if let Some(previous) = entries.get(&key) {
                if previous.value != sample.value
                    || previous.bounds.lo.cmp(&sample.bounds.lo) != Some(0)
                    || previous.bounds.hi.cmp(&sample.bounds.hi) != Some(0)
                {
                    return None;
                }
            } else {
                entries.insert(key, sample);
            }
        }
        Some(Self {
            step,
            samples: entries,
        })
    }
    pub(crate) fn cell(&self, query: [f64; 2]) -> Option<Cell> {
        let index = query.map(|x| (x / self.step).floor());
        if index
            .iter()
            .any(|x| !x.is_finite() || x.abs() > (1u64 << 50) as f64)
        {
            return None;
        }
        let (x, y) = (index[0] as i64, index[1] as i64);
        let mut samples = Vec::with_capacity(81);
        for dy in -4..=4 {
            for dx in -4..=4 {
                samples.push(self.samples.get(&(x + dx, y + dy))?.clone());
            }
        }
        Cell::new(query, self.step, &samples)
    }
}
pub(crate) struct Cell {
    origin: [f64; 2],
    step: f64,
    coefficients: Vec<(Interval, Interval)>,
}
#[derive(Debug)]
pub(crate) struct Jet {
    // Value, gx, gy, Hxx, Hxy, Hyy, differentiated from one polynomial.
    pub(crate) components: [f64; 6],
    pub(crate) nodal_error: [f64; 6],
}
impl Cell {
    fn new(query: [f64; 2], step: f64, samples: &[Sample]) -> Option<Self> {
        if samples.len() != 81
            || !step.is_normal()
            || step <= 0.
            || step.to_bits() & ((1u64 << 52) - 1) != 0
        {
            return None;
        }
        let indices = query.map(|x| x / step);
        if indices
            .iter()
            .any(|x| !x.is_finite() || x.abs() > (1u64 << 50) as f64)
        {
            return None;
        }
        let origin = indices.map(|x| x.floor() * step);
        for y in 0..9 {
            for x in 0..9 {
                if samples[y * 9 + x].coordinate
                    != [
                        origin[0] + (x as f64 - 4.) * step,
                        origin[1] + (y as f64 - 4.) * step,
                    ]
                {
                    return None;
                }
            }
        }
        let mut coefficients = Vec::with_capacity(36);
        for y in 0..6 {
            for x in 0..6 {
                let i = (y + 2) * 9 + x + 2;
                let mut nominal = Interval::exact(B::from_f64(samples[i].value, P)).scale(2.);
                let mut bounds = samples[i].bounds.scale(2.);
                for j in [i - 1, i + 1, i - 9, i + 9] {
                    nominal = nominal
                        .add(&Interval::exact(B::from_f64(samples[j].value, P)).scale(-0.25));
                    bounds = bounds.add(&samples[j].bounds.scale(-0.25));
                }
                if !nominal.finite() || !bounds.finite() {
                    return None;
                }
                coefficients.push((nominal, bounds));
            }
        }
        Some(Self {
            origin,
            step,
            coefficients,
        })
    }
    pub(crate) fn evaluate(&self, query: [f64; 2]) -> Option<Jet> {
        let t = [
            (query[0] - self.origin[0]) / self.step,
            (query[1] - self.origin[1]) / self.step,
        ];
        if t.iter().any(|x| !x.is_finite() || *x < 0. || *x > 1.) {
            return None;
        }
        let coordinates: [Interval; 2] = std::array::from_fn(|axis| {
            let q = B::from_f64(query[axis], P);
            let origin = B::from_f64(self.origin[axis], P);
            Interval {
                lo: q.sub(&origin, P, R::Down),
                hi: q.sub(&origin, P, R::Up),
            }
            .divide_positive(&B::from_f64(self.step, P))
        });
        if coordinates.iter().any(|v| {
            !v.finite()
                || !v.lo.cmp(&B::new(P)).is_some_and(|x| x >= 0)
                || !v.hi.cmp(&B::from(1u8)).is_some_and(|x| x <= 0)
        }) {
            return None;
        }
        let wx = basis(&coordinates[0]);
        let wy = basis(&coordinates[1]);
        let mut components = [0.; 6];
        let mut nodal_error = [0.; 6];
        for (component, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
            .into_iter()
            .enumerate()
        {
            let divisor = B::from_f64(self.step, P).powi(dx + dy, P, R::ToEven);
            let mut nominal = Interval::zero();
            let mut bounds = Interval::zero();
            for y in 0..6 {
                for x in 0..6 {
                    let weight = wx[dx][x].mul(&wy[dy][y]).divide_positive(&divisor);
                    nominal = nominal.add(&weight.mul(&self.coefficients[y * 6 + x].0));
                    bounds = bounds.add(&weight.mul(&self.coefficients[y * 6 + x].1));
                }
            }
            if !nominal.finite() || !bounds.finite() {
                return None;
            }
            let value = nominal.midpoint();
            if !value.is_finite() {
                return None;
            }
            components[component] = value;
            // Conversion is a proposal; taking its successor encloses the
            // finite multiprecision upper bound, verified independently below.
            let error = bounds.deviation(value);
            let proposed = number(&error).next_up();
            if !proposed.is_finite() || B::from_f64(proposed, P).cmp(&error).is_none_or(|v| v < 0) {
                return None;
            }
            nodal_error[component] = proposed;
        }
        Some(Jet {
            components,
            nodal_error,
        })
    }
}
const POLYNOMIALS: [[i32; 6]; 6] = [
    [1, -5, 10, -10, 5, -1],
    [26, -50, 20, 20, -20, 5],
    [66, 0, -60, 0, 30, -10],
    [26, 50, 20, -20, -20, 10],
    [1, 5, 10, 10, 5, -5],
    [0, 0, 0, 0, 0, 1],
];
fn basis(argument: &Interval) -> [[Interval; 6]; 3] {
    std::array::from_fn(|derivative| {
        std::array::from_fn(|basis| {
            let mut result = Interval::zero();
            for power in (derivative..6).rev() {
                let factor = (0..derivative).fold(1, |product, i| product * (power - i) as i32);
                result = result.mul(argument).add(&Interval::exact(B::from(
                    POLYNOMIALS[basis][power] * factor,
                )));
            }
            result.divide_positive(&B::from(120u8))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sample_binding_and_declared_error_are_independent_of_witness_bounds() {
        use crate::validated_branch::cover::{bracket::OutsideWitness, build};
        let q = C::new(0., 0.);
        let cover = build(&q, 1, 0.5, 16, 128).unwrap();
        let a = OutsideWitness::certify([0.3, 0.], 128, 512).unwrap();
        let b = OutsideWitness::certify([0.4, 0.], 128, 512).unwrap();
        let a = DistanceBracket::certify(&cover, &a).unwrap();
        let b = DistanceBracket::certify(&cover, &b).unwrap();
        assert_ne!(a.upper().cmp(b.upper()), Some(0));
        let x = Sample::certify(&q, 0.2, 0.5, &a).unwrap();
        let y = Sample::certify(&q, 0.2, 0.5, &b).unwrap();
        assert!(NodalField::new(0.125, vec![x, y]).is_some());
        assert!(Sample::certify(&C::new(0.01, 0.), 0.2, 0.5, &a).is_none());
        assert!(Sample::certify(&q, f64::INFINITY, 0.5, &a).is_none());
        assert!(Sample::certify(&q, 0.2, 1e-9, &a).is_none());
    }
    fn samples(query: [f64; 2], step: f64, f: impl Fn(f64, f64) -> f64) -> Vec<Sample> {
        let o = query.map(|x| (x / step).floor() * step);
        (-4..=4)
            .flat_map(|y| (-4..=4).map(move |x| (x, y)))
            .map(|(x, y)| {
                let coordinate = [o[0] + x as f64 * step, o[1] + y as f64 * step];
                let value = f(coordinate[0], coordinate[1]);
                Sample {
                    coordinate,
                    value,
                    bounds: Interval::exact(B::from_f64(value, P)),
                }
            })
            .collect()
    }
    #[test]
    fn reproduces_quadratic_values_and_derivatives() {
        let query = [0.39265, 0.05016];
        let step = 0.001;
        let f = |x: f64, y: f64| x * x + 0.3 * y * y + 0.2 * x * y;
        let nodes = samples(query, step, f);
        let jet = Cell::new(query, step, &nodes);
        // The production h is dyadic; reject this deliberately non-dyadic input.
        assert!(jet.is_none());
        let step = 2f64.powi(-10);
        let nodes = samples(query, step, f);
        let jet = Cell::new(query, step, &nodes)
            .unwrap()
            .evaluate(query)
            .unwrap();
        for (actual, expected) in jet.components.into_iter().zip([
            f(query[0], query[1]),
            2. * query[0] + 0.2 * query[1],
            0.6 * query[1] + 0.2 * query[0],
            2.,
            0.2,
            0.6,
        ]) {
            assert!((actual - expected).abs() < 1e-8);
        }
    }
    #[test]
    fn nodal_uncertainty_reaches_every_jet_component() {
        let query = [0.28125, 0.40625];
        let step = 0.125;
        let mut nodes = samples(query, step, |x, y| x * x + y * y);
        let before = Cell::new(query, step, &nodes)
            .unwrap()
            .evaluate(query)
            .unwrap();
        nodes[40].bounds = Interval {
            lo: B::from_f64(-1., P),
            hi: B::from_f64(1., P),
        };
        let after = Cell::new(query, step, &nodes)
            .unwrap()
            .evaluate(query)
            .unwrap();
        assert_eq!(before.components, after.components);
        for (a, b) in after.nodal_error.into_iter().zip(before.nodal_error) {
            assert!(a > b);
        }
        nodes.swap(40, 41);
        assert!(Cell::new(query, step, &nodes).is_none());
    }
    #[test]
    fn adjacent_cells_share_value_gradient_and_hessian_at_the_knot() {
        let h = 0.125;
        let f = |x: f64, y: f64| (7. * x).sin() + (3. * y).cos();
        let left = [0.24, 0.4];
        let right = [0.26, 0.4];
        let mut nodes = samples(left, h, f);
        nodes.extend(samples(right, h, f));
        let assignment = NodalField::new(h, nodes).unwrap();
        let a = assignment.cell(left).unwrap();
        let b = assignment.cell(right).unwrap();
        for y in [0.375, 0.4, 0.499] {
            let x = a.evaluate([0.25, y]).unwrap();
            let z = b.evaluate([0.25, y]).unwrap();
            for (u, v) in x.components.into_iter().zip(z.components) {
                assert!((u - v).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn conflicting_shared_assignments_and_missing_nodes_refuse() {
        let first = Sample {
            coordinate: [0., 0.],
            value: 0.,
            bounds: Interval {
                lo: B::from(-1),
                hi: B::from(1),
            },
        };
        let mut second = first.clone();
        second.value = 0.25;
        assert!(NodalField::new(0.125, vec![first.clone(), second]).is_none());
        let mut second = first.clone();
        second.bounds.hi = B::from(2);
        assert!(NodalField::new(0.125, vec![first.clone(), second]).is_none());
        let field = NodalField::new(0.125, vec![first.clone(), first]).unwrap();
        assert!(field.cell([0., 0.]).is_none());
    }
    fn reference_basis(t: &B, derivative: usize, index: usize) -> B {
        // Independent truncated-power cardinal spline formula, at 512 bits.
        let s = t.add(&B::from((5 - index) as i32), 512, R::ToEven);
        let choose = [1i32, 6, 15, 20, 15, 6, 1];
        let factor = (0..derivative).fold(1i32, |a, j| a * (5 - j) as i32);
        let mut sum = B::new(512);
        for (k, binomial) in choose.into_iter().enumerate() {
            let v = s.sub(&B::from(k as i32), 512, R::ToEven);
            if v.cmp(&B::new(512)).is_some_and(|x| x > 0) {
                let signed = if k % 2 == 0 {
                    binomial * factor
                } else {
                    -binomial * factor
                };
                sum = sum.add(
                    &v.powi(5 - derivative, 512, R::ToEven)
                        .mul(&B::from(signed), 512, R::ToEven),
                    512,
                    R::ToEven,
                );
            }
        }
        sum.div(&B::from(120), 512, R::ToEven)
    }
    #[test]
    fn arbitrary_nodal_values_are_enclosed_against_independent_512_bit_reconstruction() {
        let query = [0.39265, 0.05016];
        let h = 2f64.powi(-10);
        let mut nodes = samples(query, h, |x, y| (37. * x).sin() + (23. * y).cos() + x * y);
        for node in &mut nodes {
            let v = B::from_f64(node.value, P);
            let e = B::from_f64(2f64.powi(-40), P);
            node.bounds = Interval {
                lo: v.sub(&e, P, R::Down),
                hi: v.add(&e, P, R::Up),
            };
        }
        let field = Cell::new(query, h, &nodes)
            .unwrap()
            .evaluate(query)
            .unwrap();
        let origin = query.map(|x| (x / h).floor() * h);
        let t: [B; 2] = std::array::from_fn(|axis| {
            B::from_f64(query[axis], 512)
                .sub(&B::from_f64(origin[axis], 512), 512, R::ToEven)
                .div(&B::from_f64(h, 512), 512, R::ToEven)
        });
        for seed in 0..8usize {
            let values: Vec<_> = nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    if (i * 17 + seed * 23) % 7 < 3 {
                        node.bounds.lo.clone()
                    } else {
                        node.bounds.hi.clone()
                    }
                })
                .collect();
            for (component, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
                .into_iter()
                .enumerate()
            {
                let mut reference = B::new(512);
                for y in 0..6 {
                    for x in 0..6 {
                        let i = (y + 2) * 9 + x + 2;
                        let mut coefficient = values[i].mul(&B::from(2), 512, R::ToEven);
                        for j in [i - 1, i + 1, i - 9, i + 9] {
                            coefficient = coefficient.sub(
                                &values[j].mul(&B::from_f64(0.25, 512), 512, R::ToEven),
                                512,
                                R::ToEven,
                            );
                        }
                        let weight = reference_basis(&t[0], dx, x)
                            .mul(&reference_basis(&t[1], dy, y), 512, R::ToEven)
                            .div(
                                &B::from_f64(h, 512).powi(dx + dy, 512, R::ToEven),
                                512,
                                R::ToEven,
                            );
                        reference = reference.add(
                            &weight.mul(&coefficient, 512, R::ToEven),
                            512,
                            R::ToEven,
                        );
                    }
                }
                let difference = reference
                    .sub(&B::from_f64(field.components[component], 512), 512, R::Up)
                    .abs();
                assert!(difference
                    .cmp(&B::from_f64(field.nodal_error[component], 512))
                    .is_some_and(|v| v <= 0));
            }
        }
    }
}
