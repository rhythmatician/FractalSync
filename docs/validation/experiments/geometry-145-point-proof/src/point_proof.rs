use astro_float::{BigFloat, RoundingMode as R, Sign};
use std::collections::VecDeque;

#[derive(Debug)]
pub struct EscapeProof {
    pub update: usize,
    pub precision: usize,
    pub radius_exponent: Option<i32>,
}
#[derive(Debug)]
pub enum MembershipProof {
    Outside(EscapeProof),
    Inside {
        prefix: usize,
        period: usize,
        precision: usize,
    },
}
#[derive(Clone)]
struct Complex {
    re: BigFloat,
    im: BigFloat,
}
impl Complex {
    fn norm_squared(&self, p: usize, round: R) -> BigFloat {
        self.re
            .mul(&self.re, p, round)
            .add(&self.im.mul(&self.im, p, round), p, round)
    }
    fn norm_upper(&self, p: usize) -> BigFloat {
        self.norm_squared(p, R::Up).sqrt(p, R::Up)
    }
    fn advance(&self, radius: &BigFloat, c: &Self, p: usize) -> Option<(Self, BigFloat)> {
        let (next, error) = self.step(c, p)?;
        let next_radius = BigFloat::from(2u8)
            .mul(&self.norm_upper(p), p, R::Up)
            .mul(radius, p, R::Up)
            .add(&radius.mul(radius, p, R::Up), p, R::Up)
            .add(&error, p, R::Up);
        if next_radius.is_nan() || next_radius.is_inf() {
            return None;
        }
        Some((next, next_radius))
    }
    fn distance_upper(&self, other: &Self, p: usize) -> Option<BigFloat> {
        let component = |a: &BigFloat, b: &BigFloat| {
            let abs = |mut v: BigFloat| {
                if v.sign() == Some(Sign::Neg) {
                    v.inv_sign();
                }
                v
            };
            let lo = abs(a.sub(b, p, R::Down));
            let hi = abs(a.sub(b, p, R::Up));
            if lo.is_nan() || hi.is_nan() || lo.is_inf() || hi.is_inf() {
                return None;
            }
            Some(if lo.cmp(&hi).is_some_and(|v| v >= 0) {
                lo
            } else {
                hi
            })
        };
        let result = Self {
            re: component(&self.re, &other.re)?,
            im: component(&self.im, &other.im)?,
        }
        .norm_upper(p);
        (!result.is_nan() && !result.is_inf()).then_some(result)
    }
    fn step(&self, c: &Self, p: usize) -> Option<(Self, BigFloat)> {
        // Outward intervals enclose the exact operation on the stored centers.
        // Their differences from the rounded center supply its local error ball.
        let two = BigFloat::from(2u8);
        let re = self
            .re
            .mul(&self.re, p, R::ToEven)
            .sub(&self.im.mul(&self.im, p, R::ToEven), p, R::ToEven)
            .add(&c.re, p, R::ToEven);
        let im = self
            .re
            .mul(&self.im, p, R::ToEven)
            .mul(&two, p, R::ToEven)
            .add(&c.im, p, R::ToEven);
        let lo_re = self
            .re
            .mul(&self.re, p, R::Down)
            .sub(&self.im.mul(&self.im, p, R::Up), p, R::Down)
            .add(&c.re, p, R::Down);
        let hi_re = self
            .re
            .mul(&self.re, p, R::Up)
            .sub(&self.im.mul(&self.im, p, R::Down), p, R::Up)
            .add(&c.re, p, R::Up);
        let lo_im = self
            .re
            .mul(&self.im, p, R::Down)
            .mul(&two, p, R::Down)
            .add(&c.im, p, R::Down);
        let hi_im = self
            .re
            .mul(&self.im, p, R::Up)
            .mul(&two, p, R::Up)
            .add(&c.im, p, R::Up);
        if [&lo_re, &hi_re, &lo_im, &hi_im]
            .into_iter()
            .any(|v| v.is_nan() || v.is_inf())
        {
            return None;
        }
        let difference = |lo: BigFloat, hi: BigFloat, center: &BigFloat| {
            let abs = |mut v: BigFloat| {
                if v.sign() == Some(Sign::Neg) {
                    v.inv_sign();
                }
                v
            };
            let a = abs(lo.sub(center, p, R::Down));
            let b = abs(hi.sub(center, p, R::Up));
            if a.cmp(&b).is_some_and(|order| order >= 0) {
                a
            } else {
                b
            }
        };
        let error = Self {
            re: difference(lo_re, hi_re, &re),
            im: difference(lo_im, hi_im, &im),
        }
        .norm_upper(p);
        if re.is_nan()
            || im.is_nan()
            || error.is_nan()
            || re.is_inf()
            || im.is_inf()
            || error.is_inf()
        {
            return None;
        }
        Some((Self { re, im }, error))
    }
}
fn strictly_inside(distance: &BigFloat, radius: &BigFloat, outer: &BigFloat, p: usize) -> bool {
    distance
        .add(radius, p, R::Up)
        .cmp(outer)
        .is_some_and(|v| v < 0)
}
fn trap(
    z: &Complex,
    radius: &BigFloat,
    centers: &VecDeque<Complex>,
    c: &Complex,
    p: usize,
) -> Option<usize> {
    let last = centers.len().checked_sub(1)?;
    let mut candidate = None;
    for period in 1..=last.min(4096) {
        let separation = z.distance_upper(&centers[last - period], p)?;
        if candidate
            .as_ref()
            .is_none_or(|(_, old): &(usize, BigFloat)| separation.cmp(old).is_some_and(|v| v < 0))
        {
            candidate = Some((period, separation));
        }
    }
    let (period, separation) = candidate?;
    let minimum = BigFloat::from_f64(2f64.powi(-(p as i32 / 2)), p);
    let two = BigFloat::from(2u8);
    for factor in [2u8, 8, 32, 128] {
        // Guesses only choose candidate disks; strict inclusion below is the proof.
        // Each disk contains the complete critical-orbit enclosure at this prefix.
        let mut initial = radius
            .add(&separation, p, R::Up)
            .mul(&BigFloat::from(factor), p, R::Up);
        if initial.cmp(&minimum).is_some_and(|v| v < 0) {
            initial = minimum.clone();
        }
        if z.norm_upper(p)
            .add(&initial, p, R::Up)
            .cmp(&two)
            .is_none_or(|v| v > 0)
        {
            continue;
        }
        let mut center = z.clone();
        let mut bound = initial.clone();
        let mut bounded = true;
        for _ in 0..period {
            let Some((next, next_radius)) = center.advance(&bound, c, p) else {
                bounded = false;
                break;
            };
            center = next;
            bound = next_radius;
            if center
                .norm_upper(p)
                .add(&bound, p, R::Up)
                .cmp(&two)
                .is_none_or(|v| v > 0)
            {
                bounded = false;
                break;
            }
        }
        if bounded && strictly_inside(&center.distance_upper(z, p)?, &bound, &initial, p) {
            return Some(period);
        }
    }
    None
}
pub fn membership(c: [f64; 2], p: usize, budget: usize) -> Option<MembershipProof> {
    if !c.iter().all(|x| x.is_finite()) || ![128, 256, 512].contains(&p) || budget > 32768 {
        return None;
    }
    let parameter = Complex {
        re: BigFloat::from_f64(c[0], p),
        im: BigFloat::from_f64(c[1], p),
    };
    let mut z = Complex {
        re: BigFloat::new(p),
        im: BigFloat::new(p),
    };
    let mut radius = BigFloat::new(p);
    let two = BigFloat::from(2u8);
    let one = BigFloat::from(1u8);
    let mut centers = VecDeque::from([z.clone()]);
    for n in 0..budget {
        let threshold = two.add(&radius, p, R::Up);
        if z.norm_squared(p, R::Down)
            .cmp(&threshold.mul(&threshold, p, R::Up))
            .is_some_and(|order| order > 0)
        {
            return Some(MembershipProof::Outside(EscapeProof {
                update: n,
                precision: p,
                radius_exponent: radius.exponent(),
            }));
        }
        // A lost point enclosure is a refusal, never a membership decision.
        if radius.cmp(&one).is_none_or(|order| order >= 0) {
            return None;
        }
        let (next, next_radius) = z.advance(&radius, &parameter, p)?;
        z = next;
        radius = next_radius;
        centers.push_back(z.clone());
        if centers.len() > 4097 {
            centers.pop_front();
        }
        if n + 1 >= 512 && (n + 1).is_power_of_two() {
            if let Some(period) = trap(&z, &radius, &centers, &parameter, p) {
                return Some(MembershipProof::Inside {
                    prefix: n + 1,
                    period,
                    precision: p,
                });
            }
        }
    }
    None
}
pub fn escape(c: [f64; 2], p: usize, budget: usize) -> Option<EscapeProof> {
    match membership(c, p, budget)? {
        MembershipProof::Outside(proof) => Some(proof),
        MembershipProof::Inside { .. } => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_escape_budget_and_radius() {
        for p in [128, 256, 512] {
            assert!(membership([1., 0.], p, 3).is_none());
            assert!(matches!(
                membership([1., 0.], p, 4),
                Some(MembershipProof::Outside(EscapeProof { update: 3, .. }))
            ));
            assert!(membership([2., 0.], p, 2).is_none());
            assert!(matches!(
                membership([2., 0.], p, 3),
                Some(MembershipProof::Outside(EscapeProof { update: 2, .. }))
            ));
        }
    }
    #[test]
    fn interior_requires_a_strict_trap() {
        let one = BigFloat::from(1u8);
        let two = BigFloat::from(2u8);
        let three = BigFloat::from(3u8);
        assert!(!strictly_inside(&one, &one, &two, 128));
        assert!(strictly_inside(&one, &one, &three, 128));
        for p in [128, 256, 512] {
            assert!(matches!(
                membership([0., 0.], p, 512),
                Some(MembershipProof::Inside { period: 1, .. })
            ));
            for c in [[0.25, 0.], [-0.75, 0.]] {
                assert!(!matches!(
                    membership(c, p, 2048),
                    Some(MembershipProof::Inside { .. })
                ));
            }
        }
    }
    #[test]
    fn invalid_requests_refuse() {
        for c in [[f64::NAN, 0.], [0., f64::INFINITY]] {
            assert!(membership(c, 128, 512).is_none());
        }
        assert!(membership([0., 0.], 64, 512).is_none());
        assert!(membership([0., 0.], 128, 32769).is_none());
        assert!(membership([0., 0.], 128, 0).is_none());
    }
    #[test]
    fn recorded_crossing_certificates() {
        assert!(matches!(
            membership([0.3204946517944336, 0.04734092950820923], 128, 32768),
            Some(MembershipProof::Outside(EscapeProof { update: 32747, .. }))
        ));
        assert!(matches!(
            membership([0.32049988639807619, 0.04733531447355364], 128, 32768),
            Some(MembershipProof::Inside { period: 85, .. })
        ));
        assert!(matches!(
            membership([0.35659381055433242, 0.06404656757776669], 128, 32768),
            Some(MembershipProof::Inside { period: 308, .. })
        ));
    }
}
