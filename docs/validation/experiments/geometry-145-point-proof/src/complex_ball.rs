//! Shared high-precision complex-ball arithmetic.
use astro_float::{BigFloat, RoundingMode as R, Sign};
#[derive(Clone)]
pub(super) struct Complex {
    pub(super) re: BigFloat,
    pub(super) im: BigFloat,
}
impl Complex {
    pub(super) fn norm_squared(&self, p: usize, round: R) -> BigFloat {
        self.re
            .mul(&self.re, p, round)
            .add(&self.im.mul(&self.im, p, round), p, round)
    }
    pub(super) fn norm_upper(&self, p: usize) -> BigFloat {
        self.norm_squared(p, R::Up).sqrt(p, R::Up)
    }
    pub(super) fn advance(
        &self,
        radius: &BigFloat,
        c: &Self,
        p: usize,
    ) -> Option<(Self, BigFloat)> {
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
    pub(super) fn distance_upper(&self, other: &Self, p: usize) -> Option<BigFloat> {
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
    pub(super) fn step(&self, c: &Self, p: usize) -> Option<(Self, BigFloat)> {
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
