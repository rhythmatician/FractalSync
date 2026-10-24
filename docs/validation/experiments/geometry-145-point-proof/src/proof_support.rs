//! Shared directed-rounding primitives for parameter enclosures.
use super::point_proof::Complex;
use astro_float::{BigFloat, RoundingMode as R, Sign};
pub(super) fn finite(x: &BigFloat) -> bool {
    !x.is_nan() && !x.is_inf()
}
pub(super) fn error(lo: BigFloat, hi: BigFloat, center: &BigFloat, p: usize) -> Option<BigFloat> {
    if !finite(&lo) || !finite(&hi) || !finite(center) {
        return None;
    }
    let abs = |mut x: BigFloat| {
        if x.sign() == Some(Sign::Neg) {
            x.inv_sign();
        }
        x
    };
    let a = abs(lo.sub(center, p, R::Down));
    let b = abs(hi.sub(center, p, R::Up));
    if !finite(&a) || !finite(&b) {
        return None;
    }
    Some(if a.cmp(&b).is_some_and(|v| v >= 0) {
        a
    } else {
        b
    })
}
pub(super) fn derivative(z: &Complex, d: &Complex, p: usize) -> Option<(Complex, BigFloat)> {
    let two = BigFloat::from(2u8);
    let one = BigFloat::from(1u8);
    let real = |round: R, opposite: R| {
        z.re.mul(&d.re, p, round)
            .sub(&z.im.mul(&d.im, p, opposite), p, round)
            .mul(&two, p, round)
            .add(&one, p, round)
    };
    let imaginary = |round: R| {
        z.re.mul(&d.im, p, round)
            .add(&z.im.mul(&d.re, p, round), p, round)
            .mul(&two, p, round)
    };
    let re = real(R::ToEven, R::ToEven);
    let im = imaginary(R::ToEven);
    let roundoff = Complex {
        re: error(real(R::Down, R::Up), real(R::Up, R::Down), &re, p)?,
        im: error(imaginary(R::Down), imaginary(R::Up), &im, p)?,
    }
    .norm_upper(p);
    finite(&roundoff).then_some((Complex { re, im }, roundoff))
}
