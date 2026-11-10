//! High-precision quadratic parameter-disk enclosure. Experimental, not runtime.
use super::point_proof::{trap, Complex, EscapeProof, MembershipProof};
use super::proof_support::{derivative, error, finite};
use astro_float::{BigFloat, RoundingMode as R};
use std::collections::VecDeque;
fn quadratic_coefficient(
    z: &Complex,
    a: &Complex,
    b: &Complex,
    p: usize,
) -> Option<(Complex, BigFloat)> {
    let two = BigFloat::from(2u8);
    let real = |round: R, opposite: R| {
        a.re.mul(&a.re, p, round)
            .sub(&a.im.mul(&a.im, p, opposite), p, round)
            .add(
                &z.re
                    .mul(&b.re, p, round)
                    .sub(&z.im.mul(&b.im, p, opposite), p, round)
                    .mul(&two, p, round),
                p,
                round,
            )
    };
    let imaginary = |round: R| {
        a.re.mul(&a.im, p, round)
            .add(&z.re.mul(&b.im, p, round), p, round)
            .add(&z.im.mul(&b.re, p, round), p, round)
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
pub(super) fn membership(
    x: f64,
    y: f64,
    width: f64,
    p: usize,
    budget: usize,
) -> Option<MembershipProof> {
    if ![x, y, width].into_iter().all(f64::is_finite)
        || width < 0.
        || width > 4.
        || ![128, 256, 512].contains(&p)
        || budget > 32768
    {
        return None;
    }
    // The center is accepted only if the two directed sums agree. Thus no
    // unaccounted rounding of the parameter center enters the affine model.
    let half = BigFloat::from_f64(width, p).div(&BigFloat::from(2u8), p, R::Up);
    let center = |v: f64| {
        let a = BigFloat::from_f64(v, p).add(&half, p, R::Down);
        let b = BigFloat::from_f64(v, p).add(&half, p, R::Up);
        (a.cmp(&b) == Some(0) && finite(&a)).then_some(a)
    };
    let parameter = Complex {
        re: center(x)?,
        im: center(y)?,
    };
    let cr = half.mul(&BigFloat::from(2u8).sqrt(p, R::Up), p, R::Up);
    let mut z = Complex {
        re: BigFloat::new(p),
        im: BigFloat::new(p),
    };
    let mut d = z.clone();
    let mut b = z.clone();
    let cr2 = cr.mul(&cr, p, R::Up);
    let mut remainder = BigFloat::new(p);
    let two = BigFloat::from(2u8);
    let one = BigFloat::from(1u8);
    let mut centers = VecDeque::from([z.clone()]);
    for n in 0..budget {
        let linear = d.norm_upper(p).mul(&cr, p, R::Up);
        let nonlinear = b
            .norm_upper(p)
            .mul(&cr2, p, R::Up)
            .add(&remainder, p, R::Up);
        let radius = linear.add(&nonlinear, p, R::Up);
        if !finite(&radius) {
            return None;
        }
        let threshold = two.add(&radius, p, R::Up);
        if z.norm_squared(p, R::Down)
            .cmp(&threshold.mul(&threshold, p, R::Up))
            .is_some_and(|v| v > 0)
        {
            return Some(MembershipProof::Outside(EscapeProof {
                update: n,
                precision: p,
                radius_exponent: radius.exponent(),
            }));
        }
        if radius.cmp(&one).is_none_or(|v| v >= 0) {
            return None;
        }
        let (next, ez) = z.step(&parameter, p)?;
        let (next_d, ed) = derivative(&z, &d, p)?;
        let (next_b, eb) = quadratic_coefficient(&z, &d, &b, p)?;
        // z(c0+delta) = z_center + d_center*delta + b_center*delta^2 + error.
        // Retain quadratic dependence; enclose the remaining higher terms and
        // all three rounded coefficient updates with outward arithmetic.
        remainder = two
            .mul(&z.norm_upper(p), p, R::Up)
            .mul(&remainder, p, R::Up)
            .add(
                &two.mul(&linear, p, R::Up).mul(&nonlinear, p, R::Up),
                p,
                R::Up,
            )
            .add(&nonlinear.mul(&nonlinear, p, R::Up), p, R::Up)
            .add(&ez, p, R::Up)
            .add(&ed.mul(&cr, p, R::Up), p, R::Up)
            .add(&eb.mul(&cr2, p, R::Up), p, R::Up);
        z = next;
        d = next_d;
        b = next_b;
        centers.push_back(z.clone());
        if centers.len() > 4097 {
            centers.pop_front();
        }
        if n + 1 >= 512 && (n + 1).is_power_of_two() {
            let total = d
                .norm_upper(p)
                .mul(&cr, p, R::Up)
                .add(&b.norm_upper(p).mul(&cr2, p, R::Up), p, R::Up)
                .add(&remainder, p, R::Up);
            if finite(&total) {
                if let Some(period) = trap(&z, &total, &centers, &parameter, p, &cr) {
                    return Some(MembershipProof::Inside {
                        prefix: n + 1,
                        period,
                        precision: p,
                    });
                }
            }
        }
    }
    None
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trap_must_include_parameter_uncertainty() {
        let p = 128;
        let zero = BigFloat::new(p);
        let z = Complex {
            re: zero.clone(),
            im: zero.clone(),
        };
        let history = VecDeque::from([z.clone(), z.clone()]);
        assert!(trap(&z, &zero, &history, &z, p, &BigFloat::from(3u8)).is_none());
    }
    #[test]
    fn invalid_boxes_refuse_at_all_precisions() {
        for p in [128, 256, 512] {
            assert!(membership(f64::NAN, 0., 0., p, 512).is_none());
            assert!(membership(0., 0., f64::INFINITY, p, 512).is_none());
            assert!(membership(0., 0., -0.1, p, 512).is_none());
            assert!(membership(0., 0., 0., p, 32769).is_none());
            assert!(matches!(
                membership(-0.01, -0.01, 0.02, p, 512),
                Some(MembershipProof::Inside { period: 1, .. })
            ));
        }
        assert!(membership(0., 0., 0., 64, 512).is_none());
    }

    #[test]
    fn zero_width_matches_required_point_outcomes() {
        assert!(matches!(
            membership(1., 0., 0., 128, 4),
            Some(MembershipProof::Outside(EscapeProof { update: 3, .. }))
        ));
        assert!(membership(1., 0., 0., 128, 3).is_none());
        assert!(matches!(
            membership(0., 0., 0., 128, 512),
            Some(MembershipProof::Inside { period: 1, .. })
        ));
    }
    #[test]
    fn boxes_near_known_interior_and_exterior() {
        assert!(matches!(
            membership(-0.01, -0.01, 0.02, 128, 512),
            Some(MembershipProof::Inside { period: 1, .. })
        ));
        assert!(matches!(
            membership(1., 0., 0.001, 128, 512),
            Some(MembershipProof::Outside(_))
        ));
        assert!(membership(-1., -1., 2., 128, 512).is_none());
        assert!(membership(0., -2., 4., 128, 512).is_none());
    }
}
