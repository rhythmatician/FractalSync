//! Local parameter-dependent contraction certificate for periodic-cycle branches.
//! No global component or nearest-boundary claim is made here.
use super::complex_ball::Complex as BoundComplex;
use super::proof_support::{error, finite};
use super::{cycle, C, P};
use astro_float::{BigFloat, RoundingMode as R};

type B = BigFloat;
fn zero() -> C {
    C::new(0., 0.)
}
fn abs(a: &C) -> B {
    BoundComplex {
        re: a.re.clone(),
        im: a.im.clone(),
    }
    .norm_upper(P)
}
fn add(a: &C, b: &C) -> Option<(C, B)> {
    let center = a.add(b);
    let re = error(
        a.re.add(&b.re, P, R::Down),
        a.re.add(&b.re, P, R::Up),
        &center.re,
        P,
    )?;
    let im = error(
        a.im.add(&b.im, P, R::Down),
        a.im.add(&b.im, P, R::Up),
        &center.im,
        P,
    )?;
    Some((center, abs(&C { re, im })))
}
fn mul(a: &C, b: &C) -> Option<(C, B)> {
    let center = a.mul(b);
    let re = error(
        a.re.mul(&b.re, P, R::Down)
            .sub(&a.im.mul(&b.im, P, R::Up), P, R::Down),
        a.re.mul(&b.re, P, R::Up)
            .sub(&a.im.mul(&b.im, P, R::Down), P, R::Up),
        &center.re,
        P,
    )?;
    let im = error(
        a.re.mul(&b.im, P, R::Down)
            .add(&a.im.mul(&b.re, P, R::Down), P, R::Down),
        a.re.mul(&b.im, P, R::Up)
            .add(&a.im.mul(&b.re, P, R::Up), P, R::Up),
        &center.im,
        P,
    )?;
    Some((center, abs(&C { re, im })))
}
#[derive(Clone)]
struct Affine {
    center: C,
    coefficients: [C; 3],
    remainder: B,
}
impl Affine {
    fn constant(center: C) -> Self {
        Self {
            center,
            coefficients: std::array::from_fn(|_| zero()),
            remainder: B::new(P),
        }
    }
    fn variable(center: C, coefficients: [C; 3]) -> Self {
        Self {
            center,
            coefficients,
            remainder: B::new(P),
        }
    }
    fn finite(&self) -> bool {
        self.center.finite()
            && self.coefficients.iter().all(C::finite)
            && finite(&self.remainder)
            && self.remainder.cmp(&B::new(P)).is_some_and(|v| v >= 0)
    }
    fn perturbation(&self, radii: &[B; 3]) -> B {
        let mut value = self.remainder.clone();
        for i in 0..3 {
            value = value.add(
                &abs(&self.coefficients[i]).mul(&radii[i], P, R::Up),
                P,
                R::Up,
            );
        }
        value
    }
    fn norm(&self, radii: &[B; 3]) -> B {
        abs(&self.center).add(&self.perturbation(radii), P, R::Up)
    }
    fn neg(&self) -> Self {
        let neg = |x: &C| {
            let mut y = x.clone();
            y.re.inv_sign();
            y.im.inv_sign();
            y
        };
        Self {
            center: neg(&self.center),
            coefficients: std::array::from_fn(|i| neg(&self.coefficients[i])),
            remainder: self.remainder.clone(),
        }
    }
    fn add(&self, b: &Self, radii: &[B; 3]) -> Option<Self> {
        let (center, roundoff) = add(&self.center, &b.center)?;
        let mut remainder = self
            .remainder
            .add(&b.remainder, P, R::Up)
            .add(&roundoff, P, R::Up);
        let mut coefficients = std::array::from_fn(|_| zero());
        for i in 0..3 {
            let (value, error) = add(&self.coefficients[i], &b.coefficients[i])?;
            coefficients[i] = value;
            remainder = remainder.add(&error.mul(&radii[i], P, R::Up), P, R::Up);
        }
        let result = Self {
            center,
            coefficients,
            remainder,
        };
        result.finite().then_some(result)
    }
    fn sub(&self, b: &Self, radii: &[B; 3]) -> Option<Self> {
        self.add(&b.neg(), radii)
    }
    fn mul(&self, b: &Self, radii: &[B; 3]) -> Option<Self> {
        let (center, roundoff) = mul(&self.center, &b.center)?;
        let mut remainder = abs(&self.center)
            .mul(&b.remainder, P, R::Up)
            .add(&abs(&b.center).mul(&self.remainder, P, R::Up), P, R::Up)
            .add(
                &self
                    .perturbation(radii)
                    .mul(&b.perturbation(radii), P, R::Up),
                P,
                R::Up,
            )
            .add(&roundoff, P, R::Up);
        let mut coefficients = std::array::from_fn(|_| zero());
        for i in 0..3 {
            let (a, ea) = mul(&self.center, &b.coefficients[i])?;
            let (bb, eb) = mul(&b.center, &self.coefficients[i])?;
            let (value, ec) = add(&a, &bb)?;
            coefficients[i] = value;
            remainder = remainder.add(
                &ea.add(&eb, P, R::Up)
                    .add(&ec, P, R::Up)
                    .mul(&radii[i], P, R::Up),
                P,
                R::Up,
            );
        }
        let result = Self {
            center,
            coefficients,
            remainder,
        };
        result.finite().then_some(result)
    }
    fn scale(&self, c: &C, radii: &[B; 3]) -> Option<Self> {
        self.mul(&Self::constant(c.clone()), radii)
    }
}
struct AffineJet {
    z: Affine,
    m: Affine,
    a: Affine,
    b: Affine,
    d: Affine,
}
fn evaluate(z: &Affine, c: &Affine, period: usize, radii: &[B; 3]) -> Option<AffineJet> {
    let one = Affine::constant(C::new(1., 0.));
    let two = C::new(2., 0.);
    let nil = Affine::constant(zero());
    let mut j = AffineJet {
        z: z.clone(),
        m: one.clone(),
        a: nil.clone(),
        b: nil.clone(),
        d: nil,
    };
    for _ in 0..period {
        j = AffineJet {
            z: j.z.mul(&j.z, radii)?.add(c, radii)?,
            m: j.z.mul(&j.m, radii)?.scale(&two, radii)?,
            a: j.z.mul(&j.a, radii)?.scale(&two, radii)?.add(&one, radii)?,
            b: j.m
                .mul(&j.m, radii)?
                .add(&j.z.mul(&j.b, radii)?, radii)?
                .scale(&two, radii)?,
            d: j.a
                .mul(&j.m, radii)?
                .add(&j.z.mul(&j.d, radii)?, radii)?
                .scale(&two, radii)?,
        };
    }
    Some(j)
}
#[derive(Clone)]
struct Model {
    period: usize,
    parameter: C,
    parameter_radius: B,
    centers: [C; 2],
    predictors: [C; 2],
    requested: [B; 2],
    inclusion: [B; 2],
}
#[derive(Clone)]
pub(super) struct Certificate {
    mu_radius: f64,
    z_error: f64,
    c_error: f64,
    inclusion: [f64; 2],
    contraction: f64,
    model: Model,
}
impl Certificate {
    pub(super) fn period(&self) -> usize {
        self.model.period
    }
}
impl std::fmt::Debug for Certificate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Certificate")
            .field("period", &self.period())
            .field("mu_radius", &self.mu_radius)
            .field("z_error", &self.z_error)
            .field("c_error", &self.c_error)
            .field("inclusion", &self.inclusion)
            .field("contraction", &self.contraction)
            .finish()
    }
}
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Refusal {
    InvalidRequest,
    NumericalFailure,
}
/// For every complex delta in the supplied closed disk, certifies one root of
/// (f_c^period(z)-z, (f_c^period)'(z)-mu-delta) in the predicted polydisk.
/// `z_error` and `c_error` bound deviations from the linear predictors, not
/// deviations from the center parameter itself. No attraction is asserted.
pub(super) fn certify(
    z: &C,
    c: &C,
    mu: &C,
    period: usize,
    mu_radius: f64,
    z_error: f64,
    c_error: f64,
) -> Result<Certificate, Refusal> {
    if !z.finite()
        || !c.finite()
        || !mu.finite()
        || period == 0
        || period > 4096
        || ![mu_radius, z_error, c_error]
            .into_iter()
            .all(f64::is_finite)
        || mu_radius < 0.
        || z_error <= 0.
        || c_error <= 0.
    {
        return Err(Refusal::InvalidRequest);
    }
    let numerical = || -> Option<Certificate> {
        let j = cycle(z, c, period)?;
        let one = C::new(1., 0.);
        let det = j.m.sub(&one).mul(&j.d).sub(&j.a.mul(&j.b));
        // This rounded matrix is a fixed exact dyadic preconditioner in the proof.
        let inverse = [
            [j.d.div(&det), j.a.scale(-1.).div(&det)],
            [j.b.scale(-1.).div(&det), j.m.sub(&one).div(&det)],
        ];
        if !inverse.iter().flatten().all(C::finite) {
            return None;
        }
        let radii = [
            B::from_f64(mu_radius, P),
            B::from_f64(z_error, P),
            B::from_f64(c_error, P),
        ];
        let az = Affine::variable(z.clone(), [inverse[0][1].clone(), one.clone(), zero()]);
        let ac = Affine::variable(c.clone(), [inverse[1][1].clone(), zero(), one.clone()]);
        let amu = Affine::variable(mu.clone(), [one.clone(), zero(), zero()]);
        let evaluated = evaluate(&az, &ac, period, &radii)?;
        let residual = [
            evaluated.z.sub(&az, &radii)?,
            evaluated.m.sub(&amu, &radii)?,
        ];
        let errors = [
            Affine::variable(zero(), [zero(), one.clone(), zero()]),
            Affine::variable(zero(), [zero(), zero(), one.clone()]),
        ];
        let mut inclusion = [B::new(P), B::new(P)];
        for row in 0..2 {
            let correction = residual[0]
                .scale(&inverse[row][0], &radii)?
                .add(&residual[1].scale(&inverse[row][1], &radii)?, &radii)?;
            inclusion[row] = errors[row].sub(&correction, &radii)?.norm(&radii);
        }
        let jacobian = [
            [
                evaluated.m.sub(&Affine::constant(one), &radii)?,
                evaluated.a,
            ],
            [evaluated.b, evaluated.d],
        ];
        let mut contraction = B::new(P);
        for row in 0..2 {
            let mut bound = B::new(P);
            for column in 0..2 {
                let value = jacobian[0][column].scale(&inverse[row][0], &radii)?.add(
                    &jacobian[1][column].scale(&inverse[row][1], &radii)?,
                    &radii,
                )?;
                let identity = Affine::constant(C::new(if row == column { 1. } else { 0. }, 0.));
                bound = bound.add(
                    &identity
                        .sub(&value, &radii)?
                        .norm(&radii)
                        .mul(&radii[column + 1], P, R::Up)
                        .div(&radii[row + 1], P, R::Up),
                    P,
                    R::Up,
                );
            }
            if !finite(&bound) {
                return None;
            }
            if bound.cmp(&contraction).is_some_and(|v| v > 0) {
                contraction = bound;
            }
        }
        // Comparisons use outward high-precision bounds, never converted f64 values.
        if !inclusion.iter().all(finite) || !finite(&contraction) {
            return None;
        }
        let included = inclusion[0].cmp(&radii[1]).is_some_and(|v| v < 0)
            && inclusion[1].cmp(&radii[2]).is_some_and(|v| v < 0);
        let contracting = contraction.cmp(&B::from(1u8)).is_some_and(|v| v < 0);
        let conversion = |x: &B| super::number(x);
        eprintln!("[BRANCH_BOUNDS145] inclusion_z={} inclusion_c={} contraction={} included={included} contracting={contracting}",conversion(&inclusion[0]),conversion(&inclusion[1]),conversion(&contraction));
        Some(Certificate {
            mu_radius,
            z_error,
            c_error,
            inclusion: [conversion(&inclusion[0]), conversion(&inclusion[1])],
            contraction: conversion(&contraction),
            model: Model {
                period,
                parameter: mu.clone(),
                parameter_radius: radii[0].clone(),
                centers: [z.clone(), c.clone()],
                predictors: [inverse[0][1].clone(), inverse[1][1].clone()],
                requested: [radii[1].clone(), radii[2].clone()],
                inclusion: inclusion.clone(),
            },
        })
        .filter(|_| included && contracting)
    };
    // Failure is conservative. Diagnostic distinctions are intentionally limited.
    numerical().ok_or(Refusal::NumericalFailure)
}

fn subtract(a: &C, b: &C) -> Option<(C, B)> {
    add(a, &b.scale(-1.))
}
fn distance(a: &C, b: &C) -> Option<B> {
    let (center, roundoff) = subtract(a, b)?;
    let bound = abs(&center).add(&roundoff, P, R::Up);
    finite(&bound).then_some(bound)
}
fn contains(model: &Model, parameter: &C) -> Option<bool> {
    Some(
        distance(parameter, &model.parameter)?
            .cmp(&model.parameter_radius)
            .is_some_and(|v| v <= 0),
    )
}
/// Rounded linear predictor with an outward arithmetic-error disk. Does not
/// include the proven root error; callers distinguish that from the domain.
fn predictor(model: &Model, parameter: &C, index: usize) -> Option<(C, B)> {
    let (delta, e0) = subtract(parameter, &model.parameter)?;
    let (linear, e1) = mul(&model.predictors[index], &delta)?;
    let (center, e2) = add(&model.centers[index], &linear)?;
    let radius = abs(&model.predictors[index])
        .mul(&e0, P, R::Up)
        .add(&e1, P, R::Up)
        .add(&e2, P, R::Up);
    finite(&radius).then_some((center, radius))
}
#[derive(Debug, PartialEq, Eq)]
pub(super) enum GlueRefusal {
    InvalidParameter,
    PeriodMismatch,
    OutsidePatch,
    NumericalFailure,
    DistinctBranch,
}
#[derive(Debug)]
pub(super) struct Glue {
    pub phase_advance: usize,
    pub z_margin: f64,
    pub c_margin: f64,
}
/// Proves the two certified roots at this exact multiplier belong to the same
/// periodic cycle, allowing an explicitly validated cycle phase advance.
/// Both parameter coordinates must lie in the certified multiplier disks.
pub(super) fn glue(
    old: &Certificate,
    next: &Certificate,
    parameter: &C,
) -> Result<Glue, GlueRefusal> {
    if !parameter.finite() {
        return Err(GlueRefusal::InvalidParameter);
    }
    if old.period() != next.period() {
        return Err(GlueRefusal::PeriodMismatch);
    }
    let numerical = || -> Option<Result<Glue, GlueRefusal>> {
        if !contains(&old.model, parameter)? || !contains(&next.model, parameter)? {
            return Some(Err(GlueRefusal::OutsidePatch));
        }
        let (mut z, z_arithmetic) = predictor(&old.model, parameter, 0)?;
        let (c, c_arithmetic) = predictor(&old.model, parameter, 1)?;
        let mut z_radius = z_arithmetic.add(&old.model.inclusion[0], P, R::Up);
        let c_radius = c_arithmetic.add(&old.model.inclusion[1], P, R::Up);
        let (next_z, next_z_arithmetic) = predictor(&next.model, parameter, 0)?;
        let (next_c, next_c_arithmetic) = predictor(&next.model, parameter, 1)?;
        let c_bound =
            distance(&c, &next_c)?
                .add(&c_radius, P, R::Up)
                .add(&next_c_arithmetic, P, R::Up);
        let c_margin = next.model.requested[1].sub(&c_bound, P, R::Down);
        if !finite(&c_margin) || !c_margin.cmp(&B::new(P)).is_some_and(|v| v > 0) {
            return Some(Err(GlueRefusal::DistinctBranch));
        }
        for phase in 0..old.period() {
            let z_bound =
                distance(&z, &next_z)?
                    .add(&z_radius, P, R::Up)
                    .add(&next_z_arithmetic, P, R::Up);
            let z_margin = next.model.requested[0].sub(&z_bound, P, R::Down);
            if finite(&z_margin) && z_margin.cmp(&B::new(P)).is_some_and(|v| v > 0) {
                return Some(Ok(Glue {
                    phase_advance: phase,
                    z_margin: super::number(&z_margin),
                    c_margin: super::number(&c_margin),
                }));
            }
            let (square, e1) = mul(&z, &z)?;
            let (new_z, e2) = add(&square, &c)?;
            z_radius = B::from(2u8)
                .mul(&abs(&z), P, R::Up)
                .mul(&z_radius, P, R::Up)
                .add(&z_radius.mul(&z_radius, P, R::Up), P, R::Up)
                .add(&c_radius, P, R::Up)
                .add(&e1, P, R::Up)
                .add(&e2, P, R::Up);
            if !finite(&z_radius) {
                return None;
            }
            z = new_z;
        }
        Some(Err(GlueRefusal::DistinctBranch))
    };
    numerical().ok_or(GlueRefusal::NumericalFailure)?
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum SegmentRefusal {
    InvalidParameter,
    OutsidePatch,
    NotAttracting,
    NumericalFailure,
}
#[derive(Debug)]
pub(super) struct AttractingSegment {
    period: usize,
    multiplier_norm_upper: f64,
}
/// Certifies the continuous local root branch over the straight segment between
/// these exact dyadic multipliers. Convexity keeps the segment in the patch's
/// parameter disk and strictly inside the unit multiplier disk. This does not
/// identify a global component boundary or certify a nearest feature.
pub(super) fn attracting_segment(
    patch: &Certificate,
    start: &C,
    end: &C,
) -> Result<AttractingSegment, SegmentRefusal> {
    if !start.finite() || !end.finite() {
        return Err(SegmentRefusal::InvalidParameter);
    }
    let a = abs(start);
    let b = abs(end);
    if !finite(&a) || !finite(&b) {
        return Err(SegmentRefusal::NumericalFailure);
    }
    let bound = if a.cmp(&b).is_some_and(|v| v >= 0) {
        a
    } else {
        b
    };
    if !bound.cmp(&B::from(1u8)).is_some_and(|v| v < 0) {
        return Err(SegmentRefusal::NotAttracting);
    }
    if !contains(&patch.model, start).ok_or(SegmentRefusal::NumericalFailure)?
        || !contains(&patch.model, end).ok_or(SegmentRefusal::NumericalFailure)?
    {
        return Err(SegmentRefusal::OutsidePatch);
    }
    Ok(AttractingSegment {
        period: patch.period(),
        multiplier_norm_upper: super::number(&bound),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_cardioid_patch_is_certified() {
        let mu = C::new(-0.4, 0.6);
        let z = mu.scale(0.5);
        let c = mu.scale(0.5).sub(&mu.mul(&mu).scale(0.25));
        let certificate = certify(&z, &c, &mu, 1, 1e-4, 1e-8, 1e-8)
            .expect("closed-form regular branch must certify");
        assert!(certificate.contraction < 1.);
    }
    #[test]
    fn insufficient_quadratic_remainder_refuses() {
        let mu = C::new(-0.4, 0.6);
        let z = mu.scale(0.5);
        let c = mu.scale(0.5).sub(&mu.mul(&mu).scale(0.25));
        // c(mu+delta)-c(mu)-c'(mu)delta = -delta^2/4 exactly.
        assert!(certify(&z, &c, &mu, 1, 1e-4, 1e-8, 1e-10).is_err());
    }
    #[test]
    fn self_inclusion_without_contraction_refuses() {
        // At z=c=mu=0 and fixed mu, K_z=0 and K_c=-e_z^2.
        // Its image fits (r_z,r_c)=(1,1.5), but its weighted derivative bound is 4/3.
        assert!(certify(&zero(), &zero(), &zero(), 1, 0., 1., 1.5).is_err());
    }
    #[test]
    fn center_roundoff_must_not_disappear() {
        let mut negative_controls = 0;
        for numerator_re in 1..=8u8 {
            for numerator_im in 1..=8u8 {
                let mu = C {
                    re: B::from(numerator_re).div(&B::from(numerator_re + 2), P, R::ToEven),
                    im: B::from(numerator_im).div(&B::from(numerator_im + 3), P, R::ToEven),
                };
                let z = mu.scale(0.5);
                let c = mu.scale(0.5).sub(&mu.mul(&mu).scale(0.25));
                let q = 512;
                let two = B::from(2u8);
                let four = B::from(4u8);
                // Products of these exact 128-bit dyadics, and their sums, fit
                // exactly in 512 bits. This checks the true closed-form root.
                let exact_re = mu.re.div(&two, q, R::ToEven).sub(
                    &mu.re
                        .mul(&mu.re, q, R::ToEven)
                        .sub(&mu.im.mul(&mu.im, q, R::ToEven), q, R::ToEven)
                        .div(&four, q, R::ToEven),
                    q,
                    R::ToEven,
                );
                let exact_im = mu.im.div(&two, q, R::ToEven).sub(
                    &mu.re.mul(&mu.im, q, R::ToEven).div(&two, q, R::ToEven),
                    q,
                    R::ToEven,
                );
                let dr = exact_re.sub(&c.re, q, R::ToEven);
                let di = exact_im.sub(&c.im, q, R::ToEven);
                let distance_lower = dr
                    .mul(&dr, q, R::Down)
                    .add(&di.mul(&di, q, R::Down), q, R::Down)
                    .sqrt(q, R::Down);
                if distance_lower
                    .cmp(&B::from_f64(1e-50, q))
                    .is_some_and(|v| v > 0)
                {
                    negative_controls += 1;
                    assert!(certify(&z,&c,&mu,1,0.,1e-50,1e-50).is_err(),"false roundoff-free certificate at rational seeds {numerator_re}, {numerator_im}");
                }
            }
        }
        assert!(negative_controls >= 20);
    }

    #[test]
    fn all_input_domains_and_subnormal_radii_are_checked() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for slot in 0..6 {
                let mut values = [0.; 6];
                values[slot] = value;
                assert!(certify(
                    &C::new(values[0], values[1]),
                    &C::new(values[2], values[3]),
                    &C::new(values[4], values[5]),
                    1,
                    0.,
                    1e-8,
                    1e-8
                )
                .is_err());
            }
        }
        for radius in [0., -1., f64::NAN, f64::INFINITY] {
            assert!(certify(&zero(), &zero(), &zero(), 1, 0., radius, 1e-8).is_err());
            assert!(certify(&zero(), &zero(), &zero(), 1, 0., 1e-8, radius).is_err());
        }
        assert!(certify(&zero(), &zero(), &zero(), 4097, 0., 1e-8, 1e-8).is_err());
        let tiny = f64::from_bits(1);
        assert!(certify(&zero(), &zero(), &zero(), 1, 0., tiny, tiny).is_ok());
    }
    #[test]
    fn adjacent_cardioid_patches_glue_and_outside_parameters_refuse() {
        let make = |mu: C| {
            let z = mu.scale(0.5);
            let c = mu.scale(0.5).sub(&mu.mul(&mu).scale(0.25));
            certify(&z, &c, &mu, 1, 2e-4, 1e-7, 1e-7).unwrap()
        };
        let a = make(C::new(-0.4, 0.6));
        let b = make(C::new(-0.4, 0.6001));
        assert_eq!(
            glue(&a, &b, &C::new(-0.4, 0.60005)).unwrap().phase_advance,
            0
        );
        assert!(matches!(
            glue(&a, &b, &zero()),
            Err(GlueRefusal::OutsidePatch)
        ));
        assert!(matches!(
            glue(&a, &a, &zero()),
            Err(GlueRefusal::OutsidePatch)
        ));
        assert!(matches!(
            glue(&a, &b, &C::new(f64::NAN, 0.)),
            Err(GlueRefusal::InvalidParameter)
        ));
    }
    #[test]
    fn cycle_phase_is_checked_when_gluing_period_two() {
        let mu = C::new(-0.5, 0.2);
        let c = C::new(-1., 0.).add(&mu.scale(0.25));
        let mut z = zero();
        for _ in 0..4096 {
            z = z.mul(&z).add(&c);
        }
        let a = certify(&z, &c, &mu, 2, 1e-4, 1e-7, 1e-7).unwrap();
        let rotated = z.mul(&z).add(&c);
        let b = certify(&rotated, &c, &mu, 2, 1e-4, 1e-7, 1e-7).unwrap();
        assert_eq!(glue(&a, &b, &mu).unwrap().phase_advance, 1);
    }
    #[test]
    fn attracting_segment_requires_both_domain_and_multiplier_guards() {
        let mu = C::new(0.99, 0.);
        let z = mu.scale(0.5);
        let c = mu.scale(0.5).sub(&mu.mul(&mu).scale(0.25));
        let patch = certify(&z, &c, &mu, 1, 0.02, 1e-3, 1e-3).unwrap();
        assert!(attracting_segment(&patch, &C::new(0.98, 0.), &C::new(0.999, 0.)).is_ok());
        assert!(matches!(
            attracting_segment(&patch, &C::new(0.98, 0.), &C::new(1., 0.)),
            Err(SegmentRefusal::NotAttracting)
        ));
        assert!(matches!(
            attracting_segment(&patch, &C::new(0.98, 0.), &C::new(0., 0.)),
            Err(SegmentRefusal::OutsidePatch)
        ));
        assert!(matches!(
            attracting_segment(&patch, &C::new(f64::NAN, 0.), &mu),
            Err(SegmentRefusal::InvalidParameter)
        ));
    }
    #[test]
    fn distinct_period_three_components_do_not_glue() {
        let make = |seed: C| {
            let (z, c) = super::super::solve(zero(), seed, 3, &zero()).unwrap();
            certify(&z, &c, &zero(), 3, 1e-6, 1e-9, 1e-9).unwrap()
        };
        let a = make(C::new(-1.7548776662466927, 0.));
        let b = make(C::new(-0.1225611668766536, 0.7448617666197442));
        assert!(matches!(
            glue(&a, &b, &zero()),
            Err(GlueRefusal::DistinctBranch)
        ));
    }
    #[test]
    fn multiplying_a_remainder_disk_encloses_its_boundary() {
        let radii = std::array::from_fn(|_| B::new(P));
        let a = Affine::constant(C::new(3., 4.));
        let mut b = Affine::constant(zero());
        b.remainder = B::from(2u8);
        let product = a.mul(&b, &radii).unwrap();
        // The admissible input b=2i gives (3+4i)b=-8+6i, norm exactly ten.
        assert!(product
            .norm(&radii)
            .cmp(&B::from(10u8))
            .is_some_and(|v| v >= 0));
    }
    #[test]
    fn closed_form_roots_at_disk_boundary_fit_the_proven_predictor_errors() {
        let mu = C::new(-0.4, 0.6);
        let z = mu.scale(0.5);
        let c = mu.scale(0.5).sub(&mu.mul(&mu).scale(0.25));
        let h = 2f64.powi(-12);
        let patch = certify(&z, &c, &mu, 1, h, 1e-6, 1e-6).unwrap();
        for delta in [
            C::new(h, 0.),
            C::new(-h, 0.),
            C::new(0., h),
            C::new(0., -h),
            C::new(h / 2., h / 2.),
        ] {
            let parameter = mu.add(&delta);
            let exact_z = parameter.scale(0.5);
            let exact_c = parameter
                .scale(0.5)
                .sub(&parameter.mul(&parameter).scale(0.25));
            for (index, root) in [exact_z, exact_c].iter().enumerate() {
                let (predicted, arithmetic) = predictor(&patch.model, &parameter, index).unwrap();
                assert!(distance(root, &predicted)
                    .unwrap()
                    .cmp(&patch.model.inclusion[index].add(&arithmetic, P, R::Up))
                    .is_some_and(|v| v <= 0));
            }
        }
    }
    #[test]
    fn singular_period_two_multiplier_branch_refuses() {
        // At the period-two root, the two cycle phases coalesce into a fixed
        // point. The two-equation Jacobian is singular even at an exact seed.
        assert!(certify(
            &C::new(-0.5, 0.),
            &C::new(-0.75, 0.),
            &C::new(1., 0.),
            2,
            0.,
            1e-12,
            1e-12
        )
        .is_err());
    }
    #[test]
    fn wrong_center_or_invalid_input_refuses() {
        assert!(certify(&zero(), &zero(), &C::new(0.5, 0.), 1, 0., 1e-12, 1e-12).is_err());
        assert!(certify(&zero(), &zero(), &zero(), 0, 0., 1e-12, 1e-12).is_err());
        assert!(certify(&zero(), &zero(), &zero(), 1, f64::NAN, 1e-12, 1e-12).is_err());
    }
}
