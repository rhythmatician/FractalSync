//! High-precision numerical component tracing. Not a validated boundary cover.
use astro_float::{BigFloat, RoundingMode as R};
const P: usize = 128;
#[derive(Clone)]
struct C {
    re: BigFloat,
    im: BigFloat,
}
impl C {
    fn new(re: f64, im: f64) -> Self {
        Self {
            re: BigFloat::from_f64(re, P),
            im: BigFloat::from_f64(im, P),
        }
    }
    fn add(&self, b: &Self) -> Self {
        Self {
            re: self.re.add(&b.re, P, R::ToEven),
            im: self.im.add(&b.im, P, R::ToEven),
        }
    }
    fn sub(&self, b: &Self) -> Self {
        Self {
            re: self.re.sub(&b.re, P, R::ToEven),
            im: self.im.sub(&b.im, P, R::ToEven),
        }
    }
    fn mul(&self, b: &Self) -> Self {
        Self {
            re: self.re.mul(&b.re, P, R::ToEven).sub(
                &self.im.mul(&b.im, P, R::ToEven),
                P,
                R::ToEven,
            ),
            im: self.re.mul(&b.im, P, R::ToEven).add(
                &self.im.mul(&b.re, P, R::ToEven),
                P,
                R::ToEven,
            ),
        }
    }
    fn scale(&self, x: f64) -> Self {
        let b = BigFloat::from_f64(x, P);
        Self {
            re: self.re.mul(&b, P, R::ToEven),
            im: self.im.mul(&b, P, R::ToEven),
        }
    }
    fn norm2(&self) -> BigFloat {
        self.re
            .mul(&self.re, P, R::ToEven)
            .add(&self.im.mul(&self.im, P, R::ToEven), P, R::ToEven)
    }
    fn div(&self, b: &Self) -> Self {
        let d = b.norm2();
        Self {
            re: self
                .re
                .mul(&b.re, P, R::ToEven)
                .add(&self.im.mul(&b.im, P, R::ToEven), P, R::ToEven)
                .div(&d, P, R::ToEven),
            im: self
                .im
                .mul(&b.re, P, R::ToEven)
                .sub(&self.re.mul(&b.im, P, R::ToEven), P, R::ToEven)
                .div(&d, P, R::ToEven),
        }
    }
    fn finite(&self) -> bool {
        !self.re.is_nan() && !self.im.is_nan() && !self.re.is_inf() && !self.im.is_inf()
    }
    fn norm(&self) -> f64 {
        number(&self.norm2().sqrt(P, R::ToEven))
    }
    fn f64(&self) -> [f64; 2] {
        [number(&self.re), number(&self.im)]
    }
}
fn number(x: &BigFloat) -> f64 {
    x.to_string().parse().expect("decimal BigFloat output")
}
struct Jet {
    z: C,
    m: C,
    a: C,
    b: C,
    d: C,
}
fn cycle(z: &C, c: &C, p: usize) -> Option<Jet> {
    let mut j = Jet {
        z: z.clone(),
        m: C::new(1., 0.),
        a: C::new(0., 0.),
        b: C::new(0., 0.),
        d: C::new(0., 0.),
    };
    for _ in 0..p {
        j = Jet {
            z: j.z.mul(&j.z).add(c),
            m: j.z.mul(&j.m).scale(2.),
            a: j.z.mul(&j.a).scale(2.).add(&C::new(1., 0.)),
            b: j.m.mul(&j.m).add(&j.z.mul(&j.b)).scale(2.),
            d: j.a.mul(&j.m).add(&j.z.mul(&j.d)).scale(2.),
        };
        if !j.z.finite() || !j.m.finite() || !j.a.finite() || !j.b.finite() || !j.d.finite() {
            return None;
        }
    }
    Some(j)
}
fn solve(mut z: C, mut c: C, p: usize, mu: &C) -> Option<(C, C)> {
    let tolerance = BigFloat::from_f64(1e-54, P);
    let one = C::new(1., 0.);
    for iteration in 0..40 {
        let j = cycle(&z, &c, p)?;
        let f = j.z.sub(&z);
        let g = j.m.sub(mu);
        if f.norm2().cmp(&tolerance).is_some_and(|v| v < 0)
            && g.norm2().cmp(&tolerance).is_some_and(|v| v < 0)
        {
            return Some((z, c));
        }
        if iteration == 39 {
            eprintln!(
                "[HP_NEWTON_REFUSAL145] p={p} mu={:?} closure={} multiplier_error={}",
                mu.f64(),
                f.norm(),
                g.norm()
            );
        }
        let det = j.m.sub(&one).mul(&j.d).sub(&j.a.mul(&j.b));
        let dz = j.d.mul(&f).sub(&j.a.mul(&g)).div(&det);
        let dc = j.m.sub(&one).mul(&g).sub(&j.b.mul(&f)).div(&det);
        let old = f.norm2().add(&g.norm2(), P, R::ToEven);
        let mut accepted = None;
        for k in 0..24 {
            let alpha = 2f64.powi(-k);
            let nz = z.sub(&dz.scale(alpha));
            let nc = c.sub(&dc.scale(alpha));
            if !nz.finite() || !nc.finite() {
                continue;
            }
            if let Some(next) = cycle(&nz, &nc, p) {
                let nf = next.z.sub(&nz);
                let ng = next.m.sub(mu);
                if nf
                    .norm2()
                    .add(&ng.norm2(), P, R::ToEven)
                    .cmp(&old)
                    .is_some_and(|v| v < 0)
                {
                    accepted = Some((nz, nc));
                    break;
                }
            }
        }
        (z, c) = accepted?;
    }
    None
}

fn curve_derivatives(z: &C, c: &C, p: usize, mu: &C) -> Option<(C, C)> {
    let j = cycle(z, c, p)?;
    let one = C::new(1., 0.);
    let zero = C::new(0., 0.);
    let det = j.m.sub(&one).mul(&j.d).sub(&j.a.mul(&j.b));
    let mut minus_im = mu.im.clone();
    minus_im.inv_sign();
    let mu1 = C {
        re: minus_im,
        im: mu.re.clone(),
    };
    let z1 = j.a.mul(&mu1).scale(-1.).div(&det);
    let c1 = j.m.sub(&one).mul(&mu1).div(&det);
    let mut v = z.clone();
    let mut v1 = z1;
    let mut v2 = zero.clone();
    let mut m = one;
    let mut m1 = zero.clone();
    let mut m2 = zero;
    for _ in 0..p {
        let nv2 = v1.mul(&v1).add(&v.mul(&v2)).scale(2.);
        let nm2 = v2
            .mul(&m)
            .add(&v1.mul(&m1).scale(2.))
            .add(&v.mul(&m2))
            .scale(2.);
        let nv1 = v.mul(&v1).scale(2.).add(&c1);
        let nm1 = v1.mul(&m).add(&v.mul(&m1)).scale(2.);
        let nv = v.mul(&v).add(c);
        let nm = v.mul(&m).scale(2.);
        (v, v1, v2, m, m1, m2) = (nv, nv1, nv2, nm, nm1, nm2);
    }
    let rhs_f = v2.scale(-1.);
    let rhs_g = mu.scale(-1.).sub(&m2);
    let c2 =
        j.m.sub(&C::new(1., 0.))
            .mul(&rhs_g)
            .sub(&j.b.mul(&rhs_f))
            .div(&det);
    (c1.finite() && c2.finite()).then_some((c1, c2))
}
#[derive(Debug)]
struct DistanceJet {
    d: f64,
    g: [f64; 2],
    h: [[f64; 2]; 2],
}
fn projection(query: &C, seed_z: &C, seed_c: &C, p: usize, mut theta: f64) -> Option<DistanceJet> {
    let mut z = seed_z.clone();
    let mut c = seed_c.clone();
    for _ in 0..12 {
        let mu = C::new(theta.cos(), theta.sin());
        (z, c) = solve(z, c, p, &mu)?;
        let (d1, d2) = curve_derivatives(&z, &c, p, &mu)?;
        let a = d1.f64();
        let b = d2.f64();
        let r = c.sub(query).f64();
        let denominator = a[0] * a[0] + a[1] * a[1] + r[0] * b[0] + r[1] * b[1];
        if !denominator.is_finite() || denominator <= 0. {
            return None;
        }
        let delta = (r[0] * a[0] + r[1] * a[1]) / denominator;
        if delta.abs() < 1e-13 {
            let speed = a[0].hypot(a[1]);
            let d = r[0].hypot(r[1]);
            let k = (a[0] * b[1] - a[1] * b[0]) / speed.powi(3);
            let factor = -k / (1. - k * d);
            let t = [a[0] / speed, a[1] / speed];
            let inward_alignment = (r[0] * t[1] - r[1] * t[0]) / d;
            if !speed.is_finite()
                || speed <= 0.
                || !d.is_finite()
                || !inward_alignment.is_finite()
                || inward_alignment <= 0.
            {
                return None;
            }
            if !factor.is_finite() || d <= 0. || 1. - k * d <= 0. {
                return None;
            }
            return Some(DistanceJet {
                d,
                g: [-r[0] / d, -r[1] / d],
                h: [
                    [factor * t[0] * t[0], factor * t[0] * t[1]],
                    [factor * t[0] * t[1], factor * t[1] * t[1]],
                ],
            });
        }
        if !delta.is_finite() || delta.abs() > 0.1 {
            return None;
        }
        theta -= delta;
    }
    None
}
fn check_distance_derivatives(
    name: &str,
    query: &C,
    z: &C,
    c: &C,
    p: usize,
    theta: f64,
) -> Option<()> {
    let jet = projection(query, z, c, p, theta)?;
    let step = jet.d * 1e-3;
    if !step.is_finite() || step <= 0. {
        return None;
    }
    let mut gradient_error: f64 = 0.;
    let mut hessian_error: f64 = 0.;
    for axis in 0..2 {
        let offset = if axis == 0 {
            C::new(step, 0.)
        } else {
            C::new(0., step)
        };
        let plus = projection(&query.add(&offset), z, c, p, theta)?;
        let minus = projection(&query.sub(&offset), z, c, p, theta)?;
        let ge = ((plus.d - minus.d) / (2. * step) - jet.g[axis]).abs();
        if !ge.is_finite() {
            return None;
        }
        gradient_error = gradient_error.max(ge);
        for row in 0..2 {
            let he = ((plus.g[row] - minus.g[row]) / (2. * step) - jet.h[row][axis]).abs() * jet.d;
            if !he.is_finite() {
                return None;
            }
            hessian_error = hessian_error.max(he);
        }
    }
    println!("[HP_COMPONENT_JET145] name={name} jet={jet:?} step={step} gradient_error={gradient_error} distance_scaled_hessian_error={hessian_error}");
    assert!(
        gradient_error < 1e-5 && hessian_error < 1e-5,
        "analytic curve jet disagrees with neighboring queries"
    );
    Some(())
}
fn trace(name: &str, query: C, p: usize) -> Option<()> {
    let mut z = C::new(0., 0.);
    for _ in 0..32768 {
        z = z.mul(&z).add(&query);
        if !z.finite() {
            return None;
        }
    }
    let initial = cycle(&z, &query, p)?;
    let mu = initial.m;
    println!(
        "[HP_COMPONENT145] name={name} period={p} multiplier_norm={} closure={}",
        mu.norm(),
        initial.z.sub(&z).norm()
    );
    if mu.norm() >= 1. {
        return None;
    }
    let mut c = query.clone();
    for k in 1..=64 {
        (z, c) = solve(z, c, p, &mu.scale(1. - k as f64 / 64.))?;
    }
    let center = c.clone();
    let center_z = z.clone();
    let mut minimum = f64::INFINITY;
    for q in 1..p {
        if p % q == 0 {
            minimum = minimum.min(cycle(&C::new(0., 0.), &center, q)?.z.norm());
        }
    }
    let critical_closure = cycle(&C::new(0., 0.), &center, p)?.z.norm();
    if !critical_closure.is_finite()
        || critical_closure > 1e-24
        || (p > 1 && (!minimum.is_finite() || minimum < 1e-16))
    {
        return None;
    }
    println!("[HP_COMPONENT_CENTER145] name={name} coordinate={:?} critical_closure={} minimum_proper_divisor_closure={minimum}",center.f64(),cycle(&C::new(0.,0.),&center,p)?.z.norm());
    for radius in [0.9, 0.99, 0.999, 0.9999, 0.999999, 1.] {
        z = center_z.clone();
        c = center.clone();
        let start_angle = std::f64::consts::PI / 512.;
        for k in 1..=128 {
            let r = radius * k as f64 / 128.;
            (z, c) = solve(
                z,
                c,
                p,
                &C::new(r * start_angle.cos(), r * start_angle.sin()),
            )?;
        }
        let (start_z, start_c) = (z.clone(), c.clone());
        let mut best = f64::INFINITY;
        let mut best_c = c.clone();
        let mut best_z = z.clone();
        let mut theta = 0.;
        for k in 0..=512 {
            let angle = std::f64::consts::TAU * (k as f64 + 0.5) / 512.;
            (z, c) = solve(z, c, p, &C::new(radius * angle.cos(), radius * angle.sin()))?;
            let distance = c.sub(&query).norm();
            if distance < best {
                best = distance;
                best_c = c.clone();
                best_z = z.clone();
                theta = angle;
            }
        }
        let closure_c = c.sub(&start_c).norm();
        if !closure_c.is_finite() || closure_c > 1e-16 {
            return None;
        }
        // Refine the nearest sampled angle; this is numerical optimization, not
        // a proof that all other branches or unsampled root arcs are farther away.
        let step = std::f64::consts::TAU / 512.;
        let mut left = theta - step;
        let mut right = theta + step;
        for _ in 0..48 {
            let a = left + (right - left) / 3.;
            let b = right - (right - left) / 3.;
            let (za, ca) = solve(
                best_z.clone(),
                best_c.clone(),
                p,
                &C::new(radius * a.cos(), radius * a.sin()),
            )?;
            let (zb, cb) = solve(
                best_z.clone(),
                best_c.clone(),
                p,
                &C::new(radius * b.cos(), radius * b.sin()),
            )?;
            let da = ca.sub(&query).norm();
            let db = cb.sub(&query).norm();
            if da < db {
                right = b;
                if da < best {
                    best = da;
                    best_c = ca;
                    best_z = za;
                    theta = a;
                }
            } else {
                left = a;
                if db < best {
                    best = db;
                    best_c = cb;
                    best_z = zb;
                    theta = b;
                }
            }
        }
        if radius == 1. {
            check_distance_derivatives(name, &query, &best_z, &best_c, p, theta)?;
        }
        println!("[HP_COMPONENT_CURVE145] name={name} radius={radius} samples=512 nearest_distance={best:.17} nearest={:?} theta={theta} closure_c={} closure_z={}",best_c.f64(),c.sub(&start_c).norm(),z.sub(&start_z).norm());
    }
    Some(())
}
fn main() {
    let selection = std::env::args().nth(1).unwrap_or("native".into());
    if !["cardioid", "native", "python"].contains(&selection.as_str()) {
        std::process::exit(2);
    }
    for (name, c, p) in [
        ("cardioid", C::new(-0.5, 0.1), 1),
        (
            "native",
            C::new(0.32049988639807619, 0.04733531447355364),
            85,
        ),
        (
            "python",
            C::new(0.35659381055433242, 0.06404656757776669),
            154,
        ),
    ] {
        if name != selection {
            continue;
        }
        let start = std::time::Instant::now();
        let result = trace(name, c, p);
        println!(
            "[HP_COMPONENT_STATUS145] name={name} complete={} elapsed_s={}",
            result.is_some(),
            start.elapsed().as_secs_f64()
        );
        if result.is_none() {
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cardioid_curve_derivatives_match_closed_form() {
        let mu = C::new(-0.4, 0.6);
        let z = mu.scale(0.5);
        let c = mu.scale(0.5).sub(&mu.mul(&mu).scale(0.25));
        let (d1, d2) = curve_derivatives(&z, &c, 1, &mu).unwrap();
        let first = mu.sub(&mu.mul(&mu)).scale(0.5);
        let mut minus_im = first.im.clone();
        minus_im.inv_sign();
        let expected1 = C {
            re: minus_im,
            im: first.re,
        };
        let expected2 = mu.mul(&mu).sub(&mu.scale(0.5));
        assert!(d1.sub(&expected1).norm() < 1e-28);
        assert!(d2.sub(&expected2).norm() < 1e-28);
    }
    #[test]
    fn period_two_cut_locus_and_exterior_refuse() {
        let center = C::new(-1., 0.);
        let zero = C::new(0., 0.);
        assert!(projection(&center, &zero, &center, 2, std::f64::consts::PI).is_none());
        assert!(projection(&C::new(-1., 0.3), &zero, &center, 2, std::f64::consts::FRAC_PI_2).is_none());
    }
    #[test]
    fn period_two_distance_jet_matches_circle() {
        let query = C::new(-0.9, 0.05);
        let mut z = C::new(0., 0.);
        for _ in 0..4096 {
            z = z.mul(&z).add(&query);
        }
        let q = query.f64();
        let r = (q[0] + 1.).hypot(q[1]);
        let theta = q[1].atan2(q[0] + 1.);
        let jet = projection(&query, &z, &query, 2, theta).unwrap();
        assert!((jet.d - (0.25 - r)).abs() < 1e-14);
        for i in 0..2 {
            let x = [q[0] + 1., q[1]];
            assert!((jet.g[i] + x[i] / r).abs() < 1e-12);
            for j in 0..2 {
                let expected = x[i] * x[j] / r.powi(3) - if i == j { 1. / r } else { 0. };
                assert!((jet.h[i][j] - expected).abs() < 1e-11);
            }
        }
    }
}
