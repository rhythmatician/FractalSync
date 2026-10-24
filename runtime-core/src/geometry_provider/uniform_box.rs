//! Conservative empty-contour shortcut. An uncertain box uses ordinary tiles.

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
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
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            hi: products
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }

    fn square(self) -> Self {
        let lo = if self.lo <= 0.0 && self.hi >= 0.0 {
            0.0
        } else {
            {
                let magnitude = self.lo.abs().min(self.hi.abs());
                (magnitude * magnitude).next_down()
            }
        };
        Self {
            lo,
            hi: {
                let magnitude = self.lo.abs().max(self.hi.abs());
                (magnitude * magnitude).next_up()
            },
        }
    }

    fn finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }
}

/// `Some` proves that every point has the same membership. `None` is only a
/// failed shortcut, never a membership refusal. Outward-rounded intervals may
/// overestimate the orbit enclosure, which costs work but cannot hide a Shore.
pub(super) fn membership(re0: f64, im0: f64, width: f64) -> Option<bool> {
    if !re0.is_finite() || !im0.is_finite() || !width.is_finite() || width < 0.0 {
        return None;
    }
    let re = Interval {
        lo: re0.next_down(),
        hi: (re0 + width).next_up(),
    };
    let im = Interval {
        lo: im0.next_down(),
        hi: (im0 + width).next_up(),
    };
    if !re.finite() || !im.finite() {
        return None;
    }
    let y2 = im.square();
    let x = re.sub(Interval::point(0.25));
    let q = x.square().add(y2);
    let cardioid = q.mul(q.add(x)).sub(y2.mul(Interval::point(0.25)));
    let bulb = re.add(Interval::point(1.0)).square().add(y2);
    if (cardioid.finite() && cardioid.hi < 0.0) || (bulb.finite() && bulb.hi < 0.0625) {
        return Some(true);
    }
    let mut zr = Interval::point(0.0);
    let mut zi = Interval::point(0.0);
    // A shortcut only: ordinary tile construction retains its full budgets.
    for _ in 0..64 {
        let next_re = zr.square().sub(zi.square()).add(re);
        let next_im = zr.mul(zi).mul(Interval::point(2.0)).add(im);
        zr = next_re;
        zi = next_im;
        if !zr.finite() || !zi.finite() {
            return disk_membership(re, im).or_else(|| affine_membership(re, im));
        }
        if zr.square().add(zi.square()).lo > 4.0 {
            return Some(false);
        }
    }
    disk_membership(re, im).or_else(|| affine_membership(re, im))
}

fn upper_add(a: f64, b: f64) -> f64 {
    (a + b).next_up()
}
fn upper_mul(a: f64, b: f64) -> f64 {
    (a * b).next_up()
}
fn norm_upper(z: [f64; 2]) -> f64 {
    Interval::point(z[0])
        .square()
        .add(Interval::point(z[1]).square())
        .hi
        .sqrt()
        .next_up()
}
fn distance_upper(a: [f64; 2], b: [f64; 2]) -> f64 {
    let re = Interval::point(a[0]).sub(Interval::point(b[0]));
    let im = Interval::point(a[1]).sub(Interval::point(b[1]));
    re.square().add(im.square()).hi.sqrt().next_up()
}
#[derive(Clone, Copy)]
struct Ball {
    z: [f64; 2],
    radius: f64,
}
impl Ball {
    fn bounded(self) -> bool {
        upper_add(norm_upper(self.z), self.radius) <= 2.
    }
    fn step(self, c: [f64; 2], cr: f64) -> Option<Self> {
        let z = [
            self.z[0] * self.z[0] - self.z[1] * self.z[1] + c[0],
            2. * self.z[0] * self.z[1] + c[1],
        ];
        let a = upper_add(norm_upper(self.z), self.radius);
        let c_bound = upper_add(norm_upper(c), cr);
        // Let A bound |z| and C bound |c|. Each real/imaginary component
        // uses at most four rounded operations, so gamma_4 < 4*EPSILON bounds
        // its error by 4*EPSILON*(A^2+C) plus four MIN_POSITIVE terms for
        // underflow. Summing component magnitudes bounds the complex error
        // by 8*EPSILON*(A^2+C) + 8*MIN_POSITIVE. Account for both the actual
        // machine orbit and this machine center, then double that bound.
        // Accepted proofs keep every disk within |z| <= 2; nonfinite results
        // cannot certify. Radius arithmetic below is rounded upward too.
        let round = machine_round_error(a, c_bound);
        let radius = upper_add(
            upper_add(
                upper_mul(upper_mul(2., norm_upper(self.z)), self.radius),
                upper_mul(self.radius, self.radius),
            ),
            upper_add(cr, round),
        );
        (radius.is_finite() && radius <= 4. && z.iter().all(|x| x.is_finite()))
            .then_some(Self { z, radius })
    }
}
fn disk_membership(re: Interval, im: Interval) -> Option<bool> {
    let c = [re.lo + (re.hi - re.lo) * 0.5, im.lo + (im.hi - im.lo) * 0.5];
    if !c.iter().all(|x| x.is_finite()) {
        return None;
    }
    let cr = norm_upper([
        (re.lo - c[0]).abs().max((re.hi - c[0]).abs()).next_up(),
        (im.lo - c[1]).abs().max((im.hi - c[1]).abs()).next_up(),
    ]);
    let mut orbit = Ball {
        z: [0.; 2],
        radius: 0.,
    };
    let mut centers = [[0.; 2]; 65];
    let mut bounded = true;
    for slot in centers.iter_mut().skip(1) {
        orbit = orbit.step(c, cr)?;
        *slot = orbit.z;
        bounded &= orbit.bounded();
    }
    if !bounded {
        return None;
    }
    // A returned disk must contain the entire current critical-orbit disk.
    // Strict inclusion after p steps makes it forward invariant under f_c^p.
    // Every intervening disk must remain below the same escape radius, so
    // this also proves the ordinary finite-budget machine orbit stays inside.
    periodic_trap(orbit.z, orbit.radius, &centers, c, cr)
}

#[cfg(test)]
mod tests {
    use super::membership;

    #[test]
    fn certifies_uniform_boxes_and_keeps_boundary_boxes_uncertain() {
        assert_eq!(membership(-0.1, -0.1, 0.2), Some(true));
        assert_eq!(membership(-1.05, -0.05, 0.1), Some(true));
        assert_eq!(membership(0.8, 0.5, 0.1), Some(false));
        assert_eq!(membership(0.24, -0.01, 0.02), None);
        assert_eq!(membership(-0.76, -0.01, 0.02), None);
    }

    #[test]
    fn box_certificates_agree_with_all_ordinary_tile_nodes() {
        for (re0, im0, width) in [
            (-0.1, -0.1, 0.2),
            (-1.05, -0.05, 0.1),
            (0.8, 0.5, 0.1),
            (0.4, 0.0, 0.02),
            (-1.8, 0.8, 0.1),
        ] {
            let expected = membership(re0, im0, width).expect("uniform fixture");
            for y in 0..=56 {
                for x in 0..=56 {
                    let c = num_complex::Complex64::new(
                        re0 + x as f64 * width / 56.0,
                        im0 + y as f64 * width / 56.0,
                    );
                    for budget in [512, 32768] {
                        assert_eq!(
                            crate::visual_metrics::mandelbrot_membership(c, budget),
                            expected,
                            "{c:?}, budget={budget}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn certificates_match_membership_across_the_map() {
        let mut certified = 0;
        let mut uncertain = 0;
        for y in -10..=10 {
            for x in -15..=5 {
                let re0 = x as f64 / 8.0;
                let im0 = y as f64 / 8.0;
                let width = 1.0 / 64.0;
                let Some(expected) = membership(re0, im0, width) else {
                    uncertain += 1;
                    continue;
                };
                certified += 1;
                for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)] {
                    let c = num_complex::Complex64::new(re0 + dx * width, im0 + dy * width);
                    for budget in [512, 32768] {
                        assert_eq!(
                            crate::visual_metrics::mandelbrot_membership(c, budget),
                            expected,
                            "{c:?}, budget={budget}"
                        );
                    }
                }
            }
        }
        assert!(certified > 100);
        assert!(uncertain > 0);
    }

    #[test]
    fn periodic_disk_proof_covers_conjugate_period_three_tiles() {
        for im in [-0.7448617666197442, 0.7448617666197442] {
            let re0 = -0.1225611668766536 - 0.0005;
            let im0 = im - 0.0005;
            let width = 0.001;
            assert_eq!(membership(re0, im0, width), Some(true));
            for y in 0..=56 {
                for x in 0..=56 {
                    let c = num_complex::Complex64::new(
                        re0 + x as f64 * width / 56.0,
                        im0 + y as f64 * width / 56.0,
                    );
                    for budget in [512, 32768] {
                        assert!(crate::visual_metrics::mandelbrot_membership(c, budget));
                    }
                }
            }
        }
    }

    #[test]
    fn neutral_cycles_do_not_supply_an_interior_disk_certificate() {
        for re in [0.25, 0.2500001, -0.75] {
            assert_eq!(membership(re, 0.0, 0.0), None);
        }
    }

    #[test]
    fn invalid_boxes_do_not_supply_certificates() {
        for (re0, im0, width) in [
            (f64::NAN, 0.0, 1.0),
            (0.0, f64::INFINITY, 1.0),
            (0.0, 0.0, -1.0),
            (f64::MAX, 0.0, f64::MAX),
        ] {
            assert_eq!(membership(re0, im0, width), None);
        }
    }
}

// Retain the linear dependence on delta_c instead of adding the full
// parameter radius independently at every iteration:
// z(c) = z_center + derivative * delta_c + remainder_disk.
// A result remains a whole-box certificate for the ordinary machine orbit.
fn affine_membership(re: Interval, im: Interval) -> Option<bool> {
    let c = [re.lo + (re.hi - re.lo) * 0.5, im.lo + (im.hi - im.lo) * 0.5];
    if !c.iter().all(|x| x.is_finite()) {
        return None;
    }
    let cr = norm_upper([
        (re.lo - c[0]).abs().max((re.hi - c[0]).abs()).next_up(),
        (im.lo - c[1]).abs().max((im.hi - c[1]).abs()).next_up(),
    ]);
    let mut z = [0.; 2];
    let mut d = [0.; 2];
    let mut remainder = 0.;
    let mut centers = vec![z];
    let mut all_bounded = true;
    let mut radius = 0.;
    for n in 1..=512 {
        let magnitude = norm_upper(z);
        let sensitivity = norm_upper(d);
        let total = upper_add(upper_mul(sensitivity, cr), remainder);
        let a = upper_add(magnitude, total);
        let cb = upper_add(norm_upper(c), cr);
        // Bounds both the machine critical-orbit update and center update.
        let round = machine_round_error(a, cb);
        let next = [z[0] * z[0] - z[1] * z[1] + c[0], 2. * z[0] * z[1] + c[1]];
        let next_d = [
            2. * (z[0] * d[0] - z[1] * d[1]) + 1.,
            2. * (z[0] * d[1] + z[1] * d[0]),
        ];
        let p = Interval::point;
        let dr = p(2.)
            .mul(p(z[0]).mul(p(d[0])).sub(p(z[1]).mul(p(d[1]))))
            .add(p(1.));
        let di = p(2.).mul(p(z[0]).mul(p(d[1])).add(p(z[1]).mul(p(d[0]))));
        let de = dr.sub(p(next_d[0]));
        let ie = di.sub(p(next_d[1]));
        let coefficient_error = de.square().add(ie.square()).hi.sqrt().next_up();
        remainder = upper_add(
            upper_add(
                upper_mul(upper_mul(2., magnitude), remainder),
                upper_mul(total, total),
            ),
            upper_add(round, upper_mul(coefficient_error, cr)),
        );
        z = next;
        d = next_d;
        radius = upper_add(upper_mul(norm_upper(d), cr), remainder);
        if !radius.is_finite() || radius > 4. || !z.iter().all(|x| x.is_finite()) {
            return None;
        }
        let low = p(z[0])
            .square()
            .add(p(z[1]).square())
            .lo
            .max(0.)
            .sqrt()
            .next_down();
        // The ordinary 512-update query checks escape before each update.
        if n < 512 && (low - radius).next_down() > 2. {
            return Some(false);
        }
        all_bounded &= Ball { z, radius }.bounded();
        centers.push(z);
    }
    if !all_bounded {
        return None;
    }
    periodic_trap(z, radius, &centers, c, cr)
}

#[cfg(test)]
fn affine_box(re0: f64, im0: f64, width: f64) -> Option<bool> {
    if !re0.is_finite() || !im0.is_finite() || !width.is_finite() || width < 0. {
        return None;
    }
    let re = Interval {
        lo: re0.next_down(),
        hi: (re0 + width).next_up(),
    };
    let im = Interval {
        lo: im0.next_down(),
        hi: (im0 + width).next_up(),
    };
    if !re.finite() || !im.finite() {
        return None;
    }
    affine_membership(re, im)
}

fn periodic_trap(
    z: [f64; 2],
    radius: f64,
    centers: &[[f64; 2]],
    c: [f64; 2],
    cr: f64,
) -> Option<bool> {
    let last = centers.len().checked_sub(1)?;
    let mut period = 1;
    let mut separation = f64::INFINITY;
    for p in 1..=last {
        let distance = distance_upper(z, centers[last - p]);
        if distance < separation {
            period = p;
            separation = distance;
        }
    }
    for factor in [2., 8., 32., 128.] {
        let r = upper_mul(factor, upper_add(radius, separation));
        let start = Ball { z, radius: r };
        if !start.bounded() {
            continue;
        }
        let mut end = start;
        let mut bounded = true;
        // The parameter box is fixed throughout this return; intermediate disks
        // must remain bounded as well as the final disk lying strictly inside.
        for _ in 0..period {
            let Some(next) = end.step(c, cr) else {
                bounded = false;
                break;
            };
            end = next;
            bounded &= end.bounded();
        }
        if bounded && upper_add(distance_upper(end.z, start.z), end.radius) < r {
            return Some(true);
        }
    }
    None
}
fn machine_round_error(a: f64, c_bound: f64) -> f64 {
    // Encloses componentwise complex square/add rounding for both the actual
    // machine update and the rounded center, including underflow allowances.
    upper_add(
        upper_mul(32. * f64::EPSILON, upper_add(upper_mul(a, a), c_bound)),
        32. * f64::MIN_POSITIVE,
    )
}

#[cfg(test)]
mod affine_tests {
    use super::*;
    #[test]
    fn recorded_training_box_is_provably_outside() {
        let c = [0.3075204398270623, 0.02876420131721053];
        let width = 56. * 2f64.powi(-22);
        let re = c[0] - width / 2.;
        let im = c[1] - width / 2.;
        assert_eq!(affine_box(re, im, width), Some(false));
        assert_eq!(membership(re, im, width), Some(false));
        for y in 0..=16 {
            for x in 0..=16 {
                let point = num_complex::Complex64::new(
                    re + width * x as f64 / 16.,
                    im + width * y as f64 / 16.,
                );
                for budget in [512, 32768] {
                    assert!(!crate::visual_metrics::mandelbrot_membership(point, budget));
                }
            }
        }
    }
    #[test]
    fn affine_certificates_match_independent_ordinary_nodes() {
        let mut accepted = 0;
        for y in -10..=10 {
            for x in -15..=5 {
                let re = x as f64 / 8.;
                let im = y as f64 / 8.;
                let width = 1. / 64.;
                let Some(expected) = affine_box(re, im, width) else {
                    continue;
                };
                accepted += 1;
                for (dx, dy) in [(0., 0.), (1., 0.), (0., 1.), (1., 1.), (0.5, 0.5)] {
                    let c = num_complex::Complex64::new(re + dx * width, im + dy * width);
                    for budget in [512, 32768] {
                        assert_eq!(
                            crate::visual_metrics::mandelbrot_membership(c, budget),
                            expected,
                            "{c:?}"
                        );
                    }
                }
            }
        }
        assert!(accepted > 100);
    }
    #[test]
    fn neutral_and_invalid_boxes_cannot_supply_affine_certificates() {
        for x in [0.25, 0.2500001, -0.75] {
            assert_eq!(affine_box(x, 0., 0.), None);
        }
        for (x, y, w) in [
            (f64::NAN, 0., 1.),
            (0., f64::INFINITY, 1.),
            (0., 0., -1.),
            (f64::MAX, 0., f64::MAX),
        ] {
            assert_eq!(affine_box(x, y, w), None);
        }
    }
    #[test]
    fn affine_exterior_obeys_the_preupdate_escape_budget() {
        let fixtures = [
            (511, 0.2500375092511157),
            (512, 0.25003736342837773),
            (513, 0.2500372184543587),
        ];
        for (first_escape, c) in fixtures {
            let mut z = 0.;
            let mut detected = None;
            for n in 0..=513 {
                if z * z > 4. {
                    detected = Some(n);
                    break;
                }
                z = z * z + c;
            }
            assert_eq!(detected, Some(first_escape));
            let point = num_complex::Complex64::new(c, 0.);
            assert_eq!(
                crate::visual_metrics::mandelbrot_membership(point, 512),
                first_escape >= 512
            );
            assert!(!crate::visual_metrics::mandelbrot_membership(point, 32768));
            assert_eq!(
                affine_box(c, 0., 0.),
                if first_escape < 512 {
                    Some(false)
                } else {
                    None
                }
            );
        }
    }
    #[test]
    fn affine_interior_proves_a_whole_period_three_box() {
        let center = [-0.1225611668766536, 0.7448617666197442];
        let width = 1e-7;
        let re = center[0] - width / 2.;
        let im = center[1] - width / 2.;
        assert_eq!(affine_box(re, im, width), Some(true));
        for y in 0..=8 {
            for x in 0..=8 {
                let point = num_complex::Complex64::new(
                    re + width * x as f64 / 8.,
                    im + width * y as f64 / 8.,
                );
                for budget in [512, 32768] {
                    assert!(crate::visual_metrics::mandelbrot_membership(point, budget));
                }
            }
        }
    }
}
