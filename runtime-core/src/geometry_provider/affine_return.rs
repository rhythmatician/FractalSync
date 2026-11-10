//! Correlated periodic return of an already enclosed critical-orbit prefix.
use super::{
    distance_upper, machine_round_error, norm_upper, upper_add, upper_mul, Ball, Interval,
};

#[derive(Clone, Copy)]
pub(super) struct AffineBall {
    pub(super) z: [f64; 2],
    pub(super) d: [f64; 2],
    pub(super) remainder: f64,
}

impl AffineBall {
    pub(super) fn radius(self, cr: f64) -> f64 {
        upper_add(upper_mul(norm_upper(self.d), cr), self.remainder)
    }

    pub(super) fn bounded(self, cr: f64) -> bool {
        Ball {
            z: self.z,
            radius: self.radius(cr),
        }
        .bounded()
    }

    pub(super) fn step(self, c: [f64; 2], cr: f64) -> Option<Self> {
        let magnitude = norm_upper(self.z);
        let total = self.radius(cr);
        let a = upper_add(magnitude, total);
        let cb = upper_add(norm_upper(c), cr);
        let round = machine_round_error(a, cb);
        let z = [
            self.z[0] * self.z[0] - self.z[1] * self.z[1] + c[0],
            2. * self.z[0] * self.z[1] + c[1],
        ];
        let d = [
            2. * (self.z[0] * self.d[0] - self.z[1] * self.d[1]) + 1.,
            2. * (self.z[0] * self.d[1] + self.z[1] * self.d[0]),
        ];
        let p = Interval::point;
        let dr = p(2.)
            .mul(
                p(self.z[0])
                    .mul(p(self.d[0]))
                    .sub(p(self.z[1]).mul(p(self.d[1]))),
            )
            .add(p(1.));
        let di = p(2.).mul(
            p(self.z[0])
                .mul(p(self.d[1]))
                .add(p(self.z[1]).mul(p(self.d[0]))),
        );
        let de = dr.sub(p(d[0]));
        let ie = di.sub(p(d[1]));
        let coefficient_error = de.square().add(ie.square()).hi.sqrt().next_up();
        let remainder = upper_add(
            upper_add(
                upper_mul(upper_mul(2., magnitude), self.remainder),
                upper_mul(total, total),
            ),
            upper_add(round, upper_mul(coefficient_error, cr)),
        );
        let next = Self { z, d, remainder };
        let radius = next.radius(cr);
        (radius.is_finite() && radius <= 4. && z.iter().all(|x| x.is_finite())).then_some(next)
    }

    /// For every fixed parameter delta in its disk, the return maps the disk
    /// z + d*delta + |error|<=radius strictly into itself. It contains the
    /// enclosed prefix and all intermediate machine-orbit disks stay bounded.
    pub(super) fn traps(self, c: [f64; 2], cr: f64, period: usize, radius: f64) -> bool {
        if period == 0
            || !cr.is_finite()
            || cr < 0.
            || !radius.is_finite()
            || !self.remainder.is_finite()
            || self.remainder < 0.
            || radius < self.remainder
        {
            return false;
        }
        let mut end = Self {
            remainder: radius,
            ..self
        };
        if !end.bounded(cr) {
            return false;
        }
        for _ in 0..period {
            let Some(next) = end.step(c, cr) else {
                return false;
            };
            if !next.bounded(cr) {
                return false;
            }
            end = next;
        }
        let coefficient_gap = distance_upper(end.d, self.d);
        let returned = upper_add(
            distance_upper(end.z, self.z),
            upper_add(upper_mul(coefficient_gap, cr), end.remainder),
        );
        returned.is_finite() && returned < radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn correlated_return_covers_a_parameter_disk() {
        let prefix = AffineBall {
            z: [0.; 2],
            d: [1., 0.],
            remainder: 1e-4,
        };
        assert!(prefix.traps([0.; 2], 0.01, 1, 0.001));
        assert!(!prefix.traps([0.; 2], 0.01, 1, 1e-5));
    }
    #[test]
    fn return_must_account_for_the_changed_parameter_coefficient() {
        let prefix = AffineBall {
            z: [0.1, 0.],
            d: [0.; 2],
            remainder: 1e-5,
        };
        assert!(!prefix.traps([0.09, 0.], 0.001, 1, 1e-5));
    }
    #[test]
    fn neutral_and_invalid_returns_refuse() {
        let prefix = AffineBall {
            z: [0.5, 0.],
            d: [0.; 2],
            remainder: 1e-6,
        };
        assert!(!prefix.traps([0.25, 0.], 0., 1, 1e-5));
        for cr in [-1., f64::NAN, f64::INFINITY] {
            assert!(!prefix.traps([0.25, 0.], cr, 1, 1e-5));
        }
        assert!(!prefix.traps([0.25, 0.], 0., 0, 1e-5));
    }
}
