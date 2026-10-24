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
            self.lo.abs().min(self.hi.abs()).powi(2).next_down()
        };
        Self {
            lo,
            hi: self.lo.abs().max(self.hi.abs()).powi(2).next_up(),
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
            return None;
        }
        if zr.square().add(zi.square()).lo > 4.0 {
            return Some(false);
        }
    }
    None
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
