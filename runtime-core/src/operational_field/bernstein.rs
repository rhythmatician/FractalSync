//! Whole-cell bounds for the exact real spline represented by a prepared asset.

use super::PreparedField;

// Rows are the six Bernstein control positions, columns are source lattice
// coefficients. For a basis polynomial with power coefficients p_j / 120,
// control k is sum_{j<=k} p_j * binom(k,j) / binom(5,j) / 120.
// Every entry below is an integer numerator, and every row sums to 120.
const WEIGHTS: [[u8; 6]; 6] = [
    [1, 26, 66, 26, 1, 0],
    [0, 16, 66, 36, 2, 0],
    [0, 8, 60, 48, 4, 0],
    [0, 4, 48, 60, 8, 0],
    [0, 2, 36, 66, 16, 0],
    [0, 1, 26, 66, 26, 1],
];

#[derive(Clone, Copy, Default)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn centered(value: f64, center: f64) -> Self {
        if value == center {
            return Self::default();
        }
        let shift = value - center;
        Self {
            lo: shift.next_down(),
            hi: shift.next_up(),
        }
    }

    fn weighted_sum(values: [Self; 6], weights: [u8; 6]) -> Self {
        let mut total = Self::default();
        for (value, weight) in values.into_iter().zip(weights) {
            if weight == 0 || (value.lo == 0.0 && value.hi == 0.0) {
                continue;
            }
            let weight = f64::from(weight);
            total.lo = (total.lo + (value.lo * weight).next_down()).next_down();
            total.hi = (total.hi + (value.hi * weight).next_up()).next_up();
        }
        Self {
            lo: (total.lo / 120.0).next_down(),
            hi: (total.hi / 120.0).next_up(),
        }
    }

    fn difference(self, other: Self) -> Self {
        Self {
            lo: (self.lo - other.hi).next_down(),
            hi: (self.hi - other.lo).next_up(),
        }
    }

    fn difference_abs_upper(self, other: Self) -> f64 {
        (self.hi - other.lo).max(other.hi - self.lo).next_up()
    }
}

impl PreparedField {
    /// Bound signed value and the absolute physical derivatives in x and y
    /// on the closed polynomial cell. Missing support or nonfinite bounds
    /// yield no certificate. Assumes round-to-nearest binary64 operations.
    pub(super) fn bernstein_cell_bounds(
        &self,
        [cell_x, cell_y]: [usize; 2],
    ) -> Option<([f64; 2], [f64; 2])> {
        self.bernstein_cell_derivatives::<false>([cell_x, cell_y])
            .map(|(value, gradient, _)| (value, gradient))
    }

    pub(super) fn bernstein_cell_derivatives<const HESSIAN: bool>(
        &self,
        [cell_x, cell_y]: [usize; 2],
    ) -> Option<([f64; 2], [f64; 2], [f64; 3])> {
        let [width, height] = self.shape;
        if cell_x < 2 || cell_x >= width - 3 || cell_y < 2 || cell_y >= height - 3 {
            return None;
        }

        let mut stencil = [[0.0; 6]; 6];
        for (y, row) in stencil.iter_mut().enumerate() {
            for (x, value) in row.iter_mut().enumerate() {
                let index = (cell_y - 2 + y) * width + cell_x - 2 + x;
                if self.available.as_ref().is_some_and(|mask| !mask[index]) {
                    return None;
                }
                *value = self.coefficients[index];
            }
        }
        let center = stencil[2][2];
        let coefficient_min = stencil.iter().flatten().copied().fold(center, f64::min);
        let coefficient_max = stencil.iter().flatten().copied().fold(center, f64::max);
        if coefficient_min == coefficient_max {
            return Some(([center, center], [0.0; 2], [0.0; 3]));
        }

        // Centering commutes with the transform by partition of unity. It
        // removes the shared offset before taking control differences.
        let rows = stencil.map(|row| {
            let centered = row.map(|value| Interval::centered(value, center));
            WEIGHTS.map(|weights| Interval::weighted_sum(centered, weights))
        });
        let mut controls = [[Interval::default(); 6]; 6];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (y, row) in controls.iter_mut().enumerate() {
            for (x, control) in row.iter_mut().enumerate() {
                *control = Interval::weighted_sum(rows.map(|row| row[x]), WEIGHTS[y]);
                if !control.lo.is_finite() || !control.hi.is_finite() {
                    return None;
                }
                lo = lo.min(control.lo);
                hi = hi.max(control.hi);
            }
        }
        let values = [
            (lo + center).next_down().max(coefficient_min),
            (hi + center).next_up().min(coefficient_max),
        ];

        // The derivative in normalized cell coordinates has Bernstein
        // degree four along its axis and controls 5 * (next - current).
        // The nonnegative tensor basis is a partition of unity, so the
        // maximum absolute control bounds that derivative everywhere.
        let mut differences = [0.0_f64; 2];
        for (y, row) in controls.iter().enumerate() {
            for (x, &control) in row.iter().enumerate() {
                if x > 0 {
                    differences[0] = differences[0].max(control.difference_abs_upper(row[x - 1]));
                }
                if y > 0 {
                    differences[1] =
                        differences[1].max(control.difference_abs_upper(controls[y - 1][x]));
                }
            }
        }
        let gradient = differences.map(|delta| ((5.0 * delta).next_up() / self.spacing).next_up());
        let mut hessian = [0.0_f64; 3];
        if HESSIAN {
            // Differentiating the Bernstein control net again gives factors
            // 5*4 on pure differences and 5*5 on mixed differences.
            for (y, row) in controls.iter().enumerate() {
                for (x, &control) in row.iter().enumerate() {
                    if x >= 2 {
                        let first = control.difference(row[x - 1]);
                        let prior = row[x - 1].difference(row[x - 2]);
                        hessian[0] = hessian[0].max(first.difference_abs_upper(prior));
                    }
                    if x >= 1 && y >= 1 {
                        let first = control.difference(row[x - 1]);
                        let prior = controls[y - 1][x].difference(controls[y - 1][x - 1]);
                        hessian[1] = hessian[1].max(first.difference_abs_upper(prior));
                    }
                    if y >= 2 {
                        let first = control.difference(controls[y - 1][x]);
                        let prior = controls[y - 1][x].difference(controls[y - 2][x]);
                        hessian[2] = hessian[2].max(first.difference_abs_upper(prior));
                    }
                }
            }
            for (bound, weight) in hessian.iter_mut().zip([20.0, 25.0, 20.0]) {
                *bound = (((weight * *bound).next_up() / self.spacing).next_up() / self.spacing)
                    .next_up();
            }
        }
        gradient
            .into_iter()
            .chain(hessian)
            .all(f64::is_finite)
            .then_some((values, gradient, hessian))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_field::basis;

    #[test]
    fn bernstein_controls_match_the_literal_evaluator_basis() {
        const BINOMIAL: [u32; 6] = [1, 5, 10, 10, 5, 1];
        // Before division by 120 both calculations are exact dyadic values.
        // More than six points identify each degree-five polynomial.
        for i in 0_u32..=8 {
            let weights = basis(f64::from(i) / 8.0)[0];
            for (j, actual) in weights.into_iter().enumerate() {
                let expected: u32 = WEIGHTS
                    .iter()
                    .enumerate()
                    .map(|(k, row)| {
                        let k = k as u32;
                        u32::from(row[j]) * BINOMIAL[k as usize] * i.pow(k) * (8 - i).pow(5 - k)
                    })
                    .sum();
                assert_eq!(actual, f64::from(expected) / (120.0 * 8.0_f64.powi(5)));
            }
        }
    }

    #[test]
    fn bernstein_cell_bounds_preserve_affine_jets_and_remove_coefficient_halo_range() {
        let origin = [-1.5, 0.25];
        let h = 0.25;
        let coefficients = (0..110)
            .map(|i| {
                7.0 + 2.0 * (origin[0] + f64::from(i % 10) * h)
                    - 4.0 * (origin[1] + f64::from(i / 10) * h)
            })
            .collect();
        let field = PreparedField::new(origin, h, [10, 11], coefficients).unwrap();
        let (value, gradient) = field.bernstein_cell_bounds([3, 4]).unwrap();
        assert!(value[0] <= -0.5 && value[0] > -0.50000000001, "{value:?}");
        assert!(value[1] >= 1.0 && value[1] < 1.00000000001, "{value:?}");
        assert!(
            gradient[0] >= 2.0 && gradient[0] < 2.00000000001,
            "{gradient:?}"
        );
        assert!(
            gradient[1] >= 4.0 && gradient[1] < 4.00000000001,
            "{gradient:?}"
        );
    }

    #[test]
    fn bernstein_cell_bounds_cover_asymmetric_polynomial_samples() {
        let coefficients = (0..72)
            .map(|i| f64::from((i * i + 3 * i) % 23 - 11))
            .collect();
        let field = PreparedField::new([0.0; 2], 0.125, [8, 9], coefficients).unwrap();
        let (value, gradient) = field.bernstein_cell_bounds([4, 3]).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let point = [
                    (4.0 + f64::from(x) / 16.0) / 8.0,
                    (3.0 + f64::from(y) / 16.0) / 8.0,
                ];
                let jet = field.query(point).unwrap();
                assert!(value[0] - 1e-10 <= jet.value && jet.value <= value[1] + 1e-10);
                for (derivative, upper) in jet.gradient.into_iter().zip(gradient) {
                    assert!(derivative.abs() <= upper + 1e-10);
                }
            }
        }
        assert!(value[0] > -10.0 && value[1] < 10.0);
    }

    #[test]
    fn bernstein_derivatives_bound_all_hessian_components_of_a_quadratic() {
        let spacing = 0.25;
        let coefficients = (0..110)
            .map(|i| {
                let x = f64::from(i % 10) * spacing;
                let y = f64::from(i / 10) * spacing;
                x * x + 3.0 * x * y + 4.0 * y * y
            })
            .collect();
        let field = PreparedField::new([0.0; 2], spacing, [10, 11], coefficients).unwrap();
        let (_, _, hessian) = field.bernstein_cell_derivatives::<true>([4, 3]).unwrap();
        for (bound, expected) in hessian.into_iter().zip([2.0, 3.0, 8.0]) {
            assert!(expected <= bound && bound < expected + 1e-10, "{hessian:?}");
        }
    }

    #[test]
    fn bernstein_cell_bounds_require_complete_support() {
        let mut coefficients = vec![Some(3.0); 56];
        coefficients[6 * 8 + 7] = None;
        let field = PreparedField::new_sparse([0.0; 2], 1.0, [8, 7], coefficients).unwrap();
        assert_eq!(
            field.bernstein_cell_bounds([3, 3]),
            Some(([3.0; 2], [0.0; 2]))
        );
        assert_eq!(field.bernstein_cell_bounds([4, 3]), None);
        assert_eq!(field.bernstein_cell_bounds([1, 3]), None);
        assert_eq!(field.bernstein_cell_bounds([usize::MAX, 3]), None);
    }
}
