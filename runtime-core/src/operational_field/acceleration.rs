//! Outward bounds on the nominal continuous phase dynamics of a prepared spline.

use super::{navigation::ruler_upper, PreparedField};
use crate::intrinsic_bounds::{add_up, div_up, mul_up};
use crate::manifold::ManifoldConfig;

impl PreparedField {
    /// Enclose coordinate accelerations throughout a closed nominal phase box.
    ///
    /// Includes geodesic, Shore potential, metric drag and an externally
    /// bounded control covector. Wall forces are supported only when disabled.
    /// The caller must establish that its control and reference trajectory
    /// satisfy the supplied caps and that the box stays in its reference chart.
    pub fn nominal_acceleration_abs_upper(
        &self,
        bounds: [[f64; 2]; 2],
        velocity_abs_upper: [f64; 2],
        control_abs_upper: [f64; 2],
        beta: f64,
        config: &ManifoldConfig,
    ) -> Option<[f64; 2]> {
        let epsilon = config.epsilon;
        if config.mu != 0.0
            || epsilon <= 0.0
            || ![epsilon, beta, config.lambda_sq, config.kappa]
                .into_iter()
                .chain(velocity_abs_upper)
                .chain(control_abs_upper)
                .all(|bound| bound.is_finite() && bound >= 0.0)
        {
            return None;
        }
        let [first, last] = self.navigation_cells(bounds)?;
        let [lo, hi] = bounds;
        let [x, y] = std::array::from_fn(|i| lo[i].abs().max(hi[i].abs()));
        if add_up(mul_up(x, x), mul_up(y, y)) >= 4.0 {
            return None;
        }

        let mut distance = 0.0_f64;
        let mut gradient = [0.0_f64; 2];
        let mut hessian = [0.0_f64; 3];
        for y in first[1]..=last[1] {
            for x in first[0]..=last[0] {
                let ([lo, hi], grad, hess) = self.bernstein_cell_derivatives::<true>([x, y])?;
                distance = distance.max(lo.abs()).max(hi.abs());
                for (bound, cell) in gradient.iter_mut().zip(grad) {
                    *bound = bound.max(cell);
                }
                for (bound, cell) in hessian.iter_mut().zip(hess) {
                    *bound = bound.max(cell);
                }
            }
        }

        // rho=sqrt(D²+epsilon²): |rho_i| <= |D_i| and
        // |rho_ij| <= |D_ij| + |D_i D_j|/epsilon.
        let rho_upper = ruler_upper(distance, epsilon);
        let [gx, gy] = gradient;
        let [hxx, hxy, hyy] = hessian;
        let rho_xx = add_up(hxx, div_up(mul_up(gx, gx), epsilon));
        let rho_xy = add_up(hxy, div_up(mul_up(gx, gy), epsilon));
        let rho_yy = add_up(hyy, div_up(mul_up(gy, gy), epsilon));
        let [vx, vy] = velocity_abs_upper;
        let vx2 = mul_up(vx, vx);
        let vy2 = mul_up(vy, vy);
        let uv = add_up(mul_up(gx, vx), mul_up(gy, vy));
        let vhv = add_up(
            add_up(mul_up(rho_xx, vx2), mul_up(rho_yy, vy2)),
            mul_up(2.0, mul_up(rho_xy, mul_up(vx, vy))),
        );
        let ln2_lower = std::f64::consts::LN_2.next_down();
        let anisotropy = div_up(div_up(config.lambda_sq, ln2_lower), ln2_lower);
        // In the compact connection, W>=1. The bracket multiplying grad(rho)
        // is bounded by a² |vᵀ Hess(rho) v| + (|v|²+a²(grad(rho)·v)²)/epsilon.
        let common_connection = add_up(
            mul_up(anisotropy, vhv),
            div_up(
                add_up(add_up(vx2, vy2), mul_up(anisotropy, mul_up(uv, uv))),
                epsilon,
            ),
        );
        // The inverse metric norm is at most rho² and grad(rho) is its
        // eigenvector. The raised drag is exactly -beta*v.
        let control = mul_up(
            mul_up(rho_upper, rho_upper),
            add_up(control_abs_upper[0], control_abs_upper[1]),
        );
        let potential_factor = div_up(mul_up(config.kappa, rho_upper), ln2_lower);
        let acceleration = std::array::from_fn(|i| {
            let velocity = velocity_abs_upper[i];
            let connection = add_up(
                mul_up(gradient[i], common_connection),
                div_up(mul_up(2.0, mul_up(uv, velocity)), epsilon),
            );
            add_up(
                add_up(connection, control),
                add_up(
                    mul_up(potential_factor, gradient[i]),
                    mul_up(beta, velocity),
                ),
            )
        });
        acceleration
            .into_iter()
            .all(f64::is_finite)
            .then_some(acceleration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encloses_acceleration_on_a_varying_conformal_field() {
        let coefficients = (0..33 * 33).map(|i| (i % 33) as f64 / 16.0 - 1.0).collect();
        let field = PreparedField::new([-1.0; 2], 1.0 / 16.0, [33; 2], coefficients).unwrap();
        let config = ManifoldConfig {
            epsilon: 0.5,
            lambda_sq: 0.0,
            kappa: 0.0,
            mu: 0.0,
            ..ManifoldConfig::default()
        };
        let bounds = [[0.375, 0.125], [0.625, 0.25]];
        let [x, y] = field
            .nominal_acceleration_abs_upper(bounds, [0.25, 0.0], [0.125, 0.0], 0.5, &config)
            .unwrap();
        // D=x and rho²=x²+1/4. At x=1/2 and vx=1/4, the
        // geodesic, raised control, and drag terms are 1/16, 1/16 and -1/8.
        assert!(x >= 0.125 && x < 2.0);
        assert!(y >= 0.0 && y < 0.125);
        let wall = ManifoldConfig { mu: 1.0, ..config };
        assert!(field
            .nominal_acceleration_abs_upper(bounds, [0.25, 0.0], [0.0; 2], 0.0, &wall)
            .is_none());
        assert!(field
            .nominal_acceleration_abs_upper(bounds, [-0.25, 0.0], [0.0; 2], 0.0, &config)
            .is_none());
    }
}
