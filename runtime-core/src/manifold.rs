//! Mandelbrot-native manifold Physics (issue #106)
//!
//! The Player moves on a 2D configuration manifold embedded in 3D position-scale
//! space. The independent coordinates are (x,y) in the complex plane; Mandelbrot
//! scale sigma(c) is a derived geometric quantity, not an independent degree of
//! freedom.
//!
//! Core equations:
//!   q(c) = (x, y, sigma(c))           -- embedding
//!   J_q = ∂q/∂(x,y)                   -- Jacobian
//!   G(c) = J_q^T H J_q                -- induced metric
//!   K = 1/2 v^T G v                   -- kinetic energy
//!   U = kappa * sigma(c)              -- native potential
//!   E = K + U                         -- mechanical energy
//!   Gamma^i_jk = connection           -- curvature acceleration
//!   r_ddot + Gamma(r_dot,r_dot) = -G^{-1}∇U + G^{-1}Q  -- equations of motion
//!
//! Physics receives Controls as generalized forces; it does not know about music.

use num_complex::Complex64;

/// Configuration manifold parameters
#[derive(Clone, Debug)]
pub struct ManifoldConfig {
    /// Reference distance for scale definition (typically 0.1)
    pub d_ref: f64,
    /// Resolution floor for smooth regularization (typically 1e-4)
    pub epsilon: f64,
    /// Ambient scale weight lambda^2 in metric H = diag(1,1,lambda^2)
    pub lambda_sq: f64,
    /// Potential scale kappa in U = kappa*sigma
    pub kappa: f64,
}

impl Default for ManifoldConfig {
    fn default() -> Self {
        Self {
            d_ref: 0.1,
            epsilon: 1e-4,
            lambda_sq: 1.0,
            kappa: 1.0,
        }
    }
}

/// Signed distance to the Mandelbrot boundary.
/// D(c) < 0 inside M, D(c) > 0 outside M.
/// 
/// Currently delegates to the unsigned distance field and uses a heuristic
/// sign from escape iteration. Future: use proper signed distance field.
pub fn signed_distance(c: Complex64) -> Result<f64, String> {
    // Sample the unsigned distance field
    let dist_unsigned = crate::distance_field::sample_distance_field(&[c])?
        .into_iter()
        .next()
        .ok_or_else(|| "empty distance sample".to_string())?;
    
    // Heuristic sign from quick escape test (max 256 iterations)
    let mut z = Complex64::new(0.0, 0.0);
    let max_iter = 256;
    let bailout = 4.0;
    let mut escaped = false;
    
    for _ in 0..max_iter {
        z = z * z + c;
        if z.norm_sqr() > bailout {
            escaped = true;
            break;
        }
    }
    
    // Sign convention: positive outside, negative inside
    let sign = if escaped { 1.0 } else { -1.0 };
    Ok(sign * dist_unsigned as f64)
}

/// Smooth finite-resolution distance using regularization.
/// rho(c) = sqrt(D(c)^2 + epsilon^2)
///
/// This gives finite derivatives through Shore crossing and symmetric
/// treatment of inside/outside regions.
pub fn regularized_distance(c: Complex64, epsilon: f64) -> Result<f64, String> {
    let d = signed_distance(c)?;
    Ok((d * d + epsilon * epsilon).sqrt())
}

/// Mandelbrot scale sigma(c) = log2(d_ref / rho(c))
///
/// High scale at the Shore, decreases with distance from boundary.
/// The scale is symmetric across inside/outside due to regularization.
pub fn mandelbrot_scale(c: Complex64, config: &ManifoldConfig) -> Result<f64, String> {
    let rho = regularized_distance(c, config.epsilon)?;
    Ok((config.d_ref / rho).log2())
}

/// Scale gradient ∇sigma(c) computed with central differences.
///
/// Returns (∂sigma/∂x, ∂sigma/∂y) in world coordinates.
///
/// Finite-difference step: the distance field is stored f32, so h must
/// stay well above the f32 noise floor (~1.6e-7 absolute at sigma ~ 1).
/// h = 1e-4 is converged (h = 1e-6 measures quantization noise, not
/// geometry — sigma_xx came out ~40000 instead of ~22).
pub fn scale_gradient(c: Complex64, config: &ManifoldConfig) -> Result<(f64, f64), String> {
    let h = 1e-4; // finite difference step (f32 field noise floor)
    
    let c_px = Complex64::new(c.re + h, c.im);
    let sigma_px = mandelbrot_scale(c_px, config)?;
    
    let c_mx = Complex64::new(c.re - h, c.im);
    let sigma_mx = mandelbrot_scale(c_mx, config)?;
    
    let c_py = Complex64::new(c.re, c.im + h);
    let sigma_py = mandelbrot_scale(c_py, config)?;
    
    let c_my = Complex64::new(c.re, c.im - h);
    let sigma_my = mandelbrot_scale(c_my, config)?;
    
    let grad_x = (sigma_px - sigma_mx) / (2.0 * h);
    let grad_y = (sigma_py - sigma_my) / (2.0 * h);
    
    Ok((grad_x, grad_y))
}

/// Scale Hessian (second derivatives) computed with finite differences.
///
/// Returns [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]]
///
/// Same h constraint as scale_gradient: h must clear the f32 noise
/// floor of the distance field, and the second difference amplifies
/// noise by an extra 1/h.
pub fn scale_hessian(c: Complex64, config: &ManifoldConfig) -> Result<[[f64; 2]; 2], String> {
    let h = 1e-4;
    
    let c_px = Complex64::new(c.re + h, c.im);
    let (gx_px, _) = scale_gradient(c_px, config)?;
    
    let c_mx = Complex64::new(c.re - h, c.im);
    let (gx_mx, _) = scale_gradient(c_mx, config)?;
    
    let c_py = Complex64::new(c.re, c.im + h);
    let (gx_py, gy_py) = scale_gradient(c_py, config)?;
    
    let c_my = Complex64::new(c.re, c.im - h);
    let (gx_my, gy_my) = scale_gradient(c_my, config)?;
    
    let sigma_xx = (gx_px - gx_mx) / (2.0 * h);
    let sigma_yy = (gy_py - gy_my) / (2.0 * h);
    let sigma_xy = (gx_py - gx_my) / (2.0 * h); // or (gy_px - gy_mx)/(2h), should match
    
    Ok([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])
}

/// Induced metric G(c) = I + lambda^2 * grad_sigma * grad_sigma^T
///
/// Returns 2x2 symmetric positive-definite matrix as [[g11, g12], [g12, g22]]
pub fn induced_metric(c: Complex64, config: &ManifoldConfig) -> Result<[[f64; 2]; 2], String> {
    let (gx, gy) = scale_gradient(c, config)?;
    let lsq = config.lambda_sq;
    
    let g11 = 1.0 + lsq * gx * gx;
    let g12 = lsq * gx * gy;
    let g22 = 1.0 + lsq * gy * gy;
    
    Ok([[g11, g12], [g12, g22]])
}

/// Inverse of 2x2 symmetric matrix.
fn inverse_2x2(m: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2], String> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-14 {
        return Err("singular metric".to_string());
    }
    let inv_det = 1.0 / det;
    Ok([
        [m[1][1] * inv_det, -m[0][1] * inv_det],
        [-m[1][0] * inv_det, m[0][0] * inv_det],
    ])
}

/// Kinetic energy K = 1/2 v^T G v
pub fn kinetic_energy(
    v: (f64, f64),
    c: Complex64,
    config: &ManifoldConfig,
) -> Result<f64, String> {
    let g = induced_metric(c, config)?;
    let gv0 = g[0][0] * v.0 + g[0][1] * v.1;
    let gv1 = g[1][0] * v.0 + g[1][1] * v.1;
    Ok(0.5 * (v.0 * gv0 + v.1 * gv1))
}

/// Native potential U = kappa * sigma(c)
pub fn potential_energy(c: Complex64, config: &ManifoldConfig) -> Result<f64, String> {
    let sigma = mandelbrot_scale(c, config)?;
    Ok(config.kappa * sigma)
}

/// Total mechanical energy E = K + U
pub fn total_energy(
    v: (f64, f64),
    c: Complex64,
    config: &ManifoldConfig,
) -> Result<f64, String> {
    Ok(kinetic_energy(v, c, config)? + potential_energy(c, config)?)
}

/// Christoffel symbols Gamma^i_jk of the Levi-Civita connection.
///
/// Returns Gamma as [[[Gamma^0_00, Gamma^0_01], [Gamma^0_10, Gamma^0_11]],
///                    [[Gamma^1_00, Gamma^1_01], [Gamma^1_10, Gamma^1_11]]]
///
/// Computed from: Gamma^i_jk = 1/2 G^{il}(∂_j G_{kl} + ∂_k G_{jl} - ∂_l G_{jk})
pub fn christoffel_symbols(c: Complex64, config: &ManifoldConfig) -> Result<[[[f64; 2]; 2]; 2], String> {
    // Same h constraint as scale_hessian: must clear the f32 noise floor
    // of the distance field (metric derivatives amplify noise by 1/h).
    let h = 1e-4;
    let g = induced_metric(c, config)?;
    let g_inv = inverse_2x2(g)?;
    
    // Compute metric at neighboring points
    let c_px = Complex64::new(c.re + h, c.im);
    let g_px = induced_metric(c_px, config)?;
    
    let c_mx = Complex64::new(c.re - h, c.im);
    let g_mx = induced_metric(c_mx, config)?;
    
    let c_py = Complex64::new(c.re, c.im + h);
    let g_py = induced_metric(c_py, config)?;
    
    let c_my = Complex64::new(c.re, c.im - h);
    let g_my = induced_metric(c_my, config)?;
    
    // Metric derivatives: ∂_j G_{kl}
    let mut dg = [[[0.0; 2]; 2]; 2]; // dg[j][k][l]
    
    // ∂_0 G (derivative w.r.t. x)
    for k in 0..2 {
        for l in 0..2 {
            dg[0][k][l] = (g_px[k][l] - g_mx[k][l]) / (2.0 * h);
        }
    }
    
    // ∂_1 G (derivative w.r.t. y)
    for k in 0..2 {
        for l in 0..2 {
            dg[1][k][l] = (g_py[k][l] - g_my[k][l]) / (2.0 * h);
        }
    }
    
    // Compute Christoffel symbols
    let mut gamma = [[[0.0; 2]; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let mut sum = 0.0;
                for l in 0..2 {
                    sum += g_inv[i][l] * (dg[j][k][l] + dg[k][j][l] - dg[l][j][k]);
                }
                gamma[i][j][k] = 0.5 * sum;
            }
        }
    }
    
    Ok(gamma)
}

/// Geodesic acceleration term: Gamma^i_jk v^j v^k
///
/// This is the curvature-induced acceleration that redirects free motion
/// on the curved manifold.
pub fn geodesic_acceleration(
    v: (f64, f64),
    c: Complex64,
    config: &ManifoldConfig,
) -> Result<(f64, f64), String> {
    let gamma = christoffel_symbols(c, config)?;
    
    let mut a = [0.0; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let v_j = if j == 0 { v.0 } else { v.1 };
                let v_k = if k == 0 { v.0 } else { v.1 };
                a[i] += gamma[i][j][k] * v_j * v_k;
            }
        }
    }
    
    Ok((a[0], a[1]))
}

/// Potential force F_U = -G^{-1} ∇U = -kappa G^{-1} ∇sigma
pub fn potential_force(c: Complex64, config: &ManifoldConfig) -> Result<(f64, f64), String> {
    let (grad_x, grad_y) = scale_gradient(c, config)?;
    let g = induced_metric(c, config)?;
    let g_inv = inverse_2x2(g)?;
    
    let grad_u_x = config.kappa * grad_x;
    let grad_u_y = config.kappa * grad_y;
    
    let f_x = -(g_inv[0][0] * grad_u_x + g_inv[0][1] * grad_u_y);
    let f_y = -(g_inv[1][0] * grad_u_x + g_inv[1][1] * grad_u_y);
    
    Ok((f_x, f_y))
}

/// Apply generalized force through the metric: a = G^{-1} Q
pub fn apply_generalized_force(
    q: (f64, f64),
    c: Complex64,
    config: &ManifoldConfig,
) -> Result<(f64, f64), String> {
    let g = induced_metric(c, config)?;
    let g_inv = inverse_2x2(g)?;
    
    let a_x = g_inv[0][0] * q.0 + g_inv[0][1] * q.1;
    let a_y = g_inv[1][0] * q.0 + g_inv[1][1] * q.1;
    
    Ok((a_x, a_y))
}

/// Metric-consistent isotropic drag: Q_drag = -beta G v
pub fn drag_force(
    v: (f64, f64),
    c: Complex64,
    beta: f64,
    config: &ManifoldConfig,
) -> Result<(f64, f64), String> {
    let g = induced_metric(c, config)?;
    
    let gv_x = g[0][0] * v.0 + g[0][1] * v.1;
    let gv_y = g[1][0] * v.0 + g[1][1] * v.1;
    
    Ok((-beta * gv_x, -beta * gv_y))
}

/// Semi-implicit Euler integration step for manifold dynamics.
///
/// Integrates: r_ddot + Gamma(r_dot, r_dot) = -G^{-1}∇U + G^{-1}Q
///
/// Uses semi-implicit scheme:
///   v_new = v + a_total * dt
///   r_new = r + v_new * dt
///
/// Returns (new_c, new_v, energy_info)
pub fn integrate_step(
    c: Complex64,
    v: (f64, f64),
    q_control: (f64, f64),
    beta: f64,
    dt: f64,
    config: &ManifoldConfig,
) -> Result<(Complex64, (f64, f64), EnergyInfo), String> {
    // Compute forces
    let a_geodesic = geodesic_acceleration(v, c, config)?;
    let f_potential = potential_force(c, config)?;
    let q_drag = drag_force(v, c, beta, config)?;
    let q_total = (q_control.0 + q_drag.0, q_control.1 + q_drag.1);
    let a_force = apply_generalized_force(q_total, c, config)?;
    
    // Total acceleration
    let a_total = (
        -a_geodesic.0 + f_potential.0 + a_force.0,
        -a_geodesic.1 + f_potential.1 + a_force.1,
    );
    
    // Semi-implicit update
    let v_new = (v.0 + a_total.0 * dt, v.1 + a_total.1 * dt);
    let c_new = Complex64::new(c.re + v_new.0 * dt, c.im + v_new.1 * dt);
    
    // Energy accounting
    let e_old = total_energy(v, c, config)?;
    let e_new = total_energy(v_new, c_new, config)?;
    let k_old = kinetic_energy(v, c, config)?;
    let k_new = kinetic_energy(v_new, c_new, config)?;
    
    let energy_info = EnergyInfo {
        kinetic: k_new,
        potential: e_new - k_new,
        total: e_new,
        delta_total: e_new - e_old,
        delta_kinetic: k_new - k_old,
    };
    
    Ok((c_new, v_new, energy_info))
}

/// Energy diagnostic information
#[derive(Debug, Clone, Copy)]
pub struct EnergyInfo {
    pub kinetic: f64,
    pub potential: f64,
    pub total: f64,
    pub delta_total: f64,
    pub delta_kinetic: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scale_at_shore() {
        // At the Shore, scale should be high (rho small)
        let config = ManifoldConfig::default();
        let c_shore = Complex64::new(0.25, 0.0); // approximate Shore point
        let sigma = mandelbrot_scale(c_shore, &config).unwrap();
        assert!(sigma > 0.0, "Shore should have positive scale");
    }
    
    #[test]
    fn test_metric_positive_definite() {
        let config = ManifoldConfig::default();
        let c = Complex64::new(0.0, 0.0);
        let g = induced_metric(c, &config).unwrap();
        
        // Check symmetry
        assert!((g[0][1] - g[1][0]).abs() < 1e-10);
        
        // Check positive definiteness (det > 0 and g11 > 0)
        let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
        assert!(det > 0.0);
        assert!(g[0][0] > 0.0);
    }
    
    #[test]
    fn test_energy_conservation_no_forces() {
        let config = ManifoldConfig::default();
        let c = Complex64::new(0.0, 0.0);
        let v = (0.01, 0.01);
        let dt = 0.01;
        
        // Integrate without forces or drag
        let (_c_new, _v_new, info) = integrate_step(c, v, (0.0, 0.0), 0.0, dt, &config).unwrap();
        
        // Energy drift should be small
        assert!(info.delta_total.abs() < 0.01, "Energy should be approximately conserved");
    }
}
