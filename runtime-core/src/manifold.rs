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
//! Physics receives Controls as generalized force COVECTORS; it does not know
//! about music. Generalized forces are summed as covectors and converted to
//! coordinate acceleration exactly once via G^{-1} (see [`integrate_step`]).
//!
//! Signed realm classification comes from the canonical signed SDF sampler in
//! `distance_field`; this module does not reconstruct sign with an escape
//! heuristic. Derivative finite-difference steps are derived from the SDF
//! provider's pixel spacing, not a magic constant.

use num_complex::Complex64;

/// Configuration manifold parameters
#[derive(Clone, Debug)]
pub struct ManifoldConfig {
    /// Reference distance for scale definition (typically 0.1)
    pub d_ref: f64,
    /// Regularization floor epsilon in rho = sqrt(D^2 + epsilon^2).
    ///
    /// This is the smooth-scale regularization floor, NOT the SDF pixel
    /// spacing and NOT the finite-difference step. It sets the finite but
    /// high scale at The Shore and keeps derivatives smooth through the
    /// crossing. It is a capability of the current Map provider, not a
    /// permanent architectural maximum.
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
/// This delegates to the canonical signed SDF sampler in `distance_field`
/// (`sample_signed_distance_field`), which interpolates the stored signed
/// values directly. It does NOT reconstruct the sign with a separate
/// escape-iteration heuristic — the baked artifact is already signed and is
/// the single authority for realm classification.
pub fn signed_distance(c: Complex64) -> Result<f64, String> {
    let dist_signed = crate::distance_field::sample_signed_distance_field(&[c])?
        .into_iter()
        .next()
        .ok_or_else(|| "empty distance sample".to_string())?;
    Ok(dist_signed as f64)
}

/// The finite-difference step used for derivatives of the sampled scale field.
///
/// This is chosen from the SDF provider's pixel spacing (via
/// `distance_field::distance_field_metadata`), NOT a magic constant. The
/// sampled field is a smooth distance estimate; the finite-difference step
/// must clear the f32 quantization noise floor of the raster while staying
/// small enough that the local curvature of sigma ~ log2(d_ref/rho) is
/// resolved. A step of one full pixel is too coarse near the Shore (where
/// sigma varies as 1/D) and degrades energy conservation, so we use a small
/// fraction of a pixel. If no field is loaded a conservative fallback is
/// used so callers still get a deterministic value.
pub fn derivative_step() -> f64 {
    match crate::distance_field::distance_field_metadata() {
        Some((_, _, _, _, _, _, dx, dy)) => {
            let px = dx.max(dy);
            if px > 0.0 {
                (px * DERIVATIVE_STEP_PIXEL_FRACTION).max(MIN_DERIVATIVE_STEP)
            } else {
                DEFAULT_DERIVATIVE_STEP
            }
        }
        None => DEFAULT_DERIVATIVE_STEP,
    }
}

/// Fraction of a pixel used as the finite-difference step. The field is a
/// smooth distance estimate, so a step well below one pixel resolves the
/// local curvature while still clearing the f32 quantization noise floor.
const DERIVATIVE_STEP_PIXEL_FRACTION: f64 = 1.0 / 24.0;

/// Floor on the pixel-derived step (keeps the step from collapsing if a
/// provider reports an unusually fine raster).
const MIN_DERIVATIVE_STEP: f64 = 1e-5;

/// Fallback finite-difference step when no distance field is loaded.
const DEFAULT_DERIVATIVE_STEP: f64 = 1e-4;

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
/// The finite-difference step is derived from the SDF provider's pixel
/// spacing (see [`derivative_step`]) so it stays above the interpolation
/// cell / subpixel-refinement noise scale rather than being a magic constant.
pub fn scale_gradient(c: Complex64, config: &ManifoldConfig) -> Result<(f64, f64), String> {
    let h = derivative_step();

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
/// The finite-difference step is derived from the SDF provider's pixel
/// spacing (see [`derivative_step`]). Second differences amplify noise by an
/// extra 1/h, so the step must clear the interpolation-cell noise scale.
pub fn scale_hessian(c: Complex64, config: &ManifoldConfig) -> Result<[[f64; 2]; 2], String> {
    let h = derivative_step();

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
/// For the graph metric G = I + lambda^2 grad(sigma) grad(sigma)^T, the
/// connection has a closed form that avoids finite-differencing an already
/// finite-differenced metric:
///
///   Gamma^i_jk = lambda^2 * sigma_i * sigma_jk / (1 + lambda^2 ||grad sigma||^2)
///
/// where sigma_i = ∂_i sigma and sigma_jk = ∂_j ∂_k sigma (the Hessian).
/// This uses the same gradient/Hessian authority as the metric and reduces
/// nested finite-difference noise. (Derivation: ∂_j G_{kl} + ∂_k G_{jl} -
/// ∂_l G_{jk} = 2 lambda^2 sigma_jk sigma_l, and G^{-1} grad sigma =
/// grad sigma / (1 + lambda^2 ||grad sigma||^2) by Sherman-Morrison.)
pub fn christoffel_symbols(c: Complex64, config: &ManifoldConfig) -> Result<[[[f64; 2]; 2]; 2], String> {
    let (gx, gy) = scale_gradient(c, config)?;
    let hess = scale_hessian(c, config)?;
    let lsq = config.lambda_sq;

    // Denominator: 1 + lambda^2 ||grad sigma||^2
    let grad_sq = gx * gx + gy * gy;
    let denom = 1.0 + lsq * grad_sq;
    if !denom.is_finite() || denom.abs() < 1e-30 {
        return Err("Christoffel denominator singular".to_string());
    }

    // grad sigma components (sigma_0 = gx, sigma_1 = gy)
    let sig = [gx, gy];
    // Hessian components sigma_jk (symmetric)
    let hxx = hess[0][0];
    let hxy = hess[0][1];
    let hyy = hess[1][1];

    let mut gamma = [[[0.0; 2]; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let sigma_jk = match (j, k) {
                    (0, 0) => hxx,
                    (0, 1) | (1, 0) => hxy,
                    (1, 1) => hyy,
                    _ => unreachable!(),
                };
                gamma[i][j][k] = lsq * sig[i] * sigma_jk / denom;
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

/// Generalized potential force covector: Q_potential = -grad U = -kappa grad sigma.
///
/// This is a generalized force COVECTOR (lower index), not a coordinate
/// acceleration. It is converted to acceleration by [`apply_generalized_force`]
/// (the single place where G^{-1} maps a covector to coordinate acceleration).
/// The sign makes high-scale Shore geometry a potential ridge: the force points
/// downhill, away from the Shore.
pub fn potential_force(c: Complex64, config: &ManifoldConfig) -> Result<(f64, f64), String> {
    let (grad_x, grad_y) = scale_gradient(c, config)?;
    Ok((-config.kappa * grad_x, -config.kappa * grad_y))
}

/// Convert a generalized force covector to coordinate acceleration: a = G^{-1} Q.
///
/// This is the single place where the metric inverse maps a generalized force
/// covector (lower index) into a coordinate acceleration. All generalized
/// forces (potential, control, drag) are summed as covectors and converted here.
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

/// Metric-consistent isotropic drag covector: Q_drag = -beta G v.
///
/// This is a generalized force COVECTOR (lower index), not a coordinate
/// acceleration. Its power P = v^T Q_drag = -beta v^T G v <= 0, so drag can
/// never inject mechanical energy.
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
/// Generalized forces are summed as COVECTORS and converted to coordinate
/// acceleration exactly once via G^{-1}:
///
///   Q_potential = -grad U
///   Q_drag      = -beta G v
///   Q_total     = Q_potential + Q_control + Q_drag
///   a_force     = G^{-1} Q_total
///   a_total     = -Gamma(v,v) + a_force
///
/// `q_control` is a generalized force covector (units of force, NOT an
/// already-integrated impulse). Continuous force is integrated exactly once:
///
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
    // Geodesic (curvature) acceleration: -Gamma(v, v).
    let a_geodesic = geodesic_acceleration(v, c, config)?;

    // Sum generalized force covectors: potential + control + drag.
    let q_potential = potential_force(c, config)?;
    let q_drag = drag_force(v, c, beta, config)?;
    let q_total = (
        q_potential.0 + q_control.0 + q_drag.0,
        q_potential.1 + q_control.1 + q_drag.1,
    );

    // Single G^{-1} conversion of the summed covector into acceleration.
    let a_force = apply_generalized_force(q_total, c, config)?;

    // Total acceleration.
    let a_total = (
        -a_geodesic.0 + a_force.0,
        -a_geodesic.1 + a_force.1,
    );

    // Semi-implicit update: continuous force integrated exactly once.
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
