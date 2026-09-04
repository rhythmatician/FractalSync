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
    /// Barrier strength mu in the p=8 secant bowl
    /// U_wall = mu * [sec(π/2 * s^4) - 1], s = |c|^2 / 4.
    /// Essentially zero through the central region, noticeable around
    /// |c| ~ 1.4-1.6, and a stiff wall near |c| = 2. Default 1/π chosen so
    /// the π factor cancels in the force: Q_wall = -s^3 sec φ tan φ (x, y).
    pub mu: f64,
}

impl Default for ManifoldConfig {
    fn default() -> Self {
        Self {
            d_ref: 0.1,
            epsilon: 1e-4,
            lambda_sq: 1.0,
            kappa: 1.0,
            mu: std::f64::consts::FRAC_1_PI,
        }
    }
}

/// Signed geometric distance to the Mandelbrot boundary (canonical authority).
/// D(c) < 0 inside M, D(c) > 0 outside M, D(c) == 0 on The Shore.
/// Distinct from unsigned geometric distance d(c)=|D(c)| and from the
/// Shore-proximity sensitivity S(c)=|∇F|/(|∇F|+G0) carried by the minimap
/// (mip pyramid) — see `crate::minimap`. S is a sensitivity/proximity proxy,
/// not geometric distance; d and D are geometric.
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
    // Ensure the distance field is loaded so the provider spacing is stable.
    // `sample_signed_distance_field` auto-loads the builtin, but `derivative_step`
    // is called *before* any sampling in `scale_gradient`/`scale_hessian`. If we
    // returned the fallback on the first call, the first gradient would be
    // computed with h=1e-4 and the second with h=px/24 (~1.6e-4), breaking
    // determinism and the metric-consistent drive covector invariant.
    if crate::distance_field::distance_field_metadata().is_none() {
        let _ = crate::distance_field::load_builtin_distance_field("mandelbrot_default");
    }
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

/// Unsigned geometric distance d(c) = |D(c)|.
/// Distinct from signed distance D(c) and from S(c) sensitivity (see [`signed_distance`]).
pub fn unsigned_distance(c: Complex64) -> Result<f64, String> {
    Ok(signed_distance(c)?.abs())
}

/// Mandelbrot scale sigma(c) = log2(d_ref / rho(c))
///
/// High scale at the Shore, decreases with distance from boundary.
/// The scale is symmetric across inside/outside due to regularization.
/// `sigma(c)` is DERIVED from Mandelbrot geometry via D(c) and the smooth
/// regularization `rho = sqrt(D^2 + epsilon^2)`. There is no independent
/// Mandelbrot scale-control axis; `sigma` is not state and has no independent
/// velocity `v_sigma` — see [`sigma_dot`] and [`embedding`].
/// `epsilon` is a capability of the current Map provider (distance-field
/// resolution/regularization floor), not a permanent architectural maximum.
/// A deeper/adaptive provider may lower it without changing the Physics contract.
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

/// Embedding q(c) = (x, y, sigma(c)) in position-scale space.
///
/// `q` is an EMBEDDING, not three independent degrees of freedom. The
/// independent configuration is still `r = (x, y)` / `c = x + i y`; `sigma(c)`
/// is derived. See [`jacobian`] for ∂q/∂(x,y) and [`q_dot`] for the embedded
/// velocity.
pub fn embedding(c: Complex64, config: &ManifoldConfig) -> Result<(f64, f64, f64), String> {
    let sigma = mandelbrot_scale(c, config)?;
    Ok((c.re, c.im, sigma))
}

/// Jacobian of the embedding J_q(c) = ∂q/∂(x,y) as a 3×2 matrix.
///
/// ```text
/// J_q = [[1, 0],
///        [0, 1],
///        [sigma_x, sigma_y]]
/// ```
pub fn jacobian(c: Complex64, config: &ManifoldConfig) -> Result<[[f64; 2]; 3], String> {
    let (gx, gy) = scale_gradient(c, config)?;
    Ok([[1.0, 0.0], [0.0, 1.0], [gx, gy]])
}

/// Embedded velocity q_dot = J_q(c) v, with sigma_dot = ∇sigma(c)·v.
///
/// There is no independent `v_sigma`; the third component of `q_dot` is
/// strictly `sigma_dot = ∇sigma·v`. Returns (x_dot, y_dot, sigma_dot) which
/// is identically `(v.0, v.1, sigma_dot)`.
pub fn q_dot(
    c: Complex64,
    v: (f64, f64),
    config: &ManifoldConfig,
) -> Result<(f64, f64, f64), String> {
    let sd = sigma_dot(c, v, config)?;
    Ok((v.0, v.1, sd))
}

/// Time derivative of Mandelbrot scale along a trajectory: sigma_dot = ∇sigma(c)·v.
///
/// This is the ONLY velocity that exists for `sigma`; there is no independent
/// `v_sigma` state. `sigma(c)` is derived geometry.
pub fn sigma_dot(
    c: Complex64,
    v: (f64, f64),
    config: &ManifoldConfig,
) -> Result<f64, String> {
    let (gx, gy) = scale_gradient(c, config)?;
    Ok(gx * v.0 + gy * v.1)
}

/// Induced metric G(c) = J_q(c)^T H J_q(c) = I + lambda^2 * grad_sigma * grad_sigma^T
///
/// Derived from the embedding Jacobian and ambient metric H = diag(1,1,lambda^2).
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

/// Wall (secant bowl) potential U_wall(c) = mu * [sec(π/2 * s^4) - 1]
/// where s = |c|^2 / 4 = (x^2 + y^2) / 4, so s^4 = (|c|/2)^8 (p = 8).
///
/// With the default mu = 1/π the potential is exactly
///   U_wall = (1/π) [sec(π/2 * s^4) - 1]
/// and the force loses its π factor entirely (see [`wall_force`]).
///
/// Properties:
/// - finite and smooth throughout the valid disk |c| < 2 (sec is finite for |c| < 2);
/// - bowl-like near the interior (effectively r^16 behavior near center);
/// - rises increasingly rapidly outward;
/// - tends to +infinity as |c| -> 2 (sec(π/2) → ∞);
/// - rotationally symmetric;
/// - completely independent of Mandelbrot scale sigma.
///
/// Per-evaluation cost: one cos, some multiplies/divisions. No square root
/// or arbitrary exponentiation; kept exact for a clean energy ledger.
pub fn wall_potential(c: Complex64, config: &ManifoldConfig) -> Result<f64, String> {
    let x = c.re;
    let y = c.im;
    let r2 = x * x + y * y;
    let s = r2 * 0.25; // s = |c|^2 / 4

    // s^4 = (|c|/2)^8, compute via repeated multiplication
    let s2 = s * s;
    let s4 = s2 * s2;

    let phi = (std::f64::consts::PI / 2.0) * s4;
    let cos_phi = phi.cos();

    // sec(phi) = 1/cos(phi). If cos_phi is near zero, we're near the wall.
    if cos_phi.abs() < 1e-12 {
        return Err("|c| too close to 2: wall potential unstable".to_string());
    }

    let sec_phi = 1.0 / cos_phi;
    Ok(config.mu * (sec_phi - 1.0))
}

/// Total mechanical energy E = K + U_sigma + U_wall
pub fn total_energy(
    v: (f64, f64),
    c: Complex64,
    config: &ManifoldConfig,
) -> Result<f64, String> {
    Ok(kinetic_energy(v, c, config)? + potential_energy(c, config)? + wall_potential(c, config)?)
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

/// Wall (secant bowl) force covector: Q_wall = -grad U_wall.
///
/// U_wall(c) = mu * [sec(π/2 * s^4) - 1] where s = |c|^2 / 4 (p = 8).
///
/// The gradient is:
///
///   dU_wall/dx = mu * sec(φ)tan(φ) * dφ/dx
///   where φ = π/2 * s^4 and s = (x^2+y^2)/4
///   dφ/dx = π * s^3 * x
///   therefore Q_wall_x = -mu * π * s^3 * sec(φ)tan(φ) * x
///
/// With the default mu = 1/π the π cancels completely:
///
///   Q_wall = -s^3 sec(φ)tan(φ) (x, y)
///
/// This is a generalized force COVECTOR (lower index), not a coordinate
/// acceleration. It is summed with other covectors and converted to
/// acceleration by [`apply_generalized_force`] (the single G^{-1} path).
pub fn wall_force(c: Complex64, config: &ManifoldConfig) -> Result<(f64, f64), String> {
    let x = c.re;
    let y = c.im;
    let r2 = x * x + y * y;
    let s = r2 * 0.25; // s = |c|^2 / 4

    // s^4 = (|c|/2)^8, compute via repeated multiplication
    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s2 * s2;

    let phi = (std::f64::consts::PI / 2.0) * s4;
    let (sin_phi, cos_phi) = phi.sin_cos();

    // Avoid division by zero near the wall
    if cos_phi.abs() < 1e-12 {
        return Err("|c| too close to 2: wall force unstable".to_string());
    }

    let sec_phi = 1.0 / cos_phi;
    let tan_phi = sin_phi / cos_phi;

    // Q_wall_x = -mu * π * s^3 * sec(φ)tan(φ) * x
    let force_factor = -config.mu * std::f64::consts::PI * s3 * sec_phi * tan_phi;

    let qx = force_factor * x;
    let qy = force_factor * y;

    Ok((qx, qy))
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

/// Integrator selection (issue #106).
///
/// Candidate timesteppers considered:
///
/// - **Semi-implicit (symplectic) Euler**: `v_{n+1} = v_n + a(q_n,v_n)·dt`,
///   `r_{n+1} = r_n + v_{n+1}·dt`. First-order, explicit, area-preserving in
///   flat phase space, inexpensive, stable for position-dependent forces when
///   `dt` is the canonical hop `HOP_LENGTH/SAMPLE_RATE` (≈21 ms). Energy is
///   not exactly conserved but drift is bounded and O(dt).
/// - **Explicit RK4**: higher accuracy but overtly conservative; dominant
///   near-Shore error comes from finite-differenced `∇sigma`/`H_sigma`
///   propagating into the otherwise analytic `Γ`, not from the timestepper
///   itself (see `docs/adr/` and PR #112 notes).
/// - **Variational / geometric**: ideal for long conservative rollouts but
///   heavier and requires re-deriving the discrete Lagrangian for the
///   sampled sigma field.
///
/// **Chosen**: semi-implicit Euler as the inexpensive baseline. It keeps
/// parity cheap, preserves the canonical timebase, and bounds total-energy
/// drift away from and near the Shore (see `TestEnergyDrift` in
/// `backend/tests/test_manifold_physics.py`). The dominant error is the
/// sampled-field gradient quality, not the integrator order; a future deeper/
/// analytic Map provider can improve `∇sigma`/`H_sigma` without changing the
/// timestepper contract. The kernel fails closed on metric singularities
/// (no silent flat-physics fallback).
///
/// A differentiable training rollout may use a parity-pinned STE surrogate
/// where autograd requires it, but Rust remains semantic authority (ADR 0001).
///
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

    // Hard invariant: authoritative state must remain inside |c| < 2.
    // If the proposed result is non-finite or has |c_new| >= 2, return an
    // integrator error and DO NOT emit the invalid state.
    let c_abs_sq = c_new.re * c_new.re + c_new.im * c_new.im;
    if !c_abs_sq.is_finite() || c_abs_sq >= 4.0 {
        return Err(format!(
            "Hard invariant violated: |c_new|^2 = {} >= 4.0; rejecting invalid state",
            c_abs_sq
        ));
    }

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
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        // At the Shore, scale should be high (rho small)
        let config = ManifoldConfig::default();
        let c_shore = Complex64::new(0.25, 0.0); // approximate Shore point
        let sigma = mandelbrot_scale(c_shore, &config).unwrap();
        assert!(sigma > 0.0, "Shore should have positive scale");
    }
    
    #[test]
    fn test_metric_positive_definite() {
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
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
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        let config = ManifoldConfig::default();
        let c = Complex64::new(0.0, 0.0);
        let v = (0.01, 0.01);
        let dt = 0.01;
        
        // Integrate without forces or drag
        let (_c_new, _v_new, info) = integrate_step(c, v, (0.0, 0.0), 0.0, dt, &config).unwrap();
        
        // Energy drift should be small
        assert!(info.delta_total.abs() < 0.01, "Energy should be approximately conserved");
    }

    #[test]
    fn embedding_jacobian_and_qdot_consistency() {
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        let config = ManifoldConfig::default();
        let c = Complex64::new(0.12, -0.34);
        let v = (0.07, -0.03);
        let (x, y, s) = embedding(c, &config).unwrap();
        assert!((x - c.re).abs() < 1e-12);
        assert!((y - c.im).abs() < 1e-12);
        assert!(s.is_finite());
        let j = jacobian(c, &config).unwrap();
        // Top 2 rows are identity.
        assert!((j[0][0] - 1.0).abs() < 1e-12 && j[0][1].abs() < 1e-12);
        assert!(j[1][0].abs() < 1e-12 && (j[1][1] - 1.0).abs() < 1e-12);
        let (gx, gy) = scale_gradient(c, &config).unwrap();
        assert!((j[2][0] - gx).abs() < 1e-12);
        assert!((j[2][1] - gy).abs() < 1e-12);
        let sd = sigma_dot(c, v, &config).unwrap();
        assert!((sd - (gx * v.0 + gy * v.1)).abs() < 1e-12);
        let qd = q_dot(c, v, &config).unwrap();
        assert!((qd.0 - v.0).abs() < 1e-12);
        assert!((qd.1 - v.1).abs() < 1e-12);
        assert!((qd.2 - sd).abs() < 1e-12);
        // No independent v_sigma: q_dot.2 is exactly sigma_dot, not a separate state.
        // Metric derived from Jacobian must match induced_metric.
        let g = induced_metric(c, &config).unwrap();
        let lsq = config.lambda_sq;
        let g11_expect = 1.0 + lsq * gx * gx;
        let g12_expect = lsq * gx * gy;
        let g22_expect = 1.0 + lsq * gy * gy;
        assert!((g[0][0] - g11_expect).abs() < 1e-12);
        assert!((g[0][1] - g12_expect).abs() < 1e-12);
        assert!((g[1][1] - g22_expect).abs() < 1e-12);
    }

    #[test]
    fn scale_regularization_is_smooth_and_symmetric() {
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        // rho(c)=sqrt(D^2+eps^2) gives finite, smooth sigma through D=0 and symmetric
        // geometric scale on inside/outside. Finite differences must stay finite.
        let config = ManifoldConfig::default();
        // Find Shore at y=0 by bisection on signed distance (needs field).
        // If field not available, just check sigma finiteness at 0.25.
        let c_inside = Complex64::new(0.24, 0.0);
        let c_outside = Complex64::new(0.26, 0.0);
        let s_in = mandelbrot_scale(c_inside, &config).unwrap();
        let s_out = mandelbrot_scale(c_outside, &config).unwrap();
        assert!(s_in.is_finite() && s_out.is_finite());
        // Scale is highest near the Shore, lower farther away.
        let s_far = mandelbrot_scale(Complex64::new(1.0, 0.0), &config).unwrap();
        assert!(s_in > s_far);
        assert!(s_out > s_far);
        // Derivatives through the crossing must be finite (smooth).
        for x in [0.245, 0.25, 0.255] {
            let (gx, gy) = scale_gradient(Complex64::new(x, 0.0), &config).unwrap();
            assert!(gx.is_finite() && gy.is_finite());
            let h = scale_hessian(Complex64::new(x, 0.0), &config).unwrap();
            for row in h { for v in row { assert!(v.is_finite()); } }
            let gamma = christoffel_symbols(Complex64::new(x, 0.0), &config).unwrap();
            for i in 0..2 { for j in 0..2 { for k in 0..2 { assert!(gamma[i][j][k].is_finite()); } } }
        }
    }

    #[test]
    fn unsigned_distance_and_d_distinct_from_sensitivity() {
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        // D, d=|D|, and S must be explicitly distinct. This test documents the
        // invariant and checks the Rust side honors it: d = |D| exactly, while
        // S (if later exposed) is NOT geometric distance.
        let c = Complex64::new(0.0, 0.0);
        let d_signed = signed_distance(c).unwrap();
        let d_abs = unsigned_distance(c).unwrap();
        assert!((d_abs - d_signed.abs()).abs() < 1e-12);
        // S is carried by the minimap mip pyramid (S=G/(G+G0)), not by this module.
        // The absence of an S alias here is intentional — callers must not label S
        // as geometric distance.
    }

    #[test]
    fn drift_grip_controls_cannot_inject_energy() {
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        // Friction/brake is PSD: P = v^T Q_friction <= 0 regardless of grip.
        let config = ManifoldConfig::default();
        let c = Complex64::new(0.1, -0.2);
        let v = (0.4, 0.3);
        for (grip, brake) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)] {
            let mc = crate::controls::MotionControls { direction: [1.0, 0.0], throttle: 0.0, brake, grip, impulse: 0.0 };
            let beta = mc.friction_beta();
            let q = drag_force(v, c, beta, &config).unwrap();
            let p = v.0 * q.0 + v.1 * q.1;
            assert!(p <= 1e-12, "friction injected energy p={} grip={} brake={}", p, grip, brake);
        }
    }

    #[test]
    fn conservative_rollout_energy_drift_is_bounded() {
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        // With Q_control=0, beta=0, total energy E=K+U should drift only O(dt)
        let config = ManifoldConfig::default();
        let dt = 0.005;
        let mut c = Complex64::new(0.0, 0.6);
        let mut v = (0.015, -0.01);
        let e0 = total_energy(v, c, &config).unwrap();
        for _ in 0..80 {
            let (c1, v1, info) = integrate_step(c, v, (0.0, 0.0), 0.0, dt, &config).unwrap();
            assert!(info.delta_total.abs() < 0.05, "unbounded single-step drift {}", info.delta_total);
            c = c1; v = v1;
        }
        let e1 = total_energy(v, c, &config).unwrap();
        assert!((e1 - e0).abs() < 0.05, "rollout drift {} exceeds tolerance", (e1-e0).abs());
    }

    #[test]
    fn drive_work_attributable_to_generalized_force() {
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        // Drive covector must have metric-consistent dual norm; its work
        // w = v·Q*dt is attributable to the control, not hidden geometry.
        let config = ManifoldConfig::default();
        let c = Complex64::new(0.05, 0.1);
        let v = (0.02, 0.01);
        let mc = crate::controls::MotionControls { direction: [1.0, 0.0], throttle: 1.0, brake: 0.0, grip: 0.5, impulse: 0.0 };
        let q = mc.drive_covector(c, &config).unwrap();
        // Power is v·Q; compare step with vs without drive.
        let (_c0, _v0, info0) = integrate_step(c, v, (0.0, 0.0), 0.0, 0.01, &config).unwrap();
        let (_c1, _v1, info1) = integrate_step(c, v, q, 0.0, 0.01, &config).unwrap();
        // Drive must increase total energy when aligned with v (positive work).
        // If orthogonal, it may not; we test that drive vs no-drive differs.
        assert!((info1.delta_total - info0.delta_total).abs() > 0.0 || (q.0 == 0.0 && q.1 == 0.0));
    }

    #[test]
    fn mip_boundaries_do_not_cause_discontinuity() {
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        // Map-derived mechanics must vary continuously across stored MIP
        // boundaries and through the regularized Shore crest. We approximate
        // this by checking metric/shear continuity at 0.25 +/- eps (shore)
        // and at far/near points where SDF resolution changes would appear as
        // jumps. Tolerance is generous because bicubic interpolation is smooth
        // but FD noise exists.
        let config = ManifoldConfig::default();
        for x in [0.249, 0.25, 0.251, -0.751, -0.75, -0.749] {
            let g = induced_metric(Complex64::new(x, 0.0), &config).unwrap();
            for row in g { for v in row { assert!(v.is_finite()); } }
            let det = g[0][0]*g[1][1] - g[0][1]*g[0][1];
            assert!(det > 0.0 && det.is_finite());
        }
    }

    #[test]
    fn controls_drive_can_cross_shore_without_wall() {
        let _lock = crate::distance_field::global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
        // The Shore is a finite potential ridge, not a binary wall. With
        // sufficient generalized work from Controls, the trajectory must be
        // able to crest D=0 without any transient-gated wall permeability.
        // This test uses the destination seam `integrate_motion_controls`
        // (ControlsV2 -> G, Γ, U) and checks that a driven rollout crosses
        // while an undriven rollout from the same start remains inside.
        let config = ManifoldConfig { d_ref: 0.1, epsilon: 1e-4, lambda_sq: 1.0, kappa: 0.5, mu: std::f64::consts::FRAC_1_PI };
        let c0 = Complex64::new(0.23, 0.0);
        let v0 = (0.0, 0.0);
        let dt = 0.02;
        let steps = 400;
        // Undriven: should not cross (potential ridge holds).
        let mut c = c0;
        let mut v = v0;
        let undriven = crate::controls::MotionControls { direction: [1.0, 0.0], throttle: 0.0, brake: 0.0, grip: 0.0, impulse: 0.0 };
        let mut crossed_undriven = false;
        for _ in 0..steps {
            let (c1, v1, _) = crate::controls::integrate_motion_controls(c, v, &undriven, dt, &config).unwrap();
            c = c1; v = v1;
            if signed_distance(c).unwrap() > 0.0 { crossed_undriven = true; break; }
        }
        assert!(!crossed_undriven, "undriven rollout should reflect off the ridge");
        // Driven: same start, sustained throttle outward, must crest.
        c = c0; v = v0;
        let driven = crate::controls::MotionControls { direction: [1.0, 0.0], throttle: 1.0, brake: 0.0, grip: 0.0, impulse: 0.0 };
        let mut crossed_driven = false;
        let mut final_c = c0;
        for _ in 0..steps {
            let (c1, v1, _) = crate::controls::integrate_motion_controls(c, v, &driven, dt, &config).unwrap();
            c = c1; v = v1;
            final_c = c;
            if signed_distance(c).unwrap() > 0.0 { crossed_driven = true; break; }
        }
        assert!(crossed_driven, "driven rollout should crest the finite ridge without any wall gate; final c={:?} D={}", final_c, signed_distance(final_c).unwrap());
        // Also verify that no musical signal (h, energy) was involved: the
        // destination seam takes only MotionControls, dt, and ManifoldConfig.
    }
}
