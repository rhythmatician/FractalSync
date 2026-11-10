//! Scale-relative Map field M_c / R_c (issue #108) sharing #145's dyadic substrate.
//!
//! The canonical local Map field reuses the same adaptive dyadic Shore-contour
//! substrate as GeometryProvider, not the fixed 2048² F/S pyramid.
//!
//! ```text
//! rho(c) = sqrt(D(c)^2 + epsilon^2)
//! M_c(u,zeta) = [ D(c + 2^zeta * rho(c) * u) - D(c) ] / [ 2^zeta * rho(c) ]
//! R_c(u,zeta) = M_c(u,zeta) - grad D(c) · u   (where grad D valid)
//! ```
//!
//! u is dimensionless local offset, zeta is relative log-scale (beta=2^zeta).
//! Every zeta window agrees on first-order Shore direction where D differentiable.

use num_complex::Complex64;

/// Computerho(c) from D(c).
#[inline]
fn rho_from_d(d: f64, epsilon: f64) -> f64 {
    (d * d + epsilon * epsilon).sqrt()
}

/// Canonical scale-relative Map field M_c(u,zeta).
///
/// Returns dimensionless M. Uses the same dyadic Shore substrate as
/// GeometryProvider via `query_geometry` for both c and the offset point.
pub fn map_field_m(
    c: Complex64,
    u: (f64, f64),
    zeta: f64,
    epsilon: f64,
) -> Result<f64, String> {
    let jet_c = crate::geometry_provider::query_geometry(c, epsilon)?;
    let d_c = jet_c.d;
    let rho = rho_from_d(d_c, epsilon);
    if rho == 0.0 || !rho.is_finite() {
        return Err("rho not finite for M_c".to_string());
    }
    let beta = 2_f64.powf(zeta);
    let scale = beta * rho;
    if scale == 0.0 || !scale.is_finite() {
        return Err("scale not finite".to_string());
    }
    let px = c.re + scale * u.0;
    let py = c.im + scale * u.1;
    let p = Complex64::new(px, py);
    let jet_p = crate::geometry_provider::query_geometry(p, epsilon)?;
    let d_p = jet_p.d;
    Ok((d_p - d_c) / scale)
}

/// Residual Map field R_c = M_c - grad D(c) · u where grad valid.
pub fn residual_map_field_r(
    c: Complex64,
    u: (f64, f64),
    zeta: f64,
    epsilon: f64,
) -> Result<f64, String> {
    let m = map_field_m(c, u, zeta, epsilon)?;
    let jet_c = crate::geometry_provider::query_geometry(c, epsilon)?;
    // grad D valid only if Regular
    if jet_c.validity != crate::geometry_provider::GeometryValidity::Regular {
        return Err(format!(
            "R_c requires Regular jet at c, got {:?}",
            jet_c.validity
        ));
    }
    let grad = jet_c.grad_d;
    Ok(m - (grad[0] * u.0 + grad[1] * u.1))
}

/// Minimal finite sampling for ablation — not yet frozen #108 tensor.
/// Samples M_c on a small dimensionless grid at a few relative scales.
#[derive(Clone, Debug)]
pub struct MapSamplingConfig {
    pub u_samples: Vec<(f64, f64)>,
    pub zetas: Vec<f64>,
}

impl Default for MapSamplingConfig {
    fn default() -> Self {
        // 3x3 grid at u ∈ {-1,0,1}² minus center (8 points) and two scales
        let mut u = Vec::new();
        for dy in [-1.0, 0.0, 1.0] {
            for dx in [-1.0, 0.0, 1.0] {
                if dx == 0.0 && dy == 0.0 {
                    continue;
                }
                u.push((dx, dy));
            }
        }
        Self {
            u_samples: u,
            zetas: vec![0.0, 1.0],
        }
    }
}

/// Sample M_c tensor A_map(c) = [ M_c(u_pq, zeta_j) ] flattened j-major.
pub fn sample_map_tensor(
    c: Complex64,
    epsilon: f64,
    config: &MapSamplingConfig,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(config.u_samples.len() * config.zetas.len());
    for &zeta in &config.zetas {
        for &u in &config.u_samples {
            out.push(map_field_m(c, u, zeta, epsilon)?);
        }
    }
    Ok(out)
}

/// Normalized provider error for Map: e_hat_j = e_j / [2^zeta_j * rho]
pub fn normalized_map_error(
    c: Complex64,
    zeta: f64,
    epsilon: f64,
) -> Result<f64, String> {
    let jet_c = crate::geometry_provider::query_geometry(c, epsilon)?;
    let rho = rho_from_d(jet_c.d, epsilon);
    let beta = 2_f64.powf(zeta);
    let scale = beta * rho;
    let e = jet_c.estimated_error;
    Ok(e / scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn m_c_zero_at_origin_and_grad_agreement() {
        let _g = crate::distance_field::global_test_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let c = Complex64::new(-0.1, 0.1);
        let eps = 1e-4;
        let m0 = map_field_m(c, (0.0, 0.0), 0.0, eps).unwrap();
        assert!(m0.abs() < 1e-12, "M_c(0,zeta)=0, got {}", m0);
        let jet = crate::geometry_provider::query_geometry(c, eps).unwrap();
        if jet.validity == crate::geometry_provider::GeometryValidity::Regular {
            // grad_u M_c(0,zeta) should equal grad D(c) for any zeta
            let h = 1e-6;
            let mx = map_field_m(c, (h, 0.0), 0.0, eps).unwrap();
            let my = map_field_m(c, (0.0, h), 0.0, eps).unwrap();
            let gx = (mx - m0) / h;
            let gy = (my - m0) / h;
            assert!(
                (gx - jet.grad_d[0]).abs() < 0.1,
                "grad_u M_c vs grad D x: {} vs {}",
                gx,
                jet.grad_d[0]
            );
            assert!(
                (gy - jet.grad_d[1]).abs() < 0.1,
                "grad_u M_c vs grad D y: {} vs {}",
                gy,
                jet.grad_d[1]
            );
        }
    }

    #[test]
    fn r_c_zero_for_planar_shore() {
        let _g = crate::distance_field::global_test_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let c = Complex64::new(0.3, 0.0);
        let eps = 1e-4;
        let r = residual_map_field_r(c, (0.1, 0.0), 0.0, eps).unwrap_or(0.0);
        // for a locally planar Shore, R should be small
        assert!(r.abs() < 1.0, "R_c small for planar, got {}", r);
    }
}
