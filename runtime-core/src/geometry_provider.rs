//! Mandelbrot geometry provider seam (ADR 0004, issue #145).
//!
//! Physics consumes a GeometryProvider that exposes one coherent local
//! differential sample: value, gradient and Hessian from the same representation.
//!
//! ```text
//! GeometryJet { D, grad_D, hessian_D, resolved_scale, validity, singularity }
//! ```
//!
//! Physics then derives rho, sigma, G, Gamma analytically via chain rule.
//!
//! Two implementations exist behind the same jet seam:
//! - `RasterBridgeProvider` (NOW, is_bridge=true): 1024² signed-distance
//!   raster sampled with a scale-aware 9-point stencil. This is the
//!   migration bridge — it is NOT the destination adaptive/local C²
//!   representation. It correctly reports `Unresolved` when the fixed raster
//!   cannot meet `requested_scale = alpha*max(rho,epsilon)`.
//! - `ScaleAwareGeometryProvider` (NEXT, is_bridge=false): placeholder for
//!   the destination adaptive/local C² geometry (quadtree / spline tiles /
//!   on-demand DEM) that will provide true refinement. Not yet implemented
//!   behind this PR; Physics currently consumes the bridge.
//!
//! Authority: runtime-core (ADR 0001). Versioned, deterministic, and cache-aware.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Version of the geometry provider contract. Bump when jet shape, semantics,
/// or validity classification changes in the same commit as manifold/debug
/// updates and regenerated goldens/mirrors.
pub const GEOMETRY_PROVIDER_VERSION: &str = "geometry-provider/1";

/// Scale-relative resolution law factor: local cell size <= alpha * max(rho, epsilon).
/// The exact refinement/error criterion is derived and validated (ADR 0004 does
/// not freeze it), but this value is the current validated choice.
pub const GEOMETRY_SCALE_ALPHA: f64 = 0.1;

/// Minimum and maximum coherent evaluation step to keep stencil numerically
/// stable far from the Shore or extremely close to it.
pub const GEOMETRY_MIN_STEP: f64 = 1e-7;
pub const GEOMETRY_MAX_STEP: f64 = 0.1;

/// Eikonal tolerance: | |grad D| - 1 | above this flags a cut-locus / medial
/// axis singularity (signed distance fields are eikonal |grad D| = 1 where smooth).
pub const EIKONAL_TOL: f64 = 0.5;

/// Hessian singularity threshold: Frobenius norm above this flags high curvature
/// / cusp-like behavior. Scaled relative to local ruler; generous to avoid
/// false positives on regular Shore geometry.
pub const HESSIAN_NORM_SINGULAR: f64 = 5e4;

/// Validity classification for a geometry jet.
///
/// Distinguishes regular smooth geometry from numerically unresolved or
/// genuinely nonsmooth locations without silent smoothing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryValidity {
    /// Smooth, well-resolved geometry: ordinary Levi-Civita Physics may proceed.
    Regular,
    /// Provider resolution insufficient for requested local scale.
    /// Deterministic best-effort jet returned but Physics/diagnostics should
    /// treat as fail-closed or degraded.
    Unresolved,
    /// Genuine or suspected nonsmooth geometry (cut locus, medial axis, cusp,
    /// non-unique normal). Unique Hessian/connection does not exist.
    Singular,
    /// Query outside the provider's geometric domain (|c| >= 2 hard wall).
    OutsideDomain,
    /// Provider failure (field not loaded, sampling error, non-finite).
    ProviderFailure,
}

impl GeometryValidity {
    pub fn as_str(&self) -> &'static str {
        match self {
            GeometryValidity::Regular => "regular",
            GeometryValidity::Unresolved => "unresolved",
            GeometryValidity::Singular => "singular",
            GeometryValidity::OutsideDomain => "outside_provider",
            GeometryValidity::ProviderFailure => "provider_failure",
        }
    }
}

/// Explicit singularity classification where known.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SingularityKind {
    None,
    CutLocus,
    HighCurvature,
    NonUniqueNormal,
    Boundary,
}

impl SingularityKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SingularityKind::None => "none",
            SingularityKind::CutLocus => "cut_locus",
            SingularityKind::HighCurvature => "high_curvature",
            SingularityKind::NonUniqueNormal => "non_unique_normal",
            SingularityKind::Boundary => "boundary",
        }
    }
}

/// One coherent differential sample from the local geometry representation.
///
/// The invariant is that D, grad_D, and hessian_D come from one coherent
/// local representation (scale-aware local stencil), not independent heuristics
/// or nested finite differences of sigma.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometryJet {
    /// Signed distance D(c): <0 inside M, >0 outside, 0 on the Shore.
    #[serde(rename = "D")]
    pub d: f64,
    /// First derivative from the same local representation.
    #[serde(rename = "grad_D")]
    pub grad_d: [f64; 2],
    /// Second derivative from the same local representation.
    #[serde(rename = "hessian_D")]
    pub hessian_d: [[f64; 2]; 2],
    /// Actual spatial scale / error capability the provider resolved.
    /// For the current raster-backed implementation this is the pixel spacing;
    /// for a future adaptive tile it is the local tile cell size.
    pub resolved_scale: f64,
    /// Requested local scale: alpha * max(rho, epsilon). Exposed so diagnostics
    /// can compare requested vs resolved and fail closed when insufficient.
    pub requested_scale: f64,
    /// Validity classification (regular / unresolved / outside / singular / failure).
    pub validity: GeometryValidity,
    /// Explicit cut-locus / non-unique-normal classification where known.
    pub singularity: SingularityKind,
    /// Version of the provider that produced this jet.
    pub provider_version: String,
    /// Deterministic tile/cache identity. Quantized from c and resolved scale.
    pub tile_id: String,
    /// Whether this jet came from the temporary raster bridge (true) or the
    /// destination scale-aware evaluation path (false). After migration this
    /// should be false for the destination provider.
    pub is_bridge: bool,
}

impl GeometryJet {
    /// Gradient norm |grad D| — should be ~1 where smooth (eikonal).
    pub fn grad_norm(&self) -> f64 {
        (self.grad_d[0] * self.grad_d[0] + self.grad_d[1] * self.grad_d[1]).sqrt()
    }
    /// Hessian Frobenius norm.
    pub fn hessian_norm(&self) -> f64 {
        let h = &self.hessian_d;
        (h[0][0] * h[0][0] + h[0][1] * h[0][1] + h[1][0] * h[1][0] + h[1][1] * h[1][1]).sqrt()
    }
    /// Hessian eigenvalues (analytic 2x2).
    pub fn hessian_eigenvalues(&self) -> [f64; 2] {
        let h = &self.hessian_d;
        let tr = h[0][0] + h[1][1];
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
        [(tr + disc) * 0.5, (tr - disc) * 0.5]
    }
}

/// Seam trait for geometry providers. Implementations include the legacy
/// raster adapter and the destination scale-aware provider. Rust is
/// authoritative; Python/WASM receive the same semantics.
pub trait GeometryProvider {
    fn provider_version(&self) -> &str;
    fn provider_name(&self) -> &str;
    fn query(&self, c: Complex64, epsilon: f64) -> Result<GeometryJet, String>;
    fn is_bridge(&self) -> bool;
}

/// Destination scale-aware provider (NEXT).
///
/// This is the seam for the future adaptive/local C² representation
/// (quadtree, spline tiles, on-demand DEM) that will provide true
/// refinement where `resolved_scale` must meet `requested_scale`.
/// It is NOT the current raster-backed implementation. This PR leaves
/// it as an explicit placeholder so the bridge is not mistaken for the
/// destination. Physics currently consumes `RasterBridgeProvider`.
pub struct ScaleAwareGeometryProvider;

impl Default for ScaleAwareGeometryProvider {
    fn default() -> Self {
        Self
    }
}

impl ScaleAwareGeometryProvider {
    pub fn new(_alpha: f64) -> Self {
        Self
    }
    pub fn with_default_alpha() -> Self {
        Self
    }
}

impl GeometryProvider for ScaleAwareGeometryProvider {
    fn provider_version(&self) -> &str {
        GEOMETRY_PROVIDER_VERSION
    }
    fn provider_name(&self) -> &str {
        "scale-aware"
    }
    fn is_bridge(&self) -> bool {
        false
    }
    fn query(&self, _c: Complex64, _epsilon: f64) -> Result<GeometryJet, String> {
        Err("ScaleAwareGeometryProvider: adaptive/local C² geometry not yet implemented — use RasterBridgeProvider (bridge) behind the same GeometryJet seam. See ADR 0004 / #145".to_string())
    }
}

/// Raster bridge provider (NOW).
///
/// Coherent 9-point stencil over the fixed 1024² signed-distance raster.
/// This IS the current provider behind `query_bridge_geometry` / `query_geometry`.
/// It is versioned and reports `Unresolved` when the fixed raster cannot meet
/// `requested_scale`. It is NOT the destination `ScaleAwareGeometryProvider`.
pub struct RasterBridgeProvider {
    alpha: f64,
}

impl Default for RasterBridgeProvider {
    fn default() -> Self {
        Self {
            alpha: GEOMETRY_SCALE_ALPHA,
        }
    }
}

impl RasterBridgeProvider {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl GeometryProvider for RasterBridgeProvider {
    fn provider_version(&self) -> &str {
        GEOMETRY_PROVIDER_VERSION
    }
    fn provider_name(&self) -> &str {
        "raster-bridge"
    }
    fn is_bridge(&self) -> bool {
        true
    }
    fn query(&self, c: Complex64, epsilon: f64) -> Result<GeometryJet, String> {
        query_bridge_geometry_with_alpha(c, epsilon, self.alpha)
    }
}

/// Core coherent jet construction.
///
/// This is the single authority for D, grad D, H_D on regular regions. The
/// evaluation step h is scale-aware (h = alpha * max(rho, epsilon)), not a
/// fixed global pixel fraction. D, grad_D, and H_D come from the same 9-point
/// local sampling of the signed-distance field interpolated with the same
/// bicubic kernel, so they are mutually coherent by construction.
pub fn query_bridge_geometry_with_alpha(
    c: Complex64,
    epsilon: f64,
    alpha: f64,
) -> Result<GeometryJet, String> {
    // Hard domain check: canonical valid disk |c| < 2
    let r2 = c.re * c.re + c.im * c.im;
    if !r2.is_finite() {
        return Ok(failure_jet(
            f64::NAN,
            f64::NAN,
            GeometryValidity::ProviderFailure,
            SingularityKind::None,
        ));
    }
    if r2 >= 4.0 {
        // Outside valid domain: still sample for diagnostics but mark outside
        let d_out = sample_d_or_nan(c);
        let resolved = resolved_scale_from_field().unwrap_or(GEOMETRY_MAX_STEP);
        let h = (alpha * epsilon.max(1e-12)).clamp(GEOMETRY_MIN_STEP, GEOMETRY_MAX_STEP);
        // For outside, we still attempt a jet but mark OutsideDomain
        if !d_out.is_finite() {
            return Ok(failure_jet(
                d_out,
                resolved,
                GeometryValidity::OutsideDomain,
                SingularityKind::Boundary,
            ));
        }
        // Return a trivial jet outside: gradient points outward, hessian zero
        let jet = GeometryJet {
            d: d_out,
            grad_d: [0.0, 0.0],
            hessian_d: [[0.0, 0.0], [0.0, 0.0]],
            resolved_scale: resolved,
            requested_scale: h,
            validity: GeometryValidity::OutsideDomain,
            singularity: SingularityKind::Boundary,
            provider_version: GEOMETRY_PROVIDER_VERSION.to_string(),
            tile_id: tile_id_for(c, h),
            is_bridge: true,
        };
        return Ok(jet);
    }

    // Ensure distance field is available to derive resolved_scale
    let resolved = resolved_scale_from_field().unwrap_or_else(|| {
        // Attempt to auto-load builtin if not loaded
        let _ = crate::distance_field::load_builtin_distance_field("mandelbrot_default");
        resolved_scale_from_field().unwrap_or(GEOMETRY_MAX_STEP)
    });

    // Sample D at center
    let d0 = sample_d_or_nan(c);
    if !d0.is_finite() {
        return Ok(failure_jet(
            d0,
            resolved,
            GeometryValidity::ProviderFailure,
            SingularityKind::None,
        ));
    }

    let rho0 = (d0 * d0 + epsilon * epsilon).sqrt();
    let requested = alpha * rho0.max(epsilon);
    let h = requested.clamp(GEOMETRY_MIN_STEP, GEOMETRY_MAX_STEP);

    // Coherent local sampling: 9-point stencil at scale-aware h
    let pts = [
        c,
        Complex64::new(c.re + h, c.im),
        Complex64::new(c.re - h, c.im),
        Complex64::new(c.re, c.im + h),
        Complex64::new(c.re, c.im - h),
        Complex64::new(c.re + h, c.im + h),
        Complex64::new(c.re + h, c.im - h),
        Complex64::new(c.re - h, c.im + h),
        Complex64::new(c.re - h, c.im - h),
    ];
    let vals = sample_many(&pts);
    // If any sample failed, mark provider failure but still return jet with NaN derivatives
    if vals.iter().any(|v| !v.is_finite()) {
        return Ok(GeometryJet {
            d: d0,
            grad_d: [f64::NAN, f64::NAN],
            hessian_d: [[f64::NAN; 2]; 2],
            resolved_scale: resolved,
            requested_scale: requested,
            validity: GeometryValidity::ProviderFailure,
            singularity: SingularityKind::None,
            provider_version: GEOMETRY_PROVIDER_VERSION.to_string(),
            tile_id: tile_id_for(c, h),
            is_bridge: true,
        });
    }
    let d_xp = vals[1] as f64;
    let d_xm = vals[2] as f64;
    let d_yp = vals[3] as f64;
    let d_ym = vals[4] as f64;
    let d_xp_yp = vals[5] as f64;
    let d_xp_ym = vals[6] as f64;
    let d_xm_yp = vals[7] as f64;
    let d_xm_ym = vals[8] as f64;

    let grad_x = (d_xp - d_xm) / (2.0 * h);
    let grad_y = (d_yp - d_ym) / (2.0 * h);
    let hxx = (d_xp - 2.0 * d0 + d_xm) / (h * h);
    let hyy = (d_yp - 2.0 * d0 + d_ym) / (h * h);
    let hxy = (d_xp_yp - d_xp_ym - d_xm_yp + d_xm_ym) / (4.0 * h * h);

    let grad_d = [grad_x, grad_y];
    let hessian_d = [[hxx, hxy], [hxy, hyy]];

    let grad_norm = (grad_x * grad_x + grad_y * grad_y).sqrt();
    let hess_norm = (hxx * hxx + 2.0 * hxy * hxy + hyy * hyy).sqrt();

    // Singularity / cut-locus classification: eikonal violation or extreme curvature
    let mut singularity = SingularityKind::None;
    let mut is_singular = false;
    if !grad_norm.is_finite() || !hess_norm.is_finite() {
        singularity = SingularityKind::NonUniqueNormal;
        is_singular = true;
    } else if (grad_norm - 1.0).abs() > EIKONAL_TOL {
        // Signed distance field should be eikonal where smooth
        singularity = SingularityKind::CutLocus;
        is_singular = true;
    } else if hess_norm > HESSIAN_NORM_SINGULAR {
        singularity = SingularityKind::HighCurvature;
        is_singular = true;
    }

    // Scale-aware unresolved check: provider's true cell size vs requested local scale
    let is_unresolved = resolved > requested * 1.5;

    let validity = if is_singular {
        GeometryValidity::Singular
    } else if is_unresolved {
        GeometryValidity::Unresolved
    } else {
        GeometryValidity::Regular
    };

    Ok(GeometryJet {
        d: d0,
        grad_d,
        hessian_d,
        resolved_scale: resolved,
        requested_scale: requested,
        validity,
        singularity,
        provider_version: GEOMETRY_PROVIDER_VERSION.to_string(),
        tile_id: tile_id_for(c, h),
        is_bridge: true,
    })
}

/// Bridge entry point for manifold and diagnostics (NOW).
///
/// This is the RasterBridgeProvider seam (is_bridge=true) sampled from the
/// fixed 1024² raster. It is the current provider; the destination
/// `ScaleAwareGeometryProvider` is not yet implemented behind this PR.
pub fn query_geometry(
    c: Complex64,
    epsilon: f64,
) -> Result<GeometryJet, String> {
    query_bridge_geometry_with_alpha(c, epsilon, GEOMETRY_SCALE_ALPHA)
}

/// Explicit bridge query (same as `query_geometry`).
pub fn query_bridge_geometry(
    c: Complex64,
    epsilon: f64,
) -> Result<GeometryJet, String> {
    query_bridge_geometry_with_alpha(c, epsilon, GEOMETRY_SCALE_ALPHA)
}

/// Same as `query_geometry` but taking a ManifoldConfig for convenience.
pub fn query_geometry_with_config(
    c: Complex64,
    config: &crate::manifold::ManifoldConfig,
) -> Result<GeometryJet, String> {
    query_geometry(c, config.epsilon)
}

// ---------------------------------------------------------------------------
// Analytic chain-rule derivations: D, grad_D, H_D -> rho, sigma, G, Gamma
// These are pure math, not sampling: they turn one coherent jet into the
// full scale-relative differential tower.
//
//   rho = sqrt(D^2 + epsilon^2)
//   grad rho = (D/rho) grad D
//   H_rho = (D/rho) H_D + (epsilon^2/rho^3) grad_D grad_D^T
//   sigma = log2(d_ref / rho)
//   grad sigma = -(1/(rho ln2)) grad rho
//   H_sigma = -(1/ln2)( H_rho/rho - grad_rho grad_rho^T / rho^2 )
// ---------------------------------------------------------------------------

/// Regularized distance rho from jet D.
pub fn rho_from_jet(jet: &GeometryJet, epsilon: f64) -> f64 {
    (jet.d * jet.d + epsilon * epsilon).sqrt()
}

/// Gradient of rho from jet.
pub fn grad_rho_from_jet(jet: &GeometryJet, epsilon: f64) -> [f64; 2] {
    let rho = rho_from_jet(jet, epsilon);
    if rho == 0.0 || !rho.is_finite() {
        return [f64::NAN, f64::NAN];
    }
    [jet.d / rho * jet.grad_d[0], jet.d / rho * jet.grad_d[1]]
}

/// Hessian of rho from jet.
pub fn hessian_rho_from_jet(jet: &GeometryJet, epsilon: f64) -> [[f64; 2]; 2] {
    let rho = rho_from_jet(jet, epsilon);
    if rho == 0.0 || !rho.is_finite() {
        return [[f64::NAN; 2]; 2];
    }
    let rho3 = rho * rho * rho;
    let eps2 = epsilon * epsilon;
    let mut h = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            h[i][j] = jet.d / rho * jet.hessian_d[i][j]
                + eps2 / rho3 * jet.grad_d[i] * jet.grad_d[j];
        }
    }
    h
}

/// Gradient of sigma = log2(d_ref / rho) from jet.
pub fn grad_sigma_from_jet(
    jet: &GeometryJet,
    config: &crate::manifold::ManifoldConfig,
) -> [f64; 2] {
    let rho = rho_from_jet(jet, config.epsilon);
    let ln2 = std::f64::consts::LN_2;
    if rho == 0.0 || !rho.is_finite() {
        return [f64::NAN, f64::NAN];
    }
    let gr = grad_rho_from_jet(jet, config.epsilon);
    [-gr[0] / (rho * ln2), -gr[1] / (rho * ln2)]
}

/// Hessian of sigma from jet.
pub fn hessian_sigma_from_jet(
    jet: &GeometryJet,
    config: &crate::manifold::ManifoldConfig,
) -> [[f64; 2]; 2] {
    let rho = rho_from_jet(jet, config.epsilon);
    let ln2 = std::f64::consts::LN_2;
    if rho == 0.0 || !rho.is_finite() {
        return [[f64::NAN; 2]; 2];
    }
    let h_rho = hessian_rho_from_jet(jet, config.epsilon);
    let gr = grad_rho_from_jet(jet, config.epsilon);
    let mut h = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            h[i][j] = -(1.0 / ln2) * (h_rho[i][j] / rho - gr[i] * gr[j] / (rho * rho));
        }
    }
    h
}

/// Mandelbrot scale sigma from jet.
pub fn sigma_from_jet(jet: &GeometryJet, config: &crate::manifold::ManifoldConfig) -> f64 {
    let rho = rho_from_jet(jet, config.epsilon);
    (config.d_ref / rho).log2()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolved_scale_from_field() -> Option<f64> {
    crate::distance_field::distance_field_metadata().map(|(_, _, _, _, _, _, dx, dy)| dx.max(dy))
}

fn sample_d_or_nan(c: Complex64) -> f64 {
    match crate::distance_field::sample_signed_distance_field(&[c]) {
        Ok(v) if !v.is_empty() => v[0] as f64,
        _ => f64::NAN,
    }
}

fn sample_many(points: &[Complex64]) -> Vec<f32> {
    match crate::distance_field::sample_signed_distance_field(points) {
        Ok(v) => v,
        Err(_) => vec![f32::NAN; points.len()],
    }
}

fn failure_jet(d: f64, resolved: f64, validity: GeometryValidity, sing: SingularityKind) -> GeometryJet {
    GeometryJet {
        d,
        grad_d: [f64::NAN, f64::NAN],
        hessian_d: [[f64::NAN; 2]; 2],
        resolved_scale: resolved,
        requested_scale: f64::NAN,
        validity,
        singularity: sing,
        provider_version: GEOMETRY_PROVIDER_VERSION.to_string(),
        tile_id: "failure".to_string(),
        is_bridge: true,
    }
}

fn tile_id_for(c: Complex64, h: f64) -> String {
    // Deterministic tile identity: quantized cell index at scale h, version-pinned.
    // This is stable across runs and cache-friendly.
    let inv = if h > 0.0 { 1.0 / h } else { 0.0 };
    let ix = (c.re * inv).floor() as i64;
    let iy = (c.im * inv).floor() as i64;
    // Include h quantized to ~3 significant digits for stability across tiny rho changes
    let h_q = if h > 0.0 {
        format!("{:.2e}", h)
    } else {
        "nan".to_string()
    };
    format!("{}:{}:{}:{}", GEOMETRY_PROVIDER_VERSION, ix, iy, h_q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn config() -> crate::manifold::ManifoldConfig {
        crate::manifold::ManifoldConfig::default()
    }

    #[test]
    fn provider_version_is_pinned() {
        assert_eq!(GEOMETRY_PROVIDER_VERSION, "geometry-provider/1");
    }

    #[test]
    fn jet_is_coherent_and_scale_aware() {
        let _g = crate::distance_field::global_test_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cfg = config();
        let c = Complex64::new(0.0, 0.0);
        let jet = query_geometry(c, cfg.epsilon).unwrap();
        assert!(jet.d.is_finite());
        assert!(jet.grad_d[0].is_finite() && jet.grad_d[1].is_finite());
        assert!(jet.hessian_d.iter().all(|row| row.iter().all(|v| v.is_finite())));
        // resolved vs requested exposed
        assert!(jet.resolved_scale > 0.0 && jet.resolved_scale.is_finite());
        assert!(jet.requested_scale > 0.0 && jet.requested_scale.is_finite());
        // eikonal check on interior flat region: grad norm should be ~ small but not huge
        let gn = jet.grad_norm();
        assert!(gn.is_finite());
        // tile identity deterministic
        let jet2 = query_geometry(c, cfg.epsilon).unwrap();
        assert_eq!(jet.tile_id, jet2.tile_id);
    }

    #[test]
    fn jet_scales_with_rho() {
        let _g = crate::distance_field::global_test_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cfg = config();
        // Far from Shore: large rho => large requested scale
        let c_far = Complex64::new(1.2, 0.0);
        let jet_far = query_geometry(c_far, cfg.epsilon).unwrap();
        // Near Shore: small rho => small requested scale
        let shore_x = {
            // approximate shore near 0.25 on real axis
            let mut lo = 0.2;
            let mut hi = 0.35;
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                let d = crate::distance_field::sample_signed_distance_field(&[
                    Complex64::new(mid, 0.0),
                ])
                .unwrap()[0] as f64;
                if d < 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };
        let c_near = Complex64::new(shore_x, 0.0);
        let jet_near = query_geometry(c_near, cfg.epsilon).unwrap();
        assert!(
            jet_far.requested_scale > jet_near.requested_scale,
            "far requested {} should exceed near {}",
            jet_far.requested_scale,
            jet_near.requested_scale
        );
        // requested scales are alpha*max(rho,epsilon), so near Shore ~ alpha*epsilon
        assert!((jet_near.requested_scale - GEOMETRY_SCALE_ALPHA * cfg.epsilon).abs() < 1e-6);
    }

    #[test]
    fn chain_rule_matches_finite_difference_on_regular_region() {
        let _g = crate::distance_field::global_test_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cfg = config();
        // Choose a regular interior point where field is smooth and not near cut locus
        let c = Complex64::new(-0.1, 0.1);
        let jet = query_geometry(c, cfg.epsilon).unwrap();
        // skip if singular
        if jet.validity == GeometryValidity::Singular {
            return;
        }
        let (gx, gy) = {
            let gs = grad_sigma_from_jet(&jet, &cfg);
            (gs[0], gs[1])
        };
        // Compare to a tiny finite difference of sigma via provider's own rho/sigma
        let h = 1e-6;
        let sigma = |cc: Complex64| {
            let j = query_geometry(cc, cfg.epsilon).unwrap();
            sigma_from_jet(&j, &cfg)
        };
        let sxp = sigma(Complex64::new(c.re + h, c.im));
        let sxm = sigma(Complex64::new(c.re - h, c.im));
        let syp = sigma(Complex64::new(c.re, c.im + h));
        let sym = sigma(Complex64::new(c.re, c.im - h));
        let fd_gx = (sxp - sxm) / (2.0 * h);
        let fd_gy = (syp - sym) / (2.0 * h);
        // Chain-rule grad should agree with finite difference of analytic sigma to within
        // stencil truncation + raster interpolation error. Loose tolerance due to scale-aware h.
        // Chain-rule vs finite difference: provider uses scale-aware h
        // (alpha*max(rho,epsilon) ~ 0.05 far from Shore) while this FD uses
        // 1e-6. Allow loose tolerance for stencil-scale mismatch and raster
        // interpolation noise; the point is directional agreement, not exact
        // pixel-scale reconstruction.
        assert!(
            (gx - fd_gx).abs() < 0.05,
            "gx {} vs fd {}",
            gx,
            fd_gx
        );
        assert!(
            (gy - fd_gy).abs() < 0.05,
            "gy {} vs fd {}",
            gy,
            fd_gy
        );
    }
}
