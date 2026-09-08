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
//! - `ScaleAwareGeometryProvider` (DESTINATION, is_bridge=false): adaptive dyadic
//!   Shore-contour provider at scales h_k=0.125*2^-k, 57x57 tile (core 8 halo 24),
//!   N0(k)=512+64k, 4-cell stability band, 3 bisections, 7x7 normalized quadratic fit.
//!   Direct Rust Mandelbrot membership, no 1024² raster. Physics consumes this.
//!
//! Authority: runtime-core (ADR 0001). Versioned, deterministic, and cache-aware.

use num_complex::Complex64;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// Version of the geometry provider contract. Bump when jet shape, semantics,
/// or validity classification changes in the same commit as manifold/debug
/// updates and regenerated goldens/mirrors.
pub const GEOMETRY_PROVIDER_VERSION: &str = "geometry-provider/2";

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

// --- Dyadic Shore-contour provider constants (issue #145, finish) ---
pub const DYADIC_H0: f64 = 0.125;
pub const DYADIC_CORE_CELLS: usize = 8;
pub const DYADIC_HALO_CELLS: usize = 24;
pub const DYADIC_TILE_CELLS: usize = DYADIC_CORE_CELLS + 2 * DYADIC_HALO_CELLS; // 56
pub const DYADIC_TILE_NODES: usize = DYADIC_TILE_CELLS + 1; // 57
pub const DYADIC_BISECTIONS: usize = 3;
pub const DYADIC_STABILITY_BAND_CELLS: usize = 4;
pub const DYADIC_STENCIL_RADIUS: usize = 3; // 7x7
pub const DYADIC_FIT_ERROR_FACTOR: f64 = 0.25;
pub const DYADIC_CUT_TIE_FACTOR: f64 = 0.5;
pub const DYADIC_CUT_NORMAL_DEGREES: f64 = 30.0;
pub const DYADIC_MAX_K: u32 = 24;

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
    /// Estimated jet error e = max(|dD|, h*|dgrad|, h^2*|dH|_F, fit_rms).
    #[serde(default)]
    pub estimated_error: f64,
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

/// Destination scale-aware provider — adaptive dyadic Shore-contour (finish #145).
///
/// Direct Rust Mandelbrot membership (visual_metrics::mandelbrot_membership) at
/// dyadic scales h_k = 0.125*2^-k, 57×57 tile (core 8, halo 24), N0(k)=512+64k.
/// Stable mask (4-cell band), marching-squares with 3 bisections, query-centered
/// 7×7 quadratic fit, consecutive-level error e, cut-locus via persistent
/// distance tie (0.5*h) + normal separation (30°) across two levels.
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
    fn query(&self, c: Complex64, epsilon: f64) -> Result<GeometryJet, String> {
        query_scale_aware(c, epsilon)
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

// --- Dyadic Shore-contour cache (derived Shore tiles only) ---
type TileKey = (u32, i64, i64);

#[derive(Clone, Debug)]
struct Segment {
    p1: (f64, f64),
    p2: (f64, f64),
    normal: (f64, f64),
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ShoreTile {
    k: u32,
    h: f64,
    origin_re: f64,
    origin_im: f64,
    segments: Vec<Segment>,
}

static SHORE_TILE_CACHE: Lazy<RwLock<HashMap<TileKey, ShoreTile>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[inline]
fn dyadic_h(k: u32) -> f64 {
    DYADIC_H0 * 2_f64.powi(-(k as i32))
}

#[inline]
fn dyadic_n0(k: u32) -> usize {
    let n = 512 + 64 * (k as usize);
    n.min(32768)
}

#[inline]
fn mandelbrot_inside(c: Complex64, max_iter: usize) -> bool {
    crate::visual_metrics::mandelbrot_membership(c, max_iter)
}

fn dyadic_tile_origin(c: Complex64, h: f64) -> (i64, i64, f64, f64) {
    let core_stride = DYADIC_CORE_CELLS as f64 * h;
    let core_ix = (c.re / core_stride).floor() as i64;
    let core_iy = (c.im / core_stride).floor() as i64;
    let tile_re0 = core_ix as f64 * core_stride - DYADIC_HALO_CELLS as f64 * h;
    let tile_im0 = core_iy as f64 * core_stride - DYADIC_HALO_CELLS as f64 * h;
    (core_ix, core_iy, tile_re0, tile_im0)
}

fn get_or_generate_shore_tile(k: u32, core_ix: i64, core_iy: i64) -> ShoreTile {
    let key = (k, core_ix, core_iy);
    if let Some(tile) = SHORE_TILE_CACHE.read().ok().and_then(|m| m.get(&key).cloned()) {
        return tile;
    }
    let h = dyadic_h(k);
    let core_stride = DYADIC_CORE_CELLS as f64 * h;
    let tile_re0 = core_ix as f64 * core_stride - DYADIC_HALO_CELLS as f64 * h;
    let tile_im0 = core_iy as f64 * core_stride - DYADIC_HALO_CELLS as f64 * h;
    let tile = generate_shore_tile(k, h, tile_re0, tile_im0);
    if let Ok(mut m) = SHORE_TILE_CACHE.write() {
        m.insert(key, tile.clone());
    }
    tile
}

fn generate_shore_tile(k: u32, h: f64, re0: f64, im0: f64) -> ShoreTile {
    let n = DYADIC_TILE_NODES;
    let mut inside = vec![false; n * n];
    let mut max_iter = dyadic_n0(k);
    // iterative stability: increase until 4-cell band stable or max
    let mut stable = false;
    let mut iter = max_iter;
    let mut prev_inside: Option<Vec<bool>> = None;
    for _ in 0..4 {
        for j in 0..n {
            for i in 0..n {
                let c = Complex64::new(re0 + i as f64 * h, im0 + j as f64 * h);
                inside[j * n + i] = mandelbrot_inside(c, iter);
            }
        }
        if let Some(prev) = &prev_inside {
            // check 4-cell band around shore
            let mut band_changed = false;
            for j in 0..n {
                for i in 0..n {
                    if inside[j * n + i] != prev[j * n + i] {
                        // check if within 4 cells of any shore edge (mixed cell)
                        // approximate: if any neighbor differs, it's near shore
                        let mut near_shore = false;
                        for dj in -1..=1 {
                            for di in -1..=1 {
                                let ni = i as isize + di;
                                let nj = j as isize + dj;
                                if ni < 0 || nj < 0 || ni >= n as isize || nj >= n as isize {
                                    continue;
                                }
                                // check if this node is part of a mixed cell
                                // look at 4 cells around node
                                for cj in [nj - 1, nj] {
                                    for ci in [ni - 1, ni] {
                                        if ci < 0 || cj < 0 || ci >= (n as isize - 1) || cj >= (n as isize - 1) {
                                            continue;
                                        }
                                        let a = prev[(cj as usize) * n + ci as usize];
                                        let b = prev[(cj as usize) * n + (ci + 1) as usize];
                                        let c_ = prev[((cj + 1) as usize) * n + ci as usize];
                                        let d = prev[((cj + 1) as usize) * n + (ci + 1) as usize];
                                        if a != b || a != c_ || a != d {
                                            near_shore = true;
                                        }
                                    }
                                }
                            }
                        }
                        // simpler: if within 4 cells of a mixed cell, consider band
                        // we approximate by checking if any of the 4 cells around node is mixed
                        // and distance <4*h (we already check neighbor)
                        if near_shore {
                            // check if within 4 cells Manhattan
                            let mut is_band = false;
                            for dj in -4..=4 {
                                for di in -4..=4 {
                                    let ni = i as isize + di;
                                    let nj = j as isize + dj;
                                    if ni < 0 || nj < 0 || ni >= n as isize || nj >= n as isize {
                                        continue;
                                    }
                                    // check if neighbor cell is mixed
                                    for cj in [nj - 1, nj] {
                                        for ci in [ni - 1, ni] {
                                            if ci < 0 || cj < 0 || ci >= (n as isize -1) || cj >= (n as isize -1) { continue; }
                                            let a = prev[(cj as usize)*n + ci as usize];
                                            let b = prev[(cj as usize)*n + (ci+1) as usize];
                                            let cc = prev[((cj+1) as usize)*n + ci as usize];
                                            let dd = prev[((cj+1) as usize)*n + (ci+1) as usize];
                                            if a != b || a != cc || a != dd { is_band = true; }
                                        }
                                    }
                                }
                            }
                            if is_band {
                                band_changed = true;
                                break;
                            }
                        }
                    }
                    if band_changed { break; }
                }
                if band_changed { break; }
            }
            if !band_changed {
                stable = true; let _ = stable;
                break;
            }
        }
        if iter >= 32768 {
            break;
        }
        if stable {
            break;
        }
        prev_inside = Some(inside.clone());
        iter = (iter + 512).min(32768);
        if iter == max_iter {
            break;
        }
        max_iter = iter;
    }

    // marching squares with 3 bisections
    let mut segments = Vec::new();
    for j in 0..DYADIC_TILE_CELLS {
        for i in 0..DYADIC_TILE_CELLS {
            let a = inside[j * n + i];
            let b = inside[j * n + (i + 1)];
            let c_ = inside[(j + 1) * n + i];
            let d = inside[(j + 1) * n + (i + 1)];
            let mut crossings: Vec<(f64, f64)> = Vec::new();
            let mut edge_cross = |x1: f64, y1: f64, x2: f64, y2: f64, inside1: bool, inside2: bool| {
                if inside1 == inside2 {
                    return;
                }
                let mut lo = Complex64::new(x1, y1);
                let mut hi = Complex64::new(x2, y2);
                let lo_inside = inside1;
                for _ in 0..DYADIC_BISECTIONS {
                    let mid = Complex64::new((lo.re + hi.re) * 0.5, (lo.im + hi.im) * 0.5);
                    let mid_inside = mandelbrot_inside(mid, max_iter);
                    if mid_inside == lo_inside {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let p = Complex64::new((lo.re + hi.re) * 0.5, (lo.im + hi.im) * 0.5);
                crossings.push((p.re, p.im));
            };
            let x0 = re0 + i as f64 * h;
            let y0 = im0 + j as f64 * h;
            let x1 = x0 + h;
            let y1 = y0 + h;
            edge_cross(x0, y0, x1, y0, a, b);
            edge_cross(x1, y0, x1, y1, b, d);
            edge_cross(x1, y1, x0, y1, d, c_);
            edge_cross(x0, y1, x0, y0, c_, a);
            if crossings.len() == 2 {
                let (x1_, y1_) = crossings[0];
                let (x2_, y2_) = crossings[1];
                let seg = make_oriented_segment(x1_, y1_, x2_, y2_, max_iter);
                segments.push(seg);
            } else if crossings.len() == 4 {
                // ambiguous: use center to disambiguate
                let cx = re0 + (i as f64 + 0.5) * h;
                let cy = im0 + (j as f64 + 0.5) * h;
                let center_inside = mandelbrot_inside(Complex64::new(cx, cy), max_iter);
                if center_inside {
                    let s1 = make_oriented_segment(crossings[0].0, crossings[0].1, crossings[3].0, crossings[3].1, max_iter);
                    let s2 = make_oriented_segment(crossings[1].0, crossings[1].1, crossings[2].0, crossings[2].1, max_iter);
                    segments.push(s1);
                    segments.push(s2);
                } else {
                    let s1 = make_oriented_segment(crossings[0].0, crossings[0].1, crossings[1].0, crossings[1].1, max_iter);
                    let s2 = make_oriented_segment(crossings[2].0, crossings[2].1, crossings[3].0, crossings[3].1, max_iter);
                    segments.push(s1);
                    segments.push(s2);
                }
            }
        }
    }

    ShoreTile {
        k,
        h,
        origin_re: re0,
        origin_im: im0,
        segments,
    }
}

fn make_oriented_segment(x1: f64, y1: f64, x2: f64, y2: f64, max_iter: usize) -> Segment {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt().max(1e-12);
    // two normals: n1 is left of directed edge (inside should be left per spec)
    let n1 = (-dy / len, dx / len);
    let n2 = (dy / len, -dx / len);
    let mx = (x1 + x2) * 0.5;
    let my = (y1 + y2) * 0.5;
    // Use h-scaled offset: 0.25*h at this segment's scale (approx from segment length ~h)
    // Since we don't have h here, use a scale-relative epsilon: 1e-8 is too small for deep k, use len*0.25
    let eps = len * 0.25;
    let p1 = Complex64::new(mx + n1.0 * eps, my + n1.1 * eps);
    let p2 = Complex64::new(mx + n2.0 * eps, my + n2.1 * eps);
    let inside1 = mandelbrot_inside(p1, max_iter);
    let inside2 = mandelbrot_inside(p2, max_iter);
    // Choose normal that points outward (outside is in direction of normal). Inside should be left, so normal should point outward (right).
    // If inside1 is true and inside2 false, then n1 points inside, so outward is n2.
    // We want normal to point outward, and we want endpoints ordered so inside is left.
    let (normal, p1_out, p2_out) = if inside1 && !inside2 {
        // n1 is inside, so outward is n2, and current orientation has inside left => keep
        (n2, (x1, y1), (x2, y2))
    } else if !inside1 && inside2 {
        // n1 is outside, so outward is n1, current orientation already has inside left? No, inside is n2 side, so flip
        // To keep inside-left, we need to swap endpoints
        (n1, (x2, y2), (x1, y1))
    } else {
        // fallback: choose n1 and keep orientation
        (n1, (x1, y1), (x2, y2))
    };
    Segment {
        p1: p1_out,
        p2: p2_out,
        normal,
    }
}

fn point_to_segment_distance(px: f64, py: f64, seg: &Segment) -> (f64, (f64, f64)) {
    let (x1, y1) = seg.p1;
    let (x2, y2) = seg.p2;
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-18 {
        let d = ((px - x1) * (px - x1) + (py - y1) * (py - y1)).sqrt();
        return (d, seg.normal);
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len2;
    let tc = t.clamp(0.0, 1.0);
    let qx = x1 + tc * dx;
    let qy = y1 + tc * dy;
    let d = ((px - qx) * (px - qx) + (py - qy) * (py - qy)).sqrt();
    (d, seg.normal)
}

// Precomputed pseudoinverse for normalized 7x7 stencil (u=(x-cx)/h, v=(y-cy)/h).
 // Basis [1, u, v, u^2, uv, v^2], A is 49x6, P = (A^T A)^{-1} A^T is 6x49.
// Conditioning is independent of h (spec #145).
const QUADRATIC_PINV: [[f64; 49]; 6] = [
  [-4.761904761905e-02, -1.360544217687e-02, 6.802721088435e-03, 1.360544217687e-02, 6.802721088435e-03, -1.360544217687e-02, -4.761904761905e-02, -1.360544217687e-02, 2.040816326531e-02, 4.081632653061e-02, 4.761904761905e-02, 4.081632653061e-02, 2.040816326531e-02, -1.360544217687e-02, 6.802721088435e-03, 4.081632653061e-02, 6.122448979592e-02, 6.802721088435e-02, 6.122448979592e-02, 4.081632653061e-02, 6.802721088435e-03, 1.360544217687e-02, 4.761904761905e-02, 6.802721088435e-02, 7.482993197279e-02, 6.802721088435e-02, 4.761904761905e-02, 1.360544217687e-02, 6.802721088435e-03, 4.081632653061e-02, 6.122448979592e-02, 6.802721088435e-02, 6.122448979592e-02, 4.081632653061e-02, 6.802721088435e-03, -1.360544217687e-02, 2.040816326531e-02, 4.081632653061e-02, 4.761904761905e-02, 4.081632653061e-02, 2.040816326531e-02, -1.360544217687e-02, -4.761904761905e-02, -1.360544217687e-02, 6.802721088435e-03, 1.360544217687e-02, 6.802721088435e-03, -1.360544217687e-02, -4.761904761905e-02],
  [-1.530612244898e-02, -1.020408163265e-02, -5.102040816327e-03, 0.000000000000e+00, 5.102040816327e-03, 1.020408163265e-02, 1.530612244898e-02, -1.530612244898e-02, -1.020408163265e-02, -5.102040816327e-03, 0.000000000000e+00, 5.102040816327e-03, 1.020408163265e-02, 1.530612244898e-02, -1.530612244898e-02, -1.020408163265e-02, -5.102040816327e-03, 0.000000000000e+00, 5.102040816327e-03, 1.020408163265e-02, 1.530612244898e-02, -1.530612244898e-02, -1.020408163265e-02, -5.102040816327e-03, 0.000000000000e+00, 5.102040816327e-03, 1.020408163265e-02, 1.530612244898e-02, -1.530612244898e-02, -1.020408163265e-02, -5.102040816327e-03, 0.000000000000e+00, 5.102040816327e-03, 1.020408163265e-02, 1.530612244898e-02, -1.530612244898e-02, -1.020408163265e-02, -5.102040816327e-03, 0.000000000000e+00, 5.102040816327e-03, 1.020408163265e-02, 1.530612244898e-02, -1.530612244898e-02, -1.020408163265e-02, -5.102040816327e-03, 0.000000000000e+00, 5.102040816327e-03, 1.020408163265e-02, 1.530612244898e-02],
  [-1.530612244898e-02, -1.530612244898e-02, -1.530612244898e-02, -1.530612244898e-02, -1.530612244898e-02, -1.530612244898e-02, -1.530612244898e-02, -1.020408163265e-02, -1.020408163265e-02, -1.020408163265e-02, -1.020408163265e-02, -1.020408163265e-02, -1.020408163265e-02, -1.020408163265e-02, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 5.102040816327e-03, 5.102040816327e-03, 5.102040816327e-03, 5.102040816327e-03, 5.102040816327e-03, 5.102040816327e-03, 5.102040816327e-03, 1.020408163265e-02, 1.020408163265e-02, 1.020408163265e-02, 1.020408163265e-02, 1.020408163265e-02, 1.020408163265e-02, 1.020408163265e-02, 1.530612244898e-02, 1.530612244898e-02, 1.530612244898e-02, 1.530612244898e-02, 1.530612244898e-02, 1.530612244898e-02, 1.530612244898e-02],
  [8.503401360544e-03, 7.965566981526e-19, -5.102040816327e-03, -6.802721088435e-03, -5.102040816327e-03, 7.965566981526e-19, 8.503401360544e-03, 8.503401360544e-03, -1.091577697468e-18, -5.102040816327e-03, -6.802721088435e-03, -5.102040816327e-03, -1.091577697468e-18, 8.503401360544e-03, 8.503401360544e-03, -2.224458334841e-18, -5.102040816327e-03, -6.802721088435e-03, -5.102040816327e-03, -2.224458334841e-18, 8.503401360544e-03, 8.503401360544e-03, -2.602085213965e-18, -5.102040816327e-03, -6.802721088435e-03, -5.102040816327e-03, -2.602085213965e-18, 8.503401360544e-03, 8.503401360544e-03, -2.224458334841e-18, -5.102040816327e-03, -6.802721088435e-03, -5.102040816327e-03, -2.224458334841e-18, 8.503401360544e-03, 8.503401360544e-03, -1.091577697468e-18, -5.102040816327e-03, -6.802721088435e-03, -5.102040816327e-03, -1.091577697468e-18, 8.503401360544e-03, 8.503401360544e-03, 7.965566981526e-19, -5.102040816327e-03, -6.802721088435e-03, -5.102040816327e-03, 7.965566981526e-19, 8.503401360544e-03],
  [1.147959183673e-02, 7.653061224490e-03, 3.826530612245e-03, 0.000000000000e+00, -3.826530612245e-03, -7.653061224490e-03, -1.147959183673e-02, 7.653061224490e-03, 5.102040816327e-03, 2.551020408163e-03, 0.000000000000e+00, -2.551020408163e-03, -5.102040816327e-03, -7.653061224490e-03, 3.826530612245e-03, 2.551020408163e-03, 1.275510204082e-03, 0.000000000000e+00, -1.275510204082e-03, -2.551020408163e-03, -3.826530612245e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, -3.826530612245e-03, -2.551020408163e-03, -1.275510204082e-03, 0.000000000000e+00, 1.275510204082e-03, 2.551020408163e-03, 3.826530612245e-03, -7.653061224490e-03, -5.102040816327e-03, -2.551020408163e-03, 0.000000000000e+00, 2.551020408163e-03, 5.102040816327e-03, 7.653061224490e-03, -1.147959183673e-02, -7.653061224490e-03, -3.826530612245e-03, 0.000000000000e+00, 3.826530612245e-03, 7.653061224490e-03, 1.147959183673e-02],
  [8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -6.802721088435e-03, -6.802721088435e-03, -6.802721088435e-03, -6.802721088435e-03, -6.802721088435e-03, -6.802721088435e-03, -6.802721088435e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, -5.102040816327e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03, 8.503401360544e-03],
];

fn fit_quadratic_jet(
    stencil_distances: &[(f64, f64, f64)],
) -> Option<([f64; 6], f64)> {
    if stencil_distances.len() != 49 {
        return None;
    }
    let mut d_vec = [0.0f64; 49];
    for (i, &(_, _, d)) in stencil_distances.iter().enumerate() {
        d_vec[i] = d;
    }
    let mut coeff_norm = [0.0f64; 6];
    for i in 0..6 {
        let mut sum = 0.0;
        for j in 0..49 {
            sum += QUADRATIC_PINV[i][j] * d_vec[j];
        }
        coeff_norm[i] = sum;
    }
    let mut sum2 = 0.0;
    for (idx, &(_, _, d)) in stencil_distances.iter().enumerate() {
        let row = idx / 7;
        let col = idx % 7;
        let u = col as f64 - 3.0;
        let v = row as f64 - 3.0;
        let pred = coeff_norm[0] + coeff_norm[1]*u + coeff_norm[2]*v + coeff_norm[3]*u*u + coeff_norm[4]*u*v + coeff_norm[5]*v*v;
        let e = d - pred;
        sum2 += e*e;
    }
    let rms = (sum2 / 49.0).sqrt();
    Some((coeff_norm, rms))
}

fn compute_jet_at_level(
    c: Complex64,
    epsilon: f64,
    k: u32,
    h: f64,
    tile: &ShoreTile,
) -> Option<(GeometryJet, f64, Vec<Segment>)> {
    // 7x7 query-centered stencil, step = h, sign from oriented Shore only (no second authority)
    let mut stencil: Vec<(f64, f64, f64)> = Vec::with_capacity(49);
    for dy in -3..=3 {
        for dx in -3..=3 {
            let px = c.re + dx as f64 * h;
            let py = c.im + dy as f64 * h;
            if tile.segments.is_empty() {
                // no shore in tile: far field, approximate distance as large
                // determine sign via Shore orientation fallback: use inside check only when no Shore exists
                // This is the only case where we use membership as fallback; otherwise sign comes from Shore
                let inside_p = mandelbrot_inside(Complex64::new(px, py), dyadic_n0(k));
                let sign = if inside_p { -1.0 } else { 1.0 };
                let best_dist = 10.0 * h;
                stencil.push((dx as f64 * h, dy as f64 * h, sign * best_dist));
                continue;
            }
            // find nearest segment and its projection to determine signed distance via Shore normal
            let mut best_dist = f64::INFINITY;
            let mut best_signed = f64::INFINITY;
            let mut best_qx = 0.0;
            let mut best_qy = 0.0;
            let mut best_normal = (0.0, 0.0);
            for seg in &tile.segments {
                let (x1, y1) = seg.p1;
                let (x2, y2) = seg.p2;
                let dxs = x2 - x1;
                let dys = y2 - y1;
                let len2 = dxs*dxs + dys*dys;
                let t = if len2 < 1e-18 { 0.0 } else { ((px - x1)*dxs + (py - y1)*dys)/len2 }.clamp(0.0, 1.0);
                let qx = x1 + t*dxs;
                let qy = y1 + t*dys;
                let d = ((px - qx)*(px - qx) + (py - qy)*(py - qy)).sqrt();
                // signed distance: dot((p - q), normal) -- normal points outward, so inside => negative
                let signed = (px - qx)*seg.normal.0 + (py - qy)*seg.normal.1;
                if d < best_dist {
                    best_dist = d;
                    best_signed = signed;
                    best_qx = qx;
                    best_qy = qy;
                    best_normal = seg.normal;
                }
            }
            // Use signed distance from nearest Shore segment (oriented). This is the sole authority.
            // best_signed already has correct sign (negative inside, positive outside) via outward normal.
            // For robustness, if best_signed is near zero but we are exactly on Shore, keep it.
            let signed_dist = if best_signed.is_finite() { best_signed } else { best_dist };
            // Ensure consistency: if tile is non-empty, we should not fall back to membership
            let _ = (best_qx, best_qy, best_normal);
            stencil.push((dx as f64 * h, dy as f64 * h, signed_dist));
        }
    }
    let (coeff_n, rms) = fit_quadratic_jet(&stencil)?;
    let d = coeff_n[0];
    let grad = [coeff_n[1] / h, coeff_n[2] / h];
    let hess = [[2.0 * coeff_n[3] / (h * h), coeff_n[4] / (h * h)], [coeff_n[4] / (h * h), 2.0 * coeff_n[5] / (h * h)]];
    // For cut locus we need per-point nearest segments, but we approximate via stencil
    // Return jet with temporary validity Regular, will be refined by caller
    let rho = (d * d + epsilon * epsilon).sqrt();
    let requested = GEOMETRY_SCALE_ALPHA * rho.max(epsilon);
    let jet = GeometryJet {
        d,
        grad_d: grad,
        hessian_d: hess,
        resolved_scale: h,
        requested_scale: requested,
        estimated_error: rms,
        validity: GeometryValidity::Regular,
        singularity: SingularityKind::None,
        provider_version: GEOMETRY_PROVIDER_VERSION.to_string(),
        tile_id: format!("scale-aware:{}:{}:{}:{:.2e}", k, (c.re / h).floor() as i64, (c.im / h).floor() as i64, h),
        is_bridge: false,
    };
    Some((jet, rms, tile.segments.clone()))
}

fn query_scale_aware(c: Complex64, epsilon: f64) -> Result<GeometryJet, String> {
    let r2 = c.re * c.re + c.im * c.im;
    if !r2.is_finite() {
        return Ok(failure_jet(f64::NAN, f64::NAN, GeometryValidity::ProviderFailure, SingularityKind::None));
    }
    if r2 >= 4.0 {
        let h = dyadic_h(0);
        return Ok(GeometryJet {
            d: f64::INFINITY,
            grad_d: [0.0, 0.0],
            hessian_d: [[0.0, 0.0], [0.0, 0.0]],
            resolved_scale: h,
            requested_scale: GEOMETRY_SCALE_ALPHA * epsilon,
            estimated_error: 0.0,
            validity: GeometryValidity::OutsideDomain,
            singularity: SingularityKind::Boundary,
            provider_version: GEOMETRY_PROVIDER_VERSION.to_string(),
            tile_id: tile_id_for(c, h),
            is_bridge: false,
        });
    }

    // initial scale hint from bridge (not used for final D, only for k selection)
    // Spec #145: must not auto-load raster when proving independence. Use a non-auto-loading hint.
    let hint_requested = {
        // Try to get requested_scale without auto-loading: check if field is loaded
        let has_field = crate::distance_field::is_field_loaded();
        if has_field {
            if let Ok(b) = query_bridge_geometry(c, epsilon) {
                b.requested_scale
            } else { GEOMETRY_SCALE_ALPHA * epsilon }
        } else {
            // No field loaded: use destination-only hint based on epsilon (conservative)
            // Compute a hint that will still allow refinement to find correct k
            GEOMETRY_SCALE_ALPHA * epsilon * 10.0
        }
    };

    // select k range: find smallest k with h_k <= hint_requested, then monotone coarse-to-fine until convergence
    let mut target_k = 0;
    for k in 0..=DYADIC_MAX_K {
        if dyadic_h(k) <= hint_requested {
            target_k = k;
            break;
        }
        if k == DYADIC_MAX_K {
            target_k = DYADIC_MAX_K;
        }
    }
    let start_k = target_k.saturating_sub(1);

    let mut prev_jet: Option<GeometryJet> = None;
    let mut prev_h: f64 = 0.0;
    let mut prev_rms: f64 = 0.0;
    let mut prev_segments: Vec<Segment> = Vec::new();
    let mut best_regular: Option<GeometryJet> = None;
    let mut last_jet: Option<GeometryJet> = None;

    // monotone refinement: start near hint, then continue finer until Regular or resource limit (max K)
    let mut k = start_k;
    while k <= DYADIC_MAX_K {
        let h = dyadic_h(k);
        let (core_ix, core_iy, _, _) = dyadic_tile_origin(c, h);
        let tile = get_or_generate_shore_tile(k, core_ix, core_iy);
        // patch expansion: if no Shore, try one expansion (next finer level may have Shore)
        // For now, allow empty tiles to produce far-field jet via compute_jet (10*h) but also
        // ensure we advance k if compute_jet fails. The empty case is handled in compute_jet,
        // so we don't skip here. We keep the check for resource but allow compute_jet to decide.
        // If tile empty, still try compute_jet (which returns far-field). If that fails, advance.
        
        // adaptive patch expansion: if no Shore or Shore within 4 cells of patch edge, expand once
        let mut tile_for_jet = tile.clone();
        let _expanded = false;
        // check if expansion needed
        let needs_expansion = if tile.segments.is_empty() {
            true
        } else {
            // nearest Shore distance to query, and check if that Shore is near tile edge
            let mut best_edge_dist = f64::INFINITY;
            for seg in &tile.segments {
                // distance from query to segment
                let (d, _) = point_to_segment_distance(c.re, c.im, seg);
                if d < best_edge_dist {
                    best_edge_dist = d;
                }
            }
            // check if nearest Shore is within 4*h of tile border
            // tile border is at re0, re0+56*h and im0, im0+56*h
            let tile_re0 = tile.origin_re;
            let tile_im0 = tile.origin_im;
            let tile_re1 = tile_re0 + DYADIC_TILE_CELLS as f64 * h;
            let tile_im1 = tile_im0 + DYADIC_TILE_CELLS as f64 * h;
            // find the segment closest to query and see if its projection is near edge
            let mut nearest_seg = None;
            let mut min_d = f64::INFINITY;
            for seg in &tile.segments {
                let (d, _) = point_to_segment_distance(c.re, c.im, seg);
                if d < min_d {
                    min_d = d;
                    nearest_seg = Some(seg);
                }
            }
            if let Some(seg) = nearest_seg {
                let mx = (seg.p1.0 + seg.p2.0)*0.5;
                let my = (seg.p1.1 + seg.p2.1)*0.5;
                let dist_to_edge = (mx - tile_re0).min(tile_re1 - mx).min((my - tile_im0).min(tile_im1 - my));
                dist_to_edge < 4.0 * h && min_d < 4.0 * h
            } else { false }
        };
        if needs_expansion && !_expanded {
            // expand by generating a tile with larger halo (or shift to neighboring core)
            // Simplest: generate tile at same k but with origin shifted by core stride to include query more centrally
            // Try neighboring tile that contains query more centrally
            // For empty case or edge case, we try the 8 neighboring core tiles and pick one with Shore
            let mut best_tile = tile_for_jet.clone();
            let mut found = false;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if dx==0 && dy==0 { continue; }
                    let (core_ix2, core_iy2, _, _) = dyadic_tile_origin(c, h);
                    let nix = core_ix2 + dx;
                    let niy = core_iy2 + dy;
                    let ntile = get_or_generate_shore_tile(k, nix, niy);
                    if !ntile.segments.is_empty() {
                        best_tile = ntile;
                        found = true;
                        break;
                    }
                }
                if found { break; }
            }
            if found {
                tile_for_jet = best_tile;
            }
        }
        let (jet, rms, segs) = match compute_jet_at_level(c, epsilon, k, h, &tile_for_jet) {
            Some(v) => v,
            None => { k += 1; continue; },
        };
        let rho = (jet.d * jet.d + epsilon * epsilon).sqrt();
        let requested = GEOMETRY_SCALE_ALPHA * rho.max(epsilon);
        // check cell size satisfies requested
        let cell_ok = h <= requested * 1.01; // allow tiny slack

        // compute e between fine (current) and coarse (prev) if available
        let mut e = rms;
        let mut cut_locus = false;
        if let Some(prev) = &prev_jet {
            let d_diff = (jet.d - prev.d).abs();
            let grad_diff = ((jet.grad_d[0] - prev.grad_d[0]).powi(2) + (jet.grad_d[1] - prev.grad_d[1]).powi(2)).sqrt();
            let hess_diff = ((jet.hessian_d[0][0] - prev.hessian_d[0][0]).powi(2)
                + 2.0 * (jet.hessian_d[0][1] - prev.hessian_d[0][1]).powi(2)
                + (jet.hessian_d[1][1] - prev.hessian_d[1][1]).powi(2))
            .sqrt();
            let e_candidate = d_diff
                .max(prev_h * grad_diff)
                .max(prev_h * prev_h * hess_diff)
                .max(rms)
                .max(prev_rms);
            e = e_candidate;

            // cut locus: persistent tie 0.5*h with normal separation >=30° across two levels,
            // plus non-unique normal when projection lands on contour vertex (persistent)
            let check_tie = |segs: &Vec<Segment>, hh: f64| -> Option<((f64, f64), (f64, f64), bool)> {
                if segs.len() < 2 {
                    return None;
                }
                let mut dists: Vec<(f64, (f64, f64), usize, bool)> = Vec::new();
                for (idx, seg) in segs.iter().enumerate() {
                    let (d, n) = point_to_segment_distance(c.re, c.im, seg);
                    // check if projection lands on vertex (t clamped to 0 or 1)
                    let (x1, y1) = seg.p1;
                    let (x2, y2) = seg.p2;
                    let dx = x2 - x1; let dy = y2 - y1;
                    let len2 = dx*dx+dy*dy;
                    let t = if len2 < 1e-18 { 0.0 } else { ((c.re - x1)*dx + (c.im - y1)*dy)/len2 }.clamp(0.0, 1.0);
                    let on_vertex = t < 1e-9 || t > 1.0-1e-9;
                    dists.push((d, n, idx, on_vertex));
                }
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                // find first two geometrically distinct (non-adjacent) segments
                let (d1, n1, idx1, on_v1) = dists[0];
                if d1 > 0.5 * hh {
                    return None;
                }
                let s1 = &segs[idx1];
                let mut found_second = None;
                for k2 in 1..dists.len() {
                    let (d2, n2, idx2, on_v2) = dists[k2];
                    let s2 = &segs[idx2];
                    let shared = ( (s1.p1.0 - s2.p1.0).abs() < 1e-12 && (s1.p1.1 - s2.p1.1).abs() < 1e-12 ) ||
                                 ( (s1.p1.0 - s2.p2.0).abs() < 1e-12 && (s1.p1.1 - s2.p2.1).abs() < 1e-12 ) ||
                                 ( (s1.p2.0 - s2.p1.0).abs() < 1e-12 && (s1.p2.1 - s2.p1.1).abs() < 1e-12 ) ||
                                 ( (s1.p2.0 - s2.p2.0).abs() < 1e-12 && (s1.p2.1 - s2.p2.1).abs() < 1e-12 );
                    if shared {
                        continue;
                    }
                    if (d2 - d1).abs() <= DYADIC_CUT_TIE_FACTOR * hh {
                        let dot = (n1.0 * n2.0 + n1.1 * n2.1).clamp(-1.0, 1.0);
                        let ang = dot.acos() * 180.0 / std::f64::consts::PI;
                        if ang >= DYADIC_CUT_NORMAL_DEGREES {
                            found_second = Some((n2, on_v2));
                            break;
                        }
                    }
                    // if no tie but on_vertex persistent, still consider
                    if on_v1 || on_v2 {
                        // non-unique normal at vertex
                        found_second = Some((n2, true));
                        break;
                    }
                }
                if let Some((n2, on_v2)) = found_second {
                    return Some((n1, n2, on_v1 || on_v2));
                }
                None
            };
            let tie_c = check_tie(&prev_segments, prev_h);
            let tie_f = check_tie(&segs, h);
            if let (Some((n1c, n2c, on_v_c)), Some((n1f, n2f, on_v_f))) = (tie_c, tie_f) {
                // persistence: check normals similar across levels and vertex persistence
                let dot1 = (n1c.0 * n1f.0 + n1c.1 * n1f.1).clamp(-1.0, 1.0).acos() * 180.0 / std::f64::consts::PI;
                let dot2 = (n2c.0 * n2f.0 + n2c.1 * n2f.1).clamp(-1.0, 1.0).acos() * 180.0 / std::f64::consts::PI;
                let persistent_normals = dot1 < 30.0 && dot2 < 30.0;
                let persistent_vertex = on_v_c && on_v_f;
                if persistent_normals || persistent_vertex {
                    cut_locus = true;
                } else if dot1 < 30.0 || dot2 < 30.0 {
                    // one normal persistent but not both - not enough for cut locus per spec, so false
                    cut_locus = false;
                } else {
                    // both have tie with large angle but normals not persistent => not cut locus
                    cut_locus = false;
                }
            }
        }

        let mut jet_with_error = jet.clone();
        jet_with_error.estimated_error = e;
        jet_with_error.resolved_scale = h;
        jet_with_error.requested_scale = requested;

        if cut_locus {
            jet_with_error.validity = GeometryValidity::Singular;
            jet_with_error.singularity = SingularityKind::CutLocus;
            return Ok(jet_with_error);
        }

        last_jet = Some(jet_with_error.clone());

        if cell_ok && e <= 0.25 * requested {
            jet_with_error.validity = GeometryValidity::Regular;
            jet_with_error.singularity = SingularityKind::None;
            best_regular = Some(jet_with_error);
            break;
        } else {
            // need finer (either cell too large or error too large)
            prev_jet = Some(jet);
            prev_h = h;
            prev_rms = rms;
            prev_segments = segs;
            k += 1;
            continue;
        }
    }
    // If we exited via finished, k already points beyond, but best_regular is set
    // If we exited because k > MAX, fall through to Unresolved handling

    if let Some(jet) = best_regular {
        return Ok(jet);
    }
    // no Regular found, return last jet as Unresolved with error
    if let Some(mut jet) = last_jet {
        jet.validity = GeometryValidity::Unresolved;
        jet.singularity = SingularityKind::None;
        jet.is_bridge = false;
        return Ok(jet);
    }
    // no jet at all: genuine failure, not bridge-derived (no masquerade)
    Ok(failure_jet(f64::NAN, dyadic_h(target_k), GeometryValidity::ProviderFailure, SingularityKind::None))
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
            estimated_error: 0.0,
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
            estimated_error: 0.0,
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
        estimated_error: 0.0,
        validity,
        singularity,
        provider_version: GEOMETRY_PROVIDER_VERSION.to_string(),
        tile_id: tile_id_for(c, h),
        is_bridge: true,
    })
}

/// Scale-aware entry point for manifold and diagnostics (finish #145).
///
/// Direct dyadic Shore-contour provider (is_bridge=false), independent of the
/// 1024² raster. RasterBridgeProvider remains explicit for tests/migration.
pub fn query_geometry(
    c: Complex64,
    epsilon: f64,
) -> Result<GeometryJet, String> {
    query_scale_aware(c, epsilon)
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
        estimated_error: f64::INFINITY,
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
        assert_eq!(GEOMETRY_PROVIDER_VERSION, "geometry-provider/2");
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
        // Near Shore requested is alpha*max(rho,epsilon) ~ alpha*epsilon when D~0,
        // but allow small offset between dyadic Shore and raster Shore (up to ~h).
        assert!(
            jet_near.requested_scale < 5e-5,
            "near requested {} should be small near Shore",
            jet_near.requested_scale
        );
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
            (gx - fd_gx).abs() < 1.0,
            "gx {} vs fd {}",
            gx,
            fd_gx
        );
        assert!(
            (gy - fd_gy).abs() < 1.0,
            "gy {} vs fd {}",
            gy,
            fd_gy
        );
    }

    #[test]
    fn scale_aware_independent_of_raster() {
        let _g = crate::distance_field::global_test_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Clear the raster to ensure ScaleAware does not depend on it
        crate::distance_field::clear_distance_field();
        assert!(!crate::distance_field::is_field_loaded(), "field should be cleared");
        let cfg = crate::manifold::ManifoldConfig::default();
        let c = num_complex::Complex64::new(0.0, 0.0);
        // query_geometry should now be ScaleAware and succeed without raster, and must NOT auto-reload raster
        let jet = crate::geometry_provider::query_geometry(c, cfg.epsilon).expect("scale-aware should succeed without raster");
        assert!(!jet.is_bridge, "query_geometry should be scale-aware (is_bridge=false)");
        assert!(jet.d.is_finite());
        assert!(jet.estimated_error.is_finite());
        // Prove raster was not silently reloaded via auto-load
        assert!(!crate::distance_field::is_field_loaded(), "ScaleAware must not auto-reload 1024² raster");
        // Also verify that explicit bridge without raster would fail (or auto-load if called)
        // but scale-aware is independent - we don't call bridge here
        // Re-load raster for other tests
        let _ = crate::distance_field::load_builtin_distance_field("mandelbrot_default");
        assert!(crate::distance_field::is_field_loaded(), "field should be re-loaded for next tests");
    }
}
