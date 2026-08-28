//! The minimap reader: the Player's windows onto the Map.
//!
//! Domain vocabulary (issue #88):
//! - **The Map** — the Mandelbrot set
//! - **The Shore** — the boundary of The Map
//! - **The minimaps** — the Player's 9×9 windows at c, one per selected mip level
//! - **The mip pyramid** — pre-rendered multi-scale maps (base 2048², from
//!   `scripts/bake_mandel_maps_gl.py`), stored as raw little-endian f32 planes
//! - **Slope** — the gradient vector of the grey field at a point
//!
//! The pyramid carries two fields per level:
//! - `F`: fractional escape iteration normalized to [0, 1]
//! - `S`: gradient-magnitude proximity `G/(G+G0)`, unsigned [0, 1]
//!
//! Storage is row-major with row 0 corresponding to IM_MAX (the bake script
//! flips on readback), so increasing row index means decreasing imaginary part.

use once_cell::sync::Lazy;
use std::sync::RwLock;

/// Mip levels selected for v1: even levels only, so each retained rung covers
/// 4× the world-width of the next-finer one (maximally non-redundant).
pub const MINIMAP_LEVELS: [usize; 4] = [0, 2, 4, 6];

/// Side length of a minimap window (the Player's view at c).
pub const MINIMAP_WINDOW: usize = 9;

/// Total Player observation length: 4 levels × 9×9 greys + slope per level.
pub const PLAYER_OBSERVATION_LEN: usize = MINIMAP_LEVELS.len() * MINIMAP_WINDOW * MINIMAP_WINDOW
    + 2 * MINIMAP_LEVELS.len();

/// Which field the reader samples. Both are pre-normalized to [0, 1] by the
/// bake script — no runtime normalization is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimapField {
    /// Fractional escape iteration, [0, 1].
    Escape,
    /// Gradient-magnitude proximity S = G/(G+G0), unsigned [0, 1].
    ShoreProximity,
}

impl MinimapField {
    fn slot(self) -> usize {
        match self {
            MinimapField::Escape => 0,
            MinimapField::ShoreProximity => 1,
        }
    }
}

/// A loaded mip pyramid: two fields × N levels of raw grey planes.
#[derive(Clone, Debug)]
pub struct MipPyramid {
    /// fields[slot][level] = flat row-major plane
    fields: [Vec<Vec<f32>>; 2],
    widths: Vec<usize>,
    heights: Vec<usize>,
    pub re_min: f64,
    pub re_max: f64,
    pub im_min: f64,
    pub im_max: f64,
}

impl MipPyramid {
    /// Build from explicit per-level planes. All levels must cover the same extent.
    pub fn from_levels(
        level_data: Vec<Vec<f32>>,
        widths: Vec<usize>,
        heights: Vec<usize>,
        re_min: f64,
        re_max: f64,
        im_min: f64,
        im_max: f64,
    ) -> Result<Self, String> {
        if level_data.is_empty() {
            return Err("pyramid needs at least one level".into());
        }
        if widths.len() != level_data.len() || heights.len() != level_data.len() {
            return Err("widths/heights must match level count".into());
        }
        for (i, data) in level_data.iter().enumerate() {
            let expect = widths[i].saturating_mul(heights[i]);
            if data.len() != expect {
                return Err(format!(
                    "level {} has {} values, expected {}",
                    i,
                    data.len(),
                    expect
                ));
            }
        }
        // Two field slots: Escape and ShoreProximity share the same planes
        // when built via this constructor (single-field pyramids).
        Ok(Self {
            fields: [level_data.clone(), level_data],
            widths,
            heights,
            re_min,
            re_max,
            im_min,
            im_max,
        })
    }

    /// Number of levels in the pyramid.
    pub fn num_levels(&self) -> usize {
        self.widths.len()
    }

    /// Replace the Escape-field planes (used by bindings that load F and S
    /// from separate buffers).
    pub fn set_escape_field(&mut self, levels: Vec<Vec<f32>>) {
        if levels.len() == self.widths.len() {
            self.fields[0] = levels;
        }
    }

    /// (width, height) of a level, or None if out of range.
    pub fn level_size(&self, level: usize) -> Option<(usize, usize)> {
        self.widths
            .get(level)
            .zip(self.heights.get(level))
            .map(|(&w, &h)| (w, h))
    }

    fn sample_level(&self, field: MinimapField, level: usize, col: isize, row: isize) -> f32 {
        let (w, h) = match self.level_size(level) {
            Some(s) => s,
            None => return 0.0,
        };
        let clamped_col = col.clamp(0, w as isize - 1);
        let clamped_row = row.clamp(0, h as isize - 1);
        let idx = clamped_row as usize * w + clamped_col as usize;
        self.fields[field.slot()][level]
            .get(idx)
            .copied()
            .unwrap_or(0.0)
    }

    /// World-space texel spacing for a level.
    fn texel_spacing(&self, level: usize) -> Option<(f64, f64)> {
        let (w, h) = self.level_size(level)?;
        let d_re = (self.re_max - self.re_min) / w.max(2) as f64;
        let d_im = (self.im_max - self.im_min) / h.max(2) as f64;
        Some((d_re, d_im))
    }

    /// Convert world coordinates to fractional texel coordinates on a level.
    ///
    /// Column increases with Re; row increases as Im decreases (row 0 = IM_MAX).
    fn world_to_texel(&self, level: usize, c: num_complex::Complex64) -> Option<(f64, f64)> {
        let (w, h) = self.level_size(level)?;
        let fx = (c.re - self.re_min) / (self.re_max - self.re_min) * w as f64 - 0.5;
        let fy = (self.im_max - c.im) / (self.im_max - self.im_min) * h as f64 - 0.5;
        Some((fx, fy.min(h as f64 - 0.5)))
    }

    /// Convert world coordinates to fractional texel coordinates on a level.
    ///
    /// Column increases with Re; row increases as Im decreases (row 0 = IM_MAX).
    /// Public so bindings can implement batch sampling without per-point calls.
    pub fn world_to_texel_pub(
        &self,
        level: usize,
        c: num_complex::Complex64,
    ) -> Option<(f64, f64)> {
        self.world_to_texel(level, c)
    }

    /// Sample the shore-proximity (S) field at integer texel coordinates.
    /// Public for the same reason as [`MipPyramid::world_to_texel_pub`].
    pub fn sample_field_pub(&self, level: usize, col: isize, row: isize) -> f32 {
        self.sample_level(MinimapField::ShoreProximity, level, col, row)
    }

    /// Shore proximity (S field value) at c on the given level: the Player's
    /// current distance-from-the-Shore reading, in [0, 1].
    pub fn shore_proximity_at(
        &self,
        c: num_complex::Complex64,
        level: usize,
    ) -> Option<f32> {
        let (fx, fy) = self.world_to_texel(level, c)?;
        let cx = fx.round() as isize;
        let cy = fy.round() as isize;
        Some(self.sample_level(MinimapField::ShoreProximity, level, cx, cy))
    }

    /// Extract the Player's minimap: a `half*2+1`-square window of greys
    /// centered on c at the given level, clamped at the extent edges.
    ///
    /// Returns row-major values with row 0 at higher Im than the center row.
    pub fn minimap(
        &self,
        c: num_complex::Complex64,
        level: usize,
        half: usize,
    ) -> Option<Vec<f32>> {
        let (fx, fy) = self.world_to_texel(level, c)?;
        let cx = fx.round() as isize;
        let cy = fy.round() as isize;
        let mut out = Vec::with_capacity((2 * half + 1).pow(2));
        for dr in -(half as isize)..=(half as isize) {
            for dc in -(half as isize)..=(half as isize) {
                out.push(self.sample_level(MinimapField::ShoreProximity, level, cx + dc, cy + dr));
            }
        }
        Some(out)
    }

    /// Slope (gradient vector) of the chosen field at c on the given level,
    /// computed with central differences in world units.
    pub fn slope(
        &self,
        c: num_complex::Complex64,
        level: usize,
    ) -> Option<(f64, f64)> {
        self.slope_field(c, level, MinimapField::ShoreProximity)
    }

    /// Slope of an explicit field at c on the given level.
    pub fn slope_field(
        &self,
        c: num_complex::Complex64,
        level: usize,
        field: MinimapField,
    ) -> Option<(f64, f64)> {
        let (fx, fy) = self.world_to_texel(level, c)?;
        let cx = fx.round() as isize;
        let cy = fy.round() as isize;
        let (d_re, d_im) = self.texel_spacing(level)?;

        let f_xp = self.sample_level(field, level, cx + 1, cy) as f64;
        let f_xm = self.sample_level(field, level, cx - 1, cy) as f64;
        let f_yp = self.sample_level(field, level, cx, cy - 1) as f64;
        let f_ym = self.sample_level(field, level, cx, cy + 1) as f64;

        // Row index grows as Im decreases, so dF/dIm flips sign relative to rows.
        let gx = (f_xp - f_xm) / (2.0 * d_re);
        let gy = (f_yp - f_ym) / (2.0 * d_im);
        Some((gx, gy))
    }

    /// Full Player observation at c: for each selected level (in order),
    /// the 9×9 minimap followed by nothing; then all slopes appended after
    /// the greys. Layout:
    /// `[greys(level0) .. greys(levelN), (gx,gy)(level0) .. (gx,gy)(levelN)]`
    pub fn player_observation(&self, c: num_complex::Complex64) -> Option<Vec<f32>> {
        let mut obs = Vec::with_capacity(PLAYER_OBSERVATION_LEN);
        let half = MINIMAP_WINDOW / 2;
        for &level in MINIMAP_LEVELS.iter() {
            if level >= self.num_levels() {
                return None;
            }
            obs.extend(self.minimap(c, level, half)?);
        }
        for &level in MINIMAP_LEVELS.iter() {
            let (gx, gy) = self.slope(c, level)?;
            obs.push(gx as f32);
            obs.push(gy as f32);
        }
        Some(obs)
    }
}

static PYRAMID: Lazy<RwLock<Option<MipPyramid>>> = Lazy::new(|| RwLock::new(None));

/// Contour-biased integrator step for Physics (issue #88, Q2).
///
/// The Map's boundary is a WALL in a third "fractal" dimension: J(c) grows
/// infinitely volatile as c approaches the Shore, so the same world-space
/// velocity produces enormous visual change there. Physics models this as a
/// curved surface that redirects c:
///
/// - **Wall (Skyrim clip)**: crossing the Shore (entering/exiting the set)
///   is possible but must be "clipped through" — normal (cross-boundary)
///   motion is heavily damped between transients and passes freely during
///   them (`h` near 1). Not a hard wall: spontaneous crossings happen, but
///   transients make them systematically easier.
/// - **Fractal tilt**: near the Shore (high |∇S|) the surface steepens —
///   normal motion is additionally damped by the local gradient magnitude,
///   so c appears to slow in the complex plane while its momentum is
///   "retained" in the fractal dimension (as visual change, not travel).
/// - **Music push (uphill)**: audio energy pushes c UP the slope, along
///   the shore normal toward the boundary — like a rider pumping a ramp.
///   There is no fixed target height: the resting height emerges from the
///   balance between this push and the controller's gravity (the valley
///   at the origin). Loud audio holds c near the ridgeline; quiet audio
///   lets gravity pull it back down into the valley.
///
/// When no pyramid is loaded, falls back to plain clamped motion (the
/// proposed delta clamped to max_step).
pub fn contour_biased_step(
    c_re: f64,
    c_im: f64,
    u_re: f64,
    u_im: f64,
    h: f64,
    d_star: f64,
    max_step: f64,
    level: usize,
    energy: f64,
) -> Result<(f64, f64), String> {
    let u_mag = (u_re * u_re + u_im * u_im).sqrt();
    let energy = energy.clamp(0.0, 1.0);

    with_pyramid(|pyr| {
        let pyr = match pyr {
            Some(p) => p,
            None => {
                // No map available: plain clamped motion.
                let scale = if u_mag > max_step { max_step / u_mag } else { 1.0 };
                return Ok((c_re + u_re * scale, c_im + u_im * scale));
            }
        };

        let c = num_complex::Complex64::new(c_re, c_im);
        let (gx, gy) = pyr
            .slope(c, level)
            .ok_or_else(|| "level out of range".to_string())?;
        let grad_norm = (gx * gx + gy * gy).sqrt();

        // Gradient too small to define a contour: the map is FLAT here
        // (deep interior/exterior, where F and S are constant). The minimap
        // gives no direction, but the analytic cardioid proximity
        // p = ||mu|-1| (mu = 1 - sqrt(1-4c)) still points at the Shore.
        // Run the energy servo along that analytic normal so loud audio
        // draws c toward the boundary even in featureless regions.
        if grad_norm <= 1e-12 {
            let (fr, fi) = cardioid_fallback_step(
                c_re, c_im, u_re, u_im, h, d_star, max_step, energy,
            )?;
            return Ok((fr, fi));
        }

        // Normal points toward increasing proximity (away from the Shore);
        // tangent runs along the contour.
        let nx = gx / grad_norm;
        let ny = gy / grad_norm;
        let tx = -gy / grad_norm;
        let ty = gx / grad_norm;

        let proj_t = u_re * tx + u_im * ty;
        let proj_n = u_re * nx + u_im * ny;

        // ---- The Wall (Skyrim clip) ----
        // Between transients the boundary is nearly solid (2% normal
        // bleed — slow drift across is still possible, so crossings
        // happen spontaneously); during transients the clip is easy
        // (full normal motion). This is the physics-level guarantee the
        // user asked for: crossing difficulty is integral to the engine,
        // not delegated to the model.
        let wall_no_hit = 0.02_f64;
        let wall_hit = 1.0_f64;
        let h_c = h.clamp(0.0, 1.0);
        let wall_scale = wall_no_hit + (wall_hit - wall_no_hit) * h_c;

        // ---- Fractal tilt ----
        // Near the Shore the "surface" steepens: damp normal motion by the
        // local gradient magnitude so c visually slows as J(c) volatility
        // rises (momentum is retained in the fractal dimension). The
        // damping saturates so deep-interior/exterior flat regions are
        // unaffected. During transients the surface FLATTENS (scaled by
        // 1-h): clipping through the wall means the tilt relaxes too.
        let tilt = 1.0 / (1.0 + FRACTAL_TILT_GAIN * grad_norm * (1.0 - h_c));

        // ---- Music push (uphill) ----
        // Audio energy pushes c UP the slope along the shore normal
        // (toward the boundary). This is a FORCE, not a target: the
        // resting height comes from the balance against the controller's
        // gravity. Loud audio → strong push → c rides the ridgeline;
        // silence → no push → gravity drains c back into the valley.
        let push = MUSIC_PUSH_GAIN * energy;

        let mut dx = tx * proj_t + nx * (proj_n * wall_scale * tilt + push);
        let mut dy = ty * proj_t + ny * (proj_n * wall_scale * tilt + push);

        let mag = (dx * dx + dy * dy).sqrt();
        if mag > max_step && mag > 0.0 {
            let s = max_step / mag;
            dx *= s;
            dy *= s;
        }

        Ok((c_re + dx, c_im + dy))
    })
}

/// How strongly the local gradient magnitude (J(c) volatility) damps
/// normal (cross-Shore) motion. Higher = c slows more near the boundary.
pub const FRACTAL_TILT_GAIN: f64 = 2.5;

/// How hard audio energy pushes c UP the slope (toward the Shore), per
/// frame at energy = 1. This is a force, not a target height: the resting
/// proximity emerges from the balance against the controller's gravity
/// (GRAVITY_ACCEL). Rough equilibrium: push ≈ gravity·|c| ⇒ |c| ≈ 0.55/0.01
/// would be far outside the map, so in practice the wall + tilt cap the
/// climb and loud audio pins c against the ridgeline.
pub const MUSIC_PUSH_GAIN: f64 = 0.55;

/// Servo gain for the analytic-cardioid fallback (flat-map regions).
pub const CARDIOID_FALLBACK_SERVO_GAIN: f64 = 0.05;

/// Step for flat-map regions (minimap gradient ~ 0): plain clamped motion
/// plus an energy servo along the ANALYTIC cardioid normal.
///
/// The cardioid proximity p(c) = ||mu| - 1| with mu = 1 - sqrt(1-4c) is
/// smooth everywhere off the real-axis branch cut and its gradient points
/// toward the boundary. When energy raises the effective target above the
/// current proximity, c is pulled shoreward even where the baked map is
/// featureless — this is what lets c escape the cardioid interior.
fn cardioid_fallback_step(
    c_re: f64,
    c_im: f64,
    u_re: f64,
    u_im: f64,
    _h: f64,
    _d_star: f64,
    max_step: f64,
    energy: f64,
) -> Result<(f64, f64), String> {
    // Analytic gradient of p(c) = ||mu| - 1|, mu = 1 - sqrt(1-4c).
    // Computed with central differences on the analytic formula (cheap,
    // exact enough for a direction).
    let p = |re: f64, im: f64| -> f64 {
        let cc = num_complex::Complex64::new(re, im);
        let inner = (1.0 - 4.0 * cc).sqrt();
        let mu = num_complex::Complex64::new(1.0, 0.0) - inner;
        (mu.norm() - 1.0).abs()
    };
    let eps = 1e-4_f64;
    let gx = (p(c_re + eps, c_im) - p(c_re - eps, c_im)) / (2.0 * eps);
    let gy = (p(c_re, c_im + eps) - p(c_re, c_im - eps)) / (2.0 * eps);
    let gnorm = (gx * gx + gy * gy).sqrt();

    let u_mag = (u_re * u_re + u_im * u_im).sqrt();
    let scale = if u_mag > max_step { max_step / u_mag } else { 1.0 };
    let mut dx = u_re * scale;
    let mut dy = u_im * scale;

    if gnorm > 1e-12 {
        // Uphill push along the analytic normal, proportional to energy.
        // Same force model as the pyramid path: no target height, the
        // resting height emerges from the balance against gravity.
        // NOTE THE SIGN: p(c) = ||mu|-1| DECREASES toward the boundary, so
        // the boundary direction is -∇p. The gradient points away from the
        // Shore; we must DESCEND it. (Getting this wrong shoves c into the
        // valley — the exact opposite of the music push.)
        let push = MUSIC_PUSH_GAIN * energy;
        dx -= gx / gnorm * push;
        dy -= gy / gnorm * push;
        let mag = (dx * dx + dy * dy).sqrt();
        if mag > max_step && mag > 0.0 {
            let s = max_step / mag;
            dx *= s;
            dy *= s;
        }
    }

    Ok((c_re + dx, c_im + dy))
}

/// Install the process-wide pyramid (used by bindings and tests).
pub fn set_pyramid(pyr: MipPyramid) -> Result<(), String> {
    let mut guard = PYRAMID.write().map_err(|e| format!("lock error: {}", e))?;
    *guard = Some(pyr);
    Ok(())
}

/// Clear the process-wide pyramid (test helper).
pub fn clear_pyramid() {
    if let Ok(mut g) = PYRAMID.write() {
        *g = None;
    }
}

/// Access the process-wide pyramid.
pub fn with_pyramid<T>(f: impl FnOnce(Option<&MipPyramid>) -> T) -> T {
    let guard = PYRAMID.read();
    match guard {
        Ok(g) => f(g.as_ref()),
        Err(_) => f(None),
    }
}

/// Load a mip pyramid from the baked binary artifacts produced by
/// `scripts/bake_mandel_maps_gl.py`.
///
/// Each `.bin` file is a concatenation of little-endian f32 row-major planes
/// (one per mip level); the companion JSON metadata carries the per-level
/// widths/heights/offsets and the world extent. Both fields (F = escape
/// iteration, S = shore proximity) are loaded from their respective files.
pub fn load_pyramid_from_files(
    f_bin_path: &str,
    s_bin_path: &str,
    meta_path: &str,
) -> Result<MipPyramid, String> {
    use std::fs;

    let meta_str = fs::read_to_string(meta_path)
        .map_err(|e| format!("read {}: {}", meta_path, e))?;
    let meta: serde_json::Value =
        serde_json::from_str(&meta_str).map_err(|e| format!("meta parse: {}", e))?;

    let re_min = meta["re_min"].as_f64().ok_or("meta missing re_min")?;
    let re_max = meta["re_max"].as_f64().ok_or("meta missing re_max")?;
    let im_min = meta["im_min"].as_f64().ok_or("meta missing im_min")?;
    let im_max = meta["im_max"].as_f64().ok_or("meta missing im_max")?;

    let read_field = |bin_path: &str, key: &str| -> Result<Vec<Vec<f32>>, String> {
        let m = &meta[key];
        let widths: Vec<usize> = m["mip_widths"]
            .as_array()
            .ok_or(format!("meta missing {}.mip_widths", key))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let heights: Vec<usize> = m["mip_heights"]
            .as_array()
            .ok_or(format!("meta missing {}.mip_heights", key))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let offsets: Vec<usize> = m["mip_offsets_bytes"]
            .as_array()
            .ok_or(format!("meta missing {}.mip_offsets_bytes", key))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let bytes = fs::read(bin_path).map_err(|e| format!("read {}: {}", bin_path, e))?;
        let mut levels = Vec::with_capacity(widths.len());
        for i in 0..widths.len() {
            let start = offsets[i];
            let n = widths[i].saturating_mul(heights[i]);
            let end = start + n * 4;
            if end > bytes.len() {
                return Err(format!(
                    "{} truncated at level {} (need {} bytes, have {})",
                    bin_path,
                    i,
                    end,
                    bytes.len()
                ));
            }
            let mut plane = Vec::with_capacity(n);
            for chunk in bytes[start..end].chunks_exact(4) {
                plane.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            levels.push(plane);
        }
        Ok(levels)
    };

    let f_levels = read_field(f_bin_path, "F")?;
    let s_levels = read_field(s_bin_path, "S")?;

    // Build with both fields properly separated: temporarily construct via
    // from_levels for validation, then swap in the distinct field planes.
    let mut pyr = MipPyramid::from_levels(
        s_levels.clone(),
        meta["S"]["mip_widths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect(),
        meta["S"]["mip_heights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect(),
        re_min,
        re_max,
        im_min,
        im_max,
    )?;
    pyr.fields = [f_levels, s_levels];
    Ok(pyr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_lengths() {
        let err = MipPyramid::from_levels(vec![vec![0.0; 3]], vec![2], vec![2], -2.0, 1.0, -1.5, 1.5);
        assert!(err.is_err());
    }
}
