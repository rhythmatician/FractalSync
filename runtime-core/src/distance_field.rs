use ndarray::Array2;
use once_cell::sync::Lazy;
use std::path::Path;
use std::sync::{Mutex, OnceLock, RwLock};

static GLOBAL_TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
/// Global mutex to serialize tests that mutate or depend on the distance field.
/// Controls and distance-field integration tests acquire this for the duration
/// of their sensitive sections so a parallel `clear_distance_field` cannot
/// interleave between two reads that must see the same field (e.g. Q and G).
pub fn global_test_mutex() -> &'static Mutex<()> {
    GLOBAL_TEST_MUTEX.get_or_init(|| Mutex::new(()))
}

#[derive(Clone, Debug)]
struct DistanceField {
    data: Array2<f32>,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
}

static DIST_FIELD: Lazy<RwLock<Option<DistanceField>>> = Lazy::new(|| RwLock::new(None));

/// Clear the in-memory distance field (test helper)
///
/// Public so integration tests can reset state; callers outside tests should
/// avoid calling this in production code.
pub fn clear_distance_field() {
    let _m = global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
    if let Ok(mut g) = DIST_FIELD.write() {
        *g = None;
    }
}

pub fn load_distance_field<P: AsRef<Path>>(_path: P) -> Result<(), String> {
    Err("loading .npy from Rust is not implemented in this build; use `set_distance_field_from_vec` (Python) or `load_builtin_distance_field` instead".into())
}

pub fn set_distance_field_from_vec(data: Vec<f32>, rows: usize, cols: usize, xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Result<(), String> {
    let _m = global_test_mutex().lock().unwrap_or_else(|e| e.into_inner());
    if data.len() != rows.saturating_mul(cols) {
        return Err("data length does not match rows*cols".into());
    }
    let arr = Array2::from_shape_vec((rows, cols), data).map_err(|e| format!("reshape: {}", e))?;
    let df = DistanceField { data: arr, xmin, xmax, ymin, ymax };
    let mut guard = DIST_FIELD.write().map_err(|e| format!("lock error: {}", e))?;
    *guard = Some(df);
    Ok(())
}

/// Load a built-in distance field embedded at compile time or fall back to external file.
///
/// # Performance Note
/// The embedded distance field (~4MB for the 1024x1024 resolution) is included directly
/// in the compiled binary using `include_bytes!`. This can significantly increase the size
/// of the runtime-core binary and WASM bundle, potentially impacting load times in browser
/// environments. Consider:
/// - Using a Cargo feature flag to conditionally disable the embedded field for WASM builds
/// - Providing a smaller default field for WASM (e.g., 256x256) and keeping the large one
///   for native builds
/// - Loading the field dynamically at runtime instead of embedding it at compile time
pub fn load_builtin_distance_field(name: &str) -> Result<(usize, usize, f64, f64, f64, f64), String> {
    match name {
        "mandelbrot_1024" | "mandelbrot_default" | "default" => {
            // Use raw embedded binary (.bin) with little-endian float32 values for fast compile-time embedding.
            let bin_bytes: &[u8] = include_bytes!("../data/mandelbrot_distance_1024.bin");
            let json_str: &str = include_str!("../data/mandelbrot_distance_1024.json");

            if bin_bytes.len() % 4 != 0 {
                return Err("embedded bin size is not a multiple of 4".into());
            }
            let mut flat: Vec<f32> = Vec::with_capacity(bin_bytes.len() / 4);
            for chunk in bin_bytes.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                flat.push(v);
            }

            // Parse metadata JSON
            let meta: serde_json::Value = serde_json::from_str(json_str).map_err(|e| format!("meta parse error: {}", e))?;
            let res = meta.get("res").and_then(|v| v.as_u64()).ok_or_else(|| "meta missing res".to_string())? as usize;
            let rows = res;
            let cols = res;
            let xmin = meta["xmin"].as_f64().unwrap_or(-2.5);
            let xmax = meta["xmax"].as_f64().unwrap_or(1.5);
            let ymin = meta["ymin"].as_f64().unwrap_or(-2.0);
            let ymax = meta["ymax"].as_f64().unwrap_or(2.0);

            // Set into in-memory field. If this fails return a descriptive error.
            set_distance_field_from_vec(flat, rows, cols, xmin, xmax, ymin, ymax)
                .map_err(|e| format!("failed to set builtin distance field: {}", e))?;
            Ok((rows, cols, xmin, xmax, ymin, ymax))
        }
        other => Err(format!("unknown builtin distance field: {}", other)),
    }
}

/// Metadata describing the currently loaded distance field's spatial extent and
/// resolution (pixel spacing).
///
/// This is the provider's authoritative resolution: the finite-difference step
/// used for derivatives of the sampled field must be chosen relative to this
/// spacing, not a magic constant. Returns `None` if no field is loaded.
///
/// The returned tuple is `(rows, cols, xmin, xmax, ymin, ymax, dx, dy)` where
/// `dx = (xmax - xmin) / (cols - 1)` and `dy = (ymax - ymin) / (rows - 1)` are
/// the per-pixel spacings in world coordinates.
pub fn distance_field_metadata() -> Option<(usize, usize, f64, f64, f64, f64, f64, f64)> {
    let guard = DIST_FIELD.read().ok()?;
    let df = guard.as_ref()?;
    let (rows, cols) = (df.data.nrows(), df.data.ncols());
    let dx = if cols > 1 {
        (df.xmax - df.xmin) / (cols as f64 - 1.0)
    } else {
        0.0
    };
    let dy = if rows > 1 {
        (df.ymax - df.ymin) / (rows as f64 - 1.0)
    } else {
        0.0
    };
    Some((rows, cols, df.xmin, df.xmax, df.ymin, df.ymax, dx, dy))
}

/// Sample the in-memory distance field at the given complex-valued coordinates,
/// returning **signed** distances to the Mandelbrot boundary.
///
/// This is the single authority for signed realm classification. The underlying
/// field stores signed values (positive outside M, negative inside M); this
/// sampler interpolates those signed values directly and does **not** reconstruct
/// the sign with a separate escape-iteration heuristic. Callers that need to know
/// which realm a point is in must use this function (or :func:`sample_distance_field`
/// plus an independent realm source), never a second distance authority.
///
/// - For points within the field's bounding box, bicubic interpolation (for
///   fields >= 4x4) or bilinear interpolation (for smaller fields) is used at
///   exactly the requested coordinate `c`. The returned value is the signed
///   distance to the boundary at `c` — NOT a neighborhood argmin that picks
///   a nearby sample. This is essential: the manifold kernel needs a smooth
///   scalar field whose derivatives mean something, and a local min-abs
///   selection over 17 neighboring samples is inherently piecewise/non-smooth.
/// - For points outside the bounding box, the returned distance is the signed value
///   at the nearest edge plus the Euclidean distance from the point to that edge.
///   The field's bounding box encloses the Mandelbrot set, so edge values are
///   positive (outside M) and the result is positive outside the box.
///
/// If the distance field is not loaded, this function will attempt to auto-load the
/// built-in "mandelbrot_default" field.
///
/// # Arguments
/// * `points` - complex coordinates in the plane
///
/// # Returns
/// A vector of signed distances (positive outside M, negative inside M).
pub fn sample_signed_distance_field(
    points: &[num_complex::Complex64],
) -> Result<Vec<f32>, String> {
    // If no distance field is loaded, try loading the canonical builtin
    // so callers (like tests) can sample without an explicit prior set.
    let guard = DIST_FIELD.read().map_err(|e| format!("lock error: {}", e))?;
    if guard.is_none() {
        drop(guard);
        // Best-effort: try to load the canonical built-in. If that fails still
        // return the original 'not loaded' error to the caller.
        match load_builtin_distance_field("mandelbrot_default") {
            Ok(_) => {}
            Err(e) => return Err(format!("could not auto-load builtin: {}", e)),
        }
    }
    let guard = DIST_FIELD.read().map_err(|e| format!("lock error: {}", e))?;
    let df = guard.as_ref().ok_or_else(|| "distance field not loaded".to_string())?;

    let (h, w) = (df.data.nrows() as f64, df.data.ncols() as f64);
    let dx = (df.xmax - df.xmin) as f64;
    let dy = (df.ymax - df.ymin) as f64;

    let mut out = Vec::with_capacity(points.len());

    // helper to evaluate bicubic interpolation at arbitrary (fx,fy) pixel coords
    fn eval_bicubic_at(df: &DistanceField, fx: f64, fy: f64, h: f64, w: f64) -> f32 {
        let x0 = fx.floor() as isize;
        let y0 = fy.floor() as isize;
        let sx = fx - x0 as f64;
        let sy = fy - y0 as f64;
        fn cubic_kernel(x: f64) -> f64 {
            let ax = x.abs();
            if ax <= 1.0 { 1.5 * ax * ax * ax - 2.5 * ax * ax + 1.0 }
            else if ax < 2.0 { -0.5 * ax * ax * ax + 2.5 * ax * ax - 4.0 * ax + 2.0 }
            else { 0.0 }
        }
        let mut sum = 0.0f64;
        let mut wsum = 0.0f64;
        for j in -1isize..=2isize {
            let wy = cubic_kernel((j as f64) - sy);
            let y_idx = (y0 + j).clamp(0, (h - 1.0) as isize) as usize;
            for i in -1isize..=2isize {
                let wx = cubic_kernel((i as f64) - sx);
                let x_idx = (x0 + i).clamp(0, (w - 1.0) as isize) as usize;
                let val = df.data[[y_idx, x_idx]] as f64;
                let weight = wx * wy;
                sum += val * weight;
                wsum += weight;
            }
        }
        if wsum.abs() > 0.0 {
            (sum / wsum) as f32
        } else { 0.0 }
    }

    for point in points {
        let xr = point.re;
        let yr = point.im;
        // normalized [0,1]
        let mut u = (xr - df.xmin) / dx;
        let mut v = (yr - df.ymin) / dy;
        // compute outside distance in real coordinates before clamping
        let extra_x = if xr < df.xmin { df.xmin - xr } else if xr > df.xmax { xr - df.xmax } else { 0.0 };
        let extra_y = if yr < df.ymin { df.ymin - yr } else if yr > df.ymax { yr - df.ymax } else { 0.0 };
        let outside_dist = (extra_x * extra_x + extra_y * extra_y).sqrt();

        u = u.clamp(0.0, 1.0);
        v = v.clamp(0.0, 1.0);
        // pixel coordinates
        let fx = u * (w - 1.0);
        let fy = v * (h - 1.0);
        let x0 = fx.floor() as isize;
        let y0 = fy.floor() as isize;

        // If field is small (<4) preserve the original bilinear behavior used in tests.
        if h < 4.0 || w < 4.0 {
            let sx = (fx - x0 as f64) as f32;
            let sy = (fy - y0 as f64) as f32;
            let v00 = df.data[[y0 as usize, x0 as usize]];
            let x1 = (x0 + 1).min((w - 1.0) as isize) as isize;
            let y1 = (y0 + 1).min((h - 1.0) as isize) as isize;
            let v10 = df.data[[y0 as usize, x1 as usize]];
            let v01 = df.data[[y1 as usize, x0 as usize]];
            let v11 = df.data[[y1 as usize, x1 as usize]];
            let a = v00 * (1.0 - sx) + v10 * sx;
            let b = v01 * (1.0 - sx) + v11 * sx;
            let s = a * (1.0 - sy) + b * sy;
            // Signed value at the edge plus the outside distance. The box encloses
            // M, so edge values are positive and the result stays positive outside.
            let s = s + outside_dist as f32;
            out.push(s);
        } else {
            // Bicubic interpolation evaluated at exactly c. The returned value is
            // the signed distance to the boundary at c — no min-abs selection over
            // neighboring subpixel offsets. A local argmin over 17 neighboring
            // samples is inherently piecewise/non-smooth and breaks the manifold
            // kernel's derivative semantics (the Shore oscillation in sigma was
            // caused by the winning offset switching as c moved). Outside the
            // box the edge value is positive so adding outside_dist keeps it
            // positive.
            let s = eval_bicubic_at(df, fx, fy, h, w) + outside_dist as f32;
            out.push(s);
        }
    }
    Ok(out)
}

/// Sample the in-memory distance field at the given complex-valued coordinates,
/// returning **unsigned** (absolute) distances to the Mandelbrot boundary.
///
/// This is a thin wrapper over :func:`sample_signed_distance_field`:
///
/// ```text
/// sample_distance_field(p) = |sample_signed_distance_field(p)|
/// ```
///
/// It exists for callers that only need the magnitude of the distance to the
/// boundary and cannot distinguish inside vs. outside from the sign. Realm
/// classification must come from :func:`sample_signed_distance_field` (or an
/// equivalent single authority); do not reconstruct sign with a separate
/// escape-iteration heuristic.
///
/// - For points within the field's bounding box, bicubic interpolation (for
///   fields >= 4x4) or bilinear interpolation (for smaller fields) is used at
///   exactly the requested coordinate `c`. The returned magnitude is the
///   absolute distance to the boundary at `c`; it is **not** subpixel-refined.
/// - For points outside the bounding box, the returned distance is the sum of the
///   Euclidean distance from the point to the nearest edge of the box, plus the
///   unsigned distance at that edge.
///
/// If the distance field is not loaded, this function will attempt to auto-load the
/// built-in "mandelbrot_default" field.
///
/// # Arguments
/// * `points` - complex coordinates in the plane
///
/// # Returns
/// A vector of unsigned distances (non-negative floats) to the Mandelbrot boundary.
pub fn sample_distance_field(points: &[num_complex::Complex64]) -> Result<Vec<f32>, String> {
    Ok(sample_signed_distance_field(points)?
        .into_iter()
        .map(|v| v.abs())
        .collect())
}

const DEFAULT_MANDELBROT_MAX_ITER: usize = 8192;
const DEFAULT_MANDELBROT_BAILOUT: f64 = 1e6;
const DEFAULT_MANDELBROT_PERIMETER_SAMPLES: usize = 512;

pub fn mandelbrot_distance_estimate(cs: &[num_complex::Complex64]) -> Result<Vec<f32>, String> {
    // Public, user-friendly wrapper that uses sensible defaults.
    mandelbrot_distance_estimate_with_params(
        cs,
        DEFAULT_MANDELBROT_MAX_ITER,
        DEFAULT_MANDELBROT_BAILOUT,
        DEFAULT_MANDELBROT_PERIMETER_SAMPLES,
    )
}

pub fn mandelbrot_distance_estimate_with_params(
    cs: &[num_complex::Complex64],
    max_iter: usize,
    bailout: f64,
    perimeter_samples: usize,
) -> Result<Vec<f32>, String> {
    // Ensure builtin field is loaded via sample_distance_field if needed
    let mut out: Vec<f32> = Vec::with_capacity(cs.len());

    // helper: analytic DEM estimator with bounded short-cycle detection
    // Returns (Option<distance>, cycle_detected)
    let analytic_dem = |c: num_complex::Complex64| -> (Option<f64>, bool) {
        let mut z = num_complex::Complex64::new(0.0, 0.0);
        let mut dz = num_complex::Complex64::new(0.0, 0.0);
        let max_period: usize = 20;
        let tol: f64 = 1e-12;
        let mut history: Vec<num_complex::Complex64> = Vec::new();
        for _ in 0..max_iter {
            dz = num_complex::Complex64::new(2.0, 0.0) * z * dz + num_complex::Complex64::new(1.0, 0.0);
            z = z * z + c;
            // detect short cycles
            for &prev in &history {
                if (z - prev).norm() < tol {
                    // Short cycle detected -> treat as non-escaping (inside)
                    return (None, true);
                }
            }
            history.push(z);
            if history.len() > max_period { history.remove(0); }
            if z.norm() > bailout {
                let denom = dz.norm();
                if denom == 0.0 || !denom.is_finite() { return (None, false); }
                // 2*|z|*ln|z|/|dz|
                let val: f64 = 2.0_f64 * z.norm() * z.norm().ln() / denom;
                if val.is_finite() { return (Some(val), false); } else { return (None, false); }
            }
        }
        (None, false)
    };

    // helper: perimeter min distance using signed SDF
    let perimeter_min = |xr: f64, yr: f64| -> Result<f64, String> {
        // read builtin metadata (best-effort)
        let guard = DIST_FIELD.read().map_err(|e| format!("lock error: {}", e))?;
        if guard.is_none() {
            drop(guard);
            load_builtin_distance_field("mandelbrot_default")?;
        }
        let guard = DIST_FIELD.read().map_err(|e| format!("lock error: {}", e))?;
        let df = guard.as_ref().ok_or_else(|| "distance field not loaded".to_string())?;
        let xmin = df.xmin;
        let xmax = df.xmax;
        let ymin = df.ymin;
        let ymax = df.ymax;

        let xs_top: Vec<f64> = (0..perimeter_samples).map(|i| xmin + (xmax - xmin) * (i as f64) / ((perimeter_samples - 1) as f64)).collect();
        let ys_lr: Vec<f64> = (0..perimeter_samples).map(|i| ymin + (ymax - ymin) * (i as f64) / ((perimeter_samples - 1) as f64)).collect();

        let mut perimeter_cs: Vec<num_complex::Complex64> = Vec::with_capacity(perimeter_samples * 4);
        perimeter_cs.extend(xs_top.iter().map(|&x| num_complex::Complex64::new(x, ymax)));
        perimeter_cs.extend(xs_top.iter().map(|&x| num_complex::Complex64::new(x, ymin)));
        perimeter_cs.extend(ys_lr.iter().map(|&y| num_complex::Complex64::new(xmin, y)));
        perimeter_cs.extend(ys_lr.iter().map(|&y| num_complex::Complex64::new(xmax, y)));

        let sdf_vals = sample_distance_field(&perimeter_cs)?;
        if sdf_vals.len() != perimeter_cs.len() { return Err("sdf length mismatch".to_string()) }
        let mut best = std::f64::INFINITY;
        for (p, &sdf) in perimeter_cs.iter().zip(sdf_vals.iter()) {
            let xb = p.re;
            let yb = p.im;
            let d = ((xr - xb) * (xr - xb) + (yr - yb) * (yr - yb)).sqrt() + (sdf as f64).abs();
            if d < best { best = d }
        }
        Ok(best)
    };

    for c in cs.iter() {
        let xr = c.re;
        let yr = c.im;
        let (dem_opt, cycle_detected) = analytic_dem(*c);
        if let Some(dem) = dem_opt {
            // compute perimeter-based candidate and pick min
            match perimeter_min(xr, yr) {
                Ok(per) => {
                    let per_val: f64 = per;
                    let chosen: f64 = if dem.is_finite() && dem < per_val { dem } else { per_val };
                    out.push(chosen as f32);
                }
                Err(_) => out.push(dem as f32),
            }
        } else {
            // use signed SDF, but compare against a perimeter-based outside candidate and
            // pick the value (or its magnitude) that gives the smaller absolute distance.
            let v_signed = sample_distance_field(&[*c])?;
            let v = v_signed[0] as f64;
            match perimeter_min(xr, yr) {
                Ok(per) => {
                    let mut chosen: f64 = if v.abs() > per { per } else { v };
                    // If a short-cycle was detected (analytic-dem found a cycle), we
                    // have high confidence this is an interior/non-escaping point.
                    if cycle_detected && chosen > 0.0 {
                        chosen = 0.0;
                    }
                    out.push(chosen as f32);
                }
                Err(_) => {
                    let mut chosen: f64 = v;
                    if cycle_detected && chosen > 0.0 {
                        chosen = 0.0;
                    }
                    if chosen > 0.0 && chosen.abs() < 1e-8 {
                        chosen = 0.0;
                    }
                    out.push(chosen as f32)
                },
            }
        }
    }

    Ok(out)
}