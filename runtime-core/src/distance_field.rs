use ndarray::Array2;
use once_cell::sync::Lazy;
use std::path::Path;
use std::sync::RwLock;

#[derive(Clone, Debug)]
struct DistanceField {
    data: Array2<f32>,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
}

static DIST_FIELD: Lazy<RwLock<Option<DistanceField>>> = Lazy::new(|| RwLock::new(None));

pub fn load_distance_field<P: AsRef<Path>>(_path: P) -> Result<(), String> {
    Err("loading .npy from Rust is not implemented on this build; call set_distance_field_from_vec from Python instead".into())
}

pub fn set_distance_field_from_vec(data: Vec<f32>, rows: usize, cols: usize, xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Result<(), String> {
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

            // Set into in-memory field
            set_distance_field_from_vec(flat, rows, cols, xmin, xmax, ymin, ymax)?;
            Ok((rows, cols, xmin, xmax, ymin, ymax))
        }
        other => Err(format!("unknown builtin distance field: {}", other)),
    }
}
pub fn sample_distance_field(xs: &[f64], ys: &[f64]) -> Result<Vec<f32>, String> {
    if xs.len() != ys.len() {
        return Err("xs and ys must have the same length".into());
    }
    let guard = DIST_FIELD.read().map_err(|e| format!("lock error: {}", e))?;
    let df = guard.as_ref().ok_or_else(|| "distance field not loaded".to_string())?;

    let (h, w) = (df.data.nrows() as f64, df.data.ncols() as f64);
    let dx = (df.xmax - df.xmin) as f64;
    let dy = (df.ymax - df.ymin) as f64;

    let mut out = Vec::with_capacity(xs.len());
    for (&xr, &yr) in xs.iter().zip(ys.iter()) {
        // normalized [0,1]
        let mut u = (xr - df.xmin) / dx;
        let mut v = (yr - df.ymin) / dy;
        u = u.clamp(0.0, 1.0);
        v = v.clamp(0.0, 1.0);
        // pixel coordinates
        let fx = u * (w - 1.0);
        let fy = v * (h - 1.0);
        let x0 = fx.floor() as isize;
        let y0 = fy.floor() as isize;
        let x1 = (x0 + 1).min((w - 1.0) as isize) as isize;
        let y1 = (y0 + 1).min((h - 1.0) as isize) as isize;
        let sx = (fx - x0 as f64) as f32;
        let sy = (fy - y0 as f64) as f32;
        // Access elements (row=y, col=x)
        let v00 = df.data[[y0 as usize, x0 as usize]];
        let v10 = df.data[[y0 as usize, x1 as usize]];
        let v01 = df.data[[y1 as usize, x0 as usize]];
        let v11 = df.data[[y1 as usize, x1 as usize]];
        let a = v00 * (1.0 - sx) + v10 * sx;
        let b = v01 * (1.0 - sx) + v11 * sx;
        let s = a * (1.0 - sy) + b * sy;
        out.push(s.abs());
    }
    Ok(out)
}
