//! Legacy clamped motion and on-demand minimap sampling.

/// Sample the on-demand distance estimate.
pub fn deep_zoom_field(re: &[f64], im: &[f64]) -> Result<Vec<f32>, String> {
    if re.len() != im.len() {
        return Err(format!(
            "re/im length mismatch: {} vs {}",
            re.len(),
            im.len()
        ));
    }
    let cs: Vec<num_complex::Complex64> = re
        .iter()
        .zip(im.iter())
        .map(|(&r, &i)| num_complex::Complex64::new(r, i))
        .collect();
    crate::distance_field::mandelbrot_distance_estimate(&cs)
}
