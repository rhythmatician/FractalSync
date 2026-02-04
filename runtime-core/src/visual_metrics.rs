//! Visual metrics computed in Rust for runtime feedback.
//!
//! These metrics are intended to feed controller/guardrail logic rather than
//! loss-only training objectives. They operate on rendered Julia images and
//! Mandelbrot critical-orbit membership.

#[derive(Clone, Debug)]
pub struct RuntimeVisualMetrics {
    pub edge_density: f64,
    pub color_uniformity: f64,
    pub brightness_mean: f64,
    pub brightness_std: f64,
    pub brightness_range: f64,
    pub mandelbrot_membership: bool,
}

fn clamp_index(value: isize, max: usize) -> usize {
    if value < 0 {
        0
    } else if value as usize >= max {
        max.saturating_sub(1)
    } else {
        value as usize
    }
}

fn to_gray(image: &[f64], width: usize, height: usize, channels: usize) -> Result<Vec<f64>, &'static str> {
    // Treat channels == 0 as a single-channel (grayscale) image to avoid
    // silently discarding a non-empty buffer and producing misleading metrics.
    let effective_channels = if channels == 0 { 1 } else { channels };

    // Use overflow-safe multiplication to prevent panics on large inputs
    let pixels = match width.checked_mul(height) {
        Some(p) => p,
        None => return Err("image dimensions are too large"),
    };

    let mut gray = vec![0.0; pixels];
    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * effective_channels;
            let value = if effective_channels == 1 {
                image.get(base).copied().unwrap_or(0.0)
            } else {
                let r = image.get(base).copied().unwrap_or(0.0);
                let g = image.get(base + 1).copied().unwrap_or(0.0);
                let b = image.get(base + 2).copied().unwrap_or(0.0);
                (r + g + b) / 3.0
            };
            gray[y * width + x] = value.clamp(0.0, 1.0);
        }
    }
    Ok(gray)
}

fn compute_brightness_stats(gray: &[f64]) -> (f64, f64, f64) {
    if gray.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sum = 0.0;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in gray {
        sum += v;
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    let mean = sum / gray.len() as f64;
    let mut var_sum = 0.0;
    for &v in gray {
        let delta = v - mean;
        var_sum += delta * delta;
    }
    let std = (var_sum / gray.len() as f64).sqrt();
    let range = (max_val - min_val).max(0.0);
    (mean, std, range)
}

/// Computes edge density using a Sobel filter.
///
/// Note: This function performs a 3x3 Sobel convolution at each pixel, resulting in
/// O(width * height) complexity with a constant factor of 9 kernel evaluations per pixel.
/// For runtime use at larger resolutions, consider downsampling the input or
/// restricting width/height for this API.
fn compute_edge_density(gray: &[f64], width: usize, height: usize) -> f64 {
    if gray.is_empty() || width == 0 || height == 0 {
        return 0.0;
    }
    let mut edge_count = 0usize;
    let threshold = 50.0;
    for y in 0..height {
        for x in 0..width {
            let mut gx = 0.0;
            let mut gy = 0.0;
            for ky in -1..=1 {
                for kx in -1..=1 {
                    let sample_x = clamp_index(x as isize + kx, width);
                    let sample_y = clamp_index(y as isize + ky, height);
                    let value = gray[sample_y * width + sample_x] * 255.0;
                    let weight_x = match (ky, kx) {
                        (-1, -1) | (1, -1) => -1.0,
                        (-1, 1) | (1, 1) => 1.0,
                        (0, -1) => -2.0,
                        (0, 1) => 2.0,
                        _ => 0.0,
                    };
                    let weight_y = match (ky, kx) {
                        (-1, -1) | (-1, 1) => -1.0,
                        (1, -1) | (1, 1) => 1.0,
                        (-1, 0) => -2.0,
                        (1, 0) => 2.0,
                        _ => 0.0,
                    };
                    gx += value * weight_x;
                    gy += value * weight_y;
                }
            }
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude >= threshold {
                edge_count += 1;
            }
        }
    }
    edge_count as f64 / (width * height) as f64
}

/// Computes color uniformity based on local variance.
///
/// Note: This function performs a 5x5 neighborhood variance computation at each pixel,
/// resulting in O(width * height) complexity with a constant factor of 25 kernel evaluations
/// per pixel. For runtime use at larger resolutions, consider downsampling the input or
/// restricting width/height for this API.
fn compute_color_uniformity(gray: &[f64], width: usize, height: usize) -> f64 {
    if gray.is_empty() || width == 0 || height == 0 {
        return 0.0;
    }
    let kernel_radius = 2isize;
    let kernel_size = 5usize;
    let kernel_area = (kernel_size * kernel_size) as f64;
    let mut variance_sum = 0.0;

    for y in 0..height {
        for x in 0..width {
            let mut local_sum = 0.0;
            let mut local_sq_sum = 0.0;
            for ky in -kernel_radius..=kernel_radius {
                for kx in -kernel_radius..=kernel_radius {
                    let sample_x = clamp_index(x as isize + kx, width);
                    let sample_y = clamp_index(y as isize + ky, height);
                    let value = gray[sample_y * width + sample_x];
                    local_sum += value;
                    local_sq_sum += value * value;
                }
            }
            let mean = local_sum / kernel_area;
            let variance = (local_sq_sum / kernel_area) - mean * mean;
            variance_sum += variance.max(0.0);
        }
    }

    let avg_variance = variance_sum / (width * height) as f64;
    (1.0 / (1.0 + avg_variance * 10.0)).clamp(0.0, 1.0)
}

pub fn mandelbrot_membership(c: num_complex::Complex64, max_iter: usize) -> bool {
    let mut z = num_complex::Complex64::new(0.0, 0.0);
    for _ in 0..max_iter {
        if z.re * z.re + z.im * z.im > 4.0 {
            return false;
        }
        z = z * z + c;
    }
    true
}

pub fn compute_runtime_metrics(
    image: &[f64],
    width: usize,
    height: usize,
    channels: usize,
    c: num_complex::Complex64,
    max_iter: usize,
) -> Result<RuntimeVisualMetrics, &'static str> {
    if width == 0 || height == 0 {
        return Err("width and height must be non-zero");
    }

    // Use overflow-safe multiplication and guard against unreasonably large images,
    // since this function can be called from Python/WASM with untrusted inputs.
    let pixels = match width.checked_mul(height) {
        Some(p) => p,
        None => return Err("image dimensions are too large"),
    };

    // Upper bound to avoid excessive work and potential downstream allocations.
    // 16_777_216 = 4096 x 4096 pixels, which is more than enough for our use cases.
    const MAX_PIXELS: usize = 16_777_216;
    if pixels > MAX_PIXELS {
        return Err("image dimensions are too large");
    }

    let channels_safe = channels.max(1);
    let expected_len = match pixels.checked_mul(channels_safe) {
        Some(len) => len,
        None => return Err("image buffer is too large"),
    };
    if image.len() < expected_len {
        return Err("image buffer is smaller than expected");
    }

    let gray = to_gray(image, width, height, channels_safe)?;
    let edge_density = compute_edge_density(&gray, width, height);
    let color_uniformity = compute_color_uniformity(&gray, width, height);
    let (brightness_mean, brightness_std, brightness_range) = compute_brightness_stats(&gray);
    let membership = mandelbrot_membership(c, max_iter);

    Ok(RuntimeVisualMetrics {
        edge_density,
        color_uniformity,
        brightness_mean,
        brightness_std,
        brightness_range,
        mandelbrot_membership: membership,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mandelbrot_membership_detects_inside_outside() {
        assert!(mandelbrot_membership(num_complex::Complex64::new(0.0, 0.0), 50));
        assert!(!mandelbrot_membership(num_complex::Complex64::new(2.0, 0.0), 10));
    }
}
