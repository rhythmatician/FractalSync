//! Mandelbrot minimap sampling and mip pyramid.
//!
//! This module builds a parameter-plane scalar field F(c) and provides
//! sampling helpers (bilinear point sample, patch extraction, gradients,
//! and sensitivity) for the step controller.

use crate::geometry::Complex;

pub const MAX_ITER: usize = 512;
pub const ESCAPE_RADIUS: f64 = 2.0;

pub const RE_MIN: f64 = -2.0;
pub const RE_MAX: f64 = 1.0;
pub const IM_MIN: f64 = -1.5;
pub const IM_MAX: f64 = 1.5;

pub const BASE_RES: usize = 2048;
pub const MIP_LEVELS: usize = 12; // 0..11 inclusive

pub const PATCH_SIZE: usize = 16;

#[derive(Clone, Debug)]
pub struct MipLevel {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Minimap {
    pub levels: Vec<MipLevel>,
}

#[derive(Clone, Debug)]
pub struct MinimapSample {
    pub f: f32,
    pub nu_norm: f32,
    pub membership: bool,
    pub grad_re: f32,
    pub grad_im: f32,
    pub sensitivity: f32,
    pub patch: Vec<f32>,
    pub mip_level: usize,
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

fn clamp_f32(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn map_to_texel(c: Complex, width: usize, height: usize) -> (f64, f64) {
    let u = (c.real - RE_MIN) / (RE_MAX - RE_MIN);
    let v = (c.imag - IM_MIN) / (IM_MAX - IM_MIN);
    let x = u * (width.saturating_sub(1) as f64);
    let y = v * (height.saturating_sub(1) as f64);
    (x, y)
}

fn bilinear_sample(level: &MipLevel, c: Complex) -> f32 {
    let (x, y) = map_to_texel(c, level.width, level.height);
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let x0c = clamp_index(x0, level.width);
    let x1c = clamp_index(x1, level.width);
    let y0c = clamp_index(y0, level.height);
    let y1c = clamp_index(y1, level.height);

    let fx = (x - x0 as f64).clamp(0.0, 1.0) as f32;
    let fy = (y - y0 as f64).clamp(0.0, 1.0) as f32;

    let idx00 = y0c * level.width + x0c;
    let idx10 = y0c * level.width + x1c;
    let idx01 = y1c * level.width + x0c;
    let idx11 = y1c * level.width + x1c;

    let f00 = level.data.get(idx00).copied().unwrap_or(0.0);
    let f10 = level.data.get(idx10).copied().unwrap_or(0.0);
    let f01 = level.data.get(idx01).copied().unwrap_or(0.0);
    let f11 = level.data.get(idx11).copied().unwrap_or(0.0);

    let fx0 = f00 + (f10 - f00) * fx;
    let fx1 = f01 + (f11 - f01) * fx;
    fx0 + (fx1 - fx0) * fy
}

fn nearest_patch(level: &MipLevel, c: Complex, k: usize) -> Vec<f32> {
    let (x, y) = map_to_texel(c, level.width, level.height);
    let cx = x.round() as isize;
    let cy = y.round() as isize;
    let half = (k / 2) as isize;

    let mut patch = Vec::with_capacity(k * k);
    for row in 0..k as isize {
        for col in 0..k as isize {
            let ix = clamp_index(cx + col - half, level.width);
            let iy = clamp_index(cy + row - half, level.height);
            let idx = iy * level.width + ix;
            patch.push(level.data.get(idx).copied().unwrap_or(0.0));
        }
    }
    patch
}

fn gradient(level: &MipLevel, c: Complex) -> (f32, f32) {
    let (x, y) = map_to_texel(c, level.width, level.height);
    let ix = clamp_index(x.round() as isize, level.width);
    let iy = clamp_index(y.round() as isize, level.height);

    let ix_prev = clamp_index(ix as isize - 1, level.width);
    let ix_next = clamp_index(ix as isize + 1, level.width);
    let iy_prev = clamp_index(iy as isize - 1, level.height);
    let iy_next = clamp_index(iy as isize + 1, level.height);

    let idx_left = iy * level.width + ix_prev;
    let idx_right = iy * level.width + ix_next;
    let idx_down = iy_prev * level.width + ix;
    let idx_up = iy_next * level.width + ix;

    let f_left = level.data.get(idx_left).copied().unwrap_or(0.0);
    let f_right = level.data.get(idx_right).copied().unwrap_or(0.0);
    let f_down = level.data.get(idx_down).copied().unwrap_or(0.0);
    let f_up = level.data.get(idx_up).copied().unwrap_or(0.0);

    let d_re = (RE_MAX - RE_MIN) / (level.width.saturating_sub(1).max(1) as f64);
    let d_im = (IM_MAX - IM_MIN) / (level.height.saturating_sub(1).max(1) as f64);

    let grad_re = (f_right - f_left) as f64 / (2.0 * d_re);
    let grad_im = (f_up - f_down) as f64 / (2.0 * d_im);

    (grad_re as f32, grad_im as f32)
}

pub fn escape_time(c: Complex) -> (f64, bool) {
    let mut z = Complex::new(0.0, 0.0);
    for n in 0..MAX_ITER {
        let mag_sq = z.real * z.real + z.imag * z.imag;
        if mag_sq > ESCAPE_RADIUS * ESCAPE_RADIUS {
            let mag = mag_sq.sqrt();
            let nu = n as f64 + 1.0 - (mag.ln().ln() / std::f64::consts::LN_2);
            return (nu, false);
        }
        z = z * z + c;
    }
    (MAX_ITER as f64, true)
}

pub fn nu_norm_direct(c: Complex) -> f32 {
    let (nu, _inside) = escape_time(c);
    clamp_f32((nu / MAX_ITER as f64) as f32, 0.0, 1.0)
}

pub fn membership_direct(c: Complex) -> bool {
    let (_nu, inside) = escape_time(c);
    inside
}

impl Minimap {
    pub fn new() -> Self {
        Self::new_with_resolution(BASE_RES)
    }

    pub fn new_with_resolution(base_res: usize) -> Self {
        let base = Self::build_base(base_res.max(1));
        let mut levels = vec![base];
        while levels.len() < MIP_LEVELS {
            let last = levels.last().expect("base level exists");
            if last.width == 1 && last.height == 1 {
                break;
            }
            levels.push(Self::downsample(last));
        }
        Self { levels }
    }

    fn build_base(base_res: usize) -> MipLevel {
        let width = base_res;
        let height = base_res;
        let mut data = vec![0.0_f32; width * height];
        for y in 0..height {
            let imag = IM_MIN + (IM_MAX - IM_MIN) * (y as f64) / (height.saturating_sub(1) as f64);
            for x in 0..width {
                let real = RE_MIN + (RE_MAX - RE_MIN) * (x as f64) / (width.saturating_sub(1) as f64);
                let c = Complex::new(real, imag);
                let (nu, _inside) = escape_time(c);
                let f = clamp_f32((nu / MAX_ITER as f64) as f32, 0.0, 1.0);
                data[y * width + x] = f;
            }
        }
        MipLevel { width, height, data }
    }

    fn downsample(level: &MipLevel) -> MipLevel {
        let width = (level.width / 2).max(1);
        let height = (level.height / 2).max(1);
        let mut data = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let src_x = x * 2;
                let src_y = y * 2;
                let mut sum = 0.0;
                let mut count = 0.0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let ix = (src_x + dx).min(level.width - 1);
                        let iy = (src_y + dy).min(level.height - 1);
                        sum += level.data[iy * level.width + ix];
                        count += 1.0;
                    }
                }
                data[y * width + x] = sum / count;
            }
        }
        MipLevel { width, height, data }
    }

    pub fn sample(&self, c: Complex, mip_level: usize) -> MinimapSample {
        let level = self
            .levels
            .get(mip_level.min(self.levels.len().saturating_sub(1)))
            .expect("minimap level exists");
        let f = bilinear_sample(level, c);
        let patch = nearest_patch(level, c, PATCH_SIZE);
        let (grad_re, grad_im) = gradient(level, c);
        let g = (grad_re * grad_re + grad_im * grad_im).sqrt();
        let g0 = 0.02;
        let sensitivity = g / (g + g0);
        let nu_norm = f;
        let membership = membership_direct(c);

        MinimapSample {
            f,
            nu_norm,
            membership,
            grad_re,
            grad_im,
            sensitivity,
            patch,
            mip_level: mip_level.min(self.levels.len().saturating_sub(1)),
        }
    }

    pub fn sample_sensitivity(&self, c: Complex, mip_level: usize) -> f32 {
        let level = self
            .levels
            .get(mip_level.min(self.levels.len().saturating_sub(1)))
            .expect("minimap level exists");
        let (grad_re, grad_im) = gradient(level, c);
        let g = (grad_re * grad_re + grad_im * grad_im).sqrt();
        let g0 = 0.02;
        g / (g + g0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_time_inside_outside() {
        let inside = Complex::new(0.0, 0.0);
        let outside = Complex::new(2.0, 0.0);
        let (nu_inside, inside_flag) = escape_time(inside);
        assert!(inside_flag);
        assert_eq!(nu_inside as usize, MAX_ITER);
        let (_nu_out, inside_flag_out) = escape_time(outside);
        assert!(!inside_flag_out);
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let minimap = Minimap::new_with_resolution(64);
        let level = &minimap.levels[0];
        let c = Complex::new(-0.7, 0.0);
        let (gx, gy) = gradient(level, c);

        // Finite difference approx from nearby samples
        let eps_re = (RE_MAX - RE_MIN) / (level.width.saturating_sub(1).max(1) as f64);
        let eps_im = (IM_MAX - IM_MIN) / (level.height.saturating_sub(1).max(1) as f64);
        let c_left = Complex::new(c.real - eps_re, c.imag);
        let c_right = Complex::new(c.real + eps_re, c.imag);
        let c_down = Complex::new(c.real, c.imag - eps_im);
        let c_up = Complex::new(c.real, c.imag + eps_im);

        let f_left = bilinear_sample(level, c_left);
        let f_right = bilinear_sample(level, c_right);
        let f_down = bilinear_sample(level, c_down);
        let f_up = bilinear_sample(level, c_up);

        let approx_gx = (f_right - f_left) as f64 / (2.0 * eps_re);
        let approx_gy = (f_up - f_down) as f64 / (2.0 * eps_im);

        let diff_gx = (gx as f64 - approx_gx).abs();
        let diff_gy = (gy as f64 - approx_gy).abs();

        assert!(diff_gx < 1e-3, "grad_re mismatch: {}", diff_gx);
        assert!(diff_gy < 1e-3, "grad_im mismatch: {}", diff_gy);
    }
}
