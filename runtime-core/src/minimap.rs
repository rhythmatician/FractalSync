//! Mandelbrot minimap sampler and mip pyramid.
//!
//! This module defines the minimap field F(c) used for controller context.

use crate::geometry::Complex;

pub const MINIMAP_RE_MIN: f64 = -2.0;
pub const MINIMAP_RE_MAX: f64 = 1.0;
pub const MINIMAP_IM_MIN: f64 = -1.5;
pub const MINIMAP_IM_MAX: f64 = 1.5;
pub const MINIMAP_BASE_RES: usize = 2048;
pub const MINIMAP_MIP_LEVELS: usize = 12;
pub const MINIMAP_PATCH_K: usize = 16;

pub const MAX_ITER: usize = 512;
pub const ESCAPE_RADIUS: f64 = 2.0;

#[derive(Clone, Debug)]
pub struct MinimapLevel {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Minimap {
    pub levels: Vec<MinimapLevel>,
}

#[derive(Clone, Copy, Debug)]
pub struct MinimapSample {
    pub f_value: f32,
    pub grad_re: f32,
    pub grad_im: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct EscapeInfo {
    pub nu: f64,
    pub escaped: bool,
}

#[inline]
fn clamp_unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[inline]
fn map_to_uv(c: Complex) -> (f64, f64) {
    let u = (c.real - MINIMAP_RE_MIN) / (MINIMAP_RE_MAX - MINIMAP_RE_MIN);
    let v = (c.imag - MINIMAP_IM_MIN) / (MINIMAP_IM_MAX - MINIMAP_IM_MIN);
    (clamp_unit(u), clamp_unit(v))
}

#[inline]
fn clamp_index(idx: isize, max: usize) -> usize {
    if idx < 0 {
        0
    } else if idx as usize >= max {
        max.saturating_sub(1)
    } else {
        idx as usize
    }
}

impl MinimapLevel {
    pub fn sample_bilinear(&self, c: Complex) -> f32 {
        let (u, v) = map_to_uv(c);
        let x = u * (self.width.saturating_sub(1)) as f64;
        let y = v * (self.height.saturating_sub(1)) as f64;

        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let fx = x - x0 as f64;
        let fy = y - y0 as f64;

        let x0c = clamp_index(x0, self.width);
        let x1c = clamp_index(x1, self.width);
        let y0c = clamp_index(y0, self.height);
        let y1c = clamp_index(y1, self.height);

        let idx = |xi: usize, yi: usize| yi * self.width + xi;
        let f00 = self.data[idx(x0c, y0c)] as f64;
        let f10 = self.data[idx(x1c, y0c)] as f64;
        let f01 = self.data[idx(x0c, y1c)] as f64;
        let f11 = self.data[idx(x1c, y1c)] as f64;

        let f0 = f00 * (1.0 - fx) + f10 * fx;
        let f1 = f01 * (1.0 - fx) + f11 * fx;
        let f = f0 * (1.0 - fy) + f1 * fy;

        f as f32
    }

    pub fn sample_patch(&self, c: Complex, k: usize) -> Vec<f32> {
        let (u, v) = map_to_uv(c);
        let x = (u * (self.width.saturating_sub(1)) as f64).round() as isize;
        let y = (v * (self.height.saturating_sub(1)) as f64).round() as isize;

        let half = (k / 2) as isize;
        let mut patch = Vec::with_capacity(k * k);
        for dy in 0..k as isize {
            let sy = clamp_index(y + dy - half, self.height);
            for dx in 0..k as isize {
                let sx = clamp_index(x + dx - half, self.width);
                patch.push(self.data[sy * self.width + sx]);
            }
        }
        patch
    }

    pub fn sample_gradient(&self, c: Complex) -> (f32, f32) {
        let (u, v) = map_to_uv(c);
        let x = (u * (self.width.saturating_sub(1)) as f64).round() as isize;
        let y = (v * (self.height.saturating_sub(1)) as f64).round() as isize;

        let i = clamp_index(x, self.width);
        let j = clamp_index(y, self.height);

        let i_minus = clamp_index(x - 1, self.width);
        let i_plus = clamp_index(x + 1, self.width);
        let j_minus = clamp_index(y - 1, self.height);
        let j_plus = clamp_index(y + 1, self.height);

        let idx = |xi: usize, yi: usize| yi * self.width + xi;

        let denom_re = (self.width.saturating_sub(1)).max(1) as f64;
        let denom_im = (self.height.saturating_sub(1)).max(1) as f64;
        let d_re = (MINIMAP_RE_MAX - MINIMAP_RE_MIN) / denom_re;
        let d_im = (MINIMAP_IM_MAX - MINIMAP_IM_MIN) / denom_im;

        let f_re_plus = self.data[idx(i_plus, j)] as f64;
        let f_re_minus = self.data[idx(i_minus, j)] as f64;
        let f_im_plus = self.data[idx(i, j_plus)] as f64;
        let f_im_minus = self.data[idx(i, j_minus)] as f64;

        let dfd_re = (f_re_plus - f_re_minus) / (2.0 * d_re);
        let dfd_im = (f_im_plus - f_im_minus) / (2.0 * d_im);

        (dfd_re as f32, dfd_im as f32)
    }
}

impl Minimap {
    pub fn new() -> Self {
        Self::new_with_resolution(MINIMAP_BASE_RES, MINIMAP_MIP_LEVELS)
    }

    pub fn new_with_resolution(base_res: usize, mip_levels: usize) -> Self {
        let base = generate_base_level(base_res);
        let mut levels = Vec::with_capacity(mip_levels);
        levels.push(base);

        while levels.len() < mip_levels {
            let prev = levels.last().expect("base level exists");
            if prev.width == 1 && prev.height == 1 {
                break;
            }
            let next = downsample_level(prev);
            levels.push(next);
        }

        Minimap { levels }
    }

    pub fn level(&self, mip: usize) -> &MinimapLevel {
        let idx = mip.min(self.levels.len().saturating_sub(1));
        &self.levels[idx]
    }

    pub fn sample(&self, c: Complex, mip: usize) -> MinimapSample {
        let level = self.level(mip);
        let f_value = level.sample_bilinear(c);
        let (grad_re, grad_im) = level.sample_gradient(c);
        MinimapSample {
            f_value,
            grad_re,
            grad_im,
        }
    }

    pub fn sample_patch(&self, c: Complex, mip: usize, k: usize) -> Vec<f32> {
        self.level(mip).sample_patch(c, k)
    }

    pub fn escape_info(c: Complex) -> EscapeInfo {
        let mut z = Complex::new(0.0, 0.0);
        for n in 0..MAX_ITER {
            if z.real * z.real + z.imag * z.imag > ESCAPE_RADIUS * ESCAPE_RADIUS {
                let mag = z.mag().max(1e-12);
                let nu = n as f64 + 1.0 - (mag.ln().ln() / 2.0f64.ln());
                return EscapeInfo { nu, escaped: true };
            }
            z = z * z + c;
        }
        EscapeInfo {
            nu: MAX_ITER as f64,
            escaped: false,
        }
    }

    pub fn nu_norm_from_escape(info: EscapeInfo) -> f32 {
        let f = (info.nu / MAX_ITER as f64).clamp(0.0, 1.0);
        f as f32
    }

    pub fn membership(c: Complex) -> bool {
        !Self::escape_info(c).escaped
    }
}

fn generate_base_level(base_res: usize) -> MinimapLevel {
    let width = base_res.max(1);
    let height = base_res.max(1);
    let mut data = vec![0.0f32; width * height];
    let re_span = MINIMAP_RE_MAX - MINIMAP_RE_MIN;
    let im_span = MINIMAP_IM_MAX - MINIMAP_IM_MIN;

    for y in 0..height {
        let denom_y = (height.saturating_sub(1)).max(1) as f64;
        let im = MINIMAP_IM_MIN + (y as f64) * im_span / denom_y;
        for x in 0..width {
            let denom_x = (width.saturating_sub(1)).max(1) as f64;
            let re = MINIMAP_RE_MIN + (x as f64) * re_span / denom_x;
            let c = Complex::new(re, im);
            let escape = Minimap::escape_info(c);
            let f = Minimap::nu_norm_from_escape(escape);
            data[y * width + x] = f;
        }
    }

    MinimapLevel {
        width,
        height,
        data,
    }
}

fn downsample_level(prev: &MinimapLevel) -> MinimapLevel {
    let width = (prev.width / 2).max(1);
    let height = (prev.height / 2).max(1);
    let mut data = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let src_x = x * 2;
            let src_y = y * 2;
            let idx = |ix: usize, iy: usize| iy * prev.width + ix;
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let sx = (src_x + dx).min(prev.width - 1);
                    let sy = (src_y + dy).min(prev.height - 1);
                    sum += prev.data[idx(sx, sy)];
                    count += 1.0;
                }
            }
            data[y * width + x] = sum / count;
        }
    }

    MinimapLevel {
        width,
        height,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimap_builds_mip_chain() {
        let map = Minimap::new_with_resolution(32, 6);
        assert!(map.levels.len() >= 1);
        assert_eq!(map.levels[0].width, 32);
        assert_eq!(map.levels[0].height, 32);
        let last = map.levels.last().unwrap();
        assert_eq!(last.width, 1);
        assert_eq!(last.height, 1);
    }
}
