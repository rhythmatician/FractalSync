//! Step-based controller for Julia parameter navigation.

use crate::geometry::Complex;
use crate::minimap::{Minimap, MinimapSample, MINIMAP_PATCH_K};

pub const MAX_STEP_BASE: f64 = 0.02;
pub const BETA: f64 = 2.5;
pub const GAMMA: f64 = 8.0;
pub const DOMAIN_R: f64 = 2.0;
pub const WALL_MARGIN: f64 = 0.15;
pub const WALL_K: f64 = 2.0;
pub const WALL_P: f64 = 1.0;
pub const DC_REF: f64 = 0.002;
pub const G0: f64 = 0.02;

pub const CONTEXT_LEN: usize = 2 + 2 + 1 + 1 + 2 + 1 + (MINIMAP_PATCH_K * MINIMAP_PATCH_K);

#[derive(Clone, Copy, Debug)]
pub struct StepDebug {
    pub mip_level: usize,
    pub scale_g: f64,
    pub scale_df: f64,
    pub scale: f64,
    pub wall_applied: bool,
}

#[derive(Clone, Debug)]
pub struct ControllerContext {
    pub c: Complex,
    pub prev_delta: Complex,
    pub nu_norm: f32,
    pub membership: bool,
    pub grad_re: f32,
    pub grad_im: f32,
    pub sensitivity: f32,
    pub patch: Vec<f32>,
}

impl ControllerContext {
    pub fn to_feature_vector(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(CONTEXT_LEN);
        out.push(self.c.real as f32);
        out.push(self.c.imag as f32);
        out.push(self.prev_delta.real as f32);
        out.push(self.prev_delta.imag as f32);
        out.push(self.nu_norm);
        out.push(if self.membership { 1.0 } else { 0.0 });
        out.push(self.grad_re);
        out.push(self.grad_im);
        out.push(self.sensitivity);
        out.extend_from_slice(&self.patch);
        out
    }
}

#[derive(Clone, Debug)]
pub struct StepResult {
    pub delta_applied: Complex,
    pub c_next: Complex,
    pub debug: StepDebug,
    pub context: ControllerContext,
}

#[derive(Clone, Debug)]
pub struct StepController {
    minimap: Minimap,
}

impl StepController {
    pub fn new() -> Self {
        Self {
            minimap: Minimap::new(),
        }
    }

    pub fn minimap(&self) -> &Minimap {
        &self.minimap
    }

    pub fn context_for(&self, c: Complex, prev_delta: Option<Complex>, mip_level: usize) -> ControllerContext {
        let prev_delta = prev_delta.unwrap_or_else(|| Complex::new(0.0, 0.0));
        let sample = self.minimap.sample(c, mip_level);
        let patch = self.minimap.sample_patch(c, mip_level, MINIMAP_PATCH_K);
        let sensitivity = sensitivity_from_grad(sample);
        let nu_norm = sample.f_value;
        let membership = Minimap::membership(c);
        ControllerContext {
            c,
            prev_delta,
            nu_norm,
            membership,
            grad_re: sample.grad_re,
            grad_im: sample.grad_im,
            sensitivity,
            patch,
        }
    }

    pub fn apply_step(
        &self,
        c_t: Complex,
        delta_model: Complex,
        prev_delta: Option<Complex>,
    ) -> StepResult {
        let delta0 = clamp_len(delta_model, MAX_STEP_BASE);
        let mip_level = mip_for_delta(delta0);
        let context = self.context_for(c_t, prev_delta, mip_level);
        let sample = MinimapSample {
            f_value: context.nu_norm,
            grad_re: context.grad_re,
            grad_im: context.grad_im,
        };

        let delta_f_pred = sample.grad_re as f64 * delta0.real + sample.grad_im as f64 * delta0.imag;
        let scale_g = 1.0 / (1.0 + BETA * context.sensitivity as f64);
        let scale_df = 1.0 / (1.0 + GAMMA * delta_f_pred.abs());
        let scale = scale_g.min(scale_df);
        let delta1 = delta0.scale(scale);

        let (delta2, wall_applied) = apply_wall(c_t, delta1);

        let mut c_next = c_t.add(delta2);
        let delta_applied = if c_next.mag() >= DOMAIN_R {
            let scale = (DOMAIN_R - 1e-6) / c_next.mag().max(1e-12);
            c_next = c_next.scale(scale);
            c_next.add(c_t.scale(-1.0))
        } else {
            delta2
        };

        StepResult {
            delta_applied,
            c_next,
            debug: StepDebug {
                mip_level,
                scale_g,
                scale_df,
                scale,
                wall_applied,
            },
            context,
        }
    }
}

pub fn sensitivity_from_grad(sample: MinimapSample) -> f32 {
    let g = ((sample.grad_re * sample.grad_re) + (sample.grad_im * sample.grad_im)).sqrt();
    (g / (g + G0 as f32)).clamp(0.0, 1.0)
}

pub fn clamp_len(delta: Complex, max_len: f64) -> Complex {
    let len = delta.mag();
    if len <= max_len || len == 0.0 {
        delta
    } else {
        delta.scale(max_len / len)
    }
}

pub fn mip_for_delta(delta: Complex) -> usize {
    let len = delta.mag();
    if len <= DC_REF || len == 0.0 {
        return 0;
    }
    let mip = (len / DC_REF).log2().floor() as isize;
    mip.clamp(0, 11) as usize
}

fn apply_wall(c_t: Complex, delta: Complex) -> (Complex, bool) {
    let mut delta1 = delta;
    let mut c_trial = c_t.add(delta1);
    let mut r = c_trial.mag();
    let r0 = DOMAIN_R - WALL_MARGIN;
    if r > r0 {
        let u = if r == 0.0 {
            Complex::new(1.0, 0.0)
        } else {
            c_trial.scale(1.0 / r)
        };
        let out = (delta1.real * u.real + delta1.imag * u.imag).max(0.0);
        delta1 = delta1.add(u.scale(-out));

        c_trial = c_t.add(delta1);
        r = c_trial.mag();
        if r > r0 {
            let push = u.scale(-WALL_K * (r - r0).powf(WALL_P));
            return (delta1.add(push), true);
        }
        return (delta1, true);
    }
    (delta1, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_len_caps_large_steps() {
        let delta = Complex::new(0.2, 0.0);
        let clamped = clamp_len(delta, 0.02);
        assert!(clamped.mag() <= 0.02 + 1e-6);
    }

    #[test]
    fn mip_for_delta_zero() {
        let delta = Complex::new(0.0, 0.0);
        assert_eq!(mip_for_delta(delta), 0);
    }
}
