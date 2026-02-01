//! Step-based controller for navigating the complex parameter plane.

use crate::geometry::Complex;
use crate::minimap::{Minimap, MinimapSample, PATCH_SIZE};

pub const MAX_STEP_BASE: f64 = 0.02;
pub const BETA: f64 = 2.5;
pub const GAMMA: f64 = 8.0;
pub const DOMAIN_R: f64 = 2.0;
pub const WALL_MARGIN: f64 = 0.15;
pub const WALL_K: f64 = 2.0;
pub const WALL_EXPONENT: f64 = 1.0;
pub const DC_REF: f64 = 0.002;

#[derive(Clone, Copy, Debug)]
pub struct StepState {
    pub c: Complex,
    pub prev_delta: Complex,
}

impl Default for StepState {
    fn default() -> Self {
        Self {
            c: Complex::new(0.0, 0.0),
            prev_delta: Complex::new(0.0, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepDebug {
    pub mip_level: usize,
    pub scale_g: f64,
    pub scale_df: f64,
    pub scale: f64,
    pub delta_f_pred: f64,
    pub wall_applied: bool,
}

#[derive(Clone, Debug)]
pub struct StepContext {
    pub c: Complex,
    pub prev_delta: Complex,
    pub nu_norm: f32,
    pub membership: bool,
    pub grad_re: f32,
    pub grad_im: f32,
    pub sensitivity: f32,
    pub patch: Vec<f32>,
    pub mip_level: usize,
}

impl StepContext {
    pub fn as_feature_vec(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(4 + 1 + 1 + 2 + 1 + PATCH_SIZE * PATCH_SIZE);
        features.push(self.c.real as f32);
        features.push(self.c.imag as f32);
        features.push(self.prev_delta.real as f32);
        features.push(self.prev_delta.imag as f32);
        features.push(self.nu_norm);
        features.push(if self.membership { 1.0 } else { 0.0 });
        features.push(self.grad_re);
        features.push(self.grad_im);
        features.push(self.sensitivity);
        features.extend(self.patch.iter().copied());
        features
    }
}

#[derive(Clone, Debug)]
pub struct StepResult {
    pub c_next: Complex,
    pub delta_applied: Complex,
    pub debug: StepDebug,
    pub context: StepContext,
}

#[derive(Clone, Debug)]
pub struct StepController {
    minimap: Minimap,
}

fn clamp_len(delta: Complex, max_len: f64) -> Complex {
    let len = (delta.real * delta.real + delta.imag * delta.imag).sqrt();
    if len > max_len && len > 0.0 {
        let scale = max_len / len;
        Complex::new(delta.real * scale, delta.imag * scale)
    } else {
        delta
    }
}

fn choose_mip(delta: Complex) -> usize {
    let len = (delta.real * delta.real + delta.imag * delta.imag).sqrt();
    if len <= DC_REF {
        0
    } else {
        let ratio = len / DC_REF;
        let mip = ratio.log2().floor() as isize;
        mip.clamp(0, 11) as usize
    }
}

impl StepController {
    pub fn new() -> Self {
        Self {
            minimap: Minimap::new(),
        }
    }

    pub fn new_with_minimap(minimap: Minimap) -> Self {
        Self { minimap }
    }

    pub fn minimap(&self) -> &Minimap {
        &self.minimap
    }

    pub fn context_for_state(&self, state: &StepState) -> StepContext {
        let mip_level = choose_mip(state.prev_delta);
        let sample = self.minimap.sample(state.c, mip_level);
        StepContext {
            c: state.c,
            prev_delta: state.prev_delta,
            nu_norm: sample.nu_norm,
            membership: sample.membership,
            grad_re: sample.grad_re,
            grad_im: sample.grad_im,
            sensitivity: sample.sensitivity,
            patch: sample.patch,
            mip_level: sample.mip_level,
        }
    }

    pub fn step(&self, state: &mut StepState, delta_model: Complex) -> StepResult {
        let c_before = state.c;
        let prev_delta = state.prev_delta;
        let delta0 = clamp_len(delta_model, MAX_STEP_BASE);
        let mip_level = choose_mip(delta0);
        let sample = self.minimap.sample(c_before, mip_level);
        let (delta1, debug) = apply_throttle_and_wall(c_before, delta0, &sample, mip_level);
        let c_next = c_before + delta1;
        state.prev_delta = delta1;
        state.c = c_next;

        let context = StepContext {
            c: c_before,
            prev_delta,
            nu_norm: sample.nu_norm,
            membership: sample.membership,
            grad_re: sample.grad_re,
            grad_im: sample.grad_im,
            sensitivity: sample.sensitivity,
            patch: sample.patch,
            mip_level: sample.mip_level,
        };

        StepResult {
            c_next,
            delta_applied: delta1,
            debug,
            context,
        }
    }
}

fn apply_throttle_and_wall(
    c: Complex,
    delta0: Complex,
    sample: &MinimapSample,
    mip_level: usize,
) -> (Complex, StepDebug) {
    let delta_f_pred = sample.grad_re as f64 * delta0.real + sample.grad_im as f64 * delta0.imag;
    let scale_g = 1.0 / (1.0 + BETA * sample.sensitivity as f64);
    let scale_df = 1.0 / (1.0 + GAMMA * delta_f_pred.abs());
    let scale = scale_g.min(scale_df);
    let mut delta1 = Complex::new(delta0.real * scale, delta0.imag * scale);

    let mut wall_applied = false;
    let mut c_trial = c + delta1;
    let mut r = (c_trial.real * c_trial.real + c_trial.imag * c_trial.imag).sqrt();
    let r0 = DOMAIN_R - WALL_MARGIN;
    if r > r0 {
        wall_applied = true;
        let mut u = if r > 0.0 {
            Complex::new(c_trial.real / r, c_trial.imag / r)
        } else {
            Complex::new(1.0, 0.0)
        };
        let out = (delta1.real * u.real + delta1.imag * u.imag).max(0.0);
        delta1 = Complex::new(delta1.real - out * u.real, delta1.imag - out * u.imag);

        c_trial = c + delta1;
        r = (c_trial.real * c_trial.real + c_trial.imag * c_trial.imag).sqrt();
        if r > r0 {
            if r > 0.0 {
                u = Complex::new(c_trial.real / r, c_trial.imag / r);
            }
            let push_mag = WALL_K * (r - r0).powf(WALL_EXPONENT);
            delta1 = Complex::new(delta1.real - u.real * push_mag, delta1.imag - u.imag * push_mag);
        }
    }

    let c_next = c + delta1;
    let next_r = (c_next.real * c_next.real + c_next.imag * c_next.imag).sqrt();
    let delta_applied = if next_r >= DOMAIN_R {
        let scale = (DOMAIN_R - 1e-6) / next_r.max(1e-12);
        let c_clamped = Complex::new(c_next.real * scale, c_next.imag * scale);
        Complex::new(c_clamped.real - c.real, c_clamped.imag - c.imag)
    } else {
        delta1
    };

    (
        delta_applied,
        StepDebug {
            mip_level,
            scale_g,
            scale_df,
            scale,
            delta_f_pred,
            wall_applied,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_clamps_delta() {
        let minimap = Minimap::new_with_resolution(32);
        let controller = StepController::new_with_minimap(minimap);
        let mut state = StepState::default();
        let delta_model = Complex::new(1.0, 0.0);
        let result = controller.step(&mut state, delta_model);
        let mag = (result.delta_applied.real * result.delta_applied.real
            + result.delta_applied.imag * result.delta_applied.imag)
            .sqrt();
        assert!(mag <= MAX_STEP_BASE + 1e-6);
    }
}
