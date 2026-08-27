/**
 * Deterministic mock of the wasm-orbit module used only in vitest runs.
 *
 * Implements the same math as runtime-core's `PlayerState` c-space integrator
 * for the main cardioid (lobe=1) so tests exercise real synthesis semantics
 * without the wasm binary. Mirrors `controller.rs::PlayerState`: c is held as
 * persistent state and moved toward the model-driven target point on the
 * boundary (no closed-loop carrier).
 */

export interface MockState {
  lobe: number;
  sub_lobe: number;
  s: number;
  alpha: number;
}

const TWO_PI = 2.0 * Math.PI;

function lobePoint(lobe: number, subLobe: number, theta: number, s: number) {
  if (lobe === 1) {
    // c = mu/2 - mu^2/4 with mu = s * e^{i theta}
    const muRe = s * Math.cos(theta);
    const muIm = s * Math.sin(theta);
    const mu2Re = muRe * muRe - muIm * muIm;
    const mu2Im = 2.0 * muRe * muIm;
    return { re: 0.5 * muRe - 0.25 * mu2Re, im: 0.5 * muIm - 0.25 * mu2Im };
  }
  // Circular bulbs (simplified, mirrors runtime-core geometry fallback).
  const radius = 0.25 / (lobe * lobe);
  const centre = { re: -1.75 + (lobe - 2) * 0.5, im: 0 };
  return {
    re: centre.re + s * radius * Math.cos(theta),
    im: centre.im + s * radius * Math.sin(theta),
  };
}

class MockPlayerState {
  lobe: number;
  sub_lobe: number;
  s: number;
  alpha: number;
  omega_scale: number;
  c_re: number;
  c_im: number;
  // Momentum state: pull is an acceleration; friction bleeds velocity.
  v_re = 0;
  v_im = 0;
  drag = 0.9;
  level: number;
  d_star: number;
  max_step: number;

  constructor(lobe: number, sub_lobe: number, s: number, alpha: number) {
    this.lobe = lobe;
    this.sub_lobe = sub_lobe;
    this.s = s;
    this.alpha = alpha;
    this.omega_scale = 1.0;
    this.level = 0;
    this.d_star = 0.5;
    this.max_step = 0.05;
    // Start on the boundary at (s, alpha).
    const start = lobePoint(lobe, sub_lobe, alpha * TWO_PI, s);
    this.c_re = start.re;
    this.c_im = start.im;
  }

  get speed() {
    return Math.sqrt(this.v_re * this.v_re + this.v_im * this.v_im);
  }

  set_level(v: number) {
    this.level = v;
  }
  set_d_star(v: number) {
    this.d_star = v;
  }
  set_max_step(v: number) {
    this.max_step = v;
  }

  apply_controls(s: number, alpha: number, omega_scale: number) {
    this.s = s;
    this.alpha = alpha;
    this.omega_scale = omega_scale;
  }

  set_lobe(lobe: number, sub_lobe: number) {
    this.lobe = lobe;
    this.sub_lobe = sub_lobe;
  }

  step(dt: number, h: number, bandGates?: Float64Array | null) {
    // Target point on the boundary the model wants to reach.
    const target = lobePoint(this.lobe, this.sub_lobe, this.alpha * TWO_PI, this.s);
    // Pull toward the target is an ACCELERATION (mirrors controller.rs).
    // Gain includes dt so `a` is per-second; steady-state speed = a/(1-drag).
    const accelGain = Math.max(0.1, Math.min(10.0, this.omega_scale)) * 2.0 * dt;
    let aRe = (target.re - this.c_re) * accelGain;
    let aIm = (target.im - this.c_im) * accelGain;
    // Optional residual jitter from band gates (impulse per frame).
    if (bandGates) {
      for (let k = 0; k < bandGates.length; k++) {
        const amp = (0.004 * Math.max(0.0, Math.min(1.0, bandGates[k]))) / (k + 1);
        const phase = this.alpha * TWO_PI * (k + 2);
        aRe += amp * Math.cos(phase);
        aIm += amp * Math.sin(phase);
      }
    }
    // Integrate: v = drag*v + a; c += v*dt.
    this.v_re = this.v_re * this.drag + aRe;
    this.v_im = this.v_im * this.drag + aIm;
    let uRe = this.v_re * dt;
    let uIm = this.v_im * dt;
    // Clamp to max_step (no pyramid in mock → plain clamped motion).
    const mag = Math.sqrt(uRe * uRe + uIm * uIm);
    if (mag > this.max_step && mag > 0) {
      const scale = this.max_step / mag;
      uRe *= scale;
      uIm *= scale;
    }
    this.c_re += uRe;
    this.c_im += uIm;
    return { real: this.c_re, imag: this.c_im };
  }
}

export default {
  PlayerState: MockPlayerState,
  constants() {
    return {
      sample_rate: 48000,
      hop_length: 1024,
      n_fft: 4096,
      window_frames: 10,
      default_k_residuals: 6,
      default_residual_cap: 0.5,
      default_residual_omega_scale: 2.0,
      default_base_omega: 1.0,
      default_orbit_seed: 1337,
    };
  },
};
