/**
 * Deterministic mock of the wasm-orbit module used only in vitest runs.
 *
 * Implements the same math as runtime-core's `OrbitController` — the restored
 * May-proven controller (ported verbatim from the pre-fe1087b TS). Mirrors
 * `controller.rs::OrbitController`: c is directly positioned by the model's
 * (s, alpha) via mandelbrotBoundary, plus harmonic residual epicycles.
 */

export interface MockState {
  lobe: number;
  sub_lobe: number;
  s: number;
  alpha: number;
}

const TWO_PI = 2.0 * Math.PI;

function lobePoint(lobe: number, _subLobe: number, theta: number, s: number) {
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

class MockOrbitController {
  theta: number;
  omega: number;
  s: number;
  alpha: number;
  momentum = false;
  drag = 0.9;
  thrust = 0.0;
  energy = 0.0;
  shore_bias = false;
  d_star = 0.5;
  max_step = 0.05;
  manifold_physics = false;
  manifold_drag = 0.1;
  manifold_error: string | null = null;
  planar_velocity: [number, number] = [0, 0];
  // Momentum state (used when momentum is on).
  v_re = 0;
  v_im = 0;
  c_re = 0;
  c_im = 0;

  constructor(s: number, alpha: number, omega: number) {
    this.theta = 0.0;
    this.omega = omega;
    this.s = Math.max(0.01, Math.min(3.0, s));
    this.alpha = Math.max(0.0, Math.min(1.0, alpha));
  }

  apply_controls(s: number, alpha: number) {
    this.s = Math.max(0.01, Math.min(3.0, s));
    this.alpha = Math.max(0.0, Math.min(1.0, alpha));
  }

  // May's exact mandelbrotBoundary(s, alpha).
  private boundary(): { re: number; im: number } {
    const theta = TWO_PI * this.alpha;
    const r = 0.25 * (1.0 - Math.cos(theta));
    const scale = Math.min(this.s, 1.5);
    return {
      re: r * Math.cos(theta / 2.0) * scale,
      im: r * Math.sin(theta / 2.0) * scale,
    };
  }

  stepWithControls(dt: number, motion: unknown) {
    // Destination manifold mock: integrate MotionControls as simple metric-consistent drive.
    // In the real wasm this routes through manifold::integrate_motion_controls — here we
    // simulate a plausible trajectory so stepWithControls does not throw in vitest.
    const m = motion as { direction_x?: number; direction_y?: number; throttle?: number; brake?: number; grip?: number; impulse?: number; direction?: [number, number] } | null;
    let dirX = 0, dirY = 0, throttle = 0, brake = 0, grip = 0.5, impulse = 0;
    if (m) {
      if ('direction_x' in (m as object) && 'direction_y' in (m as object)) {
        dirX = (m as { direction_x: number }).direction_x ?? 0;
        dirY = (m as { direction_y: number }).direction_y ?? 0;
        throttle = (m as { throttle: number }).throttle ?? 0;
        brake = (m as { brake: number }).brake ?? 0;
        grip = (m as { grip: number }).grip ?? 0.5;
        impulse = (m as { impulse: number }).impulse ?? 0;
      } else if ('direction' in (m as object)) {
        const d = (m as { direction: [number, number] }).direction ?? [0, 0];
        dirX = d[0] ?? 0; dirY = d[1] ?? 0;
        throttle = (m as { throttle: number }).throttle ?? 0;
        brake = (m as { brake: number }).brake ?? 0;
        grip = (m as { grip: number }).grip ?? 0.5;
        impulse = (m as { impulse: number }).impulse ?? 0;
      }
    }
    // Simple planar integration: drive as direct acceleration scaled by throttle
    const mag = Math.sqrt(dirX * dirX + dirY * dirY);
    let ax = 0, ay = 0;
    if (mag > 1e-9 && throttle > 1e-9) {
      const nX = dirX / mag, nY = dirY / mag;
      const force = throttle * 2.0;
      ax += force * nX;
      ay += force * nY;
    }
    // Drag as simple linear damping (PSD)
    const beta = 0.05 + grip * 0.15 + brake * 1.0;
    ax -= beta * this.planar_velocity[0];
    ay -= beta * this.planar_velocity[1];
    // Geodesic + potential omitted in mock — direct integration.
    this.planar_velocity[0] += ax * dt;
    this.planar_velocity[1] += ay * dt;
    if (impulse > 1e-9 && mag > 1e-9) {
      this.planar_velocity[0] += (dirX / mag) * impulse * 0.5;
      this.planar_velocity[1] += (dirY / mag) * impulse * 0.5;
    }
    this.c_re += this.planar_velocity[0] * dt;
    this.c_im += this.planar_velocity[1] * dt;
    this.theta = (this.theta + this.omega * dt) % TWO_PI;
    return { real: this.c_re, imag: this.c_im };
  }

  step(dt: number, _h = 0.0, bandGates?: Float64Array | null) {
    this.theta = (this.theta + this.omega * dt) % TWO_PI;
    const base = this.boundary();
    let re = base.re;
    let im = base.im;
    if (bandGates) {
      for (let k = 0; k < bandGates.length; k++) {
        const gate = Math.max(0.0, Math.min(1.0, bandGates[k]));
        const freq = k + 2;
        const phase = freq * this.theta;
        re += gate * 0.05 * Math.cos(phase);
        im += gate * 0.05 * Math.sin(phase);
      }
    }
    if (!this.momentum && !this.shore_bias) {
      // Baseline path: c IS the target (bit-identical to May).
      this.c_re = re;
      this.c_im = im;
      return { real: re, imag: im };
    }
    // Momentum path: pull toward target is an acceleration.
    const accelGain = 2.0 * dt;
    let aRe = (re - this.c_re) * accelGain;
    let aIm = (im - this.c_im) * accelGain;
    if (this.thrust > 0) {
      const dx = re - this.c_re;
      const dy = im - this.c_im;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > 1e-9) {
        aRe += this.thrust * (-dy / d);
        aIm += this.thrust * (dx / d);
      }
    }
    this.v_re = this.v_re * this.drag + aRe;
    this.v_im = this.v_im * this.drag + aIm;
    this.c_re += this.v_re * dt;
    this.c_im += this.v_im * dt;
    return { real: this.c_re, imag: this.c_im };
  }
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

  step(dt: number, _h: number, bandGates?: Float64Array | null) {
    // Target point on the boundary the model wants to reach.
    const target = lobePoint(this.lobe, this.sub_lobe, this.alpha * TWO_PI, this.s);
    // Pull toward the target is an ACCELERATION (mirrors controller.rs).
    // Gain includes dt so `a` is per-second; steady-state speed = a/(1-drag).
    const accelGain = Math.max(0.1, Math.min(10.0, this.omega_scale)) * 2.0 * dt;
    let aRe = (target.re - this.c_re) * accelGain;
    let aIm = (target.im - this.c_im) * accelGain;
    // Gravity valley (orbit-controller/3): restoring pull toward the origin.
    aRe -= 0.01 * this.c_re;
    aIm -= 0.01 * this.c_im;
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
  OrbitController: MockOrbitController,
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
