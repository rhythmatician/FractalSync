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
  // DebugSnapshot bookkeeping (issue #111).
  last_controls: { direction: [number, number]; throttle: number; brake: number; grip: number; impulse: number } | null = null;
  last_friction_beta = 0.0;
  last_friction_power = 0.0;
  last_delta_total: number | null = null;
  step_time_seconds = 0.0;
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

  // Authoritative player c (real, imag) and planar velocity (vx, vy) —
  // mirrors the wasm OrbitController's c / velocity getters and setC /
  // setVelocity seed methods so test harnesses and the cockpit recorder
  // can seed a non-default starting point (e.g. "approach from outside
  // M" trajectories).
  get c(): { real: number; imag: number } {
    return { real: this.c_re, imag: this.c_im };
  }
  setC(re: number, im: number): void {
    this.c_re = re;
    this.c_im = im;
  }
  get velocity(): { real: number; imag: number } {
    return { real: this.planar_velocity[0], imag: this.planar_velocity[1] };
  }
  setVelocity(vx: number, vy: number): void {
    this.planar_velocity[0] = vx;
    this.planar_velocity[1] = vy;
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
    // DebugSnapshot bookkeeping (issue #111): raw controls, friction, clock.
    this.last_controls = m
      ? {
          direction: [dirX, dirY] as [number, number],
          throttle,
          brake,
          grip,
          impulse,
        }
      : null;
    this.last_friction_beta = beta;
    this.last_friction_power = -beta * (this.planar_velocity[0] ** 2 + this.planar_velocity[1] ** 2);
    this.last_delta_total = 0.0;
    this.step_time_seconds += dt;
    return { real: this.c_re, imag: this.c_im };
  }

  /**
   * Read-only DebugSnapshot mock (issue #111). Mirrors the wasm seam's wire
   * shape with mock-physics values so the cockpit adapter is testable in
   * vitest. NOT a math mirror: the real values come from Rust.
   */
  debugSnapshot() {
    const c = [this.c_re, this.c_im] as [number, number];
    const v = [this.planar_velocity[0], this.planar_velocity[1]] as [number, number];
    // Mock signed distance: distance from the unit-ish Shore at x=0.25.
    const signedDistance = 0.25 - this.c_re;
    const sigma = Math.log2(0.1 / Math.sqrt(signedDistance * signedDistance + 1e-8));
    const kinetic = 0.5 * (v[0] * v[0] + v[1] * v[1]);
    const potential = sigma;
    const action = this.last_controls
      ? {
          raw: { ...this.last_controls },
          effective: {
            direction: [
              Math.max(-1, Math.min(1, this.last_controls.direction[0])),
              Math.max(-1, Math.min(1, this.last_controls.direction[1])),
            ] as [number, number],
            throttle: Math.max(0, Math.min(1, this.last_controls.throttle)),
            brake: Math.max(0, Math.min(1, this.last_controls.brake)),
            grip: Math.max(0, Math.min(1, this.last_controls.grip)),
            impulse: Math.max(0, Math.min(1, this.last_controls.impulse)),
          },
          driveCovector: [this.last_controls.throttle * 2.0, 0] as [number, number],
          frictionBeta: this.last_friction_beta,
          frictionPower: this.last_friction_power,
        }
      : null;
    return {
      version: 'debug-snapshot/1',
      timeSeconds: this.step_time_seconds,
      action,
      map: { pyramidLoaded: false, shoreProximity: null, minimapWindow: null, extent: null },
      physics: {
        c,
        velocity: v,
        signedDistance,
        realm: signedDistance < 0 ? -1 : signedDistance > 0 ? 1 : 0,
        rho: Math.sqrt(signedDistance * signedDistance + 1e-8),
        sigma,
        sigmaDot: 0,
        scaleGradient: [0, 0] as [number, number],
        metric: [1, 0, 1] as [number, number, number],
        metricSpeed: Math.sqrt(v[0] * v[0] + v[1] * v[1]),
        kinetic,
        potential,
        total: kinetic + potential,
        geodesicAccel: [0, 0] as [number, number],
        potentialForce: [0, 0] as [number, number],
        netAccel: [0, 0] as [number, number],
        derivativeValid: true,
      },
      diagnostics: {
        derivativeStep: 1e-4,
        valid: true,
        lastError: null,
        lastDeltaTotal: this.last_delta_total,
        crestPotential: Math.log2(0.1 / 1e-4),
      },
    };
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
  MotionControls: class MockMotionControls {
    direction_x: number; direction_y: number; throttle: number; brake: number; grip: number; impulse: number;
    constructor(dx: number, dy: number, throttle: number, brake: number, grip: number, impulse: number) {
      this.direction_x = dx; this.direction_y = dy; this.throttle = throttle; this.brake = brake; this.grip = grip; this.impulse = impulse;
    }
  },
  JuliaViewControls: class MockJuliaViewControls {
    zoom_delta: number; rotation_delta: number; hue_delta: number; chroma_delta: number; lightness_delta: number; accent_delta: number; harmony_shift: number;
    constructor(z: number, r: number, h: number, c: number, l: number, a: number, hs: number) {
      this.zoom_delta = z; this.rotation_delta = r; this.hue_delta = h; this.chroma_delta = c; this.lightness_delta = l; this.accent_delta = a; this.harmony_shift = hs;
    }
  },
  JuliaViewState: class MockJuliaViewState {
    zoom = 1.0; rotation = 0.0; harmony_cooldown = 0; harmony_armed = true;
    color = { anchor_hue: 0, chroma: 0.18, lightness: 0.55, harmony: 'analogous', accent_weight: 0.35 };
    apply_controls(c: unknown) {
      const ctrl = c as { zoom_delta?: number; rotation_delta?: number; hue_delta?: number; chroma_delta?: number; lightness_delta?: number; accent_delta?: number; harmony_shift?: number };
      const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
      const wrap01 = (v: number) => ((v % 1) + 1) % 1;
      const wrapAngle = (a: number) => { const tau = 2*Math.PI; let w = ((a % tau)+tau)%tau; if (w>Math.PI) w-=tau; return w; };
      if (this.harmony_cooldown > 0) this.harmony_cooldown -= 1;
      const clamp11 = (v: number) => Math.max(-1, Math.min(1, v));
      const cz = clamp11(ctrl.zoom_delta ?? 0);
      const cr = clamp11(ctrl.rotation_delta ?? 0);
      const ch = clamp11(ctrl.hue_delta ?? 0);
      const cc = clamp11(ctrl.chroma_delta ?? 0);
      const cl = clamp11(ctrl.lightness_delta ?? 0);
      const ca = clamp11(ctrl.accent_delta ?? 0);
      const hs = clamp11(ctrl.harmony_shift ?? 0);
      this.zoom = clamp(this.zoom * Math.exp(cz * 0.05), 0.5, 8.0);
      this.rotation = wrapAngle(this.rotation + cr * 0.08);
      this.color.anchor_hue = wrap01(this.color.anchor_hue + ch * 0.02);
      this.color.chroma = clamp(this.color.chroma + cc * 0.03, 0.0, 0.4);
      this.color.lightness = clamp(this.color.lightness + cl * 0.03, 0.2, 0.9);
      this.color.accent_weight = clamp(this.color.accent_weight + ca * 0.04, 0.0, 1.0);
      const HARMONY_SHIFT_THRESHOLD = 0.6;
      const HARMONY_RELEASE_THRESHOLD = 0.3;
      if (Math.abs(hs) < HARMONY_RELEASE_THRESHOLD) this.harmony_armed = true;
      if (Math.abs(hs) > HARMONY_SHIFT_THRESHOLD && this.harmony_armed && this.harmony_cooldown === 0) {
        const modes = ['monochrome','analogous','opponent'] as const;
        const cur = this.color.harmony as typeof modes[number];
        const idx = modes.indexOf(cur);
        const dir = hs > 0 ? 1 : 2;
        this.color.harmony = modes[(idx + dir) % 3] as unknown as typeof this.color.harmony;
        this.harmony_cooldown = 15;
        this.harmony_armed = false;
      }
    }
  },
  ControlsV2: class MockControlsV2 {
    constructor(public motion: unknown, public view: unknown) {}
  },
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
      controller_version: 'orbit-controller/4',
      feature_version: 'features/2',
      analysis_pipeline_version: 'analysis-pipeline/1',
      controls_version: 'controls/2',
    };
  },
  debugSnapshotMeta() {
    return { version: 'debug-snapshot/1', canonicalDt: 1024 / 48000 };
  },
  ManifoldConfig: class MockManifoldConfig {
    d_ref: number; epsilon: number; lambda_sq: number; kappa: number; mu: number;
    constructor(d: number, e: number, l: number, k: number, mu: number) {
      this.d_ref = d; this.epsilon = e; this.lambda_sq = l; this.kappa = k; this.mu = mu;
    }
  },
  manifold_embedding(re: number, im: number) {
    // Mock embedding with REAL slope near the Shore (x=0.25): sigma rises
    // as the rider approaches it, so per-position height sampling is
    // distinguishable from patch-center height in tests.
    const d = 0.25 - re;
    const rho = Math.sqrt(d * d + 1e-8);
    const sigma = Math.log2(0.1 / rho);
    return [re, im, sigma] as [number, number, number];
  },
  debugTerrainPatch(cx: number, cy: number, half: number, n: number) {
    // Mock terrain: same wire shape as the Rust seam. Heights vary so the
    // cockpit's mesh-building path is exercised in vitest.
    const positions: number[] = [];
    const signed: number[] = [];
    const realm: number[] = [];
    for (let row = 0; row < n; row++) {
      const im = cy + half - 2 * half * (row / (n - 1));
      for (let col = 0; col < n; col++) {
        const re = cx - half + 2 * half * (col / (n - 1));
        const d = 0.25 - re;
        const rho = Math.sqrt(d * d + 1e-8);
        const sigma = Math.log2(0.1 / rho);
        positions.push(re, im, sigma);
        signed.push(d);
        realm.push(d < 0 ? -1 : d > 0 ? 1 : 0);
      }
    }
    return { n, center: [cx, cy] as [number, number], half, positions, signed, realm };
  },
  minimapShoreProximityBatch(re: number[], _im: number[], _level: number) {
    // Mock S field over the canonical extent: a smooth ramp toward the
    // Shore band at x=0.25 (shape only — the real field comes from the
    // Rust pyramid). Points outside the extent clamp to the edge value.
    return Float32Array.from(re.map((x) => Math.max(0, Math.min(1, 1 - Math.abs(x - 0.25) * 2))));
  },
  deepZoomField(re: number[], im: number[]) {
    // Mock deep-zoom DEM: unsigned distance to the boundary (0 inside),
    // same boundary the S ramp encodes. Shape only — the real field comes
    // from the Rust escape-iteration estimator.
    return Float32Array.from(
      re.map((x, i) => {
        const y = im[i];
        const d = Math.abs(x - 0.25) + Math.abs(y) * 0.5;
        return d;
      })
    );
  },
};
