/**
 * Deterministic mock of the wasm-orbit module used only in vitest runs.
 *
 * Implements the exact same math as runtime-core's controller for the main
 * cardioid (lobe=1) so tests exercise real synthesis semantics without the
 * wasm binary. Residual phases are fixed (all zero) rather than seeded-RNG,
 * which is fine for the assertions these tests make.
 */

export interface MockState {
  lobe: number;
  sub_lobe: number;
  theta: number;
  omega: number;
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

class MockOrbitState {
  lobe: number;
  sub_lobe: number;
  theta: number;
  omega: number;
  s: number;
  alpha: number;

  constructor(
    lobe: number,
    sub_lobe: number,
    theta: number,
    omega: number,
    s: number,
    alpha: number,
    _kResiduals: number,
    _residualOmegaScale: number,
    _seed?: bigint | null
  ) {
    this.lobe = lobe;
    this.sub_lobe = sub_lobe;
    this.theta = theta;
    this.omega = omega;
    this.s = s;
    this.alpha = alpha;
  }

  set_lobe(v: number) {
    this.lobe = v;
  }
  set_sub_lobe(v: number) {
    this.sub_lobe = v;
  }
  set_s(v: number) {
    this.s = v;
  }
  set_alpha(v: number) {
    this.alpha = v;
  }
  set_omega(v: number) {
    this.omega = v;
  }

  advance(dt: number) {
    this.theta = (this.theta + this.omega * dt) % TWO_PI;
  }

  synthesize(_params: object, bandGates?: Float64Array | null) {
    const carrier = lobePoint(this.lobe, this.sub_lobe, this.theta, this.s);
    if (!bandGates || bandGates.length === 0 || this.alpha === 0.0) {
      return { real: carrier.re, imag: carrier.im };
    }
    const radius = this.lobe === 1 ? 0.25 : 0.25 / (this.lobe * this.lobe);
    let resRe = 0.0;
    let resIm = 0.0;
    for (let k = 0; k < bandGates.length; k++) {
      const amplitude = (this.alpha * (this.s * radius)) / 2.0 ** (k + 1);
      // Zero residual phase keeps the mock deterministic.
      resRe += amplitude * bandGates[k];
      resIm += 0.0;
    }
    return { real: carrier.re + resRe, imag: carrier.im + resIm };
  }
}

class MockResidualParams {
  k_residuals: number;
  residual_cap: number;
  radius_scale: number;
  constructor(k_residuals: number, residual_cap: number, radius_scale: number) {
    this.k_residuals = k_residuals;
    this.residual_cap = residual_cap;
    this.radius_scale = radius_scale;
  }
}

export default {
  OrbitState: MockOrbitState,
  ResidualParams: MockResidualParams,
  step(
    state: MockOrbitState,
    dt: number,
    params: object,
    bandGates?: Float64Array | null
  ) {
    state.advance(dt);
    return state.synthesize(params, bandGates);
  },
  synthesize(state: MockOrbitState, params: object, bandGates?: Float64Array | null) {
    return state.synthesize(params, bandGates);
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
    };
  },
};
