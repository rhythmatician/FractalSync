/**
 * Orbit synthesizer for generating Julia parameter c(t) from control signals.
 * 
 * This is a TypeScript port of backend/src/orbit_synth.py to maintain DRY principles
 * and ensure the frontend and backend use the same synthesis logic.
 * 
 * Formula: c(t) = c_carrier(θ) + c_residual(phases)
 */

import { MandelbrotGeometry } from './mandelbrotGeometry';

export interface OrbitState {
  lobe: number;           // Period number (1=cardioid, 2=period-2, etc.)
  subLobe: number;        // Sub-lobe index
  theta: number;          // Carrier angle (radians)
  omega: number;          // Angular velocity (rad/s)
  s: number;              // Radius scaling factor
  alpha: number;          // Residual amplitude (relative to lobe radius)
  residualPhases: number[];   // Phases for k residual circles (radians)
  residualOmegas: number[];   // Angular velocities for residuals (rad/s)
}

export interface ControlSignals {
  sTarget: number;        // [0.2, 3.0] Radius scaling
  alpha: number;          // [0, 1] Residual amplitude
  omegaScale: number;     // [0.1, 5.0] Angular velocity scale
  bandGates: number[];    // [0, 1]^k Per-band residual gates
}

export class OrbitSynthesizer {
  private kResiduals: number;
  private residualCap: number;

  constructor(kResiduals: number = 6, residualCap: number = 0.5) {
    this.kResiduals = kResiduals;
    this.residualCap = residualCap;
  }

  /**
   * Synthesize Julia parameter c from orbit state.
   */
  synthesize(state: OrbitState, bandGates?: number[]): { real: number; imag: number } {
    // Carrier: deterministic orbit point
    const carrier = MandelbrotGeometry.lobePointAtAngle(
      state.lobe,
      state.theta,
      state.s,
      state.subLobe
    );

    // Residual: epicyclic texture
    if (state.alpha === 0.0) {
      return carrier;
    }

    // Get lobe radius for scaling
    const radius = this.getLobeRadius(state.lobe, state.subLobe);

    // Sum residual circles
    let residualReal = 0;
    let residualImag = 0;

    for (let k = 0; k < this.kResiduals; k++) {
      // Amplitude decreases as 1/k²
      const amplitude = (state.alpha * (state.s * radius)) / Math.pow(k + 1, 2);

      // Band gate (default to 1.0 if not provided)
      const gK = bandGates ? bandGates[k] : 1.0;

      // Phasor: amplitude * g_k * exp(i * phase)
      const phase = state.residualPhases[k];
      residualReal += amplitude * gK * Math.cos(phase);
      residualImag += amplitude * gK * Math.sin(phase);
    }

    // Cap residual magnitude
    const mag = Math.sqrt(residualReal * residualReal + residualImag * residualImag);
    const cap = this.residualCap * radius;
    if (mag > cap) {
      const scale = cap / mag;
      residualReal *= scale;
      residualImag *= scale;
    }

    return {
      real: carrier.real + residualReal,
      imag: carrier.imag + residualImag
    };
  }

  /**
   * Get lobe radius for residual scaling.
   */
  private getLobeRadius(lobe: number, subLobe: number): number {
    if (lobe === 1) {
      // Cardioid: use reference scale
      return 0.25;
    } else {
      return MandelbrotGeometry.periodNBulbRadius(lobe, subLobe);
    }
  }

  /**
   * Step forward in time and synthesize c(t).
   */
  step(
    state: OrbitState,
    dt: number,
    bandGates?: number[]
  ): { c: { real: number; imag: number }; newState: OrbitState } {
    // Synthesize current c
    const c = this.synthesize(state, bandGates);

    // Advance state
    const newTheta = (state.theta + state.omega * dt) % (2 * Math.PI);
    const newResidualPhases = state.residualPhases.map(
      (phase, i) => (phase + state.residualOmegas[i] * dt) % (2 * Math.PI)
    );

    const newState: OrbitState = {
      lobe: state.lobe,
      subLobe: state.subLobe,
      theta: newTheta,
      omega: state.omega,
      s: state.s,
      alpha: state.alpha,
      residualPhases: newResidualPhases,
      residualOmegas: state.residualOmegas
    };

    return { c, newState };
  }

  /**
   * Compute velocity dc/dt at current state (analytic derivative).
   */
  computeVelocity(state: OrbitState, bandGates?: number[]): { real: number; imag: number } {
    // Carrier velocity: ω * d/dθ[lobe_point_at_angle]
    const tangent = MandelbrotGeometry.lobeTangentAtAngle(
      state.lobe,
      state.theta,
      state.s,
      state.subLobe
    );
    const vCarrierReal = state.omega * tangent.real;
    const vCarrierImag = state.omega * tangent.imag;

    // Residual velocity
    if (state.alpha === 0.0) {
      return { real: vCarrierReal, imag: vCarrierImag };
    }

    const radius = this.getLobeRadius(state.lobe, state.subLobe);
    let vResidualReal = 0;
    let vResidualImag = 0;

    for (let k = 0; k < this.kResiduals; k++) {
      const amplitude = (state.alpha * (state.s * radius)) / Math.pow(k + 1, 2);
      const gK = bandGates ? bandGates[k] : 1.0;

      // Velocity: derivative of exp(i*ϕ) is i*ω*exp(i*ϕ)
      const phase = state.residualPhases[k];
      const omegaK = state.residualOmegas[k];
      // i * ω * exp(i*ϕ) = i * ω * (cos(ϕ) + i*sin(ϕ)) = -ω*sin(ϕ) + i*ω*cos(ϕ)
      vResidualReal += amplitude * gK * omegaK * (-Math.sin(phase));
      vResidualImag += amplitude * gK * omegaK * Math.cos(phase);
    }

    return {
      real: vCarrierReal + vResidualReal,
      imag: vCarrierImag + vResidualImag
    };
  }
}

/**
 * Create initial orbit state with sensible defaults.
 */
export function createInitialState(options: {
  lobe?: number;
  subLobe?: number;
  theta?: number;
  omega?: number;
  s?: number;
  alpha?: number;
  kResiduals?: number;
  residualOmegaScale?: number;
}): OrbitState {
  const {
    lobe = 1,
    subLobe = 0,
    theta = 0.0,
    omega = 1.0,
    s = 1.02,
    alpha = 0.3,
    kResiduals = 6,
    residualOmegaScale = 2.0
  } = options;

  // Random initial phases
  const residualPhases = Array.from(
    { length: kResiduals },
    () => Math.random() * 2 * Math.PI
  );

  // Residual omegas: scale with k (higher harmonics rotate faster)
  const residualOmegas = Array.from(
    { length: kResiduals },
    (_, k) => residualOmegaScale * omega * (k + 1)
  );

  return {
    lobe,
    subLobe,
    theta,
    omega,
    s,
    alpha,
    residualPhases,
    residualOmegas
  };
}
