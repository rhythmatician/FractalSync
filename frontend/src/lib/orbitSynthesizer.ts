/**
 * Orbit-based Julia parameter synthesizer (TypeScript port)
 * Converts control signals (s, alpha, omega_scale, band_gates) to c(t)
 */

export interface ControlSignals {
  sTarget: number;
  alpha: number;
  omegaScale: number;
  bandGates: number[];
}

export interface OrbitState {
  lobe: number;
  subLobe: number;
  s: number;
  alpha: number;
  omega: number;
  theta: number;
}

export interface Complex {
  real: number;
  imag: number;
}

export interface OrbitConfig {
  kResiduals: number;
}

export function createInitialState(_config: OrbitConfig): OrbitState {
  return {
    lobe: 1,
    subLobe: 0,
    s: 0.5,
    alpha: 0.5,
    omega: 1.0,
    theta: 0.0
  };
}

export class OrbitSynthesizer {
  private kBands: number;
  private residualFreqs: number[];

  constructor(kBands: number) {
    this.kBands = kBands;
    // Residual frequencies: harmonics of base orbit
    this.residualFreqs = Array.from({ length: kBands }, (_, i) => (i + 2) * 1.0);
  }

  /**
   * Map (s, alpha) to a point on/near the Mandelbrot boundary
   */
  private mandelbrotBoundary(s: number, alpha: number): Complex {
    // Clamp inputs
    s = Math.max(0.01, Math.min(3.0, s));
    alpha = Math.max(0.0, Math.min(1.0, alpha));

    // Main cardioid: c = r * e^(iθ) where r = 0.25 * (1 - cos(2πα))
    const theta = 2.0 * Math.PI * alpha;
    const r = 0.25 * (1.0 - Math.cos(theta));
    
    let real = r * Math.cos(theta / 2.0);
    let imag = r * Math.sin(theta / 2.0);

    // Scale by s to move away from boundary
    const scale = Math.min(s, 1.5); // Cap at 1.5 to avoid escaping too far
    real *= scale;
    imag *= scale;

    return { real, imag };
  }

  /**
   * Step the orbit forward by dt, applying residual modulation
   */
  step(state: OrbitState, dt: number, bandGates: number[]): { c: Complex; newState: OrbitState } {
    // Update theta (orbit phase)
    const newTheta = (state.theta + state.omega * dt) % (2.0 * Math.PI);

    // Get base position from Mandelbrot boundary
    const cBase = this.mandelbrotBoundary(state.s, state.alpha);

    // Apply residual modulation
    let residualReal = 0.0;
    let residualImag = 0.0;

    const numGates = Math.min(bandGates.length, this.kBands);
    for (let k = 0; k < numGates; k++) {
      const gate = Math.max(0.0, Math.min(1.0, bandGates[k]));
      const freq = this.residualFreqs[k];
      const phase = freq * newTheta;
      
      // Add harmonic component weighted by gate
      residualReal += gate * 0.05 * Math.cos(phase);
      residualImag += gate * 0.05 * Math.sin(phase);
    }

    // Combine base + residuals
    const c: Complex = {
      real: cBase.real + residualReal,
      imag: cBase.imag + residualImag
    };

    const newState: OrbitState = {
      lobe: state.lobe,
      subLobe: state.subLobe,
      s: state.s,
      alpha: state.alpha,
      omega: state.omega,
      theta: newTheta
    };

    return { c, newState };
  }
}
