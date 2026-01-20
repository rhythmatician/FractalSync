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
   * Get center and radius for a specific lobe
   */
  private getLobeParams(lobe: number, subLobe: number): { center: Complex; radius: number } {
    // Hardcoded centers for common periods
    const lobeData: Record<number, { center: Complex; radius: number }[]> = {
      1: [{ center: { real: 0.0, imag: 0.0 }, radius: 0.25 }],
      2: [{ center: { real: -1.0, imag: 0.0 }, radius: 0.25 }],
      3: [
        { center: { real: -0.125, imag: 0.649519 }, radius: 0.0943 },
        { center: { real: -0.125, imag: -0.649519 }, radius: 0.0943 }
      ],
      4: [{ center: { real: 0.25, imag: 0.0 }, radius: 0.04 }]
    };

    const lobeArray = lobeData[lobe];
    if (!lobeArray) {
      // Fallback to lobe 1
      return lobeData[1][0];
    }

    const index = Math.min(subLobe, lobeArray.length - 1);
    return lobeArray[index];
  }

  /**
   * Map (lobe, s, alpha) to a point on/near the Mandelbrot boundary
   */
  private mandelbrotBoundary(lobe: number, subLobe: number, s: number, alpha: number): Complex {
    // Clamp inputs
    s = Math.max(0.01, Math.min(3.0, s));
    alpha = Math.max(0.0, Math.min(1.0, alpha));

    if (lobe === 1) {
      // Main cardioid: c = r * e^(iθ) where r = 0.25 * (1 - cos(2πα))
      const theta = 2.0 * Math.PI * alpha;
      const r = 0.25 * (1.0 - Math.cos(theta));
      
      let real = r * Math.cos(theta / 2.0);
      let imag = r * Math.sin(theta / 2.0);

      // Scale by s to move away from boundary
      const scale = Math.min(s, 1.5);
      real *= scale;
      imag *= scale;

      return { real, imag };
    } else {
      // Period-n bulbs: circular orbits around bulb center
      const { center, radius } = this.getLobeParams(lobe, subLobe);
      const theta = 2.0 * Math.PI * alpha;
      
      return {
        real: center.real + s * radius * Math.cos(theta),
        imag: center.imag + s * radius * Math.sin(theta)
      };
    }
  }

  /**
   * Step the orbit forward by dt, applying residual modulation
   */
  step(state: OrbitState, dt: number, bandGates: number[]): { c: Complex; newState: OrbitState } {
    // Update theta (orbit phase)
    const newTheta = (state.theta + state.omega * dt) % (2.0 * Math.PI);

    // Get base position from Mandelbrot boundary (now respects lobe!)
    const cBase = this.mandelbrotBoundary(state.lobe, state.subLobe, state.s, state.alpha);

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
