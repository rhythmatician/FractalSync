declare module '/wasm/orbit_synth_wasm.js' {
  export default function init(): Promise<void>;
  
  export class OrbitState {
    constructor(
      lobe: number,
      sub_lobe: number,
      theta: number,
      omega: number,
      s: number,
      alpha: number,
      k_residuals: number,
      residual_omega_scale: number
    );
    lobe: number;
    sub_lobe: number;
    theta: number;
    omega: number;
    s: number;
    alpha: number;
    residual_phases(): number[];
    residual_omegas(): number[];
  }
  
  export class OrbitSynthesizer {
    constructor(k_residuals: number, residual_cap: number);
    step(state: OrbitState, dt: number, bandGates?: number[]): any;
  }
}
