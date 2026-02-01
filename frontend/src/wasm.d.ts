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

  export interface StepContext {
    c_real: number;
    c_imag: number;
    prev_delta_real: number;
    prev_delta_imag: number;
    nu_norm: number;
    membership: boolean;
    grad_re: number;
    grad_im: number;
    sensitivity: number;
    patch: number[];
    mip_level: number;
    feature_vec: number[];
  }

  export interface StepDebug {
    mip_level: number;
    scale_g: number;
    scale_df: number;
    scale: number;
    delta_f_pred: number;
    wall_applied: boolean;
  }

  export interface StepResult {
    c_real: number;
    c_imag: number;
    delta_real: number;
    delta_imag: number;
    debug: StepDebug;
    context: StepContext;
  }

  export class StepState {
    constructor(c_real: number, c_imag: number, prev_delta_real: number, prev_delta_imag: number);
    c_real: number;
    c_imag: number;
    prev_delta_real: number;
    prev_delta_imag: number;
  }

  export class StepController {
    constructor();
    context(state: StepState): StepContext;
    step(state: StepState, delta_real: number, delta_imag: number): StepResult;
  }
}
