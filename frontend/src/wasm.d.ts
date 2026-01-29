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
      residual_omega_scale: number,
      seed?: number | null
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

  export class Minimap {
    constructor();
    contextFeatures(
      cReal: number,
      cImag: number,
      prevDeltaReal: number,
      prevDeltaImag: number,
      mipLevel: number
    ): {
      feature_vector: number[];
      sensitivity: number;
      nu_norm: number;
      membership: boolean;
      grad_re: number;
      grad_im: number;
      mip_level: number;
    };
  }

  export class StepController {
    constructor();
    contextFeatures(
      cReal: number,
      cImag: number,
      prevDeltaReal: number,
      prevDeltaImag: number,
      mipLevel: number
    ): {
      feature_vector: number[];
      sensitivity: number;
      nu_norm: number;
      membership: boolean;
      grad_re: number;
      grad_im: number;
      mip_level: number;
    };
    applyStep(
      cReal: number,
      cImag: number,
      deltaReal: number,
      deltaImag: number,
      prevDeltaReal: number,
      prevDeltaImag: number
    ): {
      delta_real: number;
      delta_imag: number;
      c_next_real: number;
      c_next_imag: number;
      sensitivity: number;
      nu_norm: number;
      membership: boolean;
      grad_re: number;
      grad_im: number;
      debug: {
        mip_level: number;
        scale_g: number;
        scale_df: number;
        scale: number;
        wall_applied: boolean;
      };
    };
  }

  export function mipForDelta(deltaReal: number, deltaImag: number): number;
  export function controllerContextLen(): number;
}
