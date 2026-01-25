declare module '/wasm/orbit_synth_wasm.js' {
  export default function init(): Promise<void>;

  export function constants(): {
    sample_rate: number;
    hop_length: number;
    n_fft: number;
    window_frames: number;
    default_k_residuals: number;
    default_residual_cap: number;
    default_residual_omega_scale: number;
    default_base_omega: number;
    default_orbit_seed: number;
  };

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
      seed?: number
    );
    static newDefault(seed: number): OrbitState;
    lobe: number;
    sub_lobe: number;
    theta: number;
    omega: number;
    s: number;
    alpha: number;
    residual_phases(): number[];
    residual_omegas(): number[];
    advance(dt: number): void;
    synthesize(residualParams: ResidualParams, bandGates?: number[]): any;
    step(dt: number, residualParams: ResidualParams, bandGates?: number[]): any;
    step_advanced(
      dt: number,
      residualParams: ResidualParams,
      bandGates: number[] | undefined,
      h: number,
      d_star?: number,
      max_step?: number,
      distanceField?: DistanceField | null
    ): any;
  }

  export class ResidualParams {
    constructor(k_residuals: number, residual_cap: number, radius_scale: number);
  }

  export class DistanceField {
    constructor(
      field: Float32Array,
      resolution: number,
      realRange: [number, number],
      imagRange: [number, number],
      maxDistance: number,
      slowdownThreshold: number
    );
    lookup(real: number, imag: number): number;
    sample_bilinear(real: number, imag: number): number;
    gradient(real: number, imag: number): number[];
    get_velocity_scale(real: number, imag: number): number;
  }
}
declare module 'wasm/orbit_synth_wasm.js' {
  export * from '/wasm/orbit_synth_wasm.js';
}