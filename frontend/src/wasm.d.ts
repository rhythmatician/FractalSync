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
      seed?: bigint | null
    );
    lobe: number;
    sub_lobe: number;
    theta: number;
    omega: number;
    s: number;
    alpha: number;
    set_lobe(v: number): void;
    set_sub_lobe(v: number): void;
    set_s(v: number): void;
    set_alpha(v: number): void;
    set_omega(v: number): void;
    advance(dt: number): void;
  }

  export class ResidualParams {
    constructor(k_residuals: number, residual_cap: number, radius_scale: number);
  }

  export class PlayerState {
    constructor(lobe: number, sub_lobe: number, s: number, alpha: number);
    c_re: number;
    c_im: number;
    speed: number;
    set_level(v: number): void;
    set_d_star(v: number): void;
    set_max_step(v: number): void;
    apply_controls(s: number, alpha: number, omega_scale: number): void;
    set_lobe(lobe: number, sub_lobe: number): void;
    step(
      dt: number,
      h: number,
      band_gates?: Float64Array | null
    ): { real: number; imag: number };
  }

  /** May-proven orbit controller (restored baseline). */
  export class OrbitController {
    constructor(s: number, alpha: number, omega: number);
    theta: number;
    apply_controls(s: number, alpha: number): void;
    step(
      dt: number,
      band_gates?: Float64Array | null
    ): { real: number; imag: number };
  }

  export function step(
    state: OrbitState,
    dt: number,
    residual_params: ResidualParams,
    band_gates?: Float64Array | null
  ): { real: number; imag: number };

  export function synthesize(
    state: OrbitState,
    residual_params: ResidualParams,
    band_gates?: Float64Array | null
  ): { real: number; imag: number };

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
}
