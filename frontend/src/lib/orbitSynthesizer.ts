/**
 * Orbit-based Julia parameter synthesizer.
 *
 * Thin adapter over the canonical Rust implementation (runtime-core) compiled
 * to WebAssembly via wasm-orbit. All synthesis math lives in
 * `runtime-core/src/controller.rs`; this module only translates between the
 * frontend's camelCase control-signal vocabulary and the wasm binding API.
 *
 * The module must be initialized with `initOrbitSynth()` before constructing
 * an `OrbitSynthesizer`. If the wasm bundle fails to load, initialization
 * throws — there is deliberately no JS fallback formula, so training and
 * browser behavior cannot drift apart.
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

// Shape of the wasm-bindgen exports we rely on (subset of orbit_synth_wasm.d.ts).
interface WasmOrbitState {
  readonly lobe: number;
  readonly sub_lobe: number;
  readonly theta: number;
  readonly omega: number;
  readonly s: number;
  readonly alpha: number;
  set_lobe(v: number): void;
  set_sub_lobe(v: number): void;
  set_s(v: number): void;
  set_alpha(v: number): void;
  set_omega(v: number): void;
  advance(dt: number): void;
}

interface WasmModule {
  OrbitState: new (
    lobe: number,
    subLobe: number,
    theta: number,
    omega: number,
    s: number,
    alpha: number,
    kResiduals: number,
    residualOmegaScale: number,
    seed?: bigint | null
  ) => WasmOrbitState;
  ResidualParams: new (
    kResiduals: number,
    residualCap: number,
    radiusScale: number
  ) => object;
  step(
    state: WasmOrbitState,
    dt: number,
    residualParams: object,
    bandGates?: Float64Array | null
  ): { real: number; imag: number };
  synthesize(
    state: WasmOrbitState,
    residualParams: object,
    bandGates?: Float64Array | null
  ): { real: number; imag: number };
  constants(): {
    default_residual_cap: number;
    default_residual_omega_scale: number;
    default_base_omega: number;
    default_orbit_seed: number;
  };
}

let wasm: WasmModule | null = null;

/**
 * Load and initialize the wasm-orbit bundle. Idempotent.
 *
 * In vitest runs (`globalThis.__vitest`), a deterministic mock with identical
 * semantics is substituted so tests do not need the wasm binary.
 */
export async function initOrbitSynth(): Promise<void> {
  if (wasm) return;

  const isTest =
    typeof globalThis !== 'undefined' && (globalThis as any).__vitest;
  if (isTest) {
    const mock = await import('./__tests__/orbitSynthesizer.mock');
    wasm = ((mock as any).default ?? mock) as WasmModule;
    return;
  }

  // Served from public/wasm/ by the dev server and vite build. The ambient
  // module declaration in src/wasm.d.ts types this dynamic import. The
  // indirection through a variable keeps bundlers from trying to resolve the
  // public-path module at build time.
  const wasmUrl = '/wasm/orbit_synth_wasm.js';
  const mod = (await import(/* @vite-ignore */ wasmUrl)) as
    WasmModule & { default?: () => Promise<void> };
  if (typeof mod.default === 'function') {
    await mod.default();
  }
  wasm = mod;
}

/** Test seam: inject a wasm-shaped module directly. */
export function setWasmModuleForTesting(mod: WasmModule | null): void {
  wasm = mod;
}

function requireWasm(): WasmModule {
  if (!wasm) {
    throw new Error(
      'orbitSynthesizer not initialized; call initOrbitSynth() first'
    );
  }
  return wasm;
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

/**
 * Step the orbit forward by dt using the canonical Rust synthesis.
 *
 * Delegates to runtime_core::controller::step: advances carrier + residual
 * phases, then synthesizes c(t) = carrier(lobe, θ) + Σ gated residual
 * epicycles with exponential amplitude decay and magnitude cap.
 */
export class OrbitSynthesizer {
  private kBands: number;
  private params: object;
  private state: WasmOrbitState;

  constructor(kBands: number, initialState?: Partial<OrbitState>) {
    const m = requireWasm();
    this.kBands = kBands;
    const constants = m.constants();
    this.params = new m.ResidualParams(kBands, constants.default_residual_cap, 1.0);
    this.state = new m.OrbitState(
      initialState?.lobe ?? 1,
      initialState?.subLobe ?? 0,
      initialState?.theta ?? 0.0,
      initialState?.omega ?? 1.0,
      initialState?.s ?? 0.5,
      initialState?.alpha ?? 0.5,
      kBands,
      constants.default_residual_omega_scale,
      BigInt(constants.default_orbit_seed)
    );
  }

  /** Current carrier phase (read-only snapshot). */
  get theta(): number {
    return this.state.theta;
  }

  get lobe(): number {
    return this.state.lobe;
  }

  get subLobe(): number {
    return this.state.sub_lobe;
  }

  /**
   * Apply model-predicted control signals to the orbit state.
   */
  applyControls(signals: ControlSignals): void {
    this.state.set_s(signals.sTarget);
    this.state.set_alpha(signals.alpha);
    this.state.set_omega(1.0 * signals.omegaScale);
  }

  /**
   * Switch the active Mandelbrot lobe (section-change handling).
   */
  setLobe(lobe: number, subLobe = 0): void {
    this.state.set_lobe(lobe);
    this.state.set_sub_lobe(subLobe);
  }

  /**
   * Advance the orbit by dt and synthesize c(t).
   */
  step(dt: number, bandGates: number[]): Complex {
    const gates = new Float64Array(Math.min(bandGates.length, this.kBands));
    for (let i = 0; i < gates.length; i++) {
      gates[i] = Math.max(0.0, Math.min(1.0, bandGates[i]));
    }
    const c = requireWasm().step(this.state, dt, this.params, gates);
    return { real: c.real, imag: c.imag };
  }
}
