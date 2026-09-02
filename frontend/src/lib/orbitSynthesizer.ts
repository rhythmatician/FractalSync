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

interface WasmMotionControls {
  direction_x: number;
  direction_y: number;
  throttle: number;
  brake: number;
  grip: number;
  impulse: number;
}

interface WasmOrbitController {
  readonly theta: number;
  apply_controls(s: number, alpha: number): void;
  set_momentum(on: boolean): void;
  set_drag(drag: number): void;
  set_thrust(thrust: number): void;
  set_energy(energy: number): void;
  set_shore_bias(on: boolean): void;
  set_d_star(d_star: number): void;
  set_max_step(max_step: number): void;
  set_manifold_physics?(on: boolean): void;
  set_manifold_config?(c: unknown): void;
  set_manifold_drag?(d: number): void;
  step(dt: number, h: number, bandGates?: Float64Array | null): { real: number; imag: number };
  stepWithControls?(dt: number, motion: unknown): { real: number; imag: number };
  step_with_controls?(dt: number, motion: unknown): { real: number; imag: number };
}

interface WasmModule {
  OrbitController: new (
    s: number,
    alpha: number,
    omega: number
  ) => WasmOrbitController;
  constants(): {
    default_residual_cap: number;
    default_residual_omega_scale: number;
    default_base_omega: number;
    default_orbit_seed: number;
    controller_version?: string;
    feature_version?: string;
    analysis_pipeline_version?: string;
    sample_rate?: number;
    hop_length?: number;
    n_fft?: number;
    window_frames?: number;
  };
  /** Canonical sample-clock timebase (issue #91), constructed per stream. */
  AnalysisTimebase?: new () => import('./analysisTimebase').WasmAnalysisTimebase;
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
  PlayerState: new (
    lobe: number,
    subLobe: number,
    s: number,
    alpha: number
  ) => WasmPlayerState;
  ResidualParams: new (
    kResiduals: number,
    residualCap: number,
    radiusScale: number
  ) => object;
  JuliaViewState?: new (
    zoom: number,
    rotation: number,
    color: unknown,
    harmony_cooldown: number,
    harmony_armed: boolean
  ) => any;
  JuliaViewControls?: new (
    zoom_delta: number,
    rotation_delta: number,
    hue_delta: number,
    chroma_delta: number,
    lightness_delta: number,
    accent_delta: number,
    harmony_shift: number
  ) => any;
  ColorIntent?: new (
    anchor_hue: number,
    chroma: number,
    lightness: number,
    harmony: string,
    accent_weight: number
  ) => any;
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

// Shape of the wasm-bindgen PlayerState export (c-space integrator).
interface WasmPlayerState {
  readonly c_re: number;
  readonly c_im: number;
  readonly speed: number;
  set_level(v: number): void;
  set_d_star(v: number): void;
  set_max_step(v: number): void;
  apply_controls(s: number, alpha: number, omega_scale: number): void;
  set_lobe(lobe: number, sub_lobe: number): void;
  step(
    dt: number,
    h: number,
    bandGates?: Float64Array | null
  ): { real: number; imag: number };
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

  // Served from public/wasm/ by the dev server and vite build. Build the URL
  // at runtime (origin-relative) so Vite treats it as a truly dynamic import
  // and serves the /public asset as-is instead of trying to transform it.
  const wasmUrl = new URL('/wasm/orbit_synth_wasm.js', globalThis.location.origin).href;
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

/**
 * The runtime's controller contract version (from the Rust constant via
 * wasm constants). Returns 'unknown' if the wasm build predates the field.
 */
export function getControllerVersion(): string {
  if (!wasm) return 'unknown';
  try {
    return wasm.constants().controller_version ?? 'unknown';
  } catch {
    return 'unknown';
  }
}

/**
 * The runtime's feature-extraction contract version (from the Rust constant
 * via wasm constants). Returns 'unknown' if the wasm build predates the field.
 */
export function getFeatureVersion(): string {
  if (!wasm) return 'unknown';
  try {
    return wasm.constants().feature_version ?? 'unknown';
  } catch {
    return 'unknown';
  }
}

/**
 * The runtime's analysis-pipeline contract version (from the Rust constant
 * via wasm constants). Versions HOW audio reaches the extractor (resampling
 * ownership, hop scheduling, epoch semantics) — distinct from the feature
 * FORMULA version. Returns 'unknown' if the wasm build predates the field.
 */
export function getAnalysisPipelineVersion(): string {
  if (!wasm) return 'unknown';
  try {
    return wasm.constants().analysis_pipeline_version ?? 'unknown';
  } catch {
    return 'unknown';
  }
}

/**
 * The runtime's Controls v2 contract version (from the Rust constant via
 * wasm constants). Returns 'unknown' if the wasm build predates the field.
 */
export function getControlsVersion(): string {
  if (!wasm) return 'unknown';
  try {
    return (wasm.constants() as Record<string, unknown>).controls_version as string ?? 'unknown';
  } catch {
    return 'unknown';
  }
}

export interface JuliaViewStateHandle {
  readonly zoom: number;
  readonly rotation: number;
  readonly color: { anchor_hue: number; chroma: number; lightness: number; harmony: string; accent_weight: number };
  readonly harmony_cooldown: number;
  readonly harmony_armed: boolean;
  apply_controls(controls: JuliaViewControlsHandle): void;
}

export interface JuliaViewControlsHandle {
  readonly zoom_delta: number;
  readonly rotation_delta: number;
  readonly hue_delta: number;
  readonly chroma_delta: number;
  readonly lightness_delta: number;
  readonly accent_delta: number;
  readonly harmony_shift: number;
}

type JuliaWasmExtension = {
  JuliaViewState: new (zoom: number, rotation: number, color: unknown, harmony_cooldown: number, harmony_armed: boolean) => JuliaViewStateHandle;
  JuliaViewControls: new (zoom_delta: number, rotation_delta: number, hue_delta: number, chroma_delta: number, lightness_delta: number, accent_delta: number, harmony_shift: number) => JuliaViewControlsHandle;
};

export function getWasmModule(): WasmModule & Partial<JuliaWasmExtension> {
  return requireWasm() as WasmModule & Partial<JuliaWasmExtension>;
}

export function createJuliaViewState(): JuliaViewStateHandle {
  const m = getWasmModule();
  if (typeof m.JuliaViewState !== 'function') {
    throw new Error('[orbitSynthesizer] wasm build has no JuliaViewState binding; rebuild wasm-orbit');
  }
  return new m.JuliaViewState(1.0, 0.0, null, 0, true);
}

export function createJuliaViewControls(view: {
  zoom_delta: number;
  rotation_delta: number;
  hue_delta: number;
  chroma_delta: number;
  lightness_delta: number;
  accent_delta: number;
  harmony_shift: number;
}): JuliaViewControlsHandle {
  const m = getWasmModule();
  if (typeof m.JuliaViewControls !== 'function') {
    throw new Error('[orbitSynthesizer] wasm build has no JuliaViewControls binding; rebuild wasm-orbit');
  }
  return new m.JuliaViewControls(
    view.zoom_delta,
    view.rotation_delta,
    view.hue_delta,
    view.chroma_delta,
    view.lightness_delta,
    view.accent_delta,
    view.harmony_shift
  );
}

/**
 * The wasm module's raw constants (runtime-core authority for SAMPLE_RATE,
 * HOP_LENGTH, N_FFT, WINDOW_FRAMES). Returns null before initOrbitSynth().
 * Consumers must handle null — there is no TypeScript fallback authority.
 */
export function getWasmConstants(): ReturnType<WasmModule['constants']> | null {
  if (!wasm) return null;
  try {
    return wasm.constants();
  } catch {
    return null;
  }
}

/**
 * Construct a fresh Rust `AnalysisTimebase` (issue #91) from the wasm
 * module. Throws if the wasm build predates the timebase binding — there is
 * no JS fallback, so the canonical clock cannot silently drift.
 */
export function createAnalysisTimebase(): import('./analysisTimebase').WasmAnalysisTimebase {
  const m = requireWasm();
  if (typeof m.AnalysisTimebase !== 'function') {
    throw new Error(
      '[orbitSynthesizer] wasm build has no AnalysisTimebase binding; rebuild wasm-orbit'
    );
  }
  return new m.AnalysisTimebase();
}

/**
 * Load the mip pyramid (minimaps) into the wasm runtime so the Player's
 * contour-biased stepper can follow the Shore. Fetches the baked artifacts
 * from the backend API and calls the wasm `set_mip_pyramid` binding.
 *
 * Best-effort: if the artifacts are unavailable, the Player falls back to
 * plain clamped motion (still audio-driven, no closed loop).
 */
export async function loadMipPyramid(): Promise<boolean> {
  const m = requireWasm();
  if (typeof (m as any).set_mip_pyramid !== 'function') {
    console.warn('[orbitSynthesizer] wasm build has no set_mip_pyramid; minimap disabled');
    return false;
  }
  try {
    const metaResp = await fetch('/api/minimap/meta', { credentials: 'same-origin' });
    if (!metaResp.ok) {
      console.warn('[orbitSynthesizer] mip pyramid metadata unavailable:', metaResp.status);
      return false;
    }
    const meta = await metaResp.json();

    const [fResp, sResp] = await Promise.all([
      fetch('/api/minimap/field/F', { credentials: 'same-origin' }),
      fetch('/api/minimap/field/S', { credentials: 'same-origin' }),
    ]);
    if (!fResp.ok || !sResp.ok) {
      console.warn('[orbitSynthesizer] mip pyramid field unavailable');
      return false;
    }

    const [fBuf, sBuf] = await Promise.all([fResp.arrayBuffer(), sResp.arrayBuffer()]);
    const fFlat = new Float32Array(fBuf);
    const sFlat = new Float32Array(sBuf);

    const widths = new Uint32Array(meta.F.mip_widths);
    const heights = new Uint32Array(meta.F.mip_heights);

    (m as any).set_mip_pyramid(
      fFlat,
      sFlat,
      widths,
      heights,
      meta.re_min,
      meta.re_max,
      meta.im_min,
      meta.im_max
    );
    console.log('[orbitSynthesizer] mip pyramid loaded:', widths.length, 'levels');
    return true;
  } catch (err) {
    console.warn('[orbitSynthesizer] failed to load mip pyramid:', err);
    return false;
  }
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
 * Orbit-based Julia parameter synthesizer.
 *
 * Thin adapter over the canonical Rust `PlayerState` c-space integrator
 * (runtime-core compiled to WebAssembly via wasm-orbit). Unlike the old
 * closed-loop carrier, `PlayerState` holds `c` as persistent state and moves
 * it toward a model-driven target point on the Mandelbrot boundary, biased
 * along the Shore's contours via the minimap. This restores audio-driven
 * wandering and actually exercises the minimap (issue #88).
 */
export class OrbitSynthesizer {
  private kBands: number;
  private state: WasmOrbitController;

  constructor(kBands: number, initialState?: Partial<OrbitState>) {
    const m = requireWasm();
    this.kBands = kBands;
    this.state = new m.OrbitController(
      initialState?.s ?? 0.5,
      initialState?.alpha ?? 0.5,
      1.0
    );
  }

  /** Current c (real part). */
  get cRe(): number {
    // OrbitController computes c on step; cache from last step.
    return this._lastC.real;
  }

  /** Current c (imaginary part). */
  get cIm(): number {
    return this._lastC.imag;
  }

  /** Current wobble phase (diagnostic). */
  get theta(): number {
    return this.state.theta;
  }

  /** Speed diagnostic: |dc| of last step (May controller has no velocity state). */
  get speed(): number {
    return this._lastSpeed;
  }

  private _lastC: Complex = { real: 0.0, imag: 0.0 };
  private _lastSpeed = 0.0;
  private _prevC: Complex | null = null;

  get lobe(): number {
    // PlayerState does not expose lobe as a getter; track it here.
    return this._lobe;
  }

  get subLobe(): number {
    return this._subLobe;
  }

  private _lobe = 1;
  private _subLobe = 0;

  /**
   * Apply model-predicted control signals to the Player state.
   */
  applyControls(signals: ControlSignals): void {
    this.state.apply_controls(signals.sTarget, signals.alpha);
  }

  /** Refinement 1: momentum (persistent velocity + drag). Default OFF. */
  setMomentum(on: boolean, drag = 0.9): void {
    // wasm-bindgen exposes Rust setters as JS property setters:
    // set_momentum(on) -> `momentum` accessor, set_drag(d) -> `drag`.
    const s = this.state as unknown as {
      momentum: boolean;
      drag: number;
      set_momentum?: (on: boolean) => void;
      set_drag?: (d: number) => void;
    };
    if (typeof s.set_momentum === 'function') {
      s.set_momentum(on);
    } else {
      s.momentum = on;
    }
    if (typeof s.set_drag === 'function') {
      s.set_drag(drag);
    } else {
      s.drag = drag;
    }
  }

  /** Refinement 2: shore bias via minimap contour stepping. Default OFF. */
  setShoreBias(on: boolean, dStar = 0.5, maxStep = 0.05): void {
    // wasm-bindgen exposes Rust setters as JS property setters; try the
    // method form first (mock/test doubles), fall back to properties.
    const s = this.state as unknown as {
      shore_bias: boolean;
      d_star: number;
      max_step: number;
      set_shore_bias?: (on: boolean) => void;
      set_d_star?: (d: number) => void;
      set_max_step?: (m: number) => void;
    };
    if (typeof s.set_shore_bias === 'function' && typeof s.set_d_star === 'function' && typeof s.set_max_step === 'function') {
      s.set_shore_bias(on);
      s.set_d_star(dStar);
      s.set_max_step(maxStep);
    } else {
      s.shore_bias = on;
      s.d_star = dStar;
      s.max_step = maxStep;
    }
  }

  /** Audio thrust: sustained loudness builds inertia (tangential acceleration). */
  setThrust(thrust: number): void {
    const s = this.state as unknown as {
      thrust: number;
      set_thrust?: (v: number) => void;
    };
    const clamped = Math.max(0, Math.min(0.2, thrust));
    if (typeof s.set_thrust === 'function') {
      s.set_thrust(clamped);
    } else {
      s.thrust = clamped;
    }
  }

  /** Audio energy in [0,1]: loud audio pulls c toward the Shore (energy servo). */
  setEnergy(energy: number): void {
    const s = this.state as unknown as {
      energy: number;
      set_energy?: (v: number) => void;
    };
    const clamped = Math.max(0, Math.min(1, energy));
    if (typeof s.set_energy === 'function') {
      s.set_energy(clamped);
    } else {
      s.energy = clamped;
    }
  }

  /**
   * Switch the active Mandelbrot lobe (section-change handling).
   * The May controller is cardioid-only; this is a no-op retained for API
   * compatibility with section-detection callers.
   */
  setLobe(lobe: number, subLobe = 0): void {
    this._lobe = lobe;
    this._subLobe = subLobe;
    const playerLike = this.state as unknown as WasmPlayerState;
    if (typeof playerLike.set_lobe === 'function') {
      playerLike.set_lobe(lobe, subLobe);
    }
  }

  /**
   * Advance the Player by dt and synthesize c(t).
   *
   * `h` is the transient/hit signal in [0, 1]; near 1 opens the Shore wall
   * (boundary crossing becomes easy — the "Skyrim clip"). Used for section
   * changes / onsets. Retained for legacy `orbit_controller` path; destination
   * manifold physics uses {@link stepWithControls} which is musically ignorant.
   */
  step(dt: number, bandGates: number[], h = 0.0): Complex {
    const gates = new Float64Array(Math.min(bandGates.length, this.kBands));
    for (let i = 0; i < gates.length; i++) {
      gates[i] = Math.max(0.0, Math.min(1.0, bandGates[i]));
    }
    const c = this.state.step(dt, Math.max(0.0, Math.min(1.0, h)), gates);
    if (this._prevC) {
      this._lastSpeed = Math.hypot(c.real - this._prevC.real, c.imag - this._prevC.imag);
    }
    this._prevC = { real: c.real, imag: c.imag };
    this._lastC = { real: c.real, imag: c.imag };
    return { real: c.real, imag: c.imag };
  }

  /**
   * Destination manifold step driven by Controls v2 (issue #107/#106).
   *
   * `controls.motion` resolves to a metric-consistent generalized drive
   * covector + PSD friction + bounded impulse; physics owns `G, Γ, U` and
   * integration. No musical feature (energy, transient h, band gates) is
   * needed — only the already-interpreted motion controls.
   */
  stepWithControls(
    dt: number,
    motion: { direction: [number, number]; throttle: number; brake: number; grip: number; impulse: number }
  ): Complex {
    const m = requireWasm() as unknown as { MotionControls?: new (...args: number[]) => unknown };
    let ctrl: unknown;
    if (m.MotionControls) {
      ctrl = new (m.MotionControls as new (...a: number[]) => unknown)(
        motion.direction[0],
        motion.direction[1],
        motion.throttle,
        motion.brake,
        motion.grip,
        motion.impulse
      );
    } else {
      // Mock / test double without WASM MotionControls binding
      ctrl = {
        direction_x: motion.direction[0],
        direction_y: motion.direction[1],
        throttle: motion.throttle,
        brake: motion.brake,
        grip: motion.grip,
        impulse: motion.impulse
      } as unknown as WasmMotionControls;
    }
    const s = this.state as unknown as {
      stepWithControls?: (dt: number, c: unknown) => { real: number; imag: number };
      step_with_controls?: (dt: number, c: unknown) => { real: number; imag: number };
    };
    let c: { real: number; imag: number };
    if (typeof s.stepWithControls === 'function') {
      c = s.stepWithControls(dt, ctrl);
    } else if (typeof s.step_with_controls === 'function') {
      c = s.step_with_controls(dt, ctrl);
    } else {
      throw new Error('[orbitSynthesizer] wasm build has no stepWithControls binding; rebuild wasm-orbit');
    }
    if (this._prevC) {
      this._lastSpeed = Math.hypot(c.real - this._prevC.real, c.imag - this._prevC.imag);
    }
    this._prevC = { real: c.real, imag: c.imag };
    this._lastC = { real: c.real, imag: c.imag };
    return { real: c.real, imag: c.imag };
  }

  /**
   * Typed accessor for the underlying wasm controller's read-only
   * DebugSnapshot seam (issue #111). Returns the controller object exposing
   * `debugSnapshot()`, or undefined when the wasm build predates the seam.
   * Diagnostic-only: the snapshot is a pure function of controller state.
   */
  debugSnapshotController(): { debugSnapshot?: () => unknown } | undefined {
    return this.state as unknown as { debugSnapshot?: () => unknown } | undefined;
  }
}
