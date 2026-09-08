/* tslint:disable */
/* eslint-disable */

/** Canonical analysis tick — the seam CycleBank will consume (issue #91). */
export interface AnalysisTick {
    features: number[];
    sampleIndex: number;
    timeSeconds: number;
    dtSeconds: number;
    streamEpoch: number;
}

/** Diagnostic snapshot for manual verification of the canonical clock. */
export interface TimebaseDiagnostics {
    sourceSampleRate: number;
    sourceFramesIngested: number;
    canonicalSampleIndex: number;
    analysisHopCount: number;
    timeSeconds: number;
    streamEpoch: number;
    detectedGaps: number;
    detectedOverlaps: number;
    lastSourceStartFrame: number;
    lastSourceEndFrame: number;
}



/** One directly observed temporal ridge (issue #92). All math is Rust-owned. */
export interface CycleMode {
    id: number;
    frequencyHz: number;
    phase: number;
    strength: number;
    confidence: number;
    channelSupport: number;
    age: number;
    missingObservations: number;
    frequencySlope: number;
    frequencyUncertainty: number;
}

/** Diagnostic rational relationship between two observed modes. */
export interface CycleRelation {
    iId: number;
    jId: number;
    m: number;
    n: number;
    freqResidual: number;
    generalizedPhase: number;
    phaseStability: number;
}

/** One named scalar evidence channel value for an explicit observation. */
export interface CycleEvidenceChannelInput {
    name: string;
    value: number;
}



/** Version of the read-only DebugSnapshot contract (issue #111). */
export interface DebugSnapshotMeta {
    version: string;
    canonicalDt: number;
}



/**
 * Authoritative sample-clock audio timebase (issue #91).
 *
 * The browser's AudioWorklet transport feeds non-overlapping PCM blocks
 * here; the Rust timebase validates monotonicity, resamples statefully to
 * the canonical 48 kHz timeline, schedules exact 1024-sample hops, and runs
 * the canonical FeatureExtractor — all in Rust, so there is no TypeScript
 * mirror of the timing/scheduling math (ADR 0001).
 */
export class AnalysisTimebase {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Diagnostic snapshot for verifying the clock manually.
     * See ``ingest`` for the typing note about the .d.ts return type.
     */
    diagnostics(): any;
    /**
     * Flush end-of-stream (recovers the deferred final sample/tick).
     * See ``ingest`` for the typing note about the .d.ts return type.
     */
    flush(): any;
    /**
     * Ingest one non-overlapping PCM block. Returns a JS array of ticks
     * (possibly empty). Throws on overlap / mid-stream rate change.
     *
     * Type note: the generated .d.ts types this as `any` because the
     * function returns a `JsValue` (it serializes via
     * ``serde_wasm_bindgen``). The TS adapter in
     * ``frontend/src/lib/analysisTimebase.ts`` re-types this signature
     * as ``AnalysisTick[]`` and the ``AnalysisTick`` interface itself
     * is provided by the ``TS_TYPES`` custom section above — so the
     * Rust source remains the single authority for the wire shape.
     */
    ingest(samples: Float32Array, source_sample_rate: number, source_start_frame: bigint): any;
    constructor();
    /**
     * Declare a stream discontinuity. `reason` is informational; the epoch
     * always bumps and the schedule resets.
     */
    reset(): void;
}

export class ColorIntent {
    free(): void;
    [Symbol.dispose](): void;
    constructor(anchor_hue: number, chroma: number, lightness: number, harmony: string, accent_weight: number);
    accent_weight: number;
    anchor_hue: number;
    chroma: number;
    harmony: string;
    lightness: number;
}

/**
 * Wrapper for complex number to/from JavaScript
 */
export class Complex {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    imag: number;
    real: number;
}

export class ControlsV2 {
    free(): void;
    [Symbol.dispose](): void;
    clamped(): ControlsV2;
    static fromModelOutput(output: Float64Array): ControlsV2;
    static modelOutputOrder(): any[];
    constructor(motion: MotionControls, view: JuliaViewControls);
    to_model_output(): Float64Array;
    motion: MotionControls;
    view: JuliaViewControls;
}

/**
 * Canonical observed-ridge CycleBank (issue #92), browser surface.
 *
 * The browser feeds one canonical `AnalysisTick` per authoritative hop and
 * reads the currently observed modes / relations. It never interprets the
 * rolling feature window's offsets itself.
 */
export class CycleBank {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Rational relations among the currently observed modes (latest batch).
     */
    latest_relations(): any;
    /**
     * Current confirmed observed modes (`CycleMode[]`).
     */
    modes(): any;
    /**
     * Construct with the canonical defaults (no config). Config overrides
     * are a Rust-side concern; the browser runs the canonical pipeline.
     */
    constructor();
    /**
     * Number of currently confirmed modes.
     */
    num_modes(): number;
    /**
     * Feed one explicit observation of named scalar evidence channels
     * (diagnostic entry point; the production path is `observe_tick`).
     */
    observe(sample_index: bigint, dt_seconds: number, stream_epoch: bigint, channels: any): any;
    /**
     * Feed one canonical analysis tick (the `AnalysisTick` produced by the
     * wasm `AnalysisTimebase.ingest`/`flush`). Returns the current observed
     * `CycleMode[]`. The newest-frame extraction is done in Rust.
     */
    observe_tick(tick: any): any;
    /**
     * Deterministic discontinuity reset.
     */
    reset(): void;
    /**
     * The Rust-owned contract version (`CYCLE_BANK_VERSION`).
     */
    readonly version: string;
}

/**
 * Audio feature extractor — the SAME Rust implementation the trainer uses.
 *
 * The browser feeds a rolling window of PCM samples (from
 * AnalyserNode.getFloatTimeDomainData) and receives feature windows laid
 * out identically to training inputs. This eliminates the entire class of
 * browser-vs-trainer extraction drift (FFT size, smoothing, dB conversion,
 * per-file normalization, window layout) by construction: there is only
 * one implementation, executed in two places.
 */
export class FeatureExtractor {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Extract the MOST RECENT flattened feature window from `audio`
     * (frame-major).
     *
     * `audio` is the rolling PCM history in chronological order; the
     * returned window covers the latest `window_frames` STFT frames,
     * matching what live inference needs. Short input is padded by
     * repeating the last frame, matching training behavior for short
     * files.
     */
    extract_window(audio: Float32Array, window_frames: number): Float64Array;
    /**
     * Create an extractor with the shared runtime defaults (48 kHz,
     * hop 1024, n_fft 4096). Callers must resample browser audio to the
     * runtime sample rate before feeding PCM here.
     */
    constructor();
    readonly num_features_per_frame: number;
}

export class JuliaViewControls {
    free(): void;
    [Symbol.dispose](): void;
    clamped(): JuliaViewControls;
    constructor(zoom_delta: number, rotation_delta: number, hue_delta: number, chroma_delta: number, lightness_delta: number, accent_delta: number, harmony_shift: number);
    accent_delta: number;
    chroma_delta: number;
    harmony_shift: number;
    hue_delta: number;
    lightness_delta: number;
    rotation_delta: number;
    zoom_delta: number;
}

export class JuliaViewState {
    free(): void;
    [Symbol.dispose](): void;
    applyControls(controls: JuliaViewControls): void;
    clamped(): JuliaViewState;
    constructor(zoom: number, rotation: number, color: ColorIntent | null | undefined, harmony_cooldown: number, harmony_armed: boolean);
    color: ColorIntent;
    harmony_armed: boolean;
    harmony_cooldown: number;
    rotation: number;
    zoom: number;
}

/**
 * Manifold configuration for the browser (issue #106).
 */
export class ManifoldConfig {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Use the same defaults as OrbitController, without a second browser authority.
     */
    static defaults(): ManifoldConfig;
    constructor(d_ref: number, epsilon: number, lambda_sq: number, kappa: number, mu: number);
    readonly d_ref: number;
    readonly epsilon: number;
    readonly kappa: number;
    readonly lambda_sq: number;
    readonly mu: number;
}

export class MotionControls {
    free(): void;
    [Symbol.dispose](): void;
    clamped(): MotionControls;
    drive_magnitude(): number;
    friction_beta(): number;
    constructor(direction_x: number, direction_y: number, throttle: number, brake: number, grip: number, impulse: number);
    brake: number;
    direction_x: number;
    direction_y: number;
    grip: number;
    impulse: number;
    throttle: number;
}

/**
 * --- May-proven OrbitController bindings (restored baseline) ---
 */
export class OrbitController {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Apply model-predicted control signals (s, alpha).
     */
    apply_controls(s: number, alpha: number): void;
    /**
     * Read-only DebugSnapshot of the current authoritative state.
     */
    debugSnapshot(): any;
    /**
     * Get the current manifold configuration.
     */
    manifold_config(): ManifoldConfig;
    constructor(s: number, alpha: number, omega: number);
    /**
     * Seed the authoritative player position from (re, im) parts. The next
     * step_with_controls call advances from this point. Parts (not a
     * Complex instance) so callers never need to construct wasm objects.
     */
    setC(re: number, im: number): void;
    /**
     * Seed the planar velocity from (vx, vy) parts. The next
     * step_with_controls call applies Q_drive and drag from this velocity.
     */
    setVelocity(vx: number, vy: number): void;
    /**
     * Set the manifold configuration (used only when manifold_physics is on).
     */
    set_manifold_config(config: ManifoldConfig): void;
    /**
     * Advance one frame; returns the new c. `h` is the transient signal
     * in [0, 1] — near 1 opens the Shore wall for boundary crossing.
     */
    step(dt: number, h: number, band_gates?: Float64Array | null): Complex;
    /**
     * Destination manifold step driven by Controls v2 (issue #107/#106).
     */
    stepWithControls(dt: number, motion: MotionControls): Complex;
    /**
     * Authoritative player position c in the complex plane.
     * Read/write so test harnesses and the debug cockpit can seed a
     * non-default starting point (e.g. "approach from outside M" trajectories
     * that begin at a seahorse-basin c without paying the launch cost of
     * crossing the cardioid ridge).
     */
    readonly c: Complex;
    /**
     * Get the drag coefficient for manifold physics.
     */
    manifold_drag: number;
    /**
     * The most recent manifold-physics failure, if any. When manifold mode is
     * selected and the integrator fails, the controller fails closed (holds
     * the last valid state) and records the error here.
     */
    readonly manifold_error: string | undefined;
    /**
     * Whether manifold physics is currently enabled.
     */
    manifold_physics: boolean;
    /**
     * Target shore proximity for the shore-bias servo.
     */
    set d_star(value: number);
    /**
     * Friction for momentum refinement (default 0.90).
     */
    set drag(value: number);
    /**
     * Audio energy in [0, 1]: raises the servo's target shore-proximity
     * (loud audio pulls c toward the Shore).
     */
    set energy(value: number);
    /**
     * Max world-space step per frame for shore bias.
     */
    set max_step(value: number);
    /**
     * Refinement 1 toggle: momentum (persistent velocity + drag).
     */
    set momentum(value: boolean);
    /**
     * Refinement 2 toggle: shore bias via minimap contour stepping.
     */
    set shore_bias(value: boolean);
    /**
     * Audio thrust for momentum: sustained energy builds inertia.
     */
    set thrust(value: number);
    /**
     * Wobble phase (diagnostic).
     */
    readonly theta: number;
    /**
     * Authoritative planar velocity (vx, vy) used by the destination
     * manifold integrator.
     */
    readonly velocity: Complex;
}

/**
 * Orbit state wrapper for WASM
 */
export class OrbitState {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Advance state by dt seconds
     */
    advance(dt: number): void;
    /**
     * Create new orbit state with optional seed
     */
    constructor(lobe: number, sub_lobe: number, theta: number, omega: number, s: number, alpha: number, k_residuals: number, residual_omega_scale: number, seed?: bigint | null);
    /**
     * Create deterministic orbit with default parameters and seed
     */
    static newDefault(seed: bigint): OrbitState;
    /**
     * Get alpha (residual amplitude)
     */
    alpha: number;
    /**
     * Get lobe
     */
    lobe: number;
    /**
     * Get s (radius scaling)
     */
    s: number;
    /**
     * Set omega (base angular velocity)
     */
    set omega(value: number);
    /**
     * Get sub_lobe
     */
    sub_lobe: number;
    /**
     * Get theta
     */
    readonly theta: number;
}

/**
 * Player c-space integrator wrapper for WASM (issue #88, Q2).
 *
 * Holds `c` as persistent state and moves it toward a model-driven target
 * point on the Mandelbrot boundary, biased along the Shore's contours via
 * the minimap. This replaces the closed-loop carrier for audio-driven
 * wandering.
 */
export class PlayerState {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Apply model-predicted control signals.
     */
    apply_controls(s: number, alpha: number, omega_scale: number): void;
    /**
     * Create a PlayerState starting on the boundary at (s, alpha).
     */
    constructor(lobe: number, sub_lobe: number, s: number, alpha: number);
    /**
     * Switch the active Mandelbrot lobe.
     */
    set_lobe(lobe: number, sub_lobe: number): void;
    /**
     * Advance the Player by dt, moving c toward the model-driven target,
     * biased along the Shore's contours. Returns the new c.
     */
    step(dt: number, h: number, band_gates?: Float64Array | null): Complex;
    /**
     * Current c (imaginary part).
     */
    readonly c_im: number;
    /**
     * Current c (real part).
     */
    readonly c_re: number;
    /**
     * Set the target shore-proximity distance the servo pulls toward.
     */
    set d_star(value: number);
    /**
     * Set the audio energy in [0, 1] (loudness). Raises the servo's
     * target shore-proximity: loud audio pulls c toward the Shore.
     */
    set energy(value: number);
    /**
     * Set the mip level used for the contour step.
     */
    set level(value: number);
    /**
     * Set the maximum world-space step per frame.
     */
    set max_step(value: number);
    /**
     * Current c-space velocity magnitude (Momentum diagnostic).
     */
    readonly speed: number;
}

/**
 * Residual parameters
 */
export class ResidualParams {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Create default residual parameters
     */
    constructor(k_residuals: number, residual_cap: number, radius_scale: number);
}

export function audioFeatureAveragesJson(values: Float64Array, features_per_frame: number): string;

/**
 * Shared constants exposed to JavaScript
 */
export function constants(): any;

/**
 * Contour-biased integrator step for Physics. Returns [new_real, new_imag].
 */
export function contour_biased_step(real: number, imag: number, u_real: number, u_imag: number, h: number, d_star: number, max_step: number, level: number, energy: number): Float64Array;

export function controlsIntegrateStep(c_re: number, c_im: number, vx: number, vy: number, motion: MotionControls, dt: number, config: ManifoldConfig): Float64Array;

export function controlsV2SchemaJson(): string;

export function controlsV2VisualParametersJson(c_re: number, c_im: number, controls_json: string, presentation_json?: string | null): string;

/**
 * Build a read-only DebugSnapshot from explicit authoritative state.
 *
 * `motion_raw` is the last raw (pre-clamp) MotionControls, or null before
 * the first step. `last_delta_total` is the last step's total-energy change
 * (NaN = none). Never mutates runtime state.
 */
export function debugSnapshotFromState(c_re: number, c_im: number, vx: number, vy: number, motion_raw: MotionControls | null | undefined, friction_beta: number, friction_power: number, manifold_drag: number, config: ManifoldConfig, last_delta_total: number, time_seconds: number): any;

/**
 * The DebugSnapshot contract version and canonical step cadence.
 */
export function debugSnapshotMeta(): any;

/**
 * Sample an n x n terrain patch of the canonical embedding
 * Q(c) = (x, y, lambda*sigma(c)) centered at (cx, cy) with half-extent
 * `half` in c-space. Returns a camelCase JSON object:
 * { n, center, half, positions, signed, realm }.
 */
export function debugTerrainPatch(cx: number, cy: number, half: number, n: number, config: ManifoldConfig): any;

export function decodeControlsV2Json(values: Float64Array): string;

export function decodeLegacyVisualJson(values: Float64Array, audio_reactive: boolean, rms: number, onset: number): string;

export function decodeOrbitControlJson(values: Float64Array, k_bands: number): string;

/**
 * Deep-zoom unsigned distance field for the minimap (issue #111 feedback:
 * the minimap is a Mandelbrot deep zoom whose zoom level follows the
 * player). Resolution-unlimited escape-iteration estimator — resolves
 * structure where the baked mip pyramid runs out of texels. Returns one
 * unsigned distance per input point (0 inside the set).
 */
export function deepZoomField(re: Float64Array, im: Float64Array): Float32Array;

export function legacyAudioFeatureAveragesJson(values: Float64Array): string;

export function legacyOrbitDriveInputsJson(rms: number, onset: number): string;

export function legacyVisualExportRangesJson(): string;

export function legacyVisualSchemaJson(): string;

/**
 * Convert a generalized force covector to coordinate acceleration: a = G^{-1} Q.
 * Returns [ax, ay]. This is the single place G^{-1} maps a covector to acceleration.
 */
export function manifold_apply_generalized_force(qx: number, qy: number, real: number, imag: number, config: ManifoldConfig): Float64Array;

/**
 * Christoffel symbols Gamma^i_jk. Returns a flat JS array of 8 values:
 * [Gamma^0_00, Gamma^0_01, Gamma^0_10, Gamma^0_11, Gamma^1_00, Gamma^1_01, Gamma^1_10, Gamma^1_11].
 */
export function manifold_christoffel_symbols(real: number, imag: number, config: ManifoldConfig): Array<any>;

/**
 * Metric-consistent isotropic drag covector: Q_drag = -beta G v. Returns [Qx, Qy].
 * This is a covector, not a coordinate acceleration; its power P = v^T Q_drag <= 0.
 */
export function manifold_drag_force(vx: number, vy: number, real: number, imag: number, beta: number, config: ManifoldConfig): Float64Array;

/**
 * Embedding q(c) = (x, y, sigma(c)). Returns [x, y, sigma].
 */
export function manifold_embedding(real: number, imag: number, config: ManifoldConfig): Array<any>;

/**
 * Geodesic acceleration term: Gamma^i_jk v^j v^k. Returns [ax, ay].
 */
export function manifold_geodesic_acceleration(vx: number, vy: number, real: number, imag: number, config: ManifoldConfig): Float64Array;

/**
 * Scale-relative induced metric
 * G(c) = rho^-2 I + lambda^2 * grad_sigma * grad_sigma^T.
 * Returns a flat JS array [g11, g12, g12, g22].
 */
export function manifold_induced_metric(real: number, imag: number, config: ManifoldConfig): Array<any>;

/**
 * Semi-implicit Euler integration step for manifold dynamics.
 *
 * Integrates: r_ddot + Gamma(r_dot, r_dot) = -G^{-1}∇U + G^{-1}Q
 *
 * Returns [new_re, new_im, new_vx, new_vy, kinetic, potential, total, delta_total, delta_kinetic].
 */
export function manifold_integrate_step(c_re: number, c_im: number, vx: number, vy: number, qx: number, qy: number, beta: number, dt: number, config: ManifoldConfig): Float64Array;

/**
 * Jacobian J_q(c) = ∂q/∂(x,y) as a 3×2 matrix. Returns flat array [1,0,0,1,sigma_x,sigma_y].
 */
export function manifold_jacobian(real: number, imag: number, config: ManifoldConfig): Array<any>;

/**
 * Kinetic energy K = 1/2 v^T G v.
 */
export function manifold_kinetic_energy(vx: number, vy: number, real: number, imag: number, config: ManifoldConfig): number;

/**
 * Mandelbrot scale sigma(c) = log2(d_ref / rho(c)).
 */
export function manifold_mandelbrot_scale(real: number, imag: number, config: ManifoldConfig): number;

/**
 * Native potential U = kappa * sigma(c).
 */
export function manifold_potential_energy(real: number, imag: number, config: ManifoldConfig): number;

/**
 * Generalized potential force covector: Q_potential = -grad U = -kappa grad sigma.
 * Returns [Qx, Qy]. This is a covector, not a coordinate acceleration; convert
 * with `manifold_apply_generalized_force`.
 */
export function manifold_potential_force(real: number, imag: number, config: ManifoldConfig): Float64Array;

/**
 * Embedded velocity q_dot = J_q(c) v. Returns [vx, vy, sigma_dot].
 */
export function manifold_q_dot(vx: number, vy: number, real: number, imag: number, config: ManifoldConfig): Array<any>;

/**
 * Regularized distance rho(c) = sqrt(D^2 + epsilon^2).
 */
export function manifold_regularized_distance(real: number, imag: number, epsilon: number): number;

/**
 * Scale gradient ∇sigma(c) = (∂sigma/∂x, ∂sigma/∂y). Returns [gx, gy].
 */
export function manifold_scale_gradient(real: number, imag: number, config: ManifoldConfig): Float64Array;

/**
 * Scale Hessian [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]].
 * Returns a flat JS array [xx, xy, xy, yy].
 */
export function manifold_scale_hessian(real: number, imag: number, config: ManifoldConfig): Array<any>;

/**
 * Time derivative of Mandelbrot scale: sigma_dot = ∇sigma·v. No independent v_sigma.
 */
export function manifold_sigma_dot(vx: number, vy: number, real: number, imag: number, config: ManifoldConfig): number;

/**
 * Signed distance to the Mandelbrot boundary. Positive outside, negative inside.
 */
export function manifold_signed_distance(real: number, imag: number): number;

/**
 * Total mechanical energy E = K + U_sigma + U_wall.
 */
export function manifold_total_energy(vx: number, vy: number, real: number, imag: number, config: ManifoldConfig): number;

/**
 * Unsigned geometric distance d(c) = |D(c)|. Distinct from S sensitivity.
 */
export function manifold_unsigned_distance(real: number, imag: number): number;

/**
 * Batch shore-proximity (S field) sampling over the canonical mip pyramid
 * (issue #111 minimap panel). Same field/level/rounding as the single-point
 * sampler; one lock for the whole batch. Returns a flat Float32Array.
 */
export function minimapShoreProximityBatch(re: Float64Array, im: Float64Array, level: number): Float32Array;

/**
 * Slope of the shore-proximity field at c on a mip level. Returns [gx, gy].
 */
export function minimap_slope(real: number, imag: number, level: number): Float64Array;

export function modelOutputKind(model_type?: string | null, controls_version?: string | null): string;

export function motionDriveCovector(c_re: number, c_im: number, motion: MotionControls, config: ManifoldConfig): Float64Array;

export function orbitControlSchemaJson(k_bands: number): string;

export function orbitVisualParametersJson(c_re: number, c_im: number, controls_json: string, rms: number, onset: number): string;

/**
 * The Player's full observation at c: 4x81 greys + 8 slope values = 332.
 */
export function player_observation(real: number, imag: number): Float32Array;

/**
 * Set the mip pyramid from host-provided flat planes (row-major, per level).
 */
export function set_mip_pyramid(f_flat: Float32Array, s_flat: Float32Array, widths: Uint32Array, heights: Uint32Array, re_min: number, re_max: number, im_min: number, im_max: number): void;

/**
 * Step the orbit forward and synthesize
 */
export function step(state: OrbitState, dt: number, residual_params: ResidualParams, band_gates?: Float64Array | null): Complex;

/**
 * Synthesize Julia parameter from orbit state
 */
export function synthesize(state: OrbitState, residual_params: ResidualParams, band_gates?: Float64Array | null): Complex;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_analysistimebase_free: (a: number, b: number) => void;
    readonly __wbg_colorintent_free: (a: number, b: number) => void;
    readonly __wbg_complex_free: (a: number, b: number) => void;
    readonly __wbg_controlsv2_free: (a: number, b: number) => void;
    readonly __wbg_cyclebank_free: (a: number, b: number) => void;
    readonly __wbg_featureextractor_free: (a: number, b: number) => void;
    readonly __wbg_get_complex_imag: (a: number) => number;
    readonly __wbg_get_complex_real: (a: number) => number;
    readonly __wbg_juliaviewcontrols_free: (a: number, b: number) => void;
    readonly __wbg_juliaviewstate_free: (a: number, b: number) => void;
    readonly __wbg_manifoldconfig_free: (a: number, b: number) => void;
    readonly __wbg_motioncontrols_free: (a: number, b: number) => void;
    readonly __wbg_orbitcontroller_free: (a: number, b: number) => void;
    readonly __wbg_orbitstate_free: (a: number, b: number) => void;
    readonly __wbg_playerstate_free: (a: number, b: number) => void;
    readonly __wbg_residualparams_free: (a: number, b: number) => void;
    readonly __wbg_set_complex_imag: (a: number, b: number) => void;
    readonly __wbg_set_complex_real: (a: number, b: number) => void;
    readonly analysistimebase_diagnostics: (a: number) => any;
    readonly analysistimebase_flush: (a: number) => any;
    readonly analysistimebase_ingest: (a: number, b: number, c: number, d: number, e: bigint) => [number, number, number];
    readonly analysistimebase_new: () => number;
    readonly analysistimebase_reset: (a: number) => void;
    readonly audioFeatureAveragesJson: (a: number, b: number, c: number) => [number, number, number, number];
    readonly colorintent_accent_weight: (a: number) => number;
    readonly colorintent_anchor_hue: (a: number) => number;
    readonly colorintent_chroma: (a: number) => number;
    readonly colorintent_harmony: (a: number) => [number, number];
    readonly colorintent_lightness: (a: number) => number;
    readonly colorintent_new: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly colorintent_set_accent_weight: (a: number, b: number) => void;
    readonly colorintent_set_anchor_hue: (a: number, b: number) => void;
    readonly colorintent_set_chroma: (a: number, b: number) => void;
    readonly colorintent_set_harmony: (a: number, b: number, c: number) => void;
    readonly colorintent_set_lightness: (a: number, b: number) => void;
    readonly constants: () => any;
    readonly contour_biased_step: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number, number];
    readonly controlsIntegrateStep: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly controlsV2SchemaJson: () => [number, number, number, number];
    readonly controlsV2VisualParametersJson: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly controlsv2_clamped: (a: number) => number;
    readonly controlsv2_fromModelOutput: (a: number, b: number) => [number, number, number];
    readonly controlsv2_modelOutputOrder: () => [number, number];
    readonly controlsv2_motion: (a: number) => number;
    readonly controlsv2_new: (a: number, b: number) => number;
    readonly controlsv2_set_motion: (a: number, b: number) => void;
    readonly controlsv2_set_view: (a: number, b: number) => void;
    readonly controlsv2_to_model_output: (a: number) => [number, number];
    readonly controlsv2_view: (a: number) => number;
    readonly cyclebank_latest_relations: (a: number) => any;
    readonly cyclebank_modes: (a: number) => any;
    readonly cyclebank_new: () => [number, number, number];
    readonly cyclebank_num_modes: (a: number) => number;
    readonly cyclebank_observe: (a: number, b: bigint, c: number, d: bigint, e: any) => [number, number, number];
    readonly cyclebank_observe_tick: (a: number, b: any) => [number, number, number];
    readonly cyclebank_reset: (a: number) => void;
    readonly cyclebank_version: (a: number) => [number, number];
    readonly debugSnapshotFromState: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number];
    readonly debugSnapshotMeta: () => any;
    readonly debugTerrainPatch: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly decodeControlsV2Json: (a: number, b: number) => [number, number, number, number];
    readonly decodeLegacyVisualJson: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly decodeOrbitControlJson: (a: number, b: number, c: number) => [number, number, number, number];
    readonly deepZoomField: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly featureextractor_extract_window: (a: number, b: number, c: number, d: number) => [number, number];
    readonly featureextractor_new: () => number;
    readonly featureextractor_num_features_per_frame: (a: number) => number;
    readonly juliaviewcontrols_accent_delta: (a: number) => number;
    readonly juliaviewcontrols_clamped: (a: number) => number;
    readonly juliaviewcontrols_harmony_shift: (a: number) => number;
    readonly juliaviewcontrols_lightness_delta: (a: number) => number;
    readonly juliaviewcontrols_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly juliaviewcontrols_set_accent_delta: (a: number, b: number) => void;
    readonly juliaviewcontrols_set_harmony_shift: (a: number, b: number) => void;
    readonly juliaviewcontrols_set_lightness_delta: (a: number, b: number) => void;
    readonly juliaviewstate_applyControls: (a: number, b: number) => void;
    readonly juliaviewstate_clamped: (a: number) => number;
    readonly juliaviewstate_color: (a: number) => number;
    readonly juliaviewstate_harmony_armed: (a: number) => number;
    readonly juliaviewstate_harmony_cooldown: (a: number) => number;
    readonly juliaviewstate_new: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly juliaviewstate_set_color: (a: number, b: number) => void;
    readonly juliaviewstate_set_harmony_armed: (a: number, b: number) => void;
    readonly juliaviewstate_set_harmony_cooldown: (a: number, b: number) => void;
    readonly legacyAudioFeatureAveragesJson: (a: number, b: number) => [number, number, number, number];
    readonly legacyOrbitDriveInputsJson: (a: number, b: number) => [number, number, number, number];
    readonly legacyVisualExportRangesJson: () => [number, number, number, number];
    readonly legacyVisualSchemaJson: () => [number, number, number, number];
    readonly manifold_apply_generalized_force: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly manifold_christoffel_symbols: (a: number, b: number, c: number) => [number, number, number];
    readonly manifold_drag_force: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly manifold_embedding: (a: number, b: number, c: number) => [number, number, number];
    readonly manifold_geodesic_acceleration: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly manifold_induced_metric: (a: number, b: number, c: number) => [number, number, number];
    readonly manifold_integrate_step: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number, number];
    readonly manifold_jacobian: (a: number, b: number, c: number) => [number, number, number];
    readonly manifold_kinetic_energy: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly manifold_mandelbrot_scale: (a: number, b: number, c: number) => [number, number, number];
    readonly manifold_potential_energy: (a: number, b: number, c: number) => [number, number, number];
    readonly manifold_potential_force: (a: number, b: number, c: number) => [number, number, number, number];
    readonly manifold_q_dot: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly manifold_regularized_distance: (a: number, b: number, c: number) => [number, number, number];
    readonly manifold_scale_gradient: (a: number, b: number, c: number) => [number, number, number, number];
    readonly manifold_scale_hessian: (a: number, b: number, c: number) => [number, number, number];
    readonly manifold_sigma_dot: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly manifold_signed_distance: (a: number, b: number) => [number, number, number];
    readonly manifold_total_energy: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly manifold_unsigned_distance: (a: number, b: number) => [number, number, number];
    readonly manifoldconfig_defaults: () => number;
    readonly manifoldconfig_new: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly minimapShoreProximityBatch: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly minimap_slope: (a: number, b: number, c: number) => [number, number, number, number];
    readonly modelOutputKind: (a: number, b: number, c: number, d: number) => [number, number];
    readonly motionDriveCovector: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly motioncontrols_clamped: (a: number) => number;
    readonly motioncontrols_drive_magnitude: (a: number) => number;
    readonly motioncontrols_friction_beta: (a: number) => number;
    readonly motioncontrols_new: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly orbitControlSchemaJson: (a: number) => [number, number, number, number];
    readonly orbitVisualParametersJson: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly orbitcontroller_apply_controls: (a: number, b: number, c: number) => void;
    readonly orbitcontroller_c: (a: number) => number;
    readonly orbitcontroller_debugSnapshot: (a: number) => [number, number, number];
    readonly orbitcontroller_manifold_config: (a: number) => number;
    readonly orbitcontroller_manifold_drag: (a: number) => number;
    readonly orbitcontroller_manifold_error: (a: number) => [number, number];
    readonly orbitcontroller_manifold_physics: (a: number) => number;
    readonly orbitcontroller_new: (a: number, b: number, c: number) => number;
    readonly orbitcontroller_setC: (a: number, b: number, c: number) => void;
    readonly orbitcontroller_setVelocity: (a: number, b: number, c: number) => void;
    readonly orbitcontroller_set_d_star: (a: number, b: number) => void;
    readonly orbitcontroller_set_drag: (a: number, b: number) => void;
    readonly orbitcontroller_set_energy: (a: number, b: number) => void;
    readonly orbitcontroller_set_manifold_config: (a: number, b: number) => void;
    readonly orbitcontroller_set_manifold_drag: (a: number, b: number) => void;
    readonly orbitcontroller_set_manifold_physics: (a: number, b: number) => void;
    readonly orbitcontroller_set_max_step: (a: number, b: number) => void;
    readonly orbitcontroller_set_momentum: (a: number, b: number) => void;
    readonly orbitcontroller_set_shore_bias: (a: number, b: number) => void;
    readonly orbitcontroller_set_thrust: (a: number, b: number) => void;
    readonly orbitcontroller_step: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly orbitcontroller_stepWithControls: (a: number, b: number, c: number) => number;
    readonly orbitcontroller_theta: (a: number) => number;
    readonly orbitcontroller_velocity: (a: number) => number;
    readonly orbitstate_advance: (a: number, b: number) => void;
    readonly orbitstate_lobe: (a: number) => number;
    readonly orbitstate_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: bigint) => number;
    readonly orbitstate_newDefault: (a: bigint) => number;
    readonly orbitstate_set_lobe: (a: number, b: number) => void;
    readonly orbitstate_set_sub_lobe: (a: number, b: number) => void;
    readonly orbitstate_sub_lobe: (a: number) => number;
    readonly player_observation: (a: number, b: number) => [number, number, number, number];
    readonly playerstate_apply_controls: (a: number, b: number, c: number, d: number) => void;
    readonly playerstate_new: (a: number, b: number, c: number, d: number) => number;
    readonly playerstate_set_d_star: (a: number, b: number) => void;
    readonly playerstate_set_energy: (a: number, b: number) => void;
    readonly playerstate_set_level: (a: number, b: number) => void;
    readonly playerstate_set_lobe: (a: number, b: number, c: number) => void;
    readonly playerstate_set_max_step: (a: number, b: number) => void;
    readonly playerstate_speed: (a: number) => number;
    readonly playerstate_step: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly residualparams_new: (a: number, b: number, c: number) => number;
    readonly set_mip_pyramid: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => [number, number];
    readonly step: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly synthesize: (a: number, b: number, c: number, d: number) => number;
    readonly juliaviewcontrols_set_chroma_delta: (a: number, b: number) => void;
    readonly juliaviewcontrols_set_hue_delta: (a: number, b: number) => void;
    readonly juliaviewcontrols_set_rotation_delta: (a: number, b: number) => void;
    readonly juliaviewcontrols_set_zoom_delta: (a: number, b: number) => void;
    readonly juliaviewstate_set_rotation: (a: number, b: number) => void;
    readonly juliaviewstate_set_zoom: (a: number, b: number) => void;
    readonly motioncontrols_set_brake: (a: number, b: number) => void;
    readonly motioncontrols_set_direction_x: (a: number, b: number) => void;
    readonly motioncontrols_set_direction_y: (a: number, b: number) => void;
    readonly motioncontrols_set_grip: (a: number, b: number) => void;
    readonly motioncontrols_set_impulse: (a: number, b: number) => void;
    readonly motioncontrols_set_throttle: (a: number, b: number) => void;
    readonly orbitstate_set_alpha: (a: number, b: number) => void;
    readonly orbitstate_set_omega: (a: number, b: number) => void;
    readonly orbitstate_set_s: (a: number, b: number) => void;
    readonly juliaviewcontrols_chroma_delta: (a: number) => number;
    readonly juliaviewcontrols_hue_delta: (a: number) => number;
    readonly juliaviewcontrols_rotation_delta: (a: number) => number;
    readonly juliaviewcontrols_zoom_delta: (a: number) => number;
    readonly juliaviewstate_rotation: (a: number) => number;
    readonly juliaviewstate_zoom: (a: number) => number;
    readonly manifoldconfig_d_ref: (a: number) => number;
    readonly manifoldconfig_epsilon: (a: number) => number;
    readonly manifoldconfig_kappa: (a: number) => number;
    readonly manifoldconfig_lambda_sq: (a: number) => number;
    readonly manifoldconfig_mu: (a: number) => number;
    readonly motioncontrols_brake: (a: number) => number;
    readonly motioncontrols_direction_x: (a: number) => number;
    readonly motioncontrols_direction_y: (a: number) => number;
    readonly motioncontrols_grip: (a: number) => number;
    readonly motioncontrols_impulse: (a: number) => number;
    readonly motioncontrols_throttle: (a: number) => number;
    readonly orbitstate_alpha: (a: number) => number;
    readonly orbitstate_s: (a: number) => number;
    readonly orbitstate_theta: (a: number) => number;
    readonly playerstate_c_im: (a: number) => number;
    readonly playerstate_c_re: (a: number) => number;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
