

export interface DistanceFieldMeta {
  resolution: number;
  real_range: [number, number];
  imag_range: [number, number];
  max_distance: number;
  slowdown_threshold: number;
}

export interface OrbitRuntimeConfig {
  kBands: number;
  residualCap: number;
  residualOmegaScale: number;
  baseOmega: number;
  seed: number;
  dfBinaryPath: string;
  dfMetaPath: string;
  dStar: number;
  maxStep: number;
}

export interface OrbitDebug {
  distance: number;
  gradX: number;
  gradY: number;
  gradNorm: number;
  velocityScale: number;
  h: number;
  maxStep: number;
  dStar: number;
}

export class OrbitRuntime {
  private ready = false;
  private orbitState: any | null = null;
  private residualParams: any | null = null;
  private distanceField: any | null = null;
  private debug: OrbitDebug | null = null;
  private config: OrbitRuntimeConfig | null = null;

  // runtime guards to avoid reentrant WASM borrows and high-frequency sampling
  private inStep: boolean = false; // true while performing a step to avoid nested DF borrows
  private samplingInFlight: boolean = false; // true while an async sampling task is pending
  private samplingCooldownUntil: number = 0; // timestamp in ms until which sampling is suppressed after an error
  private lastSampleTime: number = 0; // timestamp of last scheduled sample
  private samplingBorrowing: boolean = false; // true while a sampling callback is actively borrowing the DF (prevents DF use in step_advanced)
  private samplingFailures: number = 0; // count sampling errors
  private stepAdvancedFailures: number = 0; // count step_advanced errors (ownership/borrow failures)
  private lastC: { real: number; imag: number } = { real: 0.0, imag: 0.0 }; // last known good complex parameter
  // Diagnostics counters
  private steppingFailures: number = 0; // any stepping/synthesize errors
  private steppingCooldownUntil: number = 0; // timestamp preventing stepping attempts
  private steppingFailureStreak: number = 0; // consecutive stepping failures for exponential backoff
  
  getLastC(): { real: number; imag: number } {
    return { ...this.lastC };
  }

  async initialize(config: OrbitRuntimeConfig): Promise<void> {
    if (!this.ready) {
      const wasmModInit = (await import("wasm/orbit_synth_wasm.js")) as any;
      if (wasmModInit?.default) {
        await wasmModInit.default();
      }
      this.ready = true;
    }
    this.config = config;

    // fetch named exports dynamically (bypass tsc resolution oddities)
    const wasmMod = (await import("wasm/orbit_synth_wasm.js")) as any;
    const { DistanceField: _DistanceField, OrbitState: _OrbitState, ResidualParams: _ResidualParams, constants: _constants } = wasmMod;

    // initialize residual params and orbit state using wasm bindings
    this.residualParams = new _ResidualParams(
      config.kBands,
      config.residualCap,
      1.0
    );
    // Prefer explicit seeded constructor when available (wasm naming varies by build)
    if (typeof _OrbitState.new_with_seed === 'function') {
      // Use BigInt for the wasm generated function that expects a u64
      const seedBig = BigInt(config.seed ?? 0);
      this.orbitState = _OrbitState.new_with_seed(1, 0, 0.0, config.baseOmega, 1.02, 0.3, config.kBands, config.residualOmegaScale, seedBig);
    } else if (typeof _OrbitState.newDefault === 'function') {
      this.orbitState = _OrbitState.newDefault(config.seed);
    } else {
      // Fallback to plain constructor
      this.orbitState = new _OrbitState(1, 0, 0.0, config.baseOmega, 1.02, 0.3, config.kBands, config.residualOmegaScale);
      if (this.orbitState && typeof this.orbitState.reset === 'function') {
        try { this.orbitState.reset(config.seed); } catch (e) { /* best-effort */ }
      }
    }
    this.orbitState.omega = config.baseOmega;

    if (config.dfBinaryPath && config.dfMetaPath) {
      const df = await this.loadDistanceField(
        config.dfBinaryPath,
        config.dfMetaPath
      );
      this.distanceField = df;
    } else {
      this.distanceField = null;
    }

    const constants = _constants();
    if (constants.default_k_residuals !== config.kBands) {
      console.warn(
        `[OrbitRuntime] kBands mismatch: wasm=${constants.default_k_residuals} config=${config.kBands}`
      );
    }
  }

  isReady(): boolean {
    return this.ready && this.orbitState !== null && this.residualParams !== null;
  }

  hasDistanceField(): boolean {
    return this.distanceField !== null;
  }

  getDebug(): OrbitDebug | null {
    return this.debug;
  }

  async reset(seed: number): Promise<void> {
    if (!this.ready) return;
    const wasmMod = (await import("wasm/orbit_synth_wasm.js")) as any;
    const { OrbitState: _OrbitState } = wasmMod;
    this.orbitState = _OrbitState.newDefault(seed);
    if (this.config) {
      this.orbitState.omega = this.config.baseOmega;
    }
  }

  step(
    dt: number,
    bandGates: number[],
    h: number
  ): { real: number; imag: number } {
    if (!this.orbitState || !this.residualParams) {
      throw new Error("Orbit runtime not initialized.");
    }

    const cfg = this.config;
    const dStar = cfg?.dStar ?? 0.3;
    const maxStep = cfg?.maxStep ?? 0.03;

    // Decide whether we are allowed to pass the DistanceField into step_advanced.
    // If a step is already in progress, avoid passing the DF to prevent recursive borrows
    // which produce "recursive use of an object detected" runtime errors in WASM.
    const allowDfUse = !!this.distanceField && !this.inStep && !this.samplingBorrowing && Date.now() >= this.samplingCooldownUntil;

    // If sampling is actively borrowing or we're in a stepping cooldown from recent failures,
    // skip any attempt to call into WASM synth/step/step_advanced and return last-known c.
    const nowTs = Date.now();
    if (this.samplingBorrowing || nowTs < this.steppingCooldownUntil) {
      // Fast-return last known good c
      return { real: this.lastC.real, imag: this.lastC.imag };
    }

    let c;

    // Mark that we are executing a step so nested calls know not to pass the DF.
    this.inStep = true;
    try {
      try {
        if (allowDfUse) {
          c = this.orbitState.step_advanced(
            dt,
            this.residualParams,
            bandGates,
            h,
            dStar,
            maxStep,
            this.distanceField
          );
        } else {
          // No DF available or unsafe to use it right now — use DF-less stepping
          const cCurrent = this.orbitState.synthesize(this.residualParams, bandGates);
          const cProposed = this.orbitState.step(dt, this.residualParams, bandGates);
          let dx = cProposed.real - cCurrent.real;
          let dy = cProposed.imag - cCurrent.imag;
          const mag = Math.sqrt(dx * dx + dy * dy);
          if (mag > maxStep && mag > 0) {
            const scale = maxStep / mag;
            dx *= scale;
            dy *= scale;
          }
          c = { real: cCurrent.real + dx, imag: cCurrent.imag + dy };
        }
      } catch (e) {
        // Any wasm stepping/synthesizing error (borrow/ownership) is handled here.
        // Throttle logs for frequent stepping failures to avoid noisy spamming
        if ((this.steppingFailureStreak || 0) % 5 === 0) {
          console.warn('[OrbitRuntime] stepping failed (synthesize/step/step_advanced):', e);
        }
        this.stepAdvancedFailures += 1;
        this.steppingFailures += 1;
        // extended cooldown to avoid immediate re-use after ownership errors
        this.samplingCooldownUntil = Date.now() + 5000;
        // If failures are frequent, move to a long pause to avoid constant churn
        // Increase consecutive failure streak and set exponential backoff for stepping
        this.steppingFailureStreak = (this.steppingFailureStreak || 0) + 1;
        const backoffMs = Math.min(60000, 1000 * Math.pow(2, Math.max(0, this.steppingFailureStreak - 1)));
        this.steppingCooldownUntil = Date.now() + backoffMs;
        // Also maintain the sampling cooldown to avoid immediate DF re-use
        this.samplingCooldownUntil = Math.max(this.samplingCooldownUntil, Date.now() + 5000);
        // Throttle the detailed backoff warning to once every few failures
        if ((this.steppingFailureStreak || 0) % 5 === 0) {
          console.warn(`[OrbitRuntime] stepping failure streak=${this.steppingFailureStreak}, backing off for ${backoffMs}ms`);
        }

        // Provide conservative fallback: return last known c (no time advance)
        c = { real: this.lastC.real, imag: this.lastC.imag };
      }

      // On success, reset stepping failure streak/backoff so we resume normal operation
      if (this.steppingFailureStreak > 0 && this.stepAdvancedFailures === 0) {
        this.steppingFailureStreak = 0;
        this.steppingCooldownUntil = 0;
      }
    } finally {
      this.inStep = false;
    }

    // Update last known good c and schedule asynchronous sampling to compute debug info if DF present and sampling not suppressed.
    this.lastC = { real: c.real, imag: c.imag };
    const now = Date.now();
    if (this.distanceField && now >= this.samplingCooldownUntil) {
      // limit samples to avoid flooding and avoid overlapping samples
      const minSampleInterval = 100; // ms (avoid high-frequency sampling that can collide with stepping)
      if (!this.samplingInFlight && now - this.lastSampleTime > minSampleInterval) {
        this.samplingInFlight = true;
        this.lastSampleTime = now;

        // placeholder debug while sampling completes
        this.debug = {
          distance: 1.0,
          gradX: 0.0,
          gradY: 0.0,
          gradNorm: 0.0,
          velocityScale: 1.0,
          h,
          maxStep,
          dStar,
        };

        setTimeout(() => {
          try {
            if (!this.distanceField) return;
            // Mark that we're actively borrowing the DF during sampling so synchronous steps avoid it
            this.samplingBorrowing = true;
            const d = this.distanceField.sample_bilinear(c.real, c.imag);
            const grad = this.distanceField.gradient(c.real, c.imag);
            const gradX = grad[0] ?? 0;
            const gradY = grad[1] ?? 0;
            const gradNorm = Math.sqrt(gradX * gradX + gradY * gradY);
            // Some builds expose a velocity helper on the DF; call safely if present
            let velocityScale = 1.0;
            try {
              if (typeof (this.distanceField as any).get_velocity_scale === 'function') {
                velocityScale = (this.distanceField as any).get_velocity_scale(c.real, c.imag);
              }
            } catch (e) {
              // Best-effort: if the method exists but fails, fall back to 1.0
              console.warn('[OrbitRuntime] DistanceField.get_velocity_scale failed:', e);
            }
            this.debug = {
              distance: d,
              gradX,
              gradY,
              gradNorm,
              velocityScale,
              h,
              maxStep,
              dStar,
            };
          } catch (e) {
            console.warn('[OrbitRuntime] DistanceField sampling failed:', e);
            // record failure and backoff sampling for a short while to allow WASM to settle
            this.samplingFailures += 1;
            this.samplingCooldownUntil = Date.now() + 2000;
            this.debug = {
              distance: 1.0,
              gradX: 0.0,
              gradY: 0.0,
              gradNorm: 0.0,
              velocityScale: 1.0,
              h,
              maxStep,
              dStar,
            };
          } finally {
            this.samplingBorrowing = false;
            this.samplingInFlight = false;
          }
        }, 0);
      }
    } else {
      // No DF or sampling suppressed — provide default debug
      this.debug = {
        distance: 1.0,
        gradX: 0.0,
        gradY: 0.0,
        gradNorm: 0.0,
        velocityScale: 1.0,
        h,
        maxStep,
        dStar,
      };
    }

    return { real: c.real, imag: c.imag };
  }

  /**
   * Diagnostics about the DistanceField usage and recent failures.
   */
  getDistanceFieldDiagnostics(): string | null {
    if (!this.distanceField) return null;
    const now = Date.now();
    const samplingCooldownMs = Math.max(0, this.samplingCooldownUntil - now);
    const steppingCooldownMs = Math.max(0, this.steppingCooldownUntil - now);
    return `samplingFailures=${this.samplingFailures} stepAdvancedFailures=${this.stepAdvancedFailures} samplingCooldownMs=${samplingCooldownMs} steppingFailureStreak=${this.steppingFailureStreak} steppingCooldownMs=${steppingCooldownMs}`;
  }

  updateState(params: { s: number; alpha: number; omega: number }): void {
    if (!this.orbitState) return;
    this.orbitState.s = params.s;
    this.orbitState.alpha = params.alpha;
    this.orbitState.omega = params.omega;
  }

  setLobe(lobe: number): void {
    if (!this.orbitState) return;
    this.orbitState.lobe = lobe;
  }

  getLobe(): number {
    return this.orbitState?.lobe ?? 1;
  }

  private async loadDistanceField(
    binaryPath: string,
    metaPath: string
  ): Promise<any> {
    const metaResp = await fetch(metaPath);
    if (!metaResp.ok) {
      throw new Error(`Distance field metadata not found: ${metaPath}`);
    }
    const meta = (await metaResp.json()) as DistanceFieldMeta;

    const dataResp = await fetch(binaryPath);
    if (!dataResp.ok) {
      throw new Error(`Distance field binary not found: ${binaryPath}`);
    }
    const buf = await dataResp.arrayBuffer();
    const field = new Float32Array(buf);

    const wasmMod = (await import("wasm/orbit_synth_wasm.js")) as any;
    const { DistanceField: _DistanceField } = wasmMod;

    return new _DistanceField(
      field,
      meta.resolution,
      meta.real_range[0],
      meta.real_range[1],
      meta.imag_range[0],
      meta.imag_range[1],
      meta.max_distance,
      meta.slowdown_threshold
    );
  }
}
