

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
    this.orbitState = _OrbitState.newDefault(config.seed);
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

    let c;
    if (this.distanceField) {
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
      const cCurrent = this.orbitState.synthesize(
        this.residualParams,
        bandGates
      );
      const cProposed = this.orbitState.step(
        dt,
        this.residualParams,
        bandGates
      );
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

    if (this.distanceField) {
      const d = this.distanceField.sample_bilinear(c.real, c.imag);
      const grad = this.distanceField.gradient(c.real, c.imag);
      const gradX = grad[0] ?? 0;
      const gradY = grad[1] ?? 0;
      const gradNorm = Math.sqrt(gradX * gradX + gradY * gradY);
      const velocityScale = this.distanceField.get_velocity_scale(c.real, c.imag);
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
    } else {
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
      meta.real_range,
      meta.imag_range,
      meta.max_distance,
      meta.slowdown_threshold
    );
  }
}
