/**
 * ONNX.js model inference wrapper.
 */

import * as ort from 'onnxruntime-web';

type Complex = { real: number; imag: number };
type StepControllerModule = any;

export interface VisualParameters {
  juliaReal: number;
  juliaImag: number;
  colorHue: number;
  colorSat: number;
  colorBright: number;
  zoom: number;
  speed: number;
}

export interface ModelMetadata {
  input_shape: number[];
  output_dim: number;
  parameter_names: string[];
  parameter_ranges: Record<string, [number, number]>;
  feature_mean?: number[];
  feature_std?: number[];
  epoch?: number;
  window_frames?: number;
  input_dim?: number;
  timestamp?: string;
  git_hash?: string;
  model_type?: string; // 'orbit_control' or legacy
  k_bands?: number;
}

export interface PerformanceMetrics {
  lastInferenceTime: number; // milliseconds
  averageInferenceTime: number; // rolling average
  normalizationTime: number;
  inferenceTime: number;
  postProcessingTime: number;
}

export class ModelInference {
  private session: ort.InferenceSession | null = null;
  private metadata: ModelMetadata | null = null;
  private featureMean: Float32Array | null = null;
  private featureStd: Float32Array | null = null;
  
  // Step-based controller state
  private stepControllerModule: StepControllerModule | null = null;
  private stepController: any | null = null;
  private currentC: Complex = { real: 0.0, imag: 0.0 };
  private prevDelta: Complex = { real: 0.0, imag: 0.0 };
  private controllerContextLen: number | null = null;
  private isStepModel: boolean = true;
  
  // Audio-reactive post-processing toggle (MR #8 / commit 75c1a43)
  private useAudioReactivePostProcessing: boolean = true;
  
  // Performance tracking
  private inferenceTimings: number[] = [];
  private maxTimingHistory: number = 100;
  private lastMetrics: PerformanceMetrics = {
    lastInferenceTime: 0,
    averageInferenceTime: 0,
    normalizationTime: 0,
    inferenceTime: 0,
    postProcessingTime: 0
  };

  /**
   * Enable or disable audio-reactive post-processing (MR #8 / commit 75c1a43).
   * When enabled, mixes model outputs with raw audio features for dynamic visuals.
   * When disabled, uses only model outputs with basic normalization.
   */
  setAudioReactivePostProcessing(enabled: boolean): void {
    this.useAudioReactivePostProcessing = enabled;
    console.log(`[ModelInference] Audio-reactive post-processing ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Load ONNX model and metadata.
   */
  async loadModel(modelPath: string | ArrayBuffer | Uint8Array | Blob, metadataPath?: string): Promise<void> {
    // Simple WASM backend configuration: use a single-threaded, non-SIMD runtime by default so
    // the WASM initialization is deterministic across browsers (avoids cryptic multi-thread/SIMD failures).
    ort.env.wasm.wasmPaths = '/';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = false;

    // Normalize input: allow callers to pass bytes (ArrayBuffer/Uint8Array/Blob) or a URL string.
    let modelBytes: Uint8Array | undefined;
    let isLikelyUrl = false;

    if (typeof modelPath === 'string') {
      isLikelyUrl = modelPath.startsWith('http') || modelPath.startsWith('/') || modelPath.startsWith('blob:');
    } else if (modelPath instanceof Uint8Array) {
      modelBytes = modelPath as Uint8Array;
    } else if (modelPath instanceof ArrayBuffer) {
      modelBytes = new Uint8Array(modelPath);
    } else if (typeof Blob !== 'undefined' && modelPath instanceof Blob) {
      modelBytes = new Uint8Array(await modelPath.arrayBuffer());
    }

    // If we still don't have bytes and we have a URL-like path, fetch and validate the bytes (helps detect 404 HTML pages served as binaries)
    try {
      if (!modelBytes && isLikelyUrl) {
        const resp = await fetch(modelPath as string, { credentials: 'same-origin' });
        if (!resp.ok) {
          throw new Error(`Failed to fetch model (${resp.status} ${resp.statusText})`);
        }

        const contentType = (resp.headers.get('Content-Type') || '').toLowerCase();
        const ab = await resp.arrayBuffer();
        if (ab.byteLength < 128) {
          const snippet = new TextDecoder().decode(new Uint8Array(ab.slice(0, Math.min(128, ab.byteLength))));
          throw new Error(`Fetched model is too small (${ab.byteLength} bytes), likely not an ONNX binary. Snippet: ${snippet}`);
        }

        const head = new TextDecoder().decode(new Uint8Array(ab.slice(0, 64)));
        if (contentType.includes('text') || head.trim().startsWith('<') || head.trim().startsWith('{') || head.trim().startsWith('Error')) {
          throw new Error(`Fetched model appears to be non-binary (Content-Type: ${contentType}). Snippet: ${head.substring(0, 120)}`);
        }

        modelBytes = new Uint8Array(ab);
      }

      // Build session creation options (canonical: wasm EP only — no external .data sidecar support)
      const sessionOptions: any = { executionProviders: ['wasm'] };

      // Try creating the session. If we have bytes, pass them directly.
      if (modelBytes) {
        this.session = await ort.InferenceSession.create(modelBytes, sessionOptions);
      } else {
        this.session = await ort.InferenceSession.create(modelPath as string, sessionOptions);
      }
    } catch (err) {
      // Surface a clear error and do not attempt silent retries or configuration changes.
      throw new Error(`WASM session initialization failed: ${String(err)}`);
    }

    // Load metadata if provided
    if (metadataPath) {
      try {
        const response = await fetch(metadataPath, { credentials: 'same-origin' });
        if (!response.ok) throw new Error(`Failed to fetch metadata (${response.status} ${response.statusText})`);
        const contentType = (response.headers.get('Content-Type') || '').toLowerCase();
        if (!contentType.includes('json') && !contentType.includes('application')) {
          console.warn(`[ModelInference] metadata Content-Type looks suspicious: ${contentType}`);
        }
        this.metadata = await response.json() as ModelMetadata;

        this.isStepModel = this.metadata.model_type !== 'legacy';
        console.log(`[ModelInference] Loaded ${this.isStepModel ? 'step-based' : 'legacy'} model`);

        // Set up normalization
        if (this.metadata.feature_mean && this.metadata.feature_std) {
          this.featureMean = new Float32Array(this.metadata.feature_mean);
          this.featureStd = new Float32Array(this.metadata.feature_std);
        }
      } catch (error) {
        console.warn('Failed to load metadata:', error);
      }
    }
  }

  /**
   * Run inference on audio features with latency tracking.
   */
  async infer(features: number[]): Promise<VisualParameters> {
    if (!this.session) {
      throw new Error('Model not loaded');
    }

    if (!this.stepControllerModule || !this.stepController) {
      await this.ensureStepController();
    }

    const totalStartTime = performance.now();
    let normStartTime = performance.now();

    const stepControllerModule = this.stepControllerModule;
    const stepController = this.stepController;

    const contextFeatures = this.buildControllerContext();
    const fullFeatures = this.applyInputCompatibility([...features, ...contextFeatures]);
    let normalizedFeatures = new Float32Array(fullFeatures);
    if (this.featureMean && this.featureStd) {
      normalizedFeatures = new Float32Array(fullFeatures.length);
      for (let i = 0; i < fullFeatures.length; i++) {
        const mean = this.featureMean[i] || 0;
        const std = this.featureStd[i] || 1;
        normalizedFeatures[i] = (fullFeatures[i] - mean) / (std + 1e-8);
      }
    }

    const normTime = performance.now() - normStartTime;

    // Prepare input tensor
    const inputTensor = new ort.Tensor(
      'float32',
      normalizedFeatures,
      [1, normalizedFeatures.length]
    );

    // Run inference
    const inferStartTime = performance.now();
    const feeds = { audio_features: inputTensor };
    const results = await this.session.run(feeds);
    const inferTime = performance.now() - inferStartTime;

    const outputTensor = results.visual_parameters;
    const params = Array.from(outputTensor.data as Float32Array);

    // Post-processing
    const postStartTime = performance.now();

    let visualParams: VisualParameters;

    if (this.isStepModel && stepControllerModule && stepController) {
      const deltaModel: Complex = {
        real: params[0] ?? 0.0,
        imag: params[1] ?? 0.0
      };
      const result = stepController.applyStep(
        this.currentC.real,
        this.currentC.imag,
        deltaModel.real,
        deltaModel.imag,
        this.prevDelta.real,
        this.prevDelta.imag
      );
      this.prevDelta = { real: result.delta_real, imag: result.delta_imag };
      this.currentC = { real: result.c_next_real, imag: result.c_next_imag };

      const numFeatures = 6;
      const windowFrames = Math.floor(features.length / numFeatures);
      let avgRMS = 0, avgOnset = 0;
      for (let i = 0; i < windowFrames; i++) {
        avgRMS += features[i * numFeatures + 2];
        avgOnset += features[i * numFeatures + 4];
      }
      avgRMS /= windowFrames;
      avgOnset /= windowFrames;
      const currentHue = (avgRMS * 2.0) % 1.0;
      visualParams = {
        juliaReal: this.currentC.real,
        juliaImag: this.currentC.imag,
        colorHue: currentHue,
        colorSat: Math.max(0.5, Math.min(1.0, 0.7 + avgOnset * 0.3)),
        colorBright: Math.max(0.5, Math.min(0.9, 0.6 + avgRMS * 0.3)),
        zoom: Math.max(1.5, Math.min(4.0, 2.5)),
        speed: Math.max(0.3, Math.min(0.7, (Math.abs(result.delta_real) + Math.abs(result.delta_imag)) / 0.04))
      };
    } else {
      // LEGACY VISUAL PARAMETER MODEL
      visualParams = {
        juliaReal: params[0],
        juliaImag: params[1],
        colorHue: params[2],
        colorSat: params[3],
        colorBright: params[4],
        zoom: params[5],
        speed: params[6]
      };

      if (this.useAudioReactivePostProcessing) {
        // AUDIO-REACTIVE POST-PROCESSING (MR #8 / commit 75c1a43)
        const numFeatures = 6;
        const windowFrames = Math.floor(features.length / numFeatures);

        if (features.length % numFeatures !== 0) {
          console.warn(
            `[modelInference] features.length (${features.length}) is not a multiple of numFeatures (${numFeatures}). ` +
              `Using ${windowFrames} full frames.`
          );
        }
        
        // Average each feature type across the window
        let avgCentroid = 0, avgFlux = 0, avgRMS = 0, avgZCR = 0, avgOnset = 0, avgRolloff = 0;
        for (let i = 0; i < windowFrames; i++) {
          avgCentroid += features[i * numFeatures + 0];
          avgFlux += features[i * numFeatures + 1];
          avgRMS += features[i * numFeatures + 2];
          avgZCR += features[i * numFeatures + 3];
          avgOnset += features[i * numFeatures + 4];
          avgRolloff += features[i * numFeatures + 5];
        }
        avgCentroid /= windowFrames;
        avgFlux /= windowFrames;
        avgRMS /= windowFrames;
        avgZCR /= windowFrames;
        avgOnset /= windowFrames;
        avgRolloff /= windowFrames;
        
        // Color: Map RMS (loudness) to hue cycling, onset to saturation
        visualParams.colorHue = (params[2] + avgRMS * 2.0) % 1.0;
        visualParams.colorSat = Math.max(0.5, Math.min(1.0, 0.7 + avgOnset * 0.3));
        visualParams.colorBright = Math.max(0.5, Math.min(0.9, 0.6 + avgRMS * 0.3));
      } else {
        // Color: enforce minimum saturation
        visualParams.colorHue = visualParams.colorHue % 1.0;
        visualParams.colorSat = Math.max(0.5, Math.min(1, visualParams.colorSat * 0.8 + 0.5));
        visualParams.colorBright = Math.max(0.6, Math.min(0.9, visualParams.colorBright * 0.5 + 0.5));
      }
      
      // ORIGINAL POST-PROCESSING (pre-MR #8)
      visualParams.juliaReal = (visualParams.juliaReal * 0.6) % 1.4 - 0.7;
      visualParams.juliaImag = (visualParams.juliaImag * 0.6) % 1.4 - 0.7;

      // Zoom: stay zoomed IN (1.5-4.0 for visible detail)
      visualParams.zoom = Math.max(1.5, Math.min(4.0, visualParams.zoom * 2 + 1.5));
      visualParams.speed = Math.max(0.3, Math.min(0.7, visualParams.speed));
    }

    const postTime = performance.now() - postStartTime;
    const totalTime = performance.now() - totalStartTime;

    // Track metrics
    this.inferenceTimings.push(totalTime);
    if (this.inferenceTimings.length > this.maxTimingHistory) {
      this.inferenceTimings.shift();
    }

    const avgTime = this.inferenceTimings.reduce((a, b) => a + b, 0) / this.inferenceTimings.length;

    this.lastMetrics = {
      lastInferenceTime: totalTime,
      averageInferenceTime: avgTime,
      normalizationTime: normTime,
      inferenceTime: inferTime,
      postProcessingTime: postTime
    };

    // Debug log every 60 frames (~1 second at 60fps), but skip first frame
    if (this.inferenceTimings.length > 60 && this.inferenceTimings.length % 60 === 0) {
      console.log('[ModelInference] Visual params:', {
        julia: [visualParams.juliaReal.toFixed(3), visualParams.juliaImag.toFixed(3)],
        color: [visualParams.colorHue.toFixed(3), visualParams.colorSat.toFixed(3), visualParams.colorBright.toFixed(3)],
        zoom: visualParams.zoom.toFixed(3),
        speed: visualParams.speed.toFixed(3),
        modelType: this.isStepModel ? 'step_control' : 'legacy'
      });
    }

    return visualParams;
  }

  /**
   * Get performance metrics.
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.lastMetrics };
  }

  /**
   * Get model metadata.
   */
  getMetadata(): ModelMetadata | null {
    return this.metadata;
  }

  /**
   * Check if model is loaded.
   */
  isLoaded(): boolean {
    return this.session !== null;
  }
  
  private buildControllerContext(): number[] {
    if (!this.stepControllerModule || !this.stepController) {
      return [];
    }
    const mip = this.stepControllerModule.mipForDelta(
      this.prevDelta.real,
      this.prevDelta.imag
    );
    const context = this.stepController.contextFeatures(
      this.currentC.real,
      this.currentC.imag,
      this.prevDelta.real,
      this.prevDelta.imag,
      mip
    );
    const vector = context.feature_vector as number[];
    if (this.controllerContextLen && vector.length !== this.controllerContextLen) {
      if (vector.length < this.controllerContextLen) {
        return vector.concat(new Array(this.controllerContextLen - vector.length).fill(0.0));
      }
      return vector.slice(0, this.controllerContextLen);
    }
    return vector;
  }

  private applyInputCompatibility(features: number[]): number[] {
    const expectedDim = this.metadata?.input_dim ?? this.metadata?.input_shape?.[1];
    if (!expectedDim || expectedDim <= 0) {
      return features;
    }
    if (features.length === expectedDim) {
      return features;
    }
    if (features.length < expectedDim) {
      return features.concat(new Array(expectedDim - features.length).fill(0.0));
    }
    return features.slice(0, expectedDim);
  }

  private async ensureStepController(): Promise<void> {
    if (this.stepControllerModule && this.stepController) {
      return;
    }
    // @ts-ignore - runtime-provided wasm module
    const module = await import(/* @vite-ignore */ '/wasm/orbit_synth_wasm.js');
    await module.default();
    this.stepControllerModule = module;
    this.stepController = new module.StepController();
    this.controllerContextLen = module.controllerContextLen();
  }
}
