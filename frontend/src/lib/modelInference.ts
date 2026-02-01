/**
 * ONNX.js model inference wrapper.
 */

import * as ort from 'onnxruntime-web';
import { createInitialStepState, loadStepController, type StepController, type StepState } from './stepController';

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
  model_type?: string; // 'step_control' or legacy
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
  
  // Step-based controller (new architecture)
  private stepController: StepController | null = null;
  private stepState: StepState | null = null;
  private isStepModel: boolean = false;
  
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

  // Telemetry: if enabled, we will POST aggregated telemetry to the local backend
  private telemetryEnabled: boolean = false;
  private telemetryCounter: number = 0;

  /**
   * Enable or disable server-side telemetry logging. When enabled, the frontend will
   * periodically post compact telemetry JSON to POST /api/telemetry.
   */
  setTelemetryEnabled(enabled: boolean): void {
    this.telemetryEnabled = enabled;
    console.log(`[ModelInference] telemetry ${enabled ? 'enabled' : 'disabled'}`);

    // Send an immediate ping to verify connectivity when enabling telemetry
    if (enabled) {
      try {
        const payload = { type: 'ping', ts: new Date().toISOString(), model_type: this.metadata?.model_type ?? null };
        void this.sendTelemetry(payload);
        console.debug('[ModelInference] Sent telemetry ping', payload);
      } catch (e) {
        console.warn('[ModelInference] telemetry ping failed', e);
      }
    }
  }

  private async sendTelemetry(payload: any): Promise<void> {
    if (typeof fetch === 'undefined') return;
    try {
      await fetch('/api/telemetry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    } catch (e) {
      console.warn('[ModelInference] telemetry send failed', e);
    }
  }

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

        // Check if this is a step-based control model
        this.isStepModel = this.metadata.model_type === 'step_control';

        if (this.isStepModel) {
          this.stepController = await loadStepController();
          this.stepState = await createInitialStepState();
          console.log('[ModelInference] Loaded step-based control model');
        } else {
          console.log('[ModelInference] Loaded legacy visual parameter model');
        }

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

    const totalStartTime = performance.now();
    let normStartTime = performance.now();

    const inputFeatures = this.buildModelInput(features);

    // Normalize features if normalization stats are available
    let normalizedFeatures = new Float32Array(inputFeatures.length);
    if (this.featureMean && this.featureStd) {
      for (let i = 0; i < inputFeatures.length; i++) {
        const mean = this.featureMean[i] || 0;
        const std = this.featureStd[i] || 1;
        normalizedFeatures[i] = (inputFeatures[i] - mean) / (std + 1e-8);
      }
    } else {
      normalizedFeatures = new Float32Array(inputFeatures);
    }

    const normTime = performance.now() - normStartTime;

    // Prepare input tensor
    const inputTensor = new ort.Tensor('float32', normalizedFeatures, [1, normalizedFeatures.length]);

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

    if (this.isStepModel && this.stepController && this.stepState) {
      // NEW STEP-BASED CONTROL MODEL
      const dx = params.length > 0 ? params[0] : 0;
      const dy = params.length > 1 ? params[1] : 0;

      // The WASM controller exposes `applyStep(...)` which receives numeric
      // state fields rather than a `StepState` object. Call it directly and
      // update the local stepState with the returned result.
      const result = this.stepController.step(this.stepState, dx, dy);

      // Update local state from the returned result so future steps are
      // applied relative to the current orbit state.
      if (result) {
        // Defensive checks in case WASM returns an unexpected shape
        this.stepState.c_real = typeof result.c_real === 'number' ? result.c_real : this.stepState.c_real;
        this.stepState.c_imag = typeof result.c_imag === 'number' ? result.c_imag : this.stepState.c_imag;
        this.stepState.prev_delta_real = typeof result.delta_real === 'number' ? result.delta_real : this.stepState.prev_delta_real;
        this.stepState.prev_delta_imag = typeof result.delta_imag === 'number' ? result.delta_imag : this.stepState.prev_delta_imag;
      }

      // Extract audio features for color mapping
      const numFeatures = 6;
      const windowFrames = Math.floor(features.length / numFeatures);
      let avgRMS = 0, avgOnset = 0;
      for (let i = 0; i < windowFrames; i++) {
        avgRMS += features[i * numFeatures + 2];
        avgOnset += features[i * numFeatures + 4];
      }
      avgRMS /= windowFrames;
      avgOnset /= windowFrames;
      // Expose a telemetry-friendly value for logging
      (this as any)._lastAvgRMS = avgRMS;

      const stepMag = Math.hypot(this.stepState.prev_delta_real, this.stepState.prev_delta_imag);
      const currentHue = (avgRMS * 2.0) % 1.0;
      visualParams = {
        juliaReal: this.stepState.c_real,
        juliaImag: this.stepState.c_imag,
        colorHue: currentHue,
        colorSat: Math.max(0.5, Math.min(1.0, 0.7 + avgOnset * 0.3)),
        colorBright: Math.max(0.5, Math.min(0.9, 0.6 + avgRMS * 0.3)),
        zoom: 2.5,
        speed: Math.max(0.3, Math.min(0.9, stepMag * 10.0))
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
        // Expose for telemetry
        (this as any)._lastAvgRMS = avgRMS;
        
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

      // If telemetry is enabled, periodically POST a compact record to the backend
      if (this.telemetryEnabled) {
        this.telemetryCounter += 1;
        if (this.telemetryCounter % 30 === 0) { // ~0.5s at 60fps
          try {
            const payload = this.assembleTelemetryEntry(params);
            // Fire-and-forget; failures are non-fatal
            void this.sendTelemetry(payload);
          } catch (e) {
            console.warn('[ModelInference] telemetry assembly failed', e);
          }
        }
      }
    }

    return visualParams;
  }

  private buildModelInput(features: number[]): number[] {
    let combined = [...features];

    if (this.isStepModel) {
      if (this.stepController && this.stepState) {
        const context = this.stepController.context(this.stepState);
        combined = combined.concat(context.feature_vec ?? []);
      } else {
        combined = combined.concat(new Array(265).fill(0));
      }
    }

    return this.adjustInputLength(combined);
  }

  private adjustInputLength(features: number[]): number[] {
    const inputShape = this.metadata?.input_shape;
    const expected = this.metadata?.input_dim ?? (inputShape ? inputShape[inputShape.length - 1] : undefined);
    if (!expected || expected <= 0) {
      return features;
    }
    if (features.length > expected) {
      return features.slice(0, expected);
    }
    if (features.length < expected) {
      return features.concat(new Array(expected - features.length).fill(0));
    }
    return features;
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
   * Assemble a compact telemetry payload including model proposals and minimap context.
   * This is public to make it testable.
   */
  assembleTelemetryEntry(params: number[]): Record<string, any> {
    const model_dx = (params.length > 0 ? params[0] : 0);
    const model_dy = (params.length > 1 ? params[1] : 0);

    // Base applied delta from state
    const applied_delta = Math.hypot(this.stepState?.prev_delta_real ?? 0, this.stepState?.prev_delta_imag ?? 0) || 0;

    // Attempt to extract minimap context if available
    let c_real: number | null = this.stepState?.c_real ?? null;
    let c_imag: number | null = this.stepState?.c_imag ?? null;
    let prev_dx: number | null = this.stepState?.prev_delta_real ?? null;
    let prev_dy: number | null = this.stepState?.prev_delta_imag ?? null;
    let nu_norm: number | null = null;
    let membership: number | null = null;
    let grad_re: number | null = null;
    let grad_im: number | null = null;
    let sensitivity: number | null = null;
    let patch_mean: number | null = null;
    let patch_max: number | null = null;

    if (this.stepController && this.stepState && typeof (this.stepController as any).context === 'function') {
      try {
        const ctx = (this.stepController as any).context(this.stepState);
        if (ctx && Array.isArray(ctx.feature_vec) && ctx.feature_vec.length >= 9) {
          const fv: number[] = ctx.feature_vec;
          c_real = typeof fv[0] === 'number' ? fv[0] : c_real;
          c_imag = typeof fv[1] === 'number' ? fv[1] : c_imag;
          prev_dx = typeof fv[2] === 'number' ? fv[2] : prev_dx;
          prev_dy = typeof fv[3] === 'number' ? fv[3] : prev_dy;
          nu_norm = typeof fv[4] === 'number' ? fv[4] : null;
          membership = typeof fv[5] === 'number' ? fv[5] : null;
          grad_re = typeof fv[6] === 'number' ? fv[6] : null;
          grad_im = typeof fv[7] === 'number' ? fv[7] : null;
          sensitivity = typeof fv[8] === 'number' ? fv[8] : null;

          const patch = fv.slice(9);
          if (patch.length > 0) {
            let sum = 0;
            let maxv = -Infinity;
            for (let i = 0; i < patch.length; i++) {
              const v = patch[i];
              sum += v;
              if (v > maxv) maxv = v;
            }
            patch_mean = sum / patch.length;
            patch_max = maxv === -Infinity ? null : maxv;
          }
        }
      } catch (e) {
        // ignore context errors for telemetry
      }
    }

    return {
      ts: new Date().toISOString(),
      model_dx,
      model_dy,
      applied_delta,
      c_real,
      c_imag,
      prev_dx,
      prev_dy,
      nu_norm,
      membership,
      grad_re,
      grad_im,
      sensitivity,
      patch_mean,
      patch_max,
      avg_rms: (this as any)._lastAvgRMS ?? null,
      model_type: this.metadata?.model_type ?? null
    };
  }

  /**
   * Check if model is loaded.
   */
  isLoaded(): boolean {
    return this.session !== null;
  }
}
