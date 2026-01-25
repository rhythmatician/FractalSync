/**
 * ONNX.js model inference wrapper.
 */

import * as ort from 'onnxruntime-web';
import { OrbitRuntime } from './orbitRuntime';
import {
  DEFAULT_K_BANDS,
  INPUT_DIM,
  MODEL_INPUT_NAME,
  MODEL_OUTPUT_NAME,
  OUTPUT_NAMES,
} from './modelContract';

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
  input_name?: string;
  output_dim: number;
  output_name?: string;
  parameter_names: string[];
  parameter_ranges: Record<string, [number, number]>;
  input_feature_names?: string[];
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
  
  // Orbit runtime (runtime-core stepping)
  private orbitRuntime: OrbitRuntime | null = null;
  private isControlModel: boolean = false;
  private kBands: number = DEFAULT_K_BANDS;
  private deterministicMode: boolean = false;
  private orbitSeed: number = 1337;
  private contourDStar: number = 0.3;
  private contourMaxStep: number = 0.03;
  private dfBinaryPath: string = "/mandelbrot_distance_field.bin";
  private dfMetaPath: string = "/mandelbrot_distance_field.json";
  private dfLoadError: string | null = null;
  private showMode: boolean = false;
  private residualCap: number = 0.5;
  private residualOmegaScale: number = 2.0;
  private baseOmega: number = 1.0;
  
  // Color-based section detection for lobe switching
  private colorHistory: number[] = [];
  private colorHistorySize: number = 120; // ~2 seconds at 60fps
  private lastLobeSwitch: number = 0;
  private lobeSwitchCooldown: number = 180; // ~3 seconds at 60fps (hysteresis)
  private colorChangeThreshold: number = 0.15; // Hue change threshold
  
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

  // Counters for inference-level failures
  private inferenceFailures: number = 0;

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
  async loadModel(modelPath: string, metadataPath?: string): Promise<void> {
    // Simple WASM backend configuration
    ort.env.wasm.wasmPaths = '/';
    
    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm']
    });

    // Load metadata if provided
    if (metadataPath) {
      try {
        const response = await fetch(metadataPath);
        this.metadata = await response.json() as ModelMetadata;
        
// Check if this is a control model and validate contract
        this.isControlModel = (this.metadata.parameter_names?.length ?? 0) === this.metadata.output_dim;
        this.kBands = this.metadata.k_bands || DEFAULT_K_BANDS;
        if (this.isControlModel && this.metadata.parameter_names) {
          const expected = OUTPUT_NAMES.slice(0, 3 + this.kBands);
          const matches = expected.every((name: string, idx: number) => this.metadata?.parameter_names?.[idx] === name);
          if (!matches) {
            throw new Error(`[ModelInference] Output parameter names do not match model contract`);
          }
          if (this.metadata.input_name && this.metadata.input_name !== MODEL_INPUT_NAME) {
            throw new Error(`[ModelInference] Input tensor name mismatch: ${this.metadata.input_name}`);
          }
          if (this.metadata.output_name && this.metadata.output_name !== MODEL_OUTPUT_NAME) {
            throw new Error(`[ModelInference] Output tensor name mismatch: ${this.metadata.output_name}`);
          }
        }

        // Load show control configuration
        try {
          const configResponse = await fetch('/show_control.json');
          if (configResponse.ok) {
            const config = await configResponse.json();
            this.deterministicMode = Boolean(config.replay_mode);
            this.orbitSeed = Number(config.orbit?.seed ?? this.orbitSeed);
            this.residualCap = Number(config.orbit?.residual_cap ?? this.residualCap);
            this.residualOmegaScale = Number(
              config.orbit?.residual_omega_scale ?? this.residualOmegaScale
            );
            this.baseOmega = Number(config.orbit?.base_omega ?? this.baseOmega);
            this.contourDStar = Number(config.contour?.d_star ?? this.contourDStar);
            this.contourMaxStep = Number(config.contour?.max_step ?? this.contourMaxStep);
            this.dfBinaryPath = String(config.df?.path ?? this.dfBinaryPath);
            this.dfMetaPath = String(config.df?.meta_path ?? this.dfMetaPath);
            this.showMode = Boolean(config.show_mode);
          }
        } catch (error) {
          console.warn('[ModelInference] Failed to load show control config, using defaults', error);
        }

        if (this.isControlModel) {
          const runtime = new OrbitRuntime();
          try {
            await runtime.initialize({
              kBands: this.kBands,
              residualCap: this.residualCap,
              residualOmegaScale: this.residualOmegaScale,
              baseOmega: this.baseOmega,
              seed: this.orbitSeed,
              dfBinaryPath: this.dfBinaryPath,
              dfMetaPath: this.dfMetaPath,
              dStar: this.contourDStar,
              maxStep: this.contourMaxStep
            });
            this.orbitRuntime = runtime;
            console.log('[ModelInference] Loaded orbit control model with runtime-core stepping');
          } catch (error) {
            this.dfLoadError = error instanceof Error ? error.message : String(error);
            console.error('[ModelInference] Distance field failed to load:', this.dfLoadError);
            if (this.showMode) {
              throw new Error(`[ModelInference] Distance field required in show mode: ${this.dfLoadError}`);
            }
            // Fall back to runtime without DF
            await runtime.initialize({
              kBands: this.kBands,
              residualCap: this.residualCap,
              residualOmegaScale: this.residualOmegaScale,
              baseOmega: this.baseOmega,
              seed: this.orbitSeed,
              dfBinaryPath: '',
              dfMetaPath: '',
              dStar: this.contourDStar,
              maxStep: this.contourMaxStep
            });
            this.orbitRuntime = runtime;
          }
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
    // Outer guard: ensure any runtime or wasm error results in a safe fallback visual state
    try {
      if (!this.session) {
        throw new Error('Model not loaded');
      }

      if (this.metadata?.input_dim && features.length !== this.metadata.input_dim) {
        throw new Error(
          `[ModelInference] Input feature length ${features.length} does not match model input_dim ${this.metadata.input_dim}`
        );
      }
      if (!this.metadata?.input_dim && features.length !== INPUT_DIM) {
        console.warn(
          `[ModelInference] Input feature length ${features.length} does not match contract ${INPUT_DIM}`
        );
      }

      const totalStartTime = performance.now();
      let normStartTime = performance.now();

      // Normalize features if normalization stats are available
      let normalizedFeatures = new Float32Array(features);
      if (this.featureMean && this.featureStd) {
        normalizedFeatures = new Float32Array(features.length);
        for (let i = 0; i < features.length; i++) {
          const mean = this.featureMean[i] || 0;
          const std = this.featureStd[i] || 1;
          normalizedFeatures[i] = (features[i] - mean) / (std + 1e-8);
        }
      }

      const normTime = performance.now() - normStartTime;

      // Prepare input tensor
      const inputTensor = new ort.Tensor(
        'float32',
        normalizedFeatures,
        [1, features.length]
      );

      // Run inference
      const inferStartTime = performance.now();
      const feeds = { [MODEL_INPUT_NAME]: inputTensor };
      const results = await this.session.run(feeds);
      const inferTime = performance.now() - inferStartTime;

      const outputTensor = results[MODEL_OUTPUT_NAME] || results[Object.keys(results)[0]];
      if (this.isControlModel && outputTensor.dims?.[1] !== 3 + this.kBands) {
        throw new Error(
          `[ModelInference] Output dim mismatch: expected ${3 + this.kBands}, got ${outputTensor.dims?.[1]}`
        );
      }
      const params = Array.from(outputTensor.data as Float32Array);

      // Post-processing
      const postStartTime = performance.now();

      let visualParams: VisualParameters;

      if (this.isControlModel && this.orbitRuntime) {
        try {
          // Orbit-based control model (runtime-core step_advanced)
          const controlSignals = {
            sTarget: params[0],
            alpha: params[1],
            omegaScale: params[2],
            bandGates: params.slice(3, 3 + this.kBands)
          };

          // Store last control signals for external inspection / flight recorder
          (this as any).lastControlSignals = controlSignals;

          this.orbitRuntime.updateState({
            s: controlSignals.sTarget,
            alpha: controlSignals.alpha,
            omega: this.baseOmega * controlSignals.omegaScale
          });

          const dt = 1.0 / 60.0; // Assume 60 FPS

          // Transient strength proxy = spectral flux averaged across window
          const numFeatures = 6;
          const windowFrames = Math.floor(features.length / numFeatures);
          let avgFlux = 0;
          let avgRMS = 0;
          for (let i = 0; i < windowFrames; i++) {
            avgFlux += features[i * numFeatures + 1];
            avgRMS += features[i * numFeatures + 2];
          }
          avgFlux /= windowFrames;
          avgRMS /= windowFrames;

          const h = Math.max(0, Math.min(1, avgFlux));

          let c;
          try {
            c = this.orbitRuntime.step(dt, controlSignals.bandGates, h);
          } catch (e) {
            // If runtime stepping throws, count the failure and fall back to a safe default c
            console.warn('[ModelInference] Orbit runtime step threw, using fallback c:', e);
            this.inferenceFailures = (this.inferenceFailures || 0) + 1;
            // Prefer runtime last-known c when available
            c = (this.orbitRuntime as any).getLastC ? (this.orbitRuntime as any).getLastC() : { real: 0.0, imag: 0.0 };
          }

        // Extract audio features for color mapping
        let avgOnset = 0;
        for (let i = 0; i < windowFrames; i++) {
          avgOnset += features[i * numFeatures + 4];
        }
        avgOnset /= windowFrames;

        // Map to visual parameters
        const currentHue = (avgRMS * 2.0) % 1.0;
        visualParams = {
          juliaReal: c.real,
          juliaImag: c.imag,
          colorHue: currentHue,
          colorSat: Math.max(0.5, Math.min(1.0, 0.7 + avgOnset * 0.3)),
          colorBright: Math.max(0.5, Math.min(0.9, 0.6 + avgRMS * 0.3)),
          zoom: Math.max(1.5, Math.min(4.0, 2.5)), // Fixed zoom for orbit viewing
          speed: Math.max(0.3, Math.min(0.7, controlSignals.omegaScale / 5.0))
        };

        // Color-based section detection for lobe switching (optional)
        if (!this.deterministicMode) {
          this.detectSectionChange(currentHue);
        }

        // expose inference-level failures for diagnostics/tests
        (this as any).lastInferenceFailures = this.inferenceFailures;
      } catch (e) {
        // Top-level protection: any unexpected error during orbit-based postprocessing
        // should not crash the app. Fall back to a safe visual state.
        console.error('[ModelInference] Unexpected error during orbit postprocessing:', e);
        this.inferenceFailures += 1;
        visualParams = {
          juliaReal: 0.0,
          juliaImag: 0.0,
          colorHue: 0.5,
          colorSat: 0.8,
          colorBright: 0.7,
          zoom: 2.5,
          speed: 0.5,
        };
      }

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
        modelType: this.isControlModel ? 'orbit_control' : 'legacy'
      });
    }

    return visualParams;
    } catch (e) {
      // Global inference-level safety net: never throw to callers (UI). Log and return safe fallback.
      console.error('[ModelInference] Unhandled error during inference, returning safe fallback:', e);
      this.inferenceFailures = (this.inferenceFailures || 0) + 1;
      const fallback: VisualParameters = {
        juliaReal: 0.0,
        juliaImag: 0.0,
        colorHue: 0.5,
        colorSat: 0.8,
        colorBright: 0.7,
        zoom: 2.5,
        speed: 0.5
      };
      return fallback;
    }
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
   * Return last control signals (if orbit model). Useful for flight recorder.
   */
  getLastControlSignals(): any | null {
    return (this as any).lastControlSignals || null;
  }

  getOrbitDebug(): any | null {
    return this.orbitRuntime ? this.orbitRuntime.getDebug() : null;
  }

  getDistanceFieldError(): string | null {
    // Prefer runtime diagnostics if orbit runtime is present
    if (this.orbitRuntime && typeof (this.orbitRuntime as any).getDistanceFieldDiagnostics === 'function') {
      return (this.orbitRuntime as any).getDistanceFieldDiagnostics();
    }
    return this.dfLoadError;
  }

  isDeterministicMode(): boolean {
    return this.deterministicMode;
  }

  resetOrbit(): void {
    if (this.orbitRuntime) this.orbitRuntime.reset(this.orbitSeed);
  }

  /**
   * Check if model is loaded.
   */
  isLoaded(): boolean {
    return this.session !== null;
  }
  
  /**
   * Detect section changes using color moving average with hysteresis.
   * Switches to a random different lobe when a significant color change is detected.
   */
  private detectSectionChange(currentHue: number): void {
    if (!this.orbitRuntime) return;
    if (this.deterministicMode) return; // don't switch in deterministic replay mode

    // Add current hue to history
    this.colorHistory.push(currentHue);
    if (this.colorHistory.length > this.colorHistorySize) {
      this.colorHistory.shift();
    }

    // Need enough history to detect changes
    if (this.colorHistory.length < this.colorHistorySize) return;

    // Check cooldown (hysteresis)
    const framesSinceLastSwitch = this.colorHistory.length - this.lastLobeSwitch;
    if (framesSinceLastSwitch < this.lobeSwitchCooldown) return;

    // Compute moving average of recent colors
    const recentWindow = Math.floor(this.colorHistorySize / 4); // Last 30 frames (~0.5s)
    const oldWindow = Math.floor(this.colorHistorySize / 2); // Middle 60 frames (~1s)

    let recentAvg = 0;
    for (let i = this.colorHistory.length - recentWindow; i < this.colorHistory.length; i++) {
      recentAvg += this.colorHistory[i];
    }
    recentAvg /= recentWindow;

    let oldAvg = 0;
    const oldStart = this.colorHistory.length - oldWindow - recentWindow;
    const oldEnd = this.colorHistory.length - recentWindow;
    for (let i = oldStart; i < oldEnd; i++) {
      if (i >= 0) oldAvg += this.colorHistory[i];
    }
    oldAvg /= oldWindow;

    // Detect significant change (accounting for hue wraparound)
    let hueDiff = Math.abs(recentAvg - oldAvg);
    if (hueDiff > 0.5) hueDiff = 1.0 - hueDiff; // Wraparound correction

    if (hueDiff > this.colorChangeThreshold) {
      // Section change detected! Switch to a random different lobe
      const currentLobe = this.orbitRuntime.getLobe();
      const availableLobes = [1, 2, 3].filter(l => l !== currentLobe);
      const newLobe = availableLobes[Math.floor(Math.random() * availableLobes.length)];

      this.orbitRuntime.setLobe(newLobe);
      this.lastLobeSwitch = this.colorHistory.length;

      console.log(`🎨 Section change detected (Δhue=${hueDiff.toFixed(3)})! Switching: Lobe ${currentLobe} → ${newLobe}`);
    }
  }
}
