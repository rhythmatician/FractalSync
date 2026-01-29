/**
 * ONNX.js model inference wrapper.
 */

import * as ort from 'onnxruntime-web';
import { OrbitSynthesizer, type ControlSignals, type OrbitState, createInitialState } from './orbitSynthesizer';

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
  
  // Detected session IO names (derived from model); fallback to legacy names
  private inputName: string = 'audio_features';
  private outputName: string = 'visual_parameters';
  
  // Orbit-based synthesis (new architecture)
  private orbitSynthesizer: OrbitSynthesizer | null = null;
  private orbitState: OrbitState | null = null;
  private isOrbitModel: boolean = false;
  
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

      // Detect input/output names from the session and log them for debugging.
      if (this.session) {
        if (Array.isArray(this.session.inputNames) && this.session.inputNames.length > 0) {
          this.inputName = this.session.inputNames[0];
        }
        if (Array.isArray(this.session.outputNames) && this.session.outputNames.length > 0) {
          this.outputName = this.session.outputNames[0];
        }
        console.log(`[ModelInference] session IO: input=${this.inputName} output=${this.outputName}`);
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

        // Check if this is an orbit-based control model
        this.isOrbitModel = this.metadata.model_type === 'orbit_control';

        if (this.isOrbitModel) {
          // Initialize orbit synthesizer for control-signal models
          const kBands = this.metadata.k_bands || 6;
          this.orbitSynthesizer = new OrbitSynthesizer(kBands);
          this.orbitState = createInitialState({ kResiduals: kBands });
          console.log('[ModelInference] Loaded orbit-based control model');
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
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.inputName] = inputTensor;
    const results = await this.session.run(feeds);
    const inferTime = performance.now() - inferStartTime;

    const outputTensor = results[this.outputName];
    if (!outputTensor) {
      throw new Error(`Inference output '${this.outputName}' not found in session results`);
    }
    const params = Array.from(outputTensor.data as Float32Array);

    // Post-processing
    const postStartTime = performance.now();

    let visualParams: VisualParameters;

    if (this.isOrbitModel && this.orbitSynthesizer && this.orbitState) {
      // NEW ORBIT-BASED CONTROL MODEL
      // Parse control signals from model output
      const controlSignals: ControlSignals = {
        sTarget: params[0],
        alpha: params[1],
        omegaScale: params[2],
        bandGates: params.slice(3)
      };

      // Update orbit state with new control signals
      this.orbitState.s = controlSignals.sTarget;
      this.orbitState.alpha = controlSignals.alpha;
      this.orbitState.omega = 1.0 * controlSignals.omegaScale; // Base omega * scale

      console.log(`🎯 Orbit Controls: lobe=${this.orbitState.lobe}, s=${controlSignals.sTarget.toFixed(3)}, α=${controlSignals.alpha.toFixed(3)}, ω_scale=${controlSignals.omegaScale.toFixed(3)}`);

      // Synthesize Julia parameter c(t) from orbit
      const dt = 1.0 / 60.0; // Assume 60 FPS
      const { c, newState } = this.orbitSynthesizer.step(
        this.orbitState,
        dt,
        controlSignals.bandGates
      );
      this.orbitState = newState;

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
      
      // Color-based section detection for lobe switching
      this.detectSectionChange(currentHue);
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
        modelType: this.isOrbitModel ? 'orbit_control' : 'legacy'
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
  
  /**
   * Detect section changes using color moving average with hysteresis.
   * Switches to a random different lobe when a significant color change is detected.
   */
  private detectSectionChange(currentHue: number): void {
    if (!this.orbitState) return;
    
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
      const currentLobe = this.orbitState.lobe;
      const availableLobes = [1, 2, 3].filter(l => l !== currentLobe);
      const newLobe = availableLobes[Math.floor(Math.random() * availableLobes.length)];
      
      this.orbitState.lobe = newLobe;
      this.lastLobeSwitch = this.colorHistory.length;
      
      console.log(`🎨 Section change detected (Δhue=${hueDiff.toFixed(3)})! Switching: Lobe ${currentLobe} → ${newLobe}`);
    }
  }
}
