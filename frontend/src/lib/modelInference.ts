/**
 * ONNX.js model inference wrapper.
 */

import * as ort from 'onnxruntime-web';
import { OrbitSynthesizer, type ControlSignals, type Complex, initOrbitSynth, loadMipPyramid, getControllerVersion, getFeatureVersion } from './orbitSynthesizer';

export interface VisualParameters {
  juliaSeed: Complex;
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
  /** Controller contract stamp (ADR 0001). Missing = pre-contract legacy. */
  controller_version?: string;
  /** Feature-extraction contract stamp. Missing = pre-contract legacy. */
  feature_version?: string;
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
  
  // Orbit-based synthesis (new architecture)
  private orbitSynthesizer: OrbitSynthesizer | null = null;
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
          // Initialize orbit synthesizer for control-signal models.
          // Backed by the canonical Rust implementation via wasm-orbit.
          await initOrbitSynth();
          const kBands = this.metadata.k_bands || 6;
          this.orbitSynthesizer = new OrbitSynthesizer(kBands);

          // Controller contract check (ADR 0001): the model must have been
          // trained against the same controller semantics this runtime runs.
          const runtimeVersion = getControllerVersion();
          const modelVersion = this.metadata.controller_version;
          if (!modelVersion) {
            console.warn(
              `[ModelInference] ⚠️ Model has no controller_version stamp ` +
                `(pre-contract legacy model). Trained against UNKNOWN ` +
                `controller semantics — visuals may not match training.`
            );
          } else if (modelVersion !== runtimeVersion) {
            throw new Error(
              `Controller version mismatch: model was trained against ` +
                `'${modelVersion}' but this runtime is '${runtimeVersion}'. ` +
                `Refusing to load — retrain the model or update the runtime.`
            );
          } else {
            console.log(`[ModelInference] Controller contract OK: ${runtimeVersion}`);
          }

          // Feature-extraction contract (ADR 0001): the model must have been
          // trained on features produced by the same extraction pipeline this
          // runtime executes (same Rust code via wasm-orbit).
          const runtimeFeatureVersion = getFeatureVersion();
          const modelFeatureVersion = this.metadata.feature_version;
          if (!modelFeatureVersion) {
            console.warn(
              `[ModelInference] ⚠️ Model has no feature_version stamp ` +
                `(pre-contract legacy model). Trained on UNKNOWN feature ` +
                `semantics — inputs may not match training.`
            );
          } else if (modelFeatureVersion !== runtimeFeatureVersion) {
            throw new Error(
              `Feature version mismatch: model was trained against ` +
                `'${modelFeatureVersion}' but this runtime extracts with ` +
                `'${runtimeFeatureVersion}'. Refusing to load — retrain the ` +
                `model or update the runtime.`
            );
          } else {
            console.log(`[ModelInference] Feature contract OK: ${runtimeFeatureVersion}`);
          }

          // Load the minimaps so the Player's contour-biased stepper can
          // follow the Shore (best-effort; falls back to plain motion).
          await loadMipPyramid();
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
    const feeds = { audio_features: inputTensor };
    const results = await this.session.run(feeds);
    const inferTime = performance.now() - inferStartTime;

    const outputTensor = results.visual_parameters;
    const params = Array.from(outputTensor.data as Float32Array);

    // Post-processing
    const postStartTime = performance.now();

    let visualParams: VisualParameters;

    if (this.isOrbitModel && this.orbitSynthesizer) {
      // NEW ORBIT-BASED CONTROL MODEL (canonical Rust synthesis via wasm-orbit)
      // Parse control signals from model output
      const controlSignals: ControlSignals = {
        sTarget: params[0],
        alpha: params[1],
        omegaScale: params[2],
        bandGates: params.slice(3)
      };

      // Update orbit state with new control signals
      this.orbitSynthesizer.applyControls(controlSignals);

      console.log(`🎯 Orbit Controls: lobe=${this.orbitSynthesizer.lobe}, s=${controlSignals.sTarget.toFixed(3)}, α=${controlSignals.alpha.toFixed(3)}, ω_scale=${controlSignals.omegaScale.toFixed(3)}`);

      // Extract audio features for color mapping and the transient/hit signal.
      const numFeatures = 6;
      const windowFrames = Math.floor(features.length / numFeatures);
      let avgRMS = 0, avgOnset = 0;
      for (let i = 0; i < windowFrames; i++) {
        avgRMS += features[i * numFeatures + 2];
        avgOnset += features[i * numFeatures + 4];
      }
      avgRMS /= windowFrames;
      avgOnset /= windowFrames;

      // Synthesize Julia parameter c(t) from the Player c-space integrator.
      // `h` is the onset/transient signal: near 1 allows crossing the Shore's
      // contours (section changes), otherwise the Player hugs the contour.
      const dt = 1.0 / 60.0; // Assume 60 FPS
      const h = Math.max(0.0, Math.min(1.0, avgOnset));
      const c = this.orbitSynthesizer.step(dt, controlSignals.bandGates, h);

      console.log(
        `📍 c = (${c.real.toFixed(4)}, ${c.imag.toFixed(4)}) | speed=${this.orbitSynthesizer.speed.toFixed(5)} | h=${h.toFixed(3)}`
      );

      // Map to visual parameters
      const currentHue = (avgRMS * 2.0) % 1.0;
      visualParams = {
        juliaSeed: { ...c },
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
        juliaSeed: { real: params[0], imag: params[1] },
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
      visualParams.juliaSeed.real = (visualParams.juliaSeed.real * 0.6) % 1.4 - 0.7;
      visualParams.juliaSeed.imag = (visualParams.juliaSeed.imag * 0.6) % 1.4 - 0.7;

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
        julia: [visualParams.juliaSeed.real.toFixed(3), visualParams.juliaSeed.imag.toFixed(3)],
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
    if (!this.orbitSynthesizer) return;
    
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
      const currentLobe = this.orbitSynthesizer.lobe;
      const availableLobes = [1, 2, 3].filter(l => l !== currentLobe);
      const newLobe = availableLobes[Math.floor(Math.random() * availableLobes.length)];
      
      this.orbitSynthesizer.setLobe(newLobe);
      this.lastLobeSwitch = this.colorHistory.length;
      
      console.log(`🎨 Section change detected (Δhue=${hueDiff.toFixed(3)})! Switching: Lobe ${currentLobe} → ${newLobe}`);
    }
  }
}
