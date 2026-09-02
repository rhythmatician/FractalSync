/**
 * ONNX.js model inference wrapper.
 */

import * as ort from 'onnxruntime-web';
import { OrbitSynthesizer, type ControlSignals, type Complex, initOrbitSynth, loadMipPyramid, getControllerVersion, getFeatureVersion, getAnalysisPipelineVersion, getControlsVersion, createJuliaViewState, createJuliaViewControls } from './orbitSynthesizer';
import type { AnalysisTick } from './analysisTimebase';

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
  model_type?: string; // 'orbit_control' | 'controls_v2' or legacy
  k_bands?: number;
  /** Controls v2 contract stamp (issue #107). Missing = pre-contract legacy for controls/2. */
  controls_version?: string;
  /** Controller contract stamp (ADR 0001). Missing = pre-contract legacy. */
  controller_version?: string;
  /** Feature-extraction contract stamp. Missing = pre-contract legacy. */
  feature_version?: string;
  analysis_pipeline_version?: string;
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
  
  // Orbit-based synthesis (legacy orbit_control)
  private orbitSynthesizer: OrbitSynthesizer | null = null;
  private isOrbitModel: boolean = false;
  // Controls v2 synthesis (destination manifold seam, issue #107)
  private isControlsV2: boolean = false;
  private juliaViewState: unknown | null = null;
  
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
  // Tick-ordering queue: analysis ticks can arrive faster than async ONNX
  // inference completes. Physics is stateful, so ticks must be applied in
  // order — serialize processing rather than letting tick N+1 reorder
  // before tick N (issue #91, invariant 7).
  private tickQueue: Promise<void> = Promise.resolve();
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

        // Check model contract: controls/2 takes precedence (Rust-authoritative, issue #107)
        this.isControlsV2 = this.metadata.controls_version === 'controls/2' || this.metadata.model_type === 'controls_v2';
        this.isOrbitModel = !this.isControlsV2 && this.metadata.model_type === 'orbit_control';

        if (this.isControlsV2) {
          // Controls v2 path: unified 13-channel action surface -> manifold physics -> c, Julia view deltas -> view state
          await initOrbitSynth();
          const kBands = this.metadata.k_bands || 6;
          this.orbitSynthesizer = new OrbitSynthesizer(kBands);
          // Controls version contract (issue #107): model must have been trained against the same Controls semantics this runtime runs.
          const runtimeControlsVersion = getControlsVersion();
          const modelControlsVersion = this.metadata.controls_version;
          if (!modelControlsVersion) {
            console.warn(
              `[ModelInference] \u26a0\uFE0F Model has no controls_version stamp (pre-contract legacy). Trained against UNKNOWN controls semantics.`
            );
          } else if (modelControlsVersion !== runtimeControlsVersion) {
            throw new Error(
              `Controls version mismatch: model was trained against '${modelControlsVersion}' but this runtime is '${runtimeControlsVersion}'. Refusing to load \u2014 retrain the model or update the runtime.`
            );
          } else {
            console.log(`[ModelInference] Controls contract OK: ${runtimeControlsVersion}`);
          }
          // Controller/feature/pipeline contracts remain relevant for parity gating even for controls/2 (shared timebase + feature path)
          const runtimeVersion = getControllerVersion();
          const modelVersion = this.metadata.controller_version;
          if (modelVersion && modelVersion !== runtimeVersion) {
            throw new Error(`Controller version mismatch: model '${modelVersion}' vs runtime '${runtimeVersion}'. Refusing to load.`);
          }
          const runtimeFeatureVersion = getFeatureVersion();
          const modelFeatureVersion = this.metadata.feature_version;
          if (modelFeatureVersion && modelFeatureVersion !== runtimeFeatureVersion) {
            throw new Error(`Feature version mismatch: model '${modelFeatureVersion}' vs runtime '${runtimeFeatureVersion}'. Refusing to load.`);
          }
          const runtimePipelineVersion = getAnalysisPipelineVersion();
          const modelPipelineVersion = this.metadata.analysis_pipeline_version;
          if (!modelPipelineVersion) {
            throw new Error(`Model has no analysis_pipeline_version stamp (pre-timebase legacy). Refusing to load.`);
          } else if (modelPipelineVersion !== runtimePipelineVersion) {
            throw new Error(`Analysis pipeline version mismatch: model '${modelPipelineVersion}' vs runtime '${runtimePipelineVersion}'. Refusing to load.`);
          }
          // Initialize persistent Julia view state via Rust authority (ADR 0001, issue #95/107)
          // Deterministic shared action-to-view-state semantics belong in Rust; browser consumes via WASM.
          this.juliaViewState = createJuliaViewState();
          console.log('[ModelInference] Loaded Controls v2 model (13-channel, manifold physics)');
        } else if (this.isOrbitModel) {
          // Initialize orbit synthesizer for control-signal models.
          // Backed by the canonical Rust implementation via wasm-orbit.
          await initOrbitSynth();
          const kBands = this.metadata.k_bands || 6;
          this.orbitSynthesizer = new OrbitSynthesizer(kBands);

          // Momentum refinement (ADR 0001, opt-in flag, default OFF): c is
          // persistent state with drag; the boundary target pulls via
          // acceleration. Smooths the frame-to-frame jitter of the raw
          // baseline (c teleporting to each new boundary point) while
          // staying audio-driven — the model still chooses WHERE to go,
          // momentum just shapes HOW c travels there.
          this.orbitSynthesizer.setMomentum(true, 0.90);

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

          // Analysis-pipeline contract (issue #93): the model must have been
          // trained against the same ingestion pipeline this runtime runs
          // (resampling ownership, hop scheduling, epoch semantics) — even
          // when the feature FORMULAS are identical, inputs produced by a
          // different pipeline have different semantics. A model without the
          // stamp predates the pipeline contract and is refused (unlike
          // feature_version, where legacy models get a warning): every
          // pre-timebase model was trained on the librosa-resampled path the
          // runtime no longer executes.
          const runtimePipelineVersion = getAnalysisPipelineVersion();
          const modelPipelineVersion = this.metadata.analysis_pipeline_version;
          if (!modelPipelineVersion) {
            throw new Error(
              `Model has no analysis_pipeline_version stamp (pre-timebase ` +
                `legacy model). It was trained on an ingestion pipeline this ` +
                `runtime does not execute. Refusing to load — retrain the model.`
            );
          } else if (modelPipelineVersion !== runtimePipelineVersion) {
            throw new Error(
              `Analysis pipeline version mismatch: model was trained against ` +
                `'${modelPipelineVersion}' but this runtime runs ` +
                `'${runtimePipelineVersion}'. Refusing to load — retrain the ` +
                `model or update the runtime.`
            );
          } else {
            console.log(`[ModelInference] Analysis pipeline contract OK: ${runtimePipelineVersion}`);
          }

          // Load the minimaps so the Player's contour-biased stepper can
          // follow the Shore (best-effort; falls back to plain motion).
          const pyramidLoaded = await loadMipPyramid();

          // Shore-bias refinement (ADR 0001, opt-in flag, default OFF):
          // route motion through the minimap's contour_biased_step so c
          // hugs the Shore's contours between transients and can cross
          // them on hits (h). Requires the pyramid; silently no-ops
          // without one. d_star 0.5 = target shore proximity, max_step
          // 0.05 caps per-frame travel.
          if (pyramidLoaded) {
            this.orbitSynthesizer.setShoreBias(true, 0.5, 0.05);
            console.log('[ModelInference] Shore bias enabled (pyramid loaded)');
          } else {
            console.warn('[ModelInference] Shore bias unavailable (no pyramid)');
          }

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
   * Enqueue an analysis tick for ordered inference. Ticks are processed
   * strictly in arrival order so stateful Physics steps are never reordered
   * by asynchronous ONNX completion (issue #91, invariant 7). Returns a
   * promise resolving to the visual params for THIS tick.
   */
  inferTick(tick: AnalysisTick): Promise<VisualParameters> {
    const result = this.tickQueue.then(() => this.infer(tick));
    // Keep the queue alive even if one tick rejects, without unhandled
    // rejection warnings; the error still propagates to this tick's caller.
    this.tickQueue = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  /**
   * Run inference on one analysis tick with latency tracking. Advances
   * audio-driven Physics using the tick's authoritative sample-time delta
   * (`dtSeconds`), never a hard-coded render-rate timestep.
   */
  async infer(tick: AnalysisTick): Promise<VisualParameters> {
    const features = tick.features;
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

    if (this.isControlsV2 && this.orbitSynthesizer) {
      // Controls v2 path: 13-channel unified action surface (issue #107)
      // MotionControls -> manifold physics; JuliaViewControls -> persistent view state (Rust authority)
      // params layout: directionX,Y,throttle,brake,grip,impulse, zoomDelta,rotationDelta,hueDelta,chromaDelta,lightnessDelta,accentDelta,harmonyShift
      const motion = {
        direction: [params[0], params[1]] as [number, number],
        throttle: params[2],
        brake: params[3],
        grip: params[4],
        impulse: params[5],
      };
      const view = {
        zoom_delta: params[6],
        rotation_delta: params[7],
        hue_delta: params[8],
        chroma_delta: params[9],
        lightness_delta: params[10],
        accent_delta: params[11],
        harmony_shift: params[12],
      };
      const dt = tick.dtSeconds;
      // Apply motion via manifold physics (musically ignorant, no h/energy)
      const c = this.orbitSynthesizer.stepWithControls(dt, motion);
      // Apply view deltas via Rust authority (ADR 0001): no JS duplicate of clamps/rates/harmony logic.
      // The JS mirror is deleted; the canonical semantics live in runtime-core/src/controls.rs
      // (JuliaViewState::apply_controls) and are consumed via WASM. This satisfies the #107
      // requirement that deterministic shared action-to-view-state semantics belong in Rust.
      const jvs: any = this.juliaViewState;
      if (jvs) {
        const viewControls = createJuliaViewControls(view);
        jvs.apply_controls(viewControls);
        // Read back through WASM getters (zoom/rotation/color are Rust-owned, clamped/wrapped)
        const color = jvs.color as { anchor_hue: number; chroma: number; lightness: number; harmony: string; accent_weight: number };
        const zoom = typeof jvs.zoom === 'number' ? jvs.zoom : (jvs as any).zoom;
        visualParams = {
          juliaSeed: { ...c },
          colorHue: (color?.anchor_hue ?? 0) % 1.0,
          colorSat: color?.chroma ?? 0.6,
          colorBright: color?.lightness ?? 0.6,
          zoom: zoom ?? 2.5,
          speed: Math.hypot(motion.direction[0], motion.direction[1]) * motion.throttle,
        };
      } else {
        visualParams = {
          juliaSeed: { ...c },
          colorHue: 0,
          colorSat: 0.6,
          colorBright: 0.6,
          zoom: 2.5,
          speed: Math.hypot(motion.direction[0], motion.direction[1]) * motion.throttle,
        };
      }
    } else if (this.isOrbitModel && this.orbitSynthesizer) {
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

      // Synthesize Julia parameter c(t) from Player c-space integrator.
      // `h` onset/transient signal: near 1 OPENS THE SHORE WALL — boundary
      // crossing (the "Skyrim clip") becomes easy during transients.
      //
      // Authoritative audio time (issue #91): advance Physics by the tick's
      // canonical sample-time delta, NOT a hard-coded render-rate timestep.
      const dt = tick.dtSeconds;
      const h = Math.max(0.0, Math.min(1.0, avgOnset));
      // Audio energy: sigmoid of normalized RMS in [0,1]. Drives two
      // physics channels: (1) tangential thrust (sustained loudness builds
      // inertia), (2) the energy servo (loud audio pulls c toward the
      // Shore — contract: Energy governs distance from The Shore).
      const energy = 1 / (1 + Math.exp(-avgRMS));
      const thrust = energy * 0.06;
      const synthAny = this.orbitSynthesizer as unknown as {
        setThrust?: (v: number) => void;
        setEnergy?: (v: number) => void;
        thrust?: number;
      };
      if (typeof synthAny.setThrust === 'function') synthAny.setThrust(thrust);
      else if ('thrust' in synthAny) (synthAny as unknown as { thrust: number }).thrust = thrust;
      if (typeof synthAny.setEnergy === 'function') synthAny.setEnergy(energy);
      const c = this.orbitSynthesizer.step(dt, controlSignals.bandGates, h);
      console.log(
        `(${c.real.toFixed(4)}, ${c.imag.toFixed(4)}) speed=${this.orbitSynthesizer.speed.toFixed(5)} h=${h.toFixed(3)} energy=${energy.toFixed(3)} thrust=${thrust.toFixed(4)}`
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
        modelType: this.isControlsV2 ? 'controls_v2' : (this.isOrbitModel ? 'orbit_control' : 'legacy')
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
