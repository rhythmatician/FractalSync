/**
 * ONNX.js model inference wrapper.
 */

import * as ort from 'onnxruntime-web';

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

    // Extract parameters
    const params = Array.from(outputTensor.data as Float32Array);

    // Post-processing
    const postStartTime = performance.now();

    // Apply post-processing based on model output
    const visualParams: VisualParameters = {
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
      // Use raw audio features to add dynamic variation for undertrained models
      // Extract some key audio features from the input (assuming 6 features × window_frames)
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

      // Julia parameters: Mix model output with audio features for dynamics
      // Use spectral centroid and flux to influence julia params
      const centroidInfluence = (avgCentroid - 0.5) * 0.4; // Range: -0.2 to +0.2
      const fluxInfluence = (avgFlux - 0.5) * 0.4;
      visualParams.juliaReal = Math.max(-0.8, Math.min(0.8, params[0] * 0.2 + centroidInfluence));
      visualParams.juliaImag = Math.max(-0.8, Math.min(0.8, params[1] * 0.2 + fluxInfluence));
      
      // Color: Map RMS (loudness) to hue cycling, onset to saturation
      visualParams.colorHue = (params[2] + avgRMS * 2.0) % 1.0; // Cycle hue with loudness
      visualParams.colorSat = Math.max(0.5, Math.min(1.0, 0.7 + avgOnset * 0.3)); // Boost sat on onsets
      visualParams.colorBright = Math.max(0.5, Math.min(0.9, 0.6 + avgRMS * 0.3)); // Brightness from loudness
      
      // Zoom: Use rolloff and RMS for dynamic zooming
      const zoomBase = 1.2 + avgRolloff * 0.8; // 1.2 to 2.0
      const zoomVariation = avgRMS * 0.5; // 0 to 0.5
      visualParams.zoom = Math.max(0.8, Math.min(2.5, zoomBase + zoomVariation));
      
      // Speed: Increase with flux and onset energy
      visualParams.speed = Math.max(0.3, Math.min(1.2, 0.5 + avgFlux * 0.3 + avgOnset * 0.4));
    } else {
      // ORIGINAL POST-PROCESSING (pre-MR #8)
      // Scale down undertrained model outputs and add variation
      visualParams.juliaReal = (visualParams.juliaReal * 0.6) % 1.4 - 0.7;
      visualParams.juliaImag = (visualParams.juliaImag * 0.6) % 1.4 - 0.7;
      
      // Color: enforce minimum saturation
      visualParams.colorHue = visualParams.colorHue % 1.0;
      visualParams.colorSat = Math.max(0.5, Math.min(1, visualParams.colorSat * 0.8 + 0.5));
      visualParams.colorBright = Math.max(0.6, Math.min(0.9, visualParams.colorBright * 0.5 + 0.5));
      
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
        audioAvg: { avgRMS: avgRMS.toFixed(3), avgFlux: avgFlux.toFixed(3), avgOnset: avgOnset.toFixed(3) }
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
}
