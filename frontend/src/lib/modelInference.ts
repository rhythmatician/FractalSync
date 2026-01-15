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
   * Load ONNX model and metadata.
   */
  async loadModel(modelPath: string, metadataPath?: string): Promise<void> {
    // Configure ONNX Runtime to use WASM files from public folder
    ort.env.wasm.wasmPaths = '/';
    ort.env.wasm.numThreads = 1; // Start with single thread for compatibility
    
    // Disable WebGPU/JSEP to avoid .mjs loading issues
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false;
    
    try {
      // Create session with WASM backend only (no WebGPU/JSEP)
      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm']
      });
      console.log('Model loaded successfully with WASM backend');
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }

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

    // Scale down undertrained model outputs and add variation
    // Model with 1 epoch outputs extreme values - normalize to interesting ranges
    visualParams.juliaReal = (visualParams.juliaReal * 0.6) % 1.4 - 0.7;
    visualParams.juliaImag = (visualParams.juliaImag * 0.6) % 1.4 - 0.7;
    
    // Color: enforce minimum saturation
    visualParams.colorHue = visualParams.colorHue % 1.0;
    visualParams.colorSat = Math.max(0.5, Math.min(1, visualParams.colorSat * 0.8 + 0.5));
    visualParams.colorBright = Math.max(0.6, Math.min(0.9, visualParams.colorBright * 0.5 + 0.5));
    
    // Zoom: stay zoomed IN (1.5-4.0 for visible detail)
    visualParams.zoom = Math.max(1.5, Math.min(4.0, visualParams.zoom * 2 + 1.5));
    visualParams.speed = Math.max(0.3, Math.min(0.7, visualParams.speed));

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
