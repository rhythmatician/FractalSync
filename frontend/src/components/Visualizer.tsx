/**
 * Main visualizer component that orchestrates audio capture, inference, and rendering.
 * Includes performance monitoring, error recovery, and fallback visualization.
 */

import { useEffect, useRef, useState } from 'react';
import { JuliaRenderer, VisualParameters } from '../lib/juliaRenderer';
import { ModelInference, PerformanceMetrics } from '../lib/modelInference';
import { AudioCapture } from './AudioCapture';

export function Visualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<JuliaRenderer | null>(null);
  const modelRef = useRef<ModelInference | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const [inferenceFailures, setInferenceFailures] = useState(0);
  const metricsUpdateRef = useRef<number | null>(null);

  // Default fallback parameters (safe Julia set from training)
  const DEFAULT_PARAMS: VisualParameters = {
    juliaReal: -0.7269,
    juliaImag: 0.1889,
    colorHue: 0.5,
    colorSat: 0.8,
    colorBright: 0.9,
    zoom: 1.0,
    speed: 0.5
  };

  useEffect(() => {
    // Initialize renderer
    if (canvasRef.current && !rendererRef.current) {
      try {
        const renderer = new JuliaRenderer(canvasRef.current);
        rendererRef.current = renderer;
        renderer.start();

        // Handle window resize
        const handleResize = () => {
          renderer.resize();
        };
        window.addEventListener('resize', handleResize);

        return () => {
          window.removeEventListener('resize', handleResize);
          renderer.dispose();
        };
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to initialize renderer');
      }
    }
  }, []);

  useEffect(() => {
    // Load model with retry logic
    const loadModel = async () => {
      const maxRetries = 3;
      let lastError: Error | null = null;

      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          const model = new ModelInference();
          
          // Try to load from API, fallback to local path
          try {
            const modelResponse = await fetch('/api/model/latest');
            if (modelResponse.ok) {
              const blob = await modelResponse.blob();
              const modelUrl = URL.createObjectURL(blob);
              
              // Try to get metadata
              let metadataUrl: string | undefined;
              try {
                const metadataResponse = await fetch('/api/model/metadata');
                if (metadataResponse.ok) {
                  const metadataBlob = await metadataResponse.blob();
                  metadataUrl = URL.createObjectURL(metadataBlob);
                }
              } catch (e) {
                console.warn('Failed to load metadata:', e);
              }
              
              await model.loadModel(modelUrl, metadataUrl);
            } else {
              // Fallback: try local model path
              await model.loadModel('/models/model.onnx', '/models/model.onnx_metadata.json');
            }
          } catch (e) {
            console.warn('Failed to load model from API, trying local:', e);
            await model.loadModel('/models/model.onnx', '/models/model.onnx_metadata.json');
          }
          
          modelRef.current = model;
          setIsModelLoaded(true);
          setError(null);
          console.log('✓ Model loaded successfully');
          return;
        } catch (err) {
          lastError = err instanceof Error ? err : new Error(String(err));
          console.warn(`Model load attempt ${attempt}/${maxRetries} failed:`, lastError.message);
          
          if (attempt < maxRetries) {
            // Exponential backoff before retry
            await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt - 1) * 1000));
          }
        }
      }

      // All retries exhausted
      const errorMsg = `Failed to load model after ${maxRetries} attempts: ${lastError?.message}`;
      setError(errorMsg);
      setIsModelLoaded(false);
      console.error(errorMsg);
    };

    loadModel();
  }, []);

  const handleFeatures = async (features: number[]) => {
    if (!rendererRef.current) return;

    try {
      if (modelRef.current && modelRef.current.isLoaded()) {
        // Run inference with the model
        const params = await modelRef.current.infer(features);
        
        // Update metrics display
        const modelMetrics = modelRef.current.getMetrics();
        setMetrics(modelMetrics);
        
        // Warn if inference is taking too long for live performance
        if (modelMetrics.lastInferenceTime > 30) {
          console.warn(
            `⚠️ Slow inference: ${modelMetrics.lastInferenceTime.toFixed(1)}ms (avg: ${modelMetrics.averageInferenceTime.toFixed(1)}ms)`
          );
        }

        // Reset failure counter on success
        setInferenceFailures(0);
        
        // Debug: log parameters
        console.log('Model output params:', params);
        
        // Update renderer
        rendererRef.current.updateParameters(params);
      } else {
        // Fallback: use default parameters if model not available
        rendererRef.current.updateParameters(DEFAULT_PARAMS);
        setInferenceFailures(prev => prev + 1);
      }
    } catch (err) {
      console.error('Inference error:', err);
      setInferenceFailures(prev => {
        const newCount = prev + 1;
        if (newCount % 10 === 0) {
          console.warn(`⚠️ Inference failures: ${newCount}. Using fallback visualization.`);
        }
        return newCount;
      });
      
      // Use fallback visualization
      rendererRef.current.updateParameters(DEFAULT_PARAMS);
    }
  };

  const toggleVisualization = () => {
    setIsVisualizing(!isVisualizing);
    
    // Update metrics periodically while visualizing
    if (!isVisualizing) {
      if (metricsUpdateRef.current !== null) {
        clearInterval(metricsUpdateRef.current);
        metricsUpdateRef.current = null;
      }
    } else {
      metricsUpdateRef.current = window.setInterval(() => {
        if (modelRef.current?.isLoaded()) {
          setMetrics(modelRef.current.getMetrics());
        }
      }, 500);
    }
  };

  useEffect(() => {
    return () => {
      if (metricsUpdateRef.current !== null) {
        clearInterval(metricsUpdateRef.current);
      }
    };
  }, []);

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px', background: '#1a1a1a', color: '#fff', borderBottom: '1px solid #444' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ margin: '0 0 10px 0' }}>FractalSync - Julia Set Music Visualizer</h1>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <button
                onClick={toggleVisualization}
                disabled={!isModelLoaded}
                style={{
                  padding: '10px 20px',
                  fontSize: '16px',
                  background: isVisualizing ? '#ff4444' : '#44ff44',
                  color: '#000',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: isModelLoaded ? 'pointer' : 'not-allowed',
                  fontWeight: 'bold'
                }}
              >
                {isVisualizing ? '■ Stop' : '▶ Start'} Visualization
              </button>
              
              {!isModelLoaded && (
                <span style={{ color: '#ffff00' }}>⏳ Loading model...</span>
              )}
              
              {isModelLoaded && (
                <span style={{ color: '#44ff44' }}>✓ Ready</span>
              )}
              
              {error && (
                <span style={{ color: '#ff4444' }}>✗ {error.substring(0, 40)}...</span>
              )}
              
              {inferenceFailures > 0 && (
                <span style={{ color: '#ffaa00' }}>⚠️ {inferenceFailures} failures (fallback mode)</span>
              )}
            </div>
          </div>
          
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            style={{
              padding: '5px 10px',
              background: '#444',
              color: '#fff',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            {showMetrics ? 'Hide' : 'Show'} Metrics
          </button>
        </div>

        {showMetrics && metrics && (
          <div style={{
            marginTop: '10px',
            padding: '10px',
            background: '#000',
            border: '1px solid #444',
            fontFamily: 'monospace',
            fontSize: '12px',
            color: '#0f0'
          }}>
            <div>Last inference: {metrics.lastInferenceTime.toFixed(2)}ms</div>
            <div>Average inference: {metrics.averageInferenceTime.toFixed(2)}ms</div>
            <div>Breakdown: norm={metrics.normalizationTime.toFixed(2)}ms infer={metrics.inferenceTime.toFixed(2)}ms post={metrics.postProcessingTime.toFixed(2)}ms</div>
            {metrics.averageInferenceTime > 16 && (
              <div style={{ color: '#ff4444' }}>⚠️ Average inference exceeds 16ms frame budget (60 FPS)</div>
            )}
          </div>
        )}
      </div>

      <canvas
        ref={canvasRef}
        style={{
          flex: 1,
          width: '100%',
          height: '100%',
          display: 'block',
          background: '#000'
        }}
      />

      {isVisualizing && (
        <AudioCapture onFeatures={handleFeatures} enabled={isVisualizing} />
      )}
    </div>
  );
}
