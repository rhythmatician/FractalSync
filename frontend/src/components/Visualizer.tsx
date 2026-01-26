/**
 * Main visualizer component that orchestrates audio capture, inference, and rendering.
 * Includes performance monitoring and error reporting.
 */

import { useEffect, useRef, useState } from 'react';
import { JuliaRenderer } from '../lib/juliaRenderer';
import { ModelInference, PerformanceMetrics, ModelMetadata } from '../lib/modelInference';
import { AudioCapture } from './AudioCapture';
import { FullscreenToggle } from './FullscreenToggle';

export function Visualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<JuliaRenderer | null>(null);
  const modelRef = useRef<ModelInference | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [modelMetadata, setModelMetadata] = useState<ModelMetadata | null>(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(true);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioReactiveEnabled, setAudioReactiveEnabled] = useState(false);
  const metricsUpdateRef = useRef<number | null>(null);

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
    const loadModel = async () => {
      try {
        const model = new ModelInference();
        const modelResponse = await fetch('/api/model/latest');
        if (!modelResponse.ok) {
          throw new Error(`Failed to fetch model: ${modelResponse.status}`);
        }
        const blob = await modelResponse.blob();
        const modelUrl = URL.createObjectURL(blob);

        const metadataResponse = await fetch('/api/model/metadata');
        if (!metadataResponse.ok) {
          throw new Error(`Failed to fetch metadata: ${metadataResponse.status}`);
        }
        const metadataBlob = await metadataResponse.blob();
        const metadataUrl = URL.createObjectURL(metadataBlob);

        await model.loadModel(modelUrl, metadataUrl);
        modelRef.current = model;
        setIsModelLoaded(true);
        setError(null);
        setModelMetadata(model.getMetadata());
        console.log('✓ Model loaded successfully');
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        setError(errorMsg);
        setIsModelLoaded(false);
        console.error(errorMsg);
      }
    };

    loadModel();
  }, []);

  const handleFeatures = async (features: number[]) => {
    if (!rendererRef.current) return;

    try {
      if (!modelRef.current || !modelRef.current.isLoaded()) {
        throw new Error('Model not loaded');
      }

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

      // Debug: log final visual parameters (after orbit synthesis)
      console.debug('Final visual params (post-orbit-synthesis):', params);
      
      // Update renderer
      rendererRef.current.updateParameters(params);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('Inference error:', errorMsg);
      setError(errorMsg);
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
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
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

              <label style={{
                padding: '8px 16px',
                background: '#444',
                borderRadius: '5px',
                cursor: 'pointer',
                fontSize: '14px'
              }}>
                {audioFile ? `📁 ${audioFile.name}` : '🎵 Choose Audio File'}
                <input
                  type="file"
                  accept="audio/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      setAudioFile(file);
                      if (isVisualizing) {
                        setIsVisualizing(false);
                        setTimeout(() => setIsVisualizing(true), 100);
                      }
                    }
                  }}
                  style={{ display: 'none' }}
                />
              </label>

              {audioFile && (
                <button
                  onClick={() => setAudioFile(null)}
                  style={{
                    padding: '8px 12px',
                    background: '#ff4444',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  Use Microphone
                </button>
              )}
              
              {!isModelLoaded && (
                <span style={{ color: '#ffff00' }}>⏳ Loading model...</span>
              )}
              
              {isModelLoaded && (
                <span style={{ color: '#44ff44' }}>✓ Ready</span>
              )}
              
              {error && (
                <span style={{ color: '#ff4444' }}>✗ {error.substring(0, 40)}...</span>
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

          <button
            onClick={() => setShowModelInfo(!showModelInfo)}
            style={{
              padding: '5px 10px',
              background: '#444',
              color: '#fff',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '12px',
              marginLeft: '5px'
            }}
            title="Show model information"
          >
            {showModelInfo ? 'Hide' : 'Show'} Model
          </button>
          
          <button
            onClick={() => {
              const newState = !audioReactiveEnabled;
              setAudioReactiveEnabled(newState);
              if (modelRef.current) {
                modelRef.current.setAudioReactivePostProcessing(newState);
              }
            }}
            style={{
              padding: '5px 10px',
              background: audioReactiveEnabled ? '#4CAF50' : '#666',
              color: '#fff',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '12px',
              marginLeft: '10px'
            }}
            title="Toggle audio-reactive post-processing (MR #8 / commit 75c1a43)"
          >
            Audio-Reactive: {audioReactiveEnabled ? 'ON' : 'OFF'}
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

        {showModelInfo && modelMetadata && (
          <div style={{
            marginTop: '10px',
            padding: '10px',
            background: '#1a3a1a',
            border: '1px solid #44ff44',
            fontFamily: 'monospace',
            fontSize: '12px',
            color: '#44ff44',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div>
              <strong>📦 Model Loaded:</strong>
              {modelMetadata.epoch && <span> Epoch {modelMetadata.epoch}</span>}
              {modelMetadata.window_frames && <span> | Window: {modelMetadata.window_frames}</span>}
              {modelMetadata.input_dim && <span> | Input: {modelMetadata.input_dim}</span>}
              {modelMetadata.output_dim && <span> | Output: {modelMetadata.output_dim}</span>}
              {modelMetadata.timestamp && (
                <span> | {new Date(modelMetadata.timestamp).toLocaleString()}</span>
              )}
              {modelMetadata.git_hash && (
                <span style={{ marginLeft: '10px', color: '#88ff88' }}>
                  #{modelMetadata.git_hash.substring(0, 8)}
                </span>
              )}
            </div>
            <button
              onClick={() => setShowModelInfo(false)}
              style={{
                background: 'transparent',
                color: '#44ff44',
                border: '1px solid #44ff44',
                borderRadius: '3px',
                padding: '2px 8px',
                cursor: 'pointer',
                fontSize: '11px'
              }}
            >
              ✕
            </button>
          </div>
        )}
      </div>

      <div style={{ flex: 1, width: '100%', height: '100%', display: 'flex', position: 'relative' }}>
        <canvas
          ref={canvasRef}
          style={{
            flex: 1,
            width: '100%',
            height: '100%',
            display: 'block',
            background: '#000'
          }}
          id="visualizerCanvas"
        />

        <FullscreenToggle targetId="visualizerCanvas" position="top-right" />
      </div>

      {isVisualizing && (
        <AudioCapture onFeatures={handleFeatures} enabled={isVisualizing} audioFile={audioFile} />
      )}
    </div>
  );
}
