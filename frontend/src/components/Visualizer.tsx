/**
 * Main visualizer component that orchestrates audio capture, inference, and rendering.
 */

import { useEffect, useRef, useState } from 'react';
import { JuliaRenderer, VisualParameters } from '../lib/juliaRenderer';
import { ModelInference } from '../lib/modelInference';
import { AudioCapture } from './AudioCapture';

export function Visualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<JuliaRenderer | null>(null);
  const modelRef = useRef<ModelInference | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    // Load model
    const loadModel = async () => {
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
            await model.loadModel('/models/model.onnx', '/models/model_metadata.json');
          }
        } catch (e) {
          console.warn('Failed to load model from API, trying local:', e);
          await model.loadModel('/models/model.onnx', '/models/model_metadata.json');
        }
        
        modelRef.current = model;
        setIsModelLoaded(true);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load model');
        setIsModelLoaded(false);
      }
    };

    loadModel();
  }, []);

  const handleFeatures = async (features: number[]) => {
    if (!modelRef.current || !rendererRef.current) return;

    try {
      // Run inference
      const params = await modelRef.current.infer(features);

      // Update renderer
      rendererRef.current.updateParameters(params);
    } catch (err) {
      console.error('Inference error:', err);
    }
  };

  const toggleVisualization = () => {
    setIsVisualizing(!isVisualizing);
  };

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px', background: '#1a1a1a', color: '#fff' }}>
        <h1>FractalSync - Julia Set Music Visualizer</h1>
        <div>
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
              cursor: isModelLoaded ? 'pointer' : 'not-allowed'
            }}
          >
            {isVisualizing ? 'Stop' : 'Start'} Visualization
          </button>
          {!isModelLoaded && (
            <span style={{ marginLeft: '10px' }}>Loading model...</span>
          )}
          {error && (
            <span style={{ marginLeft: '10px', color: '#ff4444' }}>Error: {error}</span>
          )}
        </div>
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
