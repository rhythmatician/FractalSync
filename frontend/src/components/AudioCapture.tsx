/**
 * Audio capture component using Web Audio API.
 */

import { useEffect, useRef, useState } from 'react';
import { AudioFeatureExtractor } from '../lib/audioFeatures';

interface AudioCaptureProps {
  onFeatures: (features: number[]) => void;
  enabled: boolean;
}

export function AudioCapture({ onFeatures, enabled }: AudioCaptureProps) {
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const featureExtractorRef = useRef<AudioFeatureExtractor | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    if (enabled && !isCapturing) {
      startCapture();
    } else if (!enabled && isCapturing) {
      stopCapture();
    }

    return () => {
      stopCapture();
    };
  }, [enabled]);

  const startCapture = async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Create audio context
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;

      // Create analyser node
      const analyser = audioContext.createAnalyser();
      analyserRef.current = analyser;

      // Connect microphone to analyser
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      // Create feature extractor
      const featureExtractor = new AudioFeatureExtractor(audioContext, analyser);
      featureExtractorRef.current = featureExtractor;

      setIsCapturing(true);
      setError(null);

      // Start feature extraction loop
      const extractLoop = () => {
        if (!featureExtractorRef.current) return;

        const features = featureExtractorRef.current.extractWindowedFeatures(10);
        onFeatures(features);

        animationFrameRef.current = requestAnimationFrame(extractLoop);
      };

      extractLoop();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to access microphone';
      setError(errorMessage);
      setIsCapturing(false);
    }
  };

  const stopCapture = () => {
    // Stop animation frame
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Stop audio stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    analyserRef.current = null;
    featureExtractorRef.current = null;
    setIsCapturing(false);
  };

  if (error) {
    return (
      <div style={{ color: 'red', padding: '10px' }}>
        Error: {error}
      </div>
    );
  }

  return null; // This component doesn't render UI
}
