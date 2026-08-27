/**
 * Audio capture component using Web Audio API.
 */

import { useEffect, useRef, useState } from 'react';
import { initOrbitSynth } from '../lib/orbitSynthesizer';
import {
  WasmAudioFeatureSource,
  setWasmFeatureExtractor,
} from '../lib/canonicalFeatures';

interface AudioCaptureProps {
  onFeatures: (features: number[]) => void;
  enabled: boolean;
  audioFile?: File | null;
}

export function AudioCapture({ onFeatures, enabled, audioFile }: AudioCaptureProps) {
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const featureSourceRef = useRef<WasmAudioFeatureSource | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (enabled && !isCapturing) {
      if (audioFile) {
        startFilePlayback();
      } else {
        startCapture();
      }
    } else if (!enabled && isCapturing) {
      stopCapture();
    }

    return () => {
      stopCapture();
    };
  }, [enabled, audioFile]);

  const startFilePlayback = async () => {
    if (!audioFile) return;

    try {
      // Create audio context
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;

      // Create analyser node with proper settings
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      // Create audio element
      const audio = new Audio();
      audio.loop = true; // Loop for continuous testing
      audio.crossOrigin = "anonymous"; // Enable CORS for file access
      audioElementRef.current = audio;
      
      // Load file
      const url = URL.createObjectURL(audioFile);
      audio.src = url;

      // Wait for audio to be ready
      await new Promise<void>((resolve, reject) => {
        audio.addEventListener('canplaythrough', () => resolve(), { once: true });
        audio.addEventListener('error', reject, { once: true });
      });

      // Connect audio element to analyser
      const source = audioContext.createMediaElementSource(audio);
      source.connect(analyser);
      analyser.connect(audioContext.destination); // Connect to speakers

      // Create canonical feature source (wasm extractor + PCM history).
      // Features come from raw PCM fed through the same Rust extractor the
      // trainer uses — not from AnalyserNode's own FFT pipeline.
      //
      // NOTE: import the SAME wasm module instance initOrbitSynth() uses
      // (/wasm/... from public/, served by wasm-pack). The src/wasm copy is
      // a stale January build without FeatureExtractor.
      await initOrbitSynth();
      const wasmUrl = new URL('/wasm/orbit_synth_wasm.js', window.location.origin).href;
      const wasmMod = await import(/* @vite-ignore */ wasmUrl);
      setWasmFeatureExtractor((wasmMod as any).FeatureExtractor ?? null);
      featureSourceRef.current = new WasmAudioFeatureSource();

      // Start playback
      await audio.play();

      setIsCapturing(true);
      setError(null);

      // Start feature extraction loop
      const extractLoop = () => {
        if (!featureSourceRef.current) return;

        // Pull raw PCM for this frame and feed the rolling buffer.
        const pcm = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(pcm);
        featureSourceRef.current.push(pcm, audioContext.sampleRate);

        const features = featureSourceRef.current.extractWindow(10);
        onFeatures(features);

        animationFrameRef.current = requestAnimationFrame(extractLoop);
      };

      extractLoop();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load audio file';
      setError(errorMessage);
      setIsCapturing(false);
    }
  };

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

      // Create canonical feature source (wasm extractor + PCM history).
      // Features come from raw PCM fed through the same Rust extractor the
      // trainer uses — not from AnalyserNode's own FFT pipeline.
      // Same wasm instance as initOrbitSynth() — see note in startFilePlayback.
      await initOrbitSynth();
      const wasmUrl = new URL('/wasm/orbit_synth_wasm.js', window.location.origin).href;
      const wasmMod = await import(/* @vite-ignore */ wasmUrl);
      setWasmFeatureExtractor((wasmMod as any).FeatureExtractor ?? null);
      featureSourceRef.current = new WasmAudioFeatureSource();

      setIsCapturing(true);
      setError(null);

      // Start feature extraction loop
      const extractLoop = () => {
        if (!featureSourceRef.current) return;

        // Pull raw PCM for this frame and feed the rolling buffer.
        const pcm = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(pcm);
        featureSourceRef.current.push(pcm, audioContext.sampleRate);

        const features = featureSourceRef.current.extractWindow(10);
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

    // Stop audio element
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.src = '';
      audioElementRef.current = null;
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
    featureSourceRef.current = null;
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
