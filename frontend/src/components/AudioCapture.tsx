/**
 * Audio capture component using Web Audio API.
 */

import { useEffect, useRef, useState } from 'react';
import { AudioFeatureExtractor } from '../lib/audioFeatures';
import { USE_WASM_WORKER } from '../lib/config';
import { WasmFeatureWorkerClient } from '../lib/wasm/wasm_feature_worker_client';

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
  const featureExtractorRef = useRef<AudioFeatureExtractor | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);

  // Worker-backed wasm extractor (optional)
  const wasmWorkerClientRef = useRef<WasmFeatureWorkerClient | null>(null);
  const ringBufferRef = useRef<Float32Array | null>(null);
  const intervalRef = useRef<number | null>(null);

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

      // Create feature extractor or initialize wasm worker client depending on configuration
      if (USE_WASM_WORKER && typeof Worker !== 'undefined') {
        // Initialize worker-backed wasm extractor
        const client = new WasmFeatureWorkerClient();
        await client.init({ sr: audioContext.sampleRate, hop: 256, nfft: 1024, include_delta: false, include_delta_delta: false });
        wasmWorkerClientRef.current = client;

        // prepare ring buffer length = (windowFrames - 1) * hop + nfft
        const windowFrames = 10;
        const hop = 256;
        const nfft = 1024;
        const rbLen = (windowFrames - 1) * hop + nfft;
        ringBufferRef.current = new Float32Array(rbLen);

        // Fill initial buffer with zeros
        ringBufferRef.current.fill(0);

        setIsCapturing(true);
        setError(null);

        // Start periodic extraction aligned to hop interval
        const hopMs = (hop / audioContext.sampleRate) * 1000;
        intervalRef.current = window.setInterval(async () => {
          if (!analyserRef.current || !wasmWorkerClientRef.current || !ringBufferRef.current) return;
          const timeData = new Float32Array(analyserRef.current.fftSize);
          analyserRef.current.getFloatTimeDomainData(timeData);

          // shift left by hop and append new timeData at end
          const rb = ringBufferRef.current;
          rb.copyWithin(0, timeData.length);
          rb.set(timeData, rb.length - timeData.length);

          // send a copy to worker (so main thread retains rb)
          const audioCopy = rb.slice();
          try {
            const featuresBuf = await wasmWorkerClientRef.current.extract(audioCopy, windowFrames);
            // featuresBuf is Float64Array
            const features = Array.from(featuresBuf);
            onFeatures(features);
          } catch (err) {
            console.warn('WASM worker extract error', err);
          }
        }, Math.max(5, hopMs));
      } else {
        const featureExtractor = new AudioFeatureExtractor(audioContext, analyser);
        featureExtractorRef.current = featureExtractor;

        // Start feature extraction loop on RAF
        const extractLoop = () => {
          if (!featureExtractorRef.current) return;

          const features = featureExtractorRef.current.extractWindowedFeatures(10);
          onFeatures(features);

          animationFrameRef.current = requestAnimationFrame(extractLoop);
        };

        extractLoop();
      }

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

      // Create feature extractor for microphone path (fallback)
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

    // Stop interval for worker-based extraction
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Terminate worker if present
    if (wasmWorkerClientRef.current) {
      wasmWorkerClientRef.current.terminate();
      wasmWorkerClientRef.current = null;
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
    featureExtractorRef.current = null;
    ringBufferRef.current = null;
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
