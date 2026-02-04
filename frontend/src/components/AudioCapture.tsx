/**
 * Audio capture component using Web Audio API.
 */

import { useEffect, useRef, useState } from 'react';
import { AudioFeatureExtractor } from '../lib/audioFeatures';
import type { SlowState } from '../lib/slowState';

const USE_JS_FEATURES = false;
const HOP_SIZE = 1024;
const WINDOW_FRAMES = 10;

type RuntimeCoreBridge = {
  pushHop: (hop: Float32Array) => SlowState;
};

interface AudioCaptureProps {
  onSlowState: (state: SlowState) => void;
  enabled: boolean;
  audioFile?: File | null;
}

export function AudioCapture({ onSlowState, enabled, audioFile }: AudioCaptureProps) {
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const featureExtractorRef = useRef<AudioFeatureExtractor | null>(null);
  const bridgeRef = useRef<RuntimeCoreBridge | null>(null);
  const latestSlowStateRef = useRef<SlowState | null>(null);
  const lastFrameTimeRef = useRef<number | null>(null);
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

      if (USE_JS_FEATURES) {
        const featureExtractor = new AudioFeatureExtractor(audioContext, analyser);
        featureExtractorRef.current = featureExtractor;
      }

      bridgeRef.current = createMockRuntimeCoreBridge(audioContext.sampleRate);

      // Start playback
      await audio.play();

      setIsCapturing(true);
      setError(null);

      startLoops();
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
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      // Connect microphone to analyser
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      if (USE_JS_FEATURES) {
        const featureExtractor = new AudioFeatureExtractor(audioContext, analyser);
        featureExtractorRef.current = featureExtractor;
      }

      bridgeRef.current = createMockRuntimeCoreBridge(audioContext.sampleRate);

      setIsCapturing(true);
      setError(null);

      startLoops();
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
    featureExtractorRef.current = null;
    bridgeRef.current = null;
    latestSlowStateRef.current = null;
    lastFrameTimeRef.current = null;
    setIsCapturing(false);
  };

  const startLoops = () => {
    const processLoop = (timestamp: number) => {
      const analyser = analyserRef.current;
      const bridge = bridgeRef.current;
      if (!analyser || !bridge) {
        return;
      }

      const timeData = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(timeData);
      const hop = timeData.slice(0, HOP_SIZE);
      let slowState = bridge.pushHop(hop);

      if (USE_JS_FEATURES && featureExtractorRef.current) {
        slowState = {
          ...slowState,
          features: featureExtractorRef.current.extractWindowedFeatures(WINDOW_FRAMES),
        };
      }

      latestSlowStateRef.current = slowState;
      const latestState = latestSlowStateRef.current;

      const lastFrameTime = lastFrameTimeRef.current ?? timestamp;
      const dt = (timestamp - lastFrameTime) / 1000;
      lastFrameTimeRef.current = timestamp;

      if (latestState) {
        const predicted = predictFastPhase(latestState, dt);
        onSlowState(predicted);
      }

      animationFrameRef.current = requestAnimationFrame(processLoop);
    };

    animationFrameRef.current = requestAnimationFrame(processLoop);
  };

  const predictFastPhase = (slowState: SlowState, dt: number): SlowState => {
    const { beat } = slowState;
    const phaseAdvance = dt / beat.spb;
    const phase = (beat.phase + phaseAdvance) % 1;
    return {
      ...slowState,
      beat: {
        ...beat,
        phase,
      },
    };
  };

  const createMockRuntimeCoreBridge = (sampleRate: number): RuntimeCoreBridge => {
    let tSec = 0;
    let phase = 0;
    let beatCount = 0;
    const spb = 0.5;
    const structure = { section_probs: [0, 0, 0, 0], hazard_probs: [0, 0, 0, 0] };
    return {
      pushHop: (hop: Float32Array) => {
        const dt = hop.length / sampleRate;
        tSec += dt;
        const phaseAdvance = dt / spb;
        const nextPhase = phase + phaseAdvance;
        const beatsCrossed = Math.floor(nextPhase);
        phase = ((nextPhase % 1) + 1) % 1;
        if (beatsCrossed > 0) {
          beatCount += beatsCrossed;
        }
        return {
          beat: {
            t_sec: tSec,
            spb,
            phase,
            beat_count: beatCount,
            conf: 0.5,
          },
          structure,
          features: [],
        };
      },
    };
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
