/**
 * Audio capture component (issue #91).
 *
 * Lifecycle/source wiring around the canonical sample-clock timebase. This
 * component owns no timing math: an analysis-only AudioWorkletNode taps PCM
 * off the Web Audio sample clock and feeds the Rust `AnalysisTimebase`,
 * which emits timestamped `AnalysisTick`s on exact 1024-sample canonical
 * boundaries. Both file playback and microphone input converge on the same
 * ingestion abstraction.
 *
 * The render loop is independent — it only ever reads the latest state.
 */

import { useEffect, useRef, useState } from 'react';
import { initOrbitSynth, createAnalysisTimebase } from '../lib/orbitSynthesizer';
import { createPcmTap, type PcmTapHandle } from '../lib/audioWorkletTransport';
import type { AnalysisTick } from '../lib/analysisTimebase';

interface AudioCaptureProps {
  onTick: (tick: AnalysisTick) => void;
  enabled: boolean;
  audioFile?: File | null;
}

export function AudioCapture({ onTick, enabled, audioFile }: AudioCaptureProps) {
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const tapRef = useRef<PcmTapHandle | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const sourceNodeRef = useRef<AudioNode | null>(null);
  // Keep the latest onTick in a ref so the worklet message handler (bound
  // once) always calls the current consumer without re-wiring the graph.
  const onTickRef = useRef(onTick);
  onTickRef.current = onTick;

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, audioFile]);

  /** Shared setup: create context, wasm timebase, and the analysis tap. */
  const setupTap = async (audioContext: AudioContext): Promise<void> => {
    await initOrbitSynth();
    const wasmTimebase = createAnalysisTimebase();
    const handle = await createPcmTap(audioContext, wasmTimebase, (tick) =>
      onTickRef.current(tick)
    );
    tapRef.current = handle;
  };

  const startFilePlayback = async () => {
    if (!audioFile) return;

    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;

      await setupTap(audioContext);
      const tap = tapRef.current!;

      // Audio element for file playback.
      const audio = new Audio();
      audio.loop = true; // Loop for continuous testing
      audio.crossOrigin = 'anonymous';
      audioElementRef.current = audio;

      const url = URL.createObjectURL(audioFile);
      audio.src = url;

      await new Promise<void>((resolve, reject) => {
        audio.addEventListener('canplaythrough', () => resolve(), { once: true });
        audio.addEventListener('error', reject, { once: true });
      });

      const source = audioContext.createMediaElementSource(audio);
      sourceNodeRef.current = source;
      // Analysis path: source → tap (sample clock). Monitoring path: source →
      // speakers. The tap does not need to reach destination to observe PCM.
      source.connect(tap.node);
      source.connect(audioContext.destination);

      await audio.play();

      setIsCapturing(true);
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load audio file';
      setError(errorMessage);
      setIsCapturing(false);
    }
  };

  const startCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;

      await setupTap(audioContext);
      const tap = tapRef.current!;

      const source = audioContext.createMediaStreamSource(stream);
      sourceNodeRef.current = source;
      // Microphone analysis must NOT route to the speakers (no feedback).
      source.connect(tap.node);

      setIsCapturing(true);
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to access microphone';
      setError(errorMessage);
      setIsCapturing(false);
    }
  };

  const stopCapture = () => {
    // Declare the discontinuity so the next start is a new stream epoch.
    if (tapRef.current) {
      try {
        tapRef.current.timebase.reset();
        tapRef.current.timebase.dispose();
      } catch {
        /* already freed */
      }
      tapRef.current.node.disconnect();
      tapRef.current = null;
    }

    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect();
      } catch {
        /* already disconnected */
      }
      sourceNodeRef.current = null;
    }

    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.src = '';
      audioElementRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

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
