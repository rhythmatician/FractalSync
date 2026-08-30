/**
 * AudioWorklet PCM transport (issue #91).
 *
 * Thin browser-plumbing seam between the Web Audio sample clock and the
 * Rust `AnalysisTimebase` (via wasm-orbit). This module owns everything that
 * needs a real `AudioContext` / `AudioWorkletNode`; all timing, resampling,
 * and hop scheduling lives in Rust (ADR 0001) — there is no TypeScript
 * mirror of that math.
 *
 * Both file playback and microphone input converge on the same ingestion
 * abstraction: an analysis-only `AudioWorkletNode` tapped off the source.
 * File playback additionally connects the source to `destination` for
 * audible monitoring; the microphone tap never routes to the speakers, so
 * there is no feedback loop.
 */

import { AnalysisTimebase, type AnalysisTick, type TimebaseDiagnostics, type WasmAnalysisTimebase } from './analysisTimebase';

/** URL of the worklet processor module (served from public/). */
const WORKLET_URL = '/worklets/pcm-tap-processor.js';

export interface PcmTapHandle {
  /** The analysis-only worklet node (connect the source into this). */
  readonly node: AudioWorkletNode;
  /** The deterministic Rust timebase wrapper (for diagnostics/reset). */
  readonly timebase: AnalysisTimebase;
}

/**
 * Create an analysis-only PCM tap on `context`.
 *
 * Loads the worklet module (idempotently per context), constructs the tap
 * node, and wires its message port into a fresh Rust `AnalysisTimebase`.
 * The caller connects a source node into `handle.node` and, for file
 * playback, also into `context.destination`.
 *
 * @param context       The AudioContext whose sample clock is authoritative.
 * @param wasmTimebase  A constructed wasm-orbit AnalysisTimebase instance.
 * @param onTick        Consumer for each emitted AnalysisTick (audio-driven).
 */
export async function createPcmTap(
  context: AudioContext,
  wasmTimebase: WasmAnalysisTimebase,
  onTick: (tick: AnalysisTick) => void
): Promise<PcmTapHandle> {
  await context.audioWorklet.addModule(WORKLET_URL);

  const timebase = new AnalysisTimebase(wasmTimebase);

  const node = new AudioWorkletNode(context, 'pcm-tap', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });

  node.port.onmessage = (ev: MessageEvent) => {
    const { samples, startFrame, sampleRate } = ev.data as {
      samples: Float32Array;
      startFrame: number;
      sampleRate: number;
    };
    const ticks = timebase.ingest({
      samples,
      sourceSampleRate: sampleRate,
      sourceStartFrame: startFrame,
    });
    for (const tick of ticks) onTick(tick);
  };

  return { node, timebase };
}

/** Read a diagnostic snapshot from a tap's timebase. */
export function tapDiagnostics(handle: PcmTapHandle): TimebaseDiagnostics {
  return handle.timebase.diagnostics;
}
