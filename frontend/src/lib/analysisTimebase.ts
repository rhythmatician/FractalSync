/**
 * Authoritative sample-clock audio timebase (issue #91) — TypeScript seam.
 *
 * The deterministic transport/timing/scheduling/resampling math lives in
 * `runtime-core/src/timebase.rs` (Rust) and is consumed here through the
 * wasm-orbit `AnalysisTimebase` binding. Per ADR 0001 there is deliberately
 * NO TypeScript mirror of the resampler, hop scheduler, or sample
 * accounting — this module only defines the boundary types and a thin
 * wrapper that delegates to the wasm instance.
 *
 * Architecture (issue #91):
 *
 *   file or microphone source
 *           ↓
 *   Web Audio sample clock (AudioWorklet render quanta)
 *           ↓
 *   exactly-once non-overlapping PCM transport (monotonic source position)
 *           ↓
 *   continuous canonical 48 kHz stream (Rust stateful streaming resampler)
 *           ↓
 *   exact 1024-sample hop scheduler (Rust)
 *           ↓
 *   runtime-core FeatureExtractor via WASM
 *           ↓
 *   AnalysisTick { features, sampleIndex, timeSeconds, dtSeconds, streamEpoch }
 *           ↓
 *   audio-driven consumers (model inference / future CycleBank)
 */

import { RUNTIME_SAMPLE_RATE, HOP_LENGTH } from './canonicalFeatures';

/** Canonical analysis timeline sample rate (runtime-core authority). */
export const CANONICAL_SAMPLE_RATE = RUNTIME_SAMPLE_RATE;

/** Canonical hop length in canonical samples (runtime-core authority). */
export const CANONICAL_HOP_LENGTH = HOP_LENGTH;

/** A block of PCM observed from the Web Audio sample clock. */
export interface PcmBlock {
  /** Non-overlapping PCM samples at the source sample rate. */
  samples: Float32Array;
  /** Source sample rate (e.g. 44100, 48000). */
  sourceSampleRate: number;
  /**
   * Position of `samples[0]` on the source sample clock, in source frames
   * since the stream started. Must be monotonically non-decreasing.
   */
  sourceStartFrame: number;
}

/** A timestamped analysis event — the seam the future CycleBank consumes. */
export interface AnalysisTick {
  /** Flattened frame-major feature window from the Rust extractor. */
  features: number[];
  /** Canonical 48 kHz sample index of this tick's hop boundary. */
  sampleIndex: number;
  /** Derived convenience: sampleIndex / CANONICAL_SAMPLE_RATE. */
  timeSeconds: number;
  /** Derived convenience: canonical hop duration in seconds. */
  dtSeconds: number;
  /** Increments on every reset/discontinuity so consumers detect restarts. */
  streamEpoch: number;
}

/** Diagnostic snapshot for manual verification of the clock. */
export interface TimebaseDiagnostics {
  sourceSampleRate: number;
  sourceFramesIngested: number;
  canonicalSampleIndex: number;
  analysisHopCount: number;
  timeSeconds: number;
  streamEpoch: number;
  detectedGaps: number;
  detectedOverlaps: number;
  lastSourceStartFrame: number;
  lastSourceEndFrame: number;
}

/** Shape of the wasm-orbit AnalysisTimebase binding (subset). */
export interface WasmAnalysisTimebase {
  ingest(samples: Float32Array, sourceSampleRate: number, sourceStartFrame: bigint): unknown;
  flush(): unknown;
  reset(): void;
  diagnostics(): unknown;
  free(): void;
}

/** Raw tick shape as serialized by the wasm binding (snake_case). */
interface RawTick {
  features: number[];
  sample_index: number;
  time_seconds: number;
  dt_seconds: number;
  stream_epoch: number;
}

function toTick(raw: RawTick): AnalysisTick {
  return {
    features: raw.features,
    sampleIndex: raw.sample_index,
    timeSeconds: raw.time_seconds,
    dtSeconds: raw.dt_seconds,
    streamEpoch: raw.stream_epoch,
  };
}

/**
 * Thin wrapper over the wasm-orbit `AnalysisTimebase`. Owns no timing math;
 * every operation delegates to the Rust instance so the browser and any
 * future Rust consumer share one implementation.
 */
export class AnalysisTimebase {
  private inner: WasmAnalysisTimebase;

  /** Wrap a live wasm binding instance (constructed by the caller). */
  constructor(wasmInstance: WasmAnalysisTimebase) {
    this.inner = wasmInstance;
  }

  /**
   * Ingest one non-overlapping PCM block. Throws on non-monotonic source
   * position (a transport bug, not a stream discontinuity — those are
   * declared via `reset()`). Returns the ticks whose hop boundaries
   * completed within this block (zero or more).
   */
  ingest(block: PcmBlock): AnalysisTick[] {
    const raw = this.inner.ingest(
      block.samples,
      block.sourceSampleRate,
      BigInt(block.sourceStartFrame)
    ) as RawTick[];
    return (raw ?? []).map(toTick);
  }

  /** Flush end-of-stream (recovers the deferred final sample/tick). */
  flush(): AnalysisTick[] {
    const raw = this.inner.flush() as RawTick[];
    return (raw ?? []).map(toTick);
  }

  /** Declare a stream discontinuity (start/stop/source replacement). */
  reset(): void {
    this.inner.reset();
  }

  /** Diagnostic snapshot for verifying the clock manually. */
  get diagnostics(): TimebaseDiagnostics {
    const d = this.inner.diagnostics() as Record<string, number>;
    return {
      sourceSampleRate: d.source_sample_rate ?? 0,
      sourceFramesIngested: d.source_frames_ingested ?? 0,
      canonicalSampleIndex: d.canonical_sample_index ?? 0,
      analysisHopCount: d.analysis_hop_count ?? 0,
      timeSeconds: d.time_seconds ?? 0,
      streamEpoch: d.stream_epoch ?? 0,
      detectedGaps: d.detected_gaps ?? 0,
      detectedOverlaps: d.detected_overlaps ?? 0,
      lastSourceStartFrame: d.last_source_start_frame ?? 0,
      lastSourceEndFrame: d.last_source_end_frame ?? 0,
    };
  }

  /** Release the underlying wasm instance. */
  dispose(): void {
    this.inner.free();
  }
}
