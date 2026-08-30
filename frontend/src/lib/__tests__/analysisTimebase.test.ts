/**
 * Tests for the AnalysisTimebase TypeScript seam (issue #91).
 *
 * The deterministic transport/timing/scheduling math lives in Rust and is
 * proven by runtime-core/tests/test_timebase.rs. These tests only verify the
 * thin binding wrapper: correct argument marshalling to the wasm instance,
 * snake_case→camelCase tick mapping, and diagnostics mapping. The wasm
 * instance is mocked so no browser or wasm binary is needed.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  AnalysisTimebase,
  CANONICAL_SAMPLE_RATE,
  CANONICAL_HOP_LENGTH,
  type WasmAnalysisTimebase,
} from '../analysisTimebase';

/** A mock wasm AnalysisTimebase that records calls and returns canned ticks. */
function makeMockWasm(overrides: Partial<WasmAnalysisTimebase> = {}): {
  wasm: WasmAnalysisTimebase;
  ingest: ReturnType<typeof vi.fn>;
} {
  const ingest = vi.fn().mockReturnValue([
    {
      features: [1, 2, 3],
      sample_index: 1024,
      time_seconds: 1024 / 48000,
      dt_seconds: 1024 / 48000,
      stream_epoch: 0,
    },
  ]);
  const wasm: WasmAnalysisTimebase = {
    ingest,
    flush: vi.fn().mockReturnValue([]),
    reset: vi.fn(),
    diagnostics: vi.fn().mockReturnValue({
      source_sample_rate: 48000,
      source_frames_ingested: 2048,
      canonical_sample_index: 2048,
      analysis_hop_count: 2,
      time_seconds: 2048 / 48000,
      stream_epoch: 1,
      detected_gaps: 0,
      detected_overlaps: 0,
      last_source_start_frame: 1024,
      last_source_end_frame: 2048,
    }),
    free: vi.fn(),
    ...overrides,
  };
  return { wasm, ingest };
}

describe('constants', () => {
  it('canonical rate and hop come from the runtime authority', () => {
    expect(CANONICAL_SAMPLE_RATE).toBe(48000);
    expect(CANONICAL_HOP_LENGTH).toBe(1024);
  });
});

describe('AnalysisTimebase wrapper', () => {
  it('marshals PCM blocks to the wasm instance with a bigint frame', () => {
    const { wasm, ingest } = makeMockWasm();
    const tb = new AnalysisTimebase(wasm);
    const samples = new Float32Array([0.1, 0.2, 0.3]);
    tb.ingest({ samples, sourceSampleRate: 44100, sourceStartFrame: 512 });

    expect(ingest).toHaveBeenCalledTimes(1);
    const [argSamples, argRate, argFrame] = ingest.mock.calls[0];
    expect(argSamples).toBe(samples);
    expect(argRate).toBe(44100);
    expect(typeof argFrame).toBe('bigint');
    expect(argFrame).toBe(512n);
  });

  it('maps snake_case wasm ticks to camelCase AnalysisTicks', () => {
    const { wasm } = makeMockWasm();
    const tb = new AnalysisTimebase(wasm);
    const ticks = tb.ingest({
      samples: new Float32Array(1024),
      sourceSampleRate: 48000,
      sourceStartFrame: 0,
    });

    expect(ticks).toHaveLength(1);
    const t = ticks[0];
    expect(t.features).toEqual([1, 2, 3]);
    expect(t.sampleIndex).toBe(1024);
    expect(t.timeSeconds).toBeCloseTo(1024 / 48000, 12);
    expect(t.dtSeconds).toBeCloseTo(1024 / 48000, 12);
    expect(t.streamEpoch).toBe(0);
  });

  it('returns an empty tick list when the wasm instance emits none', () => {
    const { wasm, ingest } = makeMockWasm();
    ingest.mockReturnValue([]);
    const tb = new AnalysisTimebase(wasm);
    const ticks = tb.ingest({
      samples: new Float32Array(128),
      sourceSampleRate: 48000,
      sourceStartFrame: 0,
    });
    expect(ticks).toEqual([]);
  });

  it('maps diagnostics to camelCase', () => {
    const { wasm } = makeMockWasm();
    const tb = new AnalysisTimebase(wasm);
    const d = tb.diagnostics;
    expect(d.sourceSampleRate).toBe(48000);
    expect(d.canonicalSampleIndex).toBe(2048);
    expect(d.analysisHopCount).toBe(2);
    expect(d.streamEpoch).toBe(1);
    expect(d.lastSourceEndFrame).toBe(2048);
  });

  it('delegates reset and dispose to the wasm instance', () => {
    const { wasm } = makeMockWasm();
    const tb = new AnalysisTimebase(wasm);
    tb.reset();
    tb.dispose();
    expect(wasm.reset).toHaveBeenCalledTimes(1);
    expect(wasm.free).toHaveBeenCalledTimes(1);
  });
});
