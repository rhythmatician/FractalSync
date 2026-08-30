/**
 * Transport wiring tests (issue #91).
 *
 * These verify the thin AudioWorklet seam: PCM blocks posted by the worklet
 * are forwarded into the Rust AnalysisTimebase and surfaced as AnalysisTicks.
 * The deterministic timing/scheduling math is proven in Rust
 * (runtime-core/tests/test_timebase.rs); here we only prove the wiring that
 * BOTH file playback and microphone input converge on.
 */

import { describe, it, expect, vi } from 'vitest';
import { createPcmTap, tapDiagnostics } from '../audioWorkletTransport';
import type { WasmAnalysisTimebase } from '../analysisTimebase';

/** A mock AudioContext with an audioWorklet.addModule stub. */
function makeMockContext() {
  return {
    audioWorklet: { addModule: vi.fn().mockResolvedValue(undefined) },
  } as unknown as AudioContext;
}

/** A mock wasm AnalysisTimebase that echoes a tick per ingest. */
function makeMockWasm(): WasmAnalysisTimebase & { ingest: ReturnType<typeof vi.fn> } {
  const ingest = vi.fn().mockReturnValue([
    {
      features: [0.5],
      sampleIndex: 1024,
      timeSeconds: 1024 / 48000,
      dtSeconds: 1024 / 48000,
      streamEpoch: 0,
    },
  ]);
  return {
    ingest,
    flush: vi.fn().mockReturnValue([]),
    reset: vi.fn(),
    diagnostics: vi.fn().mockReturnValue({ canonicalSampleIndex: 1024 }),
    free: vi.fn(),
  };
}

/** Capture the AudioWorkletNode constructor so we can drive its port. */
function mockWorkletNodeClass() {
  const instances: Array<{ port: { onmessage: ((ev: MessageEvent) => void) | null } }> = [];
  class MockAudioWorkletNode {
    port: { onmessage: ((ev: MessageEvent) => void) | null } = { onmessage: null };
    constructor(_ctx: unknown, _name: string, _opts: unknown) {
      instances.push(this);
    }
    connect() {}
    disconnect() {}
  }
  (globalThis as any).AudioWorkletNode = MockAudioWorkletNode as any;
  return instances;
}

describe('createPcmTap (shared file/mic ingestion abstraction)', () => {
  it('loads the worklet module and constructs the tap node', async () => {
    const ctx = makeMockContext();
    mockWorkletNodeClass();
    const wasm = makeMockWasm();

    const handle = await createPcmTap(ctx, wasm, () => {});

    expect(ctx.audioWorklet.addModule).toHaveBeenCalledWith('/worklets/pcm-tap-processor.js');
    expect(handle.node).toBeDefined();
    expect(handle.timebase).toBeDefined();
  });

  it('forwards worklet PCM messages into the timebase and emits ticks', async () => {
    const ctx = makeMockContext();
    const instances = mockWorkletNodeClass();
    const wasm = makeMockWasm();
    const onTick = vi.fn();

    await createPcmTap(ctx, wasm, onTick);
    const node = instances[0];

    // Simulate a worklet message (non-overlapping block at 48 kHz).
    const samples = new Float32Array(1024);
    node.port.onmessage?.({
      data: { samples, startFrame: 0, sampleRate: 48000 },
    } as MessageEvent);

    // The timebase wrapper ingested with a bigint frame and surfaced a tick.
    expect(wasm.ingest).toHaveBeenCalledTimes(1);
    const [, rate, frame] = wasm.ingest.mock.calls[0];
    expect(rate).toBe(48000);
    expect(frame).toBe(0n);
    expect(onTick).toHaveBeenCalledTimes(1);
    expect(onTick.mock.calls[0][0].sampleIndex).toBe(1024);
  });

  it('exposes timebase diagnostics through the handle', async () => {
    const ctx = makeMockContext();
    mockWorkletNodeClass();
    const wasm = makeMockWasm();

    const handle = await createPcmTap(ctx, wasm, () => {});
    const d = tapDiagnostics(handle);
    expect(d.canonicalSampleIndex).toBe(1024);
  });
});
