/**
 * Canonical feature extraction tests.
 *
 * The browser no longer extracts features in JavaScript: AudioCapture feeds
 * raw PCM through the wasm-orbit FeatureExtractor — the SAME Rust code the
 * trainer uses. These tests validate the browser-side plumbing:
 *   - resampling to the runtime sample rate (48 kHz)
 *   - rolling PCM buffer ordering
 *   - frame-major window layout contract (FEATURE_VERSION 'features/1')
 *
 * Numeric parity against Rust is enforced by shared/golden_vectors.json
 * feature_cases via preflight check (g) on the Python side; the browser
 * executes the identical Rust binary so no separate numeric test is needed.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  WasmAudioFeatureSource,
  resampleToRuntimeRate,
  RUNTIME_SAMPLE_RATE,
  setWasmFeatureExtractor,
} from "../canonicalFeatures";

/** Deterministic mock of the wasm FeatureExtractor binding. */
function makeMockExtractor() {
  return class MockFeatureExtractor {
    num_features_per_frame = 6;
    extract_window(audio: number[] | Float32Array, windowFrames: number): number[] {
      const out: number[] = [];
      const n = audio.length;
      // Emulates the real binding's "most recent window" semantics: frame
      // i of the returned window corresponds to the i-th-from-last
      // position of the buffered audio.
      for (let i = 0; i < windowFrames; i++) {
        const idx = n > 0 ? Math.max(0, n - windowFrames + i) : 0;
        const sample = audio[idx] ?? 0;
        for (let f = 0; f < 6; f++) {
          out.push(sample * (f + 1));
        }
      }
      return out;
    }
  } as any;
}

describe("resampleToRuntimeRate", () => {
  it("passes through audio already at runtime rate", () => {
    const pcm = new Float32Array([0.1, 0.2, 0.3]);
    expect(resampleToRuntimeRate(pcm, RUNTIME_SAMPLE_RATE)).toBe(pcm);
  });

  it("upsamples 44.1kHz to 48kHz with correct length", () => {
    const pcm = new Float32Array(4410); // 0.1 s at 44.1 kHz
    const out = resampleToRuntimeRate(pcm, 44100);
    expect(out.length).toBe(4800); // 0.1 s at 48 kHz
  });

  it("preserves amplitude through resampling", () => {
    const pcm = new Float32Array(48000);
    pcm.fill(0.5);
    const out = resampleToRuntimeRate(pcm, 22050);
    // Interior samples should be ~0.5 (edges may interpolate).
    expect(out[Math.floor(out.length / 2)]).toBeCloseTo(0.5, 5);
  });
});

describe("WasmAudioFeatureSource", () => {
  beforeEach(() => {
    setWasmFeatureExtractor(makeMockExtractor());
  });

  it("throws if extractor not initialized", () => {
    setWasmFeatureExtractor(null);
    const src = new WasmAudioFeatureSource();
    expect(() => src.extractWindow(10)).toThrow(/not initialized/);
    setWasmFeatureExtractor(makeMockExtractor());
  });

  it("produces windowSize*6 features in frame-major order", () => {
    const src = new WasmAudioFeatureSource();
    const pcm = new Float32Array(2048);
    for (let i = 0; i < pcm.length; i++) pcm[i] = Math.sin(i / 10) * 0.5;
    src.push(pcm, RUNTIME_SAMPLE_RATE);

    const features = src.extractWindow(10);
    expect(features).toHaveLength(60);

    // Frame-major: features[0..5] are frame 0's six values, so
    // features[0] = sample*1 and features[1] = sample*2 share a sample.
    expect(features[1]).toBeCloseTo(features[0] * 2, 8);
    expect(features[5]).toBeCloseTo(features[0] * 6, 8);
  });

  it("windows track the most recent audio as it streams", () => {
    const src = new WasmAudioFeatureSource();

    // First push: constant 0.25 → window reads 0.25.
    const pcmA = new Float32Array(1024);
    pcmA.fill(0.25);
    src.push(pcmA, RUNTIME_SAMPLE_RATE);
    let features = src.extractWindow(10);
    expect(features[0]).toBeCloseTo(0.25, 5);

    // Second push: constant 0.75 → the newest window now reads 0.75
    // everywhere (the window slides forward with the stream).
    const pcmB = new Float32Array(1024);
    pcmB.fill(0.75);
    src.push(pcmB, RUNTIME_SAMPLE_RATE);
    features = src.extractWindow(10);
    expect(features[0]).toBeCloseTo(0.75, 5);
    expect(features[54]).toBeCloseTo(0.75, 5);
  });

  it("wraps the circular buffer without reordering", () => {
    const src = new WasmAudioFeatureSource(4096);
    for (let round = 0; round < 5; round++) {
      const pcm = new Float32Array(1024);
      pcm.fill(round === 4 ? 1.0 : 0.0);
      src.push(pcm, RUNTIME_SAMPLE_RATE);
    }
    const features = src.extractWindow(10);
    // Newest frame's first feature should reflect the final push value.
    expect(features[54]).toBeGreaterThan(0.9);
  });

  it("reports buffered sample count", () => {
    const src = new WasmAudioFeatureSource();
    expect(src.samplesBuffered).toBe(0);
    src.push(new Float32Array(100), RUNTIME_SAMPLE_RATE);
    expect(src.samplesBuffered).toBe(100);
  });
});
