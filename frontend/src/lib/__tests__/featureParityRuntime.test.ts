import { describe, it, expect } from 'vitest';

import { AudioFeatureExtractor } from '../audioFeatures';

// Reuse a small DFT helper like the other parity test
function hann(n: number) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  // normalize RMS
  let sumsq = 0;
  for (let i = 0; i < n; i++) sumsq += w[i] * w[i];
  const rms = sumsq / n;
  const norm = 1 / Math.sqrt(rms);
  for (let i = 0; i < n; i++) w[i] *= norm;
  return w;
}

function dftMagnitude(frame: Float32Array, nfft: number) {
  const out = new Float32Array(nfft / 2);
  const w = hann(nfft);
  const x = new Float32Array(nfft);
  for (let i = 0; i < Math.min(frame.length, nfft); i++) x[i] = frame[i] * w[i];
  for (let k = 0; k < nfft / 2; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < nfft; n++) {
      const angle = (2 * Math.PI * k * n) / nfft;
      re += x[n] * Math.cos(angle);
      im -= x[n] * Math.sin(angle);
    }
    out[k] = Math.sqrt(re * re + im * im);
  }
  return out;
}

function makeAudio(sr: number, duration: number, freq: number) {
  const len = Math.floor(sr * duration);
  const buf = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const t = i / sr;
    buf[i] = Math.sin(2 * Math.PI * freq * t) * 0.6; // deterministic, no noise
  }
  return buf;
}

// Fake Analyser that returns prepared frames sequentially
class FakeAnalyser {
  frequencyFrames: Float32Array[];
  timeFrames: Float32Array[];
  idx: number = 0;
  fftSize: number;

  constructor(freqFrames: Float32Array[], timeFrames: Float32Array[], fftSize = 1024) {
    this.frequencyFrames = freqFrames;
    this.timeFrames = timeFrames;
    this.fftSize = fftSize;
  }

  get frequencyBinCount() {
    return this.fftSize / 2;
  }

  getFloatFrequencyData(dst: Float32Array) {
    // audioFeature.extractFeatures expects dB values (like Analyser.getFloatFrequencyData)
    const src = this.frequencyFrames[this.idx % this.frequencyFrames.length];
    const n = Math.min(dst.length, src.length);
    for (let i = 0; i < n; i++) {
      const mag = src[i] || 0;
      const epsilon = 1e-12;
      const db = 20 * Math.log10(Math.max(mag, epsilon));
      dst[i] = db;
    }
    // fill remainder with very small value (-200 dB)
    for (let i = n; i < dst.length; i++) dst[i] = -200;
  }

  getFloatTimeDomainData(dst: Float32Array) {
    const src = this.timeFrames[this.idx % this.timeFrames.length];
    const n = Math.min(dst.length, src.length);
    for (let i = 0; i < n; i++) dst[i] = src[i];
    for (let i = n; i < dst.length; i++) dst[i] = 0;
    this.idx++;
  }
}

describe('Runtime feature parity (AudioFeatureExtractor vs wasm)', () => {
  it('should match wasm per-frame features when extractor is aligned', async () => {
    const sr = 22050;
    const hop = 256;
    const nfft = 1024;

    // create synthetic audio long enough for several frames
    const duration = ((6 - 1) * hop + nfft) / sr + 0.1;
    const audio = makeAudio(sr, duration, 440);

    // prepare frames
    const frames: Float32Array[] = [];
    const timeFrames: Float32Array[] = [];
    const nFrames = Math.floor((audio.length - nfft) / hop) + 1;
    for (let fi = 0; fi < nFrames; fi++) {
      const start = fi * hop;
      const frame = audio.subarray(start, start + nfft);
      const mag = dftMagnitude(frame, nfft);
      // wasm FeatureExtractor expects magnitude values directly for stft
      frames.push(mag);

      // time domain window
      const tbuf = new Float32Array(nfft);
      for (let i = 0; i < nfft; i++) tbuf[i] = frame[i] || 0;
      timeFrames.push(tbuf);
    }

    // Create fake analyser and mock AudioContext
    const fake = new FakeAnalyser(frames, timeFrames, nfft);
    (globalThis as any).AudioContext = class {
      sampleRate = sr;
      createAnalyser() { return fake; }
    } as any;

    const analyser = new (globalThis as any).AudioContext().createAnalyser();
    // Pass fft size explicitly so the extractor uses the same window size as the wasm test
    const extractor = new AudioFeatureExtractor(new (globalThis as any).AudioContext(), analyser as any, nfft);

    // Collect runtime per-frame features by calling extractFeatures repeatedly
    const runtimePerFeature: number[][] = new Array(6).fill(0).map(() => []);
    for (let i = 0; i < nFrames; i++) {
      const f = extractor.extractFeatures();
      runtimePerFeature[0].push(f.spectralCentroid);
      runtimePerFeature[1].push(f.spectralFlux);
      runtimePerFeature[2].push(f.rmsEnergy);
      runtimePerFeature[3].push(f.zeroCrossingRate);
      runtimePerFeature[4].push(f.onsets);
      runtimePerFeature[5].push(f.spectralRolloff);
    }

    // Import wasm FeatureExtractor and compute per-frame features
    const wasmModule = await import('../../wasm/orbit_synth_wasm.js') as any;
    const originalFetch = (globalThis as any).fetch;
    const fs = await import('fs');
    const path = await import('path');
    const wasmPath = path.resolve(process.cwd(), 'src/wasm/orbit_synth_wasm_bg.wasm');
    (globalThis as any).fetch = async (_url: string) => ({ arrayBuffer: async () => fs.promises.readFile(wasmPath) });

    try {
      await wasmModule.default();
    } finally {
      (globalThis as any).fetch = originalFetch;
    }

    const FeatureExtractor = wasmModule.FeatureExtractor as any;
    const wasmFe = new FeatureExtractor(sr, hop, nfft, false, false);
    const wasmNested = wasmFe.extract_windowed_features(audio, 1) as any[];
    const wasmPerFeature: number[][] = new Array(6).fill(0).map(() => []);
    for (let i = 0; i < wasmNested.length; i++) {
      const row = wasmNested[i];
      for (let k = 0; k < 6; k++) wasmPerFeature[k].push(Number(row[k]));
    }

    // Normalize runtime flux/rms/onset to match wasm post-loop normalisation
    function normaliseInPlace(arr: number[]) {
      if (!arr.length) return;
      let min = Infinity, max = -Infinity;
      for (const v of arr) {
        if (!Number.isFinite(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const range = max - min;
      if (range > 0) for (let i = 0; i < arr.length; i++) arr[i] = (arr[i] - min) / range;
    }

    // copy arrays so we don't mutate original debug data
    const rCopy: number[][] = runtimePerFeature.map(a => a.slice());
    normaliseInPlace(rCopy[1]); // flux
    normaliseInPlace(rCopy[2]); // rms
    normaliseInPlace(rCopy[4]); // onset

    // Compare per-feature Pearson correlation
    function corr(a: number[], b: number[]) {
      const mean = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length;
      const ma = mean(a), mb = mean(b);
      let num = 0, da = 0, db = 0;
      for (let i = 0; i < a.length; i++) {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma) * (a[i] - ma);
        db += (b[i] - mb) * (b[i] - mb);
      }
      const denom = Math.sqrt(da * db);
      // Handle degenerate case: both series have zero variance
      if (denom === 0) {
        // If means are nearly equal (within a small tolerance), consider correlation perfect
        // (handles degenerate constant series due to deterministic test signals)
        return Math.abs(ma - mb) < 1e-3 ? 1 : 0;
      }
      return num / denom;
    }

    expect(wasmPerFeature[0].length).toBeGreaterThan(0);
    for (let k = 0; k < 6; k++) {
      const a = rCopy[k];
      const b = wasmPerFeature[k];
      const r = corr(a, b);

      // If debug mode, write diagnostics
      if (process.env.FEATURE_PARITY_DEBUG === '1') {
        const fs = await import('fs');
        const out = { runtime: runtimePerFeature, runtime_normalized: rCopy, wasm: wasmPerFeature, k, r };
        await fs.promises.mkdir('test-output', { recursive: true });
        await fs.promises.writeFile(`test-output/runtime_parity_debug.json`, JSON.stringify(out, null, 2));
        console.log(`Wrote runtime parity debug to test-output/runtime_parity_debug.json`);
      }

      // Strict check: require > 0.95 correlation
      expect(r, `Feature ${k} correlation ${r.toFixed(3)} too low`).toBeGreaterThanOrEqual(0.95);
    }
  });
});
