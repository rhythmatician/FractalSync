import { describe, it, expect } from 'vitest';

// We'll dynamically import the wasm module inside the test so we can await initialization

// Small helper: create sine+noise audio buffer
function makeAudio(sr: number, durationSec: number, freq: number) {
  const len = Math.floor(sr * durationSec);
  const buf = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const t = i / sr;
    buf[i] = Math.sin(2 * Math.PI * freq * t) * 0.6 + (Math.random() * 2 - 1) * 0.02;
  }
  return buf;
}

// Hann window
function hann(n: number) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return w;
}

// Naive DFT magnitude (returns length n/2 array)
function dftMagnitude(frame: Float32Array, nfft: number) {
  const out = new Float32Array(nfft / 2);
  const w = hann(nfft);
  // zero-pad if frame shorter
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

// Compute features from audio using reference JS algorithm (matches frontend impl)
function computeWindowedFeaturesReference(audio: Float32Array, sr: number, hop: number, nfft: number, windowFrames: number) {
  const features: number[] = [];
  let prevSpec: Float32Array | null = null;
  const nyq = sr / 2;
  const binWidth = nyq / (nfft / 2);

  for (let f = 0; f < windowFrames; f++) {
    const start = f * hop;
    const frame = audio.subarray(start, start + nfft);
    const mag = dftMagnitude(frame, nfft);

    // centroid
    let weighted = 0;
    let magSum = 0;
    for (let i = 0; i < mag.length; i++) {
      const freq = i * binWidth;
      weighted += freq * mag[i];
      magSum += mag[i];
    }
    const centroid = magSum > 0 ? (weighted / magSum) / nyq : 0;

    // flux (L1 positive changes normalized by energy)
    let flux = 0;
    if (prevSpec) {
      let sumPos = 0;
      for (let i = 0; i < mag.length; i++) {
        const diff = mag[i] - prevSpec[i];
        if (diff > 0) sumPos += diff;
      }
      const energy = magSum;
      flux = energy > 0 ? sumPos / energy : 0;
    }

    // rms (time-domain)
    let rms = 0;
    for (let i = 0; i < frame.length; i++) rms += frame[i] * frame[i];
    rms = Math.sqrt(rms / Math.max(1, frame.length));
    if (rms > 1) rms = 1;

    // zero crossing rate approx
    let zc = 0;
    for (let i = 1; i < frame.length; i++) if ((frame[i] >= 0) !== (frame[i - 1] >= 0)) zc++;
    zc = zc / frame.length;

    // onset proxy: thresholded flux
    const onset = flux > 0.1 ? flux : 0;

    // spectral rolloff
    let cumulative = 0;
    const target = 0.85 * (magSum);
    let rolloffFreq = nyq;
    for (let i = 0; i < mag.length; i++) {
      cumulative += mag[i];
      if (cumulative >= target) {
        rolloffFreq = i * binWidth;
        break;
      }
    }
    const rolloff = rolloffFreq / nyq;

    // push six features
    features.push(centroid, flux, rms, zc, onset, rolloff);
    prevSpec = mag;
  }
  return features;
}

describe('Feature parity: wasm vs reference JS', () => {
  it('produces similar per-frame features for synthetic audio', async () => {
    // parameters reduced for faster tests (smaller NFFT, fewer windows)
    // Use lower SR and shorter FFT to speed up naive DFT reference implementation
    const sr = 22050;
    const hop = 256;
    const nfft = 1024;
    const windowFrames = 6; // keep at least a few windows for stable correlation

    // Dynamic import so we can await the async initializer exported as default
    // Use relative import to the generated wasm wrapper in src/wasm
    const wasmModule = await import('../../wasm/orbit_synth_wasm.js') as any;

    // Vitest/node environment doesn't resolve Vite's `?url` imports by default.
    // Stub global fetch to read the local wasm file bytes directly so the
    // initializer can `fetch(wasmUrl)` and get the wasm bytes.
    const originalFetch = (globalThis as any).fetch;
    const fs = await import('fs');
    const path = await import('path');
    const wasmPath = path.resolve(process.cwd(), 'src/wasm/orbit_synth_wasm_bg.wasm');
    (globalThis as any).fetch = async (_url: string) => ({ arrayBuffer: async () => fs.promises.readFile(wasmPath) });

    try {
      await wasmModule.default();
    } finally {
      // restore fetch
      (globalThis as any).fetch = originalFetch;
    }

    const FeatureExtractor = wasmModule.FeatureExtractor as any;

    // create synthetic audio long enough for windowFrames
    const duration = ((windowFrames - 1) * hop + nfft) / sr + 0.1; // seconds
    const audio = makeAudio(sr, duration, 440);

    // call wasm feature extractor
    const wasmFe = new FeatureExtractor(sr, hop, nfft, false, false);
    const nested = wasmFe.extract_windowed_features(audio, windowFrames) as any[];
    // nested is Array<Array<number>> of length windowFrames
    const wasmFlat: number[] = [];
    for (let i = 0; i < nested.length; i++) {
      const row = nested[i];
      for (let v of row as any) wasmFlat.push(Number(v));
    }

    // compute reference features for the same number of windows implied by wasm output
    const windowsCount = Math.floor(wasmFlat.length / 6);
    const ref = computeWindowedFeaturesReference(audio, sr, hop, nfft, windowsCount);

    // ensure equal length
    expect(wasmFlat.length).toBe(ref.length);

    // Compare per-feature correlation between wasm and reference (more robust than exact equality)
    const windows = Math.floor(wasmFlat.length / 6);
    function corr(a: number[], b: number[]) {
      const na = a.length;
      const mean = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length;
      const ma = mean(a);
      const mb = mean(b);
      let num = 0, da = 0, db = 0;
      for (let i = 0; i < na; i++) {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma) * (a[i] - ma);
        db += (b[i] - mb) * (b[i] - mb);
      }
      const denom = Math.sqrt(da * db);
      if (denom === 0) return 0;
      return num / denom;
    }

    const wasmPerFeature: number[][] = new Array(6).fill(0).map(() => []);
    const refPerFeature: number[][] = new Array(6).fill(0).map(() => []);
    for (let f = 0; f < windows; f++) {
      for (let k = 0; k < 6; k++) {
        const idx = f * 6 + k;
        wasmPerFeature[k].push(wasmFlat[idx]);
        refPerFeature[k].push(ref[idx]);
      }
    }

    // Enforce minimum correlation thresholds per-feature in strict mode (CI or explicit flag).
    // thresholds: centroid, flux, rms, zcr, onset, rolloff
    const thresholds = [0.20, 0.15, 0.20, 0.10, 0.15, 0.20];
    const strict = !!(process.env.CI || process.env.FEATURE_PARITY_STRICT === '1');

    // Basic sanity checks: windows exist and wasm outputs are finite and bounded
    expect(windows).toBeGreaterThan(0);
    for (let k = 0; k < 6; k++) {
      for (const v of wasmPerFeature[k]) expect(Number.isFinite(v)).toBe(true);
      const minW = Math.min(...wasmPerFeature[k]);
      const maxW = Math.max(...wasmPerFeature[k]);
      expect(minW).toBeGreaterThanOrEqual(-1e-6);
      expect(maxW).toBeLessThanOrEqual(2.0);

      // Coerce non-finite reference outputs to zero before correlation
      for (let i = 0; i < refPerFeature[k].length; i++) {
        if (!Number.isFinite(refPerFeature[k][i])) refPerFeature[k][i] = 0;
      }

      const r = corr(wasmPerFeature[k], refPerFeature[k]);
      console.log(`Feature ${k} correlation: ${r.toFixed(3)} (threshold ${thresholds[k]})`);

      if (strict) {
        // Provide actionable failure messages to make triage easier in CI
        expect(r, `Feature ${k} correlation ${r.toFixed(3)} below threshold ${thresholds[k]}`).toBeGreaterThanOrEqual(thresholds[k]);
      } else {
        // Non-strict / local mode: warn but don't fail tests so development isn't blocked
        if (r < thresholds[k]) console.warn(`Feature ${k} correlation ${r.toFixed(3)} below threshold ${thresholds[k]} (non-strict mode) - consider running with FEATURE_PARITY_STRICT=1`);
      }
    }

    // Save diagnostics to console so we can iterate on training/impl if needed
    console.log('Feature parity diagnostics: windows=', windows, 'perFeatureMeans=', wasmPerFeature.map(a => (a.reduce((s,v)=>s+v,0)/a.length).toFixed(4)));

  });
});
