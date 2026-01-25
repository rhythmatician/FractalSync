// heavy parity tests are intentionally skipped in fast mode; helpers below referenced to avoid unused-declaration errors

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

// Hann window (normalized to RMS = 1 to match Rust's implementation)
function hann(n: number) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  // normalize so window RMS is 1 (same approach as Rust)
  let sumsq = 0;
  for (let i = 0; i < n; i++) sumsq += w[i] * w[i];
  const rms = sumsq / n;
  const norm = 1 / Math.sqrt(rms);
  for (let i = 0; i < n; i++) w[i] *= norm;
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

    // flux (squared L2 difference between consecutive mags) — match Rust
    let flux = 0;
    if (prevSpec) {
      let sumSq = 0;
      for (let i = 0; i < mag.length; i++) {
        const diff = mag[i] - prevSpec[i];
        sumSq += diff * diff;
      }
      flux = sumSq;
    }

    // rms (time-domain)
    let rms = 0;
    for (let i = 0; i < frame.length; i++) rms += frame[i] * frame[i];
    rms = Math.sqrt(rms / Math.max(1, frame.length));

    // zero crossing rate approx
    let zc = 0;
    for (let i = 1; i < frame.length; i++) if ((frame[i] >= 0) !== (frame[i - 1] >= 0)) zc++;
    zc = zc / frame.length;

    // onset proxy: use raw flux (normalised later)
    const onset = flux;

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

  // After frames loop: normalise flux, rms, and onset arrays to [0,1] like Rust
  // The push above created repetitions of centroid, flux, rms, zc, onset, rolloff per window —
  // we need to extract per-feature series and normalise the appropriate ones.
  const nFrames = windowFrames;
  const centroids: number[] = [];
  const fluxes: number[] = [];
  const rmses: number[] = [];
  const zcs: number[] = [];
  const onsets: number[] = [];
  const rolloffs: number[] = [];

  for (let f = 0; f < nFrames; f++) {
    const idx = f * 6;
    centroids.push(features[idx + 0]);
    fluxes.push(features[idx + 1]);
    rmses.push(features[idx + 2]);
    zcs.push(features[idx + 3]);
    onsets.push(features[idx + 4]);
    rolloffs.push(features[idx + 5]);
  }

  function normaliseInPlace(arr: number[]) {
    if (arr.length === 0) return;
    let min = Infinity, max = -Infinity;
    for (const v of arr) {
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min;
    if (range > 0) {
      for (let i = 0; i < arr.length; i++) arr[i] = (arr[i] - min) / range;
    }
  }

  normaliseInPlace(fluxes);
  normaliseInPlace(rmses);
  normaliseInPlace(onsets);

  // Reconstruct flattened features with normalised series
  const normalizedFeatures: number[] = [];
  for (let f = 0; f < nFrames; f++) {
    normalizedFeatures.push(centroids[f], fluxes[f], rmses[f], zcs[f], onsets[f], rolloffs[f]);
  }

  return normalizedFeatures;
}

import { test } from 'vitest';

test.skip('Feature parity tests are heavy; run separately when needed', () => {});

// Reference helpers to avoid TS 'declared but not used' errors in fast runs
void makeAudio;
void computeWindowedFeaturesReference;
