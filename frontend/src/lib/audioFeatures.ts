/**
 * @deprecated — RETIRED per ADR 0001 (feature-extraction contract, 2026-08-27).
 *
 * The JavaScript reimplementation of audio feature extraction was retired because
 * it drifted from the canonical Rust implementation (runtime-core/src/features.rs):
 * FFT size 2048 vs 4096, different smoothing, dB-domain math, and per-file
 * min-max normalization the browser cannot reproduce. The canonical path is now:
 *   frontend/src/lib/canonicalFeatures.ts → wasm-orbit FeatureExtractor → Rust
 *
 * This file is intentionally left as a hard-failing stub so that any accidental
 * re-import is caught at runtime. Do NOT add feature-extraction logic here.
 * If you need features in the browser, use WasmAudioFeatureSource from
 * canonicalFeatures.ts (fed by raw PCM, resampled to 48 kHz).
 *
 * This stub will be deleted entirely once all historical imports are confirmed
 * removed. See ADR 0001 and shared/golden_vectors.json (feature_version = features/2).
 */

export interface AudioFeatures {
  spectralCentroid: number;
  spectralFlux: number;
  rmsEnergy: number;
  zeroCrossingRate: number;
  onsets: number;
  spectralRolloff: number;
}

function retired(): never {
  throw new Error(
    "[audioFeatures.ts] RETIRED — use WasmAudioFeatureSource from canonicalFeatures.ts / wasm-orbit. " +
      "See ADR 0001 (feature-extraction contract). This file must not be used for training or inference."
  );
}

export class AudioFeatureExtractor {
  constructor(..._args: unknown[]) {
    retired();
  }
  extractFeatures(): AudioFeatures {
    return retired();
  }
  extractWindowedFeatures(): number[] {
    return retired();
  }
  getFrameCount(): number {
    return retired();
  }
  getBufferSize(): number {
    return retired();
  }
}
