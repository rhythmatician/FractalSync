/**
 * Canonical audio feature constants for the browser (issue #91).
 *
 * As of issue #91, canonical feature extraction runs inside the Rust
 * `AnalysisTimebase` (runtime-core/src/timebase.rs) via the wasm-orbit
 * binding — the browser no longer maintains a separate rolling PCM buffer or
 * a per-chunk resampler in TypeScript. This module now only exposes the
 * canonical runtime constants (sample rate, hop length) so pure/test code
 * can reference the runtime-core authority without a wasm instance.
 *
 * The wasm `constants()` binding is the single source of truth; the literals
 * here exist only so pure/test code can run before wasm init, and are pinned
 * to runtime-core (SAMPLE_RATE / HOP_LENGTH in controller.rs).
 *
 * Contract: shared/golden_vectors.json feature_cases + FEATURE_VERSION.
 */

/** Runtime sample rate the canonical extractor expects (48 kHz). */
export const RUNTIME_SAMPLE_RATE = 48000;

/** Canonical hop length between STFT frames (runtime-core authority). */
export const HOP_LENGTH = 1024;

/**
 * Read the canonical runtime constants from the wasm module (runtime-core
 * authority) when available, falling back to the pinned literals above.
 */
export function getRuntimeConstants(wasmConstants?: {
  sample_rate?: number;
  hop_length?: number;
  n_fft?: number;
}): { sampleRate: number; hopLength: number } {
  return {
    sampleRate: wasmConstants?.sample_rate ?? RUNTIME_SAMPLE_RATE,
    hopLength: wasmConstants?.hop_length ?? HOP_LENGTH,
  };
}
