/**
 * Canonical runtime constants for the browser (issue #91).
 *
 * The wasm `constants()` binding (backed by runtime-core's SAMPLE_RATE /
 * HOP_LENGTH in controller.rs) is the SINGLE source of truth. This module
 * holds NO literal values — runtime code must read the constants from the
 * wasm module, and tests must inject/mock the binding constants rather
 * than creating a second authority.
 *
 * Contract: shared/golden_vectors.json feature_cases + FEATURE_VERSION.
 */

/**
 * Read the canonical runtime constants from the wasm module (runtime-core
 * authority). Throws if the wasm module has not supplied constants — there
 * is deliberately no literal fallback, so a missing binding surfaces as an
 * error instead of silently running against a second authority.
 */
export function getRuntimeConstants(wasmConstants: {
  sample_rate?: number;
  hop_length?: number;
  n_fft?: number;
}): { sampleRate: number; hopLength: number; nFft: number } {
  if (
    typeof wasmConstants?.sample_rate !== 'number' ||
    typeof wasmConstants?.hop_length !== 'number'
  ) {
    throw new Error(
      'Canonical runtime constants unavailable: wasm constants() did not ' +
        'supply sample_rate/hop_length. Initialize the wasm module before ' +
        'reading constants — there is no TypeScript fallback authority.'
    );
  }
  return {
    sampleRate: wasmConstants.sample_rate,
    hopLength: wasmConstants.hop_length,
    nFft: wasmConstants.n_fft ?? 0,
  };
}
