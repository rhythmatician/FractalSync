/**
 * Canonical audio feature extraction for the browser.
 *
 * Wraps the wasm-orbit FeatureExtractor — the SAME Rust implementation the
 * trainer uses (runtime-core/src/features.rs). The browser feeds a rolling
 * buffer of raw PCM samples from AnalyserNode.getFloatTimeDomainData and
 * receives feature windows laid out identically to training inputs.
 *
 * This eliminates the entire class of browser-vs-trainer extraction drift
 * by construction: there is only one implementation, executed in two places.
 *
 * Contract: shared/golden_vectors.json feature_cases + FEATURE_VERSION.
 */

/** Runtime sample rate the canonical extractor expects (48 kHz). */
export const RUNTIME_SAMPLE_RATE = 48000;

/** Samples of PCM history kept for extraction (≈1 s at 48 kHz). */
const HISTORY_SAMPLES = 48000;

interface WasmFeatureExtractorShape {
  new (): {
    num_features_per_frame: number;
    extract_window(audio: Float32Array | number[], windowFrames: number): number[];
  };
}

let wasmExtractor: InstanceType<WasmFeatureExtractorShape> | null = null;

/**
 * Inject the wasm-backed extractor constructor. Called once after
 * initOrbitSynth() resolves, using the loaded module's FeatureExtractor.
 */
export function setWasmFeatureExtractor(
  Ctor: WasmFeatureExtractorShape | null
): void {
  wasmExtractor = Ctor ? new (Ctor as any)() : null;
}

function requireExtractor(): NonNullable<typeof wasmExtractor> {
  if (!wasmExtractor) {
    throw new Error(
      '[audioFeatures] wasm FeatureExtractor not initialized — call ' +
        'initOrbitSynth() then setWasmFeatureExtractor(mod.FeatureExtractor) first'
    );
  }
  return wasmExtractor;
}

/**
 * Resample a Float32Array from `fromRate` to the runtime sample rate
 * (48 kHz) with linear interpolation. Browser AudioContexts commonly run
 * at 44.1 kHz; feeding that straight into the 48 kHz-tuned extractor would
 * silently shift every spectral feature.
 */
export function resampleToRuntimeRate(
  input: Float32Array,
  fromRate: number
): Float32Array {
  if (fromRate === RUNTIME_SAMPLE_RATE) return input;
  const ratio = RUNTIME_SAMPLE_RATE / fromRate;
  const outLen = Math.max(1, Math.round(input.length * ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcPos = i / ratio;
    const i0 = Math.floor(srcPos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = srcPos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

/**
 * Rolling PCM feeder: accumulates resampled audio and extracts canonical
 * feature windows on demand.
 */
export class WasmAudioFeatureSource {
  private history: Float32Array;
  private writePos = 0;
  private filled = 0;

  constructor(historySamples: number = HISTORY_SAMPLES) {
    this.history = new Float32Array(historySamples);
  }

  /** Push newly captured PCM (already at context rate); resamples if needed. */
  push(pcm: Float32Array, contextSampleRate: number): void {
    const resampled = resampleToRuntimeRate(pcm, contextSampleRate);
    for (let i = 0; i < resampled.length; i++) {
      this.history[this.writePos] = resampled[i];
      this.writePos = (this.writePos + 1) % this.history.length;
      if (this.filled < this.history.length) this.filled++;
    }
  }

  /**
   * Extract one flattened frame-major feature window over the buffered
   * audio, in chronological order (oldest → newest).
   */
  extractWindow(windowFrames: number): number[] {
    const fe = requireExtractor();
    // Unroll the circular buffer into chronological order.
    const chronological = new Float32Array(this.filled);
    const start = (this.writePos - this.filled + this.history.length) % this.history.length;
    for (let i = 0; i < this.filled; i++) {
      chronological[i] = this.history[(start + i) % this.history.length];
    }
    return fe.extract_window(Array.from(chronological), windowFrames);
  }

  get samplesBuffered(): number {
    return this.filled;
  }
}
