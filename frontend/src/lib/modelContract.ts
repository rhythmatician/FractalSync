export const MODEL_INPUT_NAME = "audio_features";
export const MODEL_OUTPUT_NAME = "control_signals";

export const FEATURE_NAMES = [
  "spectral_centroid",
  "spectral_flux",
  "rms_energy",
  "zero_crossing_rate",
  "onset_strength",
  "spectral_rolloff",
] as const;

export const DEFAULT_WINDOW_FRAMES = 10;
export const DEFAULT_K_BANDS = 6;

export type FeatureName = (typeof FEATURE_NAMES)[number];

export function buildInputNames(
  windowFrames: number = DEFAULT_WINDOW_FRAMES,
  featureNames: readonly FeatureName[] = FEATURE_NAMES
): string[] {
  const names: string[] = [];
  for (let frameIdx = 0; frameIdx < windowFrames; frameIdx += 1) {
    for (const feat of featureNames) {
      names.push(`frame_${frameIdx}_${feat}`);
    }
  }
  return names;
}

export function buildOutputNames(kBands: number = DEFAULT_K_BANDS): string[] {
  const names = ["s_target", "alpha", "omega_scale"];
  for (let i = 0; i < kBands; i += 1) {
    names.push(`band_gate_${i}`);
  }
  return names;
}

export const INPUT_NAMES = buildInputNames();
export const OUTPUT_NAMES = buildOutputNames();
export const INPUT_DIM = INPUT_NAMES.length;
export const OUTPUT_DIM = OUTPUT_NAMES.length;
