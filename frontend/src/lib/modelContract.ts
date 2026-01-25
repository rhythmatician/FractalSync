export const MODEL_INPUT_NAME = "audio_features";
export const MODEL_OUTPUT_NAME = "control_signals";

export const INPUT_NAMES = [
  "frame_0_spectral_centroid",
  "frame_0_spectral_flux",
  "frame_0_rms_energy",
  "frame_0_zero_crossing_rate",
  "frame_0_onset_strength",
  "frame_0_spectral_rolloff",
  "frame_1_spectral_centroid",
  "frame_1_spectral_flux",
  "frame_1_rms_energy",
  "frame_1_zero_crossing_rate",
  "frame_1_onset_strength",
  "frame_1_spectral_rolloff",
  "frame_2_spectral_centroid",
  "frame_2_spectral_flux",
  "frame_2_rms_energy",
  "frame_2_zero_crossing_rate",
  "frame_2_onset_strength",
  "frame_2_spectral_rolloff",
  "frame_3_spectral_centroid",
  "frame_3_spectral_flux",
  "frame_3_rms_energy",
  "frame_3_zero_crossing_rate",
  "frame_3_onset_strength",
  "frame_3_spectral_rolloff",
  "frame_4_spectral_centroid",
  "frame_4_spectral_flux",
  "frame_4_rms_energy",
  "frame_4_zero_crossing_rate",
  "frame_4_onset_strength",
  "frame_4_spectral_rolloff",
  "frame_5_spectral_centroid",
  "frame_5_spectral_flux",
  "frame_5_rms_energy",
  "frame_5_zero_crossing_rate",
  "frame_5_onset_strength",
  "frame_5_spectral_rolloff",
  "frame_6_spectral_centroid",
  "frame_6_spectral_flux",
  "frame_6_rms_energy",
  "frame_6_zero_crossing_rate",
  "frame_6_onset_strength",
  "frame_6_spectral_rolloff",
  "frame_7_spectral_centroid",
  "frame_7_spectral_flux",
  "frame_7_rms_energy",
  "frame_7_zero_crossing_rate",
  "frame_7_onset_strength",
  "frame_7_spectral_rolloff",
  "frame_8_spectral_centroid",
  "frame_8_spectral_flux",
  "frame_8_rms_energy",
  "frame_8_zero_crossing_rate",
  "frame_8_onset_strength",
  "frame_8_spectral_rolloff",
  "frame_9_spectral_centroid",
  "frame_9_spectral_flux",
  "frame_9_rms_energy",
  "frame_9_zero_crossing_rate",
  "frame_9_onset_strength",
  "frame_9_spectral_rolloff",
];

export const DEFAULT_K_BANDS = 6;
export const INPUT_DIM = INPUT_NAMES.length;
export const OUTPUT_NAMES = buildOutputNames(DEFAULT_K_BANDS);
export const OUTPUT_DIM = OUTPUT_NAMES.length;

export function buildOutputNames(kBands = 6) {
  const names = ["s_target", "alpha", "omega_scale"];
  for (let i = 0; i < kBands; i += 1) { names.push(`band_gate_${i}`); }
  return names;
}
