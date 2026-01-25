# Model I/O Contract (Generated)

> Auto-generated from `contracts/model_io_contract.json`

**version**: `fc2dcd98c3dc6f33ea1dc6645fb30357e450f7cefbe4d10dd447655bc8770e34`  (SHA256 of contract body)

## Input
- name: `audio_features`
- window_frames: 10
- features_per_frame: 6

### Feature names (per-frame, oldest->newest)
- `spectral_centroid`
- `spectral_flux`
- `rms_energy`
- `zero_crossing_rate`
- `onset_strength`
- `spectral_rolloff`

## Output
- name: `control_signals`
- k_bands: 6

### Output elements (order)
- `s_target`
- `alpha`
- `omega_scale`
- `band_gate_0`
- `band_gate_1`
- `band_gate_2`
- `band_gate_3`
- `band_gate_4`
- `band_gate_5`

---
This document is generated; update `contracts/model_io_contract.json` instead.
