# Model I/O Contract (Single Source of Truth)

This document defines the **authoritative** ONNX model input/output contract used by:

1) Training/export (`backend/src`),
2) Runtime inference (`frontend/src`), and
3) Evaluation harness (`scripts/eval_show_readiness.py`).

The constants live in:
- `backend/src/model_contract.py`
- `frontend/src/lib/modelContract.ts`

Any change to the tensor order **must** update those modules and this document.

---

## 1) Inputs

**Input tensor name**: `audio_features`
**Shape**: `[batch, input_dim]`
**Dtype**: `float32`
**Normalization**: `(x - feature_mean) / (feature_std + 1e-8)` when metadata provides stats.

### Feature layout (flattened)
The input is a **sliding window** of audio features, flattened in time order
from **oldest → newest**:

```
frame_0_spectral_centroid, frame_0_spectral_flux, frame_0_rms_energy,
frame_0_zero_crossing_rate, frame_0_onset_strength, frame_0_spectral_rolloff,
frame_1_spectral_centroid, frame_1_spectral_flux, ...
frame_9_spectral_rolloff
```

**Default window_frames**: `10`
**Features per frame**: `6`
**Default input_dim**: `60`

### Reference input tensor layout table
| Index range | Name pattern | Description | Range |
|------------|--------------|-------------|-------|
| `0..5` | `frame_0_*` | Oldest frame | normalized [0,1] |
| `6..11` | `frame_1_*` | | normalized [0,1] |
| … | … | … | … |
| `54..59` | `frame_9_*` | Most recent frame | normalized [0,1] |

> **Note:** Controller state features and DF probes are **not** part of the current model input.
> The distance field is used downstream in the runtime stepper.

---

## 2) Outputs

**Output tensor name**: `control_signals`
**Shape**: `[batch, output_dim]`
**Dtype**: `float32`

We use **absolute control semantics** (not deltas).

### Output elements (order matters)
| Index | Name | Range | Interpretation |
|-------|------|-------|----------------|
| 0 | `s_target` | [0.2, 3.0] | Radius scaling toward/away from boundary |
| 1 | `alpha` | [0.0, 1.0] | Residual amplitude |
| 2 | `omega_scale` | [0.1, 5.0] | Angular velocity scale (multiplied by base ω) |
| 3..(3+k-1) | `band_gate_i` | [0.0, 1.0] | Per-band residual gate (k = number of bands) |

**Default k (bands)**: `6`
**Default output_dim**: `9`

### Reference output tensor layout table
| Index | Name |
|-------|------|
| 0 | `s_target` |
| 1 | `alpha` |
| 2 | `omega_scale` |
| 3 | `band_gate_0` |
| 4 | `band_gate_1` |
| 5 | `band_gate_2` |
| 6 | `band_gate_3` |
| 7 | `band_gate_4` |
| 8 | `band_gate_5` |

---

## 3) Post-processing & Runtime Integration

At runtime, the control signals are **directly applied** to the orbit state:

```
s = s_target
alpha = alpha
omega = base_omega * omega_scale
```

Then **every frame** we call:

```
OrbitState.step_advanced(dt, residual_params, band_gates, h, d_star, max_step, distance_field)
```

where:
- `h` is transient strength (derived from spectral flux),
- `distance_field` is the Mandelbrot DF for contour-biased stepping,
- `d_star` and `max_step` are configured in `show_control.json`.

---

## 4) Contract Assertions

Required checks (implemented in code):
1) **Exporter** writes `parameter_names` matching the contract list.
2) **Frontend runtime** throws if output dimension doesn’t match `3 + k_bands`.
3) **Tests** fail if metadata parameter names drift.

---

## 5) Model Metadata Fields (exported)

The ONNX metadata JSON must contain:
- `input_shape`
- `input_name`
- `input_feature_names`
- `output_dim`
- `output_name`
- `parameter_names`
- `parameter_ranges`
- `feature_mean` / `feature_std` (optional)

These fields are used by training, runtime, and evaluation to validate consistency.
