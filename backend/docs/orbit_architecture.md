# Orbit-First Architecture Implementation

## Overview

This implementation introduces a deterministic orbit-first synthesis approach for the FractalSync visualizer. Instead of predicting raw Julia parameter `c(t)` directly, the model predicts **control signals** that drive a **deterministic orbit synthesizer**.

## Architecture Components

### 1. **orbit_synth.py** - Orbit Synthesizer (NEW)
The authoritative source for converting control signals → c(t).

**Formula:**
```
c(t) = c_carrier(θ) + c_residual(ϕ_1, ..., ϕ_k)

where:
  c_carrier = s * LobePoint(lobe, sub_lobe, θ)
  c_residual = α * R * Σ(g_k / k² * exp(i*ϕ_k))
```

**Key Classes:**
- `OrbitState`: Complete state for synthesis (lobe, sub_lobe, θ, ω, s, α, residual_phases, residual_omegas)
- `OrbitSynthesizer`: Synthesizes c(t) from state using MandelbrotGeometry functions

**Authoritative Constants (from mandelbrot_orbits.py):**
- `MandelbrotGeometry.lobe_point_at_angle()` - Carrier position
- `MandelbrotGeometry.lobe_tangent_at_angle()` - Carrier velocity
- `MandelbrotGeometry.period_n_bulb_radius()` - Residual scaling

### 2. **control_model.py** - Control Signal Model (NEW)
Predicts control signals instead of raw c(t).

**Outputs:**
- `s_target`: Radius scaling [0.2, 3.0]
- `alpha`: Residual amplitude [0, 1]
- `omega_scale`: Angular velocity scale [0.1, 5.0]
- `band_gates`: Per-band residual gates [0, 1]^6

**NOT predicted per-frame:**
- `lobe` / `sub_lobe`: Controlled by slow section detection (boundary detector)

### 3. **control_trainer.py** - Control Signal Trainer (NEW)
Trains the control model with orbit synthesis in the loop.

**Training Flow:**
1. Model predicts control signals from audio features
2. Orbit synthesizer generates c(t) from controls
3. Julia sets rendered from synthesized c(t)
4. Correlation losses computed between audio and visual metrics
5. Curriculum loss compares controls to targets derived from preset orbits

**Loss Components:**
- Control loss: MSE between predicted and curriculum control targets
- Timbre-color correlation: spectral_centroid ↔ color_hue
- Transient-impact correlation: spectral_flux ↔ temporal_change

### 4. **live_controller.py** - Updated Live Controller
Now uses `OrbitSynthesizer` internally.

**Changes:**
- Replaced manual carrier/residual computation with `OrbitSynthesizer.step()`
- State now managed via `OrbitState` dataclass
- Impact envelope modulates `s` and `alpha` for punch
- Transition system updates `OrbitState.lobe` and `OrbitState.sub_lobe`

### 5. **train_orbit.py** - New Training Entry Point (NEW)
Command-line interface for orbit-based training.

**Usage:**
```bash
python backend/train_orbit.py --data-dir data/audio --epochs 100 --use-curriculum
```

**Key Arguments:**
- `--k-bands`: Number of residual epicycles (default: 6)
- `--curriculum-weight`: Weight for curriculum loss (default: 1.0)
- `--curriculum-decay`: Decay per epoch (default: 0.50)

## Integration with Existing System

### Training
- `train_orbit.py` → trains `AudioToControlModel` (control-based)

The orbit-based approach provides:
- **Deterministic behavior**: Same audio → same orbit
- **Interpretable controls**: s, alpha, omega have clear geometric meaning
- **Live performance**: Fast synthesis without model inference per frame

### Frontend Integration
The frontend will need to:
1. Run ONNX inference to get control signals
2. Call orbit synthesizer (JavaScript port) to get c(t)
3. Render Julia set with c(t)

## Testing

### Unit Tests
```bash
# Test orbit synthesizer
cd backend
python -c "from src.orbit_synth import OrbitSynthesizer, create_initial_state; synth = OrbitSynthesizer(); state = create_initial_state(); c = synth.synthesize(state); print(f'c = {c}')"

# Test control model
python test_control_model.py

# Test live controller with orbit synth
python -c "from src.live_controller import OrbitStateMachine; sm = OrbitStateMachine(); c = sm.step(); print(f'c = {c}')"
```

### Training Test
```bash
# Quick training run (1 epoch, CPU)
python backend/train_orbit.py \
    --data-dir data/audio \
    --epochs 1 \
    --batch-size 8 \
    --use-curriculum \
    --no-gpu-rendering \
    --julia-resolution 32 \
    --num-workers 0
```

## Mathematical Guarantees

### Carrier Orbit
- **Deterministic**: Same (lobe, sub_lobe, θ, s) → same c_carrier
- **Smooth**: Analytic derivatives via `lobe_tangent_at_angle()`
- **Boundary-aware**: s ≈ 1.0 → near Mandelbrot boundary

### Residual Texture
- **Capped**: |c_residual| ≤ residual_cap * lobe_radius
- **Harmonically structured**: Amplitude ∝ 1/k² (natural decay)
- **Audio-reactive**: band_gates modulate per-frequency components

### Velocity Computation
```
dc/dt = ω * d/dθ[c_carrier] + Σ(ω_k * i * amplitude_k * exp(i*ϕ_k))
```
Both terms are analytic, enabling smooth motion.

## Next Steps

1. **Frontend JavaScript Port**: Port `OrbitSynthesizer` to TypeScript for browser
2. **ONNX Integration**: Update frontend to use control model ONNX
3. **Live Testing**: Test with real audio input via microphone
4. **Visual Tuning**: Adjust correlation weights for desired aesthetic

## Files Created/Modified

**Created:**
- `backend/src/orbit_synth.py` - Orbit synthesizer
- `backend/src/control_model.py` - Control signal model
- `backend/src/control_trainer.py` - Control trainer
- `backend/train_orbit.py` - Training entry point
- `backend/docs/orbit_architecture.md` - This document

**Modified:**
- `backend/src/live_controller.py` - Now uses OrbitSynthesizer

**Authoritative sources:**
- `backend/src/mandelbrot_orbits.py` - Geometric constants
