# Physics-Based Model Documentation

## Overview

This implementation adds a physics-based approach to learning Julia set parameter trajectories from audio features. Instead of directly predicting the position of the Julia parameter `c`, the model predicts its **velocity**, treating `c` as a physical object moving through the complex plane.

## Key Components

### 1. Mandelbrot Orbits (`mandelbrot_orbits.py`)

Provides preset trajectories along significant features of the Mandelbrot set for curriculum learning.

**Available Presets:**
- `cardioid`: Main heart-shaped body of the Mandelbrot set
- `bulb`: Circular left bulb
- `period3_upper`, `period3_lower`: Decorative period-3 bulbs
- `horizontal_sweep`: Linear motion across interesting regions
- `vertical_sweep`: Vertical motion through upper regions
- `spiral_antenna`: Spiral inward toward the main antenna
- `figure8`: Complex path through bulb and cardioid

**Usage:**
```python
from src.mandelbrot_orbits import get_preset_orbit, generate_curriculum_sequence

# Get a specific orbit
orbit = get_preset_orbit("cardioid")
positions = orbit.sample(n_samples=100)
velocities = orbit.compute_velocities(n_samples=100)

# Generate curriculum sequence (simple to complex)
all_positions, all_velocities = generate_curriculum_sequence(n_samples=1000)
```

### 2. Physics Model (`physics_model.py`)

A neural network that predicts velocity instead of position, with speed modulated by audio loudness (RMS energy).

**Key Features:**
- **Velocity Prediction**: Predicts `dc/dt` instead of `c`
- **Audio-Driven Speed**: RMS energy modulates velocity magnitude
- **State Tracking**: Maintains position and velocity for inference
- **Damping**: Configurable damping factor for smooth motion
- **Constraint**: Keeps `|c| < 2` for visually interesting Julia sets

**Architecture:**
```
Input: Audio features (6 features × window_frames)
    ↓
Encoder: MLP [128, 256, 128] with BatchNorm, ReLU, Dropout
    ↓
Velocity Predictor: [64, 2] → [v_real, v_imag]
    ↓
RMS Modulation: velocity_magnitude × audio_rms × speed_scale
    ↓
Integration: position += velocity × dt
    ↓
Output: [v_real, v_imag, c_real, c_imag, hue, sat, bright, zoom, speed]
```

**Usage:**
```python
from src.physics_model import PhysicsAudioToVisualModel

model = PhysicsAudioToVisualModel(
    window_frames=10,
    hidden_dims=[128, 256, 128],
    output_dim=9,
    predict_velocity=True,
    damping_factor=0.95,  # Higher = less damping
    speed_scale=0.1,      # Velocity scaling
)

# Forward pass
output = model(audio_features, audio_rms=rms_values)
# output: [v_real, v_imag, c_real, c_imag, hue, sat, bright, zoom, speed]

# Reset state for new audio sequence
model.reset_state(position=(0.0, 0.0))
```

### 3. Physics Trainer (`physics_trainer.py`)

Training pipeline with curriculum learning support.

**Key Features:**
- **Curriculum Learning**: Starts with known Mandelbrot orbits
- **Velocity Loss**: Compares predicted to target velocities
- **Acceleration Smoothness**: Penalizes rapid velocity changes
- **Curriculum Decay**: Reduces curriculum weight over epochs
- **Standard Losses**: Maintains correlation and smoothness losses

**Loss Function:**
```
Total Loss = 
    timbre_color_loss +
    transient_impact_loss +
    silence_stillness_loss +
    distortion_roughness_loss +
    curriculum_weight × velocity_loss +
    acceleration_smoothness +
    parameter_smoothness
```

**Usage:**
```python
from src.physics_trainer import PhysicsTrainer

trainer = PhysicsTrainer(
    model=model,
    feature_extractor=feature_extractor,
    visual_metrics=visual_metrics,
    use_curriculum=True,
    curriculum_weight=1.0,
)

trainer.train(
    dataset=dataset,
    epochs=100,
    batch_size=32,
    curriculum_decay=0.95,  # Weight decays: 1.0 → 0.95 → 0.90...
)
```

## Training Script (`train_physics.py`)

Command-line interface for training physics-based models.

**Basic Usage:**
```bash
cd backend
python train_physics.py --data-dir data/audio --epochs 100 --use-curriculum
```

**Common Options:**
```bash
# Use curriculum learning with custom parameters
python train_physics.py \
    --data-dir data/audio \
    --epochs 100 \
    --batch-size 32 \
    --use-curriculum \
    --curriculum-weight 2.0 \
    --curriculum-decay 0.9 \
    --damping-factor 0.95 \
    --speed-scale 0.1 \
    --export-onnx

# Train without curriculum (pure correlation learning)
python train_physics.py \
    --data-dir data/audio \
    --epochs 100 \
    --batch-size 32

# Quick test run
python train_physics.py \
    --data-dir data/audio \
    --epochs 1 \
    --batch-size 16
```

**All Options:**
- `--data-dir`: Audio files directory
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--learning-rate`: Learning rate (default: 1e-4)
- `--window-frames`: Input window size (default: 10)
- `--use-curriculum`: Enable curriculum learning
- `--curriculum-weight`: Initial curriculum weight (default: 1.0)
- `--curriculum-decay`: Curriculum decay per epoch (default: 0.95)
- `--damping-factor`: Velocity damping (default: 0.95)
- `--speed-scale`: Velocity magnitude scaling (default: 0.1)
- `--save-dir`: Checkpoint directory (default: models/physics)
- `--device`: Training device (cuda/cpu)
- `--export-onnx`: Export final model to ONNX

## Physics Formulation

### Position vs Velocity Prediction

**Direct Position (Original Model):**
```
c = model(audio_features)
```

**Physics-Based Velocity (New Model):**
```
v = model(audio_features)
v_magnitude = |v| × audio_rms × speed_scale
c_new = c_old + v × dt
```

### Audio-Speed Coupling

The model learns a direction for motion, but the speed is modulated by audio loudness:
- High RMS → fast movement → dramatic visual changes
- Low RMS → slow movement → subtle morphing
- Direction is learned from other audio features (spectral, temporal)

### State Integration

For training:
```python
positions, velocities = integrate_across_batch(predicted_velocity, curriculum_positions)
```

For inference:
```python
model.reset_state(position=(0.0, 0.0))
for audio_chunk in audio_stream:
    output = model(audio_chunk)
    # Position is automatically integrated and constrained
```

## Curriculum Learning Strategy

The curriculum progresses from simple to complex trajectories:

1. **Phase 1 (Early Epochs)**: Simple 1D sweeps
   - `horizontal_sweep`: Left-right motion
   - `vertical_sweep`: Up-down motion

2. **Phase 2 (Mid Epochs)**: Smooth curves
   - `cardioid`: Main Mandelbrot body
   - `bulb`: Circular motion

3. **Phase 3 (Later Epochs)**: Complex paths
   - `figure8`: Crossing trajectories
   - `spiral_antenna`: Accelerating motion
   - `period3_upper/lower`: Small-scale features

The curriculum weight decays exponentially, so the model gradually transitions from mimicking known trajectories to freely learning from audio-visual correlations.

## Comparison: Direct vs Physics Models

| Aspect | Direct Model | Physics Model |
|--------|--------------|---------------|
| Output | Position `c` | Velocity `dc/dt` |
| Audio coupling | Implicit | Explicit via RMS |
| Temporal consistency | Smoothness loss | Physics integration + damping |
| Curriculum | Difficult | Natural (velocity targets) |
| Interpretability | Low | High (speed = loudness) |
| Complexity | Lower | Higher |

## Model Selection

**Use Direct Model (`model.py`) when:**
- You want simplicity
- Audio-to-visual mapping is non-physical
- Training data is limited

**Use Physics Model (`physics_model.py`) when:**
- You want physical interpretability
- Speed should correlate with loudness
- You have curriculum data available
- Temporal consistency is critical

## Testing

```bash
cd backend

# Test Mandelbrot orbits
python -c "from src.mandelbrot_orbits import list_preset_names; print(list_preset_names())"

# Test physics model creation
python -c "from src.physics_model import PhysicsAudioToVisualModel; print('OK')"

# Run full test suite (requires pytest and dependencies)
python -m pytest tests/test_physics_model.py -v
```

## Integration with Existing Code

The physics model is fully compatible with existing infrastructure:

1. **Export**: Works with existing `export_model.py` ONNX export
2. **API**: Can be served via existing `api/server.py` endpoints
3. **Frontend**: ONNX model can be loaded by existing `modelInference.ts`

To use in API:
```python
# In api/server.py, add endpoint for physics training
from src.physics_model import PhysicsAudioToVisualModel
from src.physics_trainer import PhysicsTrainer

# Create physics model instead of regular model
model = PhysicsAudioToVisualModel(...)
trainer = PhysicsTrainer(...)
```

## Future Extensions

Possible enhancements:
1. **Acceleration Prediction**: Predict `d²c/dt²` for even more complex dynamics
2. **Multi-Body**: Multiple interacting Julia parameters
3. **Force Fields**: Audio creates potential fields that attract/repel `c`
4. **Learned Physics**: Learn damping and speed scaling instead of fixing them
5. **Orbit Generation**: Use GAN to generate new Mandelbrot orbits

## References

- Julia set theory: Points `z` where `z → z² + c` remains bounded
- Mandelbrot set: Set of `c` values where Julia set is connected
- Curriculum learning: Bengio et al., 2009
- Physics-informed neural networks: Raissi et al., 2019
