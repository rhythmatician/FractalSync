# Boundary-Crossing Reward System

This document describes the boundary-crossing reward system implemented for FractalSync, which synchronizes Julia parameter `c` movement across the Mandelbrot set boundary with musical transitions.

## Overview

The boundary-crossing system rewards the model when the Julia parameter `c` crosses the Mandelbrot set boundary during significant musical events (hits, structural changes, dynamics shifts). This creates more engaging visualizations that respond meaningfully to the music.

## Key Components

### 1. Boundary Detection

#### `compute_boundary_distance(c_real, c_imag)`
Computes the approximate distance from a point to the Mandelbrot set boundary using the escape time algorithm.

- **Returns**: Normalized distance [0, 1] where 0 = on boundary, 1 = far from boundary
- **Example**:
  ```python
  from src.mandelbrot_orbits import compute_boundary_distance
  
  dist = compute_boundary_distance(0.0, 0.0)  # Origin
  print(f"Distance: {dist}")  # ~1.0 (inside set)
  ```

#### `detect_boundary_crossing(prev_real, prev_imag, curr_real, curr_imag)`
Detects if the trajectory crossed the Mandelbrot set boundary between two consecutive positions.

- **Returns**: Boolean indicating if a crossing occurred
- **Example**:
  ```python
  from src.mandelbrot_orbits import detect_boundary_crossing
  
  crossed = detect_boundary_crossing(0.0, 0.0, 0.5, 0.0)
  print(f"Crossed: {crossed}")
  ```

#### `compute_crossing_score(c_real, c_imag)`
Computes a score for how close a point is to the Mandelbrot boundary (higher = more interesting).

- **Returns**: Score [0, 1] where 1 = on boundary (most interesting)
- **Example**:
  ```python
  from src.mandelbrot_orbits import compute_crossing_score
  
  score = compute_crossing_score(-0.5, 0.3)
  print(f"Score: {score}")
  ```

### 2. Musical Transition Detection

#### `detect_musical_transitions(spectral_flux, onset_strength, rms_energy)`
Detects musical transitions by combining multiple audio features:
- Spectral flux spikes (sudden spectral changes)
- Onset detections (percussive hits)
- Energy changes (dynamics shifts)

- **Returns**: Binary array where 1 = transition detected
- **Example**:
  ```python
  from src.audio_features import detect_musical_transitions
  
  transitions = detect_musical_transitions(flux, onset, rms)
  print(f"Transitions: {np.sum(transitions)}")
  ```

#### `compute_transition_score(spectral_flux, onset_strength, rms_change)`
Computes a continuous transition score for a single frame.

- **Returns**: Score [0, 1] where higher = stronger transition
- **Weights**: (flux: 0.4, onset: 0.4, energy: 0.2)
- **Example**:
  ```python
  from src.audio_features import compute_transition_score
  
  score = compute_transition_score(0.8, 0.7, 0.3)
  print(f"Transition score: {score}")
  ```

### 3. Boundary Crossing Loss

The `BoundaryCrossingLoss` rewards the model when boundary crossings align with musical transitions:

```python
from src.physics_trainer import BoundaryCrossingLoss

loss_fn = BoundaryCrossingLoss(weight=0.5, sync_window=3)

# During training
loss = loss_fn(
    current_positions,  # (batch_size, 2)
    previous_positions, # (batch_size, 2)
    transition_scores,  # (batch_size,)
)
```

**Loss computation**:
- Detects boundary crossings between consecutive frames
- Computes crossing score (how interesting is the position)
- Multiplies crossing score × transition score
- Returns negative reward (minimize loss = maximize synchronization)

## Training with Boundary Crossing

### Command-Line Training

Add the `--boundary-crossing-weight` parameter to control the influence:

```bash
cd backend
python train_physics.py \
    --data-dir data/audio \
    --epochs 100 \
    --use-curriculum \
    --boundary-crossing-weight 0.5
```

**Recommended weights**:
- `0.0`: Disable boundary crossing reward (default behavior)
- `0.3-0.5`: Moderate influence (recommended for most cases)
- `0.7-1.0`: Strong influence (may overshadow other losses)

### Synthetic Trajectory Generation

Generate trajectories with intentional boundary crossings for testing:

```python
from src.mandelbrot_orbits import generate_boundary_crossing_trajectory

positions, velocities, crossings = generate_boundary_crossing_trajectory(
    n_samples=1000,
    n_crossings=10,
    crossing_intensity=0.5
)

print(f"Generated {np.sum(crossings)} crossings")
```

## Supervised Data Collection

The `collect_supervised_data.py` tool allows manual collection of trajectory data:

### Usage

```bash
cd backend
python collect_supervised_data.py path/to/audio.mp3 --output-dir data/supervised
```

### Controls

- **Mouse**: Move `c` parameter in complex plane
- **SPACE**: Play/Pause audio playback
- **R**: Toggle recording
- **S**: Save collected data
- **H**: Toggle help overlay
- **Q/ESC**: Quit

### Output Format

Data is saved as JSON with the following structure:

```json
{
  "metadata": {
    "audio_file": "path/to/audio.mp3",
    "audio_duration": 306.7,
    "recorded_frames": 1321
  },
  "trajectory": [
    {
      "frame": 0,
      "timestamp": 0.023,
      "c": {"real": 0.123, "imag": -0.456},
      "zoom": 1.0,
      "audio_features": {
        "spectral_centroid": 0.45,
        "spectral_flux": 0.67,
        "rms_energy": 0.82,
        ...
      },
      "transition_score": 0.75,
      "boundary_crossed": true,
      "boundary_distance": 0.12,
      "crossing_score": 0.88
    },
    ...
  ]
}
```

## Integration with Physics Trainer

The boundary crossing system is integrated into `PhysicsTrainer`:

1. **Transition Score Computation**: Computed for each frame during training
2. **Boundary Detection**: Checked between consecutive positions
3. **Loss Contribution**: Added to total loss weighted by `boundary_crossing_weight`
4. **History Tracking**: Logged in `training_history.json`

### Training History

New metrics added:
- `boundary_crossing_loss`: Average boundary crossing loss per epoch

## Benefits

1. **Musical Synchronization**: Visual changes align with musical events
2. **Engaging Visuals**: Crossing the boundary creates dramatic Julia set changes
3. **Interpretability**: Clear connection between audio and visual features
4. **Flexibility**: Adjustable weight allows tuning the effect strength

## Future Enhancements

Possible improvements:
1. **Learned Weights**: Automatically learn optimal boundary crossing weight
2. **Multi-Scale Transitions**: Detect transitions at different time scales
3. **Genre-Specific Models**: Train separate models for different music genres
4. **Anticipation**: Predict upcoming transitions and prepare crossings
5. **Trajectory Optimization**: Use RL to find optimal crossing trajectories

## Troubleshooting

### Low Crossing Score
If crossings aren't being rewarded:
- Check `boundary_crossing_weight` is > 0
- Verify transition detection is working (inspect `transition_scores`)
- Ensure model is learning (check other loss components)

### Too Many/Few Crossings
Adjust parameters:
- Increase `boundary_crossing_weight` for more crossings
- Decrease for fewer, more meaningful crossings
- Modify transition detection thresholds in `detect_musical_transitions()`

### Model Not Learning
- Check that curriculum learning is working (`--use-curriculum`)
- Verify audio features are normalized
- Ensure batch size and learning rate are appropriate
- Review training logs for NaN or exploding gradients

## References

- Mandelbrot set: https://en.wikipedia.org/wiki/Mandelbrot_set
- Julia sets: https://en.wikipedia.org/wiki/Julia_set
- Onset detection: https://librosa.org/doc/main/generated/librosa.onset.onset_detect.html
- Spectral features: https://librosa.org/doc/main/feature.html
