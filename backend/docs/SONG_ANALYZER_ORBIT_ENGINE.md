# Song Analyzer and Orbit Engine

This directory contains two new modules for advanced audio analysis and synthetic data generation.

## Modules

### 1. `song_analyzer.py` - Real-Time Audio Analysis

The `SongAnalyzer` class provides comprehensive audio analysis capabilities:

**Features:**
- **Tempo Detection**: Global and local tempo estimation
- **Section Boundaries**: Automatic detection of song sections (intro, verse, chorus, etc.)
- **Onset Detection**: Identifies hits, transients, and significant audio events
- **Beat Tracking**: Extracts beat positions throughout the song
- **Time Conversion**: Utilities for converting between frames and seconds

**Usage Example:**
```python
from src.song_analyzer import SongAnalyzer
import numpy as np

# Create analyzer
analyzer = SongAnalyzer(sr=22050, hop_length=512)

# Analyze audio
audio = np.random.randn(5 * 22050)  # 5 seconds
analysis = analyzer.analyze_song(audio)

# Access results
print(f"Tempo: {analysis['tempo']} BPM")
print(f"Section boundaries: {analysis['section_boundaries']}")
print(f"Onsets: {analysis['onset_frames']}")
```

**Key Methods:**
- `analyze_song(audio)`: Comprehensive analysis returning all features
- `get_tempo_at_frame(frame_idx, local_tempo, global_tempo)`: Get tempo at specific frame
- `is_near_section_boundary(frame_idx, boundaries, tolerance)`: Check proximity to boundaries
- `get_hit_events(onset_frames, onset_strength, threshold)`: Extract strong hit events
- `frames_to_time(frames)`: Convert frame indices to seconds
- `time_to_frames(time_sec)`: Convert seconds to frame indices

### 2. `orbit_engine.py` - Synthetic Training Data Generation

The `OrbitEngine` class generates synthetic trajectories around Mandelbrot set structures with correlated audio features.

**Features:**
- **Multiple Correlation Types**: velocity, position, or acceleration-based
- **Curriculum Learning**: Generates data from multiple orbit presets
- **Windowed Format**: Outputs data ready for model training
- **Realistic Audio Features**: Synthetic features that mimic real audio properties

**Usage Example:**
```python
from src.orbit_engine import OrbitEngine, create_synthetic_dataset

# Create engine
engine = OrbitEngine(n_audio_features=6)

# Generate trajectory
audio_features, visual_params = engine.generate_synthetic_trajectory(
    orbit_name="cardioid_boundary",
    n_samples=100,
    audio_correlation="velocity"
)

# Or use convenience function for complete dataset
windowed_audio, windowed_visual, metadata = create_synthetic_dataset(
    n_samples=1000,
    window_frames=10,
    n_audio_features=6
)
```

**Key Methods:**
- `generate_synthetic_trajectory(orbit_name, n_samples, audio_correlation)`: Generate single trajectory
- `generate_mixed_curriculum(n_samples, orbit_names, correlation_types)`: Generate diverse dataset
- `generate_windowed_features(audio_features, visual_params, window_frames)`: Convert to model input format

**Correlation Types:**
- **velocity**: Audio features correlate with trajectory speed (faster motion → higher energy)
- **position**: Audio features correlate with position in complex plane
- **acceleration**: Audio features correlate with trajectory acceleration (changes in speed)

## Integration with Training

### Using Synthetic Data Augmentation

The orbit engine can be used to augment real audio training data:

```python
from src.data_loader import AudioDataset
from src.orbit_engine import create_synthetic_dataset
from torch.utils.data import ConcatDataset

# Load real audio data
real_dataset = AudioDataset(data_dir="data/audio", ...)

# Generate synthetic data
synthetic_audio, synthetic_visual, _ = create_synthetic_dataset(n_samples=1000)

# Combine datasets
# (You would need to wrap synthetic data in a PyTorch Dataset)
combined_dataset = ...  # See examples/mixed_dataset_demo.py
```

### Benefits of Synthetic Data

1. **Parameter Space Coverage**: Ensures the model sees diverse c-parameter values
2. **Curriculum Learning**: Can order training from simple to complex trajectories
3. **Data Augmentation**: Increases training data size without collecting more audio
4. **Controllable Correlations**: Explicitly define how audio maps to visual parameters

## Examples

### Running the Demos

1. **Basic functionality demo:**
```bash
cd backend
python examples/analyzer_orbit_demo.py
```

2. **Mixed dataset demo:**
```bash
cd backend
python examples/mixed_dataset_demo.py
```

### Using with Training

See `examples/mixed_dataset_demo.py` for a complete example of:
- Creating a mixed dataset (real + synthetic)
- Using it with PyTorch DataLoader
- Inspecting sample metadata

## Testing

Run the test suites:

```bash
cd backend
python -m unittest tests.test_song_analyzer -v
python -m unittest tests.test_orbit_engine -v
```

**Test Coverage:**
- `test_song_analyzer.py`: 10 tests covering all analyzer functionality
- `test_orbit_engine.py`: 15 tests covering trajectory generation and feature creation

## Implementation Details

### Song Analyzer

The analyzer uses librosa for audio processing:
- **Tempo**: librosa's beat tracking with tempogram for local variations
- **Section boundaries**: Self-similarity matrix analysis using MFCCs and chroma features
- **Onsets**: Spectral flux-based onset detection
- **Beats**: Beat tracking algorithm from librosa

### Orbit Engine

The orbit engine builds on the existing `mandelbrot_orbits.py` module:
- Uses preset orbits (cardioid, period-2, period-3, etc.)
- Generates audio features using mathematical correlations
- Supports variable feature counts (not just 6)
- Adds realistic noise for feature variability

## Future Enhancements

Potential improvements:
1. **Boundary Crossing Rewards**: Add loss function that rewards model when trajectory crosses section boundaries
2. **Real-time Analysis**: Optimize song analyzer for streaming audio
3. **More Correlation Types**: Add harmonic, rhythmic, or spectral correlations
4. **Adaptive Curriculum**: Automatically adjust synthetic data difficulty based on model performance
5. **Audio Synthesis**: Generate actual audio from trajectories for validation

## References

- [librosa documentation](https://librosa.org/doc/latest/index.html)
- [Mandelbrot set geometry](../src/mandelbrot_orbits.py)
- [Audio feature extraction](../src/audio_features.py)
