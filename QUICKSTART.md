# Quick Start: Orbit-First Training

## Prerequisites

Ensure you have audio files in `backend/data/audio/` directory.

## Basic Training

Run orbit-based training with default settings:

```bash
cd backend
python train_orbit.py \
    --data-dir data/audio \
    --epochs 100 \
    --use-curriculum
```

## Common Configurations

### Fast Test (1 epoch, CPU)
```bash
python train_orbit.py \
    --data-dir data/audio \
    --epochs 1 \
    --batch-size 8 \
    --no-gpu-rendering \
    --julia-resolution 32 \
    --num-workers 0
```

### Full Training (GPU-accelerated)
```bash
python train_orbit.py \
    --data-dir data/audio \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 5e-4 \
    --use-curriculum \
    --curriculum-weight 1.0 \
    --curriculum-decay 0.50 \
    --k-bands 6 \
    --julia-resolution 64 \
    --julia-max-iter 50 \
    --num-workers 4 \
    --device cuda
```

### Extended Training (More residual bands)
```bash
python train_orbit.py \
    --data-dir data/audio \
    --epochs 200 \
    --k-bands 8 \
    --use-curriculum
```

## Parameters Explained

### Architecture
- `--window-frames`: Audio context window (default: 10 frames)
- `--k-bands`: Number of residual epicycles (default: 6)

### Training
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Adam learning rate (default: 5e-4)

### Curriculum Learning
- `--use-curriculum`: Enable curriculum learning from Mandelbrot orbits
- `--curriculum-weight`: Initial weight for curriculum loss (default: 1.0)
- `--curriculum-decay`: Decay factor per epoch (default: 0.50)

### Performance
- `--device`: Training device (cuda/cpu)
- `--no-gpu-rendering`: Disable GPU Julia rendering (use CPU)
- `--julia-resolution`: Julia set resolution (default: 64x64)
- `--julia-max-iter`: Julia iterations (default: 50)
- `--num-workers`: DataLoader parallel workers (default: 4)

### Output
- `--save-dir`: Checkpoint directory (default: checkpoints_orbit)

## Output Files

After training, you'll find:

```
checkpoints_orbit/
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_20.pt
├── ...
├── checkpoint_epoch_100.pt
├── model_orbit_control_TIMESTAMP.onnx
└── training_history.json
```

## Using the Trained Model

### Python Inference
```python
import torch
from src.control_model import AudioToControlModel
from src.orbit_synth import OrbitSynthesizer, OrbitState

# Load model
model = AudioToControlModel(window_frames=10, k_bands=6)
checkpoint = torch.load('checkpoints_orbit/checkpoint_epoch_100.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare audio features (shape: [1, 60])
audio_features = torch.randn(1, 60)

# Get control signals
controls = model(audio_features)
parsed = model.parse_output(controls)

# Synthesize c(t)
synthesizer = OrbitSynthesizer(k_residuals=6)
state = OrbitState(
    lobe=1,
    sub_lobe=0,
    theta=0.0,
    omega=float(parsed['omega_scale'][0]),
    s=float(parsed['s_target'][0]),
    alpha=float(parsed['alpha'][0]),
    residual_phases=np.random.uniform(0, 2*np.pi, 6),
    residual_omegas=np.ones(6),
)

c = synthesizer.synthesize(state, band_gates=parsed['band_gates'][0].detach().numpy())
print(f"Julia parameter c: {c}")
```

### ONNX Inference (Frontend)
The exported ONNX model can be loaded in the frontend:

```javascript
// Load ONNX model
const session = await ort.InferenceSession.create('model_orbit_control.onnx');

// Run inference
const feeds = { input: audioFeaturesTensor };
const results = await session.run(feeds);

// Parse control signals
const controls = results.output.data;
const s_target = controls[0];
const alpha = controls[1];
const omega_scale = controls[2];
const band_gates = controls.slice(3, 9);

// Synthesize c(t) using JavaScript orbit synthesizer
const c = orbitSynthesizer.synthesize({
    lobe: 1,
    sub_lobe: 0,
    theta: currentTheta,
    omega: omega_scale,
    s: s_target,
    alpha: alpha,
    residual_phases: residualPhases,
    residual_omegas: residualOmegas,
}, band_gates);
```

## Monitoring Training

Watch training progress:
```bash
tail -f checkpoints_orbit/training_history.json
```

Key metrics:
- `loss`: Total loss (lower is better)
- `control_loss`: Curriculum learning loss
- `timbre_color_loss`: Audio-color correlation
- `transient_impact_loss`: Audio-motion correlation

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (try 16 or 8)
- Reduce `--julia-resolution` (try 32)
- Reduce `--num-workers` (try 0)

### Slow Training
- Enable GPU rendering: remove `--no-gpu-rendering`
- Increase `--num-workers` (try 8)
- Reduce `--julia-max-iter` (try 30)

### Poor Convergence
- Increase `--curriculum-weight` (try 2.0)
- Decrease `--curriculum-decay` (try 0.70 for slower decay)
- Increase `--epochs` (try 200)

## Comparison with Old Training

**Orbit-based (new):**
```bash
python train_orbit.py --data-dir data/audio --epochs 100
```
- Predicts control signals (s, alpha, omega, gates)
- Deterministic synthesis via OrbitSynthesizer
- Explicit lobe-orbit geometry

**Velocity-based (old):**
```bash
python train.py --data-dir data/audio --epochs 100
```
- Predicts velocity and integrates to position
- Free-form exploration with physics constraints
- Implicit geometry via loss functions

Both are valid approaches. Orbit-based is recommended for live performance.
