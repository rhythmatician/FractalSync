# Advanced Loss Functions for Emotional Coherence and Variety

This document describes the advanced loss components introduced to improve emotional coherence between music and visuals, and to encourage variety in the model's exploration during training.

## Overview

The enhanced loss function addresses two key objectives:

1. **Emotional Coherence**: Ensuring that the visuals match the emotional content of the music
2. **Variety/Exploration**: Encouraging the model to explore different regions of the parameter space (different lobes) instead of staying in one small neighborhood

## Loss Components

### 1. Membership Proximity Loss

**Purpose**: Encourages visually beautiful Julia sets during high-intensity audio moments.

**How it works**: 
- Computes a membership proxy for the Mandelbrot set using escape-time iteration
- **NEW (Boundary Mode)**: During exciting music, encourages c values **near the Mandelbrot boundary** (membership ~0.9-0.99), not deep inside, where the most intricate, beautiful Julia sets occur
- During calm moments, uses a minimum membership threshold to avoid sparse sets
- Loss is weighted by audio intensity (RMS energy)

**Why boundary proximity matters**:
- **Near boundary** (membership ~0.9-0.99): Complex, intricate, "lacy" or "spiral" patterns - the most visually beautiful Julia sets
- **Deep inside M** (membership = 1.0): Simpler, less interesting patterns
- **Outside M** (membership < 0.5): Sparse, disconnected "Cantor dust"

This is based on the principle that the most beautiful Julia sets come from c values on or very close to the Mandelbrot boundary, including Misiurewicz points (pre-periodic points on the boundary).

**Configuration**:
```python
MembershipProximityLoss(
    target_membership=0.75,      # Min membership (avoid sparse sets)
    boundary_membership=0.95,    # Target boundary proximity [0-1]
    boundary_width=0.1,          # Allowed width around boundary
    max_iter=50,                 # Max iterations for escape-time
    escape_radius=2.0,           # Escape radius threshold
    weight=0.5,                  # Default weight
    use_boundary_mode=True,      # Enable boundary targeting
)
```

**Training weight**: `correlation_weights['membership_proximity']` (default: 0.5)

**Key insight**: During exciting music moments, the loss encourages c values in the boundary region where membership ≈ 0.95 ± 0.1, producing the most visually stunning, intricate Julia set patterns.

### 2. Edge Density Correlation Loss

**Purpose**: Correlates visual detail (jaggedness) with audio spectral brightness.

**How it works**:
- Encourages jagged, detailed fractals during bright/high-frequency audio
- Encourages smooth, blobby fractals during dark/low-frequency audio
- Uses negative correlation loss (maximizes positive correlation)

**Configuration**:
```python
EdgeDensityCorrelationLoss(
    weight=0.3,  # Default weight
)
```

**Training weight**: `correlation_weights['edge_density_correlation']` (default: 0.3)

**Key insight**: From EMOTIONAL_COHERENCE.md:
> "Map it to audio: correlate high-frequency spectral flux / centroid with edge density."

### 3. Lobe Variety Loss

**Purpose**: Encourages exploration of different lobes across the song.

**How it works**:
- Maintains a history buffer of recent c values (default: 100 samples)
- Computes variety score based on standard deviation in c-space
- Penalizes when variety falls below a target threshold
- Target is based on expected number of clusters (lobes)

**Configuration**:
```python
LobeVarietyLoss(
    history_size=100,  # Number of recent c values to track
    n_clusters=5,      # Expected number of different regions
    weight=0.2,        # Default weight
)
```

**Training weight**: `correlation_weights['lobe_variety']` (default: 0.2)

**Key insight**: Addresses the user request:
> "Lets also incentivize variety over the course of a whole song... we just want to incentivize exploration and penalize hanging out in a small neighborhood the whole time."

### 4. Neighborhood Penalty Loss

**Purpose**: Penalizes staying in a small neighborhood of c-space for extended periods.

**How it works**:
- Tracks recent c values within a sliding window (default: 32 frames)
- Computes the radius of recent values from their centroid
- Penalizes when radius is below a minimum threshold
- Encourages movement and exploration

**Configuration**:
```python
NeighborhoodPenaltyLoss(
    window_size=32,     # Number of recent frames
    min_radius=0.1,     # Minimum acceptable radius
    weight=0.1,         # Default weight
)
```

**Training weight**: `correlation_weights['neighborhood_penalty']` (default: 0.1)

**Key insight**: Complements lobe variety by operating on a shorter timescale (recent frames vs. song-wide history).

## Integration with ControlTrainer

The new losses are integrated into the training loop as follows:

1. **Compute visual metrics**: Edge density is computed from rendered Julia sets
2. **Compute audio features**: RMS energy, spectral centroid, etc.
3. **Compute advanced losses**:
   - Membership proximity: Uses c values, audio intensity
   - Edge density correlation: Uses edge density, spectral centroid
   - Lobe variety: Uses c values, long-term history
   - Neighborhood penalty: Uses c values, short-term window
4. **Combine losses**: All losses are added to the total loss with their respective weights

## Default Weights

The default weights are chosen to balance the different objectives:

```python
correlation_weights = {
    "timbre_color": 1.0,                  # Existing correlation
    "transient_impact": 1.0,              # Existing correlation
    "control_loss": 1.0,                  # Curriculum learning
    "membership_proximity": 0.5,          # NEW: Prevent sparse visuals
    "edge_density_correlation": 0.3,      # NEW: Visual detail ↔ brightness
    "lobe_variety": 0.2,                  # NEW: Long-term exploration
    "neighborhood_penalty": 0.1,          # NEW: Short-term movement
}
```

These weights can be customized when initializing the trainer:

```python
trainer = ControlTrainer(
    model=model,
    visual_metrics=visual_metrics,
    correlation_weights={
        "membership_proximity": 1.0,  # Increase emphasis
        "lobe_variety": 0.5,          # Increase exploration incentive
    },
    ...
)
```

## Training History

The training history now tracks all loss components:

```python
history = {
    "loss": [],                              # Total loss
    "control_loss": [],                      # Curriculum learning loss
    "timbre_color_loss": [],                 # Timbre-color correlation
    "transient_impact_loss": [],             # Transient-impact correlation
    "membership_proximity_loss": [],          # NEW: Membership proximity
    "edge_density_correlation_loss": [],      # NEW: Edge density correlation
    "lobe_variety_loss": [],                  # NEW: Lobe variety
    "neighborhood_penalty_loss": [],          # NEW: Neighborhood penalty
}
```

This allows monitoring each component's contribution over training epochs.

## Expected Benefits

1. **Better emotional coherence**:
   - Visuals stay interesting during intense moments (no sparse fractals)
   - Visual detail tracks audio brightness naturally
   
2. **More variety**:
   - Model explores different lobes throughout the song
   - Avoids getting stuck in one region
   - Transitions feel more dynamic and engaging

3. **Natural transitions**:
   - Model learns to switch lobes at appropriate moments
   - No hard-coded transition points
   - Exploration is learned from the data

## Monitoring Training

When training, watch for:

1. **Membership proximity loss**: Should decrease as model learns to stay near M during intense audio
2. **Edge density correlation**: Should become more negative (stronger positive correlation)
3. **Lobe variety loss**: Should decrease as model explores more regions
4. **Neighborhood penalty loss**: Should decrease as model moves more actively

## Future Enhancements

Potential improvements based on EMOTIONAL_COHERENCE.md:

1. **Interest field**: Precompute a field over c-space indicating visual interest
2. **Chord-aware selection**: Map musical harmony to lobe characteristics
3. **Section-aware objectives**: Different loss weights for verse/chorus/breakdown
4. **Disconnected Julia sets**: Allow controlled use of points outside M for special effects

## References

- `backend/src/loss_components.py` - Implementation of all loss functions
- `backend/src/control_trainer.py` - Integration with training loop
- `backend/docs/EMOTIONAL_COHERENCE.md` - Design philosophy and insights
- `backend/docs/LOBE_TRANSMISSION.md` - Mathematical foundations of lobes
- `backend/tests/test_loss_components.py` - Unit tests

---

*Document Version: 2026-01-22*
*Based on requirements for emotional coherence and variety in music visualization*
