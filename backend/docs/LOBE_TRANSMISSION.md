# Lobe Transmission System: Mathematical Foundations and Implementation

## Overview

The **Lobe Transmission System** is a strategic framework for selecting Mandelbrot lobes based on musical structure and emotional intent. It treats each lobe as a "gear" in a transmission, with smooth transitions guided by music theory and Julia set mathematics.

---

## Mathematical Foundations

### 1. The Fundamental Dichotomy Theorem (Julia-Fatou, 1919)

**Theorem:** For any polynomial f(z) = z² + c, the Julia set J is either:
- **Path-connected** (a single connected piece), OR
- **Totally disconnected** (Cantor dust - infinitely many isolated points)

**Implication for FractalSync:**
- Points **inside** the Mandelbrot set → **connected Julia sets** (smooth, whole, visually stable)
- Points **outside** the Mandelbrot set → **Cantor dust** (chaotic, disconnected, sparse)

**Our Strategy:**
- Use connected Julia sets (interior points) for stable musical sections
- Reserve disconnected/near-boundary regions for breakdowns and tension

---

### 2. Lobe Order and Periodicity

From Linton's Mandelbrot Map analysis:

**First Mandelbrot Theorem:**
> The order of any lobe in a sequence is the sum of the orders of the previous lobe plus the order of the lobe immediately beyond the limit.

**Formula:** Order(Lobe_n) = A + nB
- A = order of starting lobe
- B = order of terminus lobe
- n = position in sequence

**Examples:**
- Lobe 1 (Cardioid): Order = 1, period-1 fixed point
- Lobe 2: Order = 2, period-2 alternating orbit
- Lobe 3: Order = 3, period-3 triangular orbit
- Lobe 4›3: Order = 7 (4 + 3), period-7 complex orbit
- Lobe 3›2²: Order = 7 (3 + 2 + 2), period-7 with different step pattern

---

### 3. Step Size and Orbit Tracing

**Second Mandelbrot Theorem:**
> The step size of any lobe follows the same rule as order, determining how the orbit steps through periodic points.

**Step Size Properties:**
- **Step size = 1**: Sequential tracing (1→2→3→...)
- **Step size = 2**: Every-other tracing (1→3→5→2→4→...)
- **Step size = k**: Skip k-1 points between steps

**Visual Effect:**
- Step size 1: Regular polygon
- Step size 2+: Star patterns, more intricate

**Examples:**
- Lobe 7: Period-7, step size 1 → irregular heptagon
- Lobe 4›3: Period-7, step size 2 → 7-pointed star (skips 1)
- Lobe 3›2²: Period-7, step size 3 → 7-pointed star (skips 2)

---

## Implementation: The Transmission System

### Gear Metaphor

Each lobe is assigned a "gear" (1-5) based on complexity and energy:

| Gear | Lobes | Julia Properties | Musical Context |
|------|-------|------------------|-----------------|
| 1 | Cardioid (1,0) | Smooth, connected, warm | Verse, intro, resolution |
| 2 | Period-2 (2,0) | Angular, 2-fold symmetry | Pre-chorus, building |
| 3 | Period-3/4 (3,0/3,1/4,0) | Dramatic, multi-fold | Chorus, drop |
| 4 | Higher periods (4,1/4,2/3,2) | Complex, star patterns | Climax, intense moments |
| 5 | Period-8+ (8,0) | Maximum intricacy | Breakdown, chaos |

### Transmission Rules

1. **Prefer adjacent gear shifts** (Gear 2→3 over 2→5)
2. **Match energy affinity** (quiet→Gear 1, loud→Gear 3-4)
3. **Section type bonuses**:
   - Verse → Strongly favor Gear 1 (Cardioid)
   - Build → Increment gears gradually
   - Chorus → Stay in Gear 3-4 for drama
   - Breakdown → Allow Gear 4-5 for complexity
4. **History avoidance** (don't ping-pong between same lobes)
5. **Special override**: Always allow returning to Cardioid during quiet verses

### Transition Duration

Smooth visual transitions based on gear difference:
```python
duration = 2.0 + gear_difference * 1.0  # 2-5 seconds
```

Larger gear jumps need longer transitions to avoid jarring changes.

---

## Lobe Characteristics Database

Each lobe in `LOBE_CHARACTERISTICS` includes:

```python
@dataclass
class LobeCharacteristics:
    # Identity
    linton_label: str        # Linton notation (e.g., "4›3")
    
    # Mathematical properties
    period: int              # Orbit periodicity
    step_size: int           # How orbit traces points
    is_connected: bool       # Connected vs Cantor dust
    
    # Visual characteristics
    symmetry: int            # n-fold rotational symmetry
    smoothness: float        # 0=jagged, 1=smooth
    complexity: float        # 0=simple, 1=intricate
    warmth: float           # 0=cold/angular, 1=warm/rounded
    
    # Emotional mapping
    tension: float          # 0=resolved, 1=tense
    energy_affinity: float  # Preferred energy level
    
    # Transmission
    gear: int               # Transmission gear (1-5)
```

---

## Integration with Section Detection

The system uses detected section boundaries (via `SongAnalyzer` with ruptures library) to trigger gear shifts:

```python
# Section boundary detected
if boundary_detected:
    # Get section characteristics
    energy = compute_energy_level(audio_features)
    novelty = boundary_detector.get_novelty()
    
    # Select next lobe based on context
    new_lobe = scheduler.select_next_lobe(
        energy_level=energy,
        novelty=novelty,
        timestamp=current_time,
        section_type="chorus"  # or "verse", "build", "breakdown"
    )
    
    # Compute smooth transition duration
    transition_dur = scheduler.suggest_transition_duration(
        from_lobe=current_lobe,
        to_lobe=new_lobe
    )
```

---

## Future Enhancements

### 1. Linton Label Expansion
Add more lobes using Linton notation:
- `5›2` → Period-7, different step pattern
- `2›3›4` → Deep sequence lobe
- `(4›3)²` → Nested sequences

### 2. Disconnected Julia Sets
For extreme moments, use c values outside M:
- Cantor dust for total breakdown
- Controlled chaos for special effects

### 3. Visual Interest Precomputation
Build an "interest field" over c-space:
- Edge density
- Fractal dimension
- Symmetry breaking metrics

Use to guide lobe selection during climaxes.

### 4. Chord-Aware Lobe Selection
Map musical harmony to lobe characteristics:
- Major → warm lobes (Cardioid, low periods)
- Minor → cooler lobes (Period-3, angular)
- Dissonance → higher periods, near boundaries

---

## References

1. **Julia Sets and the Mandelbrot Set** (Mathematical Notes)
   - Fundamental Dichotomy Theorem
   - Connectedness properties
   - Basin of infinity

2. **The Mandelbrot Map** (J Oliver Linton, 2017)
   - Linton labeling system
   - Order and step size theorems
   - Lobe sequence rules

3. **IEEE TASLP 2006 Goto et al.** (Music Structure Segmentation)
   - Section boundary detection
   - Novelty-based segmentation
   - Temporal feature analysis

---

## Code Files

- `backend/src/live_controller.py` - LobeScheduler, LobeCharacteristics
- `backend/src/song_analyzer.py` - Section boundary detection with ruptures
- `runtime-core/src/geometry.rs` - Lobe point computation
- `frontend/src/lib/orbitSynthesizer.ts` - Client-side orbit synthesis

---

*Document Version: 2026-01-22*
*Based on mathematical foundations from Julia set theory and Mandelbrot map analysis*
