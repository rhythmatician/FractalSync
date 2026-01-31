# FractalSync — Rhythm/Structure + Control Roadmap (Session Notes)

Date: 2026-01-31  
Source-of-truth decision: **Rust (`runtime-core`) owns audio timing / clocks and emits state; frontend consumes state.**

---

## Executive summary

We agreed to build a **two-timescale system**:

- **Slow clock (audio-hop clock, Rust):** owns rhythm + structure state (BeatNet + section/anticipation model outputs).  
  - Runs at hop cadence (~46.875 Hz for 48 kHz / 1024 hop) or a decimated cadence (e.g., 12–23 Hz).
  - Emits a **held** `SlowState` that the frontend samples at 60 FPS.

- **Fast clock (render clock, frontend):** drives high-frequency Julia micro-reactivity.  
  - Runs at ~60 FPS.
  - Reads the latest `SlowState` (and optionally advances beat phase predictively between slow updates).
  - Runs the fast controller model (or uses already-produced fast control signals) to respond to musical nuance faster than beat-level events.

Key reason for the split: we want **fast nuance reactivity** without needing beat/structure inference at 60 FPS.

---

## Big decisions (locked)

### 1) Beat tracking strategy
- **Use BeatNet** (target long-term) as the beat/downbeat/tempo tracker.
- **Run BeatNet online** for runtime behavior, and use the same BeatNet-derived signals in training so we maintain **parity** between training and real use.
- Latency is not a dealbreaker because visuals can use a **predictive oscillator** (phase + seconds-per-beat) between updates.

### 2) Section/structure detection strategy
- Goal is **real-time section detection / anticipation**.
- Use **offline structure analyzers** to annotate training songs (teacher labels) when needed.
- **SongFormer** selected as the most promising offline teacher at first glance.
- Keep **pre-chorus** as a first-class label because it enables section-aware behavior and loss terms (e.g., aesthetic constraints that depend on section).

### 3) Model architecture
We moved from a single hybrid model toward a **two-model design** because we want different polling rates:

- **Anticipator model (slow-rate):** predicts song structure state + boundary hazard.
- **Controller model (fast-rate):** uses audio micro-features + BeatNet clock + anticipator outputs as conditioning to produce control signals for Julia rendering.

Anticipator outputs flow **straight into the controller** as part of its input vector (plus other inputs).

---

## Clocks & scheduling

### Slow clock (Rust, hop cadence)
- Canonical sample rate: **48 kHz**
- Hop length: **1024 samples**
- Hop cadence: `48000 / 1024 ≈ 46.875 Hz`

Responsibilities:
- Maintain and emit `BeatClockState` (BeatNet later; placeholder oscillator immediately).
- Run the **anticipator** model at hop cadence or decimated cadence (e.g. every 2–4 hops → ~12–23 Hz).
- Emit `SlowState` frames for frontend consumption.

### Fast clock (Frontend, render cadence)
- Render cadence: ~60 FPS (`requestAnimationFrame`).
Responsibilities:
- Read/hold last `SlowState`.
- Optionally **advance beat phase predictively** between slow updates using `spb` and `dt`.
- Run fast controller at render cadence (or at hop cadence with interpolation if desired).
- Apply state smoothing/hysteresis on slow signals to prevent flicker.

---

## Runtime interface: state bus

We plan to define a stable state bus from Rust → frontend.

### Beat clock state (concept)
Fields we discussed (Rust struct):
- `t_sec`: stream time for the hop/frame
- `spb`: seconds-per-beat (tempo proxy)
- `phase`: beat phase in `[0,1)`
- `beat_count`: monotonically increasing beat index (parity anchor)
- `conf`: confidence/stability measure
- optional `downbeat_prob` if we decide to include it

### Structure/anticipation state
- `section_probs` or `section_logits` (we will standardize)
- `hazard_probs` for multiple horizons (see below)

### Combined slow state
- `SlowState = { beat: BeatClockState, structure: StructureState, features?: Vec<f32> }`

Frontend should treat `SlowState` as source-of-truth, held between updates.

---

## Anticipator model contract (v1)

### Label set
We decided to use the **same label set as SongFormer**, keeping **pre-chorus**.

SongFormer’s labels (as referenced in discussion):
- `intro, verse, chorus, bridge, inst, outro, silence, pre-chorus`

We will freeze a stable ordering in a `model_contract.json` file (see below).  
Proposed ordering for v1:

```text
labels_v1 = [
  "silence",
  "intro",
  "verse",
  "pre-chorus",
  "chorus",
  "bridge",
  "inst",
  "outro"
]
L = 8
```

### Hazard horizons
We chose beat-relative horizons:

```text
horizons_beats_v1 = [1, 2, 4, 8, 16]
K = 5
```

Semantics:
- `hazard[k]` approximates `P(section boundary occurs within next horizons_beats_v1[k] beats)`.

### Inputs (slow-rate)
- Windowed audio features aligned to slow clock: `[T_slow, F_audio]`
- BeatNet-derived beat features aligned to slow clock: `[T_slow, F_beat]`

Beat feature suggestion (time-signature agnostic):
- `sin(2π phase)`, `cos(2π phase)`, `log(spb)`, `conf`  
Optional later: `downbeat_prob`

### Outputs (slow-rate step)
- `section_logits`: shape `[L]` (or `section_probs` after softmax)
- `hazard_logits`: shape `[K]` (or `hazard_probs` after sigmoid)
- optional `macro_embedding`: shape `[E]` (if useful for controller conditioning)

Runtime conversion:
- `section_probs = softmax(section_logits)`
- `hazard_probs  = sigmoid(hazard_logits)`

---

## Controller model contract (v1)

### Inputs (fast-rate)
- Micro audio features: `[T_fast, F_fast]` (short window, tuned for nuance)
- Beat features at current frame time: `[F_beat]` (phase advanced predictively)
- Conditioning from anticipator:
  - `section_probs`: `[L]`
  - `hazard_probs`: `[K]`
  - optional `macro_embedding`: `[E]`
- Optional runtime state features (previous control outputs, previous c, etc.)

### Outputs (fast-rate)
- Your existing control vector (Δc, palette, residual gains, etc.), unchanged except for incorporating new conditioning inputs.

---

## Training pipeline plan

### Source-of-truth features
- We want runtime parity; therefore:
  - **Prefer Rust `runtime-core` features as canonical** for training and inference.
  - Training data should store feature frames on the same hop clock used at runtime (48k/1024 hop).

### Offline teacher labels
- Use SongFormer (offline) to label training tracks:
  - section boundaries `(start,end,label)`
  - section labels including **pre-chorus**
- Convert to per-hop labels `y_section[t]`.

### BeatNet parity for beat-relative targets
- Run BeatNet in a streaming-like mode (same hop settings) on training audio.
- Use BeatNet beat events to define `beat_count` over time.
- For each hop `t`, compute `Δbeats_next[t]` = beats until next SongFormer boundary.

Then define hazard targets for each horizon `h ∈ [1,2,4,8,16]`:

```text
y_hazard[t, h] = 1 if 0 < Δbeats_next[t] <= h else 0
```

### Training anticipator (supervised)
Losses:
- Section classification: cross-entropy over `y_section`.
- Hazard: BCE over `y_hazard` for each horizon.
- Optional temporal smoothing regularizer to reduce flicker.

Export anticipator to ONNX.

### Training controller (supervised / task-driven)
Controller consumes anticipator signals as conditioning.

Robustness training strategy:
- Early: teacher-forced conditioning (from ground-truth/teacher labels).
- Later: feed **anticipator predictions** (realistic noise).
- Add robustness noise:
  - random delay of anticipator state by 0–2 slow steps
  - random dropout on hazard channels
  - mild noise on probabilities

Section-aware penalties:
- Use **soft section probabilities** to weight penalties, e.g.
  - penalize connected Julia sets during choruses:
    - `L += p_chorus * penalty_connected(...)`

Export controller to ONNX.

---

## Frontend vs runtime-core feature extraction (current state & refactor plan)

### Current issue (before refactor)
Frontend feature extraction advances its internal feature buffer **each time it is polled** (currently at rAF ~60 FPS).
This makes “window size” depend on polling frequency and prevents clean slow/fast consumers.

### Refactor requirement
Decouple:
- **Producer:** appends one feature frame at a fixed cadence
- **Consumers:** read windows of frames without advancing the buffer

Frontend will increasingly become a consumer only; Rust will be source-of-truth.

---

## Known bugs/issues (to fix early)

### Feature extraction correctness bug (frontend)
We identified that spectral flux is computed in a way that can update `previousSpectrum` multiple times per frame (flux computed twice), breaking onset/flux features.

This is a **parity breaker** and should be fixed before training data generation if frontend-derived features are used anywhere.

---

## Implementation roadmap (ordered to minimize rework)

### Phase 1 — Two clocks + state bus scaffolding (foundation)
- Implement `SlowClock` in Rust with deterministic hop cadence.
- Define `BeatClockState`, `StructureState`, `SlowState` and make them serde-serializable.
- Provide a “bridge” API for the host layer to call `process_hop(hop)` and serialize `SlowState`.

Frontend:
- Stop computing model inputs in JS as the primary pipeline.
- Consume `SlowState` from Rust and hold last value for render loop.

### Phase 2 — Parity-breaking bug fixes
- Fix feature extraction correctness issues (especially any duplicated flux updates).
- Confirm stability of cadence + shapes.

### Phase 3 — Anticipator training pipeline
- Run SongFormer offline to create labels for a corpus.
- Run BeatNet offline/streaming-mode to create beat-count parity signals.
- Generate per-track artifacts:
  - `features`, `beat_state`, `y_section`, `y_hazard`
- Train anticipator; export ONNX.

### Phase 4 — BeatNet runtime integration
- Export BeatNet CRNN to ONNX.
- Implement inference + particle filter update in Rust.
- Make BeatNet the provider of `BeatClockState` in `SlowClock`.

### Phase 5 — Controller training & runtime wiring
- Train controller conditioned on anticipator outputs.
- Integrate controller model into runtime at fast cadence (60 FPS), while consuming held slow state.

---

## Contract file (recommended)
Create a versioned contract file (e.g. `runtime-core/model_contracts/v1.json`) including:
- sample rate, hop size
- label set (ordering)
- hazard horizons
- beat feature definitions
- tensor names/shapes for anticipator/controller inputs and outputs

This prevents silent breaking changes.

---

## Open items (known, but not blockers for scaffolding)
- Exact feature vector definitions and normalization scheme (we should freeze for v1).
- Whether to include `downbeat_prob` in beat features from day one.
- Exact window sizes `T_slow` and `T_fast` (can be tuned after pipeline is running).
- How/where the host layer moves PCM hops from frontend to Rust (not implemented today; we only scoped the interface).

---

## Links referenced during the session
- SongFormer repo: https://github.com/ASLP-lab/SongFormer  
- All-in-one structure analyzer (alternative teacher): https://github.com/mir-aidj/all-in-one  
- BeatNet repo: https://github.com/mjhydri/BeatNet

