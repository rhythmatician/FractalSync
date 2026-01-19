# FractalSync: New Goals + New Architecture (Implementation Plan)

## 0) What we’re building

A live Julia-set visualizer that feels “musically intentional” by separating motion into two independent behaviors:

1. **Impact events (fast, frequent):**
   Triggered by big hits/transients. These cause short “crash-through-boundary” effects such as `s` crossing `1`, temporary residual boost, small speed kick — **many times per song**.

2. **Section changes (slow, occasional):**
   Triggered by sustained “novelty” changes in the audio (verse→chorus-ish). These cause **lobe switches** (even if detected late). The system must prevent getting stuck in one lobe.

This means we do **not** require perfect section detection timing. We only require:

* *some* section changes detected over an 8–13 minute song
* lobe changes happen occasionally and don’t thrash
* impacts happen frequently and feel tight to hits

---

## 1) Core design principle: Learned controller over deterministic generator

Even if you have an ONNX model already, the stable plan is:

* The fractal/orbit engine is deterministic and constrained.
* The “brains” (model or heuristics) output a **low-dimensional control vector**.
* The generator maps controls → smooth orbit state → `c(t)`.

This prevents the model from “cheating” by wandering into boring/sparse parameter space.

### Two time scales

* **Fast loop** (render rate): 60–120 Hz
  Integrates orbit angle, smooths parameters, applies impact envelope, produces `c(t)`.

* **Control loop** (decision rate): 5–20 Hz
  Computes audio features summaries, novelty score, boundary probability, impact intent, lobe-switch decisions.

---

## 2) Primary output we drive

At every render tick, we output:

* Julia parameter `c(t)` (complex)
* optional debug HUD strings (current lobe, s, novelty, last impact, etc.)

The renderer already uses GPU, so it should accept `c(t)` and update uniforms.

---

## 3) Orbit generator: carrier + residual

This is the motion model.

### 3.1 Carrier orbit (the “meaningful” motion)

We orbit a selected lobe (including cardioid). Use existing tested geometry functions (already in your codebase after PR 15).

Define:

* current `lobe_id`, `sub_lobe`
* `theta(t)` angle on orbit
* `omega(t)` angular velocity
* `s(t)` radial scaling (inside/on/outside; `s=1` is boundary)

Carrier:

* `c_carrier = lobe_point_at_angle(lobe_id, theta, s, sub_lobe)`

**Important:** Use the validated centers/radii definitions already in code. No new math.

### 3.2 Residual “texture” (small epicycles)

Add a bounded residual to make motion responsive to timbre and rhythm without destroying orbit coherence.

Residual:
[
\Delta c(t) = \sum_{k=1}^{K} \left(\alpha(t)\cdot \frac{s(t)\cdot R_l}{k^2}\right) \cdot g_k(t)\cdot e^{i\phi_k(t)}
]

Where:

* `R_l` = current lobe radius (for cardioid pick a consistent reference scale; use existing code’s radius notion or constant)
* `K` = number of residual circles (start small: 4–8)
* `g_k(t)` = band-gating in [0,1] (low k = low bands, high k = high bands)
* `phi_k(t)` integrates from `omega_k(t)` (often tied to tempo multiples)

Final:

* `c(t) = c_carrier + Δc`

### 3.3 Hard safety constraint (must have)

Residual must never overpower carrier:

* enforce `|Δc| <= residual_cap * R_l` (soft clamp)
  Example: `residual_cap = 0.5` early; tune later.

---

## 4) Impact events: many per song, tight to hits

### 4.1 Goal

On big transients, create a short “impact envelope” (200–500ms) that:

* causes `s` to cross `1` (briefly)
* boosts residual strength α briefly
* optionally applies a small `omega` kick or phase kick

This yields “boundary crashes” aligned to big hits.

### 4.2 Impact detector input features (fast)

From stereo mix:

* onset strength (spectral flux)
* transient energy (high-band energy deltas)
* optionally low-band “kick” delta

Compute at 100 Hz:

* `impact_score(t)` = weighted combination of these

Use:

* adaptive threshold + hysteresis + refractory window
  So you get repeatable triggers and no machine-gun spam.

### 4.3 Impact envelope

When triggered, start an envelope `e(t)`:

* attack 20–50ms
* decay 150–450ms

During `e(t)`:

* push `s_target` across boundary with controlled overshoot:

  * if `s_base < 1`: push outward to `s_impulse_hi` (e.g. 1.08–1.25)
  * if `s_base > 1`: either punch inward then rebound, or overshoot outward (choose one consistent style)
* increase residual α by `α_boost`
* small `omega` boost allowed (optional; keep subtle)

**Important:** Impact is independent of lobe switching. You want many impacts even if section detector is imperfect.

---

## 5) Section change detector: automatic, late is OK

### 5.1 Goal

Detect sustained changes in the audio “state” that likely correspond to section boundaries, but not necessarily precisely timed.

We don’t need perfect detection; we need enough to cause occasional lobe changes.

### 5.2 Novelty-based detector (robust baseline)

At control rate (e.g. 10 Hz):

1. Build a feature vector `F(t)` over ~0.5s hop:

   * band energies (log mel or simple octave bands)
   * spectral centroid / rolloff
   * tonalness vs noisiness
   * onset rate / flux summary
   * optional beat confidence

2. Maintain a rolling baseline `B(t)` over the last N seconds (8–20s).

3. Compute novelty:

* `novelty(t) = distance(F(t), B(t))`
  Use cosine distance or normalized L2.

4. Apply smoothing and persistence:

* require novelty above threshold for 1–2 seconds
* enforce cooldown and minimum dwell time between lobe switches

### 5.3 Lobe switching policy

When a boundary is detected:

* choose a new lobe from a curated set of “good” lobes (main cardioid + biggest bulbs you’ve mapped)
* avoid repetition:

  * don’t pick same lobe as current
  * avoid ping-pong between 2 lobes (keep short history, penalize recently used)

Then transition:

* 2–6 seconds smooth transition (late is fine)
* during transition allow additional impacts (impacts should still function)

**Hard rules**

* minimum dwell: e.g. 20–60 seconds in a lobe
* cooldown after switch: e.g. 10–20 seconds
* switch should be rare-ish compared to impacts

---

## 6) `s(t)` behavior (critical)

We want `s` to encode musical “state” while remaining controlled:

* **Loud/energetic:** `s → 1.02` (near boundary, slightly outside)
* **Quiet + noisy/chaotic:** `s` grows large (push toward |c|≈2-ish behavior)
* **Quiet + tonal/pure:** `s` settles inward (toward 0; for cardioid this approaches “tonic” feel)

### 6.1 Implement as a deterministic target + smoothing

Compute at control rate:

* loudness `L(t)` in [0,1]
* tonalness `T(t)` in [0,1] (high = pure tone/harmonic)
* noisiness `N(t)` in [0,1] (high = broadband/noisy)
* these can be crude; don’t overfit early

Define:

* `s_loud = 1.02`
* `s_quiet_noisy = s_max` (e.g. 2.0–4.0 depending on how you map |c|)
* `s_quiet_tonal = s_min` (e.g. 0.0–0.4; don’t hit exactly 0 too aggressively)

Then:

* `s_target = mix(...)` using (L,T,N) with weights you can tune.

Finally:

* smooth `s` with a rate limiter and low-pass (avoid jitter)
* impact envelope temporarily overrides / biases `s_target`

### 6.2 Loss / training intent (if learning later)

Even in heuristic mode, encode the intention:

* penalize loudness with `s` far from 1.02
* encourage `s` bigger when quiet+noisy
* encourage `s` smaller when quiet+tonal

This can be implemented as explicit loss terms if training continues.

---

## 7) ONNX model role (optional but future-proof)

You can ship the whole system using heuristics now, then later have the model learn to output controls.

### 7.1 What the model should output (small control vector)

At 5–20 Hz, output a vector like:

* `s_delta` or `s_target`
* `alpha_residual`
* `impact_intent` (optional; can remain heuristic)
* `novelty_bias` or boundary probability (optional)
* `lobe_next_logits` (optional)
* `omega_base` (small variation)
* residual tempo multipliers (soft-snapped)

### 7.2 What the model should NOT output

* It should not output raw `c(t)` directly at 60fps.
* It should not output free amplitudes for each residual circle (too many degrees of freedom).
* It should not control lobe geometry.

The deterministic generator owns those.

---

## 8) Fallback behavior (required for live)

We need a “safe mode” that always works.

Implement a “controller interface” with two implementations:

* `HeuristicController` (always available; deterministic)
* `OnnxController` (optional; can fail gracefully)

If ONNX fails / latency spikes / bad output:

* seamlessly fall back to heuristic controller outputs
* keep orbit running without discontinuities

---

## 9) Concrete modules to implement in FractalSync

(Names are suggestions; fit them to the existing code structure in the PR.)

### 9.1 `AudioFeatureStream`

Responsibilities:

* ingest stereo audio frames
* produce:

  * fast features at ~100Hz (onset, flux, band deltas)
  * slow summaries at ~10Hz (loudness, tonalness, band energies)

### 9.2 `ImpactDetector`

Inputs:

* fast features (flux/onset/band deltas)

Outputs:

* `impact_trigger` events
* `impact_score` for debug

### 9.3 `ImpactEnvelope`

State:

* current envelope value `e(t)` and phase
* refractory timer

Outputs each render frame:

* `e(t)` scalar

### 9.4 `NoveltyBoundaryDetector`

Inputs:

* slow feature vectors F(t)

State:

* rolling baseline buffer over last 8–20s
* smoothed novelty, persistence counter
* last switch time / cooldown / min dwell

Outputs:

* `boundary_detected` boolean event (can be late)
* `novelty_score` for debug

### 9.5 `LobeScheduler`

Inputs:

* boundary events
* maybe energy level
* lobe history

Outputs:

* next lobe selection + transition start

### 9.6 `OrbitStateMachine`

Holds current orbit state:

* current lobe + sub lobe
* theta, omega
* s (smoothed), s_target
* transition state (start/end lobe, progress)
* residual phases phi_k

Applies:

* slow lobe transitions
* impact envelope overrides
* residual synthesis

Outputs per render tick:

* `c(t)`
* debug info

---

## 10) Implementation sequencing (do this in order)

### Phase 1 — Make impacts amazing (fast win)

1. Compute onset/flux and implement `ImpactDetector`
2. Add `ImpactEnvelope`
3. Modify orbit generator so impacts force **brief `s` boundary crossing** + residual boost
4. Add HUD/logging so you can see: impact score, triggers, current s, s_target

**Success criterion:** you can hear drum hits and immediately see consistent “crashes” in visuals.

### Phase 2 — Add novelty boundary detection + lobe switching

1. Implement `NoveltyBoundaryDetector` with sustained novelty + dwell/cooldown
2. Implement `LobeScheduler`
3. Add transition logic between lobes (2–6 sec)
4. Ensure impacts continue to work during transitions

**Success criterion:** on a long song, you get a few lobe changes, not thrashing, not stuck.

### Phase 3 — Harden for live reliability

1. Add fallback controller interface
2. Add performance guardrails (max compute per tick)
3. Add debug event logs (why switched, why impacted)

### Phase 4 — Re-introduce ONNX model as controller (optional)

Start by having ONNX output only:

* corrections to `s_target`
* α residual
* maybe novelty bias
  Keep lobe switching and impacts partially heuristic until ONNX proves reliable.

---

## 11) Debuggability requirements (non-negotiable)

Every lobe switch event should log:

* timestamp
* novelty score
* loudness level
* chosen lobe/sub-lobe
* dwell time in previous lobe

Every impact event should log:

* timestamp
* impact score
* resulting s overshoot

Include a “debug overlay” string provider so you can see these live.

---

## 12) Parameter defaults to start with (tunable)

* Control rate: 10 Hz
* Feature rate: 100 Hz
* Render: 60 Hz

Impact:

* threshold: adaptive (e.g. percentile of last 10s)
* refractory: 150 ms
* envelope: attack 30ms, decay 300ms
* `s_impulse_hi`: 1.12 (start)
* `α_boost`: +0.3 (start)
* residual cap: 0.5 * R_l

Novelty:

* baseline window: 12s
* boundary persistence: 1.0–2.0s above threshold
* min dwell: 30s
* cooldown: 15s

Residual:

* K = 6
* amplitude law: `s*R_l/k^2`
* band gating: map low bands to low k, high bands to high k (simple)

`s`:

* loud target: 1.02
* quiet+noisy target: 2.0–3.0 (start conservative)
* quiet+tonal target: 0.2–0.6 (don’t slam to 0 instantly)
* smoothing time constant: ~0.5–2s plus rate limiting

---

## 13) What Copilot should do now

Given the “latest PR is ready,” implement **Phase 1 + Phase 2** on top of it:

1. Add / extend audio feature extraction so we have:

   * onset/flux (fast)
   * loudness and simple tonalness/noisiness (slow)
2. Implement ImpactDetector + ImpactEnvelope
3. Wire impact envelope into orbit generator:

   * force `s` crossing `1`
   * boost residual α briefly
4. Implement novelty boundary detector
5. Implement lobe scheduler + transition logic
6. Add logging + debug overlay strings

**Do not** change validated lobe geometry math. Use the existing tested centers/radii functions already in the codebase.
