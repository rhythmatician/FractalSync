# 0002 — Predictive musical timing as causal multiscale ridge tracking

Status: Accepted (2026-08-30)

## Context

FractalSync needs to react not only to what the music is doing now, but to
where the music is going next.

The Domain Contract requires the Player to anticipate important musical
Transitions rather than merely react after they occur. Much musical timing is
predictable because rhythmic and structural activity creates approximately
periodic or quasi-periodic processes across several timescales.

Examples include:

- note-level pulse;
- subdivisions;
- beat-scale recurrence;
- bar-scale recurrence;
- riffs and ostinati;
- slower phrase-like cycles;
- simultaneous or polymetric rhythmic layers.

The initial CycleBank design treated this as a finite set of adaptive oscillator
hypotheses initialized from a logarithmic frequency grid and corrected with
PLL-like dynamics.

That was directionally correct, but it chose a discrete tracking formulation too
early.

The more fundamental object is a **continuous multiscale analytic
time-frequency representation**. Periodic structure appears in that
representation as persistent ridges or modes. Those ridges have continuously
varying phase and frequency even though the transform itself must be sampled
numerically at a finite set of scales.

This changes the interpretation of the "bank":

> The bank is not a collection of fixed frequency buckets.

It is a sparse set of salient modes extracted from an underlying continuous-ish
causal time-frequency field.

ADR 0001 establishes `runtime-core` as the authority for deterministic
FractalSync domain behavior shared between runtime and training.

Issue #91 / PR #93 established an authoritative sample-clock pipeline:

```text
native-rate PCM
    ↓
runtime-core AnalysisTimebase
    ↓
canonical 48 kHz timeline
    ↓
exact 1024-sample analysis ticks
    ↓
AnalysisTick {
    features,
    sampleIndex,
    timeSeconds,
    dtSeconds,
    streamEpoch
}
```

The temporal estimator can therefore operate on a trustworthy causal clock
rather than browser render cadence.

This ADR defines the predictive musical-time abstraction built on that clock.

---

## Decision

Musical timing is represented by a **causal multiscale analytic temporal
representation whose salient ridges are exposed as a finite CycleBank**.

Broadly:

```text
causal musical evidence
        ↓
causal analytic multiscale transform
        ↓
continuous-frequency estimation / reassignment
        ↓
ridge / mode extraction and temporal tracking
        ↓
CycleBank
    └── CycleMode[]
```

A `CycleMode` represents one persistent oscillatory mode in continuous
phase-frequency space.

Conceptually:

```text
CycleMode {
    phase
    frequency
    strength
    confidence
    channelSupport
}
```

Optional future state may include:

```text
frequencySlope
frequencyUncertainty
ridgeBandwidth
age
```

Beat, subdivision, bar, phrase, riff cycle, meter, and polymeter are **not**
foundational estimator types.

They are downstream interpretations of the modes and relationships that the
CycleBank discovers.

The architectural boundary remains:

> **DSP estimates and predicts the clocks. The Player decides what those
> clocks mean visually.**

---

## The CycleBank is not a frequency-bin bank

The underlying transform is necessarily evaluated numerically at a finite
number of scales or filter center frequencies.

Those samples are **numerical sampling points**, not musical states.

For example, an implementation might evaluate a logarithmic scale grid around:

```text
1.000
1.059
1.122
1.189
1.260
...
Hz
```

but a detected mode may be:

```text
frequency = 1.0837 Hz
```

or:

```text
frequency = 2.1667 Hz
```

The mode is not permanently assigned to the nearest transform bin.

Therefore questions such as "how many bins per octave?" mean:

> How finely must scale-space be sampled for the continuous ridge estimates to
> converge to the required accuracy?

They do **not** mean:

> Which discrete frequencies is CycleBank allowed to represent?

This distinction is architectural.

Increasing scales per octave should improve numerical approximation, not change
the ontology of the timing system.

---

## Continuous time-frequency interpretation

For evidence signal \(x(t)\), consider an analytic multiscale representation:

$$
W_x(a,t)
$$

where \(a\) is scale.

A complex coefficient can be written:

$$
W_x(a,t) = A(a,t)e^{i\phi(a,t)}
$$

and therefore contains both:

* local oscillatory amplitude;
* local phase.

For a sufficiently analytic representation, temporal phase evolution also
contains instantaneous-frequency information.

Conceptually:

$$
\hat{\omega}(a,t)
\approx
\operatorname{Im}
\left(
\frac{\partial_t W_x(a,t)}
     {W_x(a,t)}
\right)
$$

subject to the exact transform and sign convention.

This permits an oscillation represented near a nominal transform scale to be
assigned a **continuous estimated frequency** rather than simply inheriting the
scale's center frequency.

This is the mathematical basis for frequency reassignment and
synchrosqueezing-like representations.

The exact transform used by #92 need not literally be textbook
synchrosqueezing, but the architecture should preserve this idea:

> **Sample scale-space discretely, estimate oscillatory modes continuously.**

---

## Causality constraint

A conventional centered Continuous Wavelet Transform is unsuitable for the
runtime path because a wavelet centered on time \(t\) generally depends on
samples after \(t\).

Runtime estimation must be causal.

The underlying representation must therefore use a causal analytic equivalent,
such as:

* a causal constant-Q analytic filter bank;
* one-sided complex wavelets;
* recursive complex resonators;
* another causal analytic multiscale transform with equivalent properties.

A representative causal kernel is:

$$
h_f(\tau)
=
e^{-\tau/\tau_f}
e^{i 2\pi f\tau}
u(\tau)
$$

where \(u(\tau)\) is the causal step function.

A recursive implementation may consequently look very similar to a bank of
complex resonators.

That similarity does not make the transform frequencies discrete musical
hypotheses. The resonators are samples of the underlying analytic field.

---

## Constant relative bandwidth

The transform should preferably have approximately constant relative bandwidth
across frequency, i.e. constant-Q or equivalent behavior.

This gives the desired temporal scaling naturally:

$$
\Delta t \propto \frac{1}{f}
$$

Slower cycles therefore integrate evidence over longer real-time intervals than
faster cycles.

This is the same property previously expressed manually as:

$$
\tau = \frac{C}{f}
$$

and:

$$
\rho = e^{-dt/\tau}
$$

where \(C\) is memory measured in cycles.

Under the new formulation, **period-scaled memory is a natural consequence of
the multiscale representation rather than a special case layered on top of
independent trackers**.

One fixed window length for all temporal frequencies is rejected.

---

## Modes are ridges, not bins

A persistent oscillatory process produces a ridge through time-frequency
space:

```text
frequency
   ↑
   │       ╭────────────╮
   │      ╱              ╲
   │─────╯                ╰────
   │
   │        another ridge ─────
   │
   └────────────────────────────→ time
```

The CycleBank represents these persistent ridges sparsely.

Conceptually:

```text
continuous causal time-frequency field
                  ↓
          local maxima / ridges
                  ↓
       temporal ridge association
                  ↓
   CycleMode {
       phase,
       frequency,
       strength,
       confidence
   }
```

The hard tracking problem is therefore not fundamentally:

> Which oscillator bucket won?

It is:

> Which ridge at the current observation corresponds to which persistent mode
> from the previous observation?

Temporal association should use continuity in quantities such as:

$$
\log f
$$

$$
\phi
$$

$$
A
$$

and potentially channel support and frequency slope.

Birth, persistence, merging, splitting, and death of ridges are hypothesis
management operations around the continuous field.

---

## Sparse approximation to a continuous belief

Conceptually, musical evidence induces a multimodal belief over oscillatory
state:

$$
p(f,\phi \mid x_{\le t})
$$

CycleBank does not need to maintain that complete probability density.

Instead, it stores its important modes.

Thus CycleBank can be understood as:

> **a sparse modal approximation to a continuous, multimodal temporal belief
> field.**

For example:

```text
frequency support

        /\                  /\
       /  \       /\       /  \
──────/────\─────/──\─────/────\────
       ↑          ↑         ↑
      mode A     mode B    mode C
```

This interpretation allows multiple simultaneous clocks without forcing a
winner.

---

## Prediction

Each tracked mode provides continuous phase and frequency.

The simplest causal prediction is:

$$
\phi(t+\Delta)
=
\operatorname{wrap}
\left(
\phi(t)+\omega(t)\Delta
\right)
$$

If reliable frequency slope is available later:

$$
\phi(t+\Delta)
\approx
\phi(t)
+
\omega(t)\Delta
+
\frac{1}{2}\dot{\omega}(t)\Delta^2
$$

Future event/reference-phase timing is derived from this state.

There is no separate foundational:

```text
nextBeatTime
nextBarTime
nextEventTimestamp
```

variable.

The estimator predicts a continuous oscillator state.

---

## No architectural requirement for a PLL

A PLL is no longer part of the architectural definition.

A PLL-like state estimator or smoother may still be useful for:

* ridge continuity;
* noisy instantaneous-frequency estimates;
* prediction through weak evidence;
* smoothing phase/frequency trajectories;
* maintaining identity through temporary ridge loss.

But it is an implementation option.

The architecture must not require every mode to originate as an independent
PLL initialized at a fixed frequency seed.

If phase derivatives and frequency reassignment recover the ridge's continuous
frequency directly, the estimator should use that information rather than
forcing a PLL to crawl from a nearby seed.

---

## Missing evidence

A tracked mode must be capable of prediction through temporary absence of
strong ridge evidence.

Therefore:

* a missing expected transient does not reset phase;
* a temporarily weak ridge may continue by free-running prediction;
* confidence may decay as evidence disappears;
* reacquisition should be continuous when the ridge reappears;
* isolated unrelated transients should not catastrophically change an
  established mode.

Ridge tracking may use predictive state to bridge short gaps.

---

## Evidence channels

The multiscale representation must support multiple causal evidence channels.

It must not be restricted to one monolithic onset envelope.

Conceptually:

```text
x_b(t)
```

for evidence channels \(b\), producing:

$$
W_b(a,t)
$$

Potential Rust-owned causal evidence includes:

* onset/transient evidence;
* spectral flux;
* Energy;
* low-band activity;
* mid-band activity;
* high-band activity;
* future canonical rhythmic evidence.

Different channels may support different modes.

For example:

```text
low-frequency evidence  → strong 1.7 Hz mode
high-frequency evidence → strong 3.4 Hz mode
texture evidence        → slower 0.23 Hz mode
```

CycleBank should preserve enough channel-support information to distinguish
these cases.

The exact evidence inventory is not fixed by this ADR.

If #92 requires additional causal evidence, that evidence belongs in
`runtime-core` under ADR 0001.

Python and TypeScript must not create competing rhythmic-analysis pipelines.

---

## Strength and confidence

`strength` and `confidence` remain distinct.

### Strength

Strength expresses local support for a ridge/mode.

It may derive from quantities such as:

* analytic coefficient magnitude;
* reassigned energy density;
* support across nearby scales;
* support across channels.

### Confidence

Confidence expresses how trustworthy the predicted mode is.

It may incorporate:

* strength;
* ridge persistence;
* phase continuity;
* frequency continuity;
* ridge sharpness;
* local scale/frequency uncertainty;
* prediction residual;
* agreement between evidence channels;
* mode age;
* recent missing evidence.

Confidence must be:

* deterministic;
* bounded;
* testable;
* distinct from raw strength.

The exact v1 formula remains tunable.

---

## Frequency uncertainty

Because the underlying object is a continuous field, uncertainty should be
treated as meaningful information rather than hidden behind arbitrary bin
choice.

Useful diagnostics may include:

* ridge width in log-frequency;
* curvature around the local spectral maximum;
* disagreement among nearby scales;
* variance of reassigned instantaneous-frequency estimates.

A future CycleMode may expose a frequency-uncertainty quantity if evaluation
shows it useful.

It is not required in the first PlayerObservation contract.

---

## Numerical scale resolution

The number of evaluated scales per octave is an implementation parameter.

No particular value such as:

```text
6 scales/octave
12 scales/octave
24 scales/octave
```

is part of the architecture.

#92 should measure convergence.

For representative synthetic signals, increasing scale resolution should cause
estimated ridge trajectories to converge:

$$
\hat f_{6}(t)
\rightarrow
\hat f_{12}(t)
\rightarrow
\hat f_{24}(t)
$$

within a defined tolerance.

This provides the principled answer to scale density:

> Use the coarsest scale sampling for which ridge frequency, phase, and
> prediction metrics have converged sufficiently for the application's timing
> error budget.

Scale density is therefore a numerical-accuracy/performance tradeoff.

It is not musical discretization.

---

## Phase reference and causal filter delay

Causal analytic filters introduce frequency-dependent transfer phase and
potential effective delay.

That phase must not silently become the definition of musical event phase.

The implementation must make phase reference semantics explicit.

If the transform introduces a deterministic phase offset or group delay, it
must be:

* analytically compensated;
* calibrated;
* or explicitly incorporated into the estimator's reference convention.

The same convention must hold in runtime and offline diagnostics.

A filter's internal phase lag is not automatically musical phase zero.

This is especially important when evaluating prediction error in tens of
milliseconds.

---

## No independent global tempo

There is no foundational global tempo state.

Each mode has its own continuous frequency:

```text
mode A: 0.72 Hz
mode B: 1.44 Hz
mode C: 2.16 Hz
mode D: 3.37 Hz
```

If a human interprets one mode as the beat, BPM is merely:

$$
\text{BPM}=60f
$$

Tempo change is represented by movement of a mode's ridge through frequency
space.

Several plausible rhythmic rates may coexist.

---

## Multiple simultaneous modes

CycleBank must support several persistent modes simultaneously.

Valid states include:

* harmonically related modes;
* integer ratios;
* non-power-of-two rational ratios;
* independent rhythmic layers;
* polymeter;
* close competing modes;
* slow and fast recurrences at the same time.

The architecture must not collapse immediately to a single "dominant tempo."

---

## Rational relationships

Relationships between modes are generic.

For two modes \(i\) and \(j\), if:

$$
m\omega_i \approx n\omega_j
$$

for small integers \(m,n\), define generalized phase difference:

$$
\psi
=
\operatorname{wrap}
\left(
m\phi_i-n\phi_j
\right)
$$

A stable \(\psi\) indicates rational phase locking.

This supports relationships such as:

```text
2:1
3:2
5:4
7:4
```

without creating meter-specific tracker types.

Relationship analysis is initially diagnostic.

This ADR does not require rational coupling to alter mode dynamics.

---

## Relationship to semantic musical structure

CycleBank represents periodic and quasi-periodic temporal structure.

It does not claim that all musical structure is periodic.

Important Sections and Transitions may instead arise from:

* timbral novelty;
* harmonic change;
* density change;
* silence;
* buildup/release;
* one-off events;
* non-periodic texture changes.

A future PlayerPolicy will combine CycleBank modes with immediate and
persistent non-periodic musical state.

CycleBank must not become a general semantic section classifier.

---

## Public Rust authority

All canonical temporal-analysis behavior belongs in `runtime-core`.

Expected concepts are approximately:

```text
CycleBank
CycleMode
CycleBankConfig
CycleObservation
CycleRelation
```

The implementation may also expose internal concepts representing:

```text
analytic filter bank
ridge candidates
ridge tracker
frequency reassignment
```

Exact names are implementation-defined.

Foundational public types should not be named:

```text
BeatTracker
BarTracker
PhraseTracker
TempoTracker
```

because those labels impose semantic interpretation at the wrong layer.

Rust owns:

* causal multiscale analysis;
* transform state;
* frequency reassignment;
* ridge extraction;
* ridge association;
* mode birth/death/merge behavior;
* phase/frequency estimation;
* free-running prediction;
* strength/confidence calculation;
* phase-reference correction;
* rational-relation diagnostics;
* deterministic CycleBank outputs.

Python and TypeScript consume bindings.

They do not reimplement this machinery.

---

## Cross-surface architecture

The intended pipeline is:

```text
                  runtime-core
      causal analytic temporal pipeline
                       ↓
                  CycleBank
                 /         \
              PyO3         WASM
               |             |
        offline Python     browser
         diagnostics       consumer
```

There is one canonical pipeline.

Python may provide:

* offline corpus iteration;
* plotting;
* tables;
* oracle comparison;
* metrics aggregation.

TypeScript may provide:

* browser lifecycle;
* WASM loading;
* UI;
* visualization of diagnostics.

Neither surface independently computes:

* wavelets/resonators;
* instantaneous frequency;
* reassignment;
* ridge extraction;
* mode tracking;
* confidence;
* phase prediction.

Per ADR 0001:

> If both Python and TypeScript need to know the behavior, that behavior belongs
> in Rust.

Rust-generated types should define shared CycleBank structures across
bindings.

---

## Relationship to AnalysisTimebase

CycleBank consumes the authoritative timebase established by issue #91.

Its state progresses from explicit sample-clock observations.

It must never use:

* `requestAnimationFrame`;
* wall-clock callback cadence;
* assumed FPS;
* ONNX completion timing.

Equivalent timestamped evidence must produce equivalent CycleBank state
regardless of how transport blocks are partitioned.

A `streamEpoch` change resets persistent temporal-analysis state
deterministically.

---

## Relationship to PlayerObservation

CycleBank does not directly define the Player's model input.

A later ADR will define a fixed, versioned `PlayerObservation`.

Cycle modes will likely be represented continuously, for example:

```text
[
    cos(phase),
    sin(phase),
    log2(frequencyHz),
    strength,
    confidence
]
```

possibly augmented with:

* channel support;
* frequency slope;
* uncertainty;
* selected rational relationships.

Raw wrapped phase should not be used directly as the long-term neural input
because of its discontinuity at wraparound.

This ADR does not decide:

* number of modes supplied to the Player;
* mode ordering;
* selection policy;
* which relationships enter PlayerObservation;
* fixed tensor shape.

Those decisions belong to the PlayerObservation contract after #92.

---

## Versioning

The canonical implementation should expose a Rust-owned:

```text
CYCLE_BANK_VERSION
```

or equivalent once public semantics exist.

The version must change when downstream CycleBank meaning changes, including:

* transform semantics;
* phase reference;
* frequency reassignment semantics;
* ridge extraction;
* ridge association;
* mode management;
* confidence semantics;
* default parameters whose changes alter outputs;
* relation semantics.

Python and WASM consume the Rust-owned version.

No independent version literals may exist outside Rust.

This version is distinct from:

```text
FEATURE_VERSION
ANALYSIS_PIPELINE_VERSION
CONTROLLER_VERSION
future PLAYER_OBSERVATION_VERSION
```

---

## Diagnostics before Player integration

CycleBank must prove that it produces useful predictive state before its modes
are allowed into PlayerObservation.

### Synthetic tests

At minimum, #92 must cover:

#### Off-grid sinusoidal frequency

Generate a known oscillation whose frequency lies between transform sample
centers.

Verify that the reported ridge frequency converges near the true continuous
frequency rather than remaining at the nearest scale center.

#### Scale-resolution convergence

Run the same signal at several scale densities.

Verify that estimated frequency/phase trajectories converge as scale-space
sampling becomes finer.

#### Clean periodic pulse train

Verify:

* stable ridge detection;
* stable phase;
* useful future event prediction.

#### Timing jitter

Add realistic temporal jitter.

Verify the mode remains coherent and prediction error stays bounded.

#### Missing events

Remove expected pulses.

Verify the mode can free-run through short unsupported intervals and confidence
decays appropriately rather than hard resetting.

#### Spurious transients

Add unrelated events.

Verify stable modes are not catastrophically rephased.

#### Frequency drift / chirp

Generate a gradually varying frequency.

Verify the ridge follows continuously.

#### Multiple simultaneous modes

Generate multiple oscillatory components.

Verify several modes survive simultaneously.

#### Rational / polymetric relation

Generate relationships such as 5:4 or 7:4.

Verify generalized phase difference remains stable when appropriate.

#### Causality

Alter future evidence.

Verify previously emitted state is unchanged.

#### Chunk invariance

Feed identical timestamped evidence with different block partitioning.

Verify CycleBank state at equivalent sample times is unchanged.

#### Phase-reference calibration

Verify the transform's reported phase corresponds to the documented musical
reference convention rather than an unaccounted filter delay.

---

## Real-song evaluation

After synthetic correctness is established, run the Rust CycleBank over several
project songs.

Offline annotations may be used as measurement oracles only.

Where a suitable event oracle exists, report:

* signed timing error;
* absolute timing error;
* median absolute error;
* p90 absolute error;
* p95 absolute error;
* fraction within 20 ms;
* fraction within 30 ms;
* fraction within 40 ms;
* acquisition time;
* mode frequency trajectory;
* mode confidence;
* behavior through weak passages.

The purpose is not to identify a single canonical beat.

The question is:

> Does at least one supported short-timescale mode causally predict salient
> temporal events with useful accuracy?

Slower modes and rational relationships should also be inspected even where no
human semantic annotation exists.

---

## Parameters deliberately left empirical

This ADR fixes the representation and ownership architecture, not every DSP
constant.

#92 should determine empirically:

* causal analytic transform family;
* transform Q / cycles of support;
* scale-frequency range;
* scales per octave;
* reassignment estimator;
* ridge detection thresholds;
* ridge association cost;
* birth/death hysteresis;
* duplicate-ridge suppression;
* weak-evidence persistence;
* confidence formula;
* channel weighting;
* rational-relation tolerances;
* whether frequency slope is worth maintaining.

Parameters must be:

* centralized;
* explicit;
* deterministic;
* testable.

They must not appear as duplicated magic values across Rust, Python, and
TypeScript.

---

## Rejected alternatives

### Fixed frequency buckets

Rejected as the musical representation.

A finite transform grid is necessary numerically, but its center frequencies
must not become the only frequencies CycleBank can represent.

### A fixed number of PLLs per octave

Rejected as the architectural primitive.

Adaptive PLLs remain a possible ridge-smoothing/tracking technique, but the
underlying object is the continuous analytic field and its ridges.

### One global tempo

Rejected because several useful clocks may coexist.

### Separate beat, bar, and phrase trackers

Rejected because these are semantic interpretations of generic temporal modes.

### Reset phase on every onset

Rejected because it produces reactive event timestamps rather than predictive
state and is fragile to missing/spurious events.

### Standard centered CWT in the runtime path

Rejected because it uses future support and violates causal live operation.

A causal analytic multiscale equivalent is required.

### Python or TypeScript wavelet/CycleBank implementations

Rejected under ADR 0001.

One Rust implementation serves both training diagnostics and runtime.

---

## Consequences

Positive:

* the model matches the underlying continuous nature of oscillatory timing;
* frequency is not artificially quantized by acquisition bins;
* constant-Q analysis naturally provides period-scaled memory;
* several simultaneous temporal modes are first-class;
* polymeter requires no special architecture;
* off-grid frequencies can be estimated directly;
* phase and frequency arise from the same analytic representation;
* CycleBank becomes a sparse ridge representation rather than a forest of
  competing hand-managed trackers;
* numerical scale density can be chosen through convergence tests;
* runtime and training share one causal implementation.

Costs:

* causal analytic filters require careful phase-reference handling;
* ridge extraction and identity tracking are nontrivial;
* time-frequency resolution and latency must be tuned;
* synchrosqueezing/reassignment becomes numerically delicate around very weak
  coefficients;
* ridge splitting/merging may require hysteresis;
* CycleBank still does not solve non-periodic semantic Transition prediction.

---

## Implementation boundary for #92

#92 implements:

```text
AnalysisTick / causal evidence
        ↓
causal analytic multiscale representation
        ↓
continuous frequency estimation / reassignment
        ↓
ridge extraction + tracking
        ↓
runtime-core CycleBank
        ↓
predictive modes + diagnostics
       / \
    PyO3  WASM
```

#92 does **not**:

* define the final `PlayerObservation`;
* feed CycleBank state into ONNX;
* retrain the Player;
* redesign Controls;
* create semantic beat/bar/phrase trackers;
* introduce a global tempo variable;
* solve all non-periodic Transition prediction;
* put wavelet/ridge math in Python or TypeScript.

The next architectural step after #92 is a separate versioned
`PlayerObservation` contract combining:

```text
immediate musical evidence
+ predictive CycleBank modes
+ Map / Physics state
```

Only after that contract is fixed should the new PlayerPolicy be trained.
