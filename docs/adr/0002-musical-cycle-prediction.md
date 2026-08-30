# 0002 — Musical timing as a bank of predictive cycle hypotheses

Status: Accepted (2026-08-29, authorized by repository owner)
Revision: 2026-08-29 — "tempo" is no longer an independently estimated
musical-state concept. Frequency is intrinsic state of every `CycleHypothesis`;
"tempo" is a human-readable interpretation of the frequency of whichever
hypothesis has been chosen to be called the beat. No code or binding change
is implied for already-implemented types; this clarifies the architecture.

## Context

FractalSync's Player must anticipate musical events rather than merely react to
them. The Domain Contract requires the Player to begin steering before a
Transition and land visually significant events, especially crossings of The
Shore, at the musically appropriate instant.

The initial design discussion described this as a progression from beat phase,
to bar phase, to phrase/section state. That framing implies separate semantic
trackers for concepts such as "beat", "bar", and "phrase", and it also implies
a single global "tempo" scalar that the bar and phrase trackers are derived
from.

That is the wrong abstraction for the music FractalSync is intended to perform.

In complex and polymetric music:

- multiple instruments may imply different simultaneous meters;
- a pattern that one listener calls a "beat" may be another pattern's
  subdivision;
- a "bar" is mathematically just a slower periodic or quasi-periodic cycle
  until musical semantics are imposed;
- several nested or rationally related cycles may coexist without any single
  one being the canonical clock;
- slow structural cycles, dynamic swells, riffs, and phrase-scale recurrences
  may carry useful predictive timing even when they do not map cleanly onto
  conventional notation.

A system that first decides on one beat, derives one bar from that beat, and
then derives one phrase from that bar would bake a particular music-theory
interpretation into the runtime. Polymeter and ambiguous hierarchy would then
become edge cases.

They should not be edge cases.

The more general mathematical object is a set of simultaneous oscillatory
hypotheses at different temporal frequencies. Terms such as beat, subdivision,
bar, riff cycle, phrase, and polymeter are possible interpretations of those
hypotheses and their relationships.

FractalSync therefore needs one causal predictive timing mechanism that works
identically across timescales.

## Decision

FractalSync will represent predictable musical timing as a **bank of generic
predictive cycle hypotheses**.

There will be no separate foundational beat tracker, bar tracker, or phrase
tracker. There will also be no separately estimated "tempo" scalar: the
angular frequency \(\omega_k\) (or \(f_k = \omega_k / 2\pi\)) that every cycle
hypothesis already carries **is** its tempo, and several hypotheses may carry
different valid values at once. "BPM" is a downstream interpretation defined
as \(60 f_k\) for the hypothesis a listener has chosen to call the beat.

The canonical musical-state timing primitive is:

```text
CycleHypothesis
    frequency
    phase
    strength
    confidence
    optional per-channel support
```

A collection of these hypotheses forms the `CycleBank`.

The same estimator equations apply to every hypothesis regardless of whether
its period is 125 ms, 500 ms, 4 s, or 20 s.

Semantic labels such as "beat", "bar", "phrase", or "7/8" are downstream
interpretations. They are not part of the mathematical identity of a cycle
hypothesis and must not be required by the core timing machinery.

The architectural flow is:

```text
audio
  ↓
canonical low-level audio evidence
  ↓
CycleBank / musical temporal state
  ↓
Player policy
  ↓
human-plausible Controls
  ↓
Rust Physics
  ↓
c
  ↓
J(c)
```

The responsibility boundary is:

> DSP estimates and predicts the clocks.
> The Player decides what those clocks mean visually.

This ADR extends ADR 0001: the runtime implementation of `CycleBank`,
`CycleHypothesis`, phase tracking, and associated causal timing math belongs in
`runtime-core`. Other languages consume bindings. Any required differentiable
or behavioral mirror is subject to ADR 0001's manifest and parity rules.

## Causal observation model

Let the canonical audio evidence contain channels

$$
x_b[n]
$$

where \(b\) may represent onset evidence, spectral flux, energy, low-frequency
activity, high-frequency activity, or other causal features.

These channels must not be collapsed prematurely into a single "beat"
envelope. Different channels may support different simultaneous cycles.

For each cycle hypothesis \(k\), define angular frequency

$$
\omega_k = 2\pi f_k
$$

and predicted phase

$$
\phi_k^{-}[n+1]
=
\operatorname{wrap}
\left(
\phi_k[n]+\omega_k[n]\Delta t
\right).
$$

The hypothesis maintains causal complex evidence for the candidate cycle.
One suitable implementation is complex demodulation / a causal complex
resonator:

$$
z_{b,k}[n]
=
\rho_k z_{b,k}[n-1]
+
(1-\rho_k)x_b[n]e^{-i\phi_k^{-}[n]}.
$$

The magnitude

$$
A_{b,k}=|z_{b,k}|
$$

measures support for that temporal frequency in channel \(b\).

The complex phase contributes evidence about phase error relative to the
free-running oscillator. The exact discriminator may combine complex-filter
phase with transient/onset evidence; the filter's arbitrary transfer phase
must not itself be assumed to define musical phase zero.

This distinction is important:

* the causal filter detects an oscillatory pattern;
* the predictive oscillator maintains the clock;
* salient evidence corrects the clock;
* evidence does not reset the clock every time an event occurs.

## Memory scales with period

The estimator will use memory measured primarily in **cycles**, not in a fixed
number of audio frames.

For a desired memory of \(M\) cycles:

$$
\tau_k = \frac{M}{f_k}
$$

and a corresponding exponential memory coefficient may be

$$
\rho_k=e^{-\Delta t/\tau_k}.
$$

Therefore the same estimator automatically observes longer history for slower
cycles.

For example, with four cycles of effective memory:

| Frequency | Period | Approximate history |
| --------: | -----: | ------------------: |
|      4 Hz | 250 ms |                 1 s |
|      2 Hz | 500 ms |                 2 s |
|    0.5 Hz |    2 s |                 8 s |
|  0.125 Hz |    8 s |                32 s |

There is no mathematical boundary at which a "beat tracker" turns into a "bar
tracker". It is the same estimator operating at another timescale.

Implementation may use bounded windows or equivalent filters for engineering
reasons, but the architectural quantity is history measured relative to cycle
period.

## Predictive phase and frequency adaptation

Each active hypothesis is a free-running adaptive oscillator.

Conceptually its state is:

$$
H_k =
(\phi_k,\omega_k,A_k,q_k)
$$

where:

* \(\phi_k\) is current phase;
* \(\omega_k\) is estimated angular frequency;
* \(A_k\) is cycle strength;
* \(q_k\) is confidence.

Between observations it predicts forward:

$$
\phi_k^{-}
=
\operatorname{wrap}
(\phi_k+\omega_k\Delta t).
$$

Given a causal phase-error measurement \(\epsilon_k\), a PLL-like correction
may update phase and frequency:

$$
\phi_k
\leftarrow
\operatorname{wrap}
\left(
\phi_k^{-}
+
K_p w_k\epsilon_k
\right)
$$

$$
\omega_k
\leftarrow
\omega_k
+
K_i w_k\frac{\epsilon_k}{\Delta t}.
$$

Here \(w_k\) is derived from evidence strength and confidence.

The exact gains, acquisition/capture ranges, confidence update, and frequency
bounds are implementation parameters, not architectural constants.

The invariant is that the oscillator **predicts first and is corrected by
evidence afterward**.

It must not simply report the timestamp of the most recently observed onset.

## Tempo

There is no independently estimated musical-state variable called "tempo".

Tempo is the frequency of a cycle hypothesis, stripped of its
music-theory interpretation:

$$
\mathrm{BPM}_k = 60\,f_k = \frac{60\,\omega_k}{2\pi}.
$$

Because \(\omega_k\) is already state of every hypothesis, several hypotheses
may simultaneously carry valid and incompatible BPM readings. For example, a
periodicity detected at 1.75 Hz is mathematically a "105 BPM" pulse; whether a
listener should also entertain a "210 BPM" half-time interpretation depends on
which higher-frequency hypothesis is also active and on the stability of their
2:1 phase relation. The estimator does not have to adjudicate that question.
It maintains both hypotheses and their generalized phase difference
\(\psi^{2:1}\), and the Player uses whichever timescale is visually relevant.

Tempo change also falls out for free. The PLL adapts \(\omega_k\) continuously:

$$
\omega_k \leftarrow \omega_k + K_i\,w_k\,\frac{\epsilon_k}{\Delta t},
$$

so a gradually accelerating performance produces a gradually rising
\(\omega_k\) on the supporting hypothesis without any separate "tempo detector"
state variable.

A possible later extension is **anticipating** a tempo change rather than only
tracking one as it happens, by augmenting a hypothesis with frequency velocity
\(\dot{\omega}_k\) and predicting

$$
\omega_k(t+\tau) \approx \omega_k(t) + \dot{\omega}_k\,\tau,
$$

$$
\phi_k(t+\tau) \approx \phi_k(t) + \omega_k\tau + \tfrac{1}{2}\dot{\omega}_k\tau^2.
$$

This is explicitly deferred: physical performance acceleration is usually
modest, and instrument meter can be exotic without the underlying tempo
actually accelerating rapidly. The first job is to make adaptive \(\omega\)
work well; \(\dot{\omega}\) is not part of the initial CycleBank contract.

The architectural consequence is that any future core type, binding, or
diagnostic that wants to talk about "the tempo" must instead identify
**which hypothesis** it refers to, or expose tempo as a derived value
\(\mathrm{BPM}_k\) computed from that hypothesis's \(f_k\).

## Future prediction

Once phase and frequency are estimated, future phase follows directly:

$$
\phi_k(t+\delta)
=
\operatorname{wrap}
\left(
\phi_k(t)+\omega_k\delta
\right).
$$

For any chosen reference phase, the estimator can therefore expose predicted
time to the next occurrence.

Predictions are not quantized to the feature hop. The oscillator free-runs in
continuous time between observations using timestamps and can produce
sub-frame predictions even when correction evidence arrives at the canonical
feature cadence.

Higher-resolution transient timestamps may later improve correction accuracy
without changing this architecture.

## Confidence

Phase without confidence is not useful musical state.

Each hypothesis must carry a confidence estimate informed by some combination
of:

* oscillatory strength;
* phase coherence;
* persistence over multiple cycles;
* prediction error;
* hypothesis age;
* consistency across evidence channels.

One possible coherence statistic is

$$
R_k
=
\frac{
\left|
\sum_t a_t e^{i\epsilon_t}
\right|
}{
\sum_t a_t+\varepsilon
}.
$$

Aligned phase errors produce \(R_k\) near one; incoherent evidence produces a
value near zero.

The exact confidence function remains an implementation decision. The
architectural requirement is that uncertainty remains explicit rather than
forcing the system to nominate a single authoritative clock.

## Multiple simultaneous meters

The CycleBank deliberately permits several strong hypotheses at once.

For example, different audio channels may support different frequencies:

```text
low-frequency evidence  → H₁
mid-frequency evidence  → H₂
high-frequency evidence → H₃
transient evidence      → H₄
```

No hypothesis is automatically declared the "real beat".

This makes polymetric material an ordinary state of the estimator rather than
a special case.

A guitar cycle, bass cycle, drum cycle, and slower structural cycle may all
remain active simultaneously.

The Player may learn that different hypotheses matter for different visual
behaviors.

## Relationships between cycles

The CycleBank may detect persistent rational relationships between hypotheses.

For two hypotheses \(i\) and \(j\), choose small integers \(m,n\). If

$$
m\omega_i \approx n\omega_j
$$

then inspect the generalized phase difference

$$
\psi_{ij}^{m:n}
=
\operatorname{wrap}
(m\phi_i-n\phi_j).
$$

If both the frequency relation and \(\psi\) remain stable, the cycles are
phase-locked.

This describes relationships conventionally interpreted as subdivisions,
meter, polymeter, or larger phrase structure without requiring those labels.

For example, a 7:4 relationship is represented simply by an integer relation
whose generalized phase remains stable.

The system may eventually represent these relations as a graph:

```text
CycleHypothesis ── phase/frequency lock ── CycleHypothesis
       │                                      │
       └──────── phase/frequency lock ─────────┘
```

The graph is mathematical temporal structure. Music-theory names may be
attached downstream if useful, but are not required.

## Player observation

The Player must not receive raw wrapped phase as a scalar because the
\(2\pi \rightarrow 0\) wrap introduces an artificial discontinuity.

A cycle hypothesis exposed to the learned policy should use a continuous
representation such as:

$$
[
\cos\phi_k,\;
\sin\phi_k,\;
\log_2 f_k,\;
A_k,\;
q_k
]
$$

with optional channel-support and coupling features.

The exact fixed-size tensor layout is deferred until the Player I/O contract
is specified.

Whatever layout is chosen must preserve these properties:

1. no semantic `beat`, `bar`, or `phrase` slot is required;
2. multiple simultaneous hypotheses can coexist;
3. phase is represented continuously;
4. frequency is explicit;
5. strength and confidence are explicit;
6. useful relationships between hypotheses can be represented.

A bounded set of adaptive oscillator slots is preferred over an unbounded
variable-length runtime structure when defining model I/O.

## Relationship to Transitions and Sections

The CycleBank models **predictive temporal structure**, not semantic section
identity.

A slow stable cycle may correspond to what a listener would call a phrase or
section-scale recurrence, and the same mathematics used for faster cycles
must be capable of representing it.

However, not every Section or Transition is periodic. A one-off breakdown,
tempo change, or structural handoff may require non-periodic evidence in
addition to the CycleBank.

Therefore:

* phrase-scale periodicity belongs in the CycleBank;
* semantic Section identity, if needed, is downstream musical state;
* Transition anticipation may combine CycleBank predictions with harmonic,
  timbral, energy, or learned structural evidence;
* the CycleBank must not be distorted into a classifier for non-periodic
  events.

This preserves one mathematical timing system without pretending all musical
structure is periodic.

## Runtime causality

The deployed estimator must be causal.

It may only use evidence available at or before the current audio timestamp.

A symmetric Morlet CWT, centered STFT window requiring future samples, or
other acausal transform may be used for:

* offline diagnostics;
* research;
* training labels/oracles;
* comparison against the causal estimator.

It may not be used as the live runtime implementation.

The runtime implementation should use causal complex resonators, demodulators,
IIR/FIR filters, PLL-like state, or mathematically equivalent causal machinery.

## Initial frequency coverage

The architectural decision does not pin a permanent frequency grid.

A useful initial diagnostic range is approximately:

```text
0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8 Hz
```

but fixed octave bins are not sufficient as the final phase estimator.

Real musical frequencies generally lie between those centers. A fixed 2 Hz
oscillator, for example, cannot remain phase-accurate for music whose actual
pulse is 2.17 Hz.

Therefore hypotheses must eventually adapt their frequency or be instantiated
densely enough that a PLL/frequency estimator can converge to the observed
cycle.

Octave-spaced values are initialization or diagnostic bands, not musical
truth.

## Implementation boundary

The first implementation shall introduce the generic mechanism, not a
beat-specific API.

Preferred conceptual types:

```text
CycleBank
CycleHypothesis
CycleConfig
CycleObservation
```

An internal generic `PhaseTracker`/PLL may exist per hypothesis, but no public
core type should imply that its mathematics applies only to beats.

The implementation lives in `runtime-core` under ADR 0001.

Python and WASM bindings expose state for diagnostics and later Player
consumption.

The initial validation effort may concentrate on the shortest strong musical
cycles because they:

* acquire quickly;
* provide many ground-truth events per song;
* make timing error easy to measure;
* expose causal prediction failures rapidly.

That validation order is **not** an architectural statement that beats are a
different kind of object from slower cycles.

Slower hypotheses should use the same machinery with proportionally longer
memory.

## Diagnostics

Before CycleBank state becomes model input, diagnostics must demonstrate that
the causal estimator actually predicts.

At minimum, evaluation should report:

* estimated frequency over time;
* predicted phase over time;
* hypothesis strength;
* confidence;
* lock acquisition time;
* signed timing error against an offline event oracle where one exists;
* absolute timing error;
* median / p90 / p95 timing error;
* fraction of predictions within 20 ms, 30 ms, and 40 ms;
* behavior through missing events;
* behavior under frequency drift (per-hypothesis \(\omega_k\) tracking);
* coexistence of multiple supported cycles;
* stability of rational phase relationships where present.

Offline beat annotations may be used as one convenient oracle for validating
short-timescale hypotheses. They do not redefine the runtime object as a beat
tracker.

## Consequences

### Positive

* Beat, subdivision, bar, polymeter, and slower recurrent structure use one
  mathematical abstraction.
* Polymeter is native rather than exceptional.
* No single global tempo or meter is required. Tempo lives inside each
  hypothesis, so several BPM readings can coexist; tempo change is a
  consequence of adapting \(\omega_k\), not a separate detector.
* The Player receives prediction rather than merely recent-event timestamps.
* Long-timescale structure is reached by scaling memory and frequency, not by
  introducing a second architecture.
* Exact timing is delegated to deterministic DSP rather than requiring the
  neural network to rediscover oscillators from short feature windows.
* The Player can focus on the learned problem: deciding how musical temporal
  structure should become visual intent.
* The architecture naturally supports future phase-coupling features.
* Runtime behavior remains causal and suitable for live performance.

### Negative

* The musical-state layer becomes persistent state rather than a stateless
  feature transform.
* Multiple hypotheses introduce ambiguity that must be represented rather than
  hidden.
* Harmonic duplicates, octave errors, and nearby competing hypotheses require
  confidence, acquisition, capture, and possibly hypothesis-suppression logic.
* Slow cycles require long observation time before confidence becomes high.
* Fixed-size model I/O for a bank of hypotheses requires an explicit packing
  contract.
* Some non-periodic musical structure will still require complementary
  evidence; CycleBank is not a universal section classifier.

## Rejected alternatives

### Separate beat, bar, and phrase trackers

Rejected because the distinction is semantic rather than fundamental to the
timing mathematics. It would hard-code a hierarchy that complex and polymetric
music routinely violates.

### One global beat/tempo PLL

Rejected because simultaneous meters and independent instrument cycles can
coexist, and because "tempo" is not a single musical-state scalar in this
architecture — it is the frequency of whichever hypothesis one chooses to call
the beat. Selecting one PLL would discard useful structure, force a tempo
interpretation onto music that may be polymetric, and make polymeter an error
condition.

### Increase the neural-network context window and let the model learn timing

Rejected as the sole timing mechanism.

A longer context is useful, but it does not justify asking the Player to
rediscover phase estimation, frequency tracking, oscillator memory, and
extrapolation from examples. Those are well-structured causal estimation
problems with deterministic mathematics.

The learned Player should consume temporal state, not spend model capacity
reinventing a clock.

### Fixed frequency bins without adaptation

Rejected for predictive phase tracking because even a small frequency mismatch
accumulates unbounded phase error.

Fixed bands remain useful for initialization, diagnostics, and measuring
modulation energy.

### Symmetric wavelets/CWT in the live runtime

Rejected because centered wavelets consume future samples and therefore cannot
provide genuinely causal anticipation.

Offline CWT remains useful as a diagnostic and research oracle.

### Event-reset timing

Rejected.

An estimator that waits for an onset and resets phase to zero reports the
present or recent past. FractalSync needs a clock that free-runs before the
event and is corrected after evidence arrives.

## Invariants

The following are architectural invariants:

1. **One timing primitive across timescales.**
   There is no separate core beat/bar/phrase mathematics.

2. **Multiple clocks may coexist.**
   No single tempo or meter is required to be authoritative. Tempo is the
   frequency of an individual `CycleHypothesis`, not a global musical-state
   variable.

3. **Prediction precedes correction.**
   Cycle hypotheses free-run between observations; evidence nudges them.

4. **Runtime timing is causal.**
   Future samples are forbidden in the deployed estimator.

5. **Memory scales with cycle period.**
   Slow cycles are not estimated from the same short fixed history as fast
   cycles.

6. **Phase is continuous at the model boundary.**
   Use sine/cosine or an equivalent representation, not raw wrapped phase.

7. **Uncertainty is explicit.**
   Every hypothesis carries strength/confidence.

8. **Semantics are downstream.**
   "Beat", "bar", "phrase", and meter names are interpretations, not required
   estimator state.

9. **DSP keeps time; the Player chooses visual meaning.**
   Cycle estimation is musical-state infrastructure, not visual policy.

10. **Rust owns runtime timing math.**
    `runtime-core` is authoritative under ADR 0001; mirrors require explicit
    justification and parity enforcement.

## Follow-up decisions

This ADR deliberately does not fix:

* the exact number of oscillator slots;
* the final frequency initialization grid;
* PLL gains and lock/acquisition thresholds;
* the exact confidence function;
* the exact low-level evidence channels;
* duplicate/harmonic hypothesis suppression;
* rational-coupling search limits;
* the fixed-size Player observation tensor;
* how much coupling information the Player receives;
* how non-periodic Transition evidence combines with the CycleBank;
* the final higher-resolution onset timestamp path;
* whether any hypothesis eventually carries a frequency-velocity
  \(\dot{\omega}_k\) for anticipatory tempo-change prediction (not part of the
  initial contract).

Those are implementation and model-I/O decisions constrained by the
architecture above.

```
