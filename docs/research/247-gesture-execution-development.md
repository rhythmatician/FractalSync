# Gesture execution, development, and traversal clocks

This is exploratory research for [#247](https://github.com/rhythmatician/FractalSync/issues/247), following [#235](https://github.com/rhythmatician/FractalSync/issues/235) and [#236](https://github.com/rhythmatician/FractalSync/issues/236). It proposes comparisons, not a Player action contract or a training gate. No experiment below has yet established playability or perceptual benefit.

The Lean module [GestureExecution](../../formal/FractalSync/GestureExecution.lean)
checks the actual product and phase-chain derivatives for one term in arbitrary
real normed spaces. Finite summation, the specific trigonometric curve, the
gauge identities, moving-anchor formula, rate budgets, and experimental claims
remain conventional arguments or open validation, not checked by this module.

## What moves

In one fixed oriented frame, let `u(φ) = (cos φ, sin φ)` and `J = [[0,-1],[1,0]]`. For differentiable phases and real 2-by-2 maps,

```text
ξ = Σ B_j u(φ_j)
ξ̇ = Σ Ḃ_j u(φ_j) + Σ B_j J u(φ_j) φ̇_j.
```

This follows from the product and chain rules. With a declared phase reference, the second sum executes the current parameterized gestures and the first develops their maps. Each term can be fast or slow. `B_j` can change size, anisotropy, orientation or phase offset, and its contributions need not be visually distinguishable. A pair of terms can cancel. Magnitudes of individual terms are therefore not additive measures of observed movement.

More generally let `ξ(t) = T(t, γ(φ(t), q(t)))`, with shape parameters `q`. In fixed vector spaces,

```text
ξ̇ = ∂ₜT + D T [∂φγ φ̇ + D_qγ q̇],
```

where derivatives of `T` are evaluated at `γ`. The parameterized curve and its image describe shape; `φ̇` selects traversal timing; changing that rate within a pass is one possible articulation; `q̇` and `∂ₜT` develop the gesture. Articulation can also come from direct control pulses, interruptions or changing shape. These names are experimental descriptions, not disjoint physical causes. A non-injective `T` or curve can erase shape differences, and a different parameterization changes the numerical phase rate.

For a moving Map anchor `b(t)` and orthonormal frame `E(t)`, write `w = E ξ` in the tangent space at `b` and `c* = Ret_b(w)`. The covariant derivative along the anchor is `D_t w = E ξ̇ + (D_t E) ξ`. Parallel transport sets the second term to zero along a smooth anchor path; another frame convention generally does not. The reference velocity is the full differential

```text
ċ* = D Ret_(b,w) [ḃ, D_t w].
```

This notation uses the connection to split the tangent of the tangent bundle into horizontal and vertical parts. It is not generally `ḃ + E ξ̇`: a curved retraction and its base-point differential matter. Chart validity, frame continuity, and the differential must be established in the selected region. Existing normal-neighborhood and clock assumptions are described in [Kepler gestures](kepler-musical-gestures.md). The reference is a target, not the actual Physics trajectory or rendered image.

## Phase gauge and bindings

With `R(δ)` the usual rotation, `B'_j = B_j R(-δ_j)` and `φ'_j = φ_j + δ_j` leave `B'_j u(φ'_j) = B_j u(φ_j)` unchanged, including for time-varying `δ_j`. But

```text
Ḃ'_j u(φ'_j) = Ḃ_j u(φ_j) - B_j J u(φ_j) δ̇_j
B'_j J u(φ'_j) φ̇'_j = B_j J u(φ_j) (φ̇_j + δ̇_j).
```

The two added terms cancel. A changing gauge can move arbitrary apparent motion between execution and development. Even a fixed offset changes the numerical phase origin. Thus report the convention before interpreting either term, and compare invariant total reference motion and actual output as well.

For an initial experiment bind each source to its unwrapped causal phase and a recorded phase epoch; use a parallel-transported anchor frame and record any map rotation separately. Fix the offset at binding. Log source identity, phase validity, confidence, resets, frame and rebinding events. Treat a reset or source replacement as an epoch boundary, not as a large inferred physical derivative. Compare a phase-continuous transition with an explicit fade/rebind, and keep both the pre- and post-transition identity. If the source phase or orientation becomes unobservable, the decomposition is unavailable until a new reference is established. Confidence is evidence about the source, not an automatic artistic gain. These are proposed experimental conventions, not claims that CycleBank IDs are stable; [#235](https://github.com/rhythmatician/FractalSync/issues/235) discusses mode turnover.

## Independent clocks

Freeze a curve and all its geometry, then change only the monotone phase-to-curve map. Uniform anomaly takes `E = nt + E₀`. For an ellipse with eccentricity `0 ≤ e < 1`, the classical clock solves `E - e sin E = nt + M₀`. It sweeps equal tangent-plane areas per equal elapsed time when `n` is constant. For `e = 0.6`, periapsis speed is four times apoapsis speed; uniform anomaly on the same ellipse is not constant arc-length speed. Match initial point, direction, duration, shape and endpoints across clocks. If a uniform-speed control is wanted, add an independently computed arc-length clock; do not call uniform anomaly uniform speed.

The classical equal-area statement concerns the fixed tangent-plane ellipse. It is not an equal-area law on the curved Map, a guarantee of free Physics motion, or a guarantee during changing anchor, frame or shape. The [existing clock analysis](kepler-musical-gestures.md) defines an intrinsic area alternative and its normal-disk and density assumptions. If musical phase drives the clock at variable rate, equal areas correspond to equal musical phase increments, not necessarily equal seconds. Rate reversal or scrubbing is a separate condition and does not satisfy the monotone-clock comparison.

The [OSCI-Render demonstration](https://polarity.me/posts/polarity-music/2024-12-31-osci-render-paint-with-sound-on-your-oscilloscope/) illustrates a creator manipulating an encoded visual shape with frequency and effects. It motivates separating traversal from shape, but its audio-rate oscilloscope display is not evidence that a Julia seed, physical tracker, or audience will behave the same way.

## Proposed falsification sequence

Use the same start, source, frame convention, fixed renderer, and recorded controls for each matched comparison. Start with simple synthetic phase and known shapes, then repeat with causal musical modes. Keep a free-navigation baseline. Bound excursion, phase rate, map rate and transition duration; measure achieved peak acceleration and jerk as well as saturation. A bound on phase rate alone does not bound acceleration, work or tracking error, especially near high eccentricity or during map changes. Choose limits from the actual Physics and actuator budget, report them, and reject invalid chart or non-finite trials rather than silently clipping them.

1. Compare frozen `B`, a smooth evolving `B`, and fast bounded direct actions. Match available travel and peak speed where feasible, log remaining differences, and include a case where a moving phase gauge represents the identical reference. This checks that a claimed development effect is not merely a different parameterization or activity level.
2. Compare uniform anomaly and classical Kepler clocks on the same fixed ellipse, including `e = 0` as a control and a moderate eccentricity such as 0.6. Record desired phase, intrinsic reference speed, actual speed, lag and saturation. Add arc-length timing if constant-speed traversal is a relevant competitor.
3. Replay every target first as a kinematic reference and then through bounded Physics. Record intrinsic path error, phase error only where phase is identifiable, applied controls, work/energy ledger, boundary interactions, discontinuities and return to free navigation. Report failures and control effort, not just successful segments. Existing [expressive-dynamics protocol](189-expressive-dynamics-protocol.md) describes sample-aligned capture and separate rendered-motion checks.
4. Hold the trajectory and presentation fixed when checking renderer effects. Conversely, hold presentation fixed across trajectory comparisons and measure actual frame changes. For [#241](https://github.com/rhythmatician/FractalSync/issues/241), use blinded listener judgments and matched-activity, timing-shifted and visual-only controls. Ask whether people identify a relationship separately from whether they prefer the result. A clear cockpit trace is not evidence of a legible Julia image.
5. For manual playability, use the same drivers and practice budget across conditions, counterbalance order, and record completion, corrections, control movements, workload and subjective preference. For future learning difficulty, compare the dimension and update bandwidth of candidate actions and held-out imitation or control performance under a fixed training budget. Action count or a successful manual trial alone cannot establish learnability.

Freeze trials, bounds, metrics and exclusions before examining scores; include silence, low-confidence sources, phase reset, rebinding, abrupt input and anchor motion. Retain raw timestamps and sample-clock epochs. Compare held-out source and trajectory families, and report individual outcomes and uncertainty rather than a single favorite clip. If a comparison cannot match speed, travel or appearance, describe it as mixed and preserve the measurements.

## Ownership and open questions

Per [ADR 0001](../adr/0001-rust-first-parity.md), runtime-consumed clocks, binding semantics, target geometry and Physics behavior belong in `runtime-core`; browser and Python surfaces consume bindings. A future semantic change must follow the applicable version, golden-vector and parity rules. This note changes no runtime behavior.

The next decision depends on measured tracking and perception: whether any separation of gesture execution and development is useful to a performer, and which clock differences survive Physics and rendering. The equations alone answer neither question. Chart bounds, rate and work budgets, source continuity, trial assets and participant results remain to be established.
