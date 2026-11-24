# Verification record: first mathematical research increment

Verified 2026-11-24 against a checkout of FractalSync base commit
`55df760ff706210dcce3c88e79e476a45c5331f0`. All checks below ran in the supplied
Linux environment. This record distinguishes kernel-checked statements from
finite numerical evidence and conventional proofs.

## Lean

Lean 4.24.0, Mathlib commit `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`, and
the committed `formal/lake-manifest.json` were used. The combined command

```text
lake build --wfail
```

completed successfully with 1,528 Lake jobs, mostly precompiled dependencies.
The four project modules contain **28 explicit theorem declarations**:

| Module | Theorems |
| --- | ---: |
| DistanceGraph | 4 |
| SensitivityBounds | 4 |
| NormalizedDistanceFields | 8 |
| CovariantStatistics | 12 |

The command `lake env lean Audit.lean` passed for **47 project declarations**,
including definitions and generated support declarations. Every transitive
axiom dependency belonged to the allowlist
`propext`, `Classical.choice`, `Quot.sound`. No `sorryAx`, native-evaluation
axiom, or additional project axiom was accepted.

A negative control inserted a deliberately unapproved axiom into a scratch
copy of the audit. The audit rejected it with
`Unapproved axiom FractalSync.auditProbe in FractalSync.auditProbe`.
That scratch file was removed and is not part of the package.

The [theorem inventory](../../formal/README.md) states the exact hypotheses and
scope of these formal results. In particular, there is no claim that the full
radial-distance integral, the L2 template geometry, topology/blow-up arguments,
or coherence perturbation theorems have been formalized.

## Independent numerical checks

Ran each script using the repository's Python environment, Python 3.12.14.
The sensitivity experiment used NumPy 2.3.5 and SciPy 1.17.0; the other three
scripts use only the standard library. All random trials have fixed seeds.

| Experiment | Observed result |
| --- | --- |
| Distance primitive against independent log-coordinate adaptive quadrature | 49 cases passed; maximum absolute discrepancy `1.862e-12`. |
| Squared radial-density remainder | 2,000 cases passed; maximum scaled discrepancy `5.023e-16`. |
| Quantitative zero-floor bound in the explicit one-dimensional puncture model | 21 cases passed. |
| Asymmetric complex-template L2 metric against finite differences of the actual analytic template | Three tested scales; maximum relative error approximately `1.287e-10`. The nonzero mixed Gram coefficient was `0.5190493241172275`. |
| Scalar exponent obstruction in the two-dimensional fold model | Condition numbers `12.5`, `1250`, `125000` at distances `1e-2`, `1e-4`, `1e-6`, for every tested exponent `-0.5, 0, 0.5, 1, 2`. |
| Normalized chart transition and combined provider/ruler error budget | 2,000 trials passed; maximum chart discrepancy `2.243e-14`; no observed budget excess. |
| Signed-distance ambiguity | 81 identical exact model samples for regions with different connectivity; sharp constants, bias cancellation, and the distinct dyadic blow-up examples also passed. |
| Finite transport/coherence identities and perturbation bounds | 4,000 trials passed; maximum gauge discrepancy `3.301e-15`, maximum common-reference pairwise discrepancy `1.423e-14`. |
| Inconsistent triangle transport | Exact integer quadratic form `-3`; the invalid squared-coherence construction gives `-1/3`. |

The scripts test analytic models, formulas, and counterexamples. They do not
sample the actual Julia renderer or certify Mandelbrot membership, provider
error, numerical transport, floating-point Physics, or audiovisual preference.
Small floating-point discrepancies are reported as numerical evidence, not
formal error bounds.

## Proof review and publication status

The radial formula, its sharp additive constant, and the quantitative
same-component limit were independently reviewed during this work. The review
prompted explicit treatment of the minimizing curve's metric/functional lengths
and of kinetic length density at nonsmooth points. The final note includes both.
All four notes give assumptions, full conventional proofs for their stated
results, limitations, and prior-art references.

The prior-art search was targeted. It found exact prior art for the requested
phase-harmonic Lipschitz bound and strong prior art for the location-scale
hyperbolic structure and Lipschitz extension envelopes. It did not establish
publication-level novelty of this package. Issues #160--#163 remain research
issues; successful Lean verification is not a novelty certificate.

## CI scope

`lean-proofs.yml` builds the pinned formal package and runs the axiom audit for
formal-package changes. Its configuration was checked locally against the
pinned official Lean action's inputs. The commands above were executed locally;
this record does not claim an observed successful GitHub-hosted CI run.
No runtime or training code was changed, so the application test suite was not
run for this research-only increment.
