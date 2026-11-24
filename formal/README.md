# Lean research contracts

This is a development-time mathematical proof package. It currently contains
**28 proved theorem declarations** supporting research issues #160--#163, plus
their definitions and a transitive axiom audit. It is independent of Rust
execution, Python training, the browser, and Aeneas/Charon.

The conventional statements, proofs, counterexamples, and originality assessment
are in [the research notes](../docs/research/README.md). Those notes contain more
mathematics than the Lean files. The inventory below is the exact boundary of
what was machine checked in this increment.

## Reproducible environment

| Component | Pin used for verification |
| --- | --- |
| Lean | `leanprover/lean4:v4.24.0` |
| Mathlib | `f897ebcf72cd16f89ab4577d0c826cd14afaafc7` (v4.24.0) |
| Transitive Lean dependencies | `lake-manifest.json` |
| Installer used in this environment | Elan 4.4.0 |
| Authoring guidance consulted | `leanprover/skills`, commit `7d3da0282e7b724b07620e45cf212f2e05e19334`, especially `lean-proof` and `mathlib-build` |

Agents own installation and maintenance. If Elan is absent, obtain it from the
[official installer](https://github.com/leanprover/elan). Respect these pins when
reproducing the proofs; no latest-version assumption is required. From `formal/`:

```sh
elan toolchain install leanprover/lean4:v4.24.0
MATHLIB_NO_CACHE_ON_UPDATE=1 lake update
lake exe cache get Mathlib.Analysis.Normed.Module.Basic Mathlib.Analysis.Normed.Group.Basic Mathlib.Algebra.BigOperators.Fin Mathlib.Algebra.BigOperators.Ring.Finset Mathlib.Data.Complex.Basic Mathlib.Tactic.NormNum Mathlib.Tactic.FieldSimp Mathlib.Tactic.Linarith Mathlib.Tactic.Ring
lake build --wfail
lake env lean Audit.lean
```

The long cache command requests only the modules used here and their imports.
It uses Mathlib's precompiled cache. The initial whole-Mathlib download in the
working environment received HTTP 429 responses; limiting the request to these
roots and setting curl's `parallel-max = 4` and `retry-delay = 3` in a temporary
curl configuration completed successfully. No Mathlib source changes or server
changes were required. Ordinary environments can use `lake exe cache get`.

`lake-manifest.json` is committed so all resolved dependencies are reviewable.
The package treats warnings as errors. The GitHub workflow builds this package
and runs the same axiom audit only for changes touching `formal/` or its workflow;
it is not a training preflight requirement.

This initial package does not close #157: its foundational metric theorem
package and persistent Lean LSP/MCP integration remain outstanding. Extend
this package when that work proceeds, rather than creating a second package.

## Exact theorem inventory

### #160: pointwise distance-graph algebra

Module: [DistanceGraph.lean](FractalSync/DistanceGraph.lean).
All names below have prefix `FractalSync.DistanceGraph.`.
Traceability: [distance note equations (5), (11), (14)--(15)](../docs/research/160-distance-geometry.md),
`runtime-core/src/manifold.rs`, and the ruler derived from `GeometryJet` in
`runtime-core/src/geometry_provider.rs`.

| Theorem | Verified mathematical content |
| --- | --- |
| `radial_remainder` | Exact squared-density remainder `(speed^2-w^2)/(r^2+epsilon^2)` when the denominator is nonzero. |
| `radial_lower_bound` | The radial squared density is at most graph speed squared when `epsilon>0` and `w^2<=speed^2`. |
| `graph_scale_comparison` | Scale-only speed squared is at most graph speed squared, which is at most `(1+a^2)` times scale-only speed squared under the same hypotheses. |
| `regularization_sandwich` | `t^2(A+C) <= tA+t^2C <= A+C` for `0<=t<=1` and `A,C>=0`. |

**Not formalized here:** absolute continuity, nearest-point existence, curve
integration, the exact radial distance/closed primitive, completeness, existence
of minimizing curves, or the zero-floor limit. The checked algebra supports
their conventional proofs; it is not a substitute for those analytic obligations.

### #161: sensitivity transfer and scalar shear

Module: [SensitivityBounds.lean](FractalSync/SensitivityBounds.lean).
Names have prefix `FractalSync.`.
Traceability: [sensitivity note Theorems 2 and 4](../docs/research/161-sensitivity-preconditioning.md),
issues #150/#151; this is an analytic control model, not a second Julia renderer.

| Theorem | Verified mathematical content |
| --- | --- |
| `sensitivity_norm_transfer` | Model response bounds and a norm-bounded perturbation transfer to lower/upper bounds for the actual response. |
| `sensitivity_squared_transfer` | Squared bounds have constants `(lower-error)^2` and `(upper+error)^2`, with nonnegative input radius and nonnegative lower margin. |
| `location_scale_complete_square` | A scalar location-scale mixed term is removed by a shear, with Schur complement `c-b^2/A`, when `A` is nonzero. |
| `location_scale_schur_pos` | `A>0` and `A*c-b^2>0` imply a positive Schur complement. |

**Not formalized here:** differentiation of the L2 affine action, positive
definiteness of its full generator Gram matrix, the global hyperbolic isometry,
curvature, exponent asymptotics, or any property of the real Julia observable.
The scalar complete-square theorem is not presented as an arbitrary-dimensional
matrix theorem.

### #162: normalized-field algebra and Lipschitz bounds

Module: [NormalizedDistanceFields.lean](FractalSync/NormalizedDistanceFields.lean).
Names have prefix `FractalSync.NormalizedDistanceFields.`.
Traceability: [normalized-field note](../docs/research/162-normalized-distance-fields.md),
Map specification #108, and provider error/validity contract #145.

| Theorem | Verified mathematical content |
| --- | --- |
| `normalizedMap_chart_transition` | Exact recentering/rescaling identity for any scalar field on a real module, with nonzero scales. |
| `normalizedDifference_error_identity` | The fixed-scale query error is the difference of endpoint errors divided by scale. |
| `normalizedDifference_anchored_error_bound` | A certified anchored error increment bounds the normalized error at positive scale. |
| `normalizedDifference_endpoint_error_bound` | Separate endpoint error guarantees give their sum divided by positive scale. |
| `normalizedMap_add_constant` | A common additive field bias cancels at fixed scale. |
| `normalizedMap_lipschitz_bound` | An ideal 1-Lipschitz field yields a 1-Lipschitz Map in normalized window coordinates for positive scale. |
| `normalizedMap_zero` | Every normalized Map is anchored at zero. |
| `normalizedMap_value_bound` | Under the same ideal Lipschitz hypothesis, `abs(M(u)) <= norm(u)`. |

**Not formalized here:** the full combined provider/recomputed-ruler error
estimate, Hessian remainder bounds, optimal finite-sample envelopes, topology
counterexamples, or blow-up limits. No ideal Lipschitz hypothesis is asserted for
an arbitrary numerical provider.

### #163: finite transport algebra and an exact counterexample

Module: [CovariantStatistics.lean](FractalSync/CovariantStatistics.lean).
Names have prefix `FractalSync.CovariantStatistics.`.
Traceability: [covariant-statistics note](../docs/research/163-covariant-phase-harmonics.md),
transported gesture specification #124 and empirical probe #126.

| Theorem | Verified mathematical content |
| --- | --- |
| `commonReferenceTerm` | A pairwise common-reference term factors into transported coordinates. |
| `commonReferenceGram` | The finite double sum is the conjugate of a single sum times that sum. |
| `commonReferenceNonnegative` | Its real quadratic form is nonnegative. |
| `transportGaugeCancel` | Matching local frame and transport gauge factors cancel. |
| `harmonicTransportGaugeCancel` | The cancellation holds for every natural harmonic order, for an already-computed harmonic coefficient. |
| `momentGaugeCovariance` | A common visual reference rotation conjugately rotates the finite moment. |
| `momentGaugeNormSq` | A unit reference rotation preserves squared moment magnitude. |
| `triangleNegative` | The explicit triangle kernel has quadratic form exactly -3 on `(1,-1,1)`. |
| `triangleDiagonal` | The counterexample's diagonal entries are one. |
| `triangleSymmetric` | The counterexample is symmetric. |
| `triangleUnit` | Every entry of the counterexample has unit squared magnitude. |
| `triangleNotPositiveSemidefinite` | The counterexample fails the quadratic-form definition of positive semidefiniteness. |

**Not formalized here:** manifold parallel transport, the analytic phase-harmonic
definition and Lipschitz theorem, negative harmonic orders, the converse
PSD/holonomy criterion, epoch-pooling interval, perturbation estimates, or
perceptual fit. Common-reference positivity itself needs no unit-modulus
assumption; the gauge and magnitude theorems state their unit hypotheses.

## Trust boundary and audit

`Audit.lean` imports the umbrella module and traverses the checked environment's
transitive dependencies for every declaration in namespace `FractalSync`.
It rejects every axiom except `propext`, `Classical.choice`, and `Quot.sound`,
the standard classical Lean/Mathlib foundations used here. It also fails if the
namespace is empty. This detects imported or indirect `sorryAx` and additional
axioms, not merely source-text placeholders. Every proof module must be imported
by `FractalSync.lean` so the audit sees it.

The current audit covers 47 declarations, including the 28 stated theorems,
definitions, and compiler-generated support declarations. No `sorry`, `admit`,
new axiom, or native-evaluation axiom is accepted. Exact real arithmetic is
distinct from IEEE 754 computation, and the formal statements do not certify
an implementation's correspondence to their definitions.

## Independent numerical checks

The four scripts under `checks/` are standalone mathematical experiments.
They are not imported by runtime, training, or bindings. Three need only the
Python standard library; `check_sensitivity_models.py` additionally uses NumPy
and SciPy. The optional versions used here are pinned in `checks/requirements.txt`.

Use the project Python environment from the repository root:

```sh
python -m pip install -r formal/checks/requirements.txt
python formal/checks/check_distance_geometry.py
python formal/checks/check_sensitivity_models.py
python formal/checks/check_normalized_distance_fields.py
python formal/checks/check_covariant_statistics.py
```

Activate the repository's environment or replace `python` with its explicit
interpreter path. Numerical results and the observed verification scope are
recorded in [verification.md](../docs/research/verification.md).
