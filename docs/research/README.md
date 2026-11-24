# Mathematical research progress

This directory records proof-oriented work on foundational issues
[#157](https://github.com/rhythmatician/FractalSync/issues/157) and
[#159](https://github.com/rhythmatician/FractalSync/issues/159), and research issues
[#160](https://github.com/rhythmatician/FractalSync/issues/160),
[#161](https://github.com/rhythmatician/FractalSync/issues/161),
[#162](https://github.com/rhythmatician/FractalSync/issues/162), and
[#163](https://github.com/rhythmatician/FractalSync/issues/163).
The research is nonblocking and does not change the Rust runtime, observation
contract, primary experiment, reward, or first-training prerequisites.

## Results and decisions

| Issue | Conventional proof note | Main progress | Practical consequence |
| --- | --- | --- | --- |
| #157/#159 | [Metric foundations](157-scale-metric.md) | Checked rank-one positive metric, two-sided inverse, arbitrary-dimensional determinant, conditional anisotropy, plane rotations, and exact impulse-energy bound | One shared metric supports the downstream proofs. Exact algebra does not certify floating-point thresholds or provider derivatives. |
| #160 | [Distance geometry](160-distance-geometry.md), [checked radial profile](160-radial-profile.md) | Checked closed primitive, optimal additive constant, exact logarithmic asymptotics, nearest-point segments, and integrated calibration under explicit trace hypotheses; conventional properness and componentwise zero-floor limit | Regularization changes cross-Shore geometry. Ideal nearest segments provide analytic controls; a restricted validity domain can exclude them. |
| #160 | [Flat and curved Shore](160-flat-shore.md) | Checked scalar curvature calculations, strict excursion-span monotonicity and unique radius; conventional full planar cut locus from prior art, threshold shortcut asymptotics, and a local curved-Shore cubic-root branch | The induced graph has positive curvature at smooth Shore points. The flat shortcut threshold is pi epsilon; curved-Shore response near it is singular. |
| #161 | [Sensitivity preconditioning](161-sensitivity-preconditioning.md) | Arbitrary localized-template geometry; derivative-error transfer; unequal-exponent and regularization obstructions | Measure both normalized singular values. A scalar distance multiplier cannot cure divergent directional anisotropy; small image residual is not a derivative certificate. |
| #162 | [Normalized distance fields](162-normalized-distance-fields.md) | Exact recenter/rescale identity; correlated provider-and-ruler error budget; Lipschitz sample envelopes; topology and blow-up counterexamples | Overlapping Map views have exact algebraic consistency laws. Finite sample accuracy does not certify connectivity. |
| #163 | [Covariant phase statistics](163-covariant-phase-harmonics.md) | Common-reference equivalence; holonomy/PSD obstruction; unknown-epoch pooling range; perturbation bounds | Use actual-trajectory transport. Avoid duplicate covariant ablations and unsupported squared-coherence or cross-epoch interpretations. |

Each note states assumptions and gives conventional proofs. The
[formal theorem inventory](../../formal/README.md) identifies the smaller subset
actually verified by Lean. The independent numerical checks in `formal/checks/`
exercise analytic models and counterexamples; they do not certify the current
GeometryProvider, Julia renderer, Controls, or floating-point implementation.

## Originality verdict

No independently publishable new result has been established in this pass.
The phase-harmonic Lipschitz bound is explicitly in the cited literature.
Location-scale hyperbolic geometry, Lipschitz extension envelopes, and much of
the covariance algebra also have direct prior art. The exact regularized graph
formulas and quantitative limits have been derived for this family, but their
publication status still needs a focused literature comparison. The complete
flat cut locus is a specialization of prior results for reflective warped
surfaces, explicitly attributed in its note. The local curved-Shore calculation
uses a standard Lyapunov--Schmidt argument; no publication-level novelty is
asserted for it.

The value of this increment is a reusable theorem package, explicit failure
cases, and narrower hypotheses for the empirical work. Potential independent
research remains in proving a transfer theorem for a nontrivial dynamical
observable, sharpening the regularized graph analysis, or establishing useful
certified provider-error guarantees. Those are research questions, not results
claimed by the current proof files.

## Relationship to the existing Lean issues

The pinned package under `formal/` now contains 172 proved theorem/lemma
declarations across eleven modules. Its transitive audit permits only the
standard classical Lean/Mathlib axioms. Lean 4, Lake, cached Mathlib and a pinned
optional LSP/MCP server were exercised locally; the reproducible setup and
precise tool boundary are in [formal/README.md](../../formal/README.md).

The core static metric results substantially advance #157. Its remaining
client-specific authoring setup and differential kinetic identities are not
silently marked complete. For #159, the ruler's first derivative and scalar
second-derivative coefficient, the exact impulse energy jump and its dual-norm
budget bound are checked. The full Hessian, sigma derivatives, connection,
continuous work-energy identity and other Phase A obligations remain open.
Aeneas/Charon (#158) is independent of this work.

Keep one proof package when those issues proceed. Extend its definitions and
inventory instead of creating parallel formal authorities. All four research
issues remain open for their explicitly stated remaining obligations.
