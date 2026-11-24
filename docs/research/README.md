# Mathematical research progress

This directory records a first proof-oriented investigation of issues
[#160](https://github.com/rhythmatician/FractalSync/issues/160),
[#161](https://github.com/rhythmatician/FractalSync/issues/161),
[#162](https://github.com/rhythmatician/FractalSync/issues/162), and
[#163](https://github.com/rhythmatician/FractalSync/issues/163).
The research is nonblocking and does not change the Rust runtime, observation
contract, primary experiment, reward, or first-training prerequisites.

## Results and decisions

| Issue | Conventional proof note | Main progress | Practical consequence |
| --- | --- | --- | --- |
| #160 | [Distance geometry](160-distance-geometry.md) | Exact shortest distance to an arbitrary closed Shore; closed primitive; sharp additive approximation; properness; quantitative zero-floor limit | Regularization changes cross-Shore geometry. Ideal radial segments provide analytic controls; the theorem does not override provider validity or Controls. |
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
publication status still needs a focused literature comparison.

The value of this increment is a reusable theorem package, explicit failure
cases, and narrower hypotheses for the empirical work. Potential independent
research remains in proving a transfer theorem for a nontrivial dynamical
observable, sharpening the regularized graph analysis, or establishing useful
certified provider-error guarantees. Those are research questions, not results
claimed by the current proof files.

## Relationship to the existing Lean issues

The small pinned package under `formal/` is an initial development-time proof
environment supporting these research results. It gives #157 a working starting
point and #159 reusable lemmas. It does not complete #157's initial metric
theorem package or the Lean LSP/MCP integration, and does not complete #159's
Phase A contracts. Aeneas/Charon (#158) is independent of this work.

Keep one proof package when those issues proceed. Extend its definitions and
inventory instead of creating parallel formal authorities. All four research
issues remain open for their explicitly stated remaining obligations.
