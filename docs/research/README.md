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
| #159 | [Differential mechanics](159-differential-mechanics.md) | Actual field Hessians and scale identities; local compact connection with Koszul uniqueness; continuous work-energy and dissipation; normalized Map differential and quadratic residual | The local geometric coefficients now connect to actual calculus. Smoothness, derivative bounds, and the equation of motion remain explicit hypotheses. |
| #159 | [Explicit potentials and controls](159-potential-controls.md), [discrete steps](159-discrete-mechanics.md) | Actual Shore and secant-wall forces, full crest budget, momentum derivative, intrinsic drive/drag and candidate control contracts; frozen-metric step energy and sharp explicit-drag condition | Continuous energy and fixed-configuration algebra have checked assumptions. The changing-metric numerical step and floating provider still need separate correspondence proofs. |
| #160 | [Distance geometry](160-distance-geometry.md), [checked radial profile](160-radial-profile.md), [AC paths and exact infimum](160-absolute-continuous-paths.md) | Unrestricted AC calibration and integrability; attained Shore-distance path minimum in both orientations; sharp Lipschitz constant; duration and kinetic-action bounds | The exact nearest-Shore result uses the actual path infimum. Its full segment equality classification is now also checked in the intrinsic-length extension. |
| #160 | [Pairwise metric and zero-floor limit](160-pairwise-metric-limit.md) | Actual AC path metric, reversal/concatenation, topology, properness and completeness; every fixed-pair zero-floor branch; quantitative same-component comparison | Different components and distinct Shore endpoints separate to infinite distance; same-component distances converge to the actual zero-floor infimum. |
| #160 | [Intrinsic curve length and minimizers](160-intrinsic-length.md) | Actual metric variation equals graph length; length-space property, constant-speed AC minimizers between every pair, and all nearest-Shore minimizers | Positive-floor pairwise infima are attained. Strict convexity classifies nearest-Shore paths as monotone AC nearest-segment traversals, including pauses and the constant case. |
| #160 | [Flat and curved Shore](160-flat-shore.md) | Actual flat coordinate connection/curvature commutator and Gram ratio, plus scalar excursion analysis; conventional full planar cut locus from prior art and curved-Shore cubic-root branch | The explicit flat metric has checked coordinate curvature +1 on Shore. Tensor/frame interfaces, flat path-metric specialization, cut locus and curved-Shore bifurcation remain separate. |
| #161 | [Sensitivity preconditioning](161-sensitivity-preconditioning.md) | Arbitrary localized-template geometry; derivative-error transfer; unequal-exponent and regularization obstructions | Measure both normalized singular values. A scalar distance multiplier cannot cure divergent directional anisotropy; small image residual is not a derivative certificate. |
| #162 | [Normalized distance fields](162-normalized-distance-fields.md) | Checked recenter/rescale identity, local differential and quadratic remainder, computed-query provider/ruler error budget and crest bias; conventional sample envelopes, topology and blow-up counterexamples | The error bounds retain the actual query location and scale. They require certificates; stencil self-consistency does not supply them or certify connectivity. |
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

The pinned package under `formal/` now contains 596 explicit theorem/lemma
declarations: 543 in thirty-nine result modules and 53 in twelve analysis-support
modules. This adds 196 declarations to the preceding 400-theorem increment.
Support includes attributed upstream AC infrastructure and separate project
proofs for AC composition, vector FTC, derivative control and proper geodesics.
The complete Lean 4.24.0 build passed with 2,619 jobs and the audit passed for
1,094 declarations, allowing only `propext`, `Classical.choice` and `Quot.sound`.
The pinned optional LSP/MCP server was reinstalled and its actual stdio smoke
test passed with 22 tools. The official proof/build skills were installed. The
[formal inventory](../../formal/README.md) and
[verification record](verification.md) give the reproducible setup and exact
trust boundary.

The current explicit potential-force catalog and additional control contracts
are checked for #159, with the continuous and frozen-metric numerical scopes
kept distinct. The combined recomputed-ruler provider bound is also checked,
but actual numerical value/derivative certificates remain to be established.
For #157, persistent Codex MCP registration was attempted and blocked by the
read-only active client configuration; installed tools and passing stdio
tests do not silently close that remaining integration step.
Aeneas/Charon (#158) remains independent.

Keep one proof package when those issues proceed. Extend its definitions and
inventory instead of creating parallel formal authorities. All four research
issues remain open for their explicitly stated remaining obligations.
