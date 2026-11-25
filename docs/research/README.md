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

| Issue | Research note | Main progress | Practical consequence |
| --- | --- | --- | --- |
| #157/#159 | [Metric foundations](157-scale-metric.md) | Checked rank-one positive metric, two-sided inverse, arbitrary-dimensional determinant, conditional anisotropy, plane rotations, and exact impulse-energy bound | One shared metric supports the downstream proofs. Exact algebra does not certify floating-point thresholds or provider derivatives. |
| #159 | [Differential mechanics](159-differential-mechanics.md) | Actual field Hessians and scale identities; local compact connection with Koszul uniqueness; continuous work-energy and dissipation; normalized Map differential and quadratic residual | The local geometric coefficients now connect to actual calculus. Smoothness, derivative bounds, and the equation of motion remain explicit hypotheses. |
| #159 | [Explicit potentials and controls](159-potential-controls.md), [discrete steps](159-discrete-mechanics.md) | Actual Shore and secant-wall forces, full crest budget, momentum derivative, intrinsic drive/drag and candidate control contracts; frozen-metric step energy and sharp explicit-drag condition | Continuous energy and fixed-configuration algebra have checked assumptions. The separate changing-metric extension now handles the exact-real ordered step; executable provider and floating-point correspondence remain open. |
| #159 | [Changing-metric dynamics and covariant steering](159-moving-metric-dynamics.md), [accumulated dynamics](159-accumulated-dynamics.md) | Actual phase-energy differential, residual power and ordered-step defect; compact-region uniform regularity; finite-history energy/work accounting with updated-position impulses and displacement budgets; a discrete steering candidate with exact endpoint norm and actual first-order consistency | Conditional finite histories now have checked energy and domain bounds. Numerical regularity constants, velocity-region persistence, trajectory convergence and IEEE/runtime correspondence remain open. |
| #160 | [Distance geometry](160-distance-geometry.md), [checked radial profile](160-radial-profile.md), [AC paths and exact infimum](160-absolute-continuous-paths.md) | Unrestricted AC calibration and integrability; attained Shore-distance path minimum in both orientations; sharp Lipschitz constant; duration and kinetic-action bounds | The exact nearest-Shore result uses the actual path infimum. Its full segment equality classification is now also checked in the intrinsic-length extension. |
| #160 | [Pairwise metric and zero-floor limit](160-pairwise-metric-limit.md) | Actual AC path metric, reversal/concatenation, topology, properness and completeness; every fixed-pair zero-floor branch; quantitative same-component comparison | Different components and distinct Shore endpoints separate to infinite distance; same-component distances converge to the actual zero-floor infimum. |
| #160 | [Intrinsic curve length and minimizers](160-intrinsic-length.md) | Actual metric variation equals graph length; length-space property, constant-speed AC minimizers between every pair, and all nearest-Shore minimizers | Positive-floor pairwise infima are attained. Strict convexity classifies nearest-Shore paths as monotone AC nearest-segment traversals, including pauses and the constant case. |
| #160 | [Flat and curved Shore](160-flat-shore.md), [Euclidean flat intrinsic bridge](160-flat-intrinsic.md) | Actual flat coordinate connection/curvature commutator and Gram ratio; full AC equality of the explicit coordinate integral with Euclidean graph length, path infimum and metric variation | The flat coordinate metric and general Euclidean path-distance construction are now formally connected, including Shore-crossing cusps. The conventional global cut-locus argument remains separate. |
| #160 | [Smooth flat geodesics and Jacobi field](160-flat-geodesics.md) | Actual geodesic energy and tangential-momentum conservation; radial first integral; unit-speed Shore geodesic and parallel normal; actual normal Jacobi equation and a field with first positive zero pi | The field-level origin of the scale pi times epsilon is checked. Arbitrary AC minimizer regularity, bundled conjugacy, second-variation criteria and global shortcut classification are not inferred. |
| #160 | [Actual flat length second variation](160-flat-second-variation.md) | Literal sine-curve length integral, actual first and second amplitude derivatives, fixed endpoints, AC and strict shortening for every sufficiently small positive amplitude when normalized span exceeds pi | A concrete local-shortening result now follows from actual path length. It does not classify all minimizers or prove the equality threshold, general index form or curved-Shore branch. |
| #160 | [Affine covariance](160-affine-covariance.md), [nonlinear covariance and curvature tensor](160-nonlinear-geometry.md) | Actual chart-Hessian connection law and full nonlinear commutator covariance; continuous trilinear tensor and field-extension independence; complete flat tensor and arbitrary regular-frame curvature ratio | The flat curvature is independent of regular field extensions, nondegenerate frames and locally invertible nonlinear coordinates. Manifold bundle/atlas construction and moving-frame connection-form laws remain separate. |
| #160 | [Curved coordinate geometry](160-curved-coordinate-geometry.md), [embedded normal neighborhoods](160-embedded-normal-neighborhoods.md) | Curved-coordinate connection and curvature; quantitative normal-chord distance/uniqueness certificate with concrete sphere instance; actual IFT normal chart, ambient distance/ruler derivative and metric pullback | Embedded distance and metric identification now follow from explicit geometric and Frenet hypotheses. Deriving a generic normal radius and transferring the ambient connection/curvature remain separate, as does bifurcation. |
| #161 | [Sensitivity preconditioning](161-sensitivity-preconditioning.md) | Arbitrary localized-template geometry; derivative-error transfer; unequal-exponent and regularization obstructions | Measure both normalized singular values. A scalar distance multiplier cannot cure divergent directional anisotropy; small image residual is not a derivative certificate. |
| #162 | [Normalized distance fields](162-normalized-distance-fields.md) | Checked recenter/rescale identity, local differential and quadratic remainder, computed-query provider/ruler error budget and crest bias; conventional sample envelopes, topology and blow-up counterexamples | The error bounds retain the actual query location and scale. They require certificates; stencil self-consistency does not supply them or certify connectivity. |
| #162/#159 | [Provider stencil certificates](162-provider-stencil-certificates.md) | Actual Taylor remainders and central-gradient/diagonal-Hessian/mixed-Hessian error bounds; correlated sampled-value budgets, actual-query perturbations, division perturbations and Euclidean component aggregation | The runtime stencil formulas now have checked sufficient error contracts at the actual positive step. Raster fidelity, actual derivative regularity and IEEE arithmetic budgets still require evidence. |
| #162 | [Concrete operational-field certificates](162-concrete-provider-certificates.md) | Actual runtime quintic coefficient polynomials and Horner jets; physical-coordinate differential and Hessian; quadratic sampling bias with exact jets; explicit coefficient-to-value and derivative sensitivity | The fixed-cell exact-real operational polynomial now has concrete certificates. Shore-distance fidelity, grid/cell-selection correspondence and IEEE evaluation remain distinct obligations. |
| #163 | [Covariant phase statistics](163-covariant-phase-harmonics.md) | Common-reference equivalence; holonomy/PSD obstruction; unknown-epoch pooling range; perturbation bounds | Use actual-trajectory transport. Avoid duplicate covariant ablations and unsupported squared-coherence or cross-epoch interpretations. |

Each note states assumptions and distinguishes kernel-checked claims from
conventional proofs and unresolved obligations. The
[formal theorem inventory](../../formal/README.md) identifies the exact subset
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

The pinned package under `formal/` now contains **906 explicit theorem/lemma
declarations across 73 modules**: 852 in sixty result modules and 54 in
thirteen analysis-support modules. This adds 180 declarations to the preceding
726-theorem increment. The count includes supporting lemmas; it is not a
count of independent research discoveries.
Support includes attributed upstream AC infrastructure and separate project
proofs for AC composition, vector FTC, derivative control, proper geodesics and
compact parametric integration.
The complete Lean 4.24.0 build passed with **2,643 jobs**. The transitive-axiom
audit passed for **1,844 project declarations**, including definitions and
generated helpers, with only `propext`, `Classical.choice` and `Quot.sound`.
The earlier 596-theorem increment installed the pinned optional LSP/MCP server and
passed its actual stdio smoke test with 22 tools. The official proof/build
skills were installed then as well. The
[formal inventory](../../formal/README.md) and
[verification record](verification.md) give the reproducible setup and exact
trust boundary.

The current explicit potential-force catalog and additional control contracts
are checked for #159. The changing-metric exact-real contracts now include
residual-aware finite histories, existence of uniform regularity on compact
smooth phase regions, whole-history position-domain budgets, and a discrete
steering candidate with a checked first derivative. The concrete operational
quintic polynomial has actual value/jet and coefficient-sensitivity results,
alongside the earlier conditional stencil bounds. Establishing reference-field
fidelity and actual numerical error certificates remains separate.
For #157, persistent Codex MCP registration was attempted and blocked by the
read-only active client configuration; installed tools and passing stdio
tests do not silently close that remaining integration step.
Aeneas/Charon (#158) remains independent.

Keep one proof package when those issues proceed. Extend its definitions and
inventory instead of creating parallel formal authorities. All four research
issues remain open for their explicitly stated remaining obligations.

## Remaining formal and implementation boundaries

| Area | Remaining work |
| --- | --- |
| Geometric interfaces | Manifold-level connection/curvature bundles and atlas gluing; moving-frame connection-form laws; coefficient regularity under sufficiently smooth coordinate changes. Nonlinear commutator covariance, a continuous trilinear coordinate tensor, extension independence and arbitrary flat-frame curvature are checked. |
| Embedded curved Shore | Derive uniform normal-chord radii for appropriate embedded curves and global separation hypotheses; identify the ambient connection and transfer curved-coordinate curvature. Actual local charts, distance identification and metric pullback under the explicit certificate are checked. |
| Smooth minimizing geometry | Prove regularity and geodesic-ODE identification for the AC minimizers; develop general conjugacy/second-variation criteria, endpoint classification and global cut-locus arguments. The explicit Shore Jacobi field and actual sine-family local shortening beyond normalized span pi are checked. |
| Curved-Shore bifurcation | Formalize the functional-analytic reduction, local branch and minimum arguments. The coordinate curvature calculation alone does not prove the bifurcation. |
| Actual providers and arithmetic | Establish reference-distance fidelity, inter-cell/global evaluator correspondence, actual query-rounding bounds and IEEE evaluation certificates. The concrete fixed-cell polynomial jets and coefficient-sensitivity constants are checked. |
| Full discrete dynamics | Compute usable regional regularity constants, prove a priori phase-region persistence including velocity bounds, propagate certified IEEE/provider errors and control trajectory error. Conditional finite-history energy/domain estimates and an exact-real discrete steering candidate are checked. |
| Optional authoring integration | Register the installed MCP server in a writable persistent client configuration. The existing managed configuration is read-only. |

The AC metric-length bridge, arbitrary-endpoint attainment, all-nearest-Shore
minimizer classification, and flat Euclidean specialization are complete at
their documented formal scope. The fixed-pair zero-floor results remain
pointwise statements; this increment adds neither uniform endpoint convergence
nor convergence of selected minimizing paths. Runtime behavior, training,
canonical parameters and provider contracts are unchanged. PR #164 remains a
draft, and the research issues remain open.
