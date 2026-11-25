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
| #159 | [Changing-metric dynamics and covariant steering](159-moving-metric-dynamics.md), [accumulated dynamics](159-accumulated-dynamics.md), [trajectory error and refinement](159-trajectory-error.md) | Actual phase-energy differential and finite-history energy accounting; ordered-step stability and ODE defect; residual/rounding propagation, simultaneous position/velocity domain control, and actual fixed-horizon grid-trajectory convergence; discrete steering consistency | Numerical histories stay in a certified phase region from an exact-curve margin. Convergence requires the stated error scaling and regional constants; numerical values, actual IEEE certificates, impulsive-history and combined steering convergence remain open. |
| #160 | [Distance geometry](160-distance-geometry.md), [checked radial profile](160-radial-profile.md), [AC paths and exact infimum](160-absolute-continuous-paths.md) | Unrestricted AC calibration and integrability; attained Shore-distance path minimum in both orientations; sharp Lipschitz constant; duration and kinetic-action bounds | The exact nearest-Shore result uses the actual path infimum. Its full segment equality classification is now also checked in the intrinsic-length extension. |
| #160 | [Pairwise metric and zero-floor limit](160-pairwise-metric-limit.md) | Actual AC path metric, reversal/concatenation, topology, properness and completeness; every fixed-pair zero-floor branch; quantitative same-component comparison | Different components and distinct Shore endpoints separate to infinite distance; same-component distances converge to the actual zero-floor infimum. |
| #160 | [Intrinsic curve length and minimizers](160-intrinsic-length.md) | Actual metric variation equals graph length; length-space property, constant-speed AC minimizers between every pair, and all nearest-Shore minimizers | Positive-floor pairwise infima are attained. Strict convexity classifies nearest-Shore paths as monotone AC nearest-segment traversals, including pauses and the constant case. |
| #160 | [Flat and curved Shore](160-flat-shore.md), [Euclidean flat intrinsic bridge](160-flat-intrinsic.md) | Actual flat coordinate connection/curvature commutator and Gram ratio; full AC equality of the explicit coordinate integral with Euclidean graph length, path infimum and metric variation | The flat coordinate metric and general Euclidean path-distance construction are now formally connected, including Shore-crossing cusps. The conventional global cut-locus argument remains separate. |
| #160 | [Smooth flat geodesics and Jacobi field](160-flat-geodesics.md) | Actual geodesic energy and tangential-momentum conservation; radial first integral; unit-speed Shore geodesic and parallel normal; actual normal Jacobi equation and a field with first positive zero pi | The field-level origin of the scale pi times epsilon is checked. Arbitrary AC minimizer regularity, bundled conjugacy, second-variation criteria and global shortcut classification are not inferred. |
| #160 | [Actual flat length second variation and pairwise shortcuts](160-flat-second-variation.md) | Literal sine-curve length derivatives, unit-interval Euclidean AC competitor and exact graph length; physical dilation and actual pairwise distance below L/epsilon for L>pi*epsilon | The strict physical shortcut is now proved for the actual path infimum. Equality-case minimality, general index forms, all-minimizer classification and the curved-Shore branch remain separate. |
| #160 | [Affine covariance](160-affine-covariance.md), [nonlinear covariance and curvature tensor](160-nonlinear-geometry.md), [moving frames and regularity](160-moving-frames.md) | Actual nonlinear covariance and extension-independent curvature; inverse-frame derivative, connection/curvature gauge laws and composition; C2-frame/C1-coefficient and C3-chart/C1-coefficient regularity; actual fiber-valued curvature tensor | Moving-frame overlap laws and sufficient transformed coefficient regularity are checked. Manifold/vector-bundle construction and atlas gluing remain separate. |
| #160 | [Curved coordinate geometry](160-curved-coordinate-geometry.md), [embedded normal neighborhoods](160-embedded-normal-neighborhoods.md), [derived normal radii](160-derived-normal-radii.md) | Actual Taylor and chord lower bounds derive an explicit normal radius from curve jets and remote separation; compact-injective curves supply separation; local normal patches give strip injectivity, unique feet and actual IFT distance/ruler/metric identification | The normal certificate is now derived under stated geometric bounds, and the chart bridge accepts a finite parameter patch. Numerical constants for concrete Shore representations, general reach and ambient connection/curvature transfer remain separate. |
| #161 | [Sensitivity preconditioning](161-sensitivity-preconditioning.md) | Arbitrary localized-template geometry; derivative-error transfer; unequal-exponent and regularization obstructions | Measure both normalized singular values. A scalar distance multiplier cannot cure divergent directional anisotropy; small image residual is not a derivative certificate. |
| #162 | [Normalized distance fields](162-normalized-distance-fields.md) | Checked recenter/rescale identity, local differential and quadratic remainder, computed-query provider/ruler error budget and crest bias; conventional sample envelopes, topology and blow-up counterexamples | The error bounds retain the actual query location and scale. They require certificates; stencil self-consistency does not supply them or certify connectivity. |
| #162/#159 | [Provider stencil certificates](162-provider-stencil-certificates.md) | Actual Taylor remainders and central-gradient/diagonal-Hessian/mixed-Hessian error bounds; correlated sampled-value budgets, actual-query perturbations, division perturbations and Euclidean component aggregation | The runtime stencil formulas now have checked sufficient error contracts at the actual positive step. Raster fidelity, actual derivative regularity and IEEE arithmetic budgets still require evidence. |
| #162 | [Concrete operational-field certificates](162-concrete-provider-certificates.md), [global evaluator and query correspondence](162-global-operational-field.md) | Concrete coherent-lattice joins through fourth order; global C4 for the actual floor-selected field; physical gradient/Hessian correspondence; ideal halo/row-major index bounds and cross-cell query stability | The mathematical global selection and jets are checked across cell boundaries. Reference-distance fidelity, machine floor/casts/masks, storage and IEEE arithmetic remain distinct obligations. |
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

The pinned package under `formal/` now contains **1,043 explicit theorem/lemma
declarations across 87 modules**: 989 in seventy-four result modules and 54 in
thirteen analysis-support modules. This adds 137 declarations in fourteen result
modules to the preceding 906-theorem increment. The count includes supporting lemmas; it is not a
count of independent research discoveries.
Support includes attributed upstream AC infrastructure and separate project
proofs for AC composition, vector FTC, derivative control, proper geodesics and
compact parametric integration.
The complete Lean 4.24.0 build passed with **2,657 jobs**. The transitive-axiom
audit passed for **2,225 project declarations**, including definitions and
generated helpers, with only `propext`, `Classical.choice` and `Quot.sound`.
The earlier 596-theorem increment installed the pinned optional LSP/MCP server and
passed its actual stdio smoke test with 22 tools. The official proof/build
skills were installed then as well. The
[formal inventory](../../formal/README.md) and
[verification record](verification.md) give the reproducible setup and exact
trust boundary.

The current explicit potential-force catalog and additional control contracts
are checked for #159. The exact-real dynamics extension now includes actual
ODE truncation and stability, conditional provider/rounding error propagation,
simultaneous numerical phase-region control, and a checked refinement limit
uniform over all grid points of each mesh. The actual global operational
quintic field is C4, with selected physical jets and ideal indices identified,
alongside cross-cell query bounds and the earlier stencil certificates.
Reference-field fidelity, certified numerical regional constants and actual
floating-point error budgets remain separate.
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
| Geometric interfaces | Construct manifold/vector-bundle connection and curvature objects and glue them over an atlas. Actual moving-frame gauge/composition laws, fiber-valued tensor covariance and sufficient nonlinear coefficient regularity are checked. |
| Embedded curved Shore | Compute jet and separation constants for concrete Shore representations, extend reach results to other embedding hypotheses, identify the ambient connection and transfer curved-coordinate curvature. A uniform radius and local distance/metric chart are now derived from the stated curve-jet/separation hypotheses. |
| Smooth minimizing geometry | Prove regularity and geodesic-ODE identification for the AC minimizers; develop general conjugacy/second-variation criteria, equality-case classification and global cut-locus arguments. The physical strict shortcut above pi*epsilon now holds for the actual pairwise infimum. |
| Curved-Shore bifurcation | Formalize the functional-analytic reduction, local branch and minimum arguments. The coordinate curvature calculation alone does not prove the bifurcation. |
| Actual providers and arithmetic | Establish reference-distance fidelity, storage/mask and machine index correspondence, actual query-rounding bounds and IEEE evaluation certificates. Coherent mathematical floor selection, global C4 jets, ideal halo/index bounds and cross-cell real-query stability are checked. |
| Full discrete dynamics | Compute usable numerical regional constants and actual provider/IEEE budgets, establish suitable reference-solution existence and margins for specific fields, and extend convergence to impulsive/combined steering dynamics and continuous interpolants. The ordered smooth-phase update now has noncircular domain control and uniform grid-trajectory convergence under stated certificates and scaling. |
| Optional authoring integration | Register the installed MCP server in a writable persistent client configuration. The existing managed configuration is read-only. |

The AC metric-length bridge, arbitrary-endpoint attainment, all-nearest-Shore
minimizer classification, and flat Euclidean specialization are complete at
their documented formal scope. The fixed-pair zero-floor results remain
pointwise statements; this increment adds neither uniform endpoint convergence
nor convergence of selected minimizing paths. Runtime behavior, training,
canonical parameters and provider contracts are unchanged. PR #164 remains a
draft, and the research issues remain open.
