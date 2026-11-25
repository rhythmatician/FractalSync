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
| #159 | [Local trajectories and continuous interpolation](159-local-trajectories.md), [quantitative existence](159-quantitative-dynamics.md) | Actual local solution, regional uniqueness and reference tube; explicit horizon R/[4(B+1)] from supplied phase-ball constants; sufficient mesh inequality for continuous interpolation | A local reference is constructed at a specified positive horizon. The same constants control every interpolated time, and the mesh inequality proves domain membership. |
| #159/#162 | [Operational ruler bounds](159-operational-ruler-bounds.md), [mechanical bounds](159-operational-mechanical-bounds.md), [finite-asset trajectory certificate](159-operational-trajectory-certificates.md) | Finite coefficient evidence gives actual ruler/gradient/Hessian norm and variation bounds, explicit mechanical acceleration constants, and an integrated actual-solution/refinement certificate | Local trajectory bounds now use explicit real formulas and a physical halo margin. Concrete asset/force evidence, actual IEEE error budgets and hybrid evolution remain separate. |
| #160 | [Distance geometry](160-distance-geometry.md), [checked radial profile](160-radial-profile.md), [AC paths and exact infimum](160-absolute-continuous-paths.md) | Unrestricted AC calibration and integrability; attained Shore-distance path minimum in both orientations; sharp Lipschitz constant; duration and kinetic-action bounds | The exact nearest-Shore result uses the actual path infimum. Its full segment equality classification is now also checked in the intrinsic-length extension. |
| #160 | [Pairwise metric and zero-floor limit](160-pairwise-metric-limit.md) | Actual AC path metric, reversal/concatenation, topology, properness and completeness; every fixed-pair zero-floor branch; quantitative same-component comparison | Different components and distinct Shore endpoints separate to infinite distance; same-component distances converge to the actual zero-floor infimum. |
| #160 | [Zero-floor geodesics and selected minimizers](160-zero-floor-geodesics.md) | Actual zero-floor infimum attained by an AC constant-speed path; uniform subsequence convergence for every selected constant-speed regularized family; positive clearance; selected curves' actual zero-floor lengths converge without a speed premise | The fixed-pair limit now includes minimizing-path compactness and attainment. Whole-selection convergence, velocity convergence and uniform varying-endpoint limits are not asserted. |
| #160 | [Intrinsic curve length and minimizers](160-intrinsic-length.md) | Actual metric variation equals graph length; length-space property, constant-speed AC minimizers between every pair, and all nearest-Shore minimizers | Positive-floor pairwise infima are attained. Strict convexity classifies nearest-Shore paths as monotone AC nearest-segment traversals, including pauses and the constant case. |
| #160 | [Flat and curved Shore](160-flat-shore.md), [Euclidean flat intrinsic bridge](160-flat-intrinsic.md) | Actual flat coordinate connection/curvature commutator and Gram ratio; full AC equality of the explicit coordinate integral with Euclidean graph length, path infimum and metric variation | The flat coordinate metric and general Euclidean path-distance construction are now formally connected, including Shore-crossing cusps. The conventional global cut-locus argument remains separate. |
| #160 | [Smooth flat geodesics and Jacobi field](160-flat-geodesics.md) | Actual geodesic energy and tangential-momentum conservation; radial first integral; unit-speed Shore geodesic and parallel normal; actual normal Jacobi equation and a field with first positive zero pi | The field-level origin of the scale pi times epsilon is checked. Arbitrary AC minimizer regularity, bundled conjugacy, second-variation criteria and global shortcut classification are not inferred. |
| #160 | [Actual flat length second variation and pairwise shortcuts](160-flat-second-variation.md) | Literal sine-curve length derivatives, unit-interval Euclidean AC competitor and exact graph length; physical dilation and actual pairwise distance below L/epsilon for L>pi*epsilon | The actual strict shortcut supplies the upper branch of the exact threshold. General index forms, minimizer classification above the threshold and the curved-Shore branch remain separate. |
| #160 | [Flat global minimum and exact threshold](160-flat-global-minimum.md) | Global AC calibration; actual distance L/epsilon for 0<=L<=pi*epsilon; all minimizers are monotone AC Shore traversals, including pauses and equality; strict shortening iff L>pi*epsilon | The threshold and its endpoint equality case hold for the full path infimum at every positive floor and tangential offset. Exactly two minimizers above threshold and the global cut locus remain conventional. |
| #160 | [Affine covariance](160-affine-covariance.md), [nonlinear covariance and curvature tensor](160-nonlinear-geometry.md), [moving frames and regularity](160-moving-frames.md) | Actual nonlinear covariance and extension-independent curvature; inverse-frame derivative, connection/curvature gauge laws and composition; C2-frame/C1-coefficient and C3-chart/C1-coefficient regularity; actual fiber-valued curvature tensor | Moving-frame overlap laws and sufficient transformed coefficient regularity are checked. Manifold/vector-bundle construction and atlas gluing remain separate. |
| #160 | [Curved coordinate geometry](160-curved-coordinate-geometry.md), [embedded normal neighborhoods](160-embedded-normal-neighborhoods.md), [derived normal radii](160-derived-normal-radii.md) | Actual Taylor and chord lower bounds derive an explicit normal radius from curve jets and remote separation; compact-injective curves supply separation; local normal patches give strip injectivity, unique feet and actual IFT distance/ruler/metric identification | The normal certificate is derived under stated geometric bounds, and the chart bridge accepts a finite parameter patch. Numerical constants for concrete Shore representations and broader reach results remain separate. |
| #160 | [Actual ambient Shore connection and curvature](160-ambient-shore-connection.md) | Ruler smoothness from the inverse normal chart; actual-gradient connection identification by Koszul uniqueness; curvature-commutator and metric-Gram transfer; embedded curvature formula and K=1 on Shore | The ambient metric connection is connected to the coordinate calculation under explicit normal-patch, Frenet and higher-regularity hypotheses. No global smooth distance, atlas gluing or minimizer regularity is inferred. |
| #161 | [Sensitivity preconditioning](161-sensitivity-preconditioning.md) | Arbitrary localized-template geometry; derivative-error transfer; unequal-exponent and regularization obstructions | Measure both normalized singular values. A scalar distance multiplier cannot cure divergent directional anisotropy; small image residual is not a derivative certificate. |
| #162 | [Normalized distance fields](162-normalized-distance-fields.md) | Checked recenter/rescale identity, local differential and quadratic remainder, computed-query provider/ruler error budget and crest bias; conventional sample envelopes, topology and blow-up counterexamples | The error bounds retain the actual query location and scale. They require certificates; stencil self-consistency does not supply them or certify connectivity. |
| #162/#159 | [Provider stencil certificates](162-provider-stencil-certificates.md) | Actual Taylor remainders and central-gradient/diagonal-Hessian/mixed-Hessian error bounds; correlated sampled-value budgets, actual-query perturbations, division perturbations and Euclidean component aggregation | The runtime stencil formulas now have checked sufficient error contracts at the actual positive step. Raster fidelity, actual derivative regularity and IEEE arithmetic budgets still require evidence. |
| #162 | [Concrete operational-field certificates](162-concrete-provider-certificates.md), [global evaluator and query correspondence](162-global-operational-field.md) | Concrete coherent-lattice joins through fourth order; global C4 for the actual floor-selected field; physical gradient/Hessian correspondence; ideal halo/row-major index bounds and cross-cell query stability | Mathematical global selection and jets are checked across cell boundaries. Finite-asset halo validity, machine floor/casts/masks, storage and IEEE arithmetic still require implementation evidence. |
| #162 | [Operational-field reference fidelity](162-operational-reference-fidelity.md) | Actual Taylor and selected-sample certificates give value, differential and Hessian error bounds; raw-sample quadratic value bias retained; explicit cross-cell Hessian query gain; combined reference/query/reported-result budgets | The reference-jet error contract is now proved for the literal global field. Sample accuracy, reference smoothness, query-rounding and IEEE discrepancy bounds remain evidence obligations; finite queries must stay within the certified halo. |
| #162 | [Physical halo and finite-asset locality](162-operational-halo.md) | Explicit four-face physical margin, phase/query displacement criterion, all 36 mathematical addresses, bounded zero extension and actual gradient/Hessian/acceleration germ independence | Only finite stored coefficient evidence is needed for the local mechanics bounds. Cell crossings are allowed; machine normalization, casts, masks, storage and displacement budgets still need evidence. |
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

The pinned package under `formal/` now contains **1,256 explicit theorem/lemma
declarations across 112 modules**: 1,201 in ninety-eight result modules and 55 in
fourteen analysis-support modules. This adds 87 declarations in nine result
modules and one support module to the preceding 1,169-declaration package.
The count includes supporting lemmas; it is not a
count of independent research discoveries.
Support includes attributed upstream AC infrastructure and separate project
proofs for AC composition, vector FTC, derivative control, proper geodesics and
compact parametric integration and uniform path compactness.
The complete Lean 4.24.0 build passed with **2,695 jobs**. The transitive-axiom
audit passed for **2,697 project declarations**, including definitions and
generated helpers, with only `propext`, `Classical.choice` and `Quot.sound`.
The earlier 596-theorem increment installed the pinned optional LSP/MCP server and
passed its actual stdio smoke test with 22 tools. The official proof/build
skills were installed then as well. The
[formal inventory](../../formal/README.md) and
[verification record](verification.md) give the reproducible setup and exact
trust boundary.

The current explicit potential-force catalog and additional control contracts
are checked for #159. The exact-real operational dynamics now have explicit
regional norm and Lipschitz formulas, a constructed positive-time reference,
a sufficient mesh criterion, and whole-interpolant/query domain control from
finite stored coefficient evidence and a physical halo margin. Force bounds
and mesh-scaled residual, rounding and initial-error contracts remain inputs.
The global operational quintic field is C4, with actual Euclidean ruler jets,
selected physical jets and ideal indices identified. Concrete sample and force
evidence, machine query/evaluation error budgets and hybrid evolution remain
separate. For #160, the actual zero-floor path infimum is now attained, every
constant-speed regularized minimizing selection has a uniform subsequential
limit, and every AC minimizing selection has convergent actual zero-floor
lengths. No smoothness of arbitrary AC minimizers is inferred.
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
| Embedded curved Shore | Compute jet and separation constants for concrete Shore representations and extend reach results to other embedding hypotheses or weaker regularity. The actual ruler, ambient connection and curvature transfer are checked under the documented normal-patch, Frenet and higher-regularity conditions. |
| Smooth minimizing geometry | Prove regularity and geodesic-ODE identification for arbitrary attained AC minimizers; develop general conjugacy/second-variation criteria, all-minimizer classification above the flat threshold and global cut-locus arguments. The exact flat threshold and every minimizing AC path at or below it are checked. |
| Curved-Shore bifurcation | Formalize the functional-analytic reduction, local branch and minimum arguments. The coordinate curvature calculation alone does not prove the bifurcation. |
| Actual providers and arithmetic | Supply sample-fidelity and reference-regularity evidence for concrete assets, certified stored coefficient/force values, actual query-rounding budgets, storage/mask and machine index correspondence, and IEEE evaluation certificates. Literal value/jet bounds, real-query margins and mathematical finite-halo address control are checked. |
| Full discrete dynamics | Instantiate the explicit regional constants and mesh inequality with concrete asset/force/arithmetic evidence; extend refinement to impulse jumps, time-dependent controls and combined covariant steering. The actual local solution, regional uniqueness, finite-asset phase/query control and continuous-interpolant refinement are checked. Global continuation and derivative convergence remain separate. |
| Zero-floor minimizing paths | Establish additional uniqueness or selection hypotheses for whole-family path convergence, convergence of velocities if needed, and uniform limits over varying endpoints. Actual componentwise zero-floor attainment, uniform subsequential path convergence and full-sequence unregularized length convergence are checked. |
| Optional authoring integration | Register the installed MCP server in a writable persistent client configuration. The existing managed configuration is read-only. |

The AC metric-length bridge, arbitrary-endpoint attainment, all-nearest-Shore
minimizer classification, flat Euclidean specialization, and flat Shore
classification at and below the threshold are complete at their documented
formal scope. Fixed-pair zero-floor limits now have the minimizing-path
continuation described above. Uniform convergence over varying endpoint pairs
and convergence of an arbitrary whole selected path family remain separate.
Runtime behavior, training,
canonical parameters and provider contracts are unchanged. PR #164 remains a
draft, and the research issues remain open.
