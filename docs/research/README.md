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

The [dynamical-ruler decision](182-dynamical-ruler-decision.md) keeps the
signed-distance graph baseline after matched Physics and frozen-renderer
comparisons. The [attachment crossing](183-attachment-crossing-fixture.md)
and [Julia sensitivity probe](151-frozen-julia-sensitivity.md) give the
reproduction commands and quantitative results for those experiments.

| Issue | Research note | Main progress | Practical consequence |
| --- | --- | --- | --- |
| #157/#159 | [Metric foundations](157-scale-metric.md) | Checked rank-one positive metric, two-sided inverse, arbitrary-dimensional determinant, conditional anisotropy, plane rotations, and exact impulse-energy bound | One shared metric supports the downstream proofs. Exact algebra does not certify floating-point thresholds or provider derivatives. |
| #159 | [Differential mechanics](159-differential-mechanics.md) | Actual field Hessians and scale identities; local compact connection with Koszul uniqueness; continuous work-energy and dissipation; normalized Map differential and quadratic residual | The local geometric coefficients now connect to actual calculus. Smoothness, derivative bounds, and the equation of motion remain explicit hypotheses. |
| #159 | [Explicit potentials and controls](159-potential-controls.md), [discrete steps](159-discrete-mechanics.md) | Actual Shore and secant-wall forces, full crest budget, momentum derivative, intrinsic drive/drag and candidate control contracts; frozen-metric step energy and sharp explicit-drag condition | Continuous energy and fixed-configuration algebra have checked assumptions. The separate changing-metric extension now handles the exact-real ordered step; executable provider and floating-point correspondence remain open. |
| #159 | [Changing-metric dynamics and covariant steering](159-moving-metric-dynamics.md), [accumulated dynamics](159-accumulated-dynamics.md), [trajectory error and refinement](159-trajectory-error.md) | Actual phase-energy differential and finite-history energy accounting; ordered-step stability and ODE defect; residual/rounding propagation, simultaneous position/velocity domain control, and actual fixed-horizon grid-trajectory convergence; discrete steering consistency | Numerical histories stay in a certified phase region from an exact-curve margin. Convergence requires the stated error scaling and regional constants; numerical values, actual IEEE certificates, impulsive-history and combined steering convergence remain open. |
| #159 | [Local trajectories and continuous interpolation](159-local-trajectories.md), [quantitative existence](159-quantitative-dynamics.md) | Actual local solution, regional uniqueness and reference tube; explicit horizon R/[4(B+1)] from supplied phase-ball constants; sufficient mesh inequality for continuous interpolation | A local reference is constructed at a specified positive horizon. The same constants control every interpolated time, and the mesh inequality proves domain membership. |
| #159/#162 | [Operational ruler bounds](159-operational-ruler-bounds.md), [mechanical bounds](159-operational-mechanical-bounds.md), [finite-asset trajectory certificate](159-operational-trajectory-certificates.md) | Finite coefficient evidence gives actual ruler/gradient/Hessian norm and variation bounds, explicit mechanical acceleration constants, and an integrated actual-solution/refinement certificate | Local trajectory bounds now use explicit real formulas and a physical halo margin. The concrete instance below discharges asset/force evidence for one fixture; actual IEEE error budgets and hybrid evolution remain separate. |
| #159/#162 | [Concrete catalog and trajectory instance](159-concrete-trajectory-certificate.md), [potential bounds](159-potential-force-bounds.md), [control bounds](159-control-force-bounds.md) | Derived actual force constants, literal nonconstant finite asset, constructed reference at an explicit rational horizon, all-time mesh/query control and an unconditional exact-real 1024-step run | The asset and force premises are discharged for this fixture. The horizon is extremely conservative; later rows give the improved horizon and exact-real RK extension. Production arithmetic and broader control evolution remain open. |
| #159/#162 | [Local fixture jets and longer horizon](159-local-horizon.md), [quadratic cancellation](162-local-fixture-bounds.md), [raised catalog](159-cancelled-mechanical-bounds.md) | Actual local ruler pairs (2,1), (1,20), (20,200), force (6,210), acceleration (8,400), constructed horizon 1/576 and unconditional 2048-step run | The former microscopic-horizon bottleneck is removed for this fixture; all-time phase error is at most 1/(36m), with the same query margins. |
| #159 | [Held-control residual](159-held-control-residual.md), [local energy bounds](159-local-energy-bounds.md) | Actual frozen-drive residual at most 680 times phase displacement plus 4 times initial covector error; actual energy differential (5,134) and finite Taylor ledger at most 1/(64m) | Holding requires stage/curve displacement evidence. The Taylor ledger retains differential increments; neither result certifies RK4, IEEE arithmetic or total energy drift. |
| #159 | [Complete held-control RK histories](159-held-runge-kutta.md) | Actual classical RK4 and any positive number of equal internal substeps; stage and endpoint domain bootstrap, actual residuals, all-time phase error<=3/(16m), explicit 16384-step target and uniform convergence | One and four internal substeps retain the same outer-anchor drive. All stage queries are covered; the next row adds signed work. IEEE evaluations, arbitrary caller inputs and provider acceptance remain separate. |
| #159 | [Signed RK work and energy ledger](159-rk-work-ledger.md) | Actual held-stage power quadrature; every-prefix energy/work defect<=1/(8mn); actual reference power integral; all-time work error<=1/(8mn)+35/(36m) and joint phase/work uniform convergence | The physical-work ledger is checked for the same fixture and held-drive policy. It preserves signed work and separates outer holding error from internal quadrature error. IEEE power/summation and compiled execution remain separate. |
| #159 | [Local RK operation errors](159-rk-operation-errors.md) | Individual RHS, constructed-stage, endpoint, power and work-update residuals imply stage containment, complete phase histories, prefix ledger<=1/(4m) and all-time work error<=11/(9m) | The final conditional bridge covers one RK substep per outer interval with nonzero finite-mesh allowances. Actual implementation evidence must establish the shrinking local error budgets. |
| #157/#159/#160 | [Field-replacement reuse boundary](157-field-replacement-boundary.md) | Source/signature inventory separates general ruler/metric/mechanics/numerical results, conditional field calculus, concrete evaluator evidence and actual distance-to-set geometry | The current formulation is retained. A replacement must supply its own field/jet/domain evidence; approximation or equal zero sets alone do not transfer distance calibration. |
| #159 | [Explicit phase-energy bounds](159-phase-energy-bounds.md) | Actual bundled energy differential, numerical norm/Lipschitz constants, and per-step Taylor remainder with domain premises discharged along the concrete finite run | The concrete energy remainder is at most 1/[400000000 m²] for m steps. The linear differential term remains, so this is not an energy-conservation or total-drift assertion. |
| #162 | [Concrete asset and reference fidelity](162-concrete-asset-instance.md) | Kernel-checked literal dyadic table, all 256 quadratic sample errors, actual full-halo value/first/second derivative errors, exact margins and Euclidean-gradient transfer | Actual fixture fidelity is now instantiated, retaining the raw-sample bias P-h². Production distance assets, machine query/IEEE errors and compiled extraction correspondence remain separate. |
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

## Foundation acceptance

| Issue | Evidence | Research disposition |
| --- | --- | --- |
| #157 | [Pinned Lean package and authoring setup](../../formal/README.md#reproducible-environment) | The proof build, audit, official skills, and project MCP configuration have reproducible checks. |
| #158 | [Aeneas and Charon spike](158-rust-verification-spike.md) | Defer integration. The isolated Rust fixtures compile, with extraction blocked before Lean generation. The report records the compiler and floating-point boundaries. |
| #159 | [Phase A acceptance](159-phase-a-acceptance.md) | The checked core geometry and mechanics contracts map to runtime test seams. Gesture and audiovisual work has separate follow-ups. |
| #160, #173 | [Distance graph disposition](160-distance-geometry.md#research-disposition), [fixed-pair examples](160-pairwise-metric-limit.md#synthetic-checks-for-the-fixed-pair-branches) | Keep the length geometry, nearest-Shore calibration, and zero-floor results as an engineering theorem package. The examples cover each fixed-pair branch. |
| #161 | [Sensitivity comparison](161-sensitivity-preconditioning.md), [response bounds](161-response-bounds.md) | Template, fold, and periodic-point examples complete the foundational comparison. Observable and Controls measurements remain separate. |
| #162 | [Normalized distance fields](162-normalized-distance-fields.md), [set limits and affine reconstruction](162-map-limits-and-reconstruction.md) | Keep the checked chart, error, limit, and restricted reconstruction results for numerical geometry. |
| #163 | [Holonomy criterion and examples](163-holonomy-acceptance.md) | A complete unit-phase kernel is PSD exactly when every triangle has trivial transport product. Empirical audiovisual fit remains with #126. |

## Geometry and steering studies

| Issues | Research note | Result |
| --- | --- | --- |
| #181, #167 | [Roundoff on a retained operational-field patch](181-operational-patch.md) | A rational error calculation bounds value, gradient, and Hessian error relative to the stored spline on one patch. A native Rust probe checks samples against those bounds. |
| #182, #184 | [Main-cardioid ruler](184-main-cardioid-ruler.md) | The reciprocal hyperbolic density matches true Shore distance to leading order along smooth boundary approaches. The ratio at the cusp depends on the approach. The note compares the candidate and existing rulers on the same paths. |
| #185 | [Fixed point-trap steering](185-orbit-trap-steering.md) | Lean checks the regular tangent and identifies the three critical points for the second critical iterate. A numerical comparison measures contour drift and near-critical angular sensitivity. |
| #184 | [Quadratic floor curvature](184-conformal-floor-curvature.md) | A quadratic floor raises conformal curvature. The main-cardioid cusp has path-dependent curvature limits, which rule out a positive $C^2$ metric extension there. |
| #185 | [Steering confidence](185-steering-confidence.md) | Lean checks a sharp angle gate for gradients with bounded error. The fixed-trap note derives sample position and binary64 evaluation budgets. |
| #184 | [Quartic ruler floor](184-quartic-ruler-floor.md) | Zero extension of the fourth power gives a positive $C^2$ conformal metric through the cusp. Curvature tends to zero at the cardioid boundary and can be positive in the interior. |
| #185 | [Finite-step trap drift](185-finite-step-trap-drift.md) | Lean checks per-step and cumulative drift bounds. The bound separates the effect of gradient error from the quadratic displacement remainder. |
| #184 | [Period-two bulb and attachment](184-period-two-attachment.md) | The exact bulb ruler and a local two-component quartic metric give a smooth crossing fixture. Offset paths expose a quadratic gap between the boundary arcs. |
| #185 | [Exact trap contours](185-exact-trap-contours.md) | Lean checks the exact lift, its tangent derivative, and its conformal speed. The lifted contour benchmarks Euler drift and distinguishes one-turn and two-turn closure. |
| #182, #184 | [Exterior ruler bounds](184-exterior-ruler-bounds.md) | Post-escape bounds enclose the ruler from the Green function and its gradient. The ruler grows as $\lvert c\rvert\log\lvert c\rvert$ far offshore. |
| #185 | [Trap-saddle curvature](185-trap-saddle-curvature.md) | The exact contour has unbounded signed curvature as the level approaches the saddle. The turning layer shrinks while its net turn tends to a quarter turn in either direction. |
| #182, #184 | [Quartic floor across the Shore](184-shore-floor-gluing.md) | Uniform local bounds give a positive $C^3$ metric across the finite Shore. The cardioid cusp precludes a $C^4$ extension. |
| #183, #184 | [Exterior ruler derivatives](184-exterior-ruler-jets.md) | Lean checks a common escape disk and Cauchy bounds for derivatives of its complex tail. Interval examples show how to choose an iteration count for a requested jet width. |
| #185 | [Trap steering budget](185-trap-steering-budget.md) | A squared-speed profile gives an acceleration-bounded restart and stop. At the saddle, a phase schedule can continue through an orthogonal handoff with zero crossing speed. |
| #183, #185 | [Steering demand under three metrics](185-matched-metric-steering.md) | The same near-saddle arcs have different acceleration demand under the signed-distance graph, signed-distance conformal and analytic cardioid metrics. |
| #184 | [Conformal proof contracts and dependency audit](184-conformal-contract-acceptance.md) | Lean checks the inland length, kinetic, and scale bounds, exact planar conformal curvature, and the ideal normal Jacobi equation. The audit identifies the classical inputs and numerical provider obligations. |
| #185 | [Interior navigation comparison](185-navigation-field-comparison.md), [exterior directions](185-exterior-navigation.md), [trap selection](185-trap-selection-validity.md) | Matched-speed paths compare natural coordinates, Map gradients, and a fixed trap. Lean checks exterior rate bounds and a finite error margin for the selected trap. |
| #168 | [Stable radial profile](168-stable-radial-profile.md) | Lean checks a positive-sum formula for intrinsic distance to the Shore. A log-scaled binary64 probe checks finite and overflow cases against high precision. |
| #169 | [Shore-crossing work cap](169-crossing-work-cap.md) | Lean checks path-length and speed-time work caps and the corresponding necessary energy conditions for a crossing. |
| #174 | [Intrinsic target bounds](174-intrinsic-target-bounds.md) | Lean checks exact target-set infima, consistent lower bounds, and tolerance and search-radius rules for navigation. |

## Musical gesture geometry

The [Kepler gesture study](kepler-musical-gestures.md) derives a normalized
swept-area clock for an ellipse in a normal chart. Lean checks its unique
inverse, flat reduction, and distortion and residual bounds under an
enclosure of the pullback area density on the radial fan. The note also
separates the prescribed gesture from the potential and forcing needed to
realize it in Physics.

## Reproducible experiments

| Area | Evidence |
| --- | --- |
| Visual sensitivity | [Frozen stencils and cockpit inspection](151-frozen-julia-sensitivity.md), [analytic calibration and second panel](171-julia-sensitivity-calibration.md), [matched pixel and color refinement](171-julia-observable-refinement.md), [geometry mask and pixel confounds](140-julia-morphology-provenance.md) |
| Physics and Map | [Sampled baseline metric](120-sampled-metric-validation.md), [signed work ledger](166-work-energy-ledger.md), [held-RK integral and refusal](166-held-rk-work.md), [Shore shortcut benchmark](172-shore-shortcut-benchmark.md), [retained field against exact Shore](181-retained-period-two-fidelity.md), [two-component Controls trial](185-contour-controls-trial.md), [recurrence and trap controls](185-finite-recurrence-controls.md), [waypoint guidance ablation](185-waypoint-guidance-ablation.md) |
| Training inputs and resets | [Causal density](101-onset-density-ablation.md), [SongFormer boundaries](102-boundary-target-evidence.md), [affect heads and provenance](103-offline-affect-screen.md), [temporal policy audit](104-temporal-policy-audit.md), [scale-stratified resets](110-reset-distribution.md), [sparse gates](152-sparse-event-gate-ablation.md) |
| Musical and expressive studies | [Teacher timeline alignment](188-teacher-timeline-alignment.md), [stimuli and randomized protocol](189-expressive-dynamics-protocol.md), [sample-aligned Julia playback](189-frozen-julia-playback.md) |

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

The pinned package passes `lake build --wfail` and the transitive axiom audit.
See the [formal inventory](../../formal/README.md) for current counts and the
scope of each statement, and the [verification record](verification.md) for
earlier checkpoints.

The optional Lean LSP server is registered in the repository
[Codex configuration](../../.codex/config.toml). A fresh configured session
passed the stdio smoke test with 22 tools and checked an actual theorem.
The [Aeneas/Charon spike](158-rust-verification-spike.md) documents the decision
to defer generated Rust proofs. The foundational issues have closed at their
documented proof boundary. The experiments and application requirements below
remain independent.

The same actual potential, normalized fixed-command drive and viscous-drag
catalog now has local force pair (6,210) and acceleration pair (8,400). The
quadratic reference supplies actual local field and Hessian-variation bounds;
inverse-metric cancellation preserves the small constants. The actual
reference horizon is now 1/576, with speed 8 kept separate from Lipschitz 400.
The exact-real m-step recurrence has all-time phase error<=1/(36m), and 2048
steps meet the uniform target 2^-16. Every admitted displaced query retains
all 36 mathematical stencil addresses. The earlier certificate remains valid.

The actual energy differential has pair (5,134); every step of this longer
run has Taylor remainder<=1/(64m²), and the accumulated defect after retaining
the sum of actual linear energy increments is<=1/(64m). This does not bound
total driven/damped energy change. The runtime holds its supplied drive
covector and recomputes drag at each stage. The new actual held-RK recurrence
closes stage displacement, stage/endpoint domains and first-order method
residuals for the fixture's outer-sampled exact-drive caller policy. For every
positive internal count, including one and four, the same actual reference
has all-time phase error<=3/(16m) for m>=16 outer steps. A 16384-step run meets
the uniform target 2^-16. All internal stage and weighted endpoint queries
retain the mathematical halo allowance.

The actual signed stage-power quadrature now has a separate certificate:
every stored prefix has energy/work defect<=1/(8mn). At m=16384 this gives
2^-17 for n=1 and 2^-19 for n=4. The actual reference power integral equals
its energy change. Combining the finite ledger with phase error gives
all-time affine work-interpolation error<=1/(8mn)+35/(36m), and joint
uniform phase/work convergence to the same reference, uniformly in positive n.
This does not bound total driven energy change; internal refinement alone
does not remove the outer holding-policy discrepancy. IEEE stage/power
arithmetic and summation, actual control construction, provider acceptance
and continuation remain implementation obligations.

The proposed dynamics-derived replacement field is recorded as future work.
The present formulation is retained. The stage, averaging, residual and
signed-work proofs are general-field results; the literal fixture instantiates their
constants. The [reuse boundary](157-field-replacement-boundary.md) identifies
the actual distance-to-set theorems that require separate transfer arguments.

For #160, the prior zero-floor attainment, uniform subsequential selected-path
convergence and full-sequence selected-length convergence remain intact. This
increment does not strengthen those geometric limits or infer smoothness of
arbitrary AC minimizers.
## Remaining formal and implementation boundaries

| Area | Remaining work |
| --- | --- |
| Geometric interfaces | Construct manifold/vector-bundle connection and curvature objects and glue them over an atlas. Actual moving-frame gauge/composition laws, fiber-valued tensor covariance and sufficient nonlinear coefficient regularity are checked. |
| Embedded curved Shore | Compute jet and separation constants for concrete Shore representations and extend reach results to other embedding hypotheses or weaker regularity. The actual ruler, ambient connection and curvature transfer are checked under the documented normal-patch, Frenet and higher-regularity conditions. |
| Smooth minimizing geometry | Prove regularity and geodesic-ODE identification for arbitrary attained AC minimizers; develop general conjugacy/second-variation criteria, all-minimizer classification above the flat threshold and global cut-locus arguments. The exact flat threshold and every minimizing AC path at or below it are checked. |
| Curved-Shore bifurcation | Formalize the functional-analytic reduction, local branch and minimum arguments. The coordinate curvature calculation alone does not prove the bifurcation. |
| Actual providers and arithmetic | Extend the concrete quadratic-fixture fidelity to production distance assets; certify compiled extraction/storage/mask/index correspondence, actual query-rounding budgets and IEEE field/force evaluation. The literal fixture, its actual reference jets, catalog force constants and real-query address control are checked. |
| Full discrete dynamics | Discharge the conditional operation-error budgets for compiled RHS, stage, power and work arithmetic; certify rounded control construction, reported energy, clock accumulation and operational acceptance/refusal. Extend to perturbed internal-substep composition, impulses, time-dependent commands, combined steering and continuation beyond T=1/576. Exact-real arbitrary-substep and conditional one-substep phase/work certificates remain distinct. |
| Zero-floor minimizing paths | Establish additional uniqueness or selection hypotheses for whole-family path convergence, convergence of velocities if needed, and uniform limits over varying endpoints. Actual componentwise zero-floor attainment, uniform subsequential path convergence and full-sequence unregularized length convergence are checked. |

The AC metric-length bridge, arbitrary-endpoint attainment, all-nearest-Shore
minimizer classification, flat Euclidean specialization, and flat Shore
classification at and below the threshold are complete at their documented
formal scope. Fixed-pair zero-floor limits now have the minimizing-path
continuation described above. Uniform convergence over varying endpoint pairs
and convergence of an arbitrary whole selected path family remain separate.
The foundational package landed in PR #164. The application obligations
above have their own acceptance criteria.
