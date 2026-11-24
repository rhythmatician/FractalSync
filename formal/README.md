# Lean research contracts

This is a development-time mathematical proof package. It currently contains
**400 explicit proved theorem/lemma declarations**: 357 in twenty-seven result
modules supporting #157, #159, and research issues #160--#163, plus 43 in nine
analysis-support modules, including attributed upstream backports. Definitions
and generated helpers are counted separately by the transitive axiom audit. It is independent of Rust
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
| Source snapshot for the AC analysis backport | Mathlib `aad247f928b1eb7c58478742119fdd15a37f0ffd`; not a dependency-pin change |
| Transitive Lean dependencies | `lake-manifest.json` |
| Installer used in this environment | Elan 4.4.0 |
| Authoring guidance consulted | `leanprover/skills`, commit `7d3da0282e7b724b07620e45cf212f2e05e19334`, especially `lean-proof` and `mathlib-build` |
| Optional Lean LSP/MCP server | `lean-lsp-mcp` 0.33.0, source commit `7932203d79db213852db215835bd79c943bb99b0` |

Agents own installation and maintenance. If Elan is absent, obtain it from the
[official installer](https://github.com/leanprover/elan). Respect these pins when
reproducing the proofs; no latest-version assumption is required. From `formal/`:

```sh
elan toolchain install leanprover/lean4:v4.24.0
MATHLIB_NO_CACHE_ON_UPDATE=1 lake update
lake exe cache get
lake build --wfail
lake env lean Audit.lean
```

Mathlib's precompiled cache avoids rebuilding its dependencies from source.
For a smaller download, pass the `Mathlib.*` import roots appearing in the
project's Lean files to `lake exe cache get`. In the working environment some
downloads received HTTP 429 responses; retrying the missing files with curl's
`parallel-max = 1` and `retry-delay = 5` in a temporary `CURL_HOME/.curlrc`
completed successfully. No Mathlib source changes were required.

`lake-manifest.json` is committed so all resolved dependencies are reviewable.
The package treats warnings as errors. The GitHub workflow builds this package
and runs the same axiom audit only for changes touching `formal/` or its workflow;
it is not a training preflight requirement.

### Optional agent authoring tools

An earlier verification run installed the pinned server in a separate Python
environment and tested it through an actual MCP stdio session. It exposed 22 tools; diagnostics for
`DistanceToSet.lean` were clean and `lean_verify` accepted
`FractalSync.DistanceGraph.exists_nearest_segment` with standard axioms only.
The current continuation reinstalled Lean in a fresh environment; the optional
server was not reinstalled or retested. Its smoke check is reproducible from the repository root:

```sh
python -m pip install -r formal/tools/requirements-mcp.txt
python formal/tools/check_mcp.py
```

Use the interpreter belonging to the environment where the server is installed.
The server also needs Elan's binaries on PATH and the Lean package built.
[tools/vscode-mcp.json](tools/vscode-mcp.json) is a pinned VS Code MCP
configuration template; merge its `servers` entry into the local
`.vscode/mcp.json` when using that client. This template does not silently
register the server in other agent clients. Python/MCP are authoring aids and
are not dependencies of proof CI or part of the kernel's trusted foundation.

The official Lean agent skills were consulted at the pin above. They are not
vendored or registered in a persistent personal skill catalog by this package.
Together with the metric and continuous mechanics results below, this makes
substantial progress on #157; client-specific persistent authoring integration
still needs its stated follow-up. Keep one shared proof package.

## Exact theorem inventory

The declaration counts include explicit helper lemmas; they are not counts of
independent research results. Compiler-generated declarations are counted only
by the separate transitive audit.

| Module | Theorem/lemma declarations | Main scope |
| --- | ---: | --- |
| `Ruler` | 9 | Scalar ruler calculus and conditional field chain rule |
| `RulerDifferential` | 20 | Actual field Hessians, scale derivatives, summit and reference origin |
| `ScaleMetric` | 31 | Rank-one metric, inverse, determinant, eigen-directions, rotations |
| `MetricImpulse` | 14 | Bilinear Cauchy--Schwarz and exact impulse energy |
| `ContinuousMechanics` | 15 | Compact connection, actual work-energy calculus, signed work and dissipation |
| `DistanceGraph` | 4 | Squared-density algebra |
| `DistanceToSet` | 6 | Actual distance field and nearest-point segments |
| `RadialProfile` | 44 | Integral calculus, sharp constant, logarithmic limits |
| `RadialLipschitz` | 8 | Global and optimal Lipschitz constants |
| `PathCalibration` | 8 | Integrated calibration and a split-at-Shore bound |
| `AbsolutelyContinuousPaths` | 14 | Unrestricted AC traces, a.e. calculus, integrability and calibration |
| `IntrinsicShoreDistance` | 18 | Actual AC path infima in both orientations, attained minimum and zero-floor divergence |
| `ComponentCrossing` | 3 | Topological Shore crossing and unrestricted AC component calibration |
| `PairwiseDistance` | 13 | Actual pairwise path infimum, component divergence and floor monotonicity |
| `PathOperations` | 16 | Actual AC reversal and concatenation with exact length laws |
| `DisplacementBounds` | 13 | Logarithmic displacement bound, positive definiteness and Shore-endpoint divergence |
| `IntrinsicMetric` | 7 | Symmetry, triangle inequality, bundled metric and endpoint divergence wrappers |
| `IntrinsicTopology` | 10 | Original topology, compact balls, properness and intrinsic completeness |
| `MetricComparison` | 6 | Scale-only versus anisotropic lengths and actual path infima |
| `ACComponentPaths` | 8 | AC reachability throughout each component of an open domain |
| `ZeroFloorLimit` | 21 | Actual zero-floor integrability, quantitative infimum sandwich and same-component limit |
| `CrossingAction` | 5 | Integral Cauchy--Schwarz, crossing duration and kinetic action |
| `FlatShore` | 32 | Scalar warp derivatives, curvature expressions, excursion integral |
| `SensitivityBounds` | 4 | Sensitivity transfer and scalar shear |
| `NormalizedDistanceFields` | 8 | Normalized-field algebra and Lipschitz bounds |
| `NormalizedMapDifferential` | 8 | Actual window differential and quadratic remainder from segment Hessian bounds |
| `CovariantStatistics` | 12 | Finite transport algebra and a PSD counterexample |
| **Result-module subtotal** | **357** | Includes supporting lemmas, not just independent research results |
| `Analysis/*` (nine modules) | 43 | Attributed AC FTC backport and compatibility/finite-dimensional support |
| **Total** | **400** | 109 more source declarations than the prior 291-theorem increment |

The analysis-support modules contribute respectively 8 (`AbsolutelyContinuous`),
6 (`AbsolutelyContinuousFTC`), 4 (`DerivIntegrable`), 2 (`Fatou`),
1 (`FiniteDimensionalAC`), 3 (`FinsetGaps`), 3 (`IntervalSlope`), and
12 (`OrderFinsetGaps`), plus 4 in the new `ACComposition` module. The
[support provenance](FractalSync/Analysis/README.md) records the exact upstream
snapshot and retained Apache 2.0 attribution. The four AC composition/gluing
proofs are project compatibility work, not attributed upstream declarations.

### #157/#159: ruler and local metric foundations

[Ruler.lean](FractalSync/Ruler.lean) uses namespace `FractalSync`.
For `rho(r)=sqrt(r²+epsilon²)` and epsilon>0 it checks positivity,
`rho²=r²+epsilon²`, `abs(r)<=rho`, `epsilon<=rho`, the actual derivative
`r/rho`, and the global one-Lipschitz bound. Composition with Mathlib's
distance-to-set function is one-Lipschitz. For an underlying field with an
explicit Fréchet derivative, `hasFDerivAt_ruler_comp` proves the field chain
rule. `hasDerivAt_ruler_slope` proves the scalar second-derivative coefficient
`epsilon²/rho³`. The other names are `ruler_pos`, `ruler_sq`, `abs_le_ruler`,
`epsilon_le_ruler`, `hasDerivAt_ruler`, `ruler_lipschitz`, and
`regularizedDistance_lipschitz`.

There is no inference of a smooth distance field from positive regularization.
The subsequent differential modules below prove full field Hessians and
continuous mechanics under explicit local regularity hypotheses.

[ScaleMetric.lean](FractalSync/ScaleMetric.lean) and
[MetricImpulse.lean](FractalSync/MetricImpulse.lean) use
`FractalSync.ScaleMetric`, with a nested `Plane` namespace for rotations.
The complete grouped identifiers, conventional formulas and assumptions are in
[the metric foundations note](../docs/research/157-scale-metric.md).

| Checked statement | Hypotheses and principal identifiers |
| --- | --- |
| `G=rho⁻²(I+a² u⊗u)` is a positive symmetric bilinear metric | A real inner-product space, rho>0; `metric_self_pos`, `metric_symm` and the bilinearity lemmas. No derivative assumption on u. |
| Exact two-sided inverse | rho nonzero; `inverse_metricAction`, `metric_inverseAction`, using `weight_pos` and Sherman--Morrison identities. |
| `norm(v)²/rho² <= g(v,v) <= (1+a² norm(u)²) norm(v)²/rho²` | `metric_norm_lower`, `metric_norm_upper`. `weight_bounds` additionally requires norm(u)<=1. |
| Tangent/normal eigenvalue equations and arbitrary finite-dimensional determinant | `metricAction_tangent`, `metricAction_normal`, `det_metricMatrix`, `det_metricMatrix_pos`. Zero vectors are not asserted to be genuine eigenvectors; dimension one need not have a tangent eigendirection. |
| Exact kinetic energy and quadratic expansion | `kinetic_formula`, `kinetic_add`, `two_kinetic`, `kinetic_nonneg`. |
| Plane quarter-turn squares to minus identity and rotations preserve all metric pairings | `Plane.quarter_square`, `Plane.quarter_isometry`, `Plane.circle_rotation_isometry`, `Plane.rotation_isometry`; the general symmetric matrix's determinant is positive. Positive definiteness is separate where a geometric interpretation requires it. |
| Exact impulse jump `Delta K=<v,J>+dualNorm(J)²/2` | Fixed configuration and nonzero rho; `impulse_energy_jump`. |
| `Delta K<=sqrt(2K) Jmax+Jmax²/2` | Actual dual-metric norm bounded by Jmax; `impulse_pairing_bound`, `impulse_energy_bound`. Mathlib's bilinear-form Cauchy--Schwarz is reused. |

Traceability: `runtime-core/src/manifold.rs` (`induced_metric`, inverse,
kinetic energy) and `runtime-core/src/geometry_provider.rs` (ruler jet),
issues #120/#145. Exact positive definiteness does not certify an absolute
floating-point determinant threshold or a provider's derivative consistency.

### #159: actual field Hessians and continuous mechanics

[RulerDifferential.lean](FractalSync/RulerDifferential.lean) uses namespace
`FractalSync`; [ContinuousMechanics.lean](FractalSync/ContinuousMechanics.lean)
extends `FractalSync.ScaleMetric`. The
[differential mechanics note](../docs/research/159-differential-mechanics.md)
gives the conventional formulas and the relationship to the field assumptions.

| Checked result | Principal declarations and hypotheses |
| --- | --- |
| Scalar logarithmic scale, first/second scalar derivatives, reference shift and summit | `scaleCoordinate_eq_log_sub`, `hasDerivAt_scaleCoordinate`, `hasDerivAt_scaleSlope`, `scaleCoordinate_reference_shift`, `scaleCoordinate_le_iff`, `scaleCoordinate_ruler_le_summit`, `ruler_zero`, `scaleCoordinate_ruler_eq_summit_iff`; positive ruler/reference/floor where used. |
| Scale differentiated directly through the regularized field | `hasDerivAt_scaleCoordinate_ruler`, `hasDerivAt_scaleRulerSlope`, `hasFDerivAt_scaleCoordinate_ruler_comp`, `hasFDerivAt_scale_ruler_differential`. |
| Actual ruler Hessian `(D/rho) H_D + (epsilon²/rho³) dD tensor dD` | `hasFDerivAt_ruler_differential`, `hasFDerivAt_fderiv_ruler_comp`; local differentiability of D and an actual Fréchet derivative of its differential field. |
| Actual scale Hessian for an arbitrary positive ruler and directly in D | `hasFDerivAt_scaleCoordinate_comp`, `hasFDerivAt_scale_differential`, `hasFDerivAt_fderiv_scaleCoordinate_comp`, `hasFDerivAt_fderiv_scaleCoordinate_ruler_comp`; the last gives `-D/(rho² log 2) H_D + (D²-epsilon²)/(rho⁴ log 2) dD tensor dD`. |
| At D=0, the field Hessian drops out and the derived Hessians have rank at most one | `hasFDerivAt_fderiv_ruler_on_zero`, `hasFDerivAt_fderiv_scale_on_zero`; the same actual local derivative hypotheses, without an ideal unit-gradient assumption. |
| Compact connection symmetry, metric compatibility, Koszul identity and uniqueness | `connection_symm`, `metric_connection`, `connection_metric_compatible`, `connection_koszul`, `connection_eq_of_koszul`; symmetric H for torsion/Koszul and positive rho for uniqueness. |
| Actual varying metric and kinetic derivatives | `hasDerivAt_metric_pairing`, `hasDerivAt_kinetic`; derivatives of the ruler, gradient and velocity are explicit local jets. |
| Actual continuous work-energy `d(K+U)/dt=inner(v,Q)` | `hasDerivAt_totalEnergy`, `hasDerivAt_totalEnergy_of_fields`; the latter derives trace jets from position and Fréchet field derivatives. Dynamics uses the existing inverse metric. |
| Nonpositive PSD friction power and exact metric-friction power `-2 beta K` | `friction_power_nonpos`, `metric_friction_power`. The latter's sign additionally uses beta>=0. |
| Full signed work integral, necessary potential-rise bound, geodesic and conservative zero derivatives | `integral_work_eq_energy_change`, `potential_increase_le_initial_kinetic_add_work`, `hasDerivAt_kinetic_geodesic`, `hasDerivAt_totalEnergy_conservative`. The integral theorem requires continuous endpoint energy and integrable power. |

No bundled manifold connection, global solution existence, global distance-field
smoothness or discrete energy conservation is claimed. The normalized Map
differential and quadratic remainder are now checked by
[NormalizedMapDifferential.lean](FractalSync/NormalizedMapDifferential.lean),
inventoried below. The full potential-force catalog and remaining specific
control contracts are still separate Phase A obligations.

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

The analytic extensions of these four algebraic statements are described next.

### #160: the actual radial integral

[RadialProfile.lean](FractalSync/RadialProfile.lean) uses namespace
`FractalSync.DistanceGraph`. The complete conventional explanation is in
[the radial-profile note](../docs/research/160-radial-profile.md).
`radialProfile` is the actual interval integral, not a stipulated antiderivative.
Unless a row says otherwise, epsilon>0 and a is any real number.

| Checked statement | Principal identifiers |
| --- | --- |
| Positive continuous density, FTC derivative, F(0)=0 and strictly increasing F | `radialDensity_pos`, `continuous_radialDensity`, `hasDerivAt_radialProfile`, `radialProfile_strictMono`, `radialProfile_zero`, `radialProfile_nonneg`, `radialProfile_pos` |
| Equality with the earlier squared radial density | `radialDensity_sq` |
| Positive dilation and normalization F(epsilon,a,r)=F(1,a,r/epsilon) | `radialDensity_scale`, `radialProfile_scale`, `radialProfile_normalize` |
| Full logarithmic closed primitive, including valid logarithm domains | `radial_log_argument_pos`, `hasDerivAt_normalizedRadialPrimitive`, `radialProfile_one_eq_primitive`, `radialProfile_closed_form` |
| Exact a=0 asinh and integrated asinh sandwich for r>=0 | `radialProfile_scale_only`, `radialDensity_comparison`, `radialProfile_asinh_bounds` |
| Sharp normalized density bounds and their attainment | `radialDensity_le_one` assumes a²<=1; `radialDensity_le_peak` assumes a>0; `radialDensity_one_at_zero`; `radialDensity_peak_attained` assumes a>=1 |
| F tends to infinity as r tends to infinity, or epsilon tends to zero with fixed r>0 | `radialProfile_tendsto_atTop`, `radialProfile_tendsto_zero_floor` |
| Defect `sqrt(1+a²) asinh(q)-F(1,a,q)` is nondecreasing and tends to `C(a)=a asinh(a)-sqrt(1+a²) log(sqrt(1+a²))` | `radialDefect_monotone`, `radialDefect_tendsto`, with checked reciprocal-coordinate compactification |
| Sharp additive bound and optimality for nonnegative radii | `radialProfile_sharp_bounds`, `radialDefectLimit_nonneg`, `radialDefectLimit_le_of_uniform_bound` |
| Exact logarithmic remainder limit, for large radius and for vanishing positive floor | `arsinh_sub_log_tendsto`, `radialProfile_log_asymptotic`, `radialProfile_zero_floor_log_asymptotic` |

The source also contains the explicit positivity, zero-value, compactification
and derivative helper theorems used by these results. The density maximum is
proved by upper bounds and attainment rather than a supremum wrapper.
[RadialLipschitz.lean](FractalSync/RadialLipschitz.lean) now packages the global
bound and the sharp Lipschitz theorem, with optimality:
`radialDensity_normalize`, `radialDensity_uniform_bound`,
`radialProfile_lipschitz`, `radialMaxSlope_pos`,
`radialDensity_le_radialMaxSlope`, `radialMaxSlope_attained`,
`radialProfile_lipschitz_sharp`, and `radialProfile_lipschitz_constant_ge`.
Sharpness uses the physical convention a>=0; the convenient sqrt(1+a²)/epsilon
bound is valid for every real a.

### #160: nearest segments and integrated path calibration

[DistanceToSet.lean](FractalSync/DistanceToSet.lean) and
[PathCalibration.lean](FractalSync/PathCalibration.lean) share namespace
`FractalSync.DistanceGraph` and use Mathlib's actual `Metric.infDist`.

| Checked statement | Exact identifiers and hypotheses |
| --- | --- |
| A Lipschitz outer map bounds the derivative of a curve composition | `norm_deriv_comp_lipschitz_le`; the curve and composite derivatives must exist, but the outer map need not be differentiable. |
| Distance-trace speed is at most curve speed | `infDist_deriv_bound`, a specialization of the preceding theorem. |
| Distance on a nearest-point segment p to x is exactly t dist(x,S) for 0<=t<=1 | `infDist_nearest_segment`; any real normed space, p in S and a nearest-point equality. No convexity or uniqueness premise. |
| Actual interior derivatives of that segment and its distance trace | `hasDerivAt_nearest_segment`, `hasDerivAt_infDist_nearest_segment`. No endpoint derivative is claimed for the distance trace. |
| Existence of a calibrated nearest segment | `exists_nearest_segment`; S is closed and nonempty in a proper real normed space, including finite-dimensional Euclidean space. |
| Nonnegative graph speed and square-root calibration | `graphSpeedSq_nonneg`, `radial_speed_le_graphSpeed`, `graphSpeed_radial`; the lower bound requires radial speed squared <= Euclidean speed squared. |
| Actual nearest-segment length integral equals F(dist(x,S)) | `radial_segment_length`, `nearest_segment_length`; the integrand uses the segment speeds justified by the preceding derivative lemmas. |
| Integrated endpoint calibration | `radialProfile_endpoint_bound`, `distance_trace_endpoint_bound`; continuity on the closed interval, derivatives on its open interior and explicitly integrable densities. |
| Shore-crossing lower bound F(d(start))+F(d(end)) | `distance_trace_crossing_bound` uses derivatives on the two open subintervals, continuity/integrability across the crossing, and a crossing point in S. No derivative at the crossing is required. |

These pointwise contracts are retained. The unrestricted absolutely continuous
theorems and actual Shore-path infimum are now established by the following
modules, with their complete proof boundary in the
[AC continuation](../docs/research/160-absolute-continuous-paths.md).

### #160: unrestricted AC paths, path infima and crossing action

[AbsolutelyContinuousPaths.lean](FractalSync/AbsolutelyContinuousPaths.lean),
[IntrinsicShoreDistance.lean](FractalSync/IntrinsicShoreDistance.lean), and
[CrossingAction.lean](FractalSync/CrossingAction.lean) share namespace
`FractalSync.DistanceGraph`. They use actual Mathlib absolute continuity,
actual derivatives and interval integrals. The AC FTC and vector derivative
integrability are proved by the attributed support package; they are not
additional hypotheses of the project path theorems.

| Checked result | Principal declarations and hypotheses |
| --- | --- |
| Distance/ruler traces preserve AC; distance derivative is bounded a.e. by curve speed | `ac_distance_trace`, `ac_ruler_trace`, `ac_distance_trace_deriv_bound`; finite-dimensional real normed E for vector differentiation. No ambient distance differentiability. |
| Calibrated composition is AC with actual a.e. derivative, integrability and FTC | `ac_radialProfile_trace`, `ac_radialProfile_deriv`, `ac_radialDensity_intervalIntegrable`, `ac_radialProfile_integral`; epsilon>0, a any real. |
| Actual graph speed is integrable from AC alone | `graphSpeed_nonneg`, `graphSpeed_eq_rulerSpeed`, `graphSpeed_le_global`, `aestronglyMeasurable_graphSpeed`, `ac_graphSpeed_intervalIntegrable`. |
| Unrestricted endpoint and arbitrary Shore-crossing lower bounds | `ac_distance_trace_endpoint_bound`, `ac_distance_trace_crossing_bound`; AC plus the endpoint order and stated crossing. No derivative at any specified time or extra density-integrability premise. |
| Length equals the original ruler-derivative integral | `acGraphLength_eq_ruler_integral`; actual a.e. ruler chain rule, including cusp traces. |
| Actual nearest and reversed-nearest segments are admissible and attain F(d(x)) | `absolutelyContinuousOnInterval_lineMap`, `acGraphLength_nearest_segment`, `hasDerivAt_infDist_reverse_nearest_segment`, `acGraphLength_reverse_nearest_segment`, `radialProfile_mem_shorePathLengths`. |
| The path-length sets are nonempty/bounded, and F(d(x)) is their least element | `acGraphLength_nonneg`, `shorePathLengths_bddBelow`, `shorePathLengths_nonempty`, `intrinsicShoreDistance_le_radialProfile`, `radialProfile_le_of_mem_shorePathLengths`, `radialProfile_isLeast_shorePathLengths`, `radialProfile_isLeast_pointToShorePathLengths`; S closed and nonempty for attainment. |
| Exact real infima in both path orientations | `intrinsicShoreDistance_eq_radialProfile`, `pointToShorePathLengths_sInf_eq`, `pointToShorePathLengths_sInf_eq_intrinsicShoreDistance`; finite-dimensional real normed E, positive epsilon. |
| Actual Shore-path infimum is zero exactly on S and diverges as epsilon decreases to zero off S | `intrinsicShoreDistance_eq_zero_iff`, `intrinsicShoreDistance_tendsto_zero_floor`; not the full pairwise componentwise limit. |
| Crossing duration and integrated kinetic-action bounds | `ac_crossing_duration_bound`, `ac_crossing_kinetic_action_bound`; the former has an a.e. speed cap, the latter positive duration and integrable squared speed. Supporting actual integral results are `integral_affine_sq`, `sq_integral_le_duration_mul_integral_sq`, `kinetic_action_lower_bound`. |

The next modules establish the actual pairwise metric, properness, completeness,
and complete pointwise componentwise zero-floor limit. Metric curve-length
identification and minimizing curves between arbitrary endpoints remain
separate formal obligations. The Euclidean classification of every
nearest-Shore minimizer is conventional, as explicitly labeled in the AC note.
Restricted provider domains can exclude a nearest segment. Kinetic action is
distinct from work and from a sufficient dynamical crossing criterion.

### #160: actual pairwise metric, topology and complete fixed-pair limit

The [pairwise continuation](../docs/research/160-pairwise-metric-limit.md)
develops the actual definitions and proofs. All modules below use namespace
`FractalSync.DistanceGraph`. Unless a row says otherwise, E is a
finite-dimensional real normed space, epsilon>0, and a is any real.

| Module and checked scope | Principal declarations and exact boundary |
| --- | --- |
| [ComponentCrossing](FractalSync/ComponentCrossing.lean): a path leaving the complement component crosses S | `continuousOn_exists_shore_crossing` works in an arbitrary topological target; `ac_exists_shore_crossing` and `ac_distance_trace_component_bound` give the AC crossing and integrated radial sum. No minimizing path is assumed. |
| [PairwiseDistance](FractalSync/PairwiseDistance.lean): the actual nonempty, bounded-below length class and its real infimum | `pairwisePathLengths_nonempty`, `pairwisePathLengths_bddBelow`, `intrinsicDistance_nonneg`, `intrinsicDistance_le_length`, `intrinsicDistance_self`. Positive floor supplies actual integrability for every AC candidate. |
| Pairwise endpoint calibration and different-component divergence | `radialProfile_difference_le_intrinsicDistance`, `radialProfile_sum_le_intrinsicDistance_of_separated_components`, `intrinsicDistance_tendsto_zero_floor_of_separated_components`. The divergence theorem requires nonempty closed S and x outside S; y is outside x's complement component and may itself lie on Shore. |
| Pairwise floor monotonicity and global comparison | `graphSpeed_antitone_floor`, `acGraphLength_antitone_floor`, `intrinsicDistance_antitone_floor`; `acGraphLength_le_global` and `intrinsicDistance_le_global` give the sqrt(1+a²)/epsilon comparison. |
| [PathOperations](FractalSync/PathOperations.lean): AC reversal and concatenation | `ac_reversePath`, `ac_concatPath`, `acGraphLength_reversePath`, `acGraphLength_concatPath`. Exact length preservation/additivity, matching endpoints for concatenation, no derivative at the join. The reversal identity is more general than the positive-floor AC application. |
| [DisplacementBounds](FractalSync/DisplacementBounds.lean): logarithmic displacement and positive definiteness | `ac_log_displacement_bound`, `log_displacement_le_intrinsicDistance`, `intrinsicDistance_pos`, `intrinsicDistance_eq_zero_iff`. The bound is log(1+dist(x,y)/rho(x)) <= D(x,y), obtained from actual AC logarithmic calculus. |
| Shore endpoints, including two distinct points on S | `intrinsicDistance_tendsto_zero_floor_of_mem_left`, plus `intrinsicDistance_tendsto_zero_floor_of_mem_right` and `intrinsicDistance_tendsto_zero_floor_of_mem_or_mem` in IntrinsicMetric. Distinct endpoints and at least one on S imply divergence; no closedness premise is needed for this branch. |
| [IntrinsicMetric](FractalSync/IntrinsicMetric.lean): actual metric axioms | `pairwisePathLengths_symm`, `intrinsicDistance_symm`, `exists_path_length_lt_intrinsicDistance_add`, `intrinsicDistance_triangle`, `intrinsicMetricSpace_dist`, and the explicit `intrinsicMetricSpace` definition. The triangle proof uses arbitrarily near-infimum paths, not attainment. |
| [IntrinsicTopology](FractalSync/IntrinsicTopology.lean): topology, properness and completeness | `intrinsicDistance_abs_sub_le`, `intrinsicDistance_lipschitz_right`, `dist_le_ruler_mul_exp_intrinsicDistance`, `intrinsicSublevel_subset_closedBall`, `isClosed_intrinsicSublevel`, `isCompact_intrinsicSublevel`, `isOpen_iff_intrinsicBall`, `intrinsicMetricSpace_topology`, `intrinsicMetricSpace_proper`, `intrinsicMetricSpace_complete`. Completeness is for the intrinsic uniformity; only topologies are identified with the ambient structure. |
| [MetricComparison](FractalSync/MetricComparison.lean): actual anisotropic/scale-only comparison | `graphSpeed_zero_le`, `graphSpeed_le_sqrt_mul_zero`, `acGraphLength_zero_le`, `acGraphLength_le_sqrt_mul_zero`, `intrinsicDistance_zero_le`, `intrinsicDistance_le_sqrt_mul_zero` prove D(epsilon,0) <= D(epsilon,a) <= sqrt(1+a²) D(epsilon,0). |
| [ACComponentPaths](FractalSync/ACComponentPaths.lean): actual off-Shore reference paths | `acJoinedIn_of_mem_connectedComponentIn` and `exists_ac_path_avoiding_of_mem_connectedComponentIn`. The open reachable class and its open relative complement force an entire connected component into the AC class. This works in any real normed space, without finite dimensionality. |
| [ZeroFloorLimit](FractalSync/ZeroFloorLimit.lean): actual zero-floor density and infimum | `ac_path_exists_positive_clearance`, `ac_graphSpeed_zero_intervalIntegrable`, `zeroFloorPathLengths_nonempty_of_mem_connectedComponentIn`. The zero-floor infimum uses paths avoiding S on the entire closed interval; its geometric theorems require a nonempty class. It is not an extended metric on empty classes or the unrestricted definition evaluated at epsilon=0. |
| Quantitative comparison and same-component convergence | `ac_log_ruler_increment_bound`, `ac_ruler_lower_of_length_le`, `ac_path_avoids_of_length_le`, `intrinsicDistance_zero_floor_sandwich_of_reference`, `intrinsicDistance_tendsto_zero_floor_of_path_nonempty`, `intrinsicDistance_tendsto_zero_floor_of_mem_connectedComponentIn`. Nonempty closed S; the final limit needs only actual component membership, with no reference-path or integrability premise. |

For a reference zero-floor length at most B, m=dist(x,S) exp(-B), and
0<epsilon<m, the checked sandwich is
`(1-epsilon²/m²) D0 <= D_epsilon <= D0`. It treats all regularized candidates
in two cases according to whether their length is at most B, then takes the
infimum. Together with the diagonal identity, different-component divergence,
and distinct Shore-endpoint divergence, this covers every fixed pair when S
is nonempty and closed. It does not claim uniform convergence over endpoint
pairs or existence of a minimizing curve between arbitrary endpoints.

### #160: flat-Shore scalar geometry

[FlatShore.lean](FractalSync/FlatShore.lean) uses `FractalSync.FlatShore` and
reuses the shared ruler. Its geometric interpretation is developed in
[the flat-Shore note](../docs/research/160-flat-shore.md). The scalar formulas
are checked; identifying them with Riemannian curvature or a global cut locus
is a conventional proof step.

| Checked statement | Principal identifiers |
| --- | --- |
| Actual derivatives of Q, 1/rho and the intrinsic warp slope | `hasDerivAt_normalNumerator`, `hasDerivAt_warp`, `warpSlope_eq`, `hasDerivAt_warpSlope`, `warp_curvature_identity` |
| Rational curvature expressions: Shore values, exact derivative, strict decrease, ranges and positive radial/tangential gap | `radialCurvature_zero`, `tangentialCurvature_zero`, `hasDerivAt_radialCurvature`, `radialCurvature_deriv_neg`, `radialCurvature_strictAntiOn`, `curvature_gap`, `tangentialCurvature_lower`, `tangentialCurvature_lt_radialCurvature`, `radialCurvature_le_one`, `radialCurvature_bounds`, `tangentialCurvature_nonpos` |
| Actual excursion integral: continuity, strict monotonicity, threshold, unbounded lower bound, unique positive turning radius for every L>pi epsilon and dilation | `continuous_excursionSpan`, `excursionSpan_strictMonoOn`, `excursionSpan_zero`, `excursionSpan_gt_threshold`, `excursionSpan_eq_threshold_iff`, `excursionSpan_linear_lower`, `excursionSpan_existsUnique_radius`, `excursionSpan_scale` |
| Specified quartic density integrates to pi(1+a²)/16>0 | `quarticDensity_eq`, `integral_quarticDensity`, `integral_quarticDensity_pos`. Identification as a Taylor coefficient remains conventional. |
| Unit speed plus specified tangential momentum imply the radial first-integral formula | `radial_first_integral`; conservation of that momentum along geodesics remains conventional. |

Four helper lemmas establish denominator positivity and integrand continuity.
No Riemannian curvature tensor, conjugate-point or cut-locus theorem,
Banach-space implicit function theorem, or local minimizer theorem is
formalized in this module.

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

### #159/#162: actual normalized-Map differential and quadratic remainder

[NormalizedMapDifferential.lean](FractalSync/NormalizedMapDifferential.lean)
extends `FractalSync.NormalizedDistanceFields`. It works in any real normed
space, without an inner-product or finite-dimensional assumption.
For `M(D,c,h,u)=(D(c+h*u)-D(c))/h`, c and h are fixed when differentiating u.

| Theorem | Verified mathematical content |
| --- | --- |
| `hasFDerivAt_normalizedMap` | For h nonzero and an actual derivative of D at c+h*u, the window derivative is exactly that differential of D. |
| `hasFDerivAt_normalizedMap_zero` | The anchor derivative is dD(c), independent of the fixed nonzero window scale. |
| `fderiv_normalizedMap` | The corresponding equality of actual `fderiv` fields under local differentiability. |
| `normalizedMap_value_bound_on_segment` | A field L-Lipschitz on the physical query segment gives abs(M(u)) <= L*norm(u). |
| `field_quadratic_remainder` | Actual field derivatives along the closed segment, with anchored linear differential variation, give the sharp factor 1/2 in the field residual. |
| `normalizedMap_quadratic_remainder` | The normalized residual is bounded by L*abs(h)*norm(u)²/2 under the corresponding anchored variation hypothesis. |
| `normalizedMap_quadratic_remainder_of_lipschitz_fderiv` | Segment-wise L-Lipschitz actual differential implies the same normalized residual. |
| `normalizedMap_quadratic_remainder_of_hessian_bound` | Actual differentiation of D and its differential field on the entire physical segment, with Hessian operator norm <= L, implies the same residual. No ideal unit-gradient hypothesis. |

**Not formalized here:** the full combined provider/recomputed-ruler error
estimate, the cross-scale comparison assembled from these local residuals,
optimal finite-sample envelopes, topology counterexamples, or blow-up limits.
Actual local differentiability hypotheses do not certify numerical provider
derivatives or imply distance-field smoothness at a cut locus. No ideal
Lipschitz hypothesis is asserted for an arbitrary numerical provider.

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
transitive dependencies for every declaration in a `FractalSync` module, as
well as every declaration in namespace `FractalSync`. Module provenance
includes adopted upstream namespaces and generated/private helpers.
It rejects every axiom except `propext`, `Classical.choice`, and `Quot.sound`,
the standard classical Lean/Mathlib foundations used here. It also fails if the
project root set is empty. This detects imported or indirect `sorryAx` and additional
axioms, not merely source-text placeholders. Every proof module must be imported
by `FractalSync.lean` so the audit sees it.

The traversal uses Lean's own checked-environment collector with a shared
visited set across all project roots. A negative control with an unapproved
axiom outside the namespace and an indirect project dependency was rejected.
This is the same transitive allowlist check without repeated Mathlib traversal.

The current audit covers 741 declarations, including the 400 stated theorem/lemma declarations,
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
