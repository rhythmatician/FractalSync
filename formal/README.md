# Lean research contracts

This is a development-time mathematical proof package. It currently contains
**1,825 explicit proved theorem/lemma declarations** across 158 result modules
and fourteen analysis-support modules, including attributed upstream backports.
The transitive axiom audit also counts definitions and generated helpers.
The package is independent of Rust execution, Python training, the browser,
and Aeneas/Charon.

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

The pinned server passed the stdio smoke check. It exposed 22 tools,
reported clean diagnostics for `DistanceToSet.lean`, and accepted
`FractalSync.DistanceGraph.exists_nearest_segment` with standard axioms only.
The stdio check also passed using the transport returned by a fresh `codex mcp get`
from this checkout. On Windows the server emitted an asyncio cleanup warning
after verification and the check exited successfully. Run the smoke check
from the repository root with an isolated Python environment:

```sh
python -m pip install -r formal/tools/requirements-mcp.txt
python formal/tools/check_mcp.py
```

Use the interpreter belonging to the environment where the server is installed.
The server needs Elan's binaries on `PATH` and the Lean package built. A fresh
Codex CLI launch from the repository root loads the
[project MCP configuration](../.codex/config.toml) when the checkout is trusted.
Make `uvx` available on the client's `PATH`. Run `codex mcp get fractalsync-lean`
from the root to check recognition. Open a new project session to load the
server if the active session predates the configuration. The
[VS Code MCP template](tools/vscode-mcp.json) has a matching pinned `servers`
entry for `.vscode/mcp.json`. Proof CI and the kernel audit run independently
of the authoring server.

The official `lean-proof`, `mathlib-build` and `lean-setup` skills were
installed from the pinned `leanprover/skills` revision and verified in the
Codex skill directory. Keep the skills and proof tools on the development
side of the project.

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
| `PotentialForces` | 29 | Actual Shore/secant-wall derivatives, raised forces, wall limit and full crest energy budget |
| `ControlMechanics` | 28 | Momentum derivative, intrinsic drive/drag, candidate board/carve and fixed-metric steering |
| `DiscreteMechanics` | 11 | Frozen-metric step energy, work defect and sharp explicit-drag criterion |
| `MovingMetricMechanics` | 18 | Actual phase-energy differential, changing-position local defect, new-position impulse and segment safety |
| `CovariantSteering` | 8 | Actual moving-metric pairing derivative and covariant steering norm preservation |
| `AccumulatedDynamics` | 14 | Actual residual power, finite ordered-step energy/impulse accounting and accumulated domain budgets |
| `PhaseEnergyRegularity` | 3 | Actual phase-energy smoothness and uniform differential bounds on compact convex regions |
| `DiscreteSteering` | 8 | Endpoint-metric normalized steering candidate, unit norm and actual first-order consistency |
| `TrajectoryError` | 18 | Actual ordered-step stability and ODE defect, residual/rounding recurrence, simultaneous phase-domain and error control |
| `TrajectoryRefinement` | 7 | Explicit exponential O(h) envelope, actual fixed-horizon grid convergence and eventual domain membership |
| `TrajectoryInterpolation` | 9 | Actual continuous stored-state interpolation, all-time error/domain bound and uniform convergence |
| `LocalDynamics` | 6 | Actual local ODE solution, positive reference tube, regional uniqueness and mechanical coefficient adapter |
| `OperationalDynamics` | 5 | Regularized literal physical spline, actual Riesz gradient/Hessian and local mechanical ODE existence |
| `BoundedLipschitz` | 14 | Exact norm/Lipschitz certificates, bilinear operations, reciprocal floors and composition |
| `MechanicalBounds` | 1 | Explicit arithmetic norm/Lipschitz budgets for the actual inverse-force and connection expression |
| `RulerJetBounds` | 10 | Scalar regularizer gains, actual differential/Hessian chain rule and bilinear variation budgets |
| `OperationalRulerBounds` | 23 | Literal Euclidean ruler/gradient/Hessian norm and variation bounds, including its actual third derivative |
| `OperationalRouteBounds` | 2 | Straight-segment length intervals and the ambient arrival-ball lower bound under regional ruler premises |
| `ConformalTransportError` | 2 | Cartesian conformal connection split and transport-angle variation under a path enclosure |
| `OperationalHalo` | 11 | Physical query margins, all 36 addresses, finite zero extension and actual metric-jet/acceleration germ independence |
| `OperationalMechanicalBounds` | 7 | Explicit actual acceleration norm/Lipschitz constants on a phase ball from finite stored coefficient evidence |
| `QuantitativeDynamics` | 5 | Actual Picard solution at an explicit horizon, fixed certificate constants, reference tube and mesh criterion |
| `OperationalTrajectoryCertificate` | 4 | Actual finite-asset solution, explicit regional constants, full interpolation/refinement and every-time displaced-query address control |
| `PotentialForceBounds` | 16 | Actual Shore/wall and total conservative norm/Lipschitz budgets, radius-derived cosine floors and rational unit-ball bounds |
| `ControlForceBounds` | 10 | Actual metric action, drive and drag bounds; derived fixed-command metric-normalization regularity |
| `OperationalCatalogBounds` | 4 | Combined actual conservative/normalized-drive/drag certificate and rational operational jet/force constants |
| `ConcreteOperationalAsset` | 23 | Literal 256-coefficient dyadic fixture, kernel-checked sample bounds, exact spacing, phase/halo margins and all 36 addresses |
| `ConcreteAssetFidelity` | 11 | Actual whole-halo value/differential/Hessian fidelity to the biased quadratic reference and Euclidean-gradient transfer |
| `RationalTrajectoryMesh` | 8 | Explicit rational horizon, bounded interpolation coefficient, all-mesh tube fit and dyadic error threshold |
| `ConcreteTrajectoryCertificate` | 17 | Actual fully instantiated force/acceleration, constructed reference, explicit horizon, perturbed and exact-real finite-run certificates |
| `PhaseEnergyBounds` | 7 | Bundled actual phase-energy differential and explicit norm/Lipschitz budgets from field and potential jets |
| `ConcreteEnergyCertificate` | 12 | Literal energy-regularity constants, contained-step Taylor bound and unconditional finite-run domain/remainder rates |
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
| `ACSubpaths` | 8 | Actual affine subpaths, derivatives and exact subinterval length |
| `DisplacementBounds` | 13 | Logarithmic displacement bound, positive definiteness and Shore-endpoint divergence |
| `IntrinsicMetric` | 7 | Symmetry, triangle inequality, bundled metric and endpoint divergence wrappers |
| `IntrinsicTopology` | 11 | Original topology, compact balls, properness, completeness and ruler distance control |
| `IntrinsicLength` | 6 | Cumulative actual graph length and intrinsic increment control |
| `GraphChord` | 10 | Lifted graph derivative/chord bounds and recovery of the full density |
| `IntrinsicVariation` | 20 | Actual intrinsic variation, partition bounds, AC and monotonicity |
| `IntrinsicGeodesics` | 5 | Constant-speed intrinsic geodesics and ordinary AC admissibility |
| `IntrinsicCurveLength` | 8 | Metric curve-length equality, length-space property and attained pairwise minimum |
| `ShoreMinimizers` | 17 | Complete nearest-Shore equality classification with pauses and zero branch |
| `MetricComparison` | 6 | Scale-only versus anisotropic lengths and actual path infima |
| `ACComponentPaths` | 8 | AC reachability throughout each component of an open domain |
| `ZeroFloorLimit` | 21 | Actual zero-floor integrability, quantitative infimum sandwich and same-component limit |
| `ZeroFloorGeodesics` | 11 | Attained actual zero-floor minimum, constant-speed geometry, selected-minimizer subsequential convergence and zero-floor length convergence |
| `CrossingAction` | 5 | Integral Cauchy--Schwarz, crossing duration and kinetic action |
| `FlatShore` | 32 | Scalar warp derivatives, curvature expressions, excursion integral |
| `FlatShoreGeometry` | 25 | Actual coordinate connection, torsion/compatibility, curvature commutator and Gram ratio |
| `FlatShoreIntrinsic` | 15 | Euclidean flat Shore, full AC coordinate/graph integral and path-infimum equality |
| `FlatShoreGeodesics` | 25 | Actual solution conservation laws, Shore geodesic and normal Jacobi equation/first zero |
| `FlatShoreSecondVariation` | 24 | Actual sine-curve length integral, second variation, AC fixed endpoints and strict shortening beyond normalized span pi |
| `FlatShoreShortcuts` | 11 | Unit-interval Euclidean competitor, exact graph length, physical dilation and actual pairwise shortcut above pi times epsilon |
| `ACScalarVariation` | 3 | Actual AC total variation equals signed increment exactly for monotone traces |
| `FlatShoreCalibration` | 13 | Actual regularized arccos calibration and global AC lower bound through normalized span pi |
| `FlatShoreMinimizers` | 6 | Actual normalized path infimum and all AC equality cases at and below the threshold |
| `FlatShoreIsometries` | 9 | Full AC candidate-length set and distance invariance under dilation and tangential translation |
| `FlatShoreThreshold` | 5 | Exact physical shortcut threshold and all AC equality cases, for arbitrary tangential source |
| `CoordinateCovariance` | 15 | Actual affine derivative, connection, bracket and curvature covariance; flat Gram-ratio specialization |
| `NonlinearCoordinates` | 14 | Actual inverse-Jacobian calculus, nonlinear connection/bracket/curvature covariance and metric transport |
| `CurvatureTensoriality` | 7 | Actual coefficient expansion, continuous trilinear curvature tensor and field-extension independence |
| `FlatCurvatureTensor` | 24 | Smooth actual flat connection, complete curvature tensor and arbitrary regular-frame ratio, including nonlinear pullback |
| `MovingFrames` | 13 | Actual inverse-frame derivative, connection/commutator gauge laws, frame composition and coefficient regularity |
| `CoordinateConnectionRegularity` | 5 | Bundled actual nonlinear pullback, C3-chart/C1-coefficient regularity and a derived invertible neighborhood |
| `ConnectionFormCurvature` | 8 | Actual fiber-valued continuous trilinear curvature tensor, gauge conjugacy and tangent-coordinate compatibility |
| `CurvedShoreGeometry` | 19 | Curved normal-chart metric, connection laws, actual compatibility and Shore curvature |
| `CurvedShoreCurvature` | 5 | Full off-Shore commutator curvature and actual normal curvature derivative at Shore |
| `TubularNeighborhood` | 9 | Normal-chord distance/uniqueness certificate, concrete sphere instance and strip injectivity |
| `EmbeddedShoreMetric` | 18 | Actual Frenet map/IFT chart, ambient distance and ruler derivative, actual metric pullback |
| `NormalRadius` | 12 | Derived normal-chord radii from actual curve jets, tangent/chord separation, compact remote separation and injectivity |
| `EmbeddedNormalPatch` | 12 | Derived local normal patch, compact-injective curve instance, actual distance/unique foot and localized IFT metric bridge |
| `MetricConnectionUniqueness` | 3 | Koszul identity and uniqueness from torsion freedom, compatibility and a positive metric |
| `EmbeddedShoreConnection` | 17 | Actual ambient regularized-distance jets and derived nonlinear normal-coordinate connection identity |
| `EmbeddedShoreCurvature` | 10 | Actual ambient curvature transport, positive normal Gram ratio, full curved formula and Shore value +1 |
| `SensitivityBounds` | 4 | Sensitivity transfer and scalar shear |
| `NormalizedDistanceFields` | 8 | Normalized-field algebra and Lipschitz bounds |
| `NormalizedMapDifferential` | 8 | Actual window differential and quadratic remainder from segment Hessian bounds |
| `NormalizedProviderError` | 18 | Actual computed-query error, recomputed ruler, scale/center stability and crest bias |
| `ProviderCertificates` | 25 | Actual Taylor and central-stencil error bounds, query/denominator perturbations and Euclidean aggregation |
| `OperationalField` | 22 | Actual quintic Horner jets, tensor-patch derivatives, weight moments and positivity |
| `OperationalFieldBounds` | 9 | Raw quadratic sampling bias, exact quadratic jets and coefficient/value stability |
| `OperationalFieldJetError` | 16 | Concrete derivative Bernstein tables and spacing-scaled coefficient/jet bounds |
| `OperationalFieldQueryError` | 11 | Coefficient-derived operator bounds, same-cell real-query and combined value errors |
| `OperationalFieldGluing` | 5 | Actual continuity and Frechet derivative gluing for mathematical floor selection |
| `OperationalFieldJoins` | 7 | Higher Horner derivatives and concrete coherent-lattice endpoint joins through order four |
| `OperationalFieldLattice` | 4 | Actual tensor-lattice patch jets and matching along both coordinate boundaries |
| `GlobalOperationalField` | 11 | Actual floor-selected global jets, gradient/Hessian identification and C4 regularity |
| `OperationalFieldQueryCorrespondence` | 17 | Physical normalization, selected query correspondence, ideal full-halo addresses and finite-asset extension independence |
| `OperationalFieldGlobalError` | 7 | Global coefficient-derived derivative bounds and cross-cell real-query error propagation |
| `OperationalFieldFidelity` | 21 | Actual Taylor/sample remainder, quadratic value bias and concrete reference jet error bounds |
| `OperationalFieldReference` | 9 | Selected physical sample accuracy and literal global value/first/second derivative fidelity |
| `OperationalFieldHessianError` | 9 | Concrete third-basis bounds and global Hessian query gain 32B/h³ |
| `OperationalFieldComputedError` | 1 | Combined reference, actual-query and reported-evaluation value/gradient/Hessian errors |
| `CovariantStatistics` | 12 | Finite transport algebra and a PSD counterexample |
| `ConcreteLocalFieldBounds` | 27 | Actual local Euclidean field jets and cross-cell Hessian variation from quadratic-reference cancellation |
| `LocalRulerBounds` | 12 | Local scalar regularizer coefficients, actual Hessian variation and convex mean-value ruler/gradient bounds |
| `RankOneInverseBounds` | 10 | Inverse rank-one contraction, exact resolvent identity and sharp regional inverse-action budgets |
| `CancelledCatalogBounds` | 7 | Actual raised-catalog cancellation, separate connection budget and local force/acceleration constants |
| `LocalTrajectoryMesh` | 6 | Separate speed/Lipschitz constants, horizon 1/576, explicit interpolation coefficient and 2048-step target |
| `ImprovedConcreteTrajectory` | 18 | Actual local jets, force and acceleration; constructed reference, all-mesh/query control and unconditional longer exact-real run |
| `HeldControlResidual` | 10 | Actual frozen-drive catalog comparison, sample-covector error and curve/stage displacement residual bounds |
| `ConcreteHeldControl` | 7 | Fully instantiated frozen-drive acceleration and numerical residual coefficients for the literal fixture |
| `ImprovedConcreteEnergy` | 18 | Actual energy differential pair (5,134), contained-step Taylor bound, finite-run remainder and accumulated Taylor ledger |
| `RungeKuttaStages` | 18 | Actual classical RK4 stages, enclosing-ball bootstrap, repeated substep containment and first-order consistency |
| `RungeKuttaPhase` | 14 | Exact telescoping stage averages and actual acceleration/stored-step residuals for any positive internal count |
| `ConcreteRungeKuttaBounds` | 6 | Actual held-field anchor variation, stage-ball speed bootstrap and smaller reference tube |
| `RungeKuttaMesh` | 6 | Held-RK error coefficient at most 108, stage/domain mesh budgets and explicit dyadic error target |
| `ConcreteRungeKutta` | 7 | Actual held-control RK recurrence, all-time finite-run certificate, stored-state containment and uniform convergence |
| `ConcreteRungeKuttaQueries` | 8 | All actual internal stages and weighted endpoints, plus final stored endpoints, retain safe mathematical query addresses |
| `ConcreteHeldPower` | 7 | Actual arbitrary-held-covector energy/power identity and concrete energy, target-field and target-power regularity |
| `RungeKuttaWork` | 8 | Actual signed stage-power quadrature, one-step energy defect and telescoped internal work with derived stage domains |
| `RungeKuttaWorkMesh` | 5 | Exact fixture ledger coefficients, all-prefix mesh rate and all-time work interpolation bounds |
| `ConcreteRungeKuttaWork` | 6 | Actual outer work and accumulated signed history; unconditional prefix/final ledger bounds and continuous interpolation |
| `ConcreteWorkIntegral` | 5 | Actual reference power trace, interval integrability, signed FTC work identity and cumulative-work Lipschitz bound |
| `ConcreteRungeKuttaWorkConvergence` | 3 | Grid work error, one common phase/work certificate and uniform convergence to the actual power integral |
| `PerturbedRungeKutta` | 8 | Explicit observed RHS, stage and endpoint errors; sequential stage containment and whole-step consistency |
| `PerturbedRungeKuttaWork` | 5 | Actual recorded power and mechanical stages, signed energy/work defect and independent stored-ledger accumulation |
| `ConcretePerturbedRungeKutta` | 12 | Nonzero local operation budgets, all-time phase and query containment, and a common reference for every admitted one-substep history |
| `ConcretePerturbedRungeKuttaWork` | 7 | Concrete local and prefix ledger bounds, actual integral comparison, and common-reference uniform phase/work convergence under shrinking local budgets |
| `MainCardioidRuler` | 10 | Multiplier-map derivative, chord distance bounds, tangent speed, and exact cusp-axis distance and ruler ratio |
| `OrbitTrapSteering` | 8 | Derivative, unit conformal speed, tangency, critical set, and saddle identity for the second-iterate point trap |
| `ConformalFloorCurvature` | 7 | Log-floor derivatives, curvature correction, and main-cardioid squared-gradient bound |
| `OrbitTrapConfidence` | 8 | Sharp dot/determinant cone and exact squared-norm gate over a gradient error ball |
| `QuarticRulerFloor` | 7 | Log-floor derivatives, quartic curvature identity, and scalar curvature bounds |
| `OrbitTrapDrift` | 11 | Exact trap increment, endpoint remainder, gradient-error contribution, and accumulated drift |
| `PeriodTwoAttachment` | 13 | Period-two multiplier and circle distance, bulb ruler, and conditional quartic $C^2$ metric gluing |
| `OrbitTrapContours` | 15 | Anchored square-root lift, exact trap level preservation, branch guard, tangent derivative and conformal speed |
| `ExteriorRulerBounds` | 10 | Escape growth, weighted potential and derivative increments, finite tail envelope, and ratio bounds |
| `OrbitTrapCurvature` | 4 | Second derivative of the anchored lift and normalized imaginary component for the heading rate |
| `ExteriorRulerJets` | 11 | Common escape disk, uniform holomorphic tail, and Cauchy bounds for its center derivatives |
| `ShoreFloorGluing` | 6 | Zero derivatives and $C^3$ gluing from uniform distance-power bounds |
| `TrapSteeringBudget` | 9 | Squared-speed acceleration components, braking bounds, and stop-to-stop schedule |
| `TrapMetricComparison` | 3 | Planar Gram identity and metric-normal acceleration at unit speed |
| `SensitivityResponse` | 5 | Actual least-squares second derivative, residual correction, and composed action response |
| `NormalizedMapLimits` | 4 | Anchored pointwise distance-function limits and affine reconstruction from symmetric samples |
| `CovariantHolonomy` | 6 | Unit-phase kernel determinant, PSD criterion, common reference, rank one, and integer harmonics |
| `ConformalRulerContracts` | 22 | Density and inland comparisons, conformal specialization, planar curvature and ideal normal Jacobi first zeros |
| `ExteriorNavigation` | 5 | Exterior ray and equipotential rates, speed, and error under an approximate logarithmic derivative |
| `OrbitTrapSelection` | 9 | Sharp finite selection margins, local regularity of a winning branch, ties, and critical return periods |
| `IntrinsicTargetBounds` | 18 | Target-set infima, consistent lower bounds, tolerance adjustment, and search radius |
| `ShoreCrossingWorkBound` | 8 | Force and path-length work caps, crest energy deficits, and impulse-adjusted travel |
| `StableRadialProfile` | 10 | Exact positive-sum radial formula, zero cases, and scaled second argument |
| `TwoEndedDisplacement` | 8 | Stronger displacement bound, compatible ruler enclosures, and fixed-goal consistency |
| **Result-module subtotal** | **1,766** | Includes supporting lemmas |
| `Analysis/*` (fourteen modules) | 55 | Attributed AC FTC backport and project analysis/metric/integral support |
| **Total** | **1,825** | Explicit theorem and lemma declarations |

The analysis-support modules contribute respectively 8 (`AbsolutelyContinuous`),
6 (`AbsolutelyContinuousFTC`), 4 (`DerivIntegrable`), 2 (`Fatou`),
1 (`FiniteDimensionalAC`), 3 (`FinsetGaps`), 3 (`IntervalSlope`), and
12 (`OrderFinsetGaps`), 4 (`ACComposition`), 2 (`ACVectorFTC`),
1 (`DerivativeControl`), 7 (`ProperGeodesics`), and 1
(`CompactParametricIntegral`). The
[support provenance](FractalSync/Analysis/README.md) records the exact upstream
snapshot and retained Apache 2.0 attribution. The AC composition/gluing,
vector FTC, derivative-control, proper-geodesic and compact parametric-integral
proofs are separate project support work, not attributed upstream declarations.

The additional `BoundedSpeedCompactness` module contributes one theorem using
the pinned bounded-continuous-function Arzelà–Ascoli and compact sequential
extraction APIs. It upgrades the existing bounded-speed endpoint family to
uniform subsequence convergence. It is project support, not an upstream
backport or a new compactness assumption.

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
inventoried below. The explicit potential-force and additional control
contracts are now checked by the following extension.

### #159: explicit potentials, controls, and discrete step contracts

[PotentialForces.lean](FractalSync/PotentialForces.lean) uses namespace
`FractalSync.PotentialForces`; [ControlMechanics.lean](FractalSync/ControlMechanics.lean)
uses `FractalSync.ControlMechanics`. The full formulas and implementation
boundary are in [the potential/control note](../docs/research/159-potential-controls.md).

| Checked result | Principal declarations and scope |
| --- | --- |
| Actual Shore potential differential and inverse-metric acceleration | `hasDerivAt_shorePotential`, `hasFDerivAt_shorePotential`, `shore_acceleration`; positive ruler/reference, actual ruler derivative, arbitrary real a and gradient. The acceleration is kappa*rho*u/(log 2*W). |
| Reference origin and exact crest potential change | `shorePotential_reference_shift`, `shorePotential_difference`, `totalPotential_reference_shift`, `totalPotential_fderiv_reference_invariant`, `totalPotential_crest_difference`; the full wall difference is retained. |
| Actual p=8 secant-wall differential and physical domain | `hasFDerivAt_wallCoordinate`, `hasFDerivAt_wallPhase`, `hasFDerivAt_wallPotential_of_cos_ne_zero`, `hasFDerivAt_wallPotential`, `wallPhase_mem`, `cos_wallPhase_pos`; norm(x)<2 ensures the cosine denominator is positive. |
| Wall sign, default strength, metric raising and boundary limit | `wallPotential_nonneg`, `wallCoefficient_nonneg`, `wallForce_radial_nonpos`, `wallForce_default`, `wall_acceleration`, `wallPotential_tendsto_atTop`. Nonnegative mu for signs and strictly positive mu for divergence; inward covector does not mean radial raised acceleration. |
| Actual total conservative force and continuous energy ledger | `hasFDerivAt_totalPotential`, `hasDerivAt_canonicalEnergy`, `canonical_work_energy`, `crest_budget_of_work_energy`, `canonical_crest_budget`. The last derives the necessary crest budget directly from trajectory derivatives and dynamics, with signed work, wall change and friction integral explicit. |
| Actual momentum differential and impulse momentum jump | `hasFDerivAt_kinetic`, `hasFDerivAt_lagrangian_velocity`, `momentum_impulse_jump`; the velocity derivative is the existing metric action Gv. |
| Metric normalization, drive and drag | `metricNormalize_unit`, `drive_acceleration`, `drive_power`, `drive_dualSq`, `drive_dualNorm_unit`, `drive_power_bound`, `metric_drag_acceleration`, `metric_drag_power_nonpos`; unit heading, nonzero vector and metric-scale conditions are explicit where needed. |
| Candidate board friction and PSD dissipation | `boardFriction_apply`, `boardFriction_quadratic`, `boardFriction_nonneg`, `board_friction_power_nonpos`, `board_friction_acceleration`, `friction_integral_nonneg`. Nonnegative coefficients suffice even for degenerate frames. |
| Candidate projection, carve and fixed-metric steering | `headingProjection_idempotent`, `headingProjection_residual_orthogonal`, `headingProjection_tangent`, `carveBias_ne_zero`, `normalized_carveBias_unit`, `hasDerivAt_steering_angle`, `hasDerivAt_steering`, `steering_preserves_unit`, `redirection_impulse_energy_jump`; 0<=alpha<1 keeps carve bias nonzero. |

The continuous ledger requires actual field/trajectory derivatives on the
open time interval, endpoint energy continuity and integrable work terms.
PSD is separate from the ledger identity. Candidate board/carve formulas
do not change the runtime's isotropic controls path. Fixed-metric steering
does not prove covariant heading transport in a varying metric.

[DiscreteMechanics.lean](FractalSync/DiscreteMechanics.lean), in
`FractalSync.ScaleMetric`, proves the actual frozen-metric velocity-step
identities `velocityEulerStep_energy_change`, `frozenForceStep_energy_change`,
`frozenForceStep_work_defect_nonneg` and `frozenForceStep_average_velocity_work`.
The drag theorems `explicitDragStep_eq_smul`, `explicitDragStep_kinetic`,
`explicitDragStep_kinetic_le`, `explicitDragStep_kinetic_le_iff_of_ne_zero`,
and `explicitDragStep_kinetic_le_iff` give the exact factor (1-beta*dt)^2
and sharp nonincrease condition v=0 or 0<=beta*dt<=2, with rho>0.
See [the discrete note](../docs/research/159-discrete-mechanics.md).
These are not changing-configuration, floating-point or full-integrator
conservation theorems.

### #159: changing-configuration steps and covariant steering

[MovingMetricMechanics.lean](FractalSync/MovingMetricMechanics.lean) and
[CovariantSteering.lean](FractalSync/CovariantSteering.lean) extend
`FractalSync.ScaleMetric`. The full hypotheses and runtime ordering are in
[the moving-metric note](../docs/research/159-moving-metric-dynamics.md).

| Checked statement | Principal identifiers and conditions |
| --- | --- |
| Actual phase-energy differential and physical work cancellation | `differentiableAt_phaseEnergy`, `hasDerivAt_phaseEnergy_line`, `fderiv_phaseEnergy_apply`, `fderiv_phaseEnergy_motion`; actual ruler, gradient-field and potential derivatives, nonzero ruler, and the stated dynamics. |
| Exact velocity-then-position increment and its first-order energy | `positionVelocityStep_eq_add`, `positionVelocityStep_increment`, `fderiv_phaseEnergy_step_increment`; the extra position increment is dt² times acceleration. |
| Quadratic local defect for energy at the new configuration | `positionVelocityStep_energy_defect`, `positionVelocityStep_energy_defect_of_hessian_bound`; actual phase-energy differential variation or Hessian certificate on the entire proposed segment. The phase product uses the max norm. |
| Explicit coefficient and exact metric/potential decomposition | `positionEnergyRate_eq_connection`, `positionEnergyRate_expanded`, `abs_positionEnergyRate_le`, `positionVelocityStep_energy_decomposition`; the coefficient bound uses positive ruler and actual operator norms. |
| Whole proposed position segment stays in an open ball | `affine_segment_norm_lt`, `positionVelocityStep_segment_norm_lt`, `positionVelocityStep_segment_norm_lt_of_acceleration_bound`; strict margin norm(x)+abs(dt)*norm(v)+dt²*norm(acc)<R suffices. |
| Impulse ledger at the new position | `impulseAtPhase_energy_change`, `positionVelocityImpulse_energy_ledger`; the impulse inverse metric is evaluated at the updated configuration. |
| Metric-skew frame action and actual moving pairing derivative | `frameTurn_orthogonal`, `frameTurn_first`, `frameTurn_second`, `frameTurn_mix`, `hasDerivAt_metric_pairing_covariant`; orthonormal-frame interpretation is confined to its two-plane. |
| Conditional continuous steering preserves metric norm | `hasDerivAt_metric_self_steering`, `metric_self_steering_eq`, `hasDerivAt_metric_self_steering_of_fields`; actual trace/field derivatives, the covariant ODE and endpoint continuity. Frames and turn rates may vary. |

These are exact-real results. The phase-Hessian certificate may require
higher field regularity than the ruler Hessian alone. Neither a numerical
provider flag nor positive regularization supplies it. The segment condition
covers the proposed affine interpolation, not an exact ODE trajectory or an
unconditional rollout. The following extension handles finite histories and
a discrete steering candidate. Solution existence, global trajectory error,
and IEEE correspondence remain separate.

### #159: accumulated energy, compact regularity, and discrete steering

[AccumulatedDynamics.lean](FractalSync/AccumulatedDynamics.lean),
[PhaseEnergyRegularity.lean](FractalSync/PhaseEnergyRegularity.lean), and
[DiscreteSteering.lean](FractalSync/DiscreteSteering.lean) extend the same
`FractalSync.ScaleMetric` namespace. The
[accumulated-dynamics note](../docs/research/159-accumulated-dynamics.md)
states the exact finite recurrence and assumptions.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Actual residual work-energy identity and residual-aware local step error | `fderiv_phaseEnergy_motion_residual`, `fderiv_phaseEnergy_step_increment_residual`, `positionVelocityStep_energy_defect_residual`; the residual is defined from actual acceleration, connection, raised force, and potential gradient. |
| Explicit residual and power estimates | `abs_metric_le_norm`, `dynamicsResidual_norm_le`, `dynamicsResidual_power_le`; actual norm error in the force and connection estimates produces a power-error budget. |
| Actual finite-history energy/work estimate including impulses | `StepJetCertificate.energy_defect`, `StepJetCertificate.impulse_energy_defect`, `positionVelocityImpulse_energy_defect_sum`; actual initial jets and segment differential variation, with impulses evaluated at the updated positions. |
| Summed error and whole-history position-domain budget | `sum_step_budget_le` gives T*(eta+hmax*C); `position_steps_norm_lt` covers every proposed affine position segment from a strict total-displacement bound. |
| Existence of uniform phase-energy regularity | `contDiffAt_phaseEnergy`, `exists_phaseEnergy_hessian_bound_on_compact`, `exists_phaseEnergy_lipschitz_fderiv_on_compact`; actual C² fields, nonzero ruler, compact phase region, and convexity for the Lipschitz conclusion. |
| Regional construction of the actual step certificate | `StepJetCertificate.ofLipschitzRegion`; actual initial jets and both phase endpoints in the assumed convex region. |
| Discrete candidate with exact endpoint metric unit norm | `covariantSteeringStep_unit`, `steering_trial_ne_zero`, `covariantSteeringStep_zero`; endpoint normalization and a strict nonvanishing bound on the Euler trial. |
| Actual first-order consistency of that candidate | `hasDerivAt_covariantSteeringStep`, `hasDerivAt_covariantSteeringStep_of_fields`; differentiating the moving-metric normalization recovers the covariant rate at zero step. |

Compactness proves the existence of a finite uniform derivative bound; it does
not compute its numerical value. The phase-region hypotheses include velocity
control, beyond the separate position-domain budget. Energy/work accounting
does not itself establish state-error convergence; that is supplied under
separate ODE/region hypotheses by the following extension. The discrete steering
candidate does not constitute an implementation or an IEEE guarantee.

### #159: actual trajectory error, phase persistence, and refinement

[TrajectoryError.lean](FractalSync/TrajectoryError.lean) and
[TrajectoryRefinement.lean](FractalSync/TrajectoryRefinement.lean) extend the
existing velocity-then-position step, using the phase product maximum norm.
The [trajectory note](../docs/research/159-trajectory-error.md) states the
actual ODE and conditional provider/rounding contracts.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Stability of the actual ordered step | `orderedStep_eq`, `phaseField_sub_le`, `orderedStep_stability`; amplification is 1+abs(h)*max(1,L)+h²*L from an acceleration Lipschitz bound. |
| Actual ODE truncation defect | `curve_quadratic_remainder`, `orderedStep_ode_defect`; actual trajectory derivatives, phase-speed bound B and acceleration variation imply a defect <=h²*(B+max(1,L)*B/2). |
| Evaluation and stored-state perturbations | `positionVelocityStep_acceleration_error`, `roundedStep_ode_error`; acceleration residual eta contributes (abs(h)+h²)*eta, with rounding and truncation separately included. |
| Finite error envelope and noncircular domain induction | `trajectoryEnvelope_eq`, `ODERegionCertificate.trajectory_error_domain`; residual/rounding contracts are required only at states already proved to lie in the region. |
| Actual derivative bounds and position/velocity margins | `acceleration_sub_le_of_derivative_bound`, `phaseBox_mem_of_error`, `ODERegionCertificate.trajectory_error_phaseBox`, `proposed_segment_mem`; proposed affine phase segments are controlled when the region is convex. |
| Checked fixed-horizon refinement bound and limit | `trajectoryEnvelope_le_exp`, `trajectoryEnvelope_fixed_horizon_bound`, `tendsto_refinementEnvelope_zero`; initial error O(h), acceleration residual O(h), stored-state error O(h²). |
| Actual numerical histories converge uniformly at grid points | `UniformODECertificate.refinement_converges`; one actual reference ODE curve on [0,T], uniform field bounds and a positive reference-tube margin give convergence and eventual domain membership. |

The final convergence statement quantifies over every grid point in each
refinement. It does not assume numerical domain membership. It does assume
existence and regularity of the reference ODE solution and certified regional
constants. The following continuation constructs local reference solutions and
handles continuous interpolants. Fixed-precision IEEE, impulsive-history and
combined steering convergence remain separate.

### #159: actual local solutions and continuous numerical trajectories

[TrajectoryInterpolation.lean](FractalSync/TrajectoryInterpolation.lean),
[LocalDynamics.lean](FractalSync/LocalDynamics.lean), and
[OperationalDynamics.lean](FractalSync/OperationalDynamics.lean) connect actual
local existence with all-time numerical error bounds. The
[local-trajectory note](../docs/research/159-local-trajectories.md) states the
phase norm, refinement contracts, and scope of the reference solution.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Actual stored-state interpolation | `trajectoryInterpolant_grid`, `continuous_trajectoryInterpolant`; a finite piecewise-affine function, including both grid endpoints. |
| All-time interpolation error | `trajectoryInterpolant_error`, `UniformODECertificate.reference_lipschitz`; grid error e becomes e+B*h/2, with reference Lipschitz regularity derived from the ODE. |
| Explicit all-time domain/error bound and limit | `UniformODECertificate.interpolation_error_bound`, `interpolation_converges`; h*(refinementCoefficient+B/2) within the actual reference tube gives error and membership throughout [0,T], and uniform convergence under the mesh-scaled contracts. |
| Actual local existence and regional uniqueness | `exists_local_ODE_certificate`, `UniformODECertificate.solution_unique`; actual C1 acceleration yields T,R>0, a solution, a uniform certificate on the phase ball and a positive reference tube. |
| Actual mechanical coefficient adapter | `contDiffAt_mechanicalAcceleration`, `exists_local_mechanical_certificate`; C1 ruler/gradient/Hessian/force coefficients and a nonzero ruler. |
| Literal operational metric specialization | `OperationalDynamics.contDiff_rulerField`, `actual_jet_regularity`, `hasFDerivAt_rulerField`, `contDiffAt_acceleration`, `exists_local_certificate`; actual C4 spline, C3 Riesz gradient, C2 gradient derivative and actual local mechanics. |

The interpolant is a specified analysis function; no renderer interpolation or
position-derivative/velocity identity is inferred between nodes. The all-time
theorems require T>0; the earlier grid theorem retains T=0. The local ODE
constants are existential. The operational specialization uses its actual
derivatives and needs no signed-distance fidelity premise, but it does not
place the existential phase ball inside a finite asset's halo or compute
usable numerical constants. Positive physical spacing gives its intended
provider interpretation.

### #159/#162: explicit operational bounds and finite-asset local trajectories

The [ruler-bound note](../docs/research/159-operational-ruler-bounds.md),
[mechanical-bound note](../docs/research/159-operational-mechanical-bounds.md),
[quantitative local-existence note](../docs/research/159-quantitative-dynamics.md),
[physical-halo note](../docs/research/162-operational-halo.md), and
[integrated trajectory certificate](../docs/research/159-operational-trajectory-certificates.md)
connect literal finite coefficient evidence to actual local mechanics.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Uniform norm and Lipschitz arithmetic | `BoundedLipschitzOn.bilinear`, `inv`, `smul`, `inner`, `clm_apply`, `comp`; a pair `(M,L)` controls the actual function on a common region. Products give `(MN,MK+NL)` and reciprocal floors give `(1/epsilon,L/epsilon²)`. |
| Actual mechanical expression | `MechanicalBounds.acceleration_bound`; `accelerationBudget` propagates the input pairs through `inverseAction - connection(v,v)`. It uses rho>=epsilon>0 and W>=1, without assuming a unit gradient or convex region. |
| Actual Euclidean operational jets | `rulerField_bounds`, `norm_gradient_rulerField_le`, `norm_hessian_rulerField_le`, the three `*_sub_le` theorems, and `actual_third_rulerField_bound`; bounded literal coefficients, positive spacing and positive floor. The L2-to-coordinate-max contraction and Riesz isometry are explicit. |
| Finite asset and physical margin | `zeroExtension_agrees`, `abs_zeroExtension_le`, `reportedQuery_mem_of_phaseBall`, `stencil_indices_of_mem_halo`; R+delta below the actual four-sided physical margin controls every selected address, including across cell changes. |
| Actual derivative and acceleration locality | `rulerField_eventuallyEq`, `actual_metricJets_eq`, `acceleration_eventuallyEq`, `acceleration_zeroExtension_eqOn`; agreement of finite stored coefficients gives neighborhood equality of actual ruler jets and acceleration inside the open halo. |
| Finite-asset mechanical bounds | `finiteAsset_acceleration_boundedLipschitz`; `operationalAccelerationBudget` supplies both actual acceleration bounds on the supplied closed phase ball. Only stored coefficients are bounded; force norm and variation remain explicit inputs. |
| Constructed quantitative reference and mesh condition | `exists_quantitative_ODE_certificate`, `exists_quantitative_trajectory_control`; T=R/[4(B+1)], displacement <=Bt<R/4, a radius-R/2 reference tube, regional uniqueness, and whole-interpolant error/domain control when 2TC<=R(N+1). |
| Complete finite-asset trajectory construction | `exists_finiteAsset_ODE_certificate`, `exists_finiteAsset_trajectory_control`, `exists_finiteAsset_interpolation_converges`; one actual reference from literal finite-asset data controls every admitted mesh, with explicit constants, uniform convergence and all-time displaced-query address bounds. |

Writing G=8B/(3h), H=68B/(9h²), and J=32B/h³ for the raw spline gains,
the actual ruler satisfies epsilon<=rho<=sqrt(B²+epsilon²), norm(u)<=G,
norm(Du)<=H+G²/epsilon, and the Lipschitz constants of rho, u, and Du are
G, H+G²/epsilon, and J+3GH/epsilon+3G³/epsilon² respectively.
The actual derivative of Du has the last bound as well. These are conservative
exact-real expressions, not numerical measurements or IEEE certificates.

The integrated finite-asset theorem supplies the acceleration constants to
the quantitative construction, then controls every continuous interpolated
state and every reported query satisfying the stated displacement budget.
Mesh-scaled residual, stored-state rounding and initial-error hypotheses remain
explicit in these generic statements. The concrete fixture below discharges
the finite coefficient and catalog-force premises. Machine evaluation,
normalization/floor/casts/masks, impulse jumps, combined steering and long-time
extension remain separate. No fixed error floor is asserted to vanish under
refinement.

### #159/#162: concrete catalog, reference fidelity, energy and trajectory certificates

The [concrete trajectory note](../docs/research/159-concrete-trajectory-certificate.md)
instantiates the preceding interfaces on the actual literal nonconstant
16-by-16 quadratic test fixture. The
[asset/fidelity note](../docs/research/162-concrete-asset-instance.md),
[potential bounds](../docs/research/159-potential-force-bounds.md),
[control bounds](../docs/research/159-control-force-bounds.md), and
[energy bounds](../docs/research/159-phase-energy-bounds.md) provide the
separate proof and correspondence boundaries.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Actual Shore and secant-wall constants | `shoreForce_bound`, `shoreForce_log_two_bound`, `wallForce_bound`, `wallForce_bound_of_radius`, `totalForce_bound`; the radius r<2 gives an explicit positive cosine denominator floor. `wallForce_default_unit_bound` gives (1/16,Lx/2) at mu=1/pi on the unit position ball. |
| Actual normalized drive and drag | `metricAction_boundedLipschitz`, `driveForce_boundedLipschitz`, `dragForce_boundedLipschitz`; `metricNormalize_unit_formula`, `metricNormalize_pos_smul`, both normalization bounds and `normalizedDriveForce_boundedLipschitz` derive fixed-nonzero-command regularity from rho/u certificates. |
| Combined actual catalog | `catalogForce_boundedLipschitz` supplies a symbolic regional pair for conservative force, normalized drive and viscous drag. `catalogForce_rational_bound` gives (400000,4000000000000) for the stated nonzero parameter instance. `rational_ruler_budget` and `rational_operational_jets` justify all numerical ruler inputs. |
| Literal finite data and geometry | `numerator_bound`, `coefficient_bound`, `coefficients_bound`, `finite_coefficients_bound`, `coefficients_of_fin`, and `coefficients_values` certify the literal table and zero extension. Spacing, center/velocity norms, exact halo margin and `safe_reported_query` certify the phase/query region and all 36 mathematical addresses. |
| Concrete sample and actual field fidelity | `sample_error_numerator_bound` and `coefficient_sample_error` discharge all 256 sample inequalities. `finite_sample_error`, `selected_sample_error`, the three physical field/query error statements, the actual `fderiv` wrappers and `actual_reference_errors` transfer them throughout the full half-open halo, including grid lines. `euclidean_gradient_reference_error` performs the actual contractive chart and Riesz transfer. |
| Rational horizon and mesh | `refinementCoefficient_add_half_le`, the rational horizon/coefficient lemmas, `rationalTrajectoryMesh_fits`, `rationalTrajectoryMesh_error_le`, and `rationalTrajectoryMesh_error_le_65536`; B>=8 and T=1/[64(B+1)] give coefficient<=B, all-mesh tube fit and error<=1/[64(N+1)]. |
| Concrete actual solution and numerical history | `regional_jets`, `position_bound`, `velocity_bound`, `rho_floor`, `heading_ne_zero`, `force_bound`, `acc_bound`, `phaseField_bound`, and the horizon lemmas discharge all coefficient premises. `exists_reference` constructs the actual solution/tube; `exists_certified_trajectory` includes regional uniqueness and all-time perturbed-mesh/query control. `exists_exact_scheme_certificate` and `exists_1024_step_certificate` instantiate the actual recursive exact-real update without numerical residual premises. |
| Explicit actual energy differential | `fderiv_phaseEnergy_eq` identifies a bundled continuous linear covector. `innerSL_bound`, `covector_comp_bound`, `phase_precomp_bound`, `positionCovector_bound`, `energyCovector_bound`, and `fderiv_phaseEnergy_bound` propagate the norm/Lipschitz pair. `stepJetCertificate` is a proved constructor with the computed regularity constant. |
| Concrete changing-metric energy | `ruler_jet`, `gradient_jet`, `potential_jet`, `potentialGradient_bound`, `energy_differential_bound`, and `phaseEnergy_differentiable` instantiate actual field and potential derivatives. `energyStepCertificate`, its literal-L theorem, and `energy_step_taylor_remainder` give a numerical Taylor remainder for contained actual updates. The finite-run corollaries discharge endpoint inclusion for the certified exact-real history. |

The fixture spacing is the exact dyadic value of binary64 `0.1`, not exactly
1/10. The field reference is P-h² for P(x,y)=x²+2xy-3y², retaining raw-sample
quintic bias. On the full physical halo the actual value, differential and
second derivative errors are at most 2^-48, 2^-43 and 2^-38. Differential/Hessian
operator norms use the physical coordinate maximum norm. A separate checked
corollary gives the same 2^-43 bound for the Euclidean gradient.

The concrete dynamics use epsilon=a=F=beta=dref=1, kappa=log 2, mu=1/pi,
x0=(1/2,1/2), v0=(1/4,0), R=1/16, and reported-query displacement delta=1/64.
The force pair is (400000,4e12), actual acceleration pair (1.1e12,4e19), and
shared speed/Lipschitz bound B=4e19. The checked horizon is exactly
1/2560000000000000000064, approximately 3.90625e-22 in model time. Every admitted
m-step interpolant has phase error at most 1/(64m); 1024 steps give 2^-16.
The exact-real velocity-first recursion discharges its residuals by definition.
The energy differential pair is (2e7,2.4e10), giving a contained-step quadratic
remainder coefficient 1.2e10 times the squared phase-increment speed.

The horizon is derived from coarse proof majorants, not measured stiffness or
a practical runtime recommendation. The source extraction models separately rounded,
non-fused binary64 operations; exact rational rounding cross-checks remain
external evidence, while the literal/sample proofs use kernel reduction.
No Rust execution or compiler-contraction equivalence is claimed. Production
Mandelbrot fidelity, actual query/IEEE errors and masks/storage/machine casts
remain open. The later held-RK extension below now connects an exact-real
outer-sampled drive policy to the reevaluated-command reference, under the
fixture's stated caller and parameter contract.

### #159/#162: local fixture bounds, a longer horizon and held control

`ConcreteLocalFieldBounds` subtracts the ideal quadratic Hessian before
estimating the finite fixture's variation. A globally bounded zero extension
of the sample errors gives actual cross-cell Hessian variation at most
32*2^-48/h³. On the closed Euclidean position ball of radius 1/16, the actual
field, differential, Hessian and Hessian Lipschitz bounds are respectively
1/4, 4, 12 and 1. These bounds retain the raw-sample reference bias P-h².

`LocalRulerBounds` uses the small field interval in the scalar regularizer's
third derivative. Its actual chain rule and Riesz transfer give local pairs
rho=(2,1), grad rho=(1,20), D grad rho=(20,200), with rho>=1. The gradient and
ruler Lipschitz bounds use convex mean-value estimates on the same region.

`RankOneInverseBounds` proves the inverse rank-one contraction and its exact
resolvent variation identity. `CancelledCatalogBounds` raises the Shore,
normalized drive and drag terms before estimating them. The actual unraised
catalog has pair (6,210); the actual changing-metric acceleration has pair
(8,400). The norm and Lipschitz constants are propagated separately.

`LocalTrajectoryMesh` and `ImprovedConcreteTrajectory` construct the reference
at **T=1/576**, retain regional uniqueness and the interior tube, and prove
all-time phase error<=1/(36m) for m steps under the explicit unit numerical
error coefficients. The actual exact-real velocity-first recursion supplies
an unconditional 2048-step certificate with uniform phase error<=2^-16.
Every admitted query within 1/64 of the interpolated position retains all
36 valid mathematical stencil addresses throughout the interval.

`HeldControlResidual` and `ConcreteHeldControl` quantify the acceleration
discrepancy when the supplied normalized-drive covector is frozen at a sample
state while potential, drag and connection are reevaluated. The concrete bound
is 680 times phase displacement, plus 4 times any admitted initial covector
error. Displacement<=8*tau gives the holding term 5440*tau. This needs a
stage/curve displacement certificate and is separate from the unit-residual
mesh specialization; it is not a runtime RK4 or floating-point error proof.

`ImprovedConcreteEnergy` uses the same actual potential and new local jets to
bound the actual phase-energy differential by (5,134). The Taylor coefficient
is 67 for contained steps. The longer exact-real run supplies the endpoint
conditions needed by the finite-run energy statements; the retained linear
differential terms account for actual energy changes.

See [the local-horizon note](../docs/research/159-local-horizon.md),
[the held-control note](../docs/research/159-held-control-residual.md), and
[the local-energy note](../docs/research/159-local-energy-bounds.md).
Production distance fidelity, IEEE errors, hybrid controls and continuation
remain separate. The exact-real RK stage/domain and residual bridge is the
next extension below.

### #159: actual held-control Runge--Kutta histories

The preceding six RK modules close the exact-real numerical bridge, using the
same fixture and reference horizon T=1/576. Their full statements and source
mapping are in [the held-RK note](../docs/research/159-held-runge-kutta.md).

`RungeKuttaStages` defines actual classical RK4 stages, their weighted endpoint,
and repeated equal substeps. A regional speed bound establishes every stage
and endpoint's membership by induction. It also checks the source-anchored
Euler-comparison bound LMh²/2. `RungeKuttaPhase` telescopes the actual increments
and derives both residuals for any positive internal substep count, retaining
one held field and one outer anchor.

`ConcreteRungeKuttaBounds` derives the actual anchored phase/acceleration
variation coefficient 1080=680+400. On a stage ball of radius 16h about an
inner anchor, the held field is bounded by 16 when h<=1/2160. The original
reference has a closed tube of radius R/4 inside the smaller open ball R/2,
leaving room for all stage queries in the original ball R=1/16.

`ConcreteRungeKutta` defines the literal outer recurrence: sample the exact
unit drive at each outer anchor and hold it through n>0 internal RK substeps.
Its actual averaged acceleration differs from the target by at most 17280h;
the exact RK endpoint differs from the velocity-first comparison by at most
32h². The latter is a method-comparison residual, not an IEEE error budget.
`RungeKuttaMesh` gives interpolation coefficient<=108 and, for m>=16 outer
steps, all-time phase error<=3/(16m). The phase norm is the maximum of the
Euclidean position and velocity norms. The 16384-step target<=2^-16 and the
quantified uniform-convergence corollary are checked for every positive n,
including one and four, with the same reference solution.

`ConcreteRungeKuttaQueries` explicitly covers the source, stages2/3/4 and
weighted endpoint of every internal substep, plus every outer endpoint.
Reported coordinates with maximum coordinate error<=1/64 retain all 36
mathematical stencil addresses. The same allowance holds at every time of
the piecewise-affine outer-state interpolant. This is not RK dense output.

The caller must supply the fixture's exact drive at each outer anchor. The
runtime accepts a supplied covector and does not enforce that policy. No
default-profile, floating-point or provider-acceptance equivalence is inferred.
The signed work extension below supplies the separate exact-real quadrature
certificate. The checked convergence rate is first order for
the held-policy comparison; there is no fourth-order or four-substep speedup
claim. The [field-replacement note](../docs/research/157-field-replacement-boundary.md)
separates these general-field numerical arguments from the actual
distance-to-set calibration and geometry below.

### #159: actual signed RK work and its continuous integral limit

The [work-ledger note](../docs/research/159-rk-work-ledger.md) connects the
runtime's separate power accumulator to the same held-RK mechanical stages.
The energy is the existing kinetic-plus-conservative-potential function.
Signed power includes both supplied drive and viscous drag; it is evaluated
at the actual stage state with the same outer-held covector.

| Module and main identifiers | Checked statement |
| --- | --- |
| `ConcreteHeldPower.fderiv_energy_heldField`, `fderiv_energy_targetField` | Actual DE(z)[F_q(z)]=inner(v,q-Gv), for arbitrary supplied q and the unit-drag fixture; the reference specializes q to the drive at z. |
| `ConcreteHeldPower.energy_differential_bound`, `targetPower_bound` | Actual energy differential norm/Lipschitz pair (5,134), reference field pair (8,400), and actual power pair (40,3072) on the certified phase ball. |
| `RungeKutta.step_energy_work_defect`, `substeps_energy_work_defect`, `outer_energy_work_defect` | Actual weighted power quadrature differs from endpoint energy change by at most KM²s² per substep and KM²h²/n over n>0 substeps; the regional wrapper derives all stages from its field bound. |
| `ConcreteRungeKutta.workTrajectory_energy_defect` | Every prefix k<=m of the actual finite history has absolute signed energy/work defect<=1/(8mn), with m>=16 and any n>0; numerical region and actual-power premises are discharged. |
| `ConcreteWorkIntegral.referenceWork_eq_energy_sub`, `referenceWork_lipschitzOn` | The actual integral from 0 to t of reference drive-plus-drag power equals E(z(t))-E(z(0)) and is 40-Lipschitz on the certified interval. |
| `ConcreteRungeKutta.exists_phase_work_certificate`, `exists_uniformly_convergent_phase_work` | One reference works for all meshes and positive internal counts; affine work interpolation has error<=1/(8mn)+35/(36m), and both phase and cumulative work converge uniformly on [0,1/576], uniformly in n. |

The fixture coefficient is 134*16²=34304 and 34304/576²=67/648<1/8.
At m=16384, the prefix and final energy/work bounds are 2^-17 for n=1
and 2^-19 for n=4. The 1/n improvement concerns this certified bound;
internal refinement alone does not remove the outer holding-policy error
relative to the continuously reevaluated command. Grid work comparison adds
15/(16m) from the phase error and energy Lipschitz bound. Affine interpolation
adds 40h/2, producing the stated all-time coefficient and a common upper
bound 9/(8m) for all n>=1.

These results concern actual physical-work samples, distinct from the earlier
Taylor ledger retaining DE(y_i)[y_{i+1}-y_i]. They do not bound total driven
energy change or certify conservation, instantaneous-power convergence,
RK dense output, IEEE power evaluation/summation or compiled execution.
The runtime has no energy/work-residual acceptance tolerance. Field/profile,
caller policy, local horizon and production-provider boundaries remain as
documented; no new distance-field hypothesis is used by the generic work proof.

### #159: local RK operation errors and complete observed histories

The [operation-error note](../docs/research/159-rk-operation-errors.md) records
the final conditional numerical bridge. `RungeKutta.Perturbed.Residuals` bounds
individual RHS, stage-construction and endpoint errors. Its `stageBounds`
theorem derives containment in evaluation order, then `euler_defect_le`
converts those local errors into a completed-step estimate. `Powers` and
`PowerResiduals` use the identical recorded mechanical states, and
`energy_work_defect` retains actual signed power and work-update arithmetic.

For the existing fixture, `ConcretePerturbedRungeKutta.History` permits RHS
errors at most h and stage/endpoint errors at most h², with h=(1/576)/m and
m>=16. `WorkHistory` permits power errors at most h and a combined weighted
work/accumulation residual at most h². All allowances are nonzero at a finite
mesh; the initial phase and zero initial work remain exact.

| Main identifier | Checked statement |
| --- | --- |
| `observed_stage_bounds`, `observed_stage_region` | Recorded stages and endpoint stay in the 18h ball around the source, from the actual field bound and local error contracts. |
| `ordered_defect`, `exists_history_certificate` | Local operation errors imply completed-step defect<=18000h², then all-time phase error<=3/(16m) and inner-region containment. |
| `history_stage_safe_query` | Every recorded stage and endpoint retains the existing mathematical query allowance 1/64. |
| `observed_work_defect`, `WorkHistory.energy_defect` | Local energy/work defect<=65536h² and every-prefix signed ledger defect<=1/(4m). |
| `exists_phase_work_certificate`, `exists_uniformly_convergent_phase_work` | One reference precedes all meshes and admitted histories; all-time work error<=11/(9m) against its actual power integral, with conditional uniform phase/work convergence. |

The last five rows use namespace `FractalSync.ConcretePerturbedRungeKutta`.
This extension covers one RK4 substep per outer interval. The prior exact-real
results for arbitrary positive internal counts are retained. Refinement assumes
a family satisfying the shrinking local budgets; no fixed-precision IEEE
scaling, rounded control construction or provider execution is proved.
The energy is mathematical E at the observed phase and the interpolation grid
is the ideal k*h; separately rounded energy reporting and machine-clock
accumulation remain outside the certificate.

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
and complete pointwise componentwise zero-floor limit. The subsequent
intrinsic-length extension identifies actual metric curve length, constructs
minimizing AC paths between arbitrary endpoints, and classifies every
nearest-Shore minimizer in strictly convex finite-dimensional normed spaces.
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
pairs. Arbitrary-endpoint minimizing curves are supplied by the separate
positive-floor construction below.

### #160: attained zero-floor paths and convergence of regularized minimizers

[ZeroFloorGeodesics.lean](FractalSync/ZeroFloorGeodesics.lean) and the
[zero-floor geodesic note](../docs/research/160-zero-floor-geodesics.md)
extend the fixed-pair distance limit to actual minimizing paths. The ambient
space is finite-dimensional and real normed; S is nonempty and closed;
x is off S and y lies in its complement component. The anisotropy parameter
may be any real number. No smooth Shore, strict convexity or unique minimizer
is assumed.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Uniform positive-floor length control gives actual zero-floor control | `infDist_lower_of_positive_floor_lengths`, `acGraphLength_zero_le_of_positive_floor_lengths`; the positive clearance is d(x,S) exp(-B), and the actual zero-floor integral is <=B. |
| Compactness of selected constant-speed regularized geodesics | `exists_uniform_limit_of_small_floor_geodesics`, `exists_uniform_limit_of_geodesics`; any positive sequence of floors tending to zero, without monotonicity, has an ambient-uniformly convergent subsequence. |
| Limit path is an actual zero-floor minimizer | `zeroFloor_minimizer_of_positive_floor_speed`, `zeroFloor_geodesic_of_positive_floor_speed`, `selected_geodesics_subsequence_zero_floor`; the limit is ordinary AC, remains in the initial component, has clearance d(x,S) exp(-D0), attains the actual integral, and satisfies D0(gamma(s),gamma(t))=D0(x,y) abs(s-t). |
| Unconditional zero-floor attainment on each component | `exists_zeroFloor_minimizer`, `zeroFloorDistance_isLeast`; the original off-Shore AC candidate class has a minimum, with no selected-geodesic premise. |
| Actual unregularized lengths of positive-floor minimizers converge | `selected_minimizer_lengths_tendsto_zero_floor`; every selected family of AC minimizers eventually avoids S and has L0(gamma_n) tending to D0. This statement needs no constant-speed parameterization. |

The last proof supplies the eventual squeeze
`D0 <= L0(gamma_n) <= D0/(1-epsilon_n²/m²)`, with
`m=d(x,S) exp(-D0)>0`. Uniform convergence of paths is only asserted after
subsequence extraction and requires the stated constant-speed selection.
Convergence of velocities, convergence of an arbitrary entire selected path
family, and uniform convergence over varying endpoint pairs are not asserted.
The proof uses the already checked metric-length bridge; it does not assume
lower semicontinuity of a derivative-dependent integral under uniform
convergence.

### #160: actual metric length, geodesics, and all nearest-Shore minimizers

These modules share `FractalSync.DistanceGraph`. The
[intrinsic-length note](../docs/research/160-intrinsic-length.md) develops
their proof dependencies and exact hypotheses.

| Module and checked result | Principal declarations and scope |
| --- | --- |
| [ACSubpaths](FractalSync/ACSubpaths.lean): actual affine subpaths | `ac_affineSubpath`, `deriv_affineSubpath`, `graphSpeed_affineSubpath`, `acGraphLength_affineSubpath`, `intrinsicDistance_le_subpath_integral`; includes zero-length parameter intervals. |
| [IntrinsicLength](FractalSync/IntrinsicLength.lean): cumulative actual graph integral | `acCumulativeLength`, endpoint identities, `continuousOn_acCumulativeLength`, `acCumulativeLength_sub`, `intrinsicDistance_le_acCumulativeLength_sub`, `intrinsicDistance_le_abs_acCumulativeLength_sub`. |
| [GraphChord](FractalSync/GraphChord.lean): lifted displacement retaining ruler change | `rulerGraphLift`, `ae_norm_deriv_rulerGraphLift`, `rulerGraphChord_le_length`, `rulerGraphChord_le_intrinsicDistance`, `norm_deriv_rulerGraphLift_le_of_control`, `ac_graphSpeed_le_control_deriv`. The chord inequality is norm(Phi(y)-Phi(x))<=rho(x)*exp(D(x,y))*D(x,y); no ambient distance-field derivative is assumed. |
| [IntrinsicVariation](FractalSync/IntrinsicVariation.lean): actual partition-supremum variation | `eVariationOn_intrinsic_le_acGraphLength`, `boundedVariationOn_intrinsic`, `intrinsicVariation_zero`, `intrinsicVariation_one`, `monotoneOn_intrinsicVariation`, `intrinsicDistance_le_intrinsicVariation_abs_sub`, `intrinsicVariation_abs_sub_le_acCumulativeLength_sub`, `absolutelyContinuousOnInterval_intrinsicVariation`; also AC and monotonicity of the cumulative integral. |
| [IntrinsicCurveLength](FractalSync/IntrinsicCurveLength.lean): full length identity and attainment | `absolutelyContinuousOnInterval_intrinsic`, `ac_graphSpeed_le_deriv_intrinsicVariation`, `acGraphLength_eq_intrinsicVariation`, `eVariationOn_intrinsic_eq_acGraphLength`, `exists_metric_length_lt_intrinsicDistance_add`, `acGraphLength_le_of_intrinsic_boundedSpeed`, `exists_acGraphLength_minimizer`, `intrinsicDistance_isLeast_pairwisePathLengths`. |
| [IntrinsicGeodesics](FractalSync/IntrinsicGeodesics.lean): constant-speed metric paths | `exists_intrinsic_boundedSpeedPath`, `exists_intrinsic_metricGeodesic`, `lipschitz_of_intrinsic_boundedSpeed`, `exists_intrinsic_ac_metricGeodesic`; properness yields compact bounded-speed families, and an explicit ordinary Lipschitz bound gives AC admissibility. |
| [ShoreMinimizers](FractalSync/ShoreMinimizers.lean): complete equality classification | `ae_speed_eq_radial_of_minimizing`, `dist_eq_distance_trace_sub`, `minimizing_shore_endpoint_is_nearest`, `minimizing_path_eq_lineMap`, `minimizing_shore_path_constant_of_zero`, `acGraphLength_eq_radial_of_monotone_nearest_segment`, `minimizing_shore_path_iff`, `minimizing_path_to_shore_classification`. |

Metric variation uses the explicit `intrinsicMetricSpace`, not the ordinary
norm metric. Finite-dimensional real normed E, positive epsilon and arbitrary
real a suffice for the length identity and arbitrary-endpoint minimizers;
S is arbitrary. The length-space property is an explicit witness theorem,
not an asserted Mathlib class instance. No smooth geodesic ODE or uniqueness
is concluded.

The Shore iff theorem assumes strict convexity only for its segment
classification. A path from Shore has radial minimum length exactly when it
is constant at zero endpoint distance, or traverses a nearest segment with
an AC nondecreasing parameter from zero to one. The converse allows pauses;
the reverse orientation has a nonincreasing parameter. No convexity of S or
unique nearest point is required. The existing nonempty-closed-set Shore
infimum theorem identifies this equality with the actual minimum.

### #160: flat-Shore scalar analysis and coordinate curvature

[FlatShore.lean](FractalSync/FlatShore.lean) uses `FractalSync.FlatShore` and
reuses the shared ruler. Its geometric interpretation is developed in
[the flat-Shore note](../docs/research/160-flat-shore.md). The scalar formulas
are checked. The following coordinate extension now identifies the planar
curvature formula through actual derivatives; the global cut locus remains
conventional.

| Checked statement | Principal identifiers |
| --- | --- |
| Actual derivatives of Q, 1/rho and the intrinsic warp slope | `hasDerivAt_normalNumerator`, `hasDerivAt_warp`, `warpSlope_eq`, `hasDerivAt_warpSlope`, `warp_curvature_identity` |
| Rational curvature expressions: Shore values, exact derivative, strict decrease, ranges and positive radial/tangential gap | `radialCurvature_zero`, `tangentialCurvature_zero`, `hasDerivAt_radialCurvature`, `radialCurvature_deriv_neg`, `radialCurvature_strictAntiOn`, `curvature_gap`, `tangentialCurvature_lower`, `tangentialCurvature_lt_radialCurvature`, `radialCurvature_le_one`, `radialCurvature_bounds`, `tangentialCurvature_nonpos` |
| Actual excursion integral: continuity, strict monotonicity, threshold, unbounded lower bound, unique positive turning radius for every L>pi epsilon and dilation | `continuous_excursionSpan`, `excursionSpan_strictMonoOn`, `excursionSpan_zero`, `excursionSpan_gt_threshold`, `excursionSpan_eq_threshold_iff`, `excursionSpan_linear_lower`, `excursionSpan_existsUnique_radius`, `excursionSpan_scale` |
| Specified quartic density integrates to pi(1+a²)/16>0 | `quarticDensity_eq`, `integral_quarticDensity`, `integral_quarticDensity_pos`. Identification as a Taylor coefficient remains conventional. |
| Unit speed plus specified tangential momentum imply the radial first-integral formula | `radial_first_integral`; actual momentum conservation is proved in the separate `FlatShoreGeodesics` extension below. |

Four helper lemmas establish denominator positivity and integrand continuity.
No Riemannian curvature tensor, conjugate-point or cut-locus theorem,
Banach-space implicit function theorem, or local minimizer theorem is
formalized in this module.

[FlatShoreGeometry.lean](FractalSync/FlatShoreGeometry.lean) uses namespace
`FractalSync.FlatShoreGeometry`. It defines the explicit diagonal plane
metric and the covariant derivative `fderiv Y x (X x) + Gamma(X,Y)`.
The derivative in the curvature definition is the actual nested covariant
derivative, including Mathlib's Lie-bracket term.

| Checked coordinate-geometric statement | Principal identifiers |
| --- | --- |
| Positive diagonal metric and actual coefficient derivatives | `radialMetric_pos`, `tangentialMetric_pos`, `metric_symm`, `metric_self_pos`, `hasDerivAt_radialMetric`, `hasDerivAt_tangentialMetric`, `hasDerivAt_gammaYY`, `hasDerivAt_gammaRY` |
| Connection laws, torsion and actual differentiated metric compatibility | `christoffel_symm`, `christoffel_metric_compatible`, `hasDerivAt_metric_pairing`, `covariantDerivative_torsion`, `covariantDerivative_first_linear`, `covariantDerivative_second_add`, `covariantDerivative_leibniz`, `covariantDerivative_metric_pairing` |
| Actual curvature commutator on coordinate fields | `curvature_normal_tangent_tangent`, `curvature_normal_tangent_tangent_eq`; both nonconstant intermediate fields are differentiated, and `lieBracket_normal_tangent` checks their coordinate bracket. |
| Metric Gram-determinant quotient equals the rational flat curvature and is +1 on Shore | `sectionalCurvature_eq_radialCurvature`, `sectionalCurvature_on_shore`; epsilon>0, every real a and signed normal coordinate. |

This module establishes global-coordinate connection and curvature. The
following extensions connect it to the actual Euclidean path metric and
prove smooth-solution and curved normal-chart results. The extensions below
now provide nonlinear covariance, a continuous trilinear coordinate tensor,
and an actual embedded chart under a normal-chord certificate. Manifold-level
bundle/atlas objects, a generic embedded-Shore reach theorem, higher-dimensional
curvature identification, smoothness of all minimizers, conjugacy/cut-locus
theory, and bifurcation remain separate.

### #160: Euclidean flat correspondence and smooth geodesic contracts

[FlatShoreIntrinsic.lean](FractalSync/FlatShoreIntrinsic.lean) uses the actual
Euclidean plane `WithLp 2 (ℝ × ℝ)`, not the coordinate product's default norm.
It checks actual `infDist` to `{r=0}` and the full AC path correspondence.
The [flat bridge note](../docs/research/160-flat-intrinsic.md) states the exact
integral and infimum definitions.

| Checked statement | Principal identifiers |
| --- | --- |
| Actual flat distance and cusp-safe derivative identity | `infDist_shore`, `hasDerivAt_normal`, `distance_weighted_deriv_eq`; differentiating d²=r² yields d*d'=r*r' without division by either trace. |
| Actual density, integrability and integral equality | `graphSpeed_eq_coordinateSpeed`, `ae_graphSpeed_eq_coordinateSpeed`, `coordinateSpeed_intervalIntegrable`, `coordinateLength_eq_acGraphLength`; full AC class, positive epsilon, arbitrary real a. |
| Candidate sets, infima and metric length | `coordinatePathLengths_eq`, `coordinateDistance_eq_intrinsicDistance`, `eVariationOn_eq_coordinateLength`, `exists_coordinateLength_minimizer`. |
| Actual coordinate chart and both AC directions | `absolutelyContinuous_chart_iff`, `coordinateLength_eq_chartLength`, `coordinatePathLengths_eq_chartPathLengths`, `intrinsicDistance_eq_coordinate_path_infimum`. |

[FlatShoreGeodesics.lean](FractalSync/FlatShoreGeodesics.lean) works with actual
`HasDerivAt` solution data for the coordinate geodesic ODE. It does not assume
that every minimizing AC path supplies such data.

| Checked statement | Principal identifiers |
| --- | --- |
| Actual energy and tangential momentum derivatives and interval conservation | `hasDerivAt_energy`, `hasDerivAt_momentum`, `energy_constant`, `momentum_constant`; conservation is derived from the ODE. |
| Radial first integrals | `tangent_velocity_eq_momentum`, `radial_first_integral`; existing metric and actual energy/momentum definitions. |
| Actual Shore geodesic, unit frame and parallel normal | `christoffel_on_shore`, `hasDerivAt_shoreCurve`, `geodesicAt_shoreCurve`, `shore_frame_orthonormal`, `alongDerivative_shore`, `shoreNormal_parallel`. |
| Actual curvature action and normal Jacobi residual | `fderiv_normalProfile`, `fderiv_tangentProfile`, `covariantDerivative_const`, `curvature_shore_normal_tangent`, `normal_profile_jacobiResidual`, `normal_profile_jacobiResidual_eq_zero_iff`; the commutator uses explicit constant coordinate extensions. |
| Explicit sine field, actual derivatives, initial data and first positive zero | `hasDerivAt_shoreJacobi`, `deriv_shoreJacobi`, `deriv_deriv_shoreJacobi`, `shoreJacobi_equation`, `shoreJacobi_initial_data`, `shoreJacobi_first_positive_zero`, `shoreCurve_pi_span`; epsilon>0 gives a unit normal and first positive zero pi. |

The [geodesic note](../docs/research/160-flat-geodesics.md) distinguishes the
actual Jacobi equation/field zero from an unproved bundled conjugate-point or
index-form/minimality theorem.

### #160: actual flat length second variation and local shortening

[FlatShoreSecondVariation.lean](FractalSync/FlatShoreSecondVariation.lean)
uses the existing flat metric at epsilon=1 and the actual sine curve
gamma_A(t)=(A*sin(t),(ell/pi)*t), t in [0,pi]. Its endpoints are (0,0) and
(0,ell); `absolutelyContinuousOnInterval_modeCurve` proves actual AC from a
global bound on its derivative. The
[second-variation note](../docs/research/160-flat-second-variation.md)
records the normalization and exact formal boundary.

`oneModeLength_eq_coordinateIntegral` identifies the defined amplitude
functional with the actual metric-speed interval integral, using
`hasDerivAt_modeCurve` and `metric_speed_modeCurve`. The radicand is positive
at every real amplitude. `Analysis/CompactParametricIntegral` derives local
domination from joint continuity and compactness, enabling actual parameter
differentiation of this integral.

`hasDerivAt_oneModeLength_zero` proves zero first variation, and
`hasDerivAt_deriv_oneModeLength_zero` proves the actual second derivative

\[
L_\ell''(0)=\frac\ell2\left[\left(\frac\pi\ell\right)^2-1\right].
\]

`deriv_deriv_oneModeLength_neg` gives negativity for ell>pi. The actual
derivative's right slope limit and the mean value theorem then give
`exists_shortening_amplitude_interval`: every sufficiently small positive
amplitude has length strictly less than ell. These are AC competitors with
fixed endpoints and an identified actual length.

[FlatShoreShortcuts.lean](FractalSync/FlatShoreShortcuts.lean) lifts the curve
to the Euclidean plane, reparameterizes it to [0,1], and proves actual AC,
exact endpoints and `acGraphLength_modePath`. The existing pairwise infimum
is bounded by this competitor. `metric_dilation`, `coordinateLength_dilation`
and `acGraphLength_dilation` check physical scaling. The resulting
`intrinsicDistance_lt_shore_length` proves, for epsilon>0 and L>pi*epsilon,
that actual D_epsilon,a((0,0),(0,L))<L/epsilon. This requires no attainment or
geodesic identification. The next extension supplies the global converse and
equality case. General second variation, all minimizers above the threshold,
and global cut-locus classification remain separate.

### #160: global flat minimum, exact threshold, and all short-segment equality cases

[FlatShoreCalibration.lean](FractalSync/FlatShoreCalibration.lean),
[FlatShoreMinimizers.lean](FractalSync/FlatShoreMinimizers.lean),
[FlatShoreIsometries.lean](FractalSync/FlatShoreIsometries.lean), and
[FlatShoreThreshold.lean](FractalSync/FlatShoreThreshold.lean) prove the global
AC result described in [the flat-minimum note](../docs/research/160-flat-global-minimum.md).
[ACScalarVariation.lean](FractalSync/ACScalarVariation.lean) supplies its generic
scalar equality criterion.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Actual regularized arccos calibration | `hasFDerivAt_calibration`, `calibrationDifferential_le_sphericalSpeed`, `calibration_lipschitz`; the argument lies in (-1,1) for 0<c<1 and a sum-of-squares certificate bounds its actual derivative. |
| Unrestricted AC global lower bound | `calibration_endpoint_bound`, `shore_span_le_sphericalLength`, `shore_span_le_acGraphLength`; actual a.e. chain rule, integration and the c→1 continuity limit for 0<=ell<=pi. |
| Actual normalized path minimum | `FlatShoreMinimizers.intrinsicDistance_eq_shore_span`; the lower bound passes through the full path infimum and the Shore segment attains it. |
| Equality rigidity and complete short-segment classification | `ae_normal_mul_velocity_eq_zero`, `minimizing_path_on_shore`, `acGraphLength_on_shore`, `minimizing_path_iff`; AC square-radius constancy forces the whole path onto Shore, and its tangent trace is monotone exactly in the equality case. |
| Generic scalar AC equality theorem | `ACScalarVariation.integral_abs_deriv_eq_sub_iff_monotoneOn`; actual total variation equals the signed endpoint increment exactly for monotone AC traces, including degenerate intervals. |
| Full path-infimum isometries | `pairwisePathLengths_dilation`, `intrinsicDistance_dilation`, `pairwisePathLengths_tangentialTranslate`, `intrinsicDistance_tangentialTranslate`; explicit inverse AC witnesses give equality of candidate-length sets. |
| Exact physical threshold and all short-segment minimizers | `FlatShoreThreshold.intrinsicDistance_eq_shore_length`, `minimizing_path_iff`, `intrinsicDistance_lt_shore_length_iff`, `exists_shorter_path_iff`; arbitrary tangential source, positive epsilon and all real a. |

The dimensional result is D_epsilon,a((0,0),(0,L))=L/epsilon for
0<=L<=pi*epsilon, with the same all-AC equality classification under positive
dilation. Combined with the previous strict competitor, a shorter path exists
exactly when L>pi*epsilon. All real a are allowed. At the threshold there is
one minimizing image, the Shore segment, with monotone AC parameterizations
and pauses allowed. The proof does not need a regularity theorem for arbitrary
AC minimizers or a general conjugate-point theorem.

### #160: actual affine coordinate covariance

[CoordinateCovariance.lean](FractalSync/CoordinateCovariance.lean) uses an
arbitrary continuous linear equivalence A and translation b. New coordinates
x represent the old point A*x+b; vector fields are transported by A inverse.
The [affine-covariance note](../docs/research/160-affine-covariance.md) records
the actual derivative hypotheses and links to the nonlinear extension below.

| Checked statement | Principal identifiers |
| --- | --- |
| Actual affine and transported-field derivatives | `hasFDerivAt_affineMap`, `hasFDerivAt_pullbackField`, `fderiv_pullbackField`; the derivative is A⁻¹*(dY)*A. |
| Actual connection and Lie-bracket covariance | `covariantDerivative_pullback`, `lieBracket_pullback`, `covariantDerivative_lieBracket_pullback`. |
| Full curvature commutator covariance | `curvature_pullback`; X and Y differentiable at the old point, Z differentiable everywhere, both intermediate covariant-derivative fields differentiable at that point. |
| Metric, Gram and ratio transport | `pullbackMetric_fields`, `pullbackMetric_pos`, `gram_pullback`, `gramRatio_pullback`; the general ratio is a totalized real expression, while its geometric interpretation requires nonzero Gram denominator. |
| Fully discharged flat specialization | `flat_covariantDerivative_eq`, `flat_curvature_eq`, `flat_gram_pullback_pos`, `flat_gramRatio_pullback`; positive epsilon, actual intermediate-field derivatives and strictly positive transformed Gram denominator. |

The general coefficient operations in this module are unbundled; geometric
connection and metric laws are supplied by the flat specialization. The next
modules prove nonlinear covariance and extension independence separately.

### #160: nonlinear covariance, actual curvature tensor, and arbitrary flat frames

[NonlinearCoordinates.lean](FractalSync/NonlinearCoordinates.lean),
[CurvatureTensoriality.lean](FractalSync/CurvatureTensoriality.lean), and
[FlatCurvatureTensor.lean](FractalSync/FlatCurvatureTensor.lean) use actual
Fréchet derivatives and Mathlib's inverse-derivative field pullback. The
[nonlinear geometry note](../docs/research/160-nonlinear-geometry.md)
explains the local regularity hypotheses.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Actual inverse-Jacobian derivative and nonlinear connection cancellation | `NonlinearCoordinates.fderiv_pullback_apply`, `covariantDerivative_pullback`; C² coordinate map, invertible actual derivative at the point, and actual field differentiation. |
| Actual bracket and full curvature covariance | `lieBracket_pullback`, `curvature_pullback`; neighborhood equality of inner covariant derivatives justifies differentiating their germs. Intermediate-field differentiability is explicit. |
| Metric and Gram-ratio transport | `pullbackMetric_fields`, `pullbackMetric_pos`, `gram_pullback`, `gramRatio_pullback`; actual field pullback and positive-definiteness transfer. |
| General actual commutator equals its coefficient expansion | `CurvatureTensoriality.curvature_eq_curvatureValue`; differentiable continuous-bilinear coefficients, differentiable X,Y, and C² Z. The first field derivatives and symmetric second derivative cancel. |
| Bundled continuous trilinear tensor and extension independence | `curvatureTensor_apply`, `curvature_eq_tensor`, `curvature_congr_values`, `curvatureTensor_swap`; the coordinate-space tensor is identified with the actual commutator. |
| Actual flat connection is smooth | `FlatCurvatureTensor.contDiff_connection`; positive epsilon, arbitrary real a, with all coefficient regularity discharged. |
| Full flat tensor and arbitrary regular-frame ratio | `tensor_eq_metric`, `curvature_eq_metric`, `gramRatio_eq_radialCurvature`; R(u,v)w=K*(g(v,w)*u-g(u,w)*v), positive Gram denominator for every nondegenerate frame. |
| Arbitrary-frame ratio after nonlinear pullback | `frame_gramRatio_pullback`, `frame_gram_pullback_pos`; actual local chart and field regularity discharged from the stated C²/differentiability hypotheses. |

The C² chart result identifies the actual commutator through equal germs. It
does not assert C¹ regularity of the transformed coefficient field, which
contains the chart Hessian and generally requires an additional derivative.
The next extension derives that coefficient regularity from a C³ chart and
proves actual moving-frame laws. Manifold connection bundles and atlas gluing
remain separate.

### #160: moving frames, actual coefficient regularity, and fiber curvature

[MovingFrames.lean](FractalSync/MovingFrames.lean),
[CoordinateConnectionRegularity.lean](FractalSync/CoordinateConnectionRegularity.lean),
and [ConnectionFormCurvature.lean](FractalSync/ConnectionFormCurvature.lean)
separate base directions from fiber coordinates. The
[moving-frame note](../docs/research/160-moving-frames.md) records the precise
regularity and completeness hypotheses.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Actual inverse-frame derivative and transported section derivative | `fderiv_inverseFrame`, `fderiv_inFrame`; inversion is differentiated at an invertible actual frame, with local invertibility derived. |
| Inhomogeneous gauge law and actual commutator covariance | `gauge_apply`, `covariantDerivative_gauge`, `curvature_gauge`; base direction fields and their bracket remain unchanged. The derivative argument uses equal germs. |
| Composition, identity and transformed regularity | `gauge_comp`, `gauge_id`, `contDiffAt_gauge`, `contDiffAt_inFrame`; C² frame plus C¹ coefficients gives C¹ gauge coefficients. |
| Actual nonlinear pullback regularity and invertible neighborhood | `pullback_apply`, `coefficients_pullback`, `contDiffAt_pullback`, `exists_ball_invertible_contDiffAt_pullback`; C³ chart, C¹ old coefficients and invertible Jacobian produce a positive ball of C¹ transformed coefficients. |
| Fiber-valued continuous trilinear tensor and gauge conjugacy | `ConnectionFormCurvature.curvature_eq_curvatureValue`, `curvature_eq_tensor`, `curvatureTensor_gauge`; section derivatives cancel from the actual commutator. |
| Compatibility with existing tangent-coordinate objects | `curvature_eq_coordinateCurvature`, `curvatureTensor_eq_coordinateTensor`; the specializations agree definitionally. |

These are coordinate and trivial-fiber objects. Gauge composition supplies a
local overlap law, without constructing an atlas-glued connection or curvature
bundle. The invertible-neighborhood radius is existential, not numerical.

### #160: curved normal-coordinate metric and curvature

[CurvedShoreGeometry.lean](FractalSync/CurvedShoreGeometry.lean) uses the supplied
metric A(r)dr²+B(r)(1-k(y)r)²dy². `metric_eq_flat_pullback`, `metric_self_pos`,
`metric_symm`, `metric_gram_determinant`, and `metric_zero_curvature` check its
pointwise metric/frame algebra. Positivity requires epsilon>0 and 1-k*r nonzero.

`christoffel_symm`, `christoffel_on_shore`, `christoffel_metric_compatible`,
`covariantDerivative_torsion`, `covariantDerivative_first_linear`,
`covariantDerivative_second_add`, and `covariantDerivative_leibniz` check its
connection laws. `hasDerivAt_metric_pairing` and
`covariantDerivative_metric_pairing` differentiate the actual varying pairing.

The two actual `fderiv` helpers, `covariantDerivative_const`,
`curvature_normal_tangent_tangent_on_shore`, and `sectionalCurvature_on_shore`
evaluate the commutator and Gram ratio at Shore. The result is +1 for every
curvature profile with the stated local derivatives of k and deriv k.

[CurvedShoreCurvature.lean](FractalSync/CurvedShoreCurvature.lean) checks the
two full intermediate field derivatives, `fderiv_gammaYY_pair_normal` and
`fderiv_gammaRY_tangent`, then `curvature_normal_tangent_tangent` evaluates
the actual off-Shore commutator. `sectionalCurvature_eq_flat_correction`
identifies its metric Gram ratio with

\[
K(r,y)=\frac{\epsilon^4-(1+a^2)r^4}
 {(\epsilon^2+(1+a^2)r^2)^2}
-\frac{k(y)(1+a^2)r(\epsilon^2+r^2)^2}
 {(1-k(y)r)(\epsilon^2+(1+a^2)r^2)^2}.
\]

The k' terms cancel. `hasDerivAt_sectionalCurvature_normal_on_shore`
proves the actual normal derivative of the commutator-defined curvature
is -(1+a²)*k(y) at r=0, using eventual equality on a nonfocal neighborhood.
The explicit sufficient hypotheses include epsilon>0, 1-k(y)*r nonzero
for the off-Shore formula, and actual derivatives of k and deriv k at y.
Constructing a normal chart from an arbitrary embedded Shore remains
separate. The [curved-coordinate note](../docs/research/160-curved-coordinate-geometry.md)
records that distinction and the local regularity assumptions. The following
module supplies the chart and metric under an explicit geometric certificate.

### #160: actual embedded distance, local normal chart, and metric pullback

[TubularNeighborhood.lean](FractalSync/TubularNeighborhood.lean) and
[EmbeddedShoreMetric.lean](FractalSync/EmbeddedShoreMetric.lean) use Mathlib's
actual distance-to-set function and inverse function theorem. The
[embedded normal-neighborhood note](../docs/research/160-embedded-normal-neighborhoods.md)
gives the geometric certificate and its limitations.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Normal-chord certificate gives a quantitative distance gap | `normalOffset_dist_sq_gap`; 2R*abs(inner n (q-p))<=norm(q-p)² over every q in S implies the gap factor 1-abs(r)/R. |
| Actual distance and unique nearest foot | `infDist_normalOffset`, `normalOffset_unique_nearest`; distance equality for abs(r)<=R, uniqueness for abs(r)<R, actual foot in S and unit normal. |
| Concrete sphere certificate | `sphere_normalChordBound`, `infDist_sphere_normalOffset`, `sphere_normalOffset_unique_nearest`; positive sphere radius and the actual inward radial normal. |
| Normal-strip injectivity | `normalMap_injOn_strip`; injective curve patch, unit normals, common chord radius over the full Shore set. |
| Actual normal-map differential and Euclidean metric factor | `hasFDerivAt_normalMap`, `inner_normalDifferential`; actual Frenet derivatives, with tangential factor 1-k*r. |
| Construction of an actual local chart and inverse derivative | `exists_normalChart`; local C¹ curve/normal, actual spanning planar linear equivalence, nonfocal 1-k*r. |
| Actual distance identity on an ambient neighborhood | `eventually_infDist_normalChart`; abs(first inverse-chart coordinate) equals actual infDist, under NormalTube and a strict radius bound. |
| Ambient regularized-distance calculus, including at Shore | `hasFDerivAt_ambientRuler`; the actual derivative follows from local equality with sqrt(epsilon²+r²). |
| Actual ambient graph metric equals the curved coordinate metric | `ambientGraphMetric_pullback`, `ambientGraphMetric_eq_scaleMetric`, `exists_normalChart_with_metric`; actual chart, ambient derivatives, and the existing rank-one metric. |

`NormalTube` supplies its chord certificate at every parameter and against the
entire Shore set. The next extension derives a radius under quantitative jet
and separation hypotheses and replaces this global-parameter requirement with
a local patch. The subsequent ambient-connection extension identifies the
connection and transfers curved-model curvature under higher local regularity.

### #160: derived normal radii and a local patch interface

[NormalRadius.lean](FractalSync/NormalRadius.lean) and
[EmbeddedNormalPatch.lean](FractalSync/EmbeddedNormalPatch.lean) derive
R=min(m²/L,d/2), with m=v-L*delta/2>0, from actual curve derivatives and remote
Shore separation. The [derived-radius note](../docs/research/160-derived-normal-radii.md)
gives the full local/remote argument and its scope.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Actual Taylor normal projection and chord lower bound | `normal_chord_taylor`, `chord_lower_bound_of_derivative`; derivative variation L, tangent speed at least v, and parameter width delta. |
| Full-Shore normal radius and uniform base patch | `normalChordBound_of_separation`, `certifiedRadius_normalChordBound`, `normal_radius_of_curve_jet`, `uniform_normal_radius`; remote distance at least d, with L,d positive. |
| Sources of derivative and remote bounds | `derivative_variation_of_second_derivative`, `exists_derivative_variation_bound`, `exists_remote_separation`, `exists_remote_separation_of_compact_injective`; compactness produces finite positive constants under the stated regularity and disjointness. |
| Derived patch and injective normal strip | `injOn_of_curve_jet`, `normalPatch_of_curve_jet`, `exists_normalPatch_of_compact_injective`; local injectivity is derived and compact embedded arcs supply remote separation. |
| Actual local distance and unique foot | `infDist_normalMap`, `unique_nearest`; distance for abs(r)<=R, uniqueness for abs(r)<R. |
| Local chart/ruler/metric bridge | `eventually_infDist_normalChart`, `hasFDerivAt_ambientRuler`, `ambientGraphMetric_pullback`, `exists_normalChart_with_metric`; the base patch need only be a neighborhood of the chosen parameter. |

The compact C² theorem uses actual `ContDiffAt` at every patch point; it is
not merely a within-set derivative assertion at endpoints. A planar spanning
frame, actual Frenet derivatives, C¹ curve/normal regularity and the nonfocal
factor remain explicit in the local inverse chart. No general reach or ambient
connection/curvature theorem is inferred.

### #160: actual ambient connection and curved-Shore curvature transfer

[MetricConnectionUniqueness.lean](FractalSync/MetricConnectionUniqueness.lean),
[EmbeddedShoreConnection.lean](FractalSync/EmbeddedShoreConnection.lean), and
[EmbeddedShoreCurvature.lean](FractalSync/EmbeddedShoreCurvature.lean) identify
the actual ambient regularized-distance connection. The
[ambient-connection note](../docs/research/160-ambient-shore-connection.md)
states the local normal-patch, Frenet and smoothness hypotheses.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Positive metric determines a compatible torsion-free connection | `MetricConnectionUniqueness.koszul`, `eq_of_compatible`, `CurvedShoreGeometry.christoffel_eq_of_compatible`. |
| Actual ambient ruler/gradient/Hessian regularity | `contDiffAt_ambientRuler_of_normalMap`, `contDiffAt_gradient`, `hasFDerivAt_inner_gradient`, `gradient_derivative_symm`; IFT and local distance identity give actual regularity through Shore. |
| Derived nonlinear connection identification | `hasDerivAt_pullback_pairing`, `pullbackConnection_eq_curved_of_metric_germ`, `ambientConnection_normalMap`, `ambientConnection_normalFrame`; both coordinate-Hessian terms are retained and the final theorem assumes no ambient jet or connection identity. |
| Actual curvature germ and ambient covariance | `curvature_congr_connection`, `eventually_ambientConnection_normalMap`, `curvature_normalMap`, `ambient_curvature_pullback`, `ambient_gramRatio_pullback`; actual inverse-Jacobian field pullback and intermediate regularity derived from the ruler. |
| Nondegenerate normal-frame curvature and explicit formula | `normalGram_pos`, `ambientNormalCurvature_eq_flat_correction`, `ambientNormalCurvature_on_shore`; the full curved correction and Shore value +1 follow for the actual ambient connection. |

Connection identification uses a C2 normal map and differentiable curvature
profile; the full curvature specialization uses a C3 normal map and locally
C2 curvature profile. The final objects use Mathlib's actual Riesz gradient
of sqrt(epsilon²+infDist²) and its actual derivative. A local `NormalPatch`,
Frenet identities, parameter neighborhood and nonfocal invertible differential
remain geometric inputs. This does not assert smoothness at fractal or
distance-cut-locus points. Atlas/bundle gluing, concrete numerical normal
radii, and arbitrary AC minimizer regularity remain separate.

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

**Not formalized by this differential module:** the smooth cross-scale comparison
assembled from these local residuals,
optimal finite-sample envelopes, topology counterexamples, or blow-up limits.
Actual local differentiability hypotheses do not certify numerical provider
derivatives or imply distance-field smoothness at a cut locus. No ideal
Lipschitz hypothesis is asserted for an arbitrary numerical provider.

### #162: combined provider and recomputed-ruler errors

[NormalizedProviderError.lean](FractalSync/NormalizedProviderError.lean)
checks the actual error decomposition at the computed query location.
`normalizedMap_scale_change_bound` gives the symmetric signed-scale bound
with denominator max(abs(h),abs(k)); `normalizedMap_center_change_bound`
gives center stability. `normalizedMap_provider_scale_error_bound` accepts
independent nonzero nominal and computed scales.

`normalizedMap_recomputed_ruler_error_bound` specializes to h=q*rho and
k=q*rhoHat, for q>0 and epsilon>0. Its correlated-error certificate is
evaluated at x+k*u, where the provider actually queries. The companion
endpoint and uniform-budget versions, exact rationalized ruler perturbation,
and `normalizedMap_crest_bias_bound` are also checked. At an ideal crest,
constant bias b gives the bound (b/epsilon)^2*norm(u).

Only the ideal field is assumed one-Lipschitz. No Lipschitz assumption is
imposed on the provider D+e, and no value/derivative error certificate is
inferred from a stencil's self-consistency checks. The runtime clamps its
stencil scale; it is not always q*rhoHat. The general h/k theorem can
accommodate an actual clamped scale with the corresponding actual-query
certificates. Coordinate/ruler/division floating errors remain separate.
See [the full error note](../docs/research/162-normalized-distance-fields.md).

### #159/#162: actual numerical stencil error contracts

[ProviderCertificates.lean](FractalSync/ProviderCertificates.lean) uses
`FractalSync.ProviderCertificates`. `first_order_remainder_bound` and
`second_order_remainder_bound` derive vector first-order and scalar cubic
Taylor remainders from actual differential fields. `LocalJetControl.remainder`
packages convex query-region hypotheses; `LocalJetControl.hessian_symm`
derives actual Hessian symmetry on a neighborhood.

`centralFirst_remainder_identity`, `centralSecond_remainder_identity`,
`centralMixed_remainder_identity`, and the corresponding
`LocalJetControl.*_error` theorems give quadratic gradient and linear Hessian
truncation bounds under anchored Lipschitz Hessian control. Each stencil also
has checked `*_add`, `*_provider_error`, `*_provider_endpoint_error`, and
`LocalJetControl.*_total_error` declarations. The total budgets retain a
separately supplied arithmetic error.

`sampled_value_error` accounts for query displacement under an actual field
Lipschitz bound; `euclidean_error_bound` aggregates component budgets in the
Euclidean norm (and Frobenius norm using a product index);
`division_perturbation_bound` controls positive-denominator uncertainty.
See [the stencil note](../docs/research/162-provider-stencil-certificates.md)
for exact constants and the runtime's clamped nine-point stencil. These
contracts do not certify actual raster values, `Regular` flags or IEEE
operations without additional evidence.

### #162: actual operational quintic polynomial and coefficient-to-jet bounds

[OperationalField.lean](FractalSync/OperationalField.lean),
[OperationalFieldBounds.lean](FractalSync/OperationalFieldBounds.lean), and
[OperationalFieldJetError.lean](FractalSync/OperationalFieldJetError.lean), with
[OperationalFieldQueryError.lean](FractalSync/OperationalFieldQueryError.lean),
use the six coefficient lists and simultaneous Horner recurrence from
`runtime-core/src/operational_field.rs`. Their domain is the exact-real
polynomial associated with a fixed cell and its 36 stored coefficients.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Returned Horner jets are actual derivatives | `hasDerivAt_hornerJet`, `hasDerivAt_basis`; actual simultaneous value/first/second recurrence. |
| Actual physical-coordinate Fréchet differential and Hessian | `hasFDerivAt_field`, `hasFDerivAt_fieldDifferential`, `fderiv_field`, `fderiv_fderiv_field`, `fieldDifferential_eq_query`, `fieldHessian_eq_query`; actual spacing factors and mixed component. |
| Concrete weight moments and nonnegativity | `basis_partition`, `basis_first_moment`, `basis_second_moment`, `basis_nonneg`; Bernstein coefficients certify positivity on the cell. |
| Raw quadratic sampling bias with exact jets | `field_quadratic`, `fieldDifferential_quadratic`, `fieldHessian_quadratic`; Q=ax²+bxy+cy²+dx+ey+f gives Q+h²*(a+c)/2 and the actual differential/Hessian of Q. |
| Coefficient error is not amplified in value | `field_coefficient_error_bound`; uniform coefficient perturbation eta gives absolute value error at most eta on the cell. |
| Concrete spacing-scaled derivative sensitivity | `fieldDifferential_coefficient_error_bound`, `fieldHessian_coefficient_error_bound`; each gradient component <=4eta/(3h), pure Hessian <=2eta/h², mixed <=16eta/(9h²), for positive h. |

The query module derives operator bounds from an actual coefficient magnitude
bound B: norm(DF)<=8B/(3h) and norm(D²F)<=68B/(9h²), in the coordinate product's
maximum norm. Mean-value estimates on the convex physical cell give actual
value and differential variation, plus the combined value budget
eta+8B/(3h)*norm(deltaQuery). Both queries must lie in the same fixed cell.
The real displacement bound does not certify the floating-point query error
or select its correct cell. The following global extension removes the same-cell
restriction under a uniform coherent-lattice bound.

The [concrete provider note](../docs/research/162-concrete-provider-certificates.md)
distinguishes the stored-coefficient model, conditional raw-sample reproduction,
and remaining Rust/query/floating-point correspondence. These are actual
polynomial-jet certificates, not a proof that the operational field equals
Shore signed distance or that the floating-point evaluator is exact.

### #162: actual global operational field and cross-cell query error

The six new modules `OperationalFieldGluing`, `OperationalFieldJoins`,
`OperationalFieldLattice`, `GlobalOperationalField`,
`OperationalFieldQueryCorrespondence`, and `OperationalFieldGlobalError`
connect the fixed-cell polynomials to the literal mathematical floor-selected
evaluator. The [global provider note](../docs/research/162-global-operational-field.md)
records the definitions, index convention and exact implementation boundary.

| Checked statement | Principal identifiers and scope |
| --- | --- |
| Actual floor-selection gluing | `hasFDerivAt_floorSelect`; derivatives on both neighboring closed cells are glued at integer boundaries using actual matching jets. |
| Concrete endpoint matching through order four | `hasDerivAt_hornerDerivative`, `hornerDerivative_low`, `latticeLineJet_join`, `latticePatchJet_join_x`, `latticePatchJet_join_y`; the coherent coefficient-index shift is essential. |
| Global actual derivatives and C4 regularity | `hasFDerivAt_globalJet`, `contDiff_globalJet`, `contDiff_globalField`; arbitrary real integer lattice, including grid edges and vertices. |
| Physical first and iterated derivatives and selected-query correspondence | `fderiv_physicalField`, `fderiv_fderiv_physicalField`, `contDiff_physicalField`, `physicalField_eq_query_patch`, `physicalDifferential_eq_query_patch`, `physicalHessian_eq_query_patch`. |
| Ideal finite-asset indices and extension independence | `floor_fraction_mem`, `halo_index_bounds`, `halo_flattened_index_bounds`, `latticeCoefficients_eq_of_halo`, `globalJet_eq_of_halo`; exact-real full-halo inequalities and mathematical integer floor. |
| Global operator and query bounds | `norm_physicalDifferential_le`, `norm_physicalHessian_le`, `physicalField_query_error_bound`, `physicalDifferential_query_error_bound`, `physicalField_combined_error_bound`; arbitrary cell crossings under a uniform lattice bound. |

For positive spacing h and coefficient magnitudes <=B, the actual global
operator bounds remain 8B/(3h) and 68B/(9h²) in the product maximum norm. The
combined value budget is eta+8B/(3h)*norm(deltaQuery). The following extension
supplies conditional reference-jet fidelity and the Hessian query bound.
The concrete quadratic fixture above now supplies its sample and actual
reference-jet accuracy. Evidence for production asset accuracy, machine
floor/casts, validity masks, storage decoding, query-rounding bounds and IEEE
evaluation remains separate.

### #162: concrete reference fidelity and combined actual-query jet budgets

[OperationalFieldFidelity.lean](FractalSync/OperationalFieldFidelity.lean),
[OperationalFieldReference.lean](FractalSync/OperationalFieldReference.lean),
[OperationalFieldHessianError.lean](FractalSync/OperationalFieldHessianError.lean),
and [OperationalFieldComputedError.lean](FractalSync/OperationalFieldComputedError.lean)
derive concrete estimates for the literal global physical field. The
[reference-fidelity note](../docs/research/162-operational-reference-fidelity.md)
states the local sample, actual reference-jet, and induced product-norm contracts.

For spacing h>0, local sample error eta, and actual reference Hessian variation
L on a convex region containing the query and its 36 selected nodes, the
value error after subtracting h²*trace(Hf)/4 is <=eta+9Lh³/2. If the region
is also a neighborhood of the query, differential error is
<=8eta/(3h)+12Lh² and Hessian error is <=68eta/(9h²)+34Lh.
The uncorrected value error retains the quadratic bias. Hessian symmetry and
literal `fderiv` identities are derived from the actual reference derivatives.

`SampleAccuracyAt` requires only the 36 selected samples. The literal-global
result `physicalField_reference_actual_jet_error` compares actual first and
iterated derivatives. The concrete third-order basis certificate yields
global Hessian query gain 32B/h³ under a uniform lattice magnitude bound B.
`reported_reference_jet_error` combines each reference error with the actual
query displacement times its gain and the separately supplied reported-result
evaluation error. These are conditional error theorems; they do not certify
an asset's true-distance samples, machine floor/casts/masks, query rounding,
storage decoding or IEEE evaluation.

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

The current audit covers 4,013 declarations, including the 1,825 stated theorem/lemma declarations,
definitions, and compiler-generated support declarations. No `sorry`, `admit`,
new axiom, or native-evaluation axiom is accepted. Exact real arithmetic is
distinct from IEEE 754 computation, and the formal statements do not certify
an implementation's correspondence to their definitions.

## Independent numerical checks

The scripts under `checks/` are standalone mathematical experiments.
Three use the Python standard library. The sensitivity and exterior-jet
checks use the optional libraries pinned in `checks/requirements.txt`.
The steering comparison calls the project's compiled Rust bindings.

Use the project Python environment from the repository root:

```sh
python -m pip install -r formal/checks/requirements.txt
python formal/checks/check_distance_geometry.py
python formal/checks/check_sensitivity_models.py
python formal/checks/check_normalized_distance_fields.py
python formal/checks/check_covariant_statistics.py
python formal/checks/check_exterior_ruler_jets.py
python formal/checks/check_trap_steering_budget.py
python formal/checks/check_navigation_fields.py --images tmp/navigation-185
python formal/checks/check_exterior_navigation.py
```

Activate the repository's environment or replace `python` with its explicit
interpreter path. See the [verification record](../docs/research/verification.md),
the [exterior jet intervals](../docs/research/184-exterior-ruler-jets.md#interval-experiment),
and the [steering comparison](../docs/research/185-matched-metric-steering.md).
