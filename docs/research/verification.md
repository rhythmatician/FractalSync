# Verification record: Lean geometry and mathematical research

Verified 2026-11-25 (America/Chicago) in the supplied Linux environment. This
continuation starts from draft PR #164 at
`57a1650fe5a2d8ce6254434252ecc0ce2e24cdd2`, whose reported main-branch base is
`55df760ff706210dcce3c88e79e476a45c5331f0`.
The preceding 1,256-declaration package and all earlier increments are retained.
This record separates the new kernel-checked results, existing analysis support,
conventional geometry, and earlier optional-tool/numerical observations.

## Environment and full Lean build

| Component | Observed version or pin |
| --- | --- |
| Elan | 4.4.0 |
| Lean | 4.24.0, commit `797c613eb9b6d4ec95db23e3e00af9ac6657f24b` |
| Lake | 5.0.0-src+797c613 |
| Mathlib dependency | `f897ebcf72cd16f89ab4577d0c826cd14afaafc7` |
| Transitive dependencies | Committed `formal/lake-manifest.json`, unchanged |
| AC backport source snapshot | Mathlib `aad247f928b1eb7c58478742119fdd15a37f0ffd` |
| Official proof guidance | `leanprover/skills` at `7d3da0282e7b724b07620e45cf212f2e05e19334` |
| Previously installed and exercised optional Lean LSP/MCP | 0.33.0, commit `7932203d79db213852db215835bd79c943bb99b0` |

The earlier increments installed Elan, Lean and the pinned Mathlib cache.
This continuation reused that installation and working cache. The new
trigonometric and exponential bound imports were built under the same pins.
No Mathlib source was changed. The toolchain, dependency sources, manifest, Lake
configuration, audit implementation/allowlist and workflow were unchanged.

From `formal/`, the complete command

```text
lake build --wfail
```

completed successfully with **2,706 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing ninety-eight result modules | 1,201 |
| `PotentialForceBounds` | 16 |
| `ControlForceBounds` | 10 |
| `OperationalCatalogBounds` | 4 |
| `ConcreteOperationalAsset` | 23 |
| `ConcreteAssetFidelity` | 11 |
| `RationalTrajectoryMesh` | 8 |
| `ConcreteTrajectoryCertificate` | 17 |
| `PhaseEnergyBounds` | 7 |
| `ConcreteEnergyCertificate` | 12 |
| Existing fourteen analysis-support modules | 55 |
| **Total across 121 source modules** | **1,364** |

This adds **108 explicit declarations in nine result modules**. The result
subtotal is 1,309 across 107 modules; support remains 55 across fourteen
modules. The counts include supporting lemmas and are not counts of independent
research results. Two additional proved energy-certificate constructors are
`def` declarations, so they enter the audit but not the theorem/lemma count.
A comment/string-stripped source inventory reproduced the preceding count
of 1,256 before counting this increment. Every result-module row in the
[formal inventory](../../formal/README.md) is reconciled against its source
count and umbrella import. No existing Lean theorem declaration changed.

## New checked scope

### Derived actual potential and control bounds

The [potential-force extension](159-potential-force-bounds.md) computes actual
Shore, secant-wall and total conservative norm/Lipschitz pairs. The position
radius r<2 supplies a positive cosine denominator floor. On the unit position
ball, the actual default wall parameter mu=1/pi gives norm bound 1/16 and
Lipschitz constant Lx/2. The Shore specialization kappa=log2 cancels the
logarithmic constant without removing the nonzero conservative term.

The [control extension](159-control-force-bounds.md) bounds the actual metric
lowering, drive and viscous covectors. It proves regularity of the actual
metric normalization of every fixed nonzero command, including the unit
formula and positive-scaling invariance. Heading regularity is derived from
the ruler and its gradient, not supplied as a new premise.

The combined [catalog certificate](159-concrete-trajectory-certificate.md)
uses epsilon=a=F=beta=1, kappa=log2, mu=1/pi and fixed positive-first-axis
command. The actual total force includes both conservative terms, normalized
drive and viscous drag. Its concrete pair is (400000,4e12). The actual
acceleration, including the changing-metric connection, has pair (1.1e12,4e19).
The parameters define a research instance and do not change canonical values.

### Literal finite asset and actual reference fidelity

The [concrete asset](162-concrete-asset-instance.md) materializes the existing
16-by-16 quadratic test fixture in the runtime source at the recorded baseline.
All 256 coefficients are literal dyadic rationals with common denominator 2^57.
Ordinary kernel-reduced finite decisions prove the coefficient magnitude
bound 7 and sample error at most 2^-48 against P(x,y)=x²+2xy-3y² at exact physical
nodes. The spacing is the exact dyadic value of binary64 0.1. Explicit unequal
entries certify that the fixture is neither constant nor zero.

The complete half-open physical halo has checked actual field fidelity:

```text
|F(q) - (P(q)-h²)| <= 2^-48
norm(DF(q) - D(P-h²)(q)) <= 2^-43
norm(D²F(q) - D²(P-h²)(q)) <= 2^-38.
```

The statements use actual first and iterated Frechet derivatives, including
at grid lines and vertices. The raw-sample quintic bias -h² is retained.
Differential/Hessian norms here use the coordinate maximum norm. A separate
contractive-chart/Riesz theorem proves Euclidean gradient error<=2^-43.
There is no remaining finite sample-fidelity or coefficient premise for this
literal fixture. Its reference is a quadratic polynomial, not Mandelbrot
distance or a production asset.

The center x0=(1/2,1/2), initial velocity (1/4,0), phase radius 1/16, reported-query
displacement 1/64, exact halo margin 1/2-2h>1/4, and all 36 mathematical stencil
addresses are also checked. Euclidean position/velocity norms and the phase
maximum norm are used by the dynamics; reported coordinate error uses its
coordinate maximum norm.

### A constructed actual reference and fully instantiated finite run

The [concrete trajectory theorem](159-concrete-trajectory-certificate.md)
constructs an actual reference from the literal field and catalog, with
regional uniqueness, endpoint derivatives, a positive reference tube, and
exact horizon

```text
T = 1 / 2560000000000000000064.
```

The shared speed/Lipschitz bound B=4e19 gives this rational horizon through
R/[4(B+1)] at R=1/16. It is approximately 3.90625e-22 in model time. This is a
very conservative consequence of coefficient magnitude propagation, not
measured stiffness or a practical runtime timestep recommendation.

For m=N+1 steps, tau=T/m, the perturbed-history theorem admits initial error
<=tau, acceleration residual<=tau and stored-step error<=tau². Its continuous
interpolant has phase error<=1/(64m) at every time, remains in the open phase
ball, and supports every reported query within 1/64 with all 36 valid
mathematical addresses. The numerical residual contracts are conditional on
state membership; the theorem proves that membership by the checked bootstrap.
No fixed-precision implementation is asserted to meet those scaled budgets.

`exactTrajectory` recursively uses the actual acceleration and velocity-first
position update over exact reals. Its residual hypotheses vanish by definition.
`exists_exact_scheme_certificate` therefore has no supplied numerical-error,
asset, force or solution premise. `exists_1024_step_certificate` instantiates
the uniform phase-error bound 2^-16 for the actual 1024-step recurrence.
Interpolated velocity is not asserted to equal the derivative of interpolated
position.

### Explicit energy regularity and unconditional finite-run remainders

The [energy extension](159-phase-energy-bounds.md) identifies the actual
phase-energy Frechet derivative with a bundled continuous linear covector and
computes its norm/Lipschitz pair from actual ruler, gradient, Hessian, velocity
and conservative potential-gradient certificates. It requires no unspecified
compactness constant or assumed phase-energy Hessian bound.

On the concrete phase region, norm(DE)<=2e7 and Lip(DE)<=2.4e10. The actual
potential-gradient identity and all regional field/force evidence are proved
for the literal fixture. Convexity supplies the full segment for every update
whose old/proposed states are in the region, giving the explicit Taylor
coefficient 1.2e10 times the squared phase-increment speed.

The finite-run theorem then derives every stored state's open-ball membership
from the actual interpolant at grid times. It discharges both endpoint
conditions at every recurrence step and proves

```text
|E(y[i+1])-E(y[i])-DE(y[i])[y[i+1]-y[i]]| <= 1/(400000000*m²),  i<m.
```

This is a per-step Taylor remainder relative to the retained differential
term. It does not assert conservation or bound total driven/damped energy
drift by the displayed quantity.

### Retained results and precise implementation boundary

All preceding #160 results, including zero-floor attainment, subsequential
constant-speed selected-path convergence, full-sequence selected-length
convergence, flat threshold/classification at and below threshold, and actual
ambient curved-Shore connection/curvature transfer are unchanged. The earlier
#161--#163 results are retained. This increment does not establish new smooth
minimizer regularity, global cut locus, bifurcation, or stronger singular limits.

The extractor models separately rounded, non-fused binary64 arithmetic and
cross-checks every operation with exact rational round-to-nearest, ties-to-even.
It is external reproducibility evidence, not Rust execution or a proof of
compiler-contraction/FMA equivalence. The kernel certifies the literal table
and its analytic properties independently of that extraction model.

Production distance assets, actual query/IEEE error budgets, machine index and
storage/mask correspondence remain open. The formal fixed-command drive is
reevaluated at every phase point. The runtime operational engine accepts a
constructed control covector and holds it through its Runge-Kutta stages;
that policy and integrator correspondence remain separate residual obligations.
No continuation beyond the local interval, impulse/time-dependent-control
convergence or combined steering result is inferred.

## Attributed analysis backport and project support

The pinned Mathlib has the AC definition but lacks its fundamental theorem.
Selected upstream proofs were adapted without replacing the definition or
adding analytic hypotheses to the target path theorem. The source mapping,
original copyright and Apache 2.0 license are recorded in
[the support README](../../formal/FractalSync/Analysis/README.md).

All six retrieved upstream files were checked against their exact blobs at
`aad247f928b1eb7c58478742119fdd15a37f0ffd`. The compatibility work uses the
pinned Fatou theorem, the pinned Vitali covering theorem, finite interval gaps,
and the original zero-derivative constancy proof. Finite-dimensional derivative
integrability is obtained by continuous linear coordinates. No `.lake/packages`
source was edited and no dependency pin was upgraded.

The earlier `ACComposition` module is separate project work. It checks the
disjoint-interval condition under monotone/antitone maps, preserves AC under
Lipschitz precomposition with the stated interval-image premise, and glues
matching AC pieces by constant extension. These four declarations use the
actual pinned Mathlib AC definition and enter the same transitive audit.

The earlier 596-theorem increment added the vector-valued AC FTC by dual separation, a
derivative estimate from actual punctured slope limits, and a proper-metric
geodesic construction using compact endpoint-path families. The latter
proves its compactness and reparameterization arguments directly; it does
not assume a minimizing curve or import an extra geometric axiom.

The preceding 906-theorem increment added one project-authored compact parametric-integral
helper. Joint continuity and actual parameter derivatives yield uniform
local domination on a compact rectangle, then the pinned differentiation
theorem applies to the actual interval integral. It supplies no new analytic
axiom and is separate from the attributed upstream backport.

The preceding increment's `BoundedSpeedCompactness` module added one project theorem using
the pinned bounded-continuous-function Arzelà–Ascoli and compact sequential
extraction APIs. It preserves the original endpoint and speed bounds while
producing actual uniform subsequence convergence. The package now has fourteen
analysis-support modules and 55 explicit support theorem/lemma declarations.

## Transitive axiom audit and negative controls

```text
lake env lean Audit.lean
```

passed for **2,961 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 2961 project declarations
```

The audit chooses roots by defining-module provenance as well as the
`FractalSync` namespace. Thus backported declarations in namespaces such as
`Finset` and `AbsolutelyContinuousOnInterval`, and private/generated helpers,
are included even when no named project theorem directly uses them. The
three-axiom allowlist and Lean checked-environment transitive traversal are
unchanged. An empty project root set is rejected.

The preceding 291-theorem increment ran a negative control that temporarily
imported `FractalSync.AuditScopeProbe`,
whose axiom and dependent theorem were both in the unrelated namespace
`ExternalAuditScope`. No `FractalSync`-named declaration depended on them.
The provenance-aware audit rejected the imported axiom with exit code 1:

```text
Unapproved axiom ExternalAuditScope.unapproved reachable from FractalSync declarations
```

The probe source, compiled module and temporary audit file were removed.
The earlier 172-theorem increment also tested an unapproved outside-namespace axiom
used indirectly by a project theorem; that transitive-dependency control was
rejected. The audit implementation and allowlist were unchanged in this
continuation, so those negative controls were not rerun. The accepted package
contains no `sorry`, `admit`, new axiom or
native-evaluation axiom. These are checks of the kernel environment rather
than only source-text scans.

## Optional MCP and official skill installation

The preceding 596-theorem increment installed the pinned 0.33.0 server in a separate Python
3.12 environment and ran `formal/tools/check_mcp.py` in a real MCP stdio
session. It discovered **22 tools**, obtained clean diagnostics
for `DistanceToSet.lean`, and used `lean_verify` to inspect
`FractalSync.DistanceGraph.exists_nearest_segment`. The result reported only
standard axioms. The check exited successfully.

The server emitted an ignored Python asyncio subprocess-finalizer warning,
`RuntimeError: Event loop is closed`, during shutdown. Diagnostics and theorem
verification had succeeded; clean server process teardown is a remaining
upstream integration issue, not a Lean proof failure. The optional server is
not a proof-CI dependency or a trusted axiom source. The checked-in VS Code
configuration is a template, not a claim that every agent client has registered
this MCP server. The checked-in Codex TOML template parsed successfully. An actual
`codex mcp add` attempt failed because the active client configuration is on
a read-only filesystem; no server entry was registered. This is the concrete
remaining client-registration blocker, distinct from the successful server
installation and stdio test. The 726-theorem increment confirmed that the
configuration mount was still read-only. This continuation did not repeat the
server smoke test or attempt to bypass that mount.

The official `lean-proof` and `mathlib-build` skills were installed from the
documented source pin and their presence in the personal catalog was verified.
Their instruction text matches the pinned upstream guidance, with the
source license retained. Neither skill is a runtime or
proof-CI dependency.

## Independent numerical checks

The new concrete-fixture extraction check was rerun in this continuation:

```text
python formal/fixtures/check_operational_quadratic.py
```

It passed for 256 coefficients, denominator 144115188075855872, magnitude bound 7,
sample-error bound 2^-48 and the exact-rational rounding cross-check. The command
also checks current source-function provenance, all recorded bit patterns,
and correspondence with the literal Lean table. Its output explicitly says
`rust_execution_claimed: false`. Rust/Cargo were unavailable; no executable
Rust comparison is claimed. The literal/sample inequalities are also separate
kernel proofs covered by the transitive audit.

The initial 28-theorem increment ran each script using its repository Python
environment, Python 3.12.14. These unchanged numerical scripts were not rerun
for the subsequent analytic extensions.
The sensitivity experiment used NumPy 2.3.5 and SciPy 1.17.0; the other three
scripts use only the standard library. All random trials have fixed seeds.

| Experiment | Observed result |
| --- | --- |
| Distance primitive against independent log-coordinate adaptive quadrature | 49 cases passed; maximum absolute discrepancy `1.862e-12`. |
| Squared radial-density remainder | 2,000 cases passed; maximum scaled discrepancy `5.023e-16`. |
| Quantitative zero-floor bound in the explicit one-dimensional puncture model | 21 cases passed. |
| Asymmetric complex-template L2 metric against finite differences of the actual analytic template | Three tested scales; maximum relative error approximately `1.287e-10`. The nonzero mixed Gram coefficient was `0.5190493241172275`. |
| Scalar exponent obstruction in the two-dimensional fold model | Condition numbers `12.5`, `1250`, `125000` at distances `1e-2`, `1e-4`, `1e-6`, for every tested exponent `-0.5, 0, 0.5, 1, 2`. |
| Normalized chart transition and combined provider/ruler error budget | 2,000 trials passed; maximum chart discrepancy `2.243e-14`; no observed budget excess. |
| Signed-distance ambiguity | 81 identical exact model samples for regions with different connectivity; sharp constants, bias cancellation, and the distinct dyadic blow-up examples also passed. |
| Finite transport/coherence identities and perturbation bounds | 4,000 trials passed; maximum gauge discrepancy `3.301e-15`, maximum common-reference pairwise discrepancy `1.423e-14`. |
| Inconsistent triangle transport | Exact integer quadratic form `-3`; the invalid squared-coherence construction gives `-1/3`. |

The scripts test analytic models, formulas, and counterexamples. They do not
sample the actual Julia renderer or certify Mandelbrot membership, provider
error, numerical transport, floating-point Physics, or audiovisual preference.
Small floating-point discrepancies are reported as numerical evidence, not
formal error bounds.

## Proof review and publication status

Independent review checked the actual catalog formulas, fixed-command
normalization, literal binary64 bit/rational correspondence, source provenance,
row/column indexing, exact spacing and halo margins, quadratic bias, phase/domain
bootstrap, and energy Taylor interpretation. It independently reproduced the
maximum finite coefficient magnitude 27/4 and sample-error record. The source
extraction is explicitly qualified as a separately rounded arithmetic model.
The actual full-halo Frechet bounds and separate Euclidean-gradient transfer
preserve their different norms. Runtime held-control/RK behavior is kept
separate from the exact-real recursive scheme.

No blocking mathematical defect was found. Source/module inventories,
relative documentation links, proof-placeholder scans and whitespace were
checked before publication. Runtime, training, canonical parameters, provider
contracts, dependency pins, audit implementation and workflow configuration
are unchanged.

The earlier targeted prior-art work did not establish publication-level
novelty of the package. Issues #157, #159 and #160--#163 remain open;
successful Lean verification is not a novelty certificate. PR #164 remains
a draft.

## CI scope

`lean-proofs.yml` builds the pinned formal package and runs the axiom audit for
formal-package changes. The commands and counts above are the observed local
results. The published workflow run for this increment is reported on
[draft PR #164](https://github.com/rhythmatician/FractalSync/pull/164).
No runtime or training code was changed, so the application test suite was
not rerun locally. Automatically triggered GitHub workflows are tracked
separately on the draft PR.
