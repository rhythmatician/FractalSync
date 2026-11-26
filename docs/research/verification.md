# Verification record: Lean geometry and mathematical research

Verified 2026-11-25 (America/Chicago) in the supplied Linux environment. The
final increment starts from PR #164 at
`6e3fea1725246787fc4e56dac3724c8ff2235f07`, with actual `main` at
`26eeb9a911a3e3005ce5b992e152ef429a53ae26` when the checkout was prepared.
The preceding 1,572-declaration package and all earlier increments are retained.
This record separates the new conditional operation-error bridge, retained
exact-real results, conventional geometry, and earlier optional-tool observations.

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

This final continuation used a fresh checkout. Elan and Lean were absent, so
Elan 4.4.0 and the official pinned Lean 4.24.0 Linux release were installed.
Mathlib and its transitive packages were fetched at the committed revisions.
An initial broad cache request encountered HTTP 429 responses; a targeted cache
request for the project's 88 direct Mathlib imports, with one concurrent download
and retries, completed and unpacked 2,568 dependency files. No toolchain or
dependency pin was changed.
No Mathlib source was changed. The toolchain, dependency sources, manifest, Lake
configuration, audit implementation/allowlist and workflow were unchanged.

From `formal/`, the complete command

```text
lake build --wfail
```

completed successfully with **2,731 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing 128 result modules | 1,517 |
| `PerturbedRungeKutta` | 8 |
| `PerturbedRungeKuttaWork` | 5 |
| `ConcretePerturbedRungeKutta` | 12 |
| `ConcretePerturbedRungeKuttaWork` | 7 |
| Existing fourteen analysis-support modules | 55 |
| **Total across 146 source modules** | **1,604** |

This adds **32 explicit declarations in four result modules**. The result
subtotal is 1,549 across 132 modules; support remains 55 across fourteen
modules. Counts include supporting lemmas and private helpers, not independent
research results. Definitions and generated helpers enter the transitive audit
separately. All 142 pre-existing proof module files are byte-for-byte unchanged.
The source inventory and umbrella import graph are reconciled with the
[formal inventory](../../formal/README.md).

## New checked scope

### Local operation errors and complete observed phase/work histories

The [operation-error extension](159-rk-operation-errors.md) records the actual
observed mechanical stages, RHS slopes, endpoint, powers and stored work.
Separate local residuals cover each operation. The generic proof establishes
stage containment in evaluation order, then bounds the complete mechanical
step and its signed energy/work defect. No distance-to-set premise is used.

For the unchanged quadratic fixture, m>=16 and h=(1/576)/m, the new one-substep
profile permits RHS and power errors up to h, and stage, endpoint and combined
work-update errors up to h². Actual local bounds close the 18h enclosing ball.
The whole-step comparison defect is at most 18000h², yielding all-time phase
error 3/(16m) and derived inner-region/query containment. The actual power
identity and energy differential pair (5,134) give local ledger defect at most
65536h² and every-prefix defect at most 1/(4m). The actual reference integral
then gives all-time work error at most 11/(9m).

One reference solution is selected before all meshes and admitted phase/work
records. Uniform convergence is conditional on a family satisfying the shrinking
operation-error budgets. It does not certify that fixed-precision arithmetic
satisfies them. The new profile uses one RK4 substep per outer interval; the
prior exact-real results for arbitrary positive internal counts remain intact.
Energy is exact mathematical energy at recorded phase values, and interpolation
uses ideal times k*h. Rounded energy reports, accumulated machine clocks,
control construction and compiled acceptance/refusal remain outside the result.

### Retained actual held power and signed RK quadrature

The [signed-work extension](159-rk-work-ledger.md) connects the runtime's
separate fifth RHS component to the existing mechanical energy. For any
supplied held covector q, the actual differential obeys
DE(z)[F_q(z)]=inner(v,q-Gv), with the fixture's unit drag. The proof uses the
actual ruler and potential jets; the conservative and connection terms cancel.
The same identity with q=drive(z) supplies the continuously reevaluated
reference power. Its norm/Lipschitz pair is (40,3072), derived from energy
differential (5,134) and reference field (8,400).

The generic quadrature uses the actual RK stage states and weights
1/6,1/3,1/3,1/6. If DE is K-Lipschitz and the stage slopes are bounded by M,
the absolute difference between endpoint energy change and signed stage work
is at most KM²s² for substep length s. The coefficient consists of an
endpoint Taylor term KM²s²/2 and an actual stage-differential variation term
KM²s²/2. The regional wrapper derives all stages and endpoints from an
enclosing-ball field bound. No high-order derivative or fourth-order claim
is used.

### Complete finite physical-work ledger

For n>0 equal internal substeps over an outer interval h, the signed energy
increments telescope, giving defect at most KM²h²/n. The fixture has K=134,
M=16 and T=1/576. Hence 134*16²=34304 and 34304*T²=67/648<1/8.
For m=N+1>=16 outer steps, every stored prefix k<=m satisfies

```text
abs(E(y_k)-E(y_0)-workTrajectory(n,N,k)) <= 1/(8*m*n).
```

All numerical state/stage membership, energy regularity and power premises
are discharged for the literal fixture and exact-drive outer caller policy.
At m=16384, the bounds are 2^-17 for n=1 and 2^-19 for n=4. The 1/n factor
improves the certified upper bound; it is not a measured runtime error ratio.
This ledger uses the actual drive-plus-drag stage power, distinct from the
retained earlier Taylor ledger of DE(y_i)[y_{i+1}-y_i].

### Actual continuous work and joint uniform convergence

The reference power trace is continuous and interval integrable. The actual
chain rule and interval FTC give integral_0^t P(z(s)) ds=E(z(t))-E(z(0))
throughout the same horizon, and cumulative reference work is 40-Lipschitz.
Combining the finite ledger with the actual energy Lipschitz constant 5 and
phase error 3/(16m) gives grid work error at most 1/(8mn)+15/(16m).
Affine interpolation adds 40h/2, giving, at every t in [0,1/576],

```text
abs(workInterpolant(n,N,t)-integral_0^t P(z(s)) ds)
  <= 1/(8*m*n)+35/(36*m) <= 9/(8*m).
```

One actual reference z is chosen before both the mesh and positive internal
count. The final quantified theorem proves simultaneous uniform convergence
of phase and cumulative work, with one mesh threshold independent of n.
It includes the actual reference work/energy identity. The interpolants
are affine through stored outer values. No instantaneous-power or derivative
convergence, RK dense output, conservation or IEEE execution is claimed.
Internal refinement at fixed m does not remove the outer holding-policy
discrepancy from the continuously reevaluated reference.

## Retained held-RK stage and trajectory results

The following results were added by the preceding 1,538-declaration increment
and supply the actual phase histories and domain evidence used above.

### Actual Runge--Kutta stages and method residuals

The [held-RK extension](159-held-runge-kutta.md) defines the actual classical
four-stage formula and its weighted endpoint on a real normed space. Regional
field boundedness and an enclosing-ball budget prove all stage and endpoint
membership in construction order. Repeated-substep induction retains the
original outer anchor and containing ball. The source-anchored consistency
bound is LMh²/2 relative to Euler; no high-order derivative assumption or
fourth-order error claim is introduced.

The phase adapter telescopes the actual substep increments and averages the
actual stage accelerations. Its second component agrees exactly with the
existing velocity-first comparison step. With phase speed bound M and
anchored acceleration-variation coefficient L, the actual averaged acceleration
residual is at most LMh and the endpoint comparison residual is at most 2Mh².
The theorem discharges stage membership from its enclosing-ball hypotheses
and retains a single held field across any positive internal substep count.

### Concrete stage closure and complete finite histories

The fixture has target acceleration Lipschitz constant 400 and holding
coefficient 680. The held acceleration's variation from the outer target
anchor is therefore at most 1080 times phase displacement. The complete
held phase vector is bounded by `8 + 1080 * displacement`. For an anchor
inside the ball R/2, with R=1/16, its enclosing ball of radius 16h lies in
the original phase region when 16h<=R/2. The actual field bound closes at
16 when h<=1/2160, before constructing any numerical stage.

The existing reference on T=1/576 has displacement<=1/72<R/4, so a closed
tube R/4 fits inside the inner ball R/2. The literal recurrence reconstructs
the exact fixture drive at each outer anchor and holds it through n>0
equal RK substeps. Actual stage averages give residual coefficients 17280
and 32 in the existing trajectory interface, with exact initial state.
The coefficient at most 108 yields all-time outer-state interpolation error
at most 3/(16m) for every m>=16. Every outer state and every interpolated
time lies in the inner region. The same reference works for every positive
internal count, including one and four. The 16384-outer-step corollary
gives uniform phase error<=2^-16; quantified uniform convergence over time
and all positive internal counts is also checked.

Position and velocity norms are Euclidean; the phase norm is their maximum.
The interpolant is affine through outer stored states, not RK dense output.
The 32h² residual compares two exact-real formulas; it is not a machine
rounding certificate despite occupying the existing stored-step-error slot.

### Every actual stage and endpoint query

The finite substep-query set explicitly contains the source, stages 2/3/4
and weighted endpoint. For every executed outer and inner index, each point
lies in the original certified phase region. Every reported coordinate query
with maximum coordinate error<=1/64 therefore retains all 36 mathematical
stencil addresses. Separate corollaries cover internal weighted endpoints
and every stored outer endpoint, including the final state. The same query
allowance is already included at every time of the outer-state interpolant.

### Runtime correspondence and field replacement

Source review checked one and four internal RK4 substeps, the single held
covector across the entire outer step, and reevaluation of potential, wall,
drag, metric and connection at every stage. The caller must supply the
fixture's exact drive at each outer anchor; `FrozenRollout::attempt_step`
itself accepts a supplied covector. The fixture parameters are epsilon=a=beta=1,
kappa=log2, mu=1/pi and unit intrinsic drive. Their runtime mapping is
lambda_sq=(log2)^2 and command magnitude 1/2 because MAX_DRIVE_FORCE=2.
No default-profile or arbitrary-caller certification is inferred.

The separate fifth RHS component accumulates power with RK weights and does
not feed back into the mechanical stages. The new signed-work extension now
certifies that exact-real quadrature separately. Provider acceptance/refusal
and IEEE stage/power arithmetic and summation remain outside the certificate.
Source review confirms that attempt_step has no energy/work-residual tolerance
or energy-drift refusal rule.

The [reuse boundary](157-field-replacement-boundary.md) records the proposed
dynamics-derived field as future work without defining it or changing this
formulation. The new stage, averaging and residual arguments depend on local
regularity evidence rather than actual distance-to-set identities. Generic
metric/mechanics results and conditional field calculus are distinguished
from literal fixture evidence and distance-derived geometry. Value approximation
or agreement of zero sets alone is not presented as a transfer theorem.

## Retained local-horizon and energy results

The following results were added by the preceding 1,479-declaration increment.
They remain unchanged and supply the concrete data used by this continuation.

### Actual local field and ruler bounds

The [local fixture extension](162-local-fixture-bounds.md) uses the same literal
256-coefficient finite asset, exact dyadic spacing, quadratic reference and
phase region as the preceding certificate. It keeps the raw-sample reference
bias P-h². The ideal quadratic has constant Hessian. Subtracting its samples
leaves a zero-extended lattice bounded by eta=2^-48, to which the actual global
Hessian query bound applies. Throughout the full physical halo, including
cell boundaries, Hessian variation is at most (32*eta/h³) times query distance.
The literal gain is checked to be at most 1.

A contractive coordinate chart and Euclidean dual norm calculation give the
reference Hessian bound 8 and local reference differential bound 7/2. The actual
finite-asset spline on the closed position ball of radius 1/16 then has value,
differential, Hessian and Hessian-variation bounds 1/4,4,12,1 respectively.
The value estimate retains both the quadratic spline bias and sample error.

`LocalRulerBounds` derives the scalar regularizer bounds on [-1/4,1/4]. It
retains the small field factor in the third derivative. The actual second
chain rule gives direct Hessian norm 19 and variation 769/4, enlarged to 20 and 200.
Riesz transfer and convex mean-value estimates give the actual regional pairs
rho=(2,1), grad rho=(1,20), D grad rho=(20,200), with rho>=1. Position and velocity
remain Euclidean; phase space uses the maximum norm.

### Inverse contraction and actual catalog cancellation

The [inverse and catalog extension](159-cancelled-mechanical-bounds.md) proves
that the actual inverse rank-one factor is a contraction, including at zero
gradient, and proves its exact resolvent identity. The inverse metric has norm
bound rho². Its regional operator pair is (4,164) for the local ruler/gradient
inputs. This avoids bounding the subtractive inverse formula term by term.

Raising the actual catalog first gives rho/W times u, the raised wall term,
normalized heading, and negative velocity. Exact checked propagated pairs are:

| Quantity | Norm | Lipschitz |
| --- | ---: | ---: |
| Actual unraised catalog before enlargement | 275/48 | 1249/6 |
| Actual unraised catalog used by certificate | 6 | 210 |
| Raised catalog | 55/12 | 701/4 |
| Connection evaluated twice on velocity | 8/3 | 1748/9 |
| Acceleration before enlargement | 29/4 | 13301/36 |
| Acceleration used by certificate | 8 | 400 |

The actual ruler, force, potential, acceleration and chosen research parameters
are unchanged. No empirical maximum or assumed local force constant is used.

### Constructed longer reference and unconditional finite run

The [local-horizon theorem](159-local-horizon.md) keeps phase speed 8 separate
from acceleration Lipschitz 400. It constructs an actual ODE reference with
regional uniqueness, endpoint derivatives, displacement<=8t and an interior
radius-R/2 tube. At R=1/16 its exact horizon is

```text
T = 1/576.
```

This is approximately 0.0017361111 in model time, about 4.4444e18 times the older
certified horizon. Neither a physical-seconds conversion nor a runtime step
size recommendation is inferred.

For m=N+1 and tau=T/m, the perturbed-history theorem retains explicit initial
error<=tau, acceleration residual<=tau and stored-step error<=tau². The actual
interpolation coefficient is at most 16. Every nonempty mesh satisfies the
tube condition, with uniform phase error<=1/(36m), open-ball membership and
all 36 mathematical stencil addresses for each reported query within 1/64 of
the interpolated position. The query displacement remains an admitted budget.

The new `exactTrajectory` recursively uses the unchanged actual acceleration
and the velocity-first position update over exact reals at the new horizon.
Its residuals vanish by definition. The unconditional 2048-step theorem gives
uniform phase error<=2^-16; the general estimate is the stronger 1/73728 on
that mesh. Every stored state lies in the open phase ball by evaluating the
interpolant theorem at actual grid times. The older microscopic-horizon
certificate is retained and not modified.

### Concrete held-drive residual

The [holding extension](159-held-control-residual.md) checks the actual
acceleration difference when the supplied drive covector is frozen at a
sample state while conservative force, drag, inverse metric and connection
are reevaluated at the current state. The actual fixture drive has pair (4,170).
The concrete result has no remaining ruler/force/asset regularity premise:

```text
norm(A_hold(q,z)-A(z)) <= 4*delta_Q + 680*dist(z0,z),
provided norm(q-Q_drive(z0)) <= delta_Q and both states lie in the region.
```

A separately supplied displacement bound dist(z0,z)<=8*tau gives the holding
term 5440*tau. This coefficient is not silently substituted for the trajectory
theorem's unit residual coefficient.

Source inspection confirms `FrozenRollout::attempt_step` holds the supplied
control covector through all RK stages; drag is recomputed at each stage.
`Rk4FourSubsteps` keeps that covector across the entire outer step. The age of
the held command is therefore measured from the outer-step start. At that
increment, actual RK stage containment and method residuals were unproved.
The held-RK extension above now supplies those exact-real results; IEEE
evaluation remains outside the certificate. The reference speed bound alone
still does not certify numerical stage motion.

### Actual energy regularity and a finite Taylor ledger

The [local energy extension](159-local-energy-bounds.md) uses the same actual
conservative potential. Its actual gradient has pair (17/16,43/2). The energy
covector budget evaluates to (601/144,267/2), giving the exported actual
Frechet differential pair (5,134). The energy is five-Lipschitz on the closed
phase ball.

The contained-step Taylor constant is 67. The new actual finite run discharges
both endpoint premises at every step. With J_i=DE(y_i)[y_(i+1)-y_i], it proves

```text
|E(y_(i+1))-E(y_i)-J_i| <= 1/(64*m²),
|E(y_m)-E(y_0)-sum_i J_i| <= 1/(64*m).
```

The 2048-step specializations are 2^-28 per step and 2^-17 for the accumulated
Taylor ledger. The retained actual differential sum is essential: these are
not conservation, total driven/damped energy-drift or physical-work-quadrature
bounds. The longer horizon also means its numerical remainder should not be
compared directly with the preceding microscopic-horizon remainder.

### Retained results and remaining scope

All preceding #160 results, including zero-floor attainment, subsequential
selected-path convergence, selected-length convergence, flat threshold and
classification at/below threshold, and ambient curved-Shore connection/curvature
transfer are unchanged. No new smooth-minimizer, cut-locus, bifurcation, atlas,
reach or stronger singular-limit theorem is claimed in this increment.

The concrete reference remains a quadratic test fixture. Production distance
fidelity, compiled extraction, storage/mask/machine-index correspondence,
query rounding and IEEE evaluation still need evidence. Actual control
construction, operational acceptance/refusal, IEEE power/summation, changing
commands, impulses, combined steering and continuation beyond the local
interval remain open. The exact-real stage/domain, method-residual and signed-work
certificates are limited to the stated fixture caller policy. Runtime behavior,
training, canonical parameters and provider contracts are unchanged.

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

passed for **3,573 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 3573 project declarations
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

The preceding 1,364-declaration increment ran the concrete-fixture extraction check:

```text
python formal/fixtures/check_operational_quadratic.py
```

It passed for 256 coefficients, denominator 144115188075855872, magnitude bound 7,
sample-error bound 2^-48 and the exact-rational rounding cross-check. The command
also checks current source-function provenance, all recorded bit patterns,
and correspondence with the literal Lean table. Its output explicitly says
`rust_execution_claimed: false`. Rust/Cargo were unavailable; no executable
Rust comparison is claimed. The literal/sample inequalities are also separate
kernel proofs covered by the transitive audit. This continuation changes neither
the fixture nor extractor and does not repeat that independent extraction check.

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

The final independent review checked sequential containment, observed-state
RHS comparisons, the completed-step error adapter, actual power identity and
stored-work telescoping, common-reference quantifiers, and the distinction
between nonzero finite-mesh allowances and shrinking-budget refinement.
Runtime-source review confirmed RK stage order and weights, the held covector,
and the power coordinate's independence from mechanical stage construction.
The final source inventory has 1,604 declarations across 146 modules, with no
unimported proof module or unexpected proof placeholder/axiom token. All 659
relative file links in the research notes and formal inventory resolve.

Independent review checked signed-power semantics, actual RK stage weights,
outer-anchor holding policy, stage/endpoint inclusion, both telescoping levels,
energy differential cancellation, actual interval-integral hypotheses and all
mesh constants. A separate review of the final convergence module checked
one-reference quantifier order, derived reference-region membership, grid work
comparison, endpoint-inclusive interpolation and a mesh threshold independent
of the positive internal count. Source review verified the runtime's actual
power rule, work initialization/addition and separation from mechanical stages;
it also checked that the engine has no work-residual acceptance criterion.
The earlier stage, trajectory and parameter-mapping reviews remain applicable
because those proof modules and runtime sources are unchanged.

No blocking mathematical defect was found. Source/module inventories,
relative documentation links, proof-placeholder scans and whitespace were
checked before publication. Runtime, training, canonical parameters, provider
contracts, dependency pins, audit implementation and workflow configuration
are unchanged.

The earlier targeted prior-art work did not establish publication-level
novelty of the package. Issues #157, #159 and #160--#163 remain open;
successful Lean verification is not a novelty certificate. The conditional
operation-error increment completes the bounded scope selected for PR #164;
its wider research and implementation obligations remain separate.

## CI scope

`lean-proofs.yml` builds the pinned formal package and runs the axiom audit for
formal-package changes. The commands and counts above are the observed local
results. Hosted workflow results for the published increment are tracked on
[PR #164](https://github.com/rhythmatician/FractalSync/pull/164).
No runtime or training code was changed, so the application test suite was
not rerun locally. Automatically triggered GitHub workflows are tracked
separately on the PR.
