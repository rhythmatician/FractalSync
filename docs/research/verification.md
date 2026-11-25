# Verification record: Lean geometry and mathematical research

Verified 2026-11-25 (America/Chicago) in the supplied Linux environment. This
continuation starts from draft PR #164 at
`57c23b158254b794d12b3a3d370a6af24a7358dd`, whose reported main-branch base is
`55df760ff706210dcce3c88e79e476a45c5331f0`.
The preceding 1,169-declaration package and all earlier increments are retained.
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
This continuation reused that installation and obtained missing pinned cache
entries for the bounded-continuous-function Arzelà–Ascoli import. No Mathlib
source was changed. The toolchain, dependency sources, manifest, Lake
configuration, audit implementation/allowlist and workflow were unchanged.

From `formal/`, the complete command

```text
lake build --wfail
```

completed successfully with **2,695 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing eighty-nine result modules | 1,115 |
| `BoundedLipschitz` | 14 |
| `MechanicalBounds` | 1 |
| `RulerJetBounds` | 10 |
| `OperationalRulerBounds` | 23 |
| `OperationalHalo` | 11 |
| `OperationalMechanicalBounds` | 7 |
| `QuantitativeDynamics` | 5 |
| `OperationalTrajectoryCertificate` | 4 |
| `ZeroFloorGeodesics` | 11 |
| Existing thirteen analysis-support modules | 54 |
| `Analysis/BoundedSpeedCompactness` | 1 |
| **Total across 112 source modules** | **1,256** |

This adds **87 explicit declarations in nine result modules and one support
module**. The result subtotal is 1,201 across ninety-eight modules; support is
55 across fourteen modules. The counts include supporting lemmas and are not
counts of independent research results. A comment/string-stripped source
inventory reproduced the preceding count of 1,169 before counting this
increment. Every result-module row in the
[formal inventory](../../formal/README.md) is reconciled against its source
count and umbrella import. No existing Lean theorem declaration changed.

## New checked scope

### Actual zero-floor attainment and selected minimizing paths

The [zero-floor geodesic extension](160-zero-floor-geodesics.md) works in any
finite-dimensional real normed space with nonempty closed Shore S, real a,
and fixed off-Shore endpoints x,y in the same component of its complement.
The existing actual off-Shore AC infimum D0 is attained by an ordinary AC
path gamma. Its entire image stays in that component, with the positive
clearance d(x,S)*exp(-D0), and

```text
L0(gamma) = D0(x,y)
D0(gamma(s),gamma(t)) = D0(x,y)*abs(s-t),  0<=s,t<=1.
```

For every positive floor sequence tending to zero and every selection of
constant-speed regularized geodesics between those endpoints, the proof
extracts a strictly increasing subsequence converging uniformly in the
original ambient metric to such a zero-floor minimizer. Floor monotonicity,
smooth Shore, strict convexity and minimizer uniqueness are not required.
Coincident endpoints are included.

The proof derives a common ambient Lipschitz bound and invokes the pinned
bounded-continuous-function form of Arzelà–Ascoli. Joint continuity of each
fixed positive-floor intrinsic distance passes the speed bound to the limit.
The existing metric-length bridge makes the limit AC and bounds every
positive-floor integral. Logarithmic ruler confinement gives actual positive
clearance; the checked density comparison then bounds its original zero-floor
integral. No weak derivative convergence or assumed lower semicontinuity of
that integral is used.

A separate theorem applies to every selected family of actual AC
positive-floor minimizers, without a constant-speed premise. It proves
eventual Shore avoidance and convergence of their actual unregularized
integrals to D0. With m=d(x,S)*exp(-D0), its eventual squeeze is

```text
D0 <= L0(gamma_n) <= D0/(1-epsilon_n^2/m^2).
```

This is full-sequence length convergence. Path convergence is subsequential
and has the explicit constant-speed selection hypothesis. No convergence of
velocities, arbitrary entire selected path family, or varying endpoint pairs
is asserted.

### Explicit actual ruler and mechanical bounds

The [ruler-bound extension](159-operational-ruler-bounds.md) concerns the
literal floor-selected spline P on the actual Euclidean plane. For coefficient
magnitude bound B>=0, spacing h>0 and floor epsilon>0, put

```text
G = 8B/(3h),  H = 68B/(9h^2),  J = 32B/h^3
Hrho = H+G^2/epsilon
Jrho = J+3GH/epsilon+3G^3/epsilon^2.
```

For rho=sqrt(P^2+epsilon^2), the checked conclusions are
epsilon<=rho<=sqrt(B^2+epsilon^2), norm(gradient rho)<=G, and
norm(D(gradient rho))<=Hrho. The global Lipschitz constants of rho, its actual
Riesz gradient and that gradient's actual derivative are G,Hrho,Jrho.
The last field also has a genuine Frechet derivative bounded by Jrho;
`HasFDerivAt` is included, so a totalized derivative supplies no shortcut.
The L2-to-coordinate-max chart is proved contractive and both covector slots
are transported before Euclidean Riesz dualization.

The [mechanical-bound extension](159-operational-mechanical-bounds.md)
propagates pairs (norm bound,Lipschitz constant) through actual sums,
differences, bilinear operations, reciprocals, and composition. It unfolds the
existing inverse-metric force minus connection(v,v), using rho>=epsilon and
W=1+a^2*norm(u)^2>=1. No unit-gradient bound is assumed.
`operationalAccelerationBudget` substitutes the actual ruler jets and the
velocity pair (norm(p.velocity)+R,1) on a closed phase ball. The phase norm is
the maximum of the two Euclidean norms. It produces explicit arithmetic
acceleration norm and Lipschitz bounds; only force norm and variation are
external coefficient hypotheses. The expressions are conservative exact-real
bounds, not sharp estimates or measured runtime constants.

### Finite-asset locality and mathematical query addresses

The [physical-halo extension](162-operational-halo.md) defines the actual
half-open normalized full-stencil halo and its strict interior. The minimum
of the four signed physical face distances is a concrete halo margin.
If phase-ball radius R and reported coordinate displacement delta satisfy
R+delta<margin, every such query stays in the open halo. All 36 selected
integer stencil indices, including row-major addresses, lie in the stored
rectangle. Cell changes are allowed.

Finite stored coefficient agreement gives equality of every physical query
jet. On the open halo it gives equality of actual ruler germs, Riesz gradients,
gradient derivatives and same-force mechanical acceleration germs. A zero
extension is globally bounded from only finite stored coefficient evidence.
The mechanical certificate is transferred back to the arbitrary original
extension by actual acceleration equality on the whole phase ball. No bound
on the original extension outside the asset is required.

These are exact-real query and mathematical integer statements. Machine
normalization, floor/casts, integer overflow, storage, masks and an actual
query-displacement budget remain separate obligations.

### Constructed quantitative trajectories and complete interpolation control

The [quantitative construction](159-quantitative-dynamics.md) uses actual
acceleration Lipschitz constant L and phase-speed bound B on a specified
closed phase ball of radius R>0. It constructs an actual solution on the
explicit horizon T=R/[4(B+1)]. The larger open Picard interval supplies
ordinary derivatives at both endpoints of [0,T]. The proof gives
norm(z(t)-p)<=B*t<R/4 and a radius-R/2 tube contained in the phase ball.
Its ODE certificate records exactly the input L,B and supports regional
uniqueness.

The [finite-asset trajectory certificate](159-operational-trajectory-certificates.md)
instantiates these constants from the actual operational acceleration budget,
using phase speed max(norm(p.velocity)+R,A0). It constructs the reference
without a preexisting solution or existential margin premise. With step
tau=T/(N+1), residual norm<=tau*eta, stored-state step error<=tau^2*r and
initial error<=tau*I, define

```text
C = refinementCoefficient(T,L,B,eta,r,I)+B/2.
```

The explicit mesh condition 2*T*C<=R*(N+1) gives continuity of the actual
stored-state interpolant, error<=tau*C at every time, whole-interpolant
phase-ball membership, and valid displaced reported queries with all 36
mathematical addresses. The residual and rounding contracts remain conditional
on a state already lying in the phase ball; the proof establishes that
membership by the existing simultaneous induction. The uniform refinement
theorem includes eventual query validity on the whole interval.

The results require concrete force bounds and the stated error scaling.
They do not certify a particular asset, IEEE execution, a runtime interpolation
policy, equality between interpolated velocity and the derivative of
interpolated position, global continuation, impulses, or combined steering.
Fixed machine precision does not automatically yield vanishing step-scaled
budgets.

The prior flat threshold and all short-segment equality cases, actual ambient
curved-Shore connection/curvature transfer, provider reference-fidelity results,
metric-length/nearest-Shore classification and #161--#163 results remain
unchanged. This increment does not assert publication-level novelty.

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

The new `BoundedSpeedCompactness` module adds one project theorem using
the pinned bounded-continuous-function Arzelà–Ascoli and compact sequential
extraction APIs. It preserves the original endpoint and speed bounds while
producing actual uniform subsequence convergence. The package now has fourteen
analysis-support modules and 55 explicit support theorem/lemma declarations.

## Transitive axiom audit and negative controls

```text
lake env lean Audit.lean
```

passed for **2,697 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 2697 project declarations
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

Independent review checked the reciprocal and bilinear budget arithmetic,
all terms of the actual mechanical expression, Euclidean/coordinate norm
transport, actual Riesz gradient/Hessian identities, and finite zero-extension
locality. The explicit scalar and field derivative constants require no
unit-gradient premise. Force regularity and sample/IEEE evidence remain
separate inputs.

The quantitative dynamics review checked the Picard interval and endpoint
derivatives, actual reference displacement and tube, phase-speed bound,
mesh inequality, conditional residual/domain induction and uniform
interpolant convergence. Physical-halo margins are in physical units, and
the address proof is about mathematical integer floor and full finite stencils.

The zero-floor review checked sup-metric uniform compactness, the finite-prefix
shift for arbitrary nonmonotone floors, ordinary AC admissibility, actual
off-Shore integrability, clearance and attainment, and constant-speed equality
without presupposing a zero-floor metric bundle. It separately checked the
arbitrary-parameter unregularized length squeeze. No derivative convergence
or whole-selection path convergence is inferred.

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
