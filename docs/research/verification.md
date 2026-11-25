# Verification record: Lean geometry and mathematical research

Verified 2026-11-24 (America/Chicago) in the supplied Linux environment. This continuation starts
from draft PR #164 at `33fe7f74f441388c781e8d6c7ef0ba203820e54c`, whose
main-branch base is `55df760ff706210dcce3c88e79e476a45c5331f0`.
The earlier 172-, 291-, 400-, 596-, 726-, 906-, and 1,043-theorem increments are retained.
This record distinguishes new kernel-checked results, retained analysis support,
conventional geometry, and earlier optional-tool and numerical observations.

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
This continuation reused that working installation and fetched the missing
pinned cache entries for inverse-trigonometric derivatives, Riesz gradients,
Picard–Lindelöf, Grönwall, and the Euclidean product inner product. The committed
toolchain, dependency sources, manifest, Lake configuration, audit, and Lean
workflow were not changed. Named local phase-norm instances avoid the earlier
generated-name collision without changing mathematical norms.

From `formal/`, the complete command

```text
lake build --wfail
```

completed successfully with **2,684 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing seventy-four result modules | 989 |
| `ACScalarVariation` | 3 |
| `FlatShoreCalibration` | 13 |
| `FlatShoreMinimizers` | 6 |
| `FlatShoreIsometries` | 9 |
| `FlatShoreThreshold` | 5 |
| `MetricConnectionUniqueness` | 3 |
| `EmbeddedShoreConnection` | 17 |
| `EmbeddedShoreCurvature` | 10 |
| `OperationalFieldFidelity` | 21 |
| `OperationalFieldReference` | 9 |
| `OperationalFieldHessianError` | 9 |
| `OperationalFieldComputedError` | 1 |
| `TrajectoryInterpolation` | 9 |
| `LocalDynamics` | 6 |
| `OperationalDynamics` | 5 |
| Existing thirteen analysis-support modules | 54 |
| **Total across 102 source modules** | **1,169** |

This adds **126 explicit declarations in fifteen result modules**. The result
subtotal is 1,115 across eighty-nine modules; support remains 54 across thirteen
modules. Counts include supporting lemmas and are not counts of independent
research results. A comment/string-stripped source inventory reproduced the
preceding count of 1,043 before counting this increment. Every result-module
row in the [formal inventory](../../formal/README.md) is reconciled against its
source count and umbrella import. No existing theorem declaration changed.
No new analysis-support module was added.

## New checked scope

### Global flat minimum, exact threshold, and all AC equality cases

The [flat calibration extension](160-flat-global-minimum.md) uses the actual
scalar function acos(c*cos(y)/sqrt(1+r²)), with 0<c<1. Its actual differential
is bounded by the comparison speed sqrt(vr²/(1+r²)²+vy²/(1+r²)) using a checked
sum-of-squares certificate. That speed is bounded by the actual flat graph
speed. Lipschitz composition, a.e. chain rule, and the AC fundamental theorem
give an integrated endpoint bound. Continuity as c tends to 1 proves the
unrestricted lower bound ell<=graphLength for 0<=ell<=pi, including the
endpoint where differentiating arccos directly would be invalid.

The lower bound passes through the actual AC candidate infimum. Full
candidate-length set equality under positive dilation and tangential
translation gives, for epsilon>0, arbitrary real a,b, and
0<=L<=pi*epsilon,

`D_epsilon,a((0,b),(0,b+L)) = L/epsilon`.

If an actual AC path has this length, equality of the post-limit integrated
speed bounds forces r*dot(r)=0 almost everywhere. Its AC squared normal trace
is therefore constant and zero. On Shore, graph length is actual scalar total
variation. The generic scalar AC theorem proves that variation equals the
signed endpoint increment exactly for nondecreasing traces. The final physical
iff thus classifies every minimizing AC path as a monotone Shore traversal,
including pauses, L=0, and L=pi*epsilon.

Combined with the previous strict sine competitor, both a strict pairwise
distance improvement and existence of an actual shorter AC path are equivalent
to L>pi*epsilon. The infimum-to-path step uses strict infimum approximation,
not attainment. All minimizers beyond the threshold, general minimizer
regularity, conjugacy, index forms, and the global cut locus remain separate.

### Actual ambient connection and embedded curvature transfer

The [ambient connection extension](160-ambient-shore-connection.md) starts with
the actual set distance and a local NormalPatch, actual local Frenet data,
a parameter neighborhood, and a nonfocal invertible normal differential.
The inverse normal chart proves actual regularity of sqrt(epsilon²+infDist²)
through Shore. Mathlib's actual Riesz gradient supplies the true differential;
its actual derivative is the Hessian, whose symmetry is derived from calculus.

A metric equality on a neighborhood is differentiated with both chart-Hessian
contributions retained. Torsion freedom and metric compatibility, together with
the new positive-metric connection uniqueness theorem, identify the nonlinear
pullback of the actual ambient compact connection with the curved coefficients.
The final connection theorem assumes neither ambient jet formulas nor a
metric/connection transformation identity. A frame wrapper derives Jacobian
invertibility from actual tangent/normal data and the nonfocal factor.

The identification persists as a germ. Actual curvature commutators agree;
inverse-Jacobian pullback also transports the actual ambient-coordinate
commutator for regular fields. Intermediate covariant-field regularity is
derived from the actual ruler. The normal-frame Gram denominator is positive.
For a C3 normal map and a locally C2 curvature profile, the actual ambient
normal-frame ratio equals the full previously computed curved formula and
equals +1 on Shore. The connection stage needs only a C2 normal map and
differentiable curvature profile.

These remain coordinate/bilinear-form/commutator objects. No manifold-bundle
atlas gluing, numerical normal radius for a concrete representation, general
reach theorem, or regularity at fractal/cut-locus points is asserted.

### Concrete reference fidelity and complete query jet budgets

The [provider fidelity extension](162-operational-reference-fidelity.md) compares
the literal operational spline with a reference field under actual local jet
regularity, anchored Hessian variation L, and sample error eta for only the
36 selected physical nodes. Those nodes lie within 3h in the coordinate product
maximum norm. Actual Taylor remainders and exact quadratic reproduction give

* corrected value error <= eta+9Lh³/2, after subtracting h²*trace(Hf)/4;
* differential error <= 8eta/(3h)+12Lh²;
* Hessian error <= 68eta/(9h²)+34Lh.

The value bias is retained for the raw sampler. The reference region is convex
and contains the query and all selected nodes; the differential/Hessian results
also require it to be a neighborhood of the query. Actual Hessian symmetry and
literal first/iterated derivative identification are derived. The trace is the
ordinary Cartesian Hessian trace, not a scale-metric Laplace–Beltrami operator.

A concrete third-basis Bernstein certificate supplies the global Hessian query
gain 32B/h³ under a uniform lattice magnitude bound B. The final combined
theorem adds each reference error, its actual-query displacement times the
appropriate gain, and a separately supplied reported-result evaluation error.
The gains are 8B/(3h), 68B/(9h²), and 32B/h³. Arbitrary cell crossings are allowed.
Evidence for actual true-distance samples and their regularity, machine
floor/casts/masks/storage, and IEEE/query-rounding error remains separate.

### Actual local mechanics and continuous-interpolant convergence

The [local trajectory extension](159-local-trajectories.md) defines the actual
piecewise-affine interpolation of stored phase states, proves continuity and
all grid values, and derives the off-grid bound e+B*h/2 from the two endpoint
errors and an actual B-Lipschitz reference curve. The reference Lipschitz bound
comes from the ODE speed certificate. The prior mesh-scaled recurrence gives
an all-time bound h*(refinementCoefficient+B/2). If it lies within the fixed
reference tube, every interpolated point is in the phase region. No convexity
of that region is needed. The uniform all-time convergence theorem includes
eventual domain membership and uses T>0; the prior grid theorem retains T=0.

Actual C1 acceleration at an initial phase state gives local Picard–Lindelöf
existence, a uniform ODE certificate on a phase ball, a positive interior tube,
and regional uniqueness. The horizon, radius, and uniform constants are
existential. The checked mechanical adapter uses the actual metric inverse
and compact connection.

The operational specialization sets rho=sqrt(epsilon²+P(chart x)²), with the
literal globally C4 physical spline P. It uses the actual Euclidean Riesz
gradient of rho and its actual derivative. Their C3/C2 regularity is derived;
the physical query differential is identified by the chain rule. A C1 force
therefore gives an actual local mechanical trajectory for this specific metric,
without a signed-distance fidelity premise or arbitrary jet identities.
Positive spacing is the intended physical regime; the regularity proof also
covers totalized exact-real normalization at other spacings.

The theorem does not compute usable numerical regional constants or prove that
the existential phase ball stays inside a finite asset halo. The interpolant
is a specified analysis function, not an asserted runtime interpolation, and
its position derivative need not equal its interpolated velocity. Fixed
nonvanishing errors, global existence, time-dependent or impulsive dynamics,
event handling, combined steering, and IEEE certification remain separate.

The earlier fixed-pair zero-floor classification, AC metric-length bridge,
arbitrary-endpoint attainment, nearest-Shore classification, radial analysis,
mechanical contracts, and #161--#163 results remain intact. This increment does
not add convergence of selected minimizing curves or a publication-level
novelty claim.

## Attributed analysis backport and retained compatibility support

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

## Transitive axiom audit and negative controls

```text
lake env lean Audit.lean
```

passed for **2,461 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 2461 project declarations
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

The earlier increments reviewed the radial formula, sharp additive constant,
AC infimum, field calculus, complete fixed-pair zero-floor classification,
metric-length identity and attained/nearest-Shore minimizing paths. Those
results remain unchanged.

Independent review checked the calibration's actual differential, denominator
domains, sum-of-squares identity, Euclidean/coordinate norm distinction, AC
integration, and the c→1 endpoint argument. Equality review checked the
post-limit integral equality, a.e. radial product, AC square-radius rigidity,
scalar variation criterion, and physical dilation/translation. The all-AC
statement includes zero length and the threshold, with pauses. Full candidate
sets and strict infimum approximation do not assume minimizing-path existence.

The ambient review checked actual regularized-distance smoothness through the
unsigned-distance cusp, Riesz-gradient identification, actual Hessian symmetry,
metric-germ differentiation with both chart-Hessian terms, positive-metric
connection uniqueness, and curvature transport with derived intermediate
regularity. Positive normal Gram denominators and sufficient local smoothness
are explicit. No global manifold gluing or fractal regularity is inferred.

Provider review checked actual local Taylor remainders and node membership,
quadratic sampling bias, induced operator norms, the concrete third-basis
table, global Hessian query stability, and the separate reference/query/report
error terms. It distinguishes sufficient mathematical sample certificates
from evidence that a real asset satisfies them and from IEEE execution.

Dynamics review checked finite interpolation joins and both endpoints,
the two-endpoint B*h/2 estimate, actual ODE speed-to-Lipschitz passage,
noncircular grid/domain induction, and all-time tube membership. Local
Picard–Lindelöf existence and Grönwall uniqueness use the pinned Mathlib APIs.
The operational specialization uses actual Euclidean Riesz jets of the
literal spline ruler. Its existential constants do not certify a finite halo
or an executable timestep, and no derivative-convergence or hybrid claim is made.

No blocking mathematical defect was found. Relative documentation links,
source/module inventories and the staged whitespace check were checked before
publication. Runtime, training, canonical parameters, provider contracts,
dependency pins, audit implementation and workflow configuration are unchanged.

The earlier prior-art search was targeted. It found exact prior art for the
phase-harmonic Lipschitz bound and strong prior art for the location-scale
hyperbolic structure and Lipschitz extension envelopes. It did not establish
publication-level novelty of this package. Issues #157, #159 and #160--#163
remain open; successful Lean verification is not a novelty certificate.
PR #164 remains a draft.

## CI scope

`lean-proofs.yml` builds the pinned formal package and runs the axiom audit for
formal-package changes. Its configuration was checked against the pinned
official Lean action's inputs in the earlier increment. The commands and
counts above are the observed local results. The published workflow run for
this increment is reported on
[draft PR #164](https://github.com/rhythmatician/FractalSync/pull/164).
No runtime or training code was changed, so the application test suite was not
rerun locally for this research-only increment. Automatically triggered
GitHub workflows are tracked separately on the draft PR.
