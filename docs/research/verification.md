# Verification record: Lean geometry and mathematical research

Verified 2026-11-24 (America/Chicago) in the supplied Linux environment. This continuation starts
from draft PR #164 at `072dd12465c7cee08cae8aa5b60cff8d872d45b7`, whose
main-branch base is `55df760ff706210dcce3c88e79e476a45c5331f0`.
The earlier 172-, 291-, 400-, 596-, 726-, and 906-theorem increments are retained.
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
This continuation reused that working installation. The committed toolchain,
dependency sources, manifest, Lake configuration, audit and Lean workflow were
not changed. Integration exposed a generated-name collision between anonymous
local product-norm instances; the new dynamics instances now have explicit,
unique names. No theorem hypotheses or mathematical norms changed to resolve it.

From `formal/`, the complete command

```text
lake build --wfail
```

completed successfully with **2,657 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing sixty result modules | 852 |
| `MovingFrames` | 13 |
| `CoordinateConnectionRegularity` | 5 |
| `ConnectionFormCurvature` | 8 |
| `NormalRadius` | 12 |
| `EmbeddedNormalPatch` | 12 |
| `FlatShoreShortcuts` | 11 |
| `OperationalFieldGluing` | 5 |
| `OperationalFieldJoins` | 7 |
| `OperationalFieldLattice` | 4 |
| `GlobalOperationalField` | 11 |
| `OperationalFieldQueryCorrespondence` | 17 |
| `OperationalFieldGlobalError` | 7 |
| `TrajectoryError` | 18 |
| `TrajectoryRefinement` | 7 |
| Existing thirteen analysis-support modules | 54 |
| **Total across eighty-seven source modules** | **1,043** |

This adds **137 explicit declarations in fourteen result modules**. The result
subtotal is 989 across seventy-four modules; support remains 54 across thirteen
modules. Counts include supporting lemmas and are not counts of independent
research results. A comment/string-stripped source inventory reproduced the
preceding count of 906 before counting this increment. Every result-module row
in the [formal inventory](../../formal/README.md) is reconciled against its
source count. No new analysis-support module was added.

## New checked scope

### Moving frames and nonlinear coefficient regularity

The [moving-frame extension](160-moving-frames.md) differentiates the actual
inverse-frame field, proves the inhomogeneous connection-form gauge law,
actual covariant-derivative and curvature-commutator covariance, and composition
of successive frame changes. Base directions and their Lie bracket remain
unchanged when only section coordinates change. A C2 frame with C1 coefficients
gives an actual C1 transformed coefficient field. The nonlinear specialization
is definitionally the earlier chart-Hessian law; a C3 chart, C1 original
coefficients and an invertible actual Jacobian give a derived positive ball of
invertibility and C1 transformed coefficients.

A continuous trilinear curvature tensor allows distinct base and fiber spaces.
Actual section derivatives cancel from the commutator; C2 frame and C1 coefficient
hypotheses discharge the regularity needed for tensor gauge conjugacy. Compatibility
lemmas identify the base-equals-fiber specialization with the existing tangent
objects. These are coordinate/trivial-fiber constructions, not atlas-glued
manifold or vector-bundle connections.

### Derived normal radii and a genuinely local chart interface

The [normal-radius extension](160-derived-normal-radii.md) derives the actual
Taylor normal projection and the chord lower bound m=v-L*delta/2 from a tangent
speed lower bound v and actual derivative variation L on a parameter patch.
Together with remote Shore distance at least d, it proves the positive uniform
radius R=min(m²/L,d/2) when L,m,d are positive. Actual second derivatives supply
the variation estimate; actual C2 neighborhood regularity on a compact convex
patch supplies an existential finite positive L.

Compactness of the base image and disjointness from the closed remote Shore
supply d. For an injectively parameterized compact curve, the latter disjointness
is derived for a compact base patch inside an open parameter patch. The same
jet bounds derive curve injectivity, a normal-patch certificate, and injectivity
of the open normal strip. The localized interface proves actual distance for
abs(r)<=R, unique foot for abs(r)<R, and an IFT chart with ambient signed-distance,
ruler derivative and metric pullback. The chart needs a parameter neighborhood,
actual local Frenet/C1 data, a planar spanning equivalence and a nonfocal factor.
Numerical constants, more general reach, and ambient connection/curvature
identification remain separate.

### Actual physical pairwise shortcut

The [flat shortcut extension](160-flat-second-variation.md) lifts the earlier
sine competitor to the actual Euclidean plane and reparameterizes it to [0,1].
Its Euclidean AC property, endpoints and graph length are checked. The length
is exactly the previously differentiated amplitude functional, so the actual
pairwise path infimum is bounded by this competitor. Actual metric, coordinate
length and AC graph-length dilation then prove, for epsilon>0 and L>pi*epsilon,
D_epsilon,a((0,0),(0,L))<L/epsilon. No attainment or geodesic regularity is assumed.
The equality case, converse below threshold and full minimizer/cut-locus
classification remain separate.

### Global operational evaluator and real query errors

The [global provider extension](162-global-operational-field.md) calculates the
coherent adjacent-cell joins through order four from the six actual quintic
coefficient rows and an arbitrary integer coefficient lattice. A generic
floor-selection lemma glues actual derivatives on neighboring closed cells;
two coordinate applications cover grid edges and vertices. The literal global
floor-selected evaluator is proved C4. Its physical first and iterated Frechet
derivatives equal the selected value/gradient/Hessian formulas, with the actual
spacing factors and cell origin.

Ideal full-halo inequalities imply every six-by-six index and row-major integer
address is in range. Arbitrary lattice extensions agreeing on the finite asset
select the same jets under those inequalities. A uniform lattice magnitude B
supplies global operator bounds 8B/(3h), 68B/(9h²), and corresponding value and
differential query bounds across arbitrary cell boundaries, in the coordinate
product maximum norm. Combined coefficient/query value error is bounded by
eta+8B/(3h)*norm(deltaQuery). Reference-distance fidelity, machine floor/casts,
validity masks, storage decoding, actual query-rounding and IEEE arithmetic
certificates remain open.

### Actual trajectory convergence and phase-region persistence

The [trajectory extension](159-trajectory-error.md) derives stability of the
existing velocity-then-position update and a local defect against an actual
autonomous ODE solution. Acceleration Lipschitz constant L and reference phase
speed bound B give amplification 1+abs(h)*max(1,L)+h²*L and defect at most
h²*(B+max(1,L)*B/2). Actual acceleration and stored-state discrepancies enter
separately. A finite geometric-sum envelope controls the numerical history;
actual reference-tube margins support simultaneous induction of error and
domain membership. Conditional residual/rounding bounds are used only after
membership of the current state is proved. Separate position and velocity
margins and the whole exact proposed affine phase segment are covered.

For h=T/(N+1), acceleration discrepancy O(h), stored-state discrepancy O(h²),
and initial error O(h), the checked finite envelope has an explicit exponential
O(h) bound and tends to zero. `UniformODECertificate.refinement_converges`
transfers this limit to actual histories uniformly at every grid point,
including both endpoints, while proving eventual numerical domain membership
from one fixed positive reference-tube margin. T=0 is included. Actual ODE
existence, usable numerical regional constants, IEEE certificates, impulsive or
time-dependent-control convergence and continuous interpolants remain separate.

The earlier fixed-pair zero-floor classification, AC metric-length bridge,
arbitrary-endpoint attainment, nearest-Shore classification, radial analysis,
mechanical contracts and #161--#163 results remain intact. This increment does
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

passed for **2,225 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 2225 project declarations
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

Independent review checked the actual inverse-frame derivative, separation of
base and fiber coordinates, gauge composition, C2-frame/C1-coefficient and
C3-chart/C1-coefficient regularity, and the actual fiber-valued curvature tensor.
The local constructions are not presented as manifold-bundle gluing.

The embedded review checked the actual Taylor and chord bounds, full-Shore
remote separation, compact injective-curve argument, strict versus non-strict
radii and derived strip injectivity. It checked the neighborhood requirement
on the local parameter patch, actual IFT construction, spanning planar frame,
nonfocal scale, ambient ruler derivative at Shore and exact scope of metric
pullback. The compactness constants are existential, not numerical certificates.

The shortcut review checked the Euclidean norm, use of the chart only as a
continuous linear equivalence, exact affine reparameterization, cusp-safe a.e.
density identity, path-infimum bound, physical dilation and L/epsilon endpoint
comparison. No converse, equality-case classification or hidden attainment
hypothesis was found.

Read-only provider review checked the concrete Horner derivative recurrence,
coherent coefficient shifts, both coordinate joins, closed-cell floor gluing,
C4 induction, physical normalization and actual derivative identities. Ideal
halo bounds and row-major addresses were compared with the unchanged Rust
source. The review distinguishes real global field correspondence from machine
index/cast/mask behavior, reference fidelity and IEEE error.

The dynamics review checked the velocity-then-position order, h² position
correction, actual derivative-based local defect, acceleration residual factor,
closed envelope and noncircular conditional-domain induction. The final theorem
uses actual histories and one fixed reference-tube margin to prove uniform grid
convergence and eventual membership. Horizon/scaling factors, endpoints and
T=0 were checked. The note retains the exact-solution and regional-bound premises
and excludes off-grid, hybrid-impulse and fixed-precision IEEE conclusions.

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
