# Verification record: Lean geometry and mathematical research

Verified 2026-11-24 (America/Chicago) in the supplied Linux environment. This continuation starts
from draft PR #164 at `9bdecaf264ea0374797087344664a726d5ef5c52`, whose
main-branch base is `55df760ff706210dcce3c88e79e476a45c5331f0`.
The earlier 172-, 291-, 400-, 596-, and 726-theorem increments are retained. This
record distinguishes new kernel-checked results, attributed analysis support,
project compatibility proofs, conventional geometry, and earlier optional-tool
and numerical observations.

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

The earlier increments restored the checkout, Elan 4.4.0, Lean 4.24.0, and
the pinned Mathlib cache in this workspace. This continuation reused that
installation and cache. One missing precompiled file for the pinned inverse
function theorem dependency was fetched through Mathlib's cache tool. The
committed toolchain, dependency sources, manifest,
Lake configuration and Lean workflow were not changed.

From `formal/`, the complete continuation command

```text
lake build --wfail
```

completed successfully with **2,643 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing forty-seven result modules | 673 |
| `NonlinearCoordinates` | 14 |
| `CurvatureTensoriality` | 7 |
| `FlatCurvatureTensor` | 24 |
| `TubularNeighborhood` | 9 |
| `EmbeddedShoreMetric` | 18 |
| `FlatShoreSecondVariation` | 24 |
| `OperationalField` | 22 |
| `OperationalFieldBounds` | 9 |
| `OperationalFieldJetError` | 16 |
| `OperationalFieldQueryError` | 11 |
| `AccumulatedDynamics` | 14 |
| `PhaseEnergyRegularity` | 3 |
| `DiscreteSteering` | 8 |
| Existing twelve analysis-support modules | 53 |
| `Analysis/CompactParametricIntegral` | 1 |
| **Total across seventy-three source modules** | **906** |

This adds **180 explicit declarations in thirteen result modules and one
support module**. The result subtotal is 852 across sixty modules; support
is 54 across thirteen modules. Counts include explicit supporting lemmas and are not counts of
independent research results. A comment/string-stripped source inventory
reproduced the preceding count of 726 before counting this increment. Every
result-module row in the [formal inventory](../../formal/README.md) is
reconciled against its source count. Attributed analysis infrastructure is
separated from the result subtotal and retains its provenance.

For #160, the nonlinear coordinate extension differentiates the actual
inverse-Jacobian pullback and includes the actual chart Hessian in the
connection transformation. It proves covariant-derivative, bracket and full
curvature covariance using neighborhood equality before differentiating the
inner covariant fields. A separate actual commutator expansion establishes
field-extension independence and identifies a bundled continuous trilinear
coordinate tensor. The flat connection is proved smooth, its complete tensor
is reconstructed, and the curvature ratio is checked in every regular
nondegenerate frame, including after nonlinear pullback. A C² chart suffices
for this local commutator theorem; it does not by itself make the transformed
coefficient field C¹. Manifold-level bundle and atlas interfaces remain open.

The embedded extension derives actual distance and unique nearest points from
a full-set normal-chord certificate, verifies a concrete sphere instance,
and proves normal-strip injectivity on an injective curve patch. Actual Frenet
derivatives, local C¹ regularity, a spanning planar linear equivalence and a
nonfocal factor allow the inverse function theorem to construct a genuine
local normal chart. Its inverse coordinate has absolute value equal to actual
distance throughout an ambient neighborhood. The proof differentiates the
actual ambient regularized distance, including at Shore, and identifies the
ambient graph metric pullback at the selected point for all tangent vectors.
Generic reach from smooth embedded data and ambient connection/curvature
transfer are still separate obligations.

The sine-family extension differentiates its actual normalized flat length
integral twice. It proves the actual curve derivative, fixed endpoints and
absolute continuity, identifies the integral with the existing metric's
curve length, and obtains the second derivative
ell/2*((pi/ell)²-1) at zero amplitude. When ell>pi, the actual derivative's
right slope limit and a strict monotonicity argument prove that every
sufficiently small positive amplitude has length less than ell. The actual
parameter interval is [0,pi]; no new wrapper into the existing [0,1]
path-infimum definition or all-minimizer classification is asserted.

For #162, the actual coefficient arrays and simultaneous Horner jet recurrence
of the experimental operational field now define a checked exact-real fixed
cell polynomial. Its returned physical-coordinate gradient and Hessian are
the actual Fréchet derivatives. Nonnegative Bernstein coefficients establish
the concrete weight positivity and stability. Raw quadratic samples give the
quadratic plus h²*(a+c)/2, with its exact differential and Hessian. Uniform
coefficient error eta gives value error at most eta, gradient-component error
4eta/(3h), pure-Hessian error 2eta/h², and mixed-Hessian error 16eta/(9h²).
Coefficient magnitudes also give actual same-cell differential and Hessian
operator bounds 8B/(3h) and 68B/(9h²), in the coordinate product maximum norm,
and corresponding real-query variation and combined value-error bounds.
These results do not identify stored coefficients with exact Shore samples,
prove global cell selection/join correspondence, or certify IEEE evaluation.

For #159, an actual residual is included in the phase-energy differential,
and the local changing-position defect is propagated through an actual finite
velocity/position/impulse recurrence. The exact impulse is evaluated at the
updated position. Explicit force/connection residual estimates and step-size
bounds give a summed energy/work budget; a strict total-displacement bound
controls every proposed position segment. Actual smooth fields on a compact
phase region yield a finite uniform phase-Hessian bound, with convexity giving
Lipschitz differential control and a constructor for local step certificates.
This establishes existence of constants, not their numerical values or
a priori phase-region persistence. A concrete endpoint-metric-normalized
covariant Euler steering candidate has checked unit norm and an actual
first-order consistency proof. These statements do not assert trajectory
convergence, exact parallel transport, or an executable IEEE guarantee.

The earlier complete fixed-pair zero-floor classification, AC metric-length
identity, arbitrary-endpoint attainment, all-nearest-Shore-minimizer
classification, radial analysis, potential/control catalog, and #161--#163
contracts remain included. This increment does not add convergence of
selected minimizing curves or a publication-level novelty claim.

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

This continuation adds one project-authored compact parametric-integral
helper. Joint continuity and actual parameter derivatives yield uniform
local domination on a compact rectangle, then the pinned differentiation
theorem applies to the actual interval integral. It supplies no new analytic
axiom and is separate from the attributed upstream backport.

## Transitive axiom audit and negative controls

```text
lake env lean Audit.lean
```

passed for **1,844 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 1844 project declarations
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

Independent review of this continuation checked the inverse-Jacobian Hessian
sign, actual neighborhood covariance, and the distinction between a C² chart's
commutator identity and the higher regularity of transformed coefficients.
The tensor review checked the actual derivative expansion, second-derivative
symmetry cancellation, full flat tensor reconstruction and positive arbitrary
frame Gram denominator.

The embedded review checked the full-set chord inequality, strict versus
non-strict normal radius, sphere-center endpoint, and strip injectivity. It
also checked that the inverse-function proof actually constructs the chart,
that the spanning planar frame is explicit, and that the ambient ruler
derivative is valid at Shore. The final wrapper gives a neighborhood distance
identity and a metric equality for all tangent vectors at its selected point;
it does not quantify the metric equality throughout that neighborhood without
corresponding Frenet data there.

Read-only provider review matched the six coefficient arrays, simultaneous
Horner recurrence, tensor-sum indexing, spacing factors and mixed-Hessian
orientation against the actual Rust source. The formal object models a fixed
real polynomial. The review distinguishes arbitrary stored coefficients from
raw samples used in the conditional quadratic reproduction theorem, and
coefficient perturbation from arithmetic error in the evaluator.

The dynamics review checked that residual power is derived from actual
Fréchet differentiation and that the finite recurrence uses the updated
configuration and proposed pre-impulse velocity in its impulse ledger. The
compact-region theorem bounds the actual phase Hessian under actual C² field
regularity; position control alone does not establish its velocity-region
hypotheses. The steering review verified the actual square-root normalization
derivative and its zero first-order correction from metric compatibility.
Endpoint norm preservation alone was not used as a substitute for consistency.

The second-variation review checked compact domination, actual twice-integrated
derivatives, all pi/ell scaling factors, the metric speed identity and the
fixed endpoints. Strict shortening follows from a negative actual first
derivative on a small positive amplitude interval, not from a weak local
maximum statement alone. The explicit sine curves also have checked AC.

The new mathematical notes were independently checked against actual
signatures. No blocking mathematical defect was found. Relative documentation
links, module counts and the staged whitespace check were also checked before
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
