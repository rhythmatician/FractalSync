# Verification record: Lean geometry and mathematical research

Verified 2026-11-24 (America/Chicago) in the supplied Linux environment. This continuation starts
from draft PR #164 at `ac4ab10ca4049f071dd84565fe7f5686696808ee`, whose
main-branch base is `55df760ff706210dcce3c88e79e476a45c5331f0`.
The earlier 172-, 291-, 400-, and 596-theorem increments are retained. This
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
installation and cache. The committed toolchain, dependency sources, manifest,
Lake configuration and Lean workflow were not changed.

From `formal/`, the complete continuation command

```text
lake build --wfail
```

completed successfully with **2,627 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing thirty-nine result modules | 543 |
| `FlatShoreIntrinsic` | 15 |
| `FlatShoreGeodesics` | 25 |
| `CoordinateCovariance` | 15 |
| `CurvedShoreGeometry` | 19 |
| `CurvedShoreCurvature` | 5 |
| `ProviderCertificates` | 25 |
| `MovingMetricMechanics` | 18 |
| `CovariantSteering` | 8 |
| Existing twelve analysis-support modules | 53 |
| **Total across fifty-nine source modules** | **726** |

This adds **130 explicit declarations in eight result modules**. The result
subtotal is 673 across forty-seven modules; support remains 53 across twelve
modules. Counts include explicit supporting lemmas and are not counts of
independent research results. A comment/string-stripped source inventory
reproduced the preceding count of 596 before counting this increment. Every
result-module row in the [formal inventory](../../formal/README.md) is
reconciled against its source count. Attributed analysis infrastructure is
separated from the result subtotal and retains its provenance.

For #160, the Euclidean flat-Shore bridge uses `WithLp 2 (ℝ × ℝ)` and the
actual distance to `{r=0}`. Differentiating the squared trace identity avoids
assuming differentiability of absolute value at a Shore crossing. It proves
the graph and coordinate densities agree almost everywhere for the full AC
class, identifies the actual coordinate integral and complete candidate
length sets, and transfers the path infimum, metric variation and attained
AC minimizers. Both directions of the global coordinate chart preserve AC;
the default coordinate product norm is not treated as Euclidean.

The smooth flat extension assumes actual first and second curve derivative
data satisfying the coordinate geodesic ODE. It differentiates the energy
and momentum, proves interval conservation and the radial first integral,
and verifies the unit-speed Shore solution with its parallel normal. An
actual iterated along-curve derivative and curvature commutator give the
normal Jacobi residual. The sine field has a checked first positive field
zero at pi. This does not establish smoothness of all AC minimizers, bundled
conjugacy, an index-form theorem or the global cut locus.

The affine-coordinate extension recomputes actual transported field
derivatives, the connection, Mathlib's Lie bracket and the full curvature
commutator. It proves their affine covariance and metric Gram-ratio
invariance. Its flat specialization discharges both intermediate-field
derivative hypotheses and proves a positive transformed Gram denominator.
The general theorem states the sufficient global differentiability of the
third field explicitly. Nonlinear charts, variable-frame naturality,
arbitrary-extension tensoriality and bundled manifold objects remain separate.

The curved normal-chart metric now has checked pointwise metric algebra,
actual compatibility, connection laws and its full off-Shore curvature.
The commutator calculation differentiates both intermediate fields and
cancels the curvature-profile derivative terms. It recovers curvature +1
at Shore and proves the actual normal derivative of the commutator-defined
curvature is -(1+a²)*k(y), using eventual equality in a nonfocal neighborhood.
These theorems take the normal-coordinate metric as supplied; construction
of a tubular chart for an embedded Shore and the functional-analytic
bifurcation remain outside the formal boundary.

For #162, actual first and second Fréchet derivatives and anchored Hessian
variation imply a third-order Taylor remainder. This gives central-gradient,
diagonal-Hessian and four-corner mixed-Hessian truncation bounds, with actual
Hessian symmetry discharged under a neighborhood hypothesis. The module
then composes signed or uniform sampled-value errors, query-coordinate
errors, explicit arithmetic and denominator errors, and Euclidean/Frobenius
component aggregation. The O(h) Hessian bound uses the stated Hessian
variation assumption; it does not silently assume fourth derivatives for
an O(h²) bound. Actual provider data and IEEE arithmetic certificates are
still premises to be supplied, not inferred from runtime flags.

For #159, the changing-configuration step uses the executable ordering
v1=v+dt*A, x1=x+dt*v1. The actual phase-energy differential, connection/potential
cancellation and a segment Taylor bound give a local dt² energy-defect
certificate. Its sufficient Hessian assumption is on actual phase energy,
which may require higher field regularity than a ruler Hessian alone.
An explicit coefficient estimate, the impulse ledger at the new position,
and safety of the entire proposed affine position segment are checked.
Moving-metric steering now has an actual covariant pairing derivative and
interval metric-norm preservation for solutions of the stated covariant ODE.
These results do not supply ODE existence, a discrete norm-preserving
steering algorithm, or a global floating-point rollout guarantee.

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

The preceding increment added the vector-valued AC FTC by dual separation, a
derivative estimate from actual punctured slope limits, and a proper-metric
geodesic construction using compact endpoint-path families. The latter
proves its compactness and reparameterization arguments directly; it does
not assume a minimizing curve or import an extra geometric axiom.

## Transitive axiom audit and negative controls

```text
lake env lean Audit.lean
```

passed for **1,375 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 1375 project declarations
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
installation and stdio test. This continuation confirmed the configuration
mount is still read-only; it did not repeat the server smoke test or attempt
to bypass that mount.

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

Independent review of this continuation checked the flat correspondence's
Euclidean norm, bidirectional AC chart transfer, cusp-safe trace identity,
complete candidate-set equality and noncircular use of existing attainment.
The smooth geodesic and Jacobi review checked the actual solution derivatives,
epsilon scaling, curvature sign, constant-extension convention, first positive
field zero, and the explicit boundary before bundled conjugacy or minimality.

The affine covariance review checked the A inverse / derivative / A chain
rule, the transported bracket and full commutator, the actual intermediate
regularity hypotheses, and the positive-Gram flat specialization. The curved
normal-chart review checked every Christoffel coefficient, the actual metric
compatibility, curvature-profile derivative cancellation, the nonfocal
denominators and the transfer of the normal derivative to actual curvature.

The provider review checked Taylor's 1/6 coefficient, the diagonal and mixed
stencil constants, the Hessian symmetry premise, correlated signed errors,
Euclidean component norms, positive denominators and the distinction between
real-arithmetic error premises and IEEE certificates. Read-only runtime
inspection confirmed the clamped stencil step, sample conversions and prior
center sample used in diagonal Hessians.

The mechanics review checked the actual phase differential, first-order
work cancellation, the phase max norm, phase-energy Hessian hypotheses,
new-position impulse and whole-segment domain certificate. The moving-metric
steering review checked the metric-skew action and interval derivative
argument, including the two-plane scope of the quarter-turn interpretation.
The proposed position interpolation is not asserted to enclose the exact ODE
trajectory. A uniform O(dt²) claim requires uniformly bounded coefficients
as the step changes.

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
