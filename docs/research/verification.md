# Verification record: Lean geometry and mathematical research

Verified 2026-11-24 (America/Chicago) in the supplied Linux environment. This continuation starts
from draft PR #164 at `d16288d2e223fc2b3efefbedf3e43510d30cb1b5`, whose
main-branch base is `55df760ff706210dcce3c88e79e476a45c5331f0`.
The earlier 172-, 291-, and 400-theorem increments are retained. This record
distinguishes new kernel-checked results, attributed analysis support, project
compatibility proofs, conventional geometry, and earlier optional-tool/numerical
observations.

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
| Installed and exercised optional Lean LSP/MCP | 0.33.0, commit `7932203d79db213852db215835bd79c943bb99b0` |

The preceding 400-theorem increment restored the checkout, Elan 4.4.0,
Lean 4.24.0, and the pinned Mathlib cache in this workspace. This continuation
reused that installation and fetched the small additional cache set needed
for strict-convexity geometry. The committed toolchain, dependency sources,
manifest, Lake configuration and Lean workflow were not changed.

From `formal/`, the complete continuation command

```text
lake build --wfail
```

completed successfully with **2,619 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing twenty-seven result modules, before this continuation | 357 |
| New ruler-distance lemma in `IntrinsicTopology` | 1 |
| Twelve new result modules | 185 |
| Existing nine analysis-support modules | 43 |
| Three new analysis-support modules | 10 |
| **Total across fifty-one source modules** | **596** |

The new result modules are `ACSubpaths` (8), `IntrinsicLength` (6),
`GraphChord` (10), `IntrinsicVariation` (20), `IntrinsicGeodesics` (5),
`IntrinsicCurveLength` (8), `ShoreMinimizers` (17), `FlatShoreGeometry` (25),
`PotentialForces` (29), `ControlMechanics` (28), `DiscreteMechanics` (11),
and `NormalizedProviderError` (18). The result subtotal is 543 across
thirty-nine modules; support contributes 53 across twelve modules.
Counts include explicit supporting lemmas and are not counts of independent
research results. The analysis-support declarations are separated because
many are attributed upstream infrastructure. The new `ACVectorFTC` (2),
`DerivativeControl` (1), and `ProperGeodesics` (7) declarations are original
project support proofs, not upstream copies.
The [formal inventory](../../formal/README.md) gives the complete module counts
and statement boundaries.

For #160, the actual AC graph integral now equals Mathlib's actual intrinsic
metric variation. The proof uses AC subpaths, cumulative graph length,
the actual AC variation trace and a lifted-chord derivative estimate.
The length-space property is stated as an explicit near-optimal-path witness,
not a new Mathlib class instance. Proper compactness then produces an exact
constant-speed metric geodesic, an explicit ordinary Lipschitz estimate makes
it AC in the original norm, and its graph length attains the original path
infimum. These positive-floor theorems hold in finite-dimensional real normed
spaces for arbitrary S and real a; arbitrary S uses Mathlib's real `infDist`
convention, including zero for the empty set.

Every nearest-Shore minimizer is now classified in a strictly convex
finite-dimensional normed space. The complete iff includes a constant
zero-distance branch and all monotone AC traversals of nearest segments,
including pauses. A checked reversal wrapper covers paths toward Shore.
The equality-case theorem requires an actual endpoint in S; the existing
nonempty-closed-set infimum theorem supplies its physical minimum interpretation.

The planar flat metric's coordinate curvature is identified using an actual
`fderiv` covariant derivative, the Mathlib Lie bracket, connection laws,
actual metric compatibility, the full curvature commutator and its metric
Gram determinant. It gives the previous rational curvature formula and +1
on flat Shore. It does not construct a bundled Riemannian tensor or prove
coordinate/frame invariance, nor formally identify this coordinate metric's
distance with the flat-Shore specialization of the general path infimum.

For #159, the explicit Shore and secant-wall potentials now have actual
Fréchet derivatives, exact raised forces and wall divergence. The full
signed continuous energy ledger gives a necessary crest budget retaining
wall change and friction. Momentum differentiation, drive, drag, candidate
board friction/carve and planar fixed-metric steering are checked.
Frozen-metric discrete energy identities give the exact explicit-drag factor
(1-beta*dt)^2 and sharp nonincrease condition v=0 or 0<=beta*dt<=2 for rho>0.

For #162, combined provider/recomputed-ruler bounds now use certificates at
the actual computed query. Center/scale stability, endpoint and relative
budgets, rationalized ruler error and quadratic crest-bias bounds are checked.
The numerical stencil clamps its scale, so the specialized proportional-scale
theorem does not directly certify that stencil; the general h/k theorem
requires actual scale and query-error data. No provider certificate is inferred.

The earlier complete fixed-pair zero-floor classification, radial analysis,
AC calibration, ruler/scale Hessians and metric foundations remain included.
The flat cut locus, curved-Shore curvature and Lyapunov--Schmidt argument,
numerical derivative/floating-point correspondence, and full changing-metric
integrator properties remain outside the checked boundary.

## Attributed analysis backport and new compatibility support

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

The new support adds the vector-valued AC FTC by dual separation, a
derivative estimate from actual punctured slope limits, and a proper-metric
geodesic construction using compact endpoint-path families. The latter
proves its compactness and reparameterization arguments directly; it does
not assume a minimizing curve or import an extra geometric axiom.

## Transitive axiom audit and negative controls

```text
lake env lean Audit.lean
```

passed for **1,094 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 1094 project declarations
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

This continuation installed the pinned 0.33.0 server in a separate Python
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
this MCP server. The new Codex TOML template parses successfully. An actual
`codex mcp add` attempt failed because the active client configuration is on
a read-only filesystem; no server entry was registered. This is the concrete
remaining client-registration blocker, distinct from the successful server
installation and stdio test.

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
AC infimum, field calculus and complete fixed-pair zero-floor classification.
Those results remain unchanged. The formerly conventional nearest-Shore
minimizer classification is now a checked iff theorem with its zero branch
and converse included.

Independent review of this continuation checked the length proof's dependency
order, actual metric instance, finite variation, lifted derivative limit,
compact path families and ordinary AC admissibility. It found no circular
assumption of attainment. The equality-case proof was reviewed against
strict convexity, actual trace derivatives and segment traversal with pauses.

The flat coordinate curvature review independently checked the Christoffel
signs, actual field derivatives, Lie-bracket convention and Gram determinant.
It confirmed the positive-one Shore convention and identified the still
separate Euclidean path-metric specialization and tensor/frame interfaces.

The mechanics review checked actual potential derivatives, signs and wall
coefficients, the distinction between covector and raised acceleration,
PSD and integrability hypotheses, and the exact explicit-drag interval.
The provider review checked the computed query location, signed-scale
denominator and zero-scale totalization. One overstatement in the conventional
note was corrected: a correlated increment certificate can be tighter than
an endpoint-error sum, but is not necessarily strictly tighter.

Read-only runtime inspection confirmed that the numerical stencil clamps
its scale, the integration step changes configuration before reporting new
energy, and the controls wrapper applies its impulse at the new position.
The notes preserve those distinctions. They do not certify the runtime's
broader energy-drift comments, sampled derivatives or floating evaluation.

The new mathematical notes were independently checked against actual
signatures. The final prose states the unit parameter interval, the real
empty-set infDist convention, the planar steering scope and the positive
floor, ordered interval and valid-domain hypotheses needed by the relevant
contracts. No blocking mathematical issue was found in the reviewed modules.

The earlier prior-art search was targeted. It found exact prior art for the requested
phase-harmonic Lipschitz bound and strong prior art for the location-scale
hyperbolic structure and Lipschitz extension envelopes. It did not establish
publication-level novelty of this package. Issues #160--#163 remain research
issues; successful Lean verification is not a novelty certificate.

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
