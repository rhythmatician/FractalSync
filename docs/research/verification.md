# Verification record: Lean geometry and mathematical research

Verified 2026-11-24 in the supplied Linux environment. This continuation starts
from draft PR #164 at `39d8614c07cf41ac3ddec30adedbc147585c724e`, whose
main-branch base is `55df760ff706210dcce3c88e79e476a45c5331f0`.
The earlier 172- and 291-theorem increments are retained. This record
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
| Previously exercised optional Lean LSP/MCP | 0.33.0, commit `7932203d79db213852db215835bd79c943bb99b0` |

The earlier local checkout and Lean installation were absent in this fresh
workspace. Elan 4.4.0 and the repository's Lean 4.24.0 toolchain were installed
again, the draft branch was restored, and the unchanged baseline built
successfully with 2,582 Lake jobs. The Mathlib cache restored all 2,540 requested
files. An initial batch encountered HTTP 429 responses; a serial retry using
a temporary curl configuration completed the remaining downloads. The
committed toolchain, dependency source and manifest pins were not changed.

From `formal/`, the complete continuation command

```text
lake build --wfail
```

completed successfully with **2,593 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing seventeen result modules | 252 |
| Ten new result modules | 105 |
| Existing eight analysis-support modules | 39 |
| New AC composition/gluing support module | 4 |
| **Total across thirty-six source modules** | **400** |

The new result modules are `ComponentCrossing` (3), `PairwiseDistance` (13),
`PathOperations` (16), `DisplacementBounds` (13), `IntrinsicMetric` (7),
`IntrinsicTopology` (10), `MetricComparison` (6), `ACComponentPaths` (8),
`ZeroFloorLimit` (21), and `NormalizedMapDifferential` (8).
Counts include explicit supporting lemmas and are not counts of independent
research results. The analysis-support declarations are separated because
many are attributed upstream infrastructure. The four new `ACComposition`
declarations are original project compatibility proofs, not upstream copies.
The [formal inventory](../../formal/README.md) gives the complete module counts
and statement boundaries.

The requested different-component result is checked for the actual pairwise
AC path infimum. A continuous path joining different complement components
must cross Shore; the existing integrated crossing calibration therefore
passes uniformly through the infimum and proves divergence as the floor
decreases to zero.

The same continuation proves actual AC reversal and concatenation with exact
length identities, metric axioms, a logarithmic displacement lower bound,
the original normed-space topology, properness and completeness. Completeness
uses the intrinsic metric's own uniform structure; no equality of ambient
and intrinsic uniformities is asserted.

For nonempty closed Shore, fixed endpoints and real a, all pointwise
zero-floor branches are covered: zero on the diagonal; divergence across
components; divergence for distinct endpoints with at least one on Shore;
and convergence to the actual off-Shore zero-floor path infimum within one
component. The last theorem derives path existence from actual component
membership and derives zero-floor integrability from compact path clearance.
Its quantitative sandwich splits all regularized candidates at an actual
reference-length bound B, without assuming a minimizing curve.

For #159/#162 the new normalized Map theorem differentiates the actual window
query at fixed anchor and nonzero scale. The residual bound
L*abs(h)*norm(u)^2/2 follows from an actual segment Hessian operator bound,
with weaker Lipschitz-differential and anchored-variation versions also proved.
No ideal signed-distance or numerical-provider derivative guarantee is inferred.

The earlier AC Shore-distance minimum, actual ruler/scale Hessians, local
connection and continuous work-energy results remain included. Metric
curve-length identification, the length-space property, arbitrary-endpoint
minimizing-curve existence, the all-nearest-Shore-minimizer classification,
Riemannian curvature identification, cut locus, and the local curved-Shore
Lyapunov--Schmidt argument remain outside the checked boundary.

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

The new `ACComposition` module is separate project work. It checks the
disjoint-interval condition under monotone/antitone maps, preserves AC under
Lipschitz precomposition with the stated interval-image premise, and glues
matching AC pieces by constant extension. These four declarations use the
actual pinned Mathlib AC definition and enter the same transitive audit.

## Transitive axiom audit and negative controls

```text
lake env lean Audit.lean
```

passed for **741 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 741 project declarations
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

## Previously checked optional MCP integration

The earlier increment ran `formal/tools/check_mcp.py` in a real MCP stdio session using the separately
installed pinned server. It discovered **22 tools**, obtained clean diagnostics
for `DistanceToSet.lean`, and used `lean_verify` to inspect
`FractalSync.DistanceGraph.exists_nearest_segment`. The result reported only
standard axioms. The check exited successfully.

The optional server was not reinstalled and its smoke check was not rerun for
this continuation.

The server emitted an ignored Python asyncio subprocess-finalizer warning,
`RuntimeError: Event loop is closed`, during shutdown. Diagnostics and theorem
verification had succeeded; clean server process teardown is a remaining
upstream integration issue, not a Lean proof failure. The optional server is
not a proof-CI dependency or a trusted axiom source. The checked-in VS Code
configuration is a template, not a claim that every agent client has registered
this MCP server. Official Lean agent skills were consulted at the documented
source pin; this increment did not install them in a personal skill catalog.

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
AC infimum, Hessian and mechanics signatures, and the conventional
nearest-Shore minimizer classification. That classification remains a
conventional equality argument, not a Lean theorem.

This continuation independently reviewed the actual zero-floor path class,
clearance and integrability proof, quantitative factor and signs, all
candidate-length cases, the one-sided limit filter, positive definiteness,
and the topology/uniformity distinction. No mathematical gap was found in
the reviewed modules. The complete fixed-pair classification was checked
case by case, including the diagonal and two distinct Shore endpoints.
The notes explicitly exclude empty-Shore and empty-admissible-class
interpretations from the zero-floor theorem.

The normalized Map note was checked against the actual `fderiv` signatures.
It differentiates only the window coordinate at fixed anchor and scale, and
retains actual ambient derivative hypotheses on the entire physical segment.
The general minimizing-curve existence argument remains outside the formal
boundary. Its use in the earlier conventional same-component proof has been
replaced by the checked split over all candidates.

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
run for this research-only increment.
