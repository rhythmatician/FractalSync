# Verification record: Lean geometry and mathematical research

Verified 2026-11-24 in the supplied Linux environment. This continuation starts
from draft PR #164 at `8feccc3c69565ea608fdd735b0af2df66448e0a5`, whose
main-branch base is `55df760ff706210dcce3c88e79e476a45c5331f0`.
The earlier 172-theorem increment is retained. This record distinguishes new
kernel-checked results, attributed analysis support, conventional geometry,
and the earlier optional-tool/numerical observations.

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

The existing local Elan/Lean installation and repository checkout were still
available. Their versions matched the published pins and were revalidated.
The unchanged baseline built successfully with 2,539 Lake jobs and passed its
310-declaration audit. The continuation fetched 24 missing Mathlib cache files
for the new imports; the dependency source and toolchain pins were not changed.

From `formal/`, the complete continuation command

```text
lake build --wfail
```

completed successfully with **2,582 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All result and support modules are reachable from
`FractalSync.lean`.

| Declaration inventory | Explicit theorem/lemma declarations |
| --- | ---: |
| Existing eleven result modules | 172 |
| Six new result modules | 80 |
| Eight new analysis-support modules | 39 |
| **Total across twenty-five source modules** | **291** |

The new result modules are `RulerDifferential` (20), `ContinuousMechanics` (15),
`RadialLipschitz` (8), `AbsolutelyContinuousPaths` (14),
`IntrinsicShoreDistance` (18), and `CrossingAction` (5).
Counts include explicit supporting lemmas and are not counts of independent
research results. The 39 analysis-support declarations are separated because
most are attributed upstream infrastructure, not project-specific discoveries.
The [formal inventory](../../formal/README.md) gives the complete module counts
and statement boundaries.

The continuation checks unrestricted AC distance/ruler traces, almost-everywhere
calculus, velocity and graph-speed integrability, endpoint and crossing
calibration, and actual attained Shore-path infima in both orientations. The
original ruler-derivative length functional is connected to the expanded
integrand by an actual a.e. chain rule. Zero-floor divergence is now proved for
that path infimum. Duration/action bounds retain their explicit speed-cap or
square-integrability hypotheses.

For #159 the full field Hessians differentiate actual `fderiv` fields. The
connection satisfies the metric differential, symmetry/Koszul identity and
positive-metric uniqueness, and continuous energy balance follows from actual
field/trajectory derivatives and the dynamics. No global smoothness, ODE
existence or discrete conservation is inferred.

The general pairwise metric, properness, full componentwise singular limit,
all-minimizer classification, Riemannian curvature identification, cut locus,
and local curved-Shore Lyapunov--Schmidt argument remain conventional.

## Attributed analysis backport

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

## Transitive axiom audit and negative controls

```text
lake env lean Audit.lean
```

passed for **581 project declarations**, including definitions, generated
helpers, and declarations in adopted upstream namespaces. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 581 project declarations
```

The audit now chooses roots by defining-module provenance as well as the
`FractalSync` namespace. Thus backported declarations in namespaces such as
`Finset` and `AbsolutelyContinuousOnInterval`, and private/generated helpers,
are included even when no named project theorem directly uses them. The
three-axiom allowlist and Lean checked-environment transitive traversal are
unchanged. An empty project root set is rejected.

A new negative control temporarily imported `FractalSync.AuditScopeProbe`,
whose axiom and dependent theorem were both in the unrelated namespace
`ExternalAuditScope`. No `FractalSync`-named declaration depended on them.
The provenance-aware audit rejected the imported axiom with exit code 1:

```text
Unapproved axiom ExternalAuditScope.unapproved reachable from FractalSync declarations
```

The probe source, compiled module and temporary audit file were removed.
The previous increment also tested an unapproved outside-namespace axiom
used indirectly by a project theorem; that transitive-dependency control was
rejected. The accepted package contains no `sorry`, `admit`, new axiom or
native-evaluation axiom. These are checks of the kernel environment rather
than only source-text scans.

## Previously checked optional MCP integration

The earlier increment ran `formal/tools/check_mcp.py` in a real MCP stdio session using the separately
installed pinned server. It discovered **22 tools**, obtained clean diagnostics
for `DistanceToSet.lean`, and used `lean_verify` to inspect
`FractalSync.DistanceGraph.exists_nearest_segment`. The result reported only
standard axioms. The check exited successfully.

The optional server smoke check was not rerun for this continuation.

The server emitted an ignored Python asyncio subprocess-finalizer warning,
`RuntimeError: Event loop is closed`, during shutdown. Diagnostics and theorem
verification had succeeded; clean server process teardown is a remaining
upstream integration issue, not a Lean proof failure. The optional server is
not a proof-CI dependency or a trusted axiom source. The checked-in VS Code
configuration is a template, not a claim that every agent client has registered
this MCP server. Official Lean agent skills were consulted at the documented
source pin; this increment did not install them in a personal skill catalog.

## Independent numerical checks

The initial 28-theorem increment ran each script using its repository Python environment, Python 3.12.14. These unchanged numerical scripts were not rerun for either analytic extension.
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

The radial formula, its sharp additive constant, and the quantitative
same-component limit were independently reviewed during this work. The review
prompted explicit treatment of the minimizing curve's metric/functional lengths
and of kinetic length density at nonsmooth points. The final note includes both.
The original four notes give assumptions, conventional proofs, limitations, and prior-art references. The new radial, metric, and flat-Shore notes add precise formal boundaries and independently reviewed conventional arguments.

The new AC, infimum, Hessian and mechanics notes were independently reviewed
against the actual Lean signatures. No formula or scope errors were found.
The conventional Euclidean minimizer classification was separately checked
by its equality argument; it is not counted as a Lean theorem.

The earlier prior-art search was targeted. It found exact prior art for the requested
phase-harmonic Lipschitz bound and strong prior art for the location-scale
hyperbolic structure and Lipschitz extension envelopes. It did not establish
publication-level novelty of this package. Issues #160--#163 remain research
issues; successful Lean verification is not a novelty certificate.

## CI scope

`lean-proofs.yml` builds the pinned formal package and runs the axiom audit for
formal-package changes. Its configuration was checked locally against the
pinned official Lean action's inputs. The commands above were executed locally;
this record does not claim an observed successful GitHub-hosted CI run.
No runtime or training code was changed, so the application test suite was not
run for this research-only increment.
