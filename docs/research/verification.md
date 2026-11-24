# Verification record: Lean geometry and mathematical research

Verified 2026-11-24 in the supplied Linux environment. The analytic extension
builds on draft PR #164, commit `a1537bb28ad06c0a66e1d8d4fffa6e581cf405cd`,
whose main-branch base is `55df760ff706210dcce3c88e79e476a45c5331f0`.
This record separates kernel-checked statements, conventional proofs, optional
authoring tools, and the numerical evidence from the initial increment.

## Environment and full Lean build

| Component | Observed version or pin |
| --- | --- |
| Elan | 4.4.0 |
| Lean | 4.24.0, commit `797c613eb9b6d4ec95db23e3e00af9ac6657f24b` |
| Lake | 5.0.0-src+797c613 |
| Mathlib | `f897ebcf72cd16f89ab4577d0c826cd14afaafc7` |
| Transitive dependencies | Committed `formal/lake-manifest.json`, unchanged |
| Optional Lean LSP/MCP | 0.33.0, commit `7932203d79db213852db215835bd79c943bb99b0` |

Elan, the pinned Lean toolchain and the optional MCP server were installed
locally. Mathlib's precompiled cache was used. Some downloads returned HTTP 429;
retrying the missing files with curl concurrency one and a five-second retry
delay succeeded. No Mathlib source modification was required.

From `formal/`, the full command

```text
lake build --wfail
```

completed successfully with **2,539 Lake jobs**, mostly cached dependencies,
and no warnings or errors. All eleven accepted proof modules are imported by
`FractalSync.lean`. They contain **172 explicit theorem/lemma declarations**,
including helper lemmas: 144 more than the initial branch's 28. This count is
not a count of independent research results.

| Module | Theorem/lemma declarations |
| --- | ---: |
| Ruler | 9 |
| ScaleMetric | 31 |
| MetricImpulse | 14 |
| DistanceGraph | 4 |
| DistanceToSet | 6 |
| RadialProfile | 44 |
| PathCalibration | 8 |
| FlatShore | 32 |
| SensitivityBounds | 4 |
| NormalizedDistanceFields | 8 |
| CovariantStatistics | 12 |
| **Total** | **172** |

The [theorem inventory](../../formal/README.md) records the hypotheses and
scope. In particular, the radial profile is an actual interval integral, its
closed primitive and sharp additive constant are checked, and nearest-point
segments use Mathlib's actual distance to a set. The unrestricted absolutely
continuous path-infimum theorem, properness, the full metric singular limit,
Riemannian curvature identification, cut locus and local curved-Shore
Lyapunov--Schmidt argument remain conventional.

## Transitive axiom audit and negative control

```text
lake env lean Audit.lean
```

passed for **310 project declarations**, including definitions and generated
support declarations. Its output was:

```text
Transitive axioms: [propext, Classical.choice, Quot.sound]
Axiom audit passed for 310 FractalSync declarations
```

The audit now runs Lean's own `CollectAxioms.collect` traversal over all project
roots with one shared visited set. This computes the same union of transitive
axioms as the previous separate calls while avoiding repeated traversal of the
large shared Mathlib graph. It still reads the checked kernel environment and
still rejects every axiom outside the three-item allowlist. It also rejects an
empty namespace.

A new negative control placed an unapproved axiom **outside** the project
namespace, then used it to prove `FractalSync.auditProbe` in a temporary audit
copy. The check rejected the indirect dependency with exit code 1:

```text
Unapproved axiom auditExternalPoison reachable from FractalSync declarations
```

The temporary file was removed. The accepted proof files contain no `sorry`,
`admit`, project axiom or native-evaluation axiom. This is a transitive audit,
not merely a source-text scan.

## Optional MCP integration

`formal/tools/check_mcp.py` ran a real MCP stdio session using the separately
installed pinned server. It discovered **22 tools**, obtained clean diagnostics
for `DistanceToSet.lean`, and used `lean_verify` to inspect
`FractalSync.DistanceGraph.exists_nearest_segment`. The result reported only
standard axioms. The check exited successfully.

The server emitted an ignored Python asyncio subprocess-finalizer warning,
`RuntimeError: Event loop is closed`, during shutdown. Diagnostics and theorem
verification had succeeded; clean server process teardown is a remaining
upstream integration issue, not a Lean proof failure. The optional server is
not a proof-CI dependency or a trusted axiom source. The checked-in VS Code
configuration is a template, not a claim that every agent client has registered
this MCP server. Official Lean agent skills were consulted at the documented
source pin; this increment did not install them in a personal skill catalog.

## Independent numerical checks

The initial 28-theorem increment ran each script using its repository Python environment, Python 3.12.14. These unchanged numerical scripts were not rerun for the analytic extension recorded above.
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

The prior-art search was targeted. It found exact prior art for the requested
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
