# 0005 — Separate reference geometry from bounded operational geometry

Status: Accepted for a bounded experimental prototype (2026-10-28, authorized by
repository owner). Shipping adoption, reference-cut motion policy and device
latency targets remain pending. ADR 0004 and the current shipping acceptance
contract remain in force. Related: #120, #145, PR #148.

FractalSync needs a coherent differential field that Physics can evaluate within
a live frame budget. The authorized experiment separates that operational field
from the Mandelbrot reference evidence used to construct and assess it. This
authorizes a bounded prototype, not shipping adoption of the changed geometry.

## Why change the contract

The experiments recorded in [the refinement report](../validation/geometry-145-refinement.md)
have produced useful derivative, integration and certification results. They
also expose cold queries lasting minutes and neighboring queries lasting
hundreds of milliseconds. Faster exact cache hits do not bound a new trajectory's
cost. More isolated proofs are not a sufficient implementation strategy for the
live visualizer.

ADR 0004 already separates Map roles and permits a coherent spline
representation. Its singularity rule still forbids silently replacing genuine
distance singularities with smooth Physics. This proposal explicitly revisits
that rule and the identity of the field supplying the metric. It preserves
ADR 0001, the Player's Controls, and the scale-relative metric formula.

## The proposed distinction

| Object | Meaning | Authority and validity |
| --- | --- | --- |
| Reference geometry | Evidence about the actual Map, Shore distance, realm and nearest-feature ambiguity | A bound or classification applies only to its stated region, scale and assumptions. Unknown stays unknown. |
| Operational field `F` | A versioned, scale-aware scalar field defining Physics | Its value, gradient and Hessian are analytic derivatives of one fixed representation. They are not reported as exact derivatives of the true Shore distance. |
| Prepared coverage | The region and scale envelope where the operational representation is available and meets its declared quality contract | Coverage is checked before stepping; a lookup cannot initiate a proof or an unbounded refinement. |

The Map, Inside, Outside and The Shore retain their meanings in the domain
contract. `F = 0` is the operational zero contour, not a redefinition of the
actual Shore. In particular, the sign of `F` alone does not certify realm inside
its reference-error band.

Physics uses

```text
rho_phys = sqrt(F^2 + epsilon^2)
sigma_phys = log2(d_ref / rho_phys)
G = rho_phys^-2 I + lambda^2 grad(sigma_phys) grad(sigma_phys)^T
```

All first and second derivatives, blend weights, metric terms and connection
terms follow from this same field. Rust owns construction semantics, evaluation
and validity; Python training and WASM consume the same versioned behavior.
This does not authorize smoothing only the Hessian or independently fitting
answers around each query.

## Fidelity and differential validity are separate obligations

A bound on `|F - D_reference|` does not bound the Hessian error, connection error
or trajectory error. The operational representation must independently pass
neighboring derivative checks, patch/refinement join checks, metric checks and
the existing energy and trajectory regressions. We do not require a Hessian of
the true distance where that Hessian does not exist.

Reference evidence must state whether a claim is certified, measured or unknown.
A point certificate cannot silently become a regional certificate. A measured
comparison is not a mathematical bound. Prepared assets retain their evidence,
error units, supported scale, construction parameters and version identity.

For the first prototype, retain the current scalar-distance error budget where
reference bounds are available. Keep the original derivative tolerances, energy
limits and both crossing test bodies intact. Any change to the claims those
checks establish belongs in an explicit acceptance-contract revision.

## Singularities require an explicit policy change

The proposed operational profile may be smooth at a reference cut locus. A
reference cut-locus flag remains visible and never becomes a claimed unique
Map normal. Operational Physics validity instead describes whether `F` supplies
a valid differential sample under the selected profile.

Allowing motion there would amend ADR 0004's current fail-closed policy. The
existing reference singularity and Physics refusal tests remain authoritative
until that amendment is accepted. They must not be silently weakened while
implementing this proposal. A prototype can be evaluated separately without
changing the shipping profile.

## Bounded evaluation and immutable field identity

The experiment loads the complete prepared asset before replay and freezes its
coordinate/scale envelope, preparation parameters and deterministic initial-state
perturbations. No coefficient replacement or coverage extension is allowed during
a rollout. A miss ends that attempt as `GeometryUnavailable`; it is not simulated
progress. A larger asset needs a new manifest and a fresh replay from the start.
Preparation may perform expensive reference calculations outside Physics. No
streaming, immutable-extension or recovery infrastructure is required here.

The rollout carries two immutable manifest identities:

- `field_id` identifies coefficients, coordinate mapping, patch structure,
  reconstruction rules and evaluator semantics.
- `physics_profile_id` identifies regularization and metric parameters, potential
  and force rules, reference-refusal policy and evidence snapshot, and numerical
  integration/work-accounting rules. Fixed timestep and drag parameters are part
  of this experiment's profile; explicit Controls remain logged inputs.

Cold prepared queries, cache eviction and exact repeats must evaluate the same
field. Cache state, query order, wall-clock deadlines and newly arrived evidence
must not choose a different local scalar function or Physics profile during a
rollout. Pin both identities across every stage and energy measurement. Within
one build, cache eviction and query reordering preserve jets and validity.
Native/browser comparisons use declared numerical parity tolerances, not an
implicit bitwise-equality requirement.

Every step is a transaction. Evaluate intermediate states into temporary storage,
check every complete differential sample, check the final weighted endpoint
separately from the fourth RK stage, and check endpoint metric/energy before
committing. Any refusal preserves position, velocity, mechanical time, integrated
work and all other mechanical state. Audio time is separate. Scalar availability
alone is insufficient when any derivative or differentiated blend support is
missing.

An unprepared query returns `GeometryUnavailable` in bounded work. It does not
fall back to a different metric, fetch data, fit a new field or launch a proof.
A refused step commits no partial movement. The prototype reports the explicit
rollout outcome and does not design frame-hold or recovery behavior. Product
response to coverage loss remains unresolved. A sampled training corpus does
not prove the Player will avoid unsupported geometry.

Proposed benchmark targets, to be ratified on named native and browser devices:

- Prepared geometry query and unavailable lookup: p99 at most 100 microseconds,
  with first-query and maximum latency reported separately.
- Complete Physics step, including every RK stage and geometry lookup: p99 at
  most 2 ms. Report work counts and maximum latency as well as percentiles.
- No proof construction, file/network access or unbounded allocation on that
  path. Deterministic work limits govern availability; timing measurements do
  not alter the mathematical answer.

Preparation time, asset size, resident memory and supported coverage are
separate gates. These proposed targets do not claim hard OS scheduling bounds
and do not count asset loading or preparation as a cheap cold query.

## A bounded next step

Acceptance is executable before constructing the corridor asset. An analytic
`F(x,y)=x` control checks regularization through the crest, metric, connection and
integration using the same downstream Rust code. Differential self-consistency,
including joins, is mandatory; reference differential fidelity is a separate
certified, measured or unknown claim. Energy tests do not establish reference
fidelity.

Keep existing crossing tests intact and report two verdicts. An operational
crossing concerns `F=0`. A reference Shore crossing needs ordered states on the
continuous reconstructed trajectory with conclusive opposite-realm evidence.
Only certified point/region-scoped distance bounds may convert signs outside
their error bands into reference realm conclusions. Measured discrepancies
cannot do this. Missing evidence yields `reference_crossing: unverified`.

Current-policy replay must honor authoritative reference-singularity refusals
even where the operational field is smooth. The experiment does not authorize
motion through a known reference cut. Unknown reference singularity information
must remain explicit in the report.

Build one experimental Rust operational representation over the original
crossing corridors, their Shore-search queries and a halo for all RK stages,
plus analytic and singularity controls. Freeze the reference evidence and field
coefficients before replay. Demonstrate coherent derivatives, unchanged crossing
and energy regressions, deterministic misses and latency on that finite domain.

That experiment answers whether the separation helps before expanding to a
general atlas. It does not establish arbitrary-depth Mandelbrot coverage. The
advertised coordinate/scale envelope, reference-fidelity claims, cut-locus
motion policy and response to missing coverage must be settled before adoption.

PR #148 remains unready. Preserve reusable Rust math and the minimal reproducible
failure evidence; do not merge the accumulated experimental archive as though it
were an accepted runtime implementation. Adoption requires a small reviewed
change, compatibility/version updates, regenerated goldens, training and browser
parity, and the finalized acceptance gates on one source snapshot.
