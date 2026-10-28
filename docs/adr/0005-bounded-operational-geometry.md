# 0005 — Separate reference geometry from bounded operational geometry

Status: Accepted for a bounded experimental prototype (2026-10-28, authorized by
repository owner). Shipping adoption, reference-cut motion policy and device
latency targets remain pending. ADR 0004 and the current shipping acceptance
contract remain in force. Related: #120, #145, PR #148.

FractalSync needs a coherent differential field that Physics can evaluate within
a live frame budget. We propose separating that operational field from the
Mandelbrot reference evidence used to construct and assess it. This changes the
meaning of the geometry consumed by Physics and must be versioned explicitly.

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

The first implementation should use prepared coefficients for a finite test
domain, with a fixed amount of Rust evaluation work. Preparation may perform
expensive reference calculations outside the Physics step. This proposal does
not require a general streaming or background-proof framework.

Cold prepared queries, cache eviction and exact repeats must evaluate the same
field. Cache state, query order, wall-clock deadlines and newly arrived evidence
must not choose a different local scalar function during a rollout. Pin the
field identity across all integrator stages and energy measurements. Extending
coverage must preserve the existing field and its derivatives on the overlap;
changing the field requires an explicit version/reset or separately designed
time-dependent mechanics.

An unprepared query returns `GeometryUnavailable` in bounded work. It does not
fall back to a different metric, fetch data, fit a new field or launch a proof.
A refused step commits no partial movement. Holding the last frame is a
candidate product response, not a successful Physics step or a guarantee of
live continuity. The domain contract requires a deliberate response to coverage
loss before shipping. A sampled training corpus does not prove the Player will
avoid unsupported geometry.

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
