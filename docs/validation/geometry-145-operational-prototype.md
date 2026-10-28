# Geometry #145: one frozen operational-field experiment

**Decision: this asset does not satisfy all retained acceptance tests.** Within
its declared corridor, the field passes differential, mechanical and provisional
evaluation-cost checks. Remaining failures identify coverage and reference
evidence gaps. They do not justify another geometry redesign or point-specific
proof expansion. PR #148 remains unready to merge.

ADR 0005 is accepted **for a bounded prototype only**, dated October 28, 2026.
Shipping adoption, reference-cut motion policy and device targets remain pending.
Its header and opening now say this consistently. The experimental Cargo feature
is disabled by default; shipping behavior and controller/golden versions are
unchanged. Original crossing bodies and thresholds remain intact.

The [machine-readable index](geometry-145-operational-prototype.json) records
identities, source/binary hashes, outcomes and timings. Exact ordered trajectories,
reference evidence, native/browser states and ledgers are preserved in the
[compressed evidence](experiments/geometry-145-operational/evidence.zip).
The [fixed recipe](geometry-145-operational-experiment.json) preceded preparation.
The [prepared asset](experiments/geometry-145-operational/field.zip) contains
`field.bin` and `manifest.json`, including mechanical profiles and reference claims.

## Field and profile identity

One immutable C4 tensor-product quintic field supplies analytic value, gradient
and Hessian. A direct query reads at most 36 coefficients, with no fitting,
membership/proof work, I/O, query cache or allocation. The existing GeometryJet
adapter allocates bounded metadata strings. Any missing differential coefficient
refuses the complete query; scalar-only or alternate-provider fallback is forbidden.

- Field ID: `c709f01272dcb1c9`.
- Asset SHA-256: `bd92d6eca3a8c124179473bc8af05af38d98a114b48aaa0c7df9b350dc0ea2c5`.
- Preparation: 1605 by 1005 coefficients, spacing 0.0005, origin (-0.1, -0.2),
  sampled from the existing 1024 raster. Full stencil coverage is approximately
  x in [-0.099, 0.701), y in [-0.199, 0.301). Adapter epsilon is at least 0.0001.
- Storage: 12,904,200 coefficient bytes; 12,904,256 including the binary header;
  approximately 6.14 MB in the archive with manifests.

These are operational coverage parameters, not certified fractal-scale resolution.
Reference scalar/differential fidelity and corridor reference-singularity
information remain unknown. The adapter reports maximal scalar uncertainty,
retaining the original scalar-quality failure.

Every rollout owns a fixed field/profile pair. The Physics profile identifies
d_ref, epsilon, lambda squared, kappa, wall strength, timestep, drag, integrator,
work-accounting rule and reference-refusal policy/evidence snapshot. Controls
remain explicit inputs. IDs use hex strings across serialization to avoid losing
64-bit identity through JavaScript numbers. Parameter changes or field replacement
inside an evaluation refuse; state from another pair cannot be stepped.

No coefficient replacement or coverage extension occurred. Reconstruction yielded
byte-identical asset data. Three fresh native processes produced identical frozen
mechanical results. Native/browser comparisons use declared numerical tolerances,
not a new cross-target bitwise requirement.

## Executable contracts

Fourteen native controls pass. Test-first failures demonstrated that changed
lambda, drag and timestep initially slipped through a field-only boundary; they
now refuse. The controls include:

- `F(x,y)=x` through regularization, crest, metric, connection and integration.
  Closed expressions independently check rho, its gradient, rho times its
  Hessian, the diagonal metric and Christoffel symbols.
- Polynomial and grid-join derivatives, complete differential support, invalid
  assets, binary roundtrip, cache stress and reordered queries/refusals.
- Intermediate RK-stage failure and a missing final weighted endpoint after all
  four stages succeed. Serialized position, velocity, mechanical time, integrated
  work, energy and identities remain unchanged after refusal.
- Authoritative reference refusal overriding a smooth operational sample, using
  both an analytic control and the canonical provider's actual singularity at
  (-1, 0). This grants no permission to move through reference cuts.
- Measured or unknown distance evidence never certifying realm. Certified point
  bounds only imply realm strictly outside their band at that same point.

Neighbor checks retain steps 1e-6/1e-7, absolute gradient tolerance 1e-4 and
normalized Hessian tolerance 1e-4. Maximum errors are 1.71e-7 and 4.18e-5.
Explicit join probes attain 9.10e-8 and 1.01e-6. These establish differential
self-consistency, not reference differential fidelity. Analytic controls run
natively; browser parity separately exercises prepared-field mechanics through
the same Rust chain.

## Mechanical and reference verdicts

The fixed recipe uses original starts, x/y offsets of plus/minus 1e-5, and speed
scales 0.99/1.01. All seven perturbations use the same asset, launch rule and
original timestep choices. Starts were not adjusted after failures.

| Check | Outcome |
| --- | --- |
| Declared corridor matrix | 35/35 mechanical expectations pass; no coverage misses |
| Above-barrier trajectories | 21/21 operational crossings; reference crossing unverified |
| Below-barrier and ridge controls | 14/14 expected non-crossing outcomes |
| Frozen high-curvature / near-Shore energy | Excursions 0.000250445 / 0.003406750, below unchanged 0.05 |
| Original Python crossing tests | Both pass, including timestep-refinement assertion |
| All retained Python energy/crossing/continuity tests | 10 pass, 3 fail |
| Original native controlled-crossing test | Coverage refusal in undriven branch |

Two Python energy tests start at (-0.5, 0.1), outside this frozen corridor, and
fail with `GeometryUnavailable`. The other failure is the unchanged assertion
`estimated_error <= 0.25 * requested_scale`. No test was skipped or weakened.

The private native `controls_drive_can_cross_shore_without_wall` body is run
verbatim in a temporary example under pinned identities. Its body hash is checked
against `9187815`, and tracked source remains unchanged. The undriven trajectory
reaches c=(0.1042700898980602, 0.30099067225167203),
v=(-0.30360212013543375, 0.27188333310990276); its attempted continuation leaves
the halo. The driven branch is not reached. No patch was added and no failed run
was resumed. This demonstrates the warned-about change in trajectory coverage.

After each successful trajectory is frozen, the existing bounded outward-rounded
membership implementation supplies reference evidence at selected ordered states.
Unknown stays unknown; no proof budget is expanded. None of the 21 operational
crossings obtains a conclusive opposite-realm pair, so its reference-crossing
verdict remains `unverified`. Continuity means the piecewise-linear reconstruction
of committed states, not certification of the exact continuous ODE solution.

Measured raster disagreement, at most 7.4821e-5 on diagnostic samples, never becomes
a certified distance bound. Reference-singularity refusal enforcement passes its
controls, while reference-singularity information for the whole corridor remains
unknown. Energy success describes operational mechanics, not reference fidelity.

## Native and browser cost

Windows 11 build 26100, Intel Core i9-14900, Rust 1.98.0 release builds. Browser:
headless Chromium 153.0.8010.12 on that device, with cross-origin isolation.
Native uses three fresh processes; browser uses a fresh context with identical
asset bytes loaded before timing.

| Measurement | Native | Browser |
| --- | --- | --- |
| First prepared query | 6.2 to 8.4 microseconds | 180 microseconds |
| Cache-stressed query p99 | 2.6 to 3.0 microseconds | 15 microseconds |
| Maximum cache-stressed query | 34.3 microseconds across final runs | 1,185 microseconds |
| First complete transactional step | 10.4 to 16.9 microseconds | 780 microseconds |
| Complete transactional step p99 | 8.7 to 14.8 microseconds | 35 microseconds |
| Maximum transactional step | 38.3 microseconds across final runs | 780 microseconds |
| Asset loading / decoding | 18.9 to 20.3 ms | 27.7 ms |

Preparation compute is 87.3 ms; serialization/write adds 10.9 ms, totaling 98.2 ms.
Browser WASM initialization is 22.9 ms and local asset fetch 40.7 ms. Preparation,
initialization and loading are excluded from query/step timers and reported separately.

Each run measures 20,000 distinct queries, 20,000 unavailable queries, 1,000 queries
after 64 MiB cache stress, and 1,000 steps near (0.4, 0.05). Complete-step timing
uses the frozen Euler profile with an explicit zero control covector, endpoint
checks, energy and mechanical-ledger commit. It performs 24 geometry evaluations,
at most 864 coefficient reads. Native/browser work counts agree. RK4 has separate
transaction regressions; these measurements do not adopt RK4.

Native sampled OS peak working set is 116.6 to 116.9 MB, including field/input
copies, reference diagnostics and 64 MiB scratch. Browser WASM linear-memory
capacity after the experiment is 31,195,136 bytes, with separate 12.9 MB input
and 64 MiB JavaScript scratch buffers. This is not renderer resident memory.
The experimental WASM binary is 4,984,870 bytes before transport compression.

The provisional 100-microsecond query and 2-ms step **p99** targets pass on these
devices. First-call/max outliers remain visible; these are not hard deadlines or
ratified shipping requirements. Browser bridge allocation/serialization is
included. Cache stress occurs outside timers and does not prove every hardware
cache is cold. There is no deferred mathematical cache miss or proof construction.

Native/browser parity passes with scalar tolerance 1e-6 absolute, gradient 1e-4
absolute, Hessian 1e-4 normalized, and state/ledger 1e-6 normalized. Availability
and rollout verdicts agree.

## Reproduce

Use the project venv and installed frontend Playwright dependency. The archive
contains the exact asset; optional reconstruction must match its SHA-256. No
global wheel installation is needed.

```powershell
rtk cargo test --manifest-path runtime-core/Cargo.toml --release --features experimental-operational-field operational_field -- --test-threads=1
rtk proxy .venv/Scripts/python.exe -m zipfile -e docs/validation/experiments/geometry-145-operational/field.zip tmp/operational-restore
Copy-Item -LiteralPath tmp/operational-restore/field.bin -Destination tmp/geometry145-frozen-operational.bin
rtk cargo build --manifest-path runtime-core/Cargo.toml --release --features experimental-operational-field --example operational_field_report
rtk proxy target/release/examples/operational_field_report.exe --prepare tmp/geometry145-frozen-operational-repro.bin > tmp/operational-field-preparation-final.json
rtk proxy .venv/Scripts/python.exe -m maturin build --manifest-path runtime-core/Cargo.toml --release --features python,experimental-operational-field --out tmp/operational-field-wheels
rtk proxy .venv/Scripts/python.exe scripts/validate_operational_field.py tmp/operational-field-wheels/runtime_core-0.1.0-cp38-abi3-win_amd64.whl --asset tmp/geometry145-frozen-operational.bin --report tmp/operational-field-pytest-frozen-final.json
rtk proxy .venv/Scripts/python.exe scripts/validate_operational_native_crossing.py tmp/geometry145-frozen-operational.bin tmp/operational-field-original-native.json
& ./scripts/measure_operational_field.ps1
rtk proxy wasm-pack build wasm-orbit --target web --release --out-dir ../tmp/operational-field-wasm -- --features experimental-operational-field
rtk proxy node scripts/validate_operational_field_browser.mjs tmp/geometry145-frozen-operational.bin tmp/operational-field-native/run-0.json tmp/operational-field-browser.json
rtk proxy node scripts/collect_operational_acceptance.mjs
```

Retained Python/native commands intentionally exit 1 for the classified failures.
The collector also exits 1 for retained acceptance failure while writing its
evidence. A passing corridor, parity or integrity check is not a green merge gate.

A larger-domain experiment requires a new manifest and fresh replay, rather than
continuing the failed native path. Reference fidelity and singularity coverage
need scoped evidence or an explicit acceptance revision. This result authorizes
neither a new proof engine nor a silent change in motion policy.
