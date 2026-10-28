# Geometry #145: bounded operational-field prototype

ADR 0005 is authorized for this experiment. PR #148 is still not ready to merge.
The new Rust implementation is behind `experimental-operational-field`, disabled
by default. Shipping geometry, controller versions, golden vectors, crossing test
bodies and acceptance thresholds are unchanged.

## Representation and coverage

The prototype evaluates one immutable tensor-product quintic B-spline. Each
successful query reads 36 coefficients and returns its analytic value, gradient
and Hessian. The field is C4 across grid joins. There is no query cache, adaptive
membership search, fitting, proof construction, I/O or allocation in the direct
field lookup. The adapter to the existing `GeometryJet` API allocates bounded
metadata strings. Field identity includes the representation and coefficient
bits. Exact repeats, neighboring queries and query order select the same field.

The first baseline samples the existing 1024 raster during preparation. It uses
1,605 by 1,005 coefficients, spacing 0.0005, origin (-0.1, -0.2), and 12,904,200
coefficient bytes. Full stencil coverage is x in [-0.099, 0.701), y in
[-0.199, 0.301), subject to floating-point coordinate conversion. The experimental
adapter supports epsilon >= 0.0001. These are declared preparation parameters,
not claims that a raster resolves the corresponding true fractal scale.

The original crossing corridors, their Shore bisections and observed step
endpoints fit this domain. It is not general Map coverage. Coordinates lacking a
full stencil and unsupported epsilon return `GeometryUnavailable`; they do not
fall back to the adaptive provider. Preparation and explicit activation happen
before replay. Activation is thread-local and intended only for this synchronous
experiment. This is not a training-worker or browser deployment interface.

Reference fidelity and reference cut information are **unknown**. The adapter
names this experimental profile explicitly, reports maximal scalar uncertainty,
and reports operational regularity rather than claiming a unique reference
normal. Smoothing a cut in this experiment does not change ADR 0004's shipping
singularity/refusal policy. The baseline has no certified reference regions.

## Results on the frozen candidate

The [machine-readable record](geometry-145-operational-prototype.json) contains
the source, original test, raster, wheel, extension and benchmark executable
SHA-256 hashes, field identity and individual pytest outcomes. Field identity is
`c709f01272dcb1c9`. The project venv staged the wheel in a temporary directory;
the installed runtime extension was not replaced.

- Both original Python crossing tests pass, including the above-barrier launch
  at dt 0.002, 0.001 and 0.0005 and its original crossing-time refinement check.
- The unchanged high-curvature and near-Shore conservative energy rollouts pass
  using the shipping semi-implicit integrator. Maximum absolute excursions are
  0.000250445 and 0.003406750, respectively, below the unchanged 0.05 limit.
  These describe the operational field, not the previous adaptive field or an
  RK4 adoption.
- The selected existing Python run is **9 passed, 1 failed**. The remaining
  failure is `test_jet_exposes_scale_and_convergence_quality`: unknown scalar
  error does not satisfy `estimated_error <= 0.25 * requested_scale`. This is
  retained as a failure; no threshold or assertion was weakened.
- Across 1,000 deterministic points, both original finite-difference step sizes
  1e-6 and 1e-7 produce maximum absolute gradient and normalized Hessian discrepancies
  1.71e-7 and 4.18e-5, below 1e-4. A first cubic prototype failed this Hessian
  diagnostic at 4.41e-4. The quintic representation fixes that join sensitivity.
- Five focused Rust controls pass: analytic polynomial derivatives through
  joins, query-order/miss behavior, malformed assets, an analytic reference cusp
  with distinct operational derivatives, and Physics refusal on coverage loss.
- Default native compilation and the actual WASM bridge's wasm32 compilation
  pass. A direct runtime-only wasm32 check failed on the existing `getrandom`
  JS feature requirement; the bridge supplies that feature. This is compilation
  evidence, not browser replay, browser latency or a browser artifact-size gate.

Measured scalar disagreement with the input raster reaches 7.4821e-5 on 1,000
points. This is a sampled raster comparison, not a uniform bound or a comparison
against true Mandelbrot distance. Passing the crossing tests establishes motion
through the operational zero contour, not certified crossing of the true Shore.

## Cold-query and preparation cost

Windows x64, Intel Core i9-14900, Rust 1.98.0, release build. Three fresh native
processes were measured. Preparation includes raster loading, coefficient
construction and fingerprinting; it is excluded from query timers.

| Measurement | Recorded result |
| --- | --- |
| Preparation | 77.5 to 105.5 ms |
| First prepared query in a fresh process | 0.9 to 1.4 microseconds |
| Distinct prepared lookups | 20,000 per run; full percentiles in JSON |
| Lookup after 64 MiB CPU-cache stress | 1,000 per run; p99 2.0 to 4.5 microseconds |
| Maximum in final cache-stress runs | 166.0 microseconds |
| Unavailable lookup | 20,000 per run; bounded coordinate checks |
| Complete existing Physics step | 1,000 per run; p99 8.3 to 14.0 microseconds; maximum 62.9 microseconds |

An earlier 64 MiB scouting run recorded a 352.8 microsecond maximum with p99
2.1 microseconds. Keep that scheduling/cache-stress outlier visible; these
measurements do not establish hard latency deadlines. Cache stress occurs
outside the timed query and is not proof that every CPU cache is cold. A
prepared query has no mathematical cache to warm or cold proof miss to hide.
Full benchmark process duration is about three seconds because it includes
cache stress and diagnostics; it is not a field-query measurement. Native
timing targets remain proposed for deployment, and browser timing is unmeasured.

## Reproduce

Use the project venv on Windows. No global wheel installation is needed.

```powershell
rtk cargo test --manifest-path runtime-core/Cargo.toml --release --features experimental-operational-field operational_field::tests -- --test-threads=1
rtk cargo run --manifest-path runtime-core/Cargo.toml --release --features experimental-operational-field --example operational_field_report
rtk proxy .venv/Scripts/python.exe -m maturin build --manifest-path runtime-core/Cargo.toml --release --features python,experimental-operational-field --out tmp/operational-field-wheels
rtk proxy .venv/Scripts/python.exe scripts/validate_operational_field.py tmp/operational-field-wheels/runtime_core-0.1.0-cp38-abi3-win_amd64.whl --report tmp/operational-field-pytest.json
rtk cargo check --manifest-path wasm-orbit/Cargo.toml --target wasm32-unknown-unknown --features runtime_core/experimental-operational-field
```

The pytest command intentionally exits 1 because the current scalar-quality gate
remains unmet. It replays existing test bodies, records every selected outcome,
and checks that the original test source did not change. The parent process
removes the temporary wheel after the child releases the Windows DLL.

## Remaining adoption work

This experiment supports separating reference preparation from immutable live
evaluation: coherence, both crossing regressions and native query cost can work
together on a finite operational field. It does not establish fractal-scale
fidelity. Next, supply region/scale-specific reference evidence and a declared
fidelity contract for prepared coefficients, including unknown reference cuts.
Evaluate that contract without changing the original crossing and energy tests.
Coverage-loss product behavior, field-version/reset semantics across coverage
extensions, training and browser parity, browser latency and deployment asset
cost remain open. Canonical adoption still requires compatibility versions,
goldens and acceptance on one final profile.
