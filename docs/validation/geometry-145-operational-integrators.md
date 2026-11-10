# Geometry #145: bounded RK4 substeps on the frozen contour field

Latest follow-up: [the neighboring-derivative verifier correction](geometry-145-operational-verifier.md)
passes self-consistency at the unchanged tolerance while preserving the raw
central-difference failure. Physical trajectories and scalar counterexamples below
remain unchanged.

October 30, 2026. The [contour field](geometry-145-operational-contour.md) and its
two scalar counterexamples remain unchanged. This follow-up fixes the observed
crossing-refinement and coarse-step drag failures with an explicit experimental
integration profile. Shipping adoption remains pending; PR #148 remains unready.

## Isolate integration from field and event sampling

The [fixed diagnostic](geometry-145-operational-integration-experiment.json)
compares Euler and RK4 on identical coefficients, initial state, launch energy
and force rules. Every run owns its immutable profile. It retains the original
first-negative-sample crossing time and separately brackets an operational zero
along the final accepted segment. The latter is a diagnostic, not a replacement
test definition or an exact ODE event.

| Timestep | Euler sampled crossing | RK4 sampled crossing | RK4 bracketed segment zero |
| --- | --- | --- | --- |
| 0.002 | 2.366 s | 2.650 s | 2.6480935215 s |
| 0.001 | 2.349 s | 2.649 s | 2.6480934053 s |
| 0.0005 | 2.412 s | 2.6485 s | 2.6480934053 s |
| 0.00025 | 2.516 s | 2.64825 s | 2.6480934053 s |
| 0.000125 | 2.673 s | 2.648125 s | 2.6480934053 s |

The original three Euler times fail the retained refinement inequality. The
original three RK4 sampled times satisfy it without changing the crossing test.
At timestep 0.001, the conservative launch's maximum energy excursion falls from
2.2783 under Euler to approximately `1.30e-7` under RK4. Interpolating the event
alone would not fix Euler's trajectory or energy error.

## The full replay caught a second coarse-step error

Single-step RK4 passes crossing refinement, but fails the existing kinetic-drag
test at `(-0.5, 0.1)`, velocity `(0.2, 0.2)`, beta 0.5 and timestep 0.01.
Potential forces are enabled in that test, so a kinetic increase alone is not a
general energy-injection proof. The independent work ledger nevertheless shows
a real numerical error here. Subdividing the same interval converges to a kinetic
decrease. The original fixture and its `delta_kinetic <= 1e-6` threshold stay intact.

| RK4 subdivisions of the same 0.01 s interval | Kinetic change | Energy change minus integrated drag work |
| --- | --- | --- |
| 1 | +0.00448636 | +0.00660744 |
| 2 | -0.00221497 | -0.0000889221 |
| 4 | -0.00212251 | +0.0000011166 |
| 8 | -0.00212323 | +0.0000003643 |
| 16 | -0.00212356 | +0.0000000254 |
| 64 | -0.00212359 | +0.000000000103 |

An additional diagnostic disables both potential and wall forces. Its kinetic
decay converges to the exact pure-drag law `K(t) = K(0) * exp(-2*beta*t)`, with
the same reduction in the energy/work residual. These controls distinguish
integration error from potential-to-kinetic conversion. No velocity, energy,
force, gradient or Hessian was clamped to make a check pass.

## Explicit four-substep profile

The [candidate declaration](geometry-145-operational-rk4-substeps-experiment.json)
selects `Rk4FourSubsteps`. Each outer step performs exactly four RK4 substeps,
holds the supplied control covector fixed, sums the four work quadratures, and
checks every complete stage and internal weighted endpoint. Position, velocity,
mechanical time, energy and integrated work commit once, after the final endpoint
and ledger checks succeed. A refusal during any substep returns no partial state.

The original `Rk4` method and its failed full replay remain preserved under a
[separate declaration](geometry-145-operational-rk4-experiment.json). The new
method has a distinct enum identity and reports `substeps: 4` and its work rule
in the profile manifest. It does not reuse the single-step RK4 identity.

The field ID remains `b45293bd1d6a1bd6`, asset SHA-256
`2a7afa0f10119b508bde3fdfb80438c3724137d824ed037239eb8173f9330dfc`.
The unchanged field-bound reference-policy SHA-256 remains
`990870ca9a516984f824c37f7aed06f2ea7551e3deddfc830c911b03a73de86d`.
The timed native/WASM profile is `54b525d2266d27eb`; the original native Controls
test uses profile `d9f1b51a967aff60` because its timestep, drag and potential differ.
Each rollout pins its own complete pair. No field, policy or profile changes
mid-rollout, and no coverage extension or resumed miss occurs.

Native, Python and WASM select the experimental method explicitly. The ordinary
activation path and shipping builds retain Euler. The legacy tuple API pins its
supplied config, drag and timestep for each complete transaction. It has no
persistent mechanical ledger; the owned `FrozenRollout` is the authority for
multi-step time/work state. The full owned native/WASM replay tests that contract
independently of the legacy compatibility tests.

## One-source validation

| Full unchanged Python physics file | Pass | Fail | Remaining failures |
| --- | --- | --- | --- |
| Euler baseline | 25 | 2 | Crossing-time refinement; scalar certification |
| Single-step RK4 | 25 | 2 | Coarse-step kinetic drag; scalar certification |
| Four-substep RK4 | 26 | 1 | Scalar certification |

The whole Python file remains byte-identical, SHA-256
`2451ab5c7ffa18f9cc8259ef109a4bca27cd1af846f9ac6843a626ed93c84536`.
The original native crossing body remains byte-identical against the fixed
checkpoint, SHA-256
`e1afb878f44a2644f1efbcf97a7b94a6e7b8f56c798dd59e41f0dfb6af340895`.
It passes under the new pinned profile and has conclusive reference Shore
crossing evidence. Its work count is 101,338 geometry evaluations over the whole
unchanged driven/undriven test.

All **24 focused Rust controls pass**, including legacy/owned RK4 agreement,
Euler-default restoration, wrong-profile rejection, four-substep analytic drag
accuracy, and full rollback after a required stage in substep two refuses.
Earlier endpoint, complete-differential-support, identity and cut-policy tests
remain intact.

Native and WASM pass all **35 per-run mechanical expectations**, including
21 operational crossings and 14 expected noncrossing controls. Fifteen crossing
trajectories have conclusive ordered reference realm pairs; six remain unverified.
All three original above-barrier timestep cases have reference crossing evidence,
as does the separate original native Controls test. These prove crossings on the
continuous reconstructed accepted-state trajectories, not exact ODE event times
or equality between the operational zero contour and the Shore.

Original high-curvature and near-Shore energy excursions are approximately
`5.06e-12` and `2.45e-11`, below their unchanged 0.05 limits. Native/WASM parity
passes at the declared tolerances for jets, trajectories, energy, time, work,
validity and profile identities. Browser initialization at the known reference
cut `(-1, 0)` refuses and preserves prior state. Unknown cuts remain unknown.

The scalar audit still proves **two quality-budget violations**. Unknown scalar
error still fails the original Python quality assertion. The original neighboring
Hessian diagnostic also remains red, with the same finite-difference truncation
witness documented in the contour report. Passing operational energy or trajectory
tests does not establish reference scalar/differential fidelity.

## Cost

Device: Intel Core i9-14900, Windows; headless Chromium 153.0.8010.12. Builds and
other validation workloads finished before timing. The timed owned step now
requires **160 geometry evaluations**, at most **5,760 coefficient reads**, and
one outer commit. A prepared query still requires 36 reads. Work is fixed and
independent of cache history or wall-clock deadlines.

| Measurement, microseconds | Native, three fresh processes | Browser |
| --- | --- | --- |
| First prepared query | 5.6 / 3.9 / 4.6 | 190 |
| Cache-stressed query p99 | 3.1 / 2.4 / 2.5 | 15 |
| Cache-stressed query maximum | 25.4 / 29.2 / 11.1 | 1,085 |
| Complete-step p99 | 57.5 / 52.5 / 60.4 | 115 |
| First complete step | 55.5 / 56.1 / 55.6 | 930 |
| Complete-step maximum | 99.1 / 113.1 / 87.9 | 930 |

The provisional 100 microsecond query p99 and 2 ms complete-step p99 targets pass.
First/max query times still do not establish a hard query deadline. Cache stress
touches 64 MiB outside the query timer. Timing excludes preparation and loading.

Native asset loading takes 0.126 to 0.136 s. Browser WASM initialization, local
fetch, decode and policy/manifest fetch measure 27.75, 136.88, 161.10 and 4.24 ms,
respectively. Coefficients remain 77,405,000 bytes. Native replay OS peak working
sets are 248,135,680 / 248,315,904 / 248,369,152 bytes, including diagnostics and
stress storage. Browser WASM capacity remains 160,301,056 bytes, plus JS input and
stress allocations; this is not renderer RSS. The prior 131.55/143.41 s offline
preparation and measured 216,313,856-byte preparation peak apply to these identical
asset bytes and are reported separately from query/step cost.

The [machine-readable record](geometry-145-operational-integrators.json) and
[raw evidence archive](experiments/geometry-145-operational/integrator-evidence.zip)
preserve all three Python replays, integration/work controls, native and browser
replays, reference pairs, fidelity witnesses, and source/binary hashes. Default
native compilation, experimental native/Python/WASM release builds, syntax and
format checks pass.

After building the experimental native example, Python wheel and WASM bundle as
in the contour report, replay with the explicit method and fresh report paths:

```powershell
rtk proxy .venv/Scripts/python.exe scripts/validate_operational_field.py tmp/operational-rk4-wheels/runtime_core-0.1.0-cp38-abi3-win_amd64.whl --asset tmp/geometry145-contour-v1.bin --reference-policy tmp/geometry145-contour-v1-reference-policy.json --integrator Rk4FourSubsteps --report tmp/fresh-four-python.json backend/tests/test_manifold_physics.py -q
rtk proxy .venv/Scripts/python.exe scripts/validate_operational_native_crossing.py tmp/geometry145-contour-v1.bin tmp/fresh-four-crossing.json tmp/geometry145-contour-v1-reference-policy.json Rk4FourSubsteps
./scripts/measure_operational_field.ps1 -Asset tmp/geometry145-contour-v1.bin -OutputDirectory tmp/fresh-four-native -Manifest docs/validation/geometry-145-operational-rk4-substeps-experiment.json -ReferencePolicy tmp/geometry145-contour-v1-reference-policy.json -Integrator Rk4FourSubsteps
rtk proxy node scripts/validate_operational_field_browser.mjs tmp/geometry145-contour-v1.bin tmp/fresh-four-native/run-2.json tmp/fresh-four-browser.json tmp/geometry145-contour-v1-reference-policy.json docs/validation/geometry-145-operational-rk4-substeps-experiment.json Rk4FourSubsteps
```

The next blockers are field fidelity and acceptance of a shipping contract, not
the demonstrated crossing/drag failures. Two scalar counterexamples, absent scalar
upper bounds, the original neighboring-Hessian check, reference derivative
fidelity, full reference-singularity information, compatibility/version work and
deployment/training validation remain unresolved. No pending policy or latency
target was ratified by this experiment.
