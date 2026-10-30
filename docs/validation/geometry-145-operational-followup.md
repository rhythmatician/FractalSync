# Geometry #145: frozen browser replay and scalar-fidelity audit

Latest follow-up: [source-raster isolation and the original native reference crossing](geometry-145-operational-source.md).
The source study rules out finer coefficient spacing as a repair. The original
native driven trace now has a certified inside/outside pair; the separate 35
fixed runs below retain their unverified reference verdicts.

October 30, 2026. ADR 0005 authorizes this bounded prototype only. Shipping
adoption, reference-cut motion policy and device targets remain pending.

The frozen larger field passes the retained operational mechanics and declared
native/WASM parity checks. It fails the retained scalar-fidelity contract at
43 fixed audit points. PR #148 is not ready to merge. The blocker is now a
demonstrated field-quality defect, rather than only an absent error certificate.

## Fixed inputs and mechanical evidence

This reuses the complete [coverage asset and policy](geometry-145-operational-coverage.md).
The field ID remains `068e8e016b4ad6b7`, asset SHA-256
`d77495871dd3fd9d0b66876ffe9148bade43234227ca4eb918a56d234069925e`, and
policy SHA-256 `029024d540c8f63397f70b87d7097456ab4165ea5ac6819c4764d734d1429101`.
No coefficients, domain, thresholds or original tests changed during this replay.
All assets, policy and manifests load before timing and before rollout.

The shared Rust report now accepts the wider-domain declaration while inheriting
the original mechanical experiment's starts, perturbations, horizons, tolerances
and targets. It cannot replace these with caller-supplied mechanics. Each rollout
pins its own field/profile pair. The WASM constructor validates the same
field-bound reference snapshot as native, and the report uses that snapshot.

- All **18 Rust contract controls pass**, including the analytic plane derivative
  chain, cache/query-order invariance, complete differential support and rollback
  after stage or final-endpoint refusal. Two additions check inherited mechanical
  settings and the certified-realm error-floor implication.
- The original native controlled-crossing body passes unchanged. Its body hash
  remains `e1afb878f44a2644f1efbcf97a7b94a6e7b8f56c798dd59e41f0dfb6af340895`.
- The full Python physics file remains **26 passed, 1 failed**. The failure is
  `test_jet_exposes_scale_and_convergence_quality`; the unknown-error sentinel
  still fails the original `0.25 * requested_scale` requirement. Test-file SHA-256
  remains `2451ab5c7ffa18f9cc8259ef109a4bca27cd1af846f9ac6843a626ed93c84536`.
- Native and WASM both pass all **35 fixed mechanical expectations**, with
  21 operational crossings. Every reference Shore crossing remains unverified.
  Neighboring and patch-join derivative checks pass their unchanged tolerances.
  These establish differential self-consistency, not reference derivative fidelity.
- Native/WASM parity passes the declared scalar, gradient, Hessian and state/ledger
  tolerances. No cross-target bitwise-equality requirement was introduced.
- In the browser, the scalar jet at the known reference cut `(-1, 0)` remains
  available, but Physics initialization refuses with `ReferenceSingularity` and
  preserves its prior complete state. Point evidence does not certify other cuts.

## Prepared-query cost

Device: Intel Core i9-14900, Windows, headless Chromium **153.0.8010.12**.
Three fresh native processes and one browser process use the same asset bytes.
Timings exclude preparation/loading. Native sampling overlapped the Python
validation initially; these are observed desktop timings, not controlled hardware
deadline guarantees. The browser replay ran after both completed.

| Measurement | Native, three processes | Browser |
| --- | --- | --- |
| First prepared query | 7.6 / 8.8 / 8.6 microseconds | 190 microseconds |
| Cache-stressed query p99, 1,000 samples | 3.5 / 3.2 / 3.9 microseconds | 30 microseconds |
| Cache-stressed query maximum | 19.2 / 21.9 / 13 microseconds | 1,815 microseconds |
| Complete transactional step p99, 1,000 samples | 8.4 / 10.8 / 24.8 microseconds | 100 microseconds |
| First complete transactional step | 21.6 / 16.3 / 17.5 microseconds | 1,785 microseconds |
| Complete-step maximum | 21.6 / 101.3 / 52 microseconds | 1,785 microseconds |

The provisional **100 microsecond query p99** and **2 ms complete-step p99**
targets pass. First and maximum browser query times exceed 100 microseconds;
passing p99 does not establish a hard query deadline. The timed step requires
24 geometry evaluations and at most 864 coefficient reads. A direct prepared
query reads 36 coefficients. Cache stress touches 64 MiB between queries.

The unchanged preparation cost was 0.461 s compute and 0.562 s including write.
Browser WASM initialization took 22.82 ms, local asset fetch 100.73 ms, decode
162.95 ms, and policy/manifest fetch 4.96 ms. These are separate observations,
not a network deployment estimate. Native asset loading was 0.128 s in run 2.

Coefficient storage is 77,405,000 bytes; encoded input is 77,405,056 bytes.
Browser WASM linear-memory capacity is 160,301,056 bytes. The JS input and 64 MiB
stress buffer are additional allocations; capacity is not renderer RSS. Native
sampled OS peak working sets are 246,018,048 / 248,586,240 / 248,590,336 bytes,
including diagnostics and serialization. The report does not equate these with
the resident coefficient allocation.

## Conclusive scalar-fidelity counterexamples

The [fixed audit plan](geometry-145-operational-fidelity-experiment.json) samples
765 points from the canonical period-one lobe parameterization at three fixed
multiplier insets. Every point is covered and has conclusive reference membership
from the existing conservative classifier. No new proof engine or escalation was
added. This offline audit never mutates the prepared field or a running rollout.

There are **44 certified sign disagreements**, of which **43 prove violations**
of the unchanged scalar-error budget. At a certified-inside point, true signed
distance D is nonpositive. A positive returned F therefore proves
`abs(F-D) >= F`. This pointwise lower bound needs no nearest-feature or derivative
claim. Sign agreement or unknown membership supplies no corresponding certificate.

The largest measured error floor is at
`(-0.5022678182841798, 0.5161133557384199)`. Reference membership is inside,
but F is `0.0004890054371243826`. Its error is therefore at least that amount,
against a retained budget of `0.000012478138821986048`, a factor of 39.19.
The budget uses `0.25 * (0.1 * hypot(F, 1e-4))`, matching the existing query scale.
All 43 witnesses and their conclusive realms are preserved in the evidence archive.

These pointwise counterexamples disprove the scalar-budget claim for this asset.
They do not provide a regional distance upper bound, certify gradients/Hessians,
or settle reference crossings. Certifying the current coefficients alone cannot
clear this failure. Another preparation experiment would need its own fixed inputs,
new field identity and fresh full replay; this report does not authorize shipping
a relaxed fidelity claim or expanding point-specific proofs indefinitely.

## Evidence and reproduction

The [machine-readable summary](geometry-145-operational-followup.json) records
source/binary hashes, identities, test outcomes and complete timing distributions.
The [compressed raw evidence](experiments/geometry-145-operational/coverage-followup-evidence.zip)
preserves all three native reports, browser report, original-test outcomes,
reference policy and every audit witness. The 77 MB asset remains reproducible
from the existing fixed recipe and is not a published shipping asset.

Build native examples and the Python wheel with `experimental-operational-field`,
plus `python` for the wheel. Build wasm-orbit with the experimental feature using
`wasm-pack`. The existing coverage report gives the asset/policy preparation and
original-test commands. For the new diagnostics, run:

```powershell
rtk cargo run --release --manifest-path runtime-core/Cargo.toml --features experimental-operational-field --example operational_fidelity_audit -- tmp/geometry145-coverage-v1.bin docs/validation/geometry-145-operational-fidelity-experiment.json
./scripts/measure_operational_field.ps1 -Asset tmp/geometry145-coverage-v1.bin -OutputDirectory tmp/fresh-native -Manifest docs/validation/geometry-145-operational-coverage-experiment.json -ReferencePolicy tmp/geometry145-coverage-v1-reference-policy.json
rtk proxy node scripts/validate_operational_field_browser.mjs tmp/geometry145-coverage-v1.bin tmp/fresh-native/run-2.json tmp/fresh-browser.json tmp/geometry145-coverage-v1-reference-policy.json docs/validation/geometry-145-operational-coverage-experiment.json
```

Current merge blockers remain scalar/reference differential fidelity, actual Shore
crossing evidence, complete reference-singularity information, shipping adoption
and compatibility, and deployment/training validation. The bounded experiment
supports the representation's mechanical coherence and provisional evaluation
cost, while rejecting this raster-backed preparation's scalar-quality claim.
