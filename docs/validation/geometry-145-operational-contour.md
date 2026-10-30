# Geometry #145: frozen finite-contour source experiment

Latest follow-up: [the explicit four-substep RK4 profile](geometry-145-operational-integrators.md)
passes the unchanged crossing-refinement and drag tests on these same field
bytes. Python is 26 passed, 1 failed under that profile. The Euler snapshot below
and both scalar counterexamples remain preserved.

October 30, 2026. This follows the [raster-source diagnosis](geometry-145-operational-source.md).
The source changes; the dense C4 quintic evaluator, coefficient spacing, coverage,
original test bodies and thresholds do not. This is a separate bounded prototype,
not shipping adoption. PR #148 remains unready.

## Fixed construction and identity

The [experiment declaration](geometry-145-operational-contour-experiment.json)
was fixed before preparation. Rust constructs a finite membership mesh on
`[-2.1, 2.1]^2`, with coarse spacing 0.002. Mixed coarse cells receive 16 by 16
subdivision, giving fine spacing 0.000125. Marching squares resolves saddle cells
using the center classification and bisects mixed edges eight times. Every
membership query uses the existing Rust routine with an 8,192-iteration cap.

A bounded nearest-segment tree supplies signed distance samples for the existing
quintic coefficients. This code runs only offline. Prepared queries still read
36 coefficients and never traverse this tree, classify membership, fit a field,
or load data. No existing asset is extended or overwritten.

The source remains **measured, not certified**. Nonescape at a finite iteration
cap does not prove inside membership. Homogeneous coarse cells may hide smaller
features. Segment distances are not a true-Shore distance certificate or a
reference cut-locus classification. No reference-error metadata was upgraded
from unknown to certified.

| Item | Frozen value |
| --- | --- |
| Field ID | `b45293bd1d6a1bd6` |
| Asset SHA-256 | `2a7afa0f10119b508bde3fdfb80438c3724137d824ed037239eb8173f9330dfc` |
| Coordinate origin / spacing / shape | (-1.002, -0.702) / 0.0005 / 3425 by 2825 |
| Encoded asset / coefficients | 77,405,056 / 77,405,000 bytes |
| Reference nodes / refined coarse cells | 4,414,201 / 11,287 |
| Segments / membership queries | 312,455 / 22,354,903 |
| Declared segment / membership-query ceilings | 500,000 / 50,000,000 |

The asset remains untracked at `tmp/geometry145-contour-v1.bin`. The complete
recipe and hashes make it reproducible. The known canonical refusal at `(-1, 0)`
is snapshotted offline into a new field-bound policy before replay. Its SHA-256
is `990870ca9a516984f824c37f7aed06f2ea7551e3deddfc830c911b03a73de86d`.
Its authority is still exact-point only; reference information elsewhere remains
unknown. Every replay starts afresh and pins a field/profile pair.

## Retained acceptance results

| Check | Result |
| --- | --- |
| Focused Rust contract and preparation controls | 21 pass |
| Original native controlled-crossing body | Pass, including a conclusive reference crossing |
| Full unchanged Python physics file | 25 pass, 2 fail |
| Fixed per-run mechanical expectations | 35 of 35 pass on native and WASM |
| Fixed operational crossings | 21 |
| Fixed reference Shore crossings with conclusive ordered pairs | 17 pass, 18 unverified |
| Native/WASM jet, state, ledger and profile parity | Pass at declared tolerances |
| Patch-join differential checks | Pass |
| Original neighboring derivative diagnostic | Fail at the larger finite-difference increment |
| Original energy limits | Pass |
| Fixed 765-point scalar-error audit | 2 conclusive budget violations |

The three preparation controls exercise a plane through its crest, disconnected
contours and nearest-component selection at an exact tie, plus budget exhaustion
and empty-source refusal. A preliminary circle-field assertion incorrectly
expected exact distance at a cut locus. That failed control was separated into
an exact source-distance tie check and field-fidelity samples at regular points.
No original crossing, derivative or energy threshold changed.

The original native test body hash remains
`e1afb878f44a2644f1efbcf97a7b94a6e7b8f56c798dd59e41f0dfb6af340895`.
Its driven/undriven trajectories use 13,925 geometry evaluations in total. The
same mechanics and reference policy retain profile ID `b61c1e0bbba796e4`; the
field ID changes because the coefficients changed. A field change need not
change the separate mechanical-profile identity.

The full Python file hash remains
`2451ab5c7ffa18f9cc8259ef109a4bca27cd1af846f9ac6843a626ed93c84536`.
Its failures are:

- `test_jet_exposes_scale_and_convergence_quality`: unknown scalar error still
  fails `estimated_error <= 0.25 * requested_scale`. No small error was invented.
- `test_shore_ridge_reflects_below_barrier_and_crosses_above_it`: all three
  above-barrier launches cross, but their crossing times do not satisfy the
  original refinement inequality. At timesteps 0.002, 0.001 and 0.0005, the times
  are approximately 2.366, 2.349 and 2.412 seconds. The fine gap of 0.063 exceeds
  the coarse gap of 0.017. The earlier raster-backed asset passed this check.

Thus the 35 per-run outcomes do not mean all mechanical acceptance passes.
The cross-run timestep-refinement obligation remains red on this candidate.
High-curvature and near-Shore energy excursions are 0.0006461 and 0.0022352,
respectively, below their unchanged 0.05 limits.

## Scalar evidence and differential diagnosis

The audit uses the same 765 points, conclusive reference membership routine and
scalar budget as before. Only its field binding changes. None of the old 43
counterexample points still fails this one-sided sign test, but two different
points now fail. This does not certify global improvement or bound errors whose
sign agrees with the reference realm.

Both new counterexamples are certified inside. Their positive returned F proves
`abs(F-D) >= F`, independently of nearest-feature differentiability:

| Point | Error lower bound | Retained budget |
| --- | --- | --- |
| (-0.6914277483783785, 0.27300884290578004) | 0.000025824394538970835 | 0.0000025820170595515567 |
| (-0.6914196246209077, 0.2730051820911295) | 0.000006636855129022978 | 0.0000025054999309024687 |

No patch or coefficient was changed after these failures. They remain part of
the frozen result.

The original neighboring check reports maximum normalized Hessian error
`0.0002047859809711231` against its unchanged `1e-4` tolerance. The maximum
gradient error is `0.000004132223399233581`. Join errors are approximately
`1.54e-7` for the gradient and `5.06e-6` for the Hessian, passing the same tolerance.

A separate finite-increment witness diagnoses the neighboring failure without
replacing its gate. At `(0.33961885656970914, 0.053156996587030714)`, the returned
Hessian yy entry is `2.0466397670854977`. Central differences at increments
`1e-6`, `5e-7`, `2.5e-7` and `1e-7` have normalized errors approximately
`2.048e-4`, `5.120e-5`, `1.280e-5` and `2.051e-6`. Richardson extrapolation from
the first increment agrees to `1.405e-9`. This is evidence of finite-difference
truncation at that witness, not of a neighboring field changing with query order.
The original diagnostic remains red. Neither this diagnosis nor operational
energy conservation establishes reference differential fidelity.

## Cost and reproducibility

Device: Intel Core i9-14900, Windows. Browser: headless Chromium 153.0.8010.12.
Builds and other validation workloads finished before native/browser timing.
Timings exclude asset preparation and loading; first and maximum latencies
remain separate from the provisional p99 targets.

| Measurement | Native, three processes | Browser |
| --- | --- | --- |
| First prepared query, microseconds | 7.1 / 6.0 / 7.9 | 200 |
| Cache-stressed query p99, microseconds | 5.3 / 4.7 / 4.4 | 25 |
| Cache-stressed query maximum, microseconds | 172.4 / 33.8 / 27.4 | 2,785 |
| Complete-step p99, microseconds | 35.4 / 34.2 / 30.1 | 75 |
| First complete step, microseconds | 32.1 / 28.2 / 25.3 | 1,055 |
| Complete-step maximum, microseconds | 158 / 446.3 / 126 | 1,055 |

The provisional 100 microsecond query p99 and 2 ms complete-step p99 targets
pass. Query maxima exceed 100 microseconds on both targets; this is not a hard
deadline guarantee. The timed Euler step uses 24 geometry evaluations and at
most 864 coefficient reads. Cache stress touches 64 MiB outside each query timer.

The first complete preparation took **131.55 s**. An independent second run
from the same manifest produced the **identical asset SHA-256**, taking 143.41 s
as a standalone process with a sampled OS peak working set of **216,313,856 bytes**.
The memory observation includes offline contour construction and serialization.
No validation benchmark ran concurrently with that second measurement.

Native measured asset loading took 0.169 to 0.219 s. Browser WASM initialization,
local fetch, decode and policy/manifest fetch took 28.97, 140.68, 171.03 and
6.53 ms, respectively. These are local observations, not network deployment
estimates. Native replay peak working sets were 247,603,200 / 247,484,416 /
247,685,120 bytes, including diagnostics, serialization and stress storage.
Browser WASM memory capacity is 160,301,056 bytes; JS input and stress allocations
are additional. Capacity is not renderer RSS.

The [machine-readable record](geometry-145-operational-contour.json) and
[raw evidence archive](experiments/geometry-145-operational/contour-v1-evidence.zip)
preserve preparation/reproduction results, all three native reports, browser
report, reference traces, audit witnesses, derivative diagnosis and unchanged
test outcomes. Native default compilation and experimental native/Python/WASM
release builds pass. Reproduce with fresh output paths:

```powershell
rtk cargo run --release --manifest-path runtime-core/Cargo.toml --features experimental-operational-field --example operational_contour_prepare -- docs/validation/geometry-145-operational-contour-experiment.json tmp/fresh-contour.bin
```

Bind the existing fixed fidelity and reference-policy plans to the resulting
field ID, then use the retained validation commands from the earlier reports.
Keep the source declaration, field bytes and policy fixed for the entire replay.

The finite source eliminates the old sign counterexamples under this audit and
provides more verified reference crossings at bounded prepared-query cost. It
still fails scalar fidelity and crossing-time refinement. Adoption, certified
scalar bounds, reference differential fidelity, full reference-singularity
information and compatibility/deployment/training remain unresolved.
