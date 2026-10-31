# Geometry #145: distinguish source sensitivity from event quantization

October 30, 2026. Direct comparison of the two frozen contour assets finds
1,099 of 1,768 fixed samples outside the retained quarter-scale comparison
budget. A separate timestep sweep shows convergence of bracketed operational
zero times despite identical sampled crossing times. Neither result changes
the original tests, runtime quality metadata or merge verdict.

This follows the [reference-grid experiment](geometry-145-operational-reference-grid.md).
That experiment removed two sign-based counterexamples but did not establish
scalar-distance fidelity. The new executable comparison tests the missing
source-stability question directly, independently of derivative self-consistency.

## Frozen comparison

The [comparison plan](geometry-145-operational-source-refinement-plan.json) pins
coarse asset `b45293bd1d6a1bd6` and refined asset `37ec6365e602ae31`. Their source
coarse spacings are 0.002 and 0.001. Both use coefficient spacing 0.0005 and final
contour subdivision spacing 0.000125. Neither asset is modified.

The fixed points comprise the original 1,000-point neighboring grid, the existing
765-point scalar-audit generator, and three named controls at the quality probe,
crossing start and ridge start. No sample is removed because of its result.
Every point is available in both assets.

For a declared coarse source spacing `h_source`, the comparison measures

```text
e_D = abs(F_f - F_c)
e_g = h_source * norm(gradient_f - gradient_c)
e_H = h_source^2 * Frobenius(Hessian_f - Hessian_c)
source_sensitivity = max(e_D, e_g, e_H)
comparison_budget = 0.25 * 0.1 * hypot(F_f, 1e-4)
```

These terms have distance units, as in the established numerical-convergence
estimator. However, `h_source` is a reference coarse-grid spacing, not the
unchanged coefficient spacing or a claimed resolved scale. This pair does not
measure coefficient-resolution convergence, fit residual, membership accuracy,
a reference-distance upper bound, or quality throughout a region. The report
labels it measured source-refinement sensitivity. It is not assigned to the
legacy `estimated_error` field.

| Fixed sample group | Samples | Outside comparison budget |
| --- | --- | --- |
| Original neighboring grid | 1,000 | 381 |
| Existing scalar-audit points | 765 | 717 |
| Quality probe `(0.3, 0.05)` | 1 | 0 |
| Original crossing start `(0.4, 0.05)` | 1 | 1 |
| Ridge start `(0.6, 0.15)` | 1 | 0 |
| Total | 1,768 | 1,099 |

At the crossing start the scalar changes from `0.0541637737` to `0.0460293755`.
The absolute difference is `0.0081343982`, against comparison budget
`0.0011507371`. The barrier changes from `0.9081186906` to `0.8846414463`, and
the original test's energy-derived launch speed changes from `0.2163846919` to
`0.2557097396`. Its 2.649-to-2.044-second crossing-time change therefore cannot
be interpreted as an integration comparison with identical initial velocity.

At the quality probe, sensitivity is only `3.386e-6`, below budget `4.308e-4`.
Using that point alone to assign a small general quality estimate would hide
the corridor failures. The largest absolute sensitivity is `0.0340303854`,
dominated by scaled Hessian change at `(0.3384152457, 0.0516211604)`. The largest
sampled sensitivity-to-budget ratio is about 5,996. Different samples attain
these two maxima.

This rejects the inference that passing analytic derivatives and a sign audit
establishes stable source geometry. It does not identify which source is closer
to the true distance at every failed point. Both still use finite-orbit evidence.

## Crossing-time diagnostic

The existing integration diagnostic now also accepts an explicit
`Rk4FourSubsteps` method plan. Its previous Euler/RK4 plan remains supported.
The [new frozen event plan](geometry-145-operational-event-experiment.json)
uses the refined asset, original launch construction and the existing five
timesteps. No field, profile or coefficient replacement occurs during a run.

| Timestep | First negative sample | Bracketed segment zero | Maximum energy excursion |
| --- | --- | --- | --- |
| 0.002 | 2.044 | 2.043985546712 | `4.546e-5` |
| 0.001 | 2.044 | 2.043983781041 | `1.336e-6` |
| 0.0005 | 2.044 | 2.043983908753 | `7.280e-8` |
| 0.00025 | 2.044 | 2.043983907712 | `3.342e-9` |
| 0.000125 | 2.044 | 2.043983907809 | `7.233e-10` |

All five runs complete without refusal. Successive bracketed-time gaps are
`1.766e-6`, `1.277e-7`, `1.041e-9` and `9.668e-11` seconds. This supports
sample quantization as the reason the original strict sample-gap assertion
fails even as these diagnostics converge. A segment zero is for the committed
piecewise-linear operational path, not the exact ODE solution or a certified
reference Shore event. It does not replace the retained assertion.

The full original Python suite remains **25 passed, two failed** on this asset:
sampled crossing-time refinement and unknown scalar-quality metadata. The
source comparison adds evidence against inventing a small quality estimate;
the event diagnostic adds evidence against tuning Physics to move sample bins.

## Checks and reproduction

All **28 experimental Rust controls pass**, including the new source-comparison
control. It checks the mixed Hessian contribution to the Frobenius norm,
gradient scaling and refusal of nonfinite evidence. The executable also refuses
swapped asset identities and an altered fixed-control plan. Default native
compilation, formatting and diff checks pass.

```powershell
rtk cargo test -p runtime_core --features experimental-operational-field operational_field --lib
rtk cargo build -p runtime_core --release --features experimental-operational-field --example operational_source_refinement --example operational_integration_diagnostic
rtk proxy target/release/examples/operational_source_refinement.exe tmp/geometry145-contour-v1.bin tmp/geometry145-reference-grid-v2.bin docs/validation/geometry-145-operational-source-refinement-plan.json
rtk proxy target/release/examples/operational_integration_diagnostic.exe tmp/geometry145-reference-grid-v2.bin tmp/geometry145-reference-grid-v2-policy.json docs/validation/geometry-145-operational-event-experiment.json
```

The [machine report](geometry-145-operational-source-refinement.json) records
source/binary/asset hashes, every event run, refusal controls, maxima and group
counts. The [raw archive](experiments/geometry-145-operational/source-refinement-evidence.zip)
contains all 1,768 point comparisons and unabridged diagnostic output.
Original asset reproduction and latency evidence remain in the preceding report.
This follow-up adds offline verification only; it adds no runtime work, changes
no shipping math and does not require new goldens or controller versions.

PR #148 remains unready. Source stability and supported quality claims must be
resolved across the prepared corridor, not inferred from one passing point.
The event assertion remains an independent retained gate. ADR 0005 shipping
adoption and reference-cut motion policy are still pending.
