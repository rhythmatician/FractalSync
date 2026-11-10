# Geometry #145: isolate raster defects and verify the native reference crossing

Latest follow-up: [the separately frozen finite-contour experiment](geometry-145-operational-contour.md)
has two different scalar counterexamples and a crossing-time refinement failure.
The raster-backed asset and source diagnosis below remain preserved.

October 30, 2026. This follows the [frozen browser/fidelity replay](geometry-145-operational-followup.md).
The field bytes, Physics profile, original tests and thresholds are unchanged.
No new asset was constructed. ADR 0005 remains authorized for a bounded prototype
only, and PR #148 remains unready.

## Source diagnosis

The prior audit proved 43 scalar-budget violations in the larger asset. The
[fixed source-audit plan](geometry-145-operational-source-experiment.json) now
compares those same 765 points against the raw source raster and against four
coefficient spacings. It uses the same quintic evaluator and conservative
reference membership. There is no online refinement or point-specific correction.
Diagnostic local stencils are not rollout assets.

The hypotheses were source-raster error, reconstruction smoothing error, or a
sign-convention mismatch. Deep inside/outside sign controls pass. A complete
offline stencil at the current spacing reproduces the actual loaded asset within
`3.544256901855114e-16`, against an explicit `1e-12` reproduction limit. This
checks that the probe isolates the actual reconstruction rather than another fit.

| Source/reconstruction | Certified scalar-budget violations |
| --- | --- |
| Raw 1024 by 1024 source raster | 60 |
| Existing frozen asset, spacing 0.0005 | 43 |
| Diagnostic reconstruction, spacing 0.001 | 14 |
| Diagnostic reconstruction, spacing 0.0005 | 43 |
| Diagnostic reconstruction, spacing 0.00025 | 58 |
| Diagnostic reconstruction, spacing 0.000125 | 60 |

All 43 current failures already fail in the raw source. Finer reconstruction
converges toward that defective source and restores more of its sign errors.
Increasing coefficient density is therefore not a remedy for these failures.
The coarser diagnostic's smaller count does not establish fidelity: the audit
only detects errors whose returned sign disagrees with a conclusive realm. More
smoothing can hide those errors while introducing others. No coarser field was
selected to optimize this count.

The embedded raster's spacings are approximately 0.002416 by 0.002194 world
units. Its metadata explicitly describes iteration-limited membership, possible
near-boundary errors and blended exterior DEM data. This experiment identifies
the defective input stage, not which raster-generation operation caused each
error. Replacing it requires better offline reference data and a fresh fixed
preparation. Certifying or upsampling the present coefficients cannot repair the
observed failures. No proof expansion or changed fidelity contract was adopted.

## Reference crossing of the original native driven trajectory

The native verifier previously hardcoded `reference_crossing: unverified`.
It now records every successful endpoint returned by the existing Rust Controls
integrator through a diagnostic wrapper. The original test body is still copied
verbatim and checked against the same fixed checkpoint. No assertion, loop,
control, timestep, crossing stop or returned mechanical result changes.

Reference classification runs only after the original test and all its motion
have finished. The rollout remains under field `068e8e016b4ad6b7` and profile
`b61c1e0bbba796e4`, with the same frozen authoritative refusal at `(-1, 0)`.
The wrapper records only successful returns. Each trace starts at its actual
initial position; separate undriven/driven traces retain their own times.

The original body passes, with the same 13,750 geometry evaluations. Its SHA-256
remains `e1afb878f44a2644f1efbcf97a7b94a6e7b8f56c798dd59e41f0dfb6af340895`.
The undriven run has 400 accepted steps and supplies no opposite-realm pair.
The driven run has 182 accepted steps and supplies this conclusive ordered pair:

| Mechanical time | Accepted position | Reference evidence |
| --- | --- | --- |
| 0 seconds | (0.3, 0.05) | Certified inside |
| 2.92 seconds | (0.3211973028902156, 0.048080168058688866) | Certified outside |

These endpoints establish a reference Shore crossing on the continuous
piecewise-linear reconstruction of this original driven trajectory. They do not
claim the exact ODE trajectory or certify all intermediate points. The verdict is
separate from its passing operational crossing. No sign of F or measured raster
discrepancy was substituted for membership evidence.

This verifies one previously unexamined native reference crossing. It does not
change the separate 35 fixed operational runs' unverified reference verdicts,
establish reference derivative fidelity, or cure the 43 scalar-budget failures.
No additional motion or continued run beyond the original crossing stop was used.

## Evidence and checks

The [machine-readable summary](geometry-145-operational-source.json) records
source hashes, reproduction error, source counts, both trace outcomes and the
reference pair. The [compressed evidence](experiments/geometry-145-operational/source-and-crossing-evidence.zip)
preserves every source-study value and every accepted trace endpoint with its
reference evidence. The original asset SHA-256 remains
`d77495871dd3fd9d0b66876ffe9148bade43234227ca4eb918a56d234069925e`.

The experimental source diagnostic builds and runs successfully. The native test
body remains unchanged and passes under the instrumented setup. All 18 focused
Rust controls still pass. No runtime evaluator, Physics implementation, Python
test body, WASM binding or shipping profile changed in this follow-up; prior
browser timing/parity and the 26-pass/1-fail Python result retain their stated
source hashes and scope.

```powershell
rtk cargo run --release --manifest-path runtime-core/Cargo.toml --features experimental-operational-field --example operational_source_audit -- tmp/geometry145-coverage-v1.bin docs/validation/geometry-145-operational-source-experiment.json
rtk proxy .venv/Scripts/python.exe scripts/validate_operational_native_crossing.py tmp/geometry145-coverage-v1.bin tmp/fresh-native-reference-trace.json tmp/geometry145-coverage-v1-reference-policy.json
```

The next preparation must improve the reference input rather than merely add
coefficients. The retained scalar-fidelity gate remains red. Broader reference
crossing evidence, reference differential fidelity, singularity coverage, shipping
adoption and compatibility/deployment/training gates remain open.
