# Component-distance field experiment

This experiment is not adopted by the production provider. Thresholds and original crossing tests remain unchanged.

The retained-node probe identified one late-escaping contour-stability blocker at `(0.3204946517944336, 0.04734092950820923)`. Its exterior certificate needs update 32,747. The budget-aware field query ultimately timed out at 1,200.28 seconds.

## Box enclosure results

The high-precision affine enclosure retains linear parameter dependence and outward roundoff bounds. Its quadratic counterpart also retains the second parameter coefficient. Both use strict periodic trapping disks with parameter uncertainty included at each update. The shared point path passes zero parameter radius. Both box variants use the same directed-rounding primitives in `proof_support.rs`; their remainder recurrences remain distinct.

Affine certificates resolve native-query boxes at dyadic levels 24, 28, 32 and 40, and Python-query boxes at levels 32 and 40. The quadratic variant certifies the same tested levels and is slower in this diagnostic; it is not selected as an optimization. Neither resolves the sampled boxes containing the late escaper.

Point witnesses alone give a native distance interval approximately `[5.33e-6,8.01e-6]` at tolerance `1e-5`, but the tighter `1e-7` and `1e-9` requests hit the 8,192-region work limit. Adding up to 256 box certificates and 16 point certificates times out at 180 seconds before completing the first query. The point-only rows share one cache and a 64-call cap; tighter rows are not fresh-cap runs. These searches do not provide usable field coefficients. Their complete observed outputs and budgets are preserved in the JSON report; they are exploratory wrappers around the preceding reconstructed distance oracle.

## Coherent local component geometry

The numerical boundary experiment solves the periodic-orbit and multiplier equations together, using 128-bit arithmetic and damped Newton continuation. It traces 512 multiplier angles at radii approaching one, then on the unit circle. Angles avoid the root at multiplier one. A local nearest-angle refinement supplies the distance; implicit first and second derivatives supply its gradient and Hessian from the same curve.

| Query | Cycle period | Local interior distance | Gradient discrepancy | Distance-scaled Hessian discrepancy |
| --- | ---: | ---: | ---: | ---: |
| Native crossing | 85 | 7.642526802481138e-6 | 7.3813e-9 | 3.1850e-9 |
| Python crossing | 154 | 1.728681341973857e-7 | 1.4844e-8 | 9.5845e-9 |

The earlier period-308 trapping certificate remains valid: it traverses the Python query's period-154 cycle twice. This is not a contradiction or a claim that the trapping heuristic always finds the least period.

The derivative discrepancies compare the analytic jet against independently projected neighboring queries. They are local numerical checks, not the project's full derivative validation. Fifteen tests across two executables pass, including closed-form cardioid derivatives, the period-two circle distance jet, strict cutoff semantics, box uncertainty, and cut-locus/exterior refusal controls. All nine source mutations fail their targeted tests; compilation failures are excluded from mutation success.

Both traces complete: approximately 29.56 seconds native and 63.38 seconds Python in the recorded runs. These cold times include orbit settling, several full curves, local minimization and neighboring-query checks. They do not establish acceptable live query cost.

## Reproduction

From the repository root:

```powershell
rtk cargo test --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision
rtk cargo run --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision --bin geometry145_precision_probe -- boxes
rtk cargo run --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision --bin geometry145_precision_probe -- quadratic-boxes
rtk cargo run --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision --bin component_probe -- native
rtk cargo run --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision --bin component_probe -- python
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-component-mutations.py
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-point-proof-mutations.py
```

Mutation scripts restore source bytes in a finally block. Run them sequentially, without concurrent builds of this experimental crate. The component executable returns nonzero for unknown selections or refused traces. Nonfinite derivative discrepancies refuse instead of disappearing through `max`.

## Work before adoption

The next requirement is a validated boundary cover and branch control. Sampled continuation plus a local Newton minimum does not prove global nearest-boundary uniqueness or exclude branch switching. A periodic cycle can change phase during a multiplier loop even when its parameter coordinate closes; cycle phase closure is not used as a boundary certificate.

A production component representation must supply distance/error capability, retain genuine cut-locus refusals, support component identity and caching across neighboring queries, and meet live query cost. Then it needs coherent-field integration, the original crossing and training gates, full derivative/energy validation, version and golden-vector updates, dependency/WASM review, and the full project checks. This experiment does not establish merge readiness.

## Production gate rerun

The original native `manifold::tests::controls_drive_can_cross_shore_without_wall` test still fails after rebuilding production: zero passed, one failed, 94 filtered out, 2.54 seconds in the test. It reports the same CutLocus refusal at `(0.21474166530729011, 0.12774018997553147)` with provider version 6. The command and complete output are preserved in the JSON report. The numerical component prototype has not changed this production result.
