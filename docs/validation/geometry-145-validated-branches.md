# Validated local periodic branches

This experiment advances branch control for the coherent component-distance representation. It is not adopted by production and does not establish merge readiness. The Physics thresholds and original crossing tests are intact.

## What is proved

For the two complex equations

```text
F(z,c,mu) = f_c^p(z) - z
G(z,c,mu) = (f_c^p)'(z) - mu
```

`validated_branch.rs` proves a unique root within a specified error polydisk for every multiplier in a supplied complex disk. It retains three complex affine dependencies: multiplier displacement, initial-state error, and parameter error. Directed-rounding disks enclose nonlinear terms and all center/coefficient arithmetic errors.

A rounded inverse is used as a fixed exact dyadic preconditioner. Acceptance requires both strict self-inclusion of the preconditioned residual map and a weighted-infinity contraction bound below one. The proof comparisons use outward high-precision bounds; diagnostic `f64` conversions do not determine acceptance.

The certificate privately retains its exact predictor, requested domains, and proven inclusion bounds. The certified period is private proof state. It cannot be changed through diagnostic metadata.

`glue` evaluates both predictors at one exact shared multiplier with outward arithmetic bounds. It proves the old root enclosure lies in the next patch's uniqueness domain. When needed, it advances the old periodic cycle through validated state disks before checking containment. This establishes cycle identity across a join; successful numerical Newton solves alone are not treated as identity evidence.

`attracting_segment` requires both endpoint multipliers to lie in the certified patch disk and strictly inside the unit disk. Convexity covers their straight segment. Uniform contraction supplies a continuous local root branch. The checked iterate count need not be the least cycle period.

## Recorded crossing-query arcs

Each recorded case requires a point certificate, four nonzero multiplier-disk certificates (radii `1e-8`, `1e-6`, `1e-4`, `1e-3`), two attracting segments, and a proved join. The segments use multipliers on the radius-`0.999999` circle. Their endpoints are exact dyadic values formed from the diagnostic trigonometric results; no exact transcendental-circle assertion is needed for the endpoint norm checks.

At disk radius `1e-3`, the native period-85 certificate has contraction approximately `0.005887`, with parameter-error inclusion about `1.82e-9` inside the requested `1e-8` domain. The Python period-154 certificate has contraction approximately `0.013029`, with parameter-error inclusion about `1.73e-10`. Both joins use phase advance zero and retain positive state and parameter margins. The period-two control independently requires and checks a phase advance of one.

The initial native neighborhood request was too small to contain its calculated parameter-error bound. Enlarging the root enclosure made that request provable; the strict acceptance inequalities were preserved. These root-domain choices are not Physics acceptance thresholds.

## Verification

The isolated crate has 28 passing tests across two executables. Shared complex-ball arithmetic is extracted into one module, so membership tests are not duplicated between executables.

The new proof checks include independent inclusion-only and contraction-only refusals, exact closed-form cardioid roots at multiple disk perturbations, nonfinite inputs in every complex component, invalid and subnormal radii, period limits, singular period-two branch refusal, known cycle-phase changes, attracting-segment guards, and refusal to join distinct period-three components.

Ten new source mutations must fail their targeted tests: removing self-inclusion, contraction, operation roundoff, nonlinear remainder, remainder transport, gluing parameter coverage, gluing parameter identity, cycle-phase containment, attraction, or segment coverage. Compilation failures do not count as caught mutations. The preceding nine mutation checks are retained and rerun.

The first single roundoff fixture did not catch omitted rounding. The strengthened test checks a family of exact 128-bit dyadic inputs against the closed-form root evaluated exactly at 512 bits. It rejects requests whose parameter-error domain is smaller than the independently established center error. Removing operation roundoff now makes that test fail.

Independent mathematical review found no enclosure or contraction defect. Review found mutable period metadata could disconnect proof operations from the certified period; that API flaw is fixed by private proof state.

## Reproduction

From the repository root:

```powershell
rtk cargo test --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision
rtk cargo run --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision --bin component_probe -- branch-cardioid
rtk cargo run --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision --bin component_probe -- branch-native
rtk cargo run --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision --bin component_probe -- branch-python
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-branch-mutations.py
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-component-mutations.py
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-point-proof-mutations.py
```

Run mutation scripts sequentially, without concurrent builds of this crate. They normalize newlines only in temporary mutations and restore original source bytes in `finally` blocks. The branch mutation script optionally accepts mutation names for focused diagnosis; the default runs all ten. Unknown names and empty selections reject; both unknown-only and mixed known/unknown negative checks preserve the source bytes.

## Remaining acceptance work

These are local attracting branches, not a complete boundary cover. The next step is to assemble a closed, validated curve with guarded joins and usable coordinate/distance enclosures around the query. This needs global coverage and nearest-feature/cut-locus control before it can supply trustworthy distance capability. A local branch certificate alone does not prove Mandelbrot boundary identity, least period, global nearest-boundary uniqueness, or the distance jet's approximation error.

Production integration, live query cost, the original crossing and training gates, full derivative/energy validation, dependency/WASM review, version and golden-vector updates, and project-wide validation remain required. The previous numerical component jets and production gate evidence are preserved in [geometry-145-component-field.md](geometry-145-component-field.md).

## Current production gate

The original native crossing test was rerun against production on October 26. It still fails at the same CutLocus point `(0.21474166530729011, 0.12774018997553147)`: zero passed, one failed, 94 filtered out, 1.85 seconds in the test. The full output is preserved in the JSON report. Local branch certification has not yet been integrated into that path.
