# Bounded refinement and canonical nodes

These are unshipped experiments based on production commit `d8ec37a`.
They do not make PR #148 ready to merge. Acceptance thresholds and both original
crossing tests are unchanged.

Apply these patches in order to reconstruct the latest candidate:

1. `experiments/geometry-145-component-nodes.patch`
2. `experiments/geometry-145-scale-component-levels.patch`
3. `experiments/geometry-145-bounded-selector.patch`
4. `experiments/geometry-145-shared-interval.patch`
5. `experiments/geometry-145-canonical-nodes.patch`

The complete composition was applied in an isolated Git directory and checked
against all three measured source files. The twelve-case energy sweep belongs
to the first three patches only. Do not attribute that result to the later
canonical-node variant.

## What changed

An outward interval bound encloses every possible coefficient in a coarse
quintic field. A known cardioid boundary point supplies an upper bound on the
true boundary distance, without requiring stationary-root completeness. Clamped
node values stay inside that bound. The interval calculation follows the actual
scalar evaluation, including de-trending, coefficient correction and affine
restoration.

Only a bound that forces the existing size gate to its constant-one plateau can
discard an uncertified coarse model. Uncertain retained finer dependencies still
prevent certification. Skipped models cannot supply raw convergence evidence.
Disabling this certificate makes its regression test fail.

The shared-interval patch reuses the existing box arithmetic for both providers.
The canonical-node patch removes query-centered source reuse. Retained nodes
then depend on their own coordinates and starting level. It also checks every
exactly tied nearest projection when excluding unstable membership nodes. A
regression reverses segment order and requires rejection in both orders.

## Measured results

- The bounded-selector variant passed all twelve energy cases in 1091.97 seconds.
  Its rollout metrics match the earlier size-only component-field experiment.
- The shared-interval variant passed twelve derivative probes in 15.28 seconds.
- The canonical-node variant passed 26 targeted tests. Its twelve derivative
  probe centers match the shared-interval variant exactly, but evaluating the
  report took 77.57 seconds. See the JSON for the checked rerun.
- The former driven refusal at `(0.32046083605181780, 0.04736134021833255)`
  passes an isolated query with the shared-interval variant.
- Both full native crossing attempts timed out at 1200 seconds. The traced run
  progressed beyond the former refusal to
  `(0.32049988639807619, 0.04733531447355364)` before timing out at a fine contour
  level. A timeout is not a passing crossing or a new classification result.
- The recorded Python crossing query timed out at 180 seconds. The earlier
  provider suite retained its training-point failure.

## Root-free distance enclosure

`experiments/geometry-145-certified-cardioid.rs` is a separate, unintegrated
global-distance enclosure. Two rational circle charts cover the cardioid.
Interval endpoint bounds and a second-derivative bound of 10 enclose each curve
piece by its chord with error at most `10 * width² / 8`. A priority queue retains
coverage of every unpruned interval until the global upper/lower gap reaches the
requested tolerance. Exhaustion returns no certificate.

Its six standalone tests pass, including invalid inputs, budget exhaustion,
known endpoint distances and an underflowing nonzero chord. After applying all
five patches, reproduce with:

```powershell
rustc --edition=2021 --test -O docs/validation/experiments/geometry-145-certified-cardioid-probe.rs -o tmp/cardioid-certificate-tests.exe
tmp/cardioid-certificate-tests.exe
```

This enclosure does not yet certify exclusion of competing nearest features.
The experimental provider still uses approximate stationary roots in that path.
Those roots cannot establish completeness or a numerical error bound. Integrating
the distance enclosure and a complete feature-exclusion certificate remains a
requirement before adopting the component shortcut.

Full crossing completion, training validity, cache-context validation and query
cost remain open. Any production adoption also needs version/golden updates and
the complete Rust, Python, parity and architecture gates.
