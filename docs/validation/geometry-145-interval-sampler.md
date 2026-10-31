# Interval-bearing coherent sampler for #145

PR #148 remains unready to merge. This experiment connects the
[certified scalar samples and direction coverage](geometry-145-direction-cones.md)
to the coherent reconstruction's interval path. Production code, versions,
thresholds and original crossing tests are unchanged.

`NodalSample` retains a coordinate, the canonical signed value and an optional
finite enclosure. All 81 stencil positions now use the same sampler, including
the outer row and column. Missing samples and coordinate mismatches refuse
evaluation. Unknown sample uncertainty remains unknown; the fitter cannot
replace it with a singleton interval. The fitted model retains its `RangeJet`
and passes that range directly into hierarchy error propagation, avoiding a
later coordinate-based reinterpretation of sampled values.

Two oracle targets remain distinct:

- The existing scalar certificates enclose Mandelbrot distance with attracting
  covers and exterior witnesses. Their adapter preserves the signed numerical
  value and takes a conservative hull if that value lies outside the distance
  bracket. It never chooses a new value from the bracket midpoint.
- Stable canonical Shore samples enclose arithmetic for the represented
  polyline. Directed projection bounds include clamped endpoints, degenerate
  segments and the minimum over every segment. The existing signed value and
  sign remain unchanged. This is not an ideal-Mandelbrot distance certificate;
  membership stability, containment, coarse/fine contour disagreement, fitting
  and refinement still carry the separate discretization checks.

The first sampler revision left represented Shore arithmetic unknown and caused
four existing exterior-query tests to return `Unresolved`. Adding explicit
polyline arithmetic enclosures repaired those regressions without changing their
assertions or adding a new ideal-Mandelbrot acceptance requirement. The companion
JSON retains that failed trial and the final passing run.

At the two difficult actual query centers, the experiment now certifies the
original scalar windows and complete direction agreement. Their 81-node coherent
reconstructions retain the full nodal uncertainty:

| Query | Propagated nodal error | Original quality budget |
| --- | ---: | ---: |
| Native `(0.32049988639807619, 0.04733531447355364)` | `1.5027446077940745e-6` | `2.5072904006121766e-6` |
| Python `(0.35659381055433242, 0.06404656757776669)` | `1.6725944147264863e-8` | `2.5000037354212323e-6` |

The budget uses the existing `0.25 * requested_scale`, with `epsilon=1e-4` from
both original crossing configurations and comparison scale `2h`. These results
establish the nodal-uncertainty portion of a fixed-cell reconstruction. They do
not establish hierarchy closure, complete runtime query acceptance or successful
crossings. All 162 canonical nodal values remain bit-identical.

Final validation passes 111 provider tests in 187.12 seconds. The unfiltered
precision-enabled baseline passes 64 tests in 81.63 seconds; all 32 injected
faults are caught in additional replays. Three independent tests pass in 126.88
seconds, covering all 164 scalar witnesses and clearances. These timings do not
establish practical runtime query cost.

Added tests vary individual nodal values to
both interval endpoints and verify value, gradient and Hessian containment. They
also check missing/moved samples, invalid certificates, unproved values, retained
uncertainty despite equal numerical fields, Shore endpoints/minima/signs, and
nonfinite quality inputs. A review found that `error > budget` could fail open
for NaN; the gate now requires finite field components, finite nonnegative error,
a finite positive budget and `error <= budget`. An injected infinite-budget fault
is caught.

The focused runner compiles every Rust dependency from an immutable temporary
snapshot, records captured-byte hashes and rejects source drift before reporting
success. An isolated control changes a temporary source copy after harness
emission: the snapshot's unit test passes, but the runner correctly exits with a
source-drift failure. Neither the main checkout nor candidate source changes.
Failure cleanup stops the owned checker and its compiler descendants before
removing temporary files. A timeout control with a sleeping descendant and an
injected first-attempt tree-kill failure verify cleanup and retry on Windows.

The independent checker additionally replays both actual query centers, for 164
exterior witnesses and clearances in total. Its scope remains scalar/cover
verification, not independent reconstruction or cone-arithmetic replay. Single-
node endpoint tests do not replace an independent mixed-assignment check.

Apply `experiments/geometry-145-interval-sampler.patch` after the direction-cones
patch and its documented prerequisites. Reproduce with:

```powershell
rtk cargo test --release --jobs 1 --manifest-path tmp/geometry145-runtime-integration/runtime-core/Cargo.toml --target-dir target/geometry145-runtime-integration geometry_provider -- --nocapture
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-generic-component-check.py --source-root tmp/geometry145-runtime-integration --target-dir target/geometry145-runtime-integration --precision-target-dir target/geometry145-precision --emit-cover --output tmp/geometry-145-interval-sampler-complete.json
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-branch-hp-check.py --input tmp/geometry-145-interval-sampler-complete.json --target-dir target/geometry145-precision --output tmp/geometry-145-interval-sampler-independent.json
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-source-drift-check.py --source-root tmp/geometry145-runtime-integration --target-dir target/geometry145-runtime-integration --snapshot tmp/geometry-145-interval-sampler-complete.json --output tmp/geometry-145-sampler-drift-final.json
```

Still required: consolidate the generic proof backend into runtime-owned Rust,
construct bounded reusable derived tiles, feed those samples and complete
direction evidence through actual hierarchy/refinement queries, preserve genuine
cuts, and rerun the original crossings, derivative consistency, inward motion,
energy, live training and complete query cost. Existing experimental 4096-entry
node/component caches are not an adopted replacement for the 64-derived-tile
contract. No generic Regular-jet shortcut has been installed.
