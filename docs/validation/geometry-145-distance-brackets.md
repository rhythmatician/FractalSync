# #145 certified distance brackets and cached transport — 2026-10-26

PR #148 remains unready to merge. This experiment combines a certified interior
cover with an independently certified exterior witness, then reuses that evidence
for nearby scalar distance queries. It supplies intervals, not a distance value,
gradient, Hessian, nearest-foot certificate, or runtime field implementation.

## Independent bounds

The lower bound comes from the [closed attracting cover](geometry-145-closed-cover.md)
and its documented winding/fullness argument. `OutsideWitness` can only be
constructed after the existing high-precision critical-orbit proof certifies
escape for its exact represented parameter. Newton projection and normal or
tangential offsets suggest coordinates but cannot certify exterior membership.
The witness point and its escape proof remain private and immutable.

A segment from the interior query to a proved exterior point crosses the
Mandelbrot boundary. Directed endpoint distance therefore supplies an upper
bound at that same query. The new API retains the query and refuses invalid or
inconsistent intervals. Exhausted escape work is not exterior evidence.

Final frozen runs at 128-bit precision give these mathematical Mandelbrot
boundary-distance enclosures:

| Query | Period | Lower bound, approximately | Upper bound, approximately | Width, approximately |
| --- | ---: | ---: | ---: | ---: |
| Native high-period query | 85 | 6.95629739922484e-6 | 7.815207700024845e-6 | 8.58910300800005e-7 |
| Python refusal query | 154 | 1.7240936517760065e-7 | 1.7459681553337673e-7 | 2.187450355776077e-9 |

The native coordinates are `(0.32049988639807619, 0.04733531447355364)`;
the Python coordinates are `(0.35659381055433242, 0.06404656757776669)`.
The native point differs from the still-failing shipped crossing's earlier
`CutLocus` point. The best native witness escapes at update 9488; the best Python
witness escapes at update 23349. Neither proof extends the fixed 32768-update
budget or changes the pre-update escape convention. Decimal table values are
diagnostic renderings; proof comparisons use directed multiprecision values.

## Compact neighborhood certificate

`InteriorDistanceDomain` retains only an exact origin, its positive certified
clearance L, and an exact exterior witness. At another exact query q it computes

* lower: downward-rounded `L - upward_distance(q, origin)`, requiring positivity;
* upper: upward-rounded `distance(q, exterior_witness)`.

Distance to a closed set is 1-Lipschitz. Strict positive remaining clearance
places the transported query inside the original certified open disk, so both
bounds remain valid. Queries on or outside that disk refuse. The query is bound
to the resulting interval; no midpoint scalar or derivative claim is returned.
The expensive cover and orbit proof are not repeated during transport.

| Final run | Full construction wall time | 10,000 transported intervals | Mean per interval |
| --- | ---: | ---: | ---: |
| Native | 143.04 s | 0.1676 s | 16.76 microseconds |
| Python | 143.49 s | 0.1602 s | 16.02 microseconds |

The two cold runs ran concurrently. The local benchmark repeats 323 coordinate
combinations in a box with half-width `0.001 * L`, converted through f64 inputs.
It includes query conversion, bound evaluation and destruction. It excludes
cover construction, exterior search, trajectory coverage, classification,
geometry jets and cache replacement. These measurements demonstrate reuse of
scalar evidence, not a full live geometry-query budget. The cold cost remains
unsuitable for a runtime cache miss.

The numerical local analytic jets also pass their retained neighboring-gradient
and Hessian consistency checks at both queries. The witness search runs those
checks before constructing bounds. That establishes local model consistency;
the bracket does not certify the model as a globally nearest boundary branch.

## Verification and reproducibility

The isolated crate passes 49 tests. The retained 26 mutations and seven new
bracket mutations are caught by assertion failures, not compilation failures.
New coverage includes rejected interior/nonfinite witnesses, exact escape
cutoffs, immutable query binding, transported-bound refusal, irrational upper
distance against a 512-bit reference, diagonal travel near the clearance
boundary, and tiny diagonal motion requiring downward clearance subtraction.
The last two tests independently catch the two rounding-direction mutations.
All mutation scripts restore original source bytes in `finally` blocks.

Build and run from the repository root:

```text
cargo test --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision
cargo build --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision --bin component_probe
target/geometry145-precision/release/component_probe bracket-native 0.999999 128 0.9 0.05
target/geometry145-precision/release/component_probe bracket-python 0.999999 128 0.9 0.05
```

The accompanying JSON records all initial and final terminal runs, source and
binary hashes, mutation results and the final test output. Two additional
`cfg(test)` regressions were added after freezing the successful executable;
the certification and benchmark functions were unchanged. A rebuilt release
binary is compared after normalizing PE timestamps and CodeView identifiers.

## Remaining merge work

These bounds concern the mathematical quadratic family at represented
parameters. Equivalence to the canonical finite floating orbit is unproven.
The runtime still needs a coherent field with stable derivatives, justified
classification, acceptable cold and warm costs, and successful original native
and Python crossings and live training. Thresholds and crossing tests are
unchanged. This experiment adopts no production provider change, so it does not
clear the existing failures recorded in the closed-cover report.

The next useful implementation work is to tighten exterior witnesses and
adaptive cover accuracy to a requested interval width, then connect reusable
certificates to a coherent nodal/analytic field. Averaging query-dependent
interval endpoints would not establish that field or its derivatives.
