# #145 reusable covers and conservative norm cost

PR #148 remains unready to merge. The experiment now reuses a validated closed
attracting curve for new queries without repeating periodic-root proofs or cycle
joins. It also reduces the cost of intermediate conservative norm bounds.
Production geometry, acceptance thresholds and original crossing tests are unchanged.

## Query reuse

`ClosedCover::clearance_at` recomputes every edge and connector separation bound
and the winding for the new exact query. The immutable cover retains the
previously proved closure, attraction and cycle identities. A private
`CurveClearance` binds its lower bound to that query. Pairing it with a separately
proved escaping endpoint supplies a scalar distance bracket.

This is stronger than transporting one clearance through a small interior ball.
It still refuses intersecting capsules, ambiguous orientations, zero winding and
nonfinite queries. A new fixture starts with a valid closed cover, then checks
that a second query separated from every edge nevertheless refuses because it
intersects a connector. A separated exterior query independently requires the
zero-winding refusal. Query substitution has its own mutation check.

## Conservative intermediate norms

The pinned AstroFloat representation exposes the normalized binary exponent and
mantissa words. The shortcut truncates the leading mantissa bits to binary64,
takes the successor to enclose the discarded tail, and rounds each square, sum
and square root upward using successors. The exponent guard keeps those
operations normal and far from overflow. Unsupported layouts, exceptional values
and exponents outside the guard retain the multiprecision path. Exact axes also
retain multiprecision rounding.

The square-root step relies on Rust's documented correctly rounded
[`f64::sqrt`](https://doc.rust-lang.org/std/primitive.f64.html#method.sqrt).
Tests compare against independent 512-bit bounds, cover both 32-bit and 64-bit
word layouts, include discarded mantissa tails, and exercise both guard endpoints
and their immediate rejected neighbors. They do not constitute execution on a
32-bit target.

The shortcut is used for conservative intermediate upper bounds. Final endpoint
distances retain the original multiprecision norm. Applying the shortcut there
initially failed the retained `1e-35` accuracy test; the implementation was scoped
back, and the test and tolerance remain unchanged.

## Frozen measurements

The native and Python runs below used the same executable and ran concurrently.
Each requested-width check passes at the original anchor.

| Query | Edges | Requested width | Proved width | Cold wall time |
| --- | ---: | ---: | ---: | ---: |
| Native, period 85 | 248 | 5e-8 | 4.9181703949299435e-8 | 25.98 s |
| Python, period 154 | 173 | 5e-10 | 4.706751315129253e-10 | 25.97 s |

The preceding frozen executable took about 184 seconds at those requests, under
a heavier concurrent workload. These are observed times, not a controlled speedup
ratio. A separate norm microbenchmark takes 0.1159073 seconds for 10,000 precise
norms and 0.0008779 seconds for 10,000 conservative shortcut norms. That measures
the helper only, not total field-query speed.

After adding the remaining tests and rerunning mutations, a second frozen
executable repeats both high-period runs successfully in 29.40 seconds each.
Bounds and acceptance counts match the table. Its whole-cover times are
0.9395554 and 0.5881129 seconds per 100 queries. Both executable hashes and
all six terminal benchmark outcomes are retained in the JSON record.

Each reusable-cover benchmark uses a precomputed 10-by-10 query grid with box
half-width `1.25 * original_clearance`. Query conversion and compact-ball
eligibility checks occur before timing. Times include both successful and refused
whole-cover queries. The scalar work remains linear in the number of edges.

| Query | Accepted / 100 | Accepted beyond transport ball | Within requested width | Whole-cover query time / 100 |
| --- | ---: | ---: | ---: | ---: |
| Native | 85 | 41 | 0 | 0.6873484 s |
| Python | 86 | 42 | 0 | 0.4731130 s |
| Cardioid control | 83 | 39 | 10 | 0.2410725 s |

The retained tiny-ball benchmark separately accepts 10,000 queries in
0.2089260 seconds for native and 0.1846478 seconds for Python. It covers a much
smaller region and does not replace the whole-cover measurement.

## What the measurements leave unresolved

The isolated crate passes 66 tests and catches all 49 retained and new mutations
by assertion failures. The new mutations remove mantissa-tail rounding,
square-root rounding, reused-query binding, reused-query winding, and independent
connector separation. Independent specification, health and standards reviews
found no material defect in this bounded experimental change.

Both original production crossing tests were rerun. Native still fails with
`CutLocus` at `(0.21474166530729011, 0.12774018997553147)`. Python still fails
with `Unresolved` at `(0.35659381055433242, 0.06404656757776669)` after a separate
successful extension rebuild. The fixture's documented `RUNTIME_CORE_NO_BUILD`
switch prevents rebuilding an already loaded Windows extension during pytest.
The accompanying JSON records the extension hash, terminal failures, benchmark
output, source and frozen-binary hashes, and every mutation result.

The larger region now has reusable membership and scalar bounds, but neither
high-period grid has a query meeting its requested width with the single retained
exterior witness. Exterior-witness coverage and interior enclosure accuracy must
improve before these bounds can supply a whole tile of accurate node values.
Cold construction at 26 seconds also remains unsuitable for a runtime miss.

These are mathematical Mandelbrot distance bounds. Equivalence to the canonical
finite floating orbit, nearest-branch classification, and certified derivative
errors remain unproved. A coherent value/gradient/Hessian field, original crossing
success, energy behavior, live training validity, full query cost, and the required
version/golden/parity/browser/CI validation remain merge requirements. No
production replacement or merge-readiness claim follows from these experiments.
