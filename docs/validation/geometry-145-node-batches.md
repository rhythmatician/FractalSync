# #145 distance bounds at neighboring dyadic nodes

PR #148 remains unready to merge. This experiment extends the reusable-cover
checkpoint at `2f8b268` to accuracy requests at multiple neighboring nodes. It
does not replace the production provider or establish a coherent interpolated
field. Thresholds and original crossing tests are unchanged.

## Diagnosis and change

The preceding broad-query benchmark reused one exterior witness and a cover
refined at one anchor. No broader high-period query met its width request. To
separate witness error from cover error, this probe visits a 3-by-3 grid of
canonical dyadic coordinates near each anchor and searches independently proved
exterior witnesses at each node. The steps are `2^-20` for native and `2^-27` for
Python. These small grids are not the production quintic stencil.

Refreshing each node's exterior witness alone meets the request at 3/9 native
nodes and 1/9 Python nodes. The remaining lower-bound gaps require cover
refinement. `build_requested_at` now accepts up to 1,024 exact query/clearance
pairs. Candidate arcs and their preceding connectors must meet every target.
After closure, each query independently repeats separation, winding and the
clearance check, including the closing connector. A partial prefix or partial
target success cannot return a certified cover.

Each target is the smallest independently proved endpoint-distance upper bound
across the retained witness pool, minus the unchanged width request, rounded
upward. Final brackets are reconstructed for each exact node and must pass
`require_width`; construction targets do not substitute for this check.

The Python batch succeeds with the existing curve proposal. Native initially
refuses at depth 24. Logging identifies a query whose demanded clearance lies
above the distance to the proposed root-dented curve. Narrowing the dent from
0.05 to 0.025 still refuses. Reducing its depth instead, with root radius 0.99
rather than 0.9, produces a valid nine-node cover. The inclusion, attraction,
gluing, separation, winding, precision and work limits remain unchanged.

The successful native construction has 537 edges and the Python construction
has 179. Their batch-refinement runs take about 63 and 30 seconds respectively,
under different concurrent loads. These are construction costs, not live-query
costs. Each diagnostic also constructs its original anchor cover and searches
witnesses before the batch pass. Work grows with both regions and target count;
a region count alone does not describe batch cost.

## Stable scalar proposals

Choosing the midpoint of an enclosure can change a node value when a different
batch produces a different enclosure. Two bounded alternatives are now checked.

`dyadic_sample` proposes a numerical bin, verifies exact binary64
representability, and accepts only when the entire certified interval lies in
one half-open dyadic rounding bin. Any accepted result for the same distance and
quantum is unique regardless of cover or witness selection. The native grid
certifies 4/9 such samples and Python 6/9. The remaining bins refuse; their values
are not inferred or smoothed.

`certifies_value` separately checks an independently supplied numerical value.
Both endpoints of the certified distance interval must fit inside
`value +/- allowed_error`, with inward-rounded acceptance boundaries. In this
diagnostic, a node's periodic numerical projection is a fixed function of its
query, period and fixture seed, computed before batch refinement or pooled
witness selection. The final diagnostic requires this error check at every
node. This certifies the proposal's scalar accuracy, not its derivatives or a
generic runtime branch-discovery rule.

## Verification and limits

The isolated crate passes 69 tests. All 55 mutations are caught by assertion
failures. New cases protect multi-query refinement, the independent final target
gate, both rounding-bin boundaries and both candidate-error endpoints. The final
target test refuses when one target succeeds and a later target fails. The
retained upper-distance precision test and crossing tests are unchanged.

Read-only specification, health and standards reviews found no material defect
in the intended change. Source restoration and the final executable are checked
after mutation runs. The accompanying JSON retains terminal successes and
refusals, full benchmark output, stderr hashes, source/binary hashes and mutation
results. Superseded refusals remain evidence rather than being counted as passes.

Reproduce the final successful diagnostic variants with the release
`component_probe` executable:

```text
component_probe tight-bracket-native .999999 128 .99 .05 5e-8 9.5367431640625e-7
component_probe tight-bracket-python .999999 128 .9 .05 5e-10 7.450580596923828e-9
```

Nine accurate scalar brackets are progress toward a reusable node oracle. They
do not prove the full stencil, C2 interpolation, derivative accuracy, cut-locus
classification, or equivalence to the canonical finite floating orbit. The
fixed fixture seed is not a general runtime component-selection algorithm.
Construction and witness costs remain too high for live misses. Production
crossing, energy, training, browser/WASM, parity/golden and CI requirements remain
open; the latest original production crossing reruns remain the failures recorded
in `geometry-145-cover-reuse.json`.
