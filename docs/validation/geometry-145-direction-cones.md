# Complete boundary-direction coverage for #145

PR #148 remains unready to merge. This unshipped follow-up to
[certified scalar windows](geometry-145-certified-scalar.md) proves sufficient
agreement of all boundary directions within a supplied query radius. The
precision-enabled experiment certifies all 162 canonical nodes, retaining the
original distance windows, 30-degree angle threshold, `0.5h` tie band and
32,768-update escape cap. Production source and original crossing tests are
unchanged.

The closed attracting curve lies inside the Mandelbrot set. Its winding is zero
outside the set, by fullness. A segment from a query with certified nonzero
winding toward any boundary point must cross the curve; exterior points
approaching that boundary point give the same conclusion by compactness. Thus
each boundary direction within the query disk is represented by an edge or
connector capsule intersecting that disk.

The Rust proof visits every edge and connector. It discards a capsule only when
its outward distance lower bound strictly exceeds the disk radius. Every
remaining direction must have a certified angle strictly less than half the
existing cut angle to one common axis. Directed dot and cross-product bounds
give an axis-cosine lower bound. The axis is only a numerical proposal: all
capsules must independently satisfy the cone inequality.

Bounded subdivision excludes irrelevant portions of long capsules. Children
retain the parent radius plus an outward midpoint-rounding allowance, covering
the entire original capsule. Subdivision stops at depth 12 or 65,536 visits per
attempt. Failure or budget exhaustion returns no certificate; neither is a
positive cut classification. The private certificate retains its query, radius,
axis, axis-cosine bound, accepted-piece count and total visits. The reported
cosine is relative to the axis, not a pairwise cosine bound.

Each node supplies the independently certified scalar upper bound plus the
outward-rounded original `0.5h` tie allowance. All true boundary feet within that
tie band are therefore covered. Witness search first attempts one quarter of
the original distance-error allowance, then falls back to the original
allowance. This changes proof effort and interval tightness, not the canonical
signed nodal value or its acceptance window.

The development trials accepted 86/162 nodes with the initial norm-ratio bound,
98/162 after dot/cross bounds, and 160/162 after capsule subdivision. The two
remaining native refusals were `(0.32049560546875, 0.04733753204345703)` and
`(0.32049560546875, 0.04733848571777344)`. Additional mesh halvings did not
resolve them. Tighter exterior witnesses did. The companion JSON preserves
terminal summaries and hashes of these unsuccessful trials.

The diagnostic tries mesh levels from native `2^-20` and Python `2^-27`, allowing
four halvings. All 162 nodes succeed within those diagnostic limits. This does
not establish runtime acceptance. The existing three contour-edge bisections
localize crossings; they are distinct from these spatial halvings. Neither
runtime policy has changed.

Final validation passed 105 provider tests in 170.50 seconds and all 53 focused
tests in 86.78 seconds, with no filter. All 26 injected faults were caught,
including capsule-radius omission, bypassed angle acceptance, equality pruning
and ignored work exhaustion. Three independent cover/scalar checks passed in
123.75 seconds. Successful cone attempts used at most 6,356 visits and 5,153
accepted pieces; these counts exclude previous failed attempts and all witness
search, cover construction and other query work.

Apply `experiments/geometry-145-direction-cones.patch` after the certified-scalar
patch and its documented prerequisites. Reproduce the focused and independent
checks with:

```powershell
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-generic-component-check.py --source-root tmp/geometry145-runtime-integration --target-dir target/geometry145-runtime-integration --precision-target-dir target/geometry145-precision --emit-cover --output tmp/geometry-145-direction-cone-final.json
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-branch-hp-check.py --input tmp/geometry-145-direction-cone-final.json --target-dir target/geometry145-precision --output tmp/geometry-145-direction-cone-independent.json
```

The independent checker replays the lower cover, exterior points and scalar
windows. It does not independently replay cone arithmetic. Cone coverage has
separate Rust adversarial tests, mutation controls and three-axis review; that
distinction remains explicit in the evidence.

Still required: integrate an interval-bearing sampler into the coherent runtime
field, propagate its uncertainty through refinement, preserve positive cut
classification, and rerun original crossings, inward convergence, derivative
consistency, energy, live training and complete query cost. The recorded search
and proof times are experiment costs, not frame-budget acceptance.
