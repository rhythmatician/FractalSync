# Geometry #145: deterministic bounded proof tiles

This unshipped candidate follows `58766f5`. It adds bounded reuse between the
runtime-owned certificate backend and the coherent fitter. The public query
loop does not call the new adapter yet. Production behavior is unchanged.

## Implementation

The new LRU retains at most 64 covered tiles. Each key fixes the level, core
coordinates and error window. A tile derives its complete 16 by 16 lattice and
its seed from that key, owns an immutable attracting cover, and retains at most
256 nodal certificates. Non-grid query centers do not grow the node cache.
The existing Shore and experimental node caches are still present; adoption
must reconcile those paths rather than claiming a completed provider cache
budget here.

Each retained node keeps its private exterior witness and proof. A new query
may reuse a witness only after recomputing its lower clearance and directed
upper distance and passing both scalar-window gates. Donor selection uses
only the requested stencil with deterministic coordinate ties. A failed
reuse attempt falls back to a fresh exterior proof.

All 81 samples flow through the shared enclosed-level fitter. The analytic
component path uses the same fitter. The new covered-level adapter preserves
the canonical nodal values, coordinate checks, ranges and existing thresholds.
It separately returns complete direction agreement rather than interpreting
a failed cone as a positive cut.

## Validation and fixes

The frozen candidate passes **124 provider tests**, with 224 tests filtered
across 23 suites. The provider test process took 343.98 seconds. The WASM bridge
checks successfully with zero errors and 93 warnings, including the currently
unused backend. All 39 captured Rust and manifest inputs remained unchanged.

The new tests certify all 162 original stencil nodes through the runtime owner,
preserve their proposal values, pass both original nodal-uncertainty budgets,
and compare complete fitted jets, ranges, errors and direction verdicts after:

- repeating a query;
- visiting another grid cell within the same tile in either order;
- evicting the actual tile key and reconstructing it.

The initial first-query-selected cover and accumulated-target extension were
sound proof owners but could change acceptance evidence with cache history.
Review caught this. The fixed lattice, immutable cover and stencil-only donor
policy remove that dependency. Review also caught an initial ordering test
whose neighbor stayed in the same grid cell. The final test asserts both an
equal tile key and distinct stencil sets. Standards, Spec and Health review
found no remaining concrete issue in this bounded stage.

Diagnostic cold setup times were 50.30 seconds and 69.75 seconds; repeated
sampling took 0.00557 seconds and 0.03926 seconds. These measurements occurred
under concurrent validation load. They are not public hierarchy-query or
trajectory benchmarks and do not establish acceptable initialization cost.

An earlier 124-pass run was rejected by its source-drift guard after review
changes landed during validation. The final evidence above uses a fresh,
unchanged snapshot and the strengthened different-cell test.

## Remaining adoption

Connect the covered sampler to actual hierarchy/refinement queries, reconcile
cache ownership, and reduce cold proof cost. Then rerun the original crossings,
inward convergence, derivative, energy and live-training validation. Browser
artifact size, real query latency, and required version/golden/binding updates
also remain outstanding. PR #148 is not ready to merge.

[Machine evidence](geometry-145-runtime-tiles.json) and
[candidate delta](experiments/geometry-145-runtime-tiles.patch).
