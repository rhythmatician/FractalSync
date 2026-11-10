# Geometry #145: runtime-owned certificate backend

This checkpoint follows `d801898` and the interval-bearing sampler. It moves
the generic proof implementation into the candidate Rust crate and introduces
one reusable certificate owner. Production queries have not adopted it.

## Changes

`DistanceCover` owns a validated closed attracting cover. Its constructor
accepts at most 289 finite, positive-error targets. Targets choose subdivision
effort; they do not choose the canonical signed distance. A sample requires
both the cover's lower clearance and a fresh directed exterior escape proof.
The retained scalar remains the original negative proposal value, with its
coordinate-bound enclosure. Private fields prevent callers from forging the
proof or replacing the canonical value.

Direction agreement uses the unchanged tie and angle constants. A failed
scalar or direction proof refuses conservatively. The runtime-owned precision
modules are byte-identical copies of the previously validated experiment,
with `astro-float` pinned to 0.9.6. The proposal, orbit seed, and cycle-trap
helpers are now compiled outside tests in the candidate crate.

## Validation

- Native provider suite: **122 passed**, 224 filtered, 23 suites, 209.87 s.
- Two focused backend tests pass, including invalid-request controls and both
  actual difficult query centers. Both centers retain bit-identical proposal
  values. The native center requires two spatial halvings for direction
  agreement; the Python center requires none. This is the existing bounded
  diagnostic of up to four spatial halvings, not a claim about runtime contour
  refinement or Regular acceptance.
- The browser bridge checks successfully for `wasm32-unknown-unknown`: zero
  errors, 79 warnings. Most newly exposed warnings report the currently
  uncalled backend. This check establishes compilation, not artifact size.
- Source hashes remained unchanged through validation. The patch applies to
  the interval-sampler baseline. Standards, Spec, and Health review found no
  concrete defect in this bounded stage.

Two preliminary tests correctly refused direction agreement when they
required the starting scale: first with a single-point cover, then with the
81-node cover. The final test exercises bounded spatial refinement without
changing thresholds. Checking `runtime-core` alone for the browser target
also refused because its random-number dependency lacked the bridge's `js`
feature; checking through `wasm-orbit` uses the actual browser configuration.

The direct focused rerun recorded roughly 10.46 s and 8.19 s for the two
cover-and-center checks under concurrent validation load. These are diagnostic
wall times, not public-query or cached-tile benchmarks.

## Next integration

The owner is not called by the runtime hierarchy yet. It must supply bounded,
reusable derived tiles and interval-bearing samples to the coherent fitter.
Per-node proposal and witness work is currently uncached. Before adoption,
measure actual query cost and browser artifact size, exercise all stencil
samples through the new owner, and retain conservative cut/refinement gates.

The original crossing, inward, derivative, energy, and live-training validation
still needs a successful run on the integrated backend. Thresholds and
original crossing tests are unchanged. Production source and its earlier
failure evidence are unchanged. PR #148 remains unready to merge.

Machine evidence: [record](geometry-145-runtime-backend.json).
Candidate delta: [patch](experiments/geometry-145-runtime-backend.patch).
