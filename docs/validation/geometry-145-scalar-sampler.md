# Geometry #145: explicit scalar sampler validation

The reconstructed coherent-field candidate passed 76 geometry-provider tests but failed `signed_stencil_preserves_distance_to_segment_endpoint`. That test supplies a synthetic line segment and expects distance `sqrt(2)` at `(2, 1)`. The candidate's canonical-node sampler instead queried actual Mandelbrot geometry and returned `1.7525267751610076`. This was a mismatch between the synthetic fixture and the sampler being exercised.

The follow-up factors node sampling into an explicit closure. Runtime still supplies the identical canonical Mandelbrot node function. The synthetic regression supplies the signed distance to its fixed line segment. Its endpoint assertion and `1e-6` tolerance are unchanged. No crossing test or acceptance threshold changes.

The endpoint test passes. The full geometry-provider rerun passes all 77 selected tests in 141.62 seconds, with 224 unrelated tests filtered out. This verifies the sampling seam and the selected provider regressions. It does not verify full crossings, live training, energy, or cost of this reconstructed runtime candidate.

Apply [the follow-up patch](experiments/geometry-145-scalar-sampler.patch) after the ten patches listed in [the ranged-field report](geometry-145-ranged-field.md), starting at `d8ec37a`. Reverse patch checking passes against the tested source. [Source hashes and test output](geometry-145-scalar-sampler.json) record the exact variant. It remains unshipped.

The isolated integration checkout also exposes a merge requirement: production's `uniform_box.rs` has changed since `d8ec37a`. Further integration must preserve those repairs rather than copy the older experimental file over them.
