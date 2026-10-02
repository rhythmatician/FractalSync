# Geometry provider migration and deletion plan

This is the #84 migration record for the #145 Physics geometry seam. The architecture route belongs to [#77](https://github.com/rhythmatician/FractalSync/issues/77); this file records only the code ownership, bridge consumers, and deletion triggers needed to change this seam safely. It supplements [ADR 0004](adr/0004-scale-aware-mandelbrot-geometry-provider.md), which owns the provider design.

## Migration rule

Migration is one way:

```text
LEGACY input -> temporary BRIDGE -> DESTINATION seam
```

New destination behavior must not depend on legacy scaffolding. Keep an adapter only while an identified caller or a named validation task needs it. Retire it when that consumer moves and its replacement is accepted.

| Class | Meaning |
|---|---|
| DESTINATION | Expected to survive; contracts and substantive tests belong here. |
| BRIDGE | Temporary adapter into the destination seam; it adds no lasting authority. |
| LEGACY | Still has a supported consumer, but is not destination Physics authority. |
| DELETE | No surviving in-repository consumer; remove rather than extend. |

## Geometry ownership and migration inventory

| Tree-backed path | Class and owner | Current consumer or evidence | Delete or migration trigger |
|---|---|---|---|
| `runtime-core/src/geometry_provider.rs`: `query_geometry` / `ScaleAwareGeometryProvider`; `runtime-core/src/manifold.rs`: `geometry_jet` and the chain-rule Physics consumers | **DESTINATION** — #145 owns the provider; #120 owns mechanics validation. Normal `query_geometry()` calls the scale-aware provider, and manifold geometry queries delegate to it. | `runtime-core/src/manifold.rs`; `runtime-core/src/scale_relative_map.rs`; `runtime-core/src/debug.rs`; provider and manifold tests. The provider returns resolution/error and validity evidence in `GeometryJet`. | Keep. Do not reintroduce fixed-raster finite differences or make PlayerObservation depend on Physics jet packing. #120 validation is pending; no acceptance claim is made here. |
| `RasterBridgeProvider`, `query_bridge_geometry[_with_alpha]`, and 9-point fixed-raster jet in `runtime-core/src/geometry_provider.rs` | **BRIDGE** — retained only for migration/comparison. It is not selected by normal `query_geometry()` in the current tree. | Repository search finds the provider and explicit bridge functions at their definitions, but no in-repository production call site. The scale-aware provider has a raster-independence test. Treat any new call site as an explicit migration exception. | After #120 records the destination comparison and confirms no external/in-repository migration consumer remains, delete the bridge type, query functions, and bridge-only code/tests. Do not switch normal Physics back to the bridge. |
| Fixed 1024² signed-distance field and sampler in `runtime-core/src/distance_field.rs` | **LEGACY** — training/loss input; not Physics geometry authority. | `backend/src/visual_metrics.py` samples it for `LossVisualMetrics.mandelbrot_distance_estimate`; `backend/src/control_trainer.py` consumes that loss path. Rust/Python sampler tests and the built-in-field tests also use it. | Delete the raster asset/sampler only after the loss metric and all training callers migrate to an accepted replacement and the sampler tests have moved or been removed. Removing `RasterBridgeProvider` does not itself satisfy this trigger. |
| 2048² `F/S` mip pyramid, `MINIMAP_LEVELS`, and `runtime-core/src/minimap.rs` readers | **LEGACY** — existing observation/cartography and shore-bias compatibility; separate from signed-distance Physics geometry. #108 owns the scale-relative `M_c/R_c` destination observation. | Runtime readers live in `runtime-core/src/minimap.rs` and its WASM bindings. `frontend/src/lib/orbitSynthesizer.ts` loads the fields, and `frontend/src/lib/modelInference.ts` enables shore bias from them. `backend/src/control_trainer.py` loads the pyramid for shore-proximity training behavior. The cockpit's `cockpitMinimap.ts` is a separate local overview renderer, not this F/S pyramid. | Delete the F/S pyramid and its loaders/readers only when the #108 replacement is accepted, every model/runtime observation and training consumer has migrated, the shore-bias path has a replacement or is retired, and any #111 global-overview consumer has moved. Do not rebuild a larger fixed pyramid or couple it to Physics authority. |
| `manifold::derivative_step()` and `DiagnosticsSnapshot.derivative_step` / `derivativeStep` | **LEGACY compatibility diagnostic** — not a live Physics derivative input. | The snapshot reports it (`runtime-core/src/debug.rs`); the cockpit displays it; `runtime-core/src/bin/diagnose_antenna_mini.rs` uses it as the base for finite-difference diagnostic sweeps; `runtime-core/tests/test_debug_snapshot.rs` and `backend/tests/test_debug_snapshot.py` assert the field. The comment in `runtime-core/src/controls.rs` about a missing-field fallback is stale and is not evidence of a current fallback. | Remove the function and wire field after the diagnostic sweep uses the provider's reported requested/resolved scale directly, the cockpit/report no longer presents this compatibility field, and both snapshot assertions are updated. Keep diagnostic finite differences explicitly diagnostic; do not let them become Physics authority. |

## Current work and evidence owners

| Frontier | Build or decision | Do not extend | Keep until | Exit state |
|---|---|---|---|---|
| #145 GeometryProvider | Adaptive provider and shared local Map substrate. | Fixed raster as Physics authority; a second Mandelbrot evaluator; nested finite differences over sampled `sigma`. | #120 completes destination mechanics validation and any bridge comparison is recorded. | Provider selection and deletion consequences recorded here; #145 acceptance remains subject to its issue checklist. |
| #120 Physics validation | Validate regular geometry, singular/cut-locus semantics, energy behavior, perturbation sensitivity, cost, and Rust/Python/WASM parity against the destination provider. | Treating code presence or canceled/incomplete runs as validation evidence. | Validation owner records results and limitations. | **Blocked:** provider contract and partial-snapshot tests pass, but Shore rollout and manifold acceptance tests remain red. See the [validation record](geometry-provider-validation.md) for commands, counts, and remaining work. |
| #111 cockpit diagnostics | Present Rust snapshot provider, scale/error, validity, singularity, and bridge provenance as STATE/DIAG evidence. | Recomputing provider geometry or turning diagnostics into PlayerObservation without its contract. | Eyes-on cockpit review. | UI panel and generated WASM snapshot were checked together: the initial sample emitted `debug-snapshot/4`; the October 2 follow-up uses `debug-snapshot/5` with nullable unavailable derivatives, `geometry-provider/3`, `estimatedError`, and regular destination provenance. Visual acceptance remains pending owner review. |
| #82 native Shore crossing | Record watchable crossing and non-crossing trajectories and judge the visual crossing. | Treating mechanical crossing alone as visual acceptance. | #120 geometry findings and eyes-on crossing verdict. | **Crossing acceptance pending; no visual verdict is asserted here.** |

## Tests and artifacts to retain during migration

- Keep scale-aware provider tests for adaptive resolution, estimated error, cut-locus classification, tile/cache determinism, and operation without the baked 1024² field. These test destination behavior.
- Keep raster sampler and visual-metric tests only while the 1024² training-loss consumer remains. They do not prove Physics provider correctness.
- Keep F/S minimap and shore-bias tests only while their observation, training, or cartography consumers remain. They do not prove signed-distance jet correctness.
- Keep `diagnose_antenna_mini` finite-difference sweeps as diagnostic evidence while useful; label them as diagnostics and remove `derivative_step()` compatibility only at the trigger above.
- Keep #82's crossing artifacts separate from #120's numerical acceptance. A trajectory crossing is not by itself evidence that the destination geometry is accurate or that the visual transition is acceptable.

When a consumer moves, update this table and the owning issue. This geometry record covers only part of #84. Its broader migration inventory still needs reconciliation; #120, #111, and #82 retain their own acceptance evidence.
