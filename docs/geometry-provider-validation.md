# Adaptive geometry validation, September 26, 2026

PR #148 was rebased onto main at `32ddb065d943374b68deba18816404acef9029bc`, including the transported camera fix. This record describes the repairs following that rebase. **Issue #145 remains open and PR #148 is not ready to merge.**

## October 2 follow-up: startup and partial diagnostics

The approved lifecycle change is implemented. An untouched controller entering manifold mode starts at the verified regular point `(-0.5, 0.1)`. Legacy initialization stays at the origin. Explicit positions, existing nonzero velocity, and established trajectories are preserved. There is no new collision or swept-path barrier at `(0, 0)`; the integrator still evaluates sampled states and endpoints and fails closed when a required derivative is unavailable.

`debug-snapshot/5` returns partial snapshots for Singular, Unresolved, and outside-provider geometry. Position, velocity, classification, provenance, and available finite scalar values remain visible. Unavailable metric, derivative, force, kinetic/total-energy, and camera geometry values serialize as `null`. Diagnostics report invalidity and an error instead of failing before the provider can be inspected. Malformed state/configuration inputs are rejected. The controller preserves the core invalidity flag. `orbit-controller/7` records the startup change; regenerated golden vectors changed only this version string.

The cockpit displays unavailable values explicitly and retains its last valid scene when the new snapshot cannot supply rendering derivatives. Manual startup no longer manufactures an explicit origin seed. Explicit saved seeds remain explicit.

Verification of this follow-up:

- Controller integration tests: 21 passed, including explicit origin refusal, preserved legacy defaults, and preservation of existing velocity at the origin.
- Snapshot-focused Rust tests: 16 passed, including singular/nonzero-velocity, unresolved, regular equivalence, JSON nulls, and malformed input.
- Full Rust suite: 225 passed, 4 failed, 1 ignored. One failure was an obsolete test expecting an error instead of a partial outside-provider snapshot; its corrected scoped test passes. The remaining three failures are the native undriven/driven rollout test and two pre-existing manifold assertions in the debug integration file. The full snapshot integration target had 31 passed, 2 failed.
- Canonical preflight parity passed. Python stub generation had no API diff.
- Final compiled Python smoke: `(0,0)` is invalid/Singular with no metric; `(-0.5,0.1)` is valid/Regular with a metric; `(0.35,0.05)` is invalid/Unresolved with no metric.
- Frontend production build passed; 27 test files / 116 tests passed. Model-dependent ONNX checks retain their existing unavailable-runtime skips.
- Three focused Python debug tests passed against `target/python-snapshot5`. The full debug file was interrupted during slow controller replay tests; no complete result is claimed.
- Rebuilt WASM passed `node frontend/scripts/test_debug_snapshot_contract.mjs`. The test caught and fixed the default serializer emitting `undefined` for Rust `None`: both snapshot bindings now serialize explicit `null`, matching Python and recorded JSON. The test covers the standalone and controller APIs, regular automatic startup, explicit origin preservation, and unresolved diagnostics.

**Remaining #145 blocker:** the native rollout test fails in its undriven phase, near `c=(0.29842574478946454, 0.051456454020995405)`, `v=(-0.003236304445455842, 0.0029941732581717675)`, with a CutLocus refusal. This is not evidence that the driven trajectory crossed the Shore and failed there. A narrow refinement probe also found that `(0.35,0.05)` remains Unresolved at level 8; level 9 cannot contain the nearest contour in the current expanded tile footprint. Its level-8 error is about 0.00657 for requested scale 0.001028. These results require further provider/classification validation; the startup and diagnostics repair does not establish that the refused rollout is a genuine geometric singularity, or that changing the footprint alone fixes it. Crossing tests remain enabled. PR #148 and issue #145 remain unaccepted.

## October 2 CI follow-up

All five initial CI failures on `a1db98b` were compiler failures at the deprecated `std::f64::INFINITY` constant with warnings denied. Replacing it with `f64::INFINITY` in `b6dfc3d` restored green CI stub generation, Python packaging, WASM compilation, artifact builds, and Python/WASM type checks.

The next run exposed a GPU harness fixture assuming every snapshot had rendering derivatives. The harness now asserts a Regular snapshot, samples a small canonical Rust terrain patch, and expands that fixture to the same 512x512 GPU mesh. All original projection tolerances, static-buffer assertions, triangle-count checks, and 1,000 camera updates remain. This explicitly tests GPU rendering rather than dense adaptive-provider throughput. The real local WebGL run passed 16 projection cases with maximum error 5.411e-8, 262,144 vertices, 531,502 triangles, and no browser/GL errors. CI now also runs the real-WASM snapshot/startup contract script.

Two Rust integration fixtures also requested smooth energy/motion at provider-classified singular points. Their ordinary-geometry assertions now use an explicitly verified Regular point; the energy fixture retains the same distance from the origin and thus the same wall contribution. The actual native rollout regression and singular-refusal tests remain enabled. **The complete Rust debug-snapshot integration target now passes 33/33.** The Python non-terrain snapshot/controller subset passes 16/16; its five terrain tests were not part of that bounded run.

One known Rust rollout regression remains. At the recorded pre-step state the jet is Regular; the candidate endpoint `(0.29835945330580743, 0.0515177858286498)` is classified Singular/CutLocus. A narrower probe has not established whether the competing normals represent genuine geometry or premature classification. Reducing epsilon does not refine this case because rho dominates the requested scale. No tie threshold, normal-angle threshold, physics rule, or crossing assertion was relaxed. CI and #145 are not claimed fully green or complete.

## September 26 repairs and evidence

The provider now measures nearest-segment distance, handles the sign at shared contour vertices, and resolves ambiguous marching-squares connectivity using membership. It requires stable contributing tiles, checks containment of the sampled distance neighborhood, and refines one dyadic level at a time. Cross-level gradient and Hessian error use the coarser level's spacing. Regular jets must satisfy both the requested resolution and error bound. Cut-locus classification excludes incidental corners but retains persistent competing nearest features. Missing contours return unresolved geometry instead of invented jets.

Loading or clearing the legacy raster does not alter destination queries. Provider failures also preserve destination provenance. Exact main-cardioid and period-two-bulb membership shortcuts reduce repeated interior iteration; regression tests compare them against the original orbit evaluator near their boundaries.

Manifold regularization, snapshots, and terrain diagnostics now query with the active epsilon. The cockpit displays the provider's resolution, estimated error, validity, singularity, and bridge provenance. Versions are `geometry-provider/3`, `debug-snapshot/4`, and `orbit-controller/6`. Golden vectors were regenerated. Their numeric values are unchanged; the controller version changed. Python stub generation produced no API diff.

## Verification

These are local Windows results using release Rust builds and a rebuilt Python extension isolated under `target/python`. The Python commands use the project's `.venv/Scripts/python.exe`, `PYTHONPATH=target/python;backend`, and `RUNTIME_CORE_NO_BUILD=1`. They are not a claim that GitHub CI is green.

| Check | Result |
|---|---|
| Rust provider unit tests | 10 passed |
| Rust public adaptive geometry contract | 3 passed |
| Rust Controls tests | 15 passed |
| Rust library suite | 65 passed, 1 failed, native Shore crossing |
| Full Rust suite with `--no-fail-fast -- --test-threads=1`, before the final snapshot epsilon repair | 207 passed, 16 failed, 1 ignored across 24 target summaries |
| Rust debug snapshot suite after that repair | 16 passed, 14 failed |
| Python manifold physics | 25 passed, 3 failed |
| Python debug snapshot subset covering versioned sections, time, control provenance, and initial action | 4 failed at the cut-locus origin |
| Python full debug snapshot file | Interrupted after nine failures and slow terrain diagnostics; no complete result |
| `scripts/preflight_parity.py` | All mandatory checks passed; manifold maximum error 1.494e-08 |
| Frontend tests and production build | 27 files, 113 tests passed; build passed |
| Final rebuilt WASM snapshot and focused cockpit tests | Live regular snapshot had current versions and estimatedError; 21 tests passed |
| Python stub generation | Completed, no API diff |
| `git diff --check` | Passed |

The provider, Controls, and public-contract results overlap the Rust library/full-suite counts. Do not add them together. A complete all-green suite, GPU rendering acceptance, and eyes-on Shore verdict are still outstanding.

Regular-geometry algebra/parity tests now assert regular fixtures explicitly. The randomized parity trajectory starts at `(-0.5, 0.1)` with low drive energy to remain in that chart. It proves binding parity there; it does not prove general navigation or successful Shore crossing. The crossing tests remain enabled and failing.

Useful reproduction commands from the repository root:

```text
cargo test -p runtime_core --release --no-fail-fast -- --test-threads=1
cargo test -p runtime_core --release --test test_adaptive_geometry_contract
cargo test -p runtime_core --release --test test_debug_snapshot -- --test-threads=1
python -m pytest backend/tests/test_manifold_physics.py -q
python scripts/preflight_parity.py
npm --prefix frontend test -- --run
npm --prefix frontend run build
cargo run --release -p runtime_core --example geometry_provider_report
```

Use the configured project interpreter for the Python commands and rebuild its extension before comparing results.

## Measured query sample

[The JSON report](validation/geometry-provider-145.json) contains five points at four epsilon values each, with jet quality and first/immediate-repeat timings. The example asserts identical repeated jets and the strict resolution/error bounds for every Regular result. Its first queries can reuse cache entries from earlier rows, so `first_us` is not an isolated cold-cache measurement.

At epsilon 1e-4, `(-0.5, 0.1)` and `(0.3, 0.05)` are Regular, `(0.35, 0.05)` is Unresolved, and the origin and cusp `(0.25, 0)` are Singular. The cusp becomes Unresolved at epsilon 1e-8. Immediate-repeat timings in this release sample were about 0.45 to 1.49 ms; first touches reached about 82 ms. These machine-specific measurements do not establish the live frame budget or deep-scale acceptance.

## September 26 blockers (updated October 2)

1. **Native Shore crossing.** The Rust crossing test still fails. Python high-curvature, near-Shore, and underpowered/above-barrier trajectories encounter Unresolved or Singular geometry. Recorded failing positions include `(0.4005984941807052, 0.054142245287925383)`, `(0.300068357754536, 0.05333168318631767)`, and `(0.38777728061779215, 0.05037298251515404)`. Determine which represent real geometric ambiguity and which are contour/jet-estimation defects, then establish the ordinary Controls behavior there. Do not relax quality thresholds merely to obtain a trajectory.
2. **Initialization and diagnostic behavior at invalid geometry — resolved October 2.** The follow-up above implements regular untouched startup and partial invalid snapshots. Explicit origin states remain covered by refusal and diagnostic tests.
3. **Broader numerical and perceptual acceptance.** Finish #120 validation of conditioning, energy drift, cut loci, perturbation sensitivity, deep scales, cost, and cross-language behavior. Then obtain the #82 crossing/non-crossing visual verdict. The passing sample and bindings checks above do not close these items.

The [migration inventory](geometry-provider-migration.md) records destination ownership and deletion triggers for this geometry seam. It is a partial #84 artifact, not completion of that broader issue.
