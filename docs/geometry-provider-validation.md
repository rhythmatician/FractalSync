# Adaptive geometry validation, September 26, 2026

PR #148 was rebased onto main at `32ddb065d943374b68deba18816404acef9029bc`, including the transported camera fix. This record describes the repairs following that rebase. **Issue #145 remains open and PR #148 is not ready to merge.**

## Repairs and evidence

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

## Remaining blockers

1. **Native Shore crossing.** The Rust crossing test still fails. Python high-curvature, near-Shore, and underpowered/above-barrier trajectories encounter Unresolved or Singular geometry. Recorded failing positions include `(0.4005984941807052, 0.054142245287925383)`, `(0.300068357754536, 0.05333168318631767)`, and `(0.38777728061779215, 0.05037298251515404)`. Determine which represent real geometric ambiguity and which are contour/jet-estimation defects, then establish the ordinary Controls behavior there. Do not relax quality thresholds merely to obtain a trajectory.
2. **Initialization and diagnostic behavior at invalid geometry.** Default-state and cut-locus snapshot tests fail because metric evaluation requires Regular geometry before the snapshot can report validity. Define a valid startup state and a diagnostic contract for singular/unresolved states. Moving all fixtures to regular points would hide the lifecycle problem.
3. **Broader numerical and perceptual acceptance.** Finish #120 validation of conditioning, energy drift, cut loci, perturbation sensitivity, deep scales, cost, and cross-language behavior. Then obtain the #82 crossing/non-crossing visual verdict. The passing sample and bindings checks above do not close these items.

The [migration inventory](geometry-provider-migration.md) records destination ownership and deletion triggers for this geometry seam. It is a partial #84 artifact, not completion of that broader issue.
