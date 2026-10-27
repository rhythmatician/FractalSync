# Geometry #145: public coherent-query integration

This unshipped candidate follows `79c83a5`. The public query now calls the
covered sampler and shared enclosed-level fitter. Both recorded higher-period
refusal points return `Regular`. Production source remains unchanged, and
PR #148 is not ready to merge.

```text
canonical nodal proposals and certificates
  -> shared quintic value, gradient and Hessian
  -> fitted ranges attached before hierarchy closure
  -> consecutive scalar-quality gates and current direction certificate
  -> explicit geometry validity
```

## Query and proof changes

Scalar certification and direction agreement are separate. A failed direction
proof does not invalidate a scalar convergence sample. The accepted level must
still prove direction agreement and pass every existing public quality gate.
A direction refusal clears pending acceptance and remains `Unresolved`.

A current size weight that is constant zero, including its interval and
derivative coefficients, eliminates the complete finer tail. Earlier weights
retain their operand certificates. Review caught a provisional exact range
being inspected before the real fitted range was attached. The final candidate
passes the actual range into construction before checking closure. A regression
keeps refinement open when the numerical weight is zero but its enclosing
interval cannot prove zero.

Three internal hierarchy fixtures were corrected for the size-only selector.
The partial-weight regression now actually exercises a partial weight; the
zero-weight fixtures distinguish retained inputs from unused lookahead.
Public thresholds, the refinement cap, and both original crossing tests remain
intact. Their constants and file hashes are in the machine record.

The fast escape path propagates outward f64 critical-orbit disks, including
local roundoff and the linear and quadratic radius terms. It preserves the
32768-update cap and pre-update escape semantics. One early fast probe precedes
the original ranked multiprecision fallback, with its original 32 candidates
per window and 128/256-bit schedule. Shared screening keeps both paths' candidate
geometry and guards together.

Exact-coordinate proposal caching avoids repeated orbit/continuation work.
Fixed even-grid witness anchors reduce repeated exterior proofs. Each anchor
is proved directly; a node always selects the same anchor and rechecks both
scalar-window bounds. The 64-entry proof-tile cache remains bounded to 256
nodes per tile. A first 81-node stencil can retain at most nine additional
anchors. Query-center donors remain restricted to the requested stencil.

## Validation

The final unchanged snapshot passes:

| Check | Result |
| --- | --- |
| Provider suite | 139 passed, 60 filtered; 314.08 seconds |
| Original derivative diagnostic | All six cases at both original step sizes pass |
| Native energy regressions | Four passed |
| Inward distance-error regression | Passed within the provider suite |
| Isolated Python physics subset | 25 passed; crossing and signed-distance continuity remain unpassed |
| Browser bridge WASM compilation | Zero errors, 18 warnings |
| Fast escape controls | Four baseline tests pass; all three missing-radius-term mutations are caught |
| Fresh independent fast-witness replay | All 59 positive fast proofs re-proved at 512 bits |

The derivative tolerances remain `1e-4`. Across the 12 probes, the largest
gradient error is `1.02e-9`; the largest relative Hessian error is `6.09e-8`.
The full provider suite includes all original 162 stencil nodes, their unchanged
nodal uncertainty budgets, different-cell query ordering, and actual tile-key
eviction/reconstruction. Cached nodal intervals, complete fitted jets, errors
and direction verdicts agree after reconstruction.

The mutation runner compiles captured copies of the actual proof inputs.
It checks that four tests execute, rejects source drift, and distinguishes
containment assertion failures from compilation failures. Review also cleared
the fitted-range ordering fix, named source types, and shared witness screening.

The first Python full-suite attempt was rejected by its environment guard:
the repository fixture automatically rebuilt the installed production extension.
Subsequent runs set `RUNTIME_CORE_NO_BUILD=1`, load a separate candidate wheel,
and verify its path and hash. The installed production rebuild was separately
checked against the frozen production reference: all 64 jets match exactly at
identical represented coordinates. Its binary hash changed; the original binary
bytes were not restored.

## Remaining work

The original native crossing exhausted a 600-second diagnostic limit after
advancing beyond the recorded refusal, reaching approximately
`(0.32055668029005002, 0.04729598596080384)`. That run precedes the final
range-ordering cleanup. The final candidate's unchanged Python crossing
exhausts a 120-second limit while locating the Shore, at the query
`(0.33125, 0.05)`. These are incomplete runs, not passing tests or new acceptance
thresholds. A noncrossing suite attempt also exceeded its limit in the original
signed-distance continuity test.

Repeated cold proof construction and the ordinary Shore fallback still cost
too much. Resolve that path, then complete both original crossings, continuity,
and live-training validation. Before production adoption, remove diagnostic
logging/replay controls, reconcile cache ownership, update versions/goldens and
bindings, measure browser artifact size, and run the applicable project gates.

[Machine evidence](geometry-145-runtime-query.json) and
[incremental candidate patch](experiments/geometry-145-runtime-query.patch).
The patch applies after the validated runtime-tile experiment series in an
isolated checkout, not directly over the production provider.

Reproduce the final suite and derivative checks there with:

```text
rtk cargo test --release --jobs 1 -p runtime_core --lib geometry_provider
rtk cargo run --release --jobs 1 -p runtime_core --example geometry_derivative_report -- --check
rtk cargo check --manifest-path wasm-orbit/Cargo.toml --target wasm32-unknown-unknown --jobs 1
```

The [fast-escape mutation runner](experiments/geometry-145-fast-escape-controls.py)
accepts explicit `--candidate`, `--deps`, `--astro-float` rlib, and `--output`
paths. For actual-stencil fast-witness replay, set `GEOMETRY145_REPLAY_FAST=1`
and run `periodic_boundary::tiles::tests::cold_profile` in the Rust test binary.
