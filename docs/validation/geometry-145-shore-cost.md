# Geometry #145: Shore fallback cost and retained coarse geometry

This diagnostic follows the unshipped public-query candidate at `20b1af8`.
Production source remains unchanged. PR #148 is not ready to merge.

The original Python crossing timed out while locating the Shore. Its query
`(0.33125, 0.05)`, with epsilon `1e-4`, now has an isolated Rust reproduction
that asserts `Regular`. The original crossing tests and geometry thresholds
remain intact.

## Cost isolation

Four cold processes ran the same query with source hashes checked before and
after execution. The first three exhausted a 120-second diagnostic limit.
These time limits are not acceptance thresholds.

| Experiment | Level-21 search, 82 calls | Whole-query result |
| --- | --- | --- |
| Existing candidate | 76.75 seconds | Timed out at level 22 |
| Exact whole-box membership memo, 4096 entries | 73.90 seconds | Timed out at level 22 |
| Also retain stable empty tiles, 4096 entries | 1.30 seconds | Timed out at level 23 |
| Both diagnostic memos, 16384 entries | 1.28 seconds | Completed in 33.65 seconds; `Unresolved` |

Level 21 visits about 690 leaf tiles per search, exceeding the existing
384-entry Shore cache. Reconstructing empty tiles evicts useful results.
The diagnostic retains fully stable, segment-free generated tiles separately.
Their complete metadata matches fresh construction. Uncertain or contour-bearing
tiles still use the existing cache. Exact whole-box membership answers, including
failed shortcuts, also have a bounded memo.

The 16 whole-box tests pass, including query-order and eviction comparison with
the uncached evaluator. A separate control compares cached empty tiles against
fresh membership and contour construction and passes. These are targeted controls,
not a full validation of a new production cache design. The 4096/16384 capacities
are diagnostic settings. Runtime cache ownership, memory and browser cost remain
unsettled. The short whole-box control ran in another process during the final
query; the timings identify dominant work, not a benchmark acceptance result.

## Numerical refusal exposed

The final query finishes all original refinement levels and fails its
`Regular` assertion. At level 22, the coherent hierarchy closes with distance
`0.00012283832704465035`. Finer raw contour distances continue moving, reaching
`0.00007161828682331185` at level 24. The final returned jet is `Unresolved`,
with estimated error `0.000018744143402759172` and requested scale
`0.000012300072767063684`. Its error exceeds the unchanged quarter-scale gate.
The failing `Regular` assertion is a diagnostic target, not proof that this
point is smooth or a new merge gate. The original crossings remain the
acceptance authority.

Caching exposes this failure; it does not repair it. The size-only hierarchy
retains coarse geometry that disagrees with later contour estimates. An early
pair can agree before still finer contour features appear. A replacement must
preserve one coherent differentiable field while ensuring retained coefficients
meet the requested error. Accepting the early pair or weakening the final gate
would conceal this case.

Next work must address that retained coarse contribution and the fallback's
changing nodal geometry, then rerun this reproduction, the original crossings,
continuity and live-training validation. The earlier 12 passing derivative
probes and energy regressions remain evidence for their recorded candidate;
they do not establish correctness at this new point.

## Reproduction

Apply the [diagnostic patch](experiments/geometry-145-shore-cost.patch) after the
validated experiment series through `20b1af8`, in an isolated checkout. It
includes explicit diagnostic logs and the failing query assertion.

```text
rtk cargo test --release --jobs 1 -p runtime_core --lib shore_timeout_profile -- --nocapture --test-threads=1
rtk cargo test --release --jobs 1 -p runtime_core --lib geometry_provider::uniform_box -- --test-threads=1
rtk cargo test --release --jobs 1 -p runtime_core --lib empty_cache_controls -- --test-threads=1
```

[Machine evidence](geometry-145-shore-cost.json) records each executable,
source snapshot, per-level search work, final jet and hierarchy trace. The
diagnostic patch applies cleanly over the recorded public-query candidate.
