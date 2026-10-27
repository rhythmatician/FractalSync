# Complete scalar windows for #145

PR #148 remains unready to merge. This unshipped experiment combines the
[shared closed-cover lower bounds](geometry-145-f64-cover.md) with independently
proved exterior witnesses. All 162 canonical nodes now certify the original
distance-error windows: `5e-8` native and `5e-10` Python. No production code,
threshold or original crossing test changes.

The Rust proposal search uses the discovered nearest-branch direction, four
travel distances inside the original upper-error window and 81 angular offsets.
Every actual represented candidate must pass an outward endpoint-distance check.
The existing shared `MembershipOrbit` screens candidates and ranks numerical
escape updates; these floating results never issue exterior authority. The first
32 proposals are eligible for the existing outward multiprecision escape proof,
using 128 bits with a 256-bit fallback and the unchanged 32,768-update cap.

The retained run proves every selected witness at 128 bits. The longest proof
escapes at update 25,345. A segment from an independently proved interior query
to a proved exterior point crosses the boundary, so its outward endpoint distance
is an upper bound. The private witness retains its exact represented point and
proof. Each scalar certificate independently evaluates the curve clearance and
endpoint upper bound at its retained query, then requires the entire interval
inside the original error window with inward-rounded acceptance limits.

The returned nodal value remains the deterministic signed numerical proposal;
it is not selected from a query-dependent bracket midpoint. Replacing it with a
midpoint is caught by a regression. Reverse-order proposal checks, the existing
quintic derivative and cell-join checks, and both endpoint-gate controls remain.
These are certified nodal samples, not true-distance derivative or between-node
accuracy certificates.

Validation:

- The broader provider suite passes 103 tests.
- The precision-enabled focused suite passes 51 tests and catches all 22 injected
  faults. New controls bypass either scalar endpoint gate, change the canonical
  value, forge exterior authority or misreport the shared escape cutoff.
- Three independent tests pass. The existing multiprecision checker revalidates
  all 4,349 lower-cover pieces and 162 clearances, then separately re-proves all
  selected exterior points at 512 bits. Independent endpoint norms fit within the
  f64 upper bounds, and exact high-precision comparisons confirm both original
  error limits. Reported scalar lower bounds match the independently replayed
  cover targets exactly.

The full focused run takes 65.98 seconds under concurrent provider validation.
The earlier filtered complete-window run takes 56.53 seconds. These include
shared-cover construction, proposal search and exterior proofs. They exclude
runtime refinement, cut classification, Physics, cache replacement and complete
query cost. They do not establish a live frame budget.

The original three normal-offset candidates certified only 33 of 162 nodes at
128 bits. Two sampled refusals also remain unescaped under the cap at 256 and
512 bits and in the canonical floating screen. Angular search addresses candidate
placement while retaining fresh proof authority; no escape cap was extended.
Those unsuccessful trials and diagnostics remain in the companion JSON.

Still required: adopt the interval-bearing sampler in the coherent runtime
field, propagate its uncertainty through refinement, prove adequate global
nearest-feature/cut coverage, and rerun original crossings, inward convergence,
energy, live training and practical query cost. The precision implementation is
currently linked into a temporary validation harness; it is not a new runtime
dependency or a Regular-jet shortcut.

Apply `experiments/geometry-145-certified-scalar.patch` after the f64-cover patch
and its documented chain. Build the reconstructed runtime checkout and the
existing point-proof experiment dependencies, then run:

```powershell
rtk cargo test --release --jobs 1 --manifest-path tmp/geometry145-runtime-integration/runtime-core/Cargo.toml --target-dir target/geometry145-runtime-integration geometry_provider::
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-generic-component-check.py --source-root tmp/geometry145-runtime-integration --target-dir target/geometry145-runtime-integration --precision-target-dir target/geometry145-precision --emit-cover --output tmp/geometry-145-mp-scalar-final.json
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-branch-hp-check.py --input tmp/geometry-145-mp-scalar-final.json --target-dir target/geometry145-precision --output tmp/geometry-145-mp-scalar-independent.json
```

The optional `--test-filter` runs a named diagnostic without mutation coverage;
the recorded final run is unfiltered. Both harnesses modify temporary copies
only. The compact evidence JSON retains all scalar interval/witness rows and
omits the 4,511 lower-cover replay rows while recording stream/input hashes.
`--emit-cover` recreates the complete independent-checker input.

Independent review found a zero-match filter could report success. The harness
now rejects missing or zero executed-test counts and records the count and filter.
An intentionally missing filter exits with failure; a genuine one-test filter
passes and records zero mutations. These controls follow the unfiltered 51-test
run; Rust proof sources were unchanged. All three review axes are clear after
rechecking this fix.
