# Geometry #145: coherent refinement and integration

PR #148 is not ready to merge. These are unshipped source snapshots following
`97370ad`; production geometry is unchanged. Public quality thresholds and the
original native/Python crossing tests remain intact.

The source-aware hierarchy replaces a disagreeing Shore model using smooth
lookahead weights. The value, gradient, Hessian and enclosing range use the same
expression. Scalar nodal certification is separate from query direction, and
acceptance requires certified membership at both convergence levels and a
resolved current direction. Covered interior samples prove membership separately
from uniqueness of their normal cone. A missing
first derivative stencil retains a finite, explicitly unresolved scalar.

The cache experiments preserve complete uncertainty masks in compressed empty
tiles. Exact-coordinate membership and certificate memos are bounded; eviction
changes recomputation, not decisions. The prescribed cache capacities and total
memory still need reconciliation before adoption.

## Verified results

Results belong to the named snapshot, rather than a mixture of candidates.

| Snapshot | Check | Result |
| --- | --- | --- |
| `shore-certificates` | Full provider subset | 152 passed, 362.23 s |
| `shore-certificates` | Original derivative diagnostic | All 12 probes pass at the original tolerance |
| `certified-rk4` | Python physics, excluding crossing and continuity | 25 passed, 450.23 s |
| `certified-rk4` | Original signed-distance continuity | Passed, 513.70 s |
| `certified-rk4` | Mandatory parity preflight | Passed, 17.10 s |
| `certified-rk4` | Browser bridge compilation | WASM target checks successfully |
| `certified-rk4` | Full provider subset | Timed out at 1,200 s; no full-suite pass |
| `nodal-finest` | Original derivative diagnostic | All 12 probes pass at the original tolerance |
| `nodal-finest` | Newly recorded Python refusal | Regular, 41.12 s |
| `targeted-proofs` | Newly recorded native refusal | Regular, 42.12 s |
| `coherent-floor` | Recorded Python/native refusals | Both Regular, 14.53 / 138.75 s |
| `coherent-witness` | Recorded Python/native refusals | Both Regular, 10.51 / 118.72 s |
| `coherent-witness` | Closed-prefix independence control | Passed |

Accepted guarded runs report no source drift. Python runs load a separately
staged extension with `RUNTIME_CORE_NO_BUILD=1` and check its path and hash;
the installed production extension is also guarded against replacement.

The previous high-curvature rollout failed with energy drift `0.10810054`
against the unchanged `0.05` limit. Probes at its largest excursions show
neighboring gradients and Hessians converge to the returned derivatives.
An RK4 diagnostic using the same geometry reduces maximum excursion to
`0.007998` at `dt=0.002` and `0.001182` at `dt=0.001`. The Rust timestepper now
uses that method in the adoption candidate. Python and Controls still delegate
to Rust. Every intermediate stage retains the open-disk invariant and geometry
refusals. Regenerated candidate goldens carry `orbit-controller/11`.

## Remaining failures and cost

The original crossings were rerun on `shore-certificates`. The native test
fails at `(0.32057277423411185, 0.04728409835060002)` because a required scalar
node remains unresolved. The Python test fails at
`(0.37490511279048488, 0.05358919829424173)` because the retained field differs
too much from finer geometry. Those are actual failures, not timeout verdicts.
Focused later experiments clear each point; complete trajectory passes on a
combined candidate are still required.

Nodal refinement is coordinate-owned and keeps the original level bounds.
Taking every available finer source clears the Python refusal but costs too
much. Refining coarse coefficient sources while avoiding unnecessary fine
sampling is cheaper. A certified closed prefix can use local contour samples
as later convergence witnesses without retaining those coefficients; its
independence control checks the full jet and range. The former first-level
contour-borrowing shortcut is removed from that experiment.

Proving every uncertain node in an entire tile is also too expensive. Directed
escape proofs now target only nodes that can affect the nearest-Shore check.
An interior result or failed proof leaves the uncertainty unresolved. Independent
high-precision replay remains in proof controls rather than running on every
cache miss during a trajectory. The full provider rerun after that test-only
change was interrupted and produced no completion record.

The cleanup also made the covered-node 512-bit replay unconditional in native
test builds. This adds diagnostic work that release builds do not perform.
The latest worktree restores explicit `GEOMETRY145_REPLAY_FAST=1` audit mode
inside `cfg(test)` and retains the dedicated proof controls. Repeated numerical
validation must distinguish that instrumentation cost from production cost.

An additional source-cover experiment is still under validation. It compares
all 81 control samples and intervals with canonical searches and checks eviction.
It also exposed a fitter inconsistency: clamping a scalar without applying the
same monotone clamp to its interval can reject a valid bounded sample. A focused
regression reproduces that refusal. This newer experiment is not included in
the acceptance results above. Applying the same monotone clamp to the value
and interval makes the regression pass; the fix is also backported into the
adoption worktree.

A later source-cover run completed the original native crossing in 718.73 s
and failed at `(0.32051424405315243, 0.04736324864591995)` with an unavailable
first fitted scalar stencil. This is a new recorded refusal, not a passing
crossing. Separately, requiring a previous unique direction for a covered
interior sample incorrectly conflated membership with differentiability. A
focused regression now verifies that distinction, and the previously refused
higher-period query returns Regular in 120.07 s. Both changes remain candidate
work pending validation on the full trajectories.

The cold-cache replay of that new native refusal timed out after 300.17 s
before reaching its final scalar-stencil diagnostic. The source guard recorded
no drift. This is a query-cost failure and provides no passing correctness
verdict. The archive includes the exact replay source and the earlier successful
membership regression as separate snapshots.

Before adoption, finish both original crossings, repeat the applicable checks
on one final source snapshot, validate live training, settle query/memory cost,
and finish browser artifact and project checks. Single-point Regular results
do not establish trajectory validity or merge readiness.

## Reproduction

[Machine records](geometry-145-refinement.json) contain source pins, binary
hashes, limits and verdicts. The
[source archive](experiments/geometry-145-refinement/sources.zip) preserves the
exact Rust source bytes for each named snapshot. Readable patches in the same
directory compare each snapshot with the source recorded by
[`20b1af8`](geometry-145-runtime-query.md).

Restore a snapshot with the hash-checking helper, for example:

```text
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-refinement/restore.py --snapshot certified-rk4 --output tmp/refinement-replay
```

The archive deduplicates unchanged files by content hash. The helper restores
the repository's unchanged
`runtime-core/data/mandelbrot_distance_1024.bin`; its hash is recorded as external
scaffolding rather than duplicating it in every archive member. Only the
`certified-rk4` and `proof-controls` scaffolds include the regenerated controller
goldens. Other snapshots remain numerical experiments, not parity-certified
adoption packages. Use the project venv for Python builds and explicit Cargo
manifest/target paths. Historical timeout results are not new acceptance limits.
