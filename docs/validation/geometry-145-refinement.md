# Geometry #145: coherent refinement and integration

Latest follow-up: [observed internal contour-edge closure](geometry-145-operational-closure.md) fixes a reproduced truncation and reduces source-comparison failures from 720 to 52. All 30 Rust controls, 35 per-run expectations, 21 reference crossings, parity and provisional p99 cost checks pass. Experimental Python remains 25 passed, 2 failed. Shipping-profile CI separately fails the original Rust/backend crossings. Feature-gated controls now have an additional CI step. Earlier assets and failures are preserved. This candidate is not merge-ready.

PR #148 is not ready to merge. These are unshipped source snapshots following
`97370ad`; production geometry is unchanged. Public quality thresholds and the
original native/Python crossing tests remain intact.

The owner authorized the bounded experimental implementation of
[ADR 0005](../adr/0005-bounded-operational-geometry.md). The
[operational prototype report](geometry-145-operational-prototype.md) records
the new feature-gated Rust field, unchanged crossing/energy replays and cold
prepared-query measurements. Its raster reference is uncertified, so these
results do not clear the current scalar-fidelity acceptance gate. The frozen
native controlled-crossing and retained energy replays also expose explicit
coverage failures; reference crossings remain unverified. These are recorded
outcomes, not requests for more point-specific proof extensions. ADR 0004
still governs shipping behavior. The older adaptive experiments below remain
failure evidence rather than the selected runtime implementation.

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

Targeted tracing found an unproved mask node at
`(0.3205142021179199, 0.047362565994262695)`. The existing directed proof
limits do not establish its membership. Building the level-20 shared cover
took 20.13 s; the subsequent membership check brought the total to 21.57 s.
This distinguishes cover construction cost from the failed membership proof.

Two interval-source experiments preserve uncertain sign instead of assigning
membership. A contour edge outside all enlarged uncertain-cell boxes survives
the permitted local contour changes, so its distance provides an unsigned
upper bound. The scalar enclosure retains both signs. With no surviving edge,
the sampler still refuses. This bounds the represented finite contour family;
it does not prove exact Mandelbrot membership or override query classification.

Both experiments pass all 12 original derivative probes at the unchanged
tolerance. The early interval variant also passes four source controls and
11 hierarchy controls. It advances the recorded native query past its missing
scalar stencil but still returns `Unresolved` after 215.33 s, with estimated
scalar error `9.77e-8`. Its regularity failure remains open. These experiments
are not an accepted crossing repair.

The deferred interval fallback subsequently timed out at 900.03 s and is not a
practical query strategy. The acceptance trace for the early variant confirms
that its coherent field is closed and certified at level 23, while the query
direction remains unresolved. Level 24 then lacks an unretained raw witness.
Increasing arithmetic precision through 2,048 bits did not resolve the three
recorded mask nodes within the unchanged 32,768-iteration proof limit.

An RK4 subdivision experiment retries unresolved predicted stages using at
most four timestep bisections. Unresolved starting states, detected singularities
and provider/domain failures still reject; failed subdivisions return no
partial movement. Four stage controls, the original 12 derivative probes and
all 25 selected Python physics tests pass on this snapshot. Its original native
crossing still fails after 299.24 s. This is not an accepted crossing repair.

The `relevance-certified` snapshot adds an outward-rounded geometric proof for
changed cells that cannot reach any nearest or competing feature. It retains
the four-cell stability band, half-cell competing-feature band and angle/error
thresholds. The new regression fails before the proof and passes afterward;
closer cells, tie-band overlap and rounding-boundary controls still refuse
exclusion. The original tied-projection test and all 12 derivative probes pass.
The original native crossing still fails after 284.58 s, now reporting a
suspected cut locus at `(0.32051319273424200, 0.04736395396592710)`. The earlier
isolated query remains Unresolved. Further geometry is being checked to
distinguish finite-resolution competition from a persistent nonsmooth feature.

The `linear-range` experiment fixes repeated accounting of the same uncertain
nodal value in the affine-reference interval evaluator. It first collects each
node's linear coefficient, then encloses that node once. The new regression
fails on the previous evaluator and passes with the correction, including
adversarial extrema for the value, gradient and Hessian. The nominal field and
public error thresholds are unchanged. All 12 original derivative probes pass.

At `(0.32051424405315243, 0.04736324864591995)`, a same-source comparison reduces
the analytic enclosure error from `0.42215383086462627` to
`3.39540053030035e-7`. The query becomes Regular with total error
`1.6042069473509945e-6`, below its unchanged `2.5000008e-6` limit. The original
native crossing passes that point but fails later at
`(0.32051451369978534, 0.04736306453144112)`, where the analytic enclosure still
dominates. Its full run fails after 569.47 s. The provider suite reports 171
passes and three failures: two unresolved rollout points and one test expecting
an unresolved result where the candidate now returns Regular. That last result
still needs independent verification before revising its expectation.
Its full 27-test Python run times out at 3600 s after reporting a ridge-start
failure. This predates the world-footprint repair and is not a passing suite.

The `bounded-weights` experiment addresses the next interval overestimate.
The quintic blend weight has a known value range and bounded first and second
derivatives, but direct interval polynomial evaluation can exceed those ranges.
The new regression fails before intersecting with these analytic bounds and
passes afterward. All ten range controls, five zero-tail controls, five
selector controls, seven integrator stage controls and 12 original derivative
probes pass. Its staged Python wheel still fails the original ridge-barrier
test before movement: `(0.6, 0.15)` returns Unresolved. The nominal field,
crossing tests and acceptance thresholds remain unchanged. The original native
crossing fails after 642.21 s at the same later stage; analytic enclosure error
falls from `0.017355738516676638` to `0.00034293200949250525`, still above the
unchanged limit. All three staged-wheel training-start controls pass.

The `world-footprint` experiment fixes a separate refusal at `(0.6, 0.15)`.
The fixed tile-count search shrank in world units as the source mesh refined,
eventually ending before the nearest Shore. The corrected search covers the
valid disk from every query while retaining the existing visit and leaf caps.
World widths use floating-point units rather than overflowing a browser-sized
cell count. The recorded ridge-start regression changes from Unresolved to
Regular, with error `0.0019229919458484956` against its unchanged approximately
`0.00414` limit. The width control and all four source-cover controls pass.
The unchanged staged-wheel ridge-barrier test passes its full 200-step rollout
in 1107.82 s, with no source drift. This repairs its immediate starting-point
refusal but leaves substantial performance work. The recorded
rollout point `(0.3075204398270623, 0.02876420131721053)` remains Unresolved;
this search correction does not resolve that field-comparison failure.

The `centered-enclosure` experiment addresses lost cancellation in the blend
enclosure. Retained weights form a partition of unity, including derivatives.
Subtracting a fixed Taylor jet from every level and restoring it once preserves
that identity during interval evaluation. A new common-field control reports a
spurious error of `162.8069792730034` before the correction and passes afterward.
Five zero-tail, five selector, 11 coherent-field, ten range, seven integrator
stage, the cut-locus and both proved/uncertain gate controls pass. All 12
original derivative probes and three staged-wheel training-start controls pass.
The nominal field is unchanged. Its original native crossing fails after
531.95 s at the same later query, with analytic enclosure error
`0.00038503165859504236`. The common-field correction therefore does not repair
this trajectory. All 25 selected Python physics tests pass in 1583.78 s on this
snapshot, with the original crossing and continuity tests excluded from that
selection. This is not a full-suite pass.
Its WASM bridge check succeeds with zero errors and 19 warnings; this is
compilation only, not browser execution or artifact-size acceptance.

A single busy-host query-cost diagnostic uses this candidate's staged wheel at
`epsilon=1e-4`. It queries `(0.6 + i*1e-6, 0.15)` in index order
`[0, 0, 1, 2, 3, 4, 0]`. The cold query takes 7.17 s, neighboring queries take
0.089–0.419 s, and exact cached repeats take 21–33 microseconds. All return
Regular, with source hashes unchanged. These measurements expose remaining
cost work; they are not browser or runtime cost acceptance.

The focused `enclosure-trace` replay confirms the later native refusal after
240.54 s. At the first retained level, scalar range is approximately
`[-7.57e-7, 7.56e-7]`; the resulting disagreement-weight Hessian interval reaches
roughly `1.28e16`, despite a nominal Hessian on the order of `1e7`. The next
level's scalar bounds are substantially tighter. This motivates the
`scalar-precision` experiment: try at most two additional source levels after
finding a wide finite fallback, retain the narrowest certified interval, and
include exact epsilon in the nodal cache key. This refines the oracle data;
the public thresholds, cut tests and outer refinement budget remain unchanged.
The replay still refuses after 253.43 s, but reduces analytic enclosure error
to `2.7973162196900137e-6`, just above the unchanged approximately `2.50e-6`
limit. It also retains the other recorded rollout refusal. A separate
`finer-source` experiment keeps the canonical source mesh four levels finer
than its coefficient mesh after the epsilon floor is reached. It encounters
missing source samples on the other recorded rollout replay and times out at
900 s on this native point. Neither experiment establishes crossing acceptance.

The `strict-scalar` variant reduces the internal coefficient uncertainty target
from one eighth to one thirty-second of the existing scalar budget, retaining
the same two extra source attempts. It passes the native-point replay in
194.22 s, with total error `3.5539708944324027e-7` below the unchanged
approximately `2.50e-6` limit. All 12 original derivative probes and three
training-start controls pass. Its original native crossing still fails after
639.26 s, now at `(0.32051477807538159, 0.04736288660490145)`. Remaining Python
runs were interrupted for contract review; they are not passing validations.

The `capped-source` counterfactual passes the earlier `(0.30752, 0.02876)` point
but exposes a discrepancy at `(0.35715, 0.06381)`, and takes 893.83 s to pass the
native-point replay. At fixed spatial level 20, the `budget-trace` diagnostic
finds no contour-distance change when raising the starting iteration budget
from its default through 16384. This does not prove higher-level spatial or
reference correctness. The separate `confidence-refinement` prototype fails a
new derivative-join control and is not an accepted alternative.

Before adoption, finish both original crossings, repeat the applicable checks
on one final source snapshot, validate live training, settle query/memory cost,
and finish browser artifact and project checks. Single-point Regular results
do not establish trajectory validity or merge readiness.

## Reproduction

[Machine records](geometry-145-refinement.json) contain source pins, binary
hashes, limits and verdicts. The
[source archive](experiments/geometry-145-refinement/sources.zip) preserves the
exact Rust source bytes for each named snapshot. Three representative readable
patches remain; redundant generated patches were removed without discarding
their archived sources. These patches compare with the source recorded by
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
