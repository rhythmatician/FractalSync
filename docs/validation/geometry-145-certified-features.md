# Certified component features and affine critical families

This extends the [bounded-refinement experiment](geometry-145-bounded-selector.md).
The field replacement remains unshipped. Quality thresholds, membership budgets
and both original crossing tests are unchanged.

## Complete cardioid candidates

The experimental provider now evaluates component inclusion with outward
interval arithmetic. A complete global distance enclosure bounds the scalar
selected by the previous root guess. Approximate roots therefore select a value
inside a certified interval; they do not establish completeness.

Two rational circle charts cover the primary cardioid boundary. Subdivision
retains every possible stationary foot within the unchanged nearest-feature tie
band. An interval derivative can exclude a region only when it proves that no
stationary point lies there. A distance lower bound can exclude a region only
when it exceeds the certified global upper bound plus the tie band. Exhaustion
returns no cover.

A regular-direction certificate requires every pair of enclosed displacement
cones to agree, including each cone with itself. Zero-distance normalization and
nonfinite bounds refuse certification. Each cover stores its original query;
callers cannot supply another query when checking agreement.

Thirteen standalone tests pass. They include rational-chart seams, the origin's
cusp projection, exact and nearby medial-axis cases, and two known equal-distance
opposite feet. Removing one chart makes the opposite-feet regression fail.
Independent Spec, Health and Standards reviews found no concrete unsafe pruning;
their API and verification findings were addressed.

The actual provider candidate passes 37 targeted tests and all twelve derivative
checks in 82.57 seconds. All twelve center jets match the preceding canonical-node
candidate exactly. These local results do not certify the complete fitted and
refined derivative error calculation. Propagating the scalar enclosure through
that calculation remains an adoption obligation.

## Affine whole-box proof

The new proof retains the linear dependence of the critical orbit on the
parameter. It encloses each state as `center + derivative * delta_c + remainder`.
The remainder includes linear propagation, the complete quadratic remainder,
coefficient-update rounding and both machine-orbit and center-update rounding.

Outside certification requires escape before update 512, preserving the original
pre-update test at the minimum ordinary tile budget. Inside certification requires
all 512 prefix enclosures to remain bounded, followed by a strict periodic return
disk with bounded intermediate steps. Bounded prefixes alone never certify
interior membership. The original disk prefix and affine prefix share their
periodic-trap and rounding implementations.

The combined experimental candidate passes 42 targeted tests. New regressions
exercise the public fallback, compare independent ordinary nodes, prove a whole
period-three interior box, and distinguish first escape at updates 511, 512 and
513. Actual-source mutations that allow update 512 or disconnect the fallback
make the corresponding tests fail.

The recorded training query at
`(0.3075204398270623, 0.02876420131721053)` now returns Regular with a cold query
time of 51.81 seconds. Its scale and error satisfy the existing gates. The same
candidate passes all twelve derivative checks in 13.43 seconds. The candidate
without the affine proof times out on that training query after 180 seconds.
This resolves one recorded query; it does not establish general training
acceptance or an acceptable live-query cost.

Every serialized derivative-probe row also matches when cases run in reverse
order, each in a fresh process. This checks the sampled cache-context behavior.
The original native crossing run and the full energy sweep both time out at
1200 seconds. Eleven energy cases finish and pass before that combined timeout.
Separate low-curvature, high-curvature, near-Shore and inward sweeps now complete
and pass all twelve cases with the unchanged 0.05 energy limit. The inward-only
sweep completes in 532.03 seconds. The original provider suite and recorded Python crossing query time
out at 180 seconds. None of these timeouts is a passing validation result.

## Isolated production proof

The affine whole-box optimization is also tested separately from the replacement
field. All 64 sampled production serialized jets are identical. The first library
run has 94 passes and the same native crossing failure. Both new mutation checks
are red-capable. Full verification is recorded separately as it completes.

The separate timing runs sum to 27.13 seconds before and 27.50 seconds after.
Other validation jobs were running, so these timings establish no general speedup.
Provider/controller/snapshot stamps are unchanged because no sampled numerical
output changed. Full production validation gives 257 Rust passes, one known
native crossing failure and one ignored test. The rebuilt Python extension is
byte-identical to the Python build target; its physics suite gives 26 passes and
the same recorded crossing refusal. Mandatory parity, architecture and canonical
surface gates pass. Any adoption of the replacement field still requires its version,
golden and binding updates.

## Reproduction

Starting at `d8ec37a`, apply the five bounded-refinement patches, then
`geometry-145-certified-features.patch` and `geometry-145-affine-prefix.patch`.
That reconstructs the candidate used by the derivative, training, crossing and
energy runs. Apply `geometry-145-affine-prefix-verified.patch` last for shared-proof
refactoring and the two additional regression tests. All five final source files
were checked against the exact patch reconstruction in an isolated directory.

The accompanying [JSON](geometry-145-certified-features.json) separates terminal
results from pending work and preserves exact variant ownership. The
[production comparison](geometry-145-production-affine.json) records all 64 jet
hashes and the separate timing measurements. Original crossing completion,
the recorded Python crossing query, full field certification and live cost remain
merge blockers.

The subsequent [enclosure follow-up](geometry-145-ranged-field.md) propagates
analytic coefficient uncertainty through the reconstructed field and output
error gate. Its validation is recorded separately from this candidate.
