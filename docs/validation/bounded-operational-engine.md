# Bounded operational engine integration

This change extracts an explicitly selected, default-disabled prepared-field
engine from PR #148 onto the current base. It does not close #145 or change the
base provider. Each rollout owns one immutable field, Physics profile, and
frozen reference-refusal snapshot; attempts commit position, velocity, time,
work, and energy together only after all stages and the weighted endpoint pass.
A missing differential stencil refuses as `GeometryUnavailable`; there is no
fallback to the base provider during an operational rollout.

Python consumers can run `python scripts/build_bounded_stubs.py OUTPUT` and
point `MYPYPATH` at `OUTPUT` to check the combined default and feature-only
API. The overlay is assembled from the canonical default stub and the separate
experimental declarations; the default runtime stubs stay unchanged.

The field is a smooth operational scalar, not a certified Mandelbrot distance.
Reference-distance accuracy and reference-cut coverage outside the supplied
snapshot are unknown. The exact known cusp and cut at `(0.25, 0)` and `(-1, 0)`
remain refused, as do the snapshot's reference-singularity regions. The policy
does not claim complete cut-locus coverage. Shipping adoption, coverage-loss
behavior, reference accuracy and production device targets remain separate
decisions, not integration acceptance claims.

The crossing regression retains its physical setup, crossing requirement and
0.05-second absolute timing limit. It interpolates between bracketing committed
states and compares the fine trajectory against a separately run half-step
trajectory. Analytic linear controls include the sample phase that caused the
old strict timestamp-gap comparison to fail.

`scripts/replay_distant_feature_failure.py` preserves the historical adaptive
provider failure against a clean checkout of its exact recorded source. It
expects that historical test to fail at the recorded point; it is not a test
of this engine or a claim that the on-demand provider is fixed.

The following checks are required before this candidate is ready: exact-source
default and experimental binaries; original default crossing and energy tests;
operational crossing and energy replay on a frozen asset; analytical, join,
refusal and transaction tests; Python and WASM parity; and recorded source,
binary, asset and policy hashes. A measured field discrepancy is not a certified
reference error bound.

The frozen-asset replay uses the #148 starting positions and launch rules for
the driven/undriven Controls case and the underpowered/above-barrier Python case.
Its crossing verdict concerns the operational contour. Reference Shore crossing
is separately unverified for this extraction. The two retained high-curvature
and near-Shore energy replays compare absolute energy excursion and drift with
the original `0.05` limit. The crossing replay also checks absolute energy
excursion against that limit.

The fixed 1,000-point neighboring check and explicit joins retain the `1e-4`
tolerance. The neighboring verifier uses Richardson extrapolation at fixed
increments and checks agreement at two resolutions. The report also exposes the
original raw central-difference outcome separately; a failure there cannot be
described as a passing raw diagnostic. These tests check field self-consistency,
not fidelity to reference derivatives.

## Recorded local integration results

On Windows build 26100, Intel Core i9-14900, Chromium 153.0.8010.12, source
`0c71d63948f29265a184011b9a8ed1b93e77995a` was clean during builds. The default
and feature-enabled wheels each passed 189 backend tests with no skips; default
Rust passed 55 tests and feature-enabled Rust passed 69. The first simultaneous
backend run produced one trainer export-message failure; its isolated rerun
and the full feature-enabled suite run serially passed. Default frontend tests
and build passed after generating the default WASM package; some pre-existing
model-inference checks returned early because model/baseline files were absent.

The operational asset has field ID `9721bb7647a0b8e6`, SHA-256
`bc2604771c2cc48fb71cda6cb22c6e4338a776b6591fa923dd065977817ad9fb`;
the policy SHA-256 is
`7931b4d06e7d1aab6e60ef35b065c5594b00bad216ccc4146dee55925d9dd1cd`.
The [frozen policy](bounded-reference-policy.json) and
[preparation recipe](bounded-field-preparation.json) are included here. To
reproduce the asset without adding the adaptive provider to this change, run
the following in a separate clean checkout of PR #148 source `e780e795`;
the output path must not already exist:

```text
cargo run --release --manifest-path runtime-core/Cargo.toml --features experimental-operational-field --example operational_contour_prepare -- docs/validation/geometry-145-operational-retained-fine-experiment.json NEW_ASSET_PATH
```

Check its SHA-256 and field ID against the values above before replay. The
previous recorded preparation took roughly 218 seconds and produced a
77,405,056-byte asset; preparation is outside all live timings. The engine
accepts supplied prepared assets and does not build or fetch one at runtime.

The Rust Controls crossing and Python barrier crossing pass on that asset;
their reference crossings remain **unverified**. The four interpolated Python
times are `3.8632321511`, `3.8632678178`, `3.8632706835`, and `3.8632705541`
seconds. High-curvature and near-Shore maximum absolute energy excursions are
`2.29e-14` and `2.43e-11`, below `0.05`. Across 1,000 fixed points, the
Richardson neighboring Hessian error is `3.00e-6`; the join error is `4.80e-6`.
The separately retained raw central-difference Hessian diagnostic fails at
`2.34e-4` against `1e-4`. Neither diagnostic measures reference fidelity.

For one 10-step frozen-asset trajectory, native Python and headless Chromium
match within `1e-9` for scalar, metric, potential, state, and ledger; the
observed maximum difference was zero, without requiring bitwise parity.
The local native run measured first prepared field query `9.2 µs`, query p99
`0.2 µs`, maximum sampled query `4.7 µs`, first complete step `46.9 µs`,
step p99 `63.8 µs`, and maximum step `231.5 µs`. Loading took `123.9 ms`;
preparation was not timed here. These are fixed-region samples without cache
stress. One complete step reported 160 geometry evaluations and an upper bound
of 5,760 coefficient reads.

The browser measured first **probe** `1.2 ms`, probe p99 `100 µs`, maximum
probe `400 µs`, first complete step `500 µs`, step p99 `200 µs`, and maximum
step `500 µs`. A probe includes metric, potential, and JSON serialization; it
is not the single geometry-query metric in the provisional target. Browser
timers are quantized, the sample was not cache-stressed, and the measurements
do not establish a hard deadline. WASM initialization, fetch and asset decoding
took `21.5`, `98.3`, and `136.3 ms` respectively, outside the timed calls.

## Committed mechanical endpoints

The experimental Rust method `FrozenRollout::attempt_step_with_trace` returns
the starting state, checked internal endpoints, and final committed endpoint
with their mechanical times. Euler and single-step RK4 expose two endpoints.
Four-substep RK4 exposes five. The step must pass its ordinary endpoint and
ledger checks to return the trace. A second-substep refusal leaves the prior
state intact and returns an error. The trace uses fixed storage and shares the
ordinary step's geometry evaluations.

These endpoints give #179 a source for transport experiments along accepted
integration steps. The trace samples the discrete rollout. A continuous path
or a transport error bound between samples requires separate analysis.

## Read-only Map route guidance

`FrozenRollout::route_guidance` and the experimental WASM method
`BoundedOperationalRollout.route_guidance` start from the current mechanical
position. The WASM method accepts a JSON request with `window` as two
coordinate corners, `gridSteps`, a `target` with `center` and `radius`, and
an optional `lengthBudget`. Both the start and center must lie in the
window, and the window must fit in the open mechanical disk and the prepared
field's chart. A radius of zero specifies a point. The lattice attaches an
off-grid start and target to the corners of their containing cells. It uses
up to 64 intervals per axis, each at most two spline cells wide.

The planner uses the prepared field's [Bernstein edge enclosures](../research/174-operational-route-enclosures.md).
It refuses a whole candidate edge if its coordinate box lacks a stencil
or overlaps a frozen refusal box or canonical reference point. Bounding by
the box may also refuse edges that miss the unsupported region. It checks
ambient arrival independently of the metric. When rounding leaves a point
ambiguous at the ball boundary, it can inform the lower bound but cannot
witness arrival. The search orders equal priorities by node index.

The report distinguishes arrival, a route witness, a graph-budget
exclusion, and absence of a certified route. The cost bracket applies to
the admitted finite graph in the exact stored-spline metric. The returned
bearing uses the compiled start metric. Potential barriers, travel time,
continuous Physics correspondence, and human driving are outside the
geometric result.

The browser check initializes a rollout near the cusp and commits a Physics
step. Chromium found a 79-vertex route from that committed position to the
`0.001`-radius target. It certified 16,518 edges and refused two. The graph
minimum lies between 6.80863 and the chosen route's upper length of 35.50192
in the stated metric. Sharing repeated cell enclosures within one request
reduced this local query from 606.4 ms to 60.0 ms and produced the same route
report. Those are single-machine timings after asset and WASM loading. The
test confirms that routing leaves the mechanical state unchanged. Use the
checksum-locked fine asset and policy above to reproduce it.

```powershell
rtk proxy wasm-pack build wasm-orbit --target web --out-dir ../tmp/map-guidance-web -- --features experimental-operational-field
rtk proxy node scripts/check_bounded_browser.mjs tmp/geometry145-retained-fine.bin docs/validation/bounded-reference-policy.json frontend/node_modules/playwright/index.mjs tmp/map-guidance-browser.json tmp/map-guidance-web --verify-cusp-route
```
