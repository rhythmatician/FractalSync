# Deadline and action bounds for navigation

The Rust check evaluates necessary conditions for a supplied arrival task. Give it
a lower bound for the whole accepted target set in the rollout's metric and a
deadline in canonical audio samples. A hard deadline exclusion needs a positive
intrinsic speed cap that applies throughout the relevant rollout. A configured
throttle or a speed measured after the run does not supply this cap. Replay and
research callers keep numerical Shore estimates tagged as estimates.
[Feasibility issue](https://github.com/rhythmatician/FractalSync/issues/169),
[navigation proofs](174-intrinsic-target-bounds.md).

For a certified lower length \(B\), duration \(T\), and a certified or enforced
speed cap \(V\), any arrival requires \(B\le VT\). For \(T>0\), a path with
square-integrable speed must also satisfy \(\int K\,dt \ge B^2/(2T)\). This
kinetic action is distinct from the ledger's signed work. A separate opt-in
crest check compares a lower work deficit on the first-arrival prefix against
an upper bound on continuous control work plus any independently bounded
impulse work. When impulses may occur and their work is unknown, this crest
check is unavailable. [Work, action and impulse assumptions](169-crossing-work-cap.md).

The implementation uses the canonical 48 kHz sample clock. It rounds elapsed
seconds and travel capacity upward, and the displayed minimum duration and
action downward. A task is **ruled out** when a supported necessary condition
fails, **insufficient evidence** when required inputs are absent and no supported
check excludes it, and **not ruled out** otherwise. At zero samples the travel
capacity is zero and the action lower bound is unavailable. These statuses
concern the supplied task assumptions. Actual route selection and dynamic
reachability require their own evidence.
[Rust assessment](../../runtime-core/src/crossing_feasibility.rs).

## Flat-Shore trial

For the analytic plane Shore \(r=0\), floor \(\varepsilon=0.25\), and zero
anisotropy, use starts at \(r=\pm0.5\) and a supplied speed cap of one. An exact
crossing needs intrinsic length \(2\operatorname{asinh}(2)\approx2.88727\).
The Rust lower bound is 2.19722 under explicit crossing evidence.
The two-second point task is ruled out. At four seconds the task is not ruled
out. The test independently integrates a constant-speed witness over the
four-second interval. The recovered length is 2.88727 and its kinetic action is
about 1.04204. Reversing the path preserves length. A quadratic time warp
increases action at the same length, and a tangential leg on the Shore makes
the route longer.

Replacing the exact target with a 0.25 intrinsic arrival radius lowers the
certified task bound to about 1.94722. The two-second exclusion no longer
follows. This is a comparison against the whole acceptable set, even though the
witness above goes to the center. Two distinct endpoints on the Shore still
have a positive displacement bound. The report also distinguishes the loss of
the crossing sum when crossing evidence is unavailable.
[Reproduction fixture](../../runtime-core/examples/check_navigation_feasibility.rs).

## Replay and current provider

The same check reads an actual read-only raster cockpit snapshot. Its optional
Shore-distance value remains visible with provider metadata and yields
insufficient evidence for an arrival deadline. The retained-field replay uses
the canonical clock to express its 400-step, eight-second arrival window. Its
operational scalar changes sign on the driven replay. It still reports
insufficient evidence for the true-Shore arrival task, because the archived
field has unknown reference correspondence and this rollout enforces no
intrinsic speed cap. The added JSON field is diagnostic. Training and replay
execution do not use it for filtering.
[Retained replay](../../runtime-core/examples/replay_bounded_motion.rs),
[asset and policy provenance](../validation/bounded-operational-engine.md).

~~~powershell
rtk cargo run --offline -p runtime_core --example check_navigation_feasibility
rtk cargo run --offline -p runtime_core --features experimental-operational-field --example replay_bounded_motion -- tmp/geometry145-retained-fine.bin docs/validation/bounded-reference-policy.json
~~~

The analytic paths and signed-work unit cases make it possible to test the
classification without a navigation policy or participants. The retained
replay supplies a negative control for missing evidence.

### Retained replay length and action

The replay samples each accepted outer RK4 step and its three accepted internal
substep endpoints. At each endpoint it computes kinetic energy from the
recorded velocity and the induced metric of the same frozen field and Physics
profile. It integrates endpoint speeds and kinetic energies separately with
composite trapezoids spaced 960, 480, and 240 canonical samples apart. The three
resolutions use the same accepted trajectory. These estimates can miss variation
between recorded states, especially where the metric changes rapidly.

The JSON reports `duration_samples` through the first positive outer sample
for the driven trial and through the full deadline for the undriven trial.
`covered_samples` is the duration each resolution actually integrated. A
panel needs valid values at all its observed endpoints, including any fine
samples in its interior. The replay keeps each step separate, so an impulse
at a step boundary can use its pre- and post-event speeds at the same sample
without interpolating across the jump. This fixture applies no impulses. A
rejected Physics step still aborts the replay.

For full coverage, `action_floor_from_measured_length` is \(L_q^2/(2T)\),
where \(L_q\) is the trapezoidal estimate of length. With the same trapezoid
weights, the difference between action and this expression is half the
duration times the weighted variance of the endpoint speeds, up to rounding.
The comparison checks the sampled values. A continuous-path lower bound
requires independent evidence. Partial coverage leaves the comparison null.

| Throttle | Duration (samples) | Panel (samples) | Length estimate | Kinetic-action estimate |
| --- | ---: | ---: | ---: | ---: |
| Driven | 179,520 | 960 | 10.359196 | 18.562221 |
| Driven | 179,520 | 480 | 10.359415 | 18.562725 |
| Driven | 179,520 | 240 | 10.359470 | 18.562852 |
| Undriven | 384,000 | 960 | 9.551215 | 6.956353 |
| Undriven | 384,000 | 480 | 9.551942 | 6.957673 |
| Undriven | 384,000 | 240 | 9.550928 | 6.955754 |

All six estimates have full coverage. At 240 samples, the driven
sampled-length comparison is 14.347408, below the action estimate of
18.562852. The signed integrated work is 17.426484, which has units of
energy. In the undriven trial, the last refinement moves both estimates down.
Its signed integrated work is \(-0.695625\) and its mechanical energy changes
by \(-0.445434\). The replay preserves this ledger discrepancy in the output.
The true-Shore deadline assessment remains `insufficientEvidence`.

The synthetic tests check refinement against exact variable-speed integrals,
a step-boundary impulse, and a gap with a missing step and a missing endpoint.

~~~powershell
rtk cargo test --offline -p runtime_core --features experimental-operational-field --example replay_bounded_motion
~~~

## Training curriculum check

Run `python backend/train.py --check-curriculum-feasibility curriculum.json`
to review timed point targets before a training run. The manifest is a JSON
array. Give each case a unique `id`, a `start` and `target` in Map coordinates,
and `startSample` and `deadlineSample` on the same canonical audio clock.
`intrinsicTolerance` defaults to zero. `kineticActionBudget`, when supplied,
caps the integral of kinetic energy over the arrival interval.

```json
[
  {
    "id": "ideal-inside-to-outside",
    "metric": "idealDistanceGraph",
    "start": [0, 0],
    "target": [2.125, 0],
    "startSample": 48000,
    "deadlineSample": 96000,
    "kineticActionBudget": 2.0
  }
]
```

For an explicitly ideal task, `idealDistanceGraph` uses certified bounds for
the ideal Mandelbrot Shore. The example's one-second interval needs at least
121.16 units of kinetic action against a budget of two. Extending the deadline
to sample 4,848,000 gives a 100-second interval and lowers the bound to 1.21,
leaving that budget open.
The default `runtimeMap` metric retains the candidate with insufficient
evidence because the ideal distance bound lacks runtime metric correspondence.
The deadline check also reports insufficient evidence without a speed cap
that applies to the rollout. The manifest accepts no speed-cap claims.

The CLI exits after writing the Rust assessment and its provenance for each
case. `eligibleCaseIds` includes every case that survives the necessary checks
for further review. The ordinary `--use-curriculum` orbits are untimed.

An offline retained-field trial also applies the action inequality to paths
that stay within a declared window in the nominal spline metric. The first
target needs at least 0.59214 units of action in one second and 0.00592 over
100 seconds. This supports a regional exclusion under an action budget of
0.05 and the stated metric and containment premises. Default `runtimeMap`
curriculum still needs evidence that those premises apply to the rollout.
[Retained Map experiment](174-operational-route-enclosures.md)

The [default raster trial](174-raster-route-bounds.md) also certifies a
pointwise metric lower bound directly from full bicubic support. Its
one-second ambient-ball task needs at least 0.06077 units of action, even
if a path first leaves the window. This fixture uses the default Physics
profile and a fixed raster generation.
