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
classification without a navigation policy or participants. Actual hard
curriculum exclusions still need a certified target and rollout metric, a
certified or enforced speed cap for the deadline test, and an acceptance
consumer in #138 or #110. The retained replay supplies an explicit negative
control for missing evidence.
