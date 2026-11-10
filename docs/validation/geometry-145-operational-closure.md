# Geometry #145: close observed internal contour edges

October 30, 2026. A reproduced contour truncation is fixed by following sampled
edge crossings into neighboring reference cells. Frozen source-comparison
failures decrease from 720 to 52, with no change to the field evaluator, Physics,
test bodies or thresholds. The PR remains unready. Experimental Python validation
is still 25 passed, two failed, and the default shipping profile also fails CI.

## Independent regression and bounded fix

A thin extension leaves a sampled disk and crosses the edge of a refined cell.
The next cell has homogeneous coarse corners, and coefficient rows miss the
extension. Previously its observed contour stopped at that shared edge. A control
above the flat extension returned `0.3172079097` instead of approximately `0.192`.
The regression was run red before implementation and is preserved in the archive.

With explicit `contour_cell_closure=true`, a finite queue schedules each reference
cell at most once. A sampled crossing on an internal edge schedules the incident
neighbor. The queue never extends the fixed reference rectangle. All added cells,
classifications and segments consume the existing limits; exhaustion refuses
preparation rather than publishing a partial field.

Closure mode uses one global fine-lattice coordinate expression for coarse
corners, fine nodes and contour corners. Shared edge samples therefore have
identical coordinates regardless of traversal. The analytic control produces
identical asset bytes from two different coarse partitions of the same fine
lattice, and a budget one query below its measured use refuses. This same-build
control adds no bitwise native/WASM parity requirement.

The option defaults off to preserve older frozen recipes and their local
coordinate convention. New recipes explicitly enable coefficient-witness seeding
and closure. This is closure of **observed sampled internal edges**, not proof
that every component or fine boundary feature has been found. No distance upper
bound or reference singularity certificate is inferred from it.

## Frozen preparations

The [refined recipe](geometry-145-operational-closure-fine-experiment.json) was
selected for replay before acceptance validation. The
[coarse recipe](geometry-145-operational-closure-coarse-experiment.json) supplies
the source-sensitivity comparison. Both retain coefficient spacing `0.0005`,
final contour spacing `0.000125`, 8,192 iterations, eight bisections and the
original 50-million-query, 500,000-segment and 80-million-coefficient-byte limits.

| Measurement | Coarse source | Refined source, selected |
| --- | --- | --- |
| Field ID | `71eaa19d09e17bac` | `cc35adf2aad104b1` |
| Source coarse spacing | 0.002 | 0.001 |
| Additional closure cells | 1,322 | 3,583 |
| Total refined cells | 14,275 | 32,378 |
| Membership queries | 23,972,542 | 35,669,103 |
| Segments | 359,548 | 357,374 |
| Process preparation time | 161.65 seconds | 154.44 seconds |
| Sampled OS peak working set | 216,367,104 bytes | 216,371,200 bytes |

Both artifacts remain 77,405,056 bytes with 77,405,000 coefficient bytes.
Selected artifact SHA-256 is
`e2b8ae78a38a950721598afb23972f80776b72dac9f8632d82b12c9a34d11847`.
The [replay manifest](geometry-145-operational-closure-replay.json) binds asset
and frozen policy hashes. Timed Physics profile remains `54b525d2266d27eb`.
Assets, coverage and profiles remain fixed throughout each fresh rollout.

## Independent outcomes

| Gate | Result |
| --- | --- |
| Fixed source-sensitivity comparison | 52 / 1,768 failures, previously 720 |
| Neighbor-grid comparison | 27 / 1,000 failures, unchanged |
| Near-Shore source comparison | 25 / 765 failures, previously 693 |
| Three named control jets | All agree exactly across source grids |
| Fixed reference-realm scalar audit | 765 conclusive realms, zero sign mismatches and counterexamples |
| Scalar-distance upper bounds | Unknown |
| Reference differential fidelity and full singularity coverage | Unknown |
| Frozen per-run expectations and coverage | 35 / 35 pass with complete coverage |
| Operational crossings / reference crossing pairs | 21 / 21, separately reported |
| Original native crossing body | Passes unchanged with verified reference crossing |
| Original Python physics file | 25 pass, two fail, source unchanged |
| Neighboring differential self-consistency / joins | Pass unchanged `1e-4` tolerances |
| Native/WASM parity, query invariance, known-cut refusal | Pass declared tolerances and preserve state |

Source sensitivity retains its previous distance-unit formula and declared
coarse source spacing. It is neither coefficient-resolution convergence nor a
reference-distance bound. Remaining component failures overlap: scalar 21,
scaled gradient 35 and scaled Hessian 50. Maximum sensitivity is `0.00540085`.
The smaller failure count is not promoted to a general quality certificate.

Normalized neighboring Hessian error is `3.002e-6`; join error is `4.802e-6`.
The raw finite-increment central diagnostic remains failed at `1.853e-4`.
High-curvature and near-Shore energy excursions are `2.287e-14` and `2.428e-11`,
below the unchanged `0.05` limits. These establish operational behavior under
the tests, not complete reference fidelity.

The original sampled crossing times remain `3.864`, `3.864`, `3.8635` seconds.
The strict gap-refinement assertion fails, and scalar quality remains unknown
in the legacy adapter. No estimate is manufactured from the now-identical
quality-probe jets. A separate five-timestep segment-event diagnostic converges
toward `3.863270565` seconds, with maximum energy excursion decreasing from
`2.412e-5` to `6.517e-10`. It replaces neither the original assertion nor a
reference Shore event. Reference crossing pairs concern the reconstructed
piecewise-linear committed trajectory, not an exact ODE solution.

## Evaluation cost

Device: Intel Core i9-14900, Windows 10.0.26100, headless Chromium 153.0.8010.12.
Native cache-stressed query p99 is `4.0 / 3.0 / 4.2` microseconds across three
fresh processes; complete-step p99 is `159.1 / 150.8 / 65.4` microseconds.
Browser p99 is 25 microseconds per query and 165 microseconds per complete step.
All pass the provisional 100-microsecond query and 2-ms step p99 targets.

Browser first prepared query is 155 microseconds; stressed-query maximum is
1.070 ms. First/maximum complete step is 1.375 ms. Native first queries are
`5.2 / 4.5 / 4.5` microseconds, stressed-query maxima `52.6 / 8.1 / 8.5`
microseconds, step maxima `544.4 / 428.5 / 528.8` microseconds. Full distributions
and unavailable-query timings remain in the machine report. These are p99
experiments, not hard deadlines or ratified shipping device requirements.

Each timed step retains 160 geometry evaluations and at most 5,760 coefficient
reads. Native loading takes `160.16 / 137.64 / 127.95` ms. Browser fetch takes
89.40 ms, decoding 132.77 ms and WASM initialization 20.88 ms, excluded from
query timings. Native sampled peaks are `248,217,600 / 250,904,576 / 250,912,768`
bytes. WASM capacity remains 160,301,056 bytes, separate from JavaScript input,
64 MiB stress scratch and other browser allocations. Preparation is reported
above and also excluded from query cost.

## Shipping gate and verification

At prior committed checkpoint `2db0b80`, the
[default-profile CI run](https://github.com/rhythmatician/FractalSync/actions/runs/41148404070)
finished with Rust **98 passed, one failed** and backend **187 passed, one failed**.
The original Rust crossing refuses a cut locus at
`(0.2147416653, 0.1277401900)`. The original backend crossing refuses unresolved
geometry at `(0.3565938106, 0.0640465676)`. Frontend tests and builds passed;
separate type/stub workflows also passed at that checkpoint. These results are
for the shipping profile, not the experimental asset. Green experimental
crossings do not clear these default-profile failures.

CI previously omitted feature-gated controls. The Rust job now adds a separate
bounded-field control step using `!cancelled()` so it runs even after a default
test failure. The default step is unchanged; no failure is suppressed. The exact
new release command passes locally with **30 controls**, as does the debug run.
Default native compilation, workflow parsing, formatting and diff checks pass.

```powershell
rtk cargo test --manifest-path runtime-core/Cargo.toml --release --features experimental-operational-field operational_field --lib
```

The [machine report](geometry-145-operational-closure.json) records identities,
hashes and outcomes. The [raw archive](experiments/geometry-145-operational/closure-evidence.zip)
contains both preparations, every fixed source comparison, original failures,
three native reports, browser parity, event diagnostics, the synthetic red
regression and the shipping CI snapshot. Python/WASM reuse unchanged evaluator
and mechanics binaries with their actual hashes recorded separately. No shipping
math, goldens, versions, original tests or thresholds change in this follow-up.

Source retention/stability at the remaining points, supported scalar-quality
evidence, original crossing refinement, default-profile CI failures and pending
ADR 0005 adoption decisions remain merge blockers. Earlier assets and failures
are preserved.
