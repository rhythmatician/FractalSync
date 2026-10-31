# Geometry #145: retain sampled geometry under source subdivision

October 30, 2026. Fixed physical probe blocks prevent a finer source partition
from forgetting components already observed by the coarser preparation. All
1,768 fixed source-comparison jets now agree exactly, previously 52 failures.
Whole-asset magnitude increases fall from 172,562 to zero. This clears the
retention regression, not the remaining reference-fidelity and merge gates.

## Independent regression and implementation

The analytic control contains a large circle that seeds a coarse cell and a
small isolated circle sampled inside that cell. After subdivision, the isolated
circle misses the new seed locations. Edge closure alone cannot recover a
disconnected component. The control was run red: its returned distance increases
from `0.2461023929` to `0.3522324806` despite unchanged final contour resolution.

The new `refinement_block_cells` parameter queues aligned blocks rather than
individual cells. Keeping `reference_spacing * refinement_block_cells` fixed
retains the physical probe region when its cells subdivide. Shared fine-lattice
coordinates, coefficient witnesses, classifier and final resolution remain fixed.
Denser initial samples can add blocks; they do not shrink the retained regions.
The rule applies during offline preparation only, within the frozen reference
rectangle. It adds no live extension, fitting or reference work.

Block sizes must be powers of two, tile the reference grid, require closure when
larger than one, and satisfy the existing effective subdivision limit of 32.
All added visits use the existing membership and segment budgets. The default
size is one, preserving previous recipes. The new coarse artifact reproduces
the preceding artifact SHA-256 exactly, confirming compatibility on the full
9,675,625-coefficient dataset.

The regression now produces identical bytes from the two partitions. Invalid
block configurations refuse. A new whole-asset checker compares every stored
coefficient on complete matching grids, rejecting unavailable support,
incompatible mappings and invalid tolerances. It checks signs and magnitudes;
it does not certify the reconstructed field or true Shore distance.

## Frozen preparations and retention results

The [refined recipe](geometry-145-operational-retained-fine-experiment.json) was
selected for replay before validation. The [coarse recipe](geometry-145-operational-retained-coarse-experiment.json)
supplies the comparison. Both retain the same coefficient grid, 8,192 iterations,
eight bisections, final contour spacing `0.000125`, 50-million-query limit,
500,000-segment limit and 80-million-coefficient-byte limit.

| Measurement | Coarse source | Refined source, selected |
| --- | --- | --- |
| Field ID | `71eaa19d09e17bac` | `9721bb7647a0b8e6` |
| Source spacing / block size | `0.002 / 1` | `0.001 / 2` |
| Physical probe-block spacing | 0.002 | 0.002 |
| Visited reference cells | 14,275 | 57,476 |
| Membership queries | 23,972,542 | 37,758,491 |
| Segments | 359,548 | 360,902 |
| Process preparation time | 244.91 seconds | 217.69 seconds |
| Sampled OS peak working set | 216,346,624 bytes | 216,940,544 bytes |

The refined run queues 43,107 cells as block peers. This is a scheduling count,
not a disjoint count of additional visits. Both assets contain 77,405,000
coefficient bytes and 77,405,056 artifact bytes. Selected artifact SHA-256 is
`bc2604771c2cc48fb71cda6cb22c6e4338a776b6591fa923dd065977817ad9fb`.
The [replay manifest](geometry-145-operational-retained-replay.json) binds the
asset and reference-policy hashes. Timed Physics profile is `54b525d2266d27eb`.
Each rollout loads the complete asset and freezes field/profile identities.

| Retention evidence | Before | After |
| --- | --- | --- |
| Fixed source-comparison failures | 52 / 1,768 | 0 / 1,768 |
| Maximum fixed-point jet sensitivity | `0.00540085` | Exactly zero |
| Coefficient magnitude increases above `1e-12` | 172,562 | 0 |
| Maximum coefficient magnitude increase | `0.00350640` | Zero |
| Coefficient sign changes | 0 | 0 |

All 9,675,625 coefficients are checked. The new refined field gains closer
sampled geometry at 223,521 coefficients, with maximum magnitude decrease
`0.04890876`. Thus the fields are not globally identical or globally converged.
Only the fixed comparison jets agree exactly. The `1e-12` diagnostic comparison
tolerance is separate from unchanged public quality and derivative thresholds.

Both preparations use the same final contour lattice and coefficient spacing.
Their agreement does not establish coefficient-resolution convergence, fit
residuals, membership certainty or true-distance upper bounds. Runtime scalar
quality remains unknown. No small legacy error estimate is inferred from this
source-retention result.

## Retained mechanical and reference gates

All 35 per-run expectations pass with complete coverage: 21 operational crossings
and 14 noncrossing controls. All 21 crossing trajectories also have separately
verified ordered opposite-reference-realm pairs. The original native crossing
body passes unchanged. Reference pairs concern the reconstructed piecewise-linear
committed trajectory, not an exact ODE event. Known reference-cut refusal and
state preservation still pass; complete singularity coverage remains unknown.

The fixed scalar audit retains 765 conclusive reference realms and zero sign
counterexamples. This one-sided evidence supplies no scalar-distance upper bound.
Reference differential fidelity is also unknown.

Neighboring self-consistency and joins pass the unchanged `1e-4` tolerances;
normalized Hessian errors are `3.002e-6` and `4.802e-6`. The raw central diagnostic
remains failed at `2.343e-4` and is preserved. High-curvature and near-Shore energy
excursions are `2.287e-14` and `2.428e-11`, below unchanged `0.05` limits.
Native/WASM parity passes the declared tolerances. No cross-target bitwise
requirement is added.

Original Python validation remains **25 passed, two failed**. Sampled crossing
times `3.864`, `3.864`, `3.8635` fail the strict gap-refinement assertion, and
legacy scalar-quality metadata remains unknown. Both test bodies and thresholds
are intact. The separate five-timestep segment-event diagnostic converges toward
`3.863270565` seconds; energy excursion decreases from `2.412e-5` to `6.517e-10`.
It does not replace the original assertion or certify an exact Shore event.

## Evaluation cost

Device: Intel Core i9-14900, Windows 10.0.26100, headless Chromium 153.0.8010.12.
Native cache-stressed query p99 is `3.1 / 3.9 / 4.6` microseconds across three
fresh processes; complete-step p99 is `154.8 / 168.5 / 202.3` microseconds.
Browser p99 is 25 microseconds per query and 190 microseconds per complete step.
These pass the provisional experimental 100-microsecond and 2-ms p99 targets.

The browser nevertheless records a **9.05-ms maximum query**, which is retained
without rerunning to select a smaller maximum. First prepared query is 185
microseconds; first/maximum complete step is 1.630 ms. Native first queries are
`8.8 / 5.3 / 4.1` microseconds, stressed-query maxima `35.4 / 6.9 / 39.3`
microseconds, and step maxima `240.7 / 467.9 / 254.7` microseconds. Browser
unavailable-query p99 is 5 microseconds and maximum 320 microseconds. Passing p99
does not establish a hard deadline or ratify shipping device requirements.

Runtime work is unchanged: 160 geometry evaluations and at most 5,760 coefficient
reads per timed step. The new whole-coefficient comparison runs offline only.
Native loading takes `154.73 / 114.86 / 149.94` ms. Browser fetch takes 129.04 ms,
decoding 155.53 ms and WASM initialization 30.42 ms, excluded from query timings.
Preparation costs are reported above. Native sampled peaks are `250,925,056 /
250,896,384 / 250,875,904` bytes. WASM capacity remains 160,301,056 bytes, separate
from JavaScript input, 64 MiB stress scratch and other browser allocations.
Full first/p50/p99/max distributions remain in the machine report.

## Verification and remaining merge work

All **32 Rust controls pass in debug and release**, including the independent
red/green retention regression and complete-grid/sign/mapping controls. Default
compilation, formatting and diff checks pass. Actual source, binary, artifact
and original-test hashes are recorded. Python/WASM reuse unchanged evaluator and
mechanics binaries; the new verifier and builder hashes are separate.

At prior committed checkpoint `9c7eb4d`, the new CI control step passes remotely
with 30 controls. The [same run](https://github.com/rhythmatician/FractalSync/actions/runs/41150341147)
still fails the default Rust crossing, **98 passed, one failed**, and backend
crossing, **187 passed, one failed**. Builds and frontend tests pass. These are
separate shipping-profile failures; prepared-field results do not clear them.

The [machine report](geometry-145-operational-retained.json) and
[raw archive](experiments/geometry-145-operational/retained-evidence.zip) preserve
both preparations, before/after whole-coefficient checks, every fixed sample,
native and browser results, original failures, event diagnostics, the synthetic
red result and the CI snapshot. Earlier assets and reports remain intact.

Source retention is now verified for this frozen comparison. Independent
reconstruction/reference-distance evidence, the original timing assertion,
default-profile crossing failures, complete singularity information and ADR 0005
shipping adoption remain unresolved. No shipping math, versions, goldens,
original tests or thresholds change in this follow-up.
