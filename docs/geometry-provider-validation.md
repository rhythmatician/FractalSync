# Adaptive geometry validation

PR #148 was rebased onto main at `32ddb065d943374b68deba18816404acef9029bc`, including the transported camera fix. This record describes the repairs following that rebase. **Issue #145 remains open and PR #148 is not ready to merge.**

## October 23 follow-up: coherent scalar-field experiment

The new candidate evaluates a globally anchored tensor quintic B-spline scalar field at each dyadic level. It accumulates scalar corrections through C2 refinement weights and differentiates the complete expression, including those weights, to obtain `D`, `grad_D` and `H_D`. This addresses the earlier moving-fit derivative mismatch and the discontinuous choice of accepted resolution. It is an **unshipped experiment**, not a replacement for provider 6. No production math, golden vector, quality threshold or crossing test changed. **PR #148 remains blocked.**

The [experimental patch](validation/experiments/geometry-145-coherent-field.patch) preserves the radius-6 candidate against the provider source at `00da849`. Its [measurements](validation/geometry-145-coherent-field-experiment.json) include the alternative radius, refinement, error-weight and membership-budget runs. The patch retains baseline version stamps only to compare diagnostics; these stamps do not identify released experimental semantics. It must not be applied to the shipping branch without resolving the failures, adding production regression coverage, bumping versions and rebuilding parity artifacts.

### What improved

The initial eight public-query derivative probes all pass with the scalar-field candidate. Extending the report to twelve probes adds exact dyadic tile boundaries. The radius-12 replay passes ten; both probes at `(0.390625, 0.05)` are Unresolved, so the diagnostic correctly rejects them even though their finite-difference residuals are small. A single case can now be replayed with `--case NAME`; unknown arguments and case names fail explicitly. All derivative tolerances are unchanged.

| Near-Shore timestep | Default excursion / largest change | Scalar-field excursion / largest change |
|---|---:|---:|
| 0.002 | 0.0217245 / 0.0246512 | 0.00694309 / 0.000933209 |
| 0.001 | 0.0224304 / 0.0244727 | 0.00348478 / 0.000235352 |
| 0.0005 | 0.0228863 / 0.0243545 | 0.00174712 / 0.0000589647 |

All three candidate near-Shore rollouts complete under the unchanged 0.05 limits. Their excursion approximately halves and largest step approximately quarters as the timestep halves. The three high-curvature rollouts also pass. The recorded live-training point is Regular under the candidate with error about `1.149e-7`; this is a point query, not a live training acceptance run.

### Why it cannot ship

The inward rollouts stop after 73/100, 146/200 and 292/400 steps. Their energy errors decrease under timestep refinement up to refusal, but incomplete trajectories fail acceptance. The coarse-timestep refusal is now a named provider-report sample, `coherent_field_inward_refusal`, at `(0.39113334515997950, 0.04985151190569804)`.

The trace separates early field termination from subsequent contour changes:

| Level | Raw distance | Accumulated scalar distance | State |
|---|---:|---:|---|
| 7 | 0.0389900609 | 0.0389900609 | Refining |
| 8 | 0.0390164990 | 0.0390164990 | Refining |
| 9 | 0.0374549954 | 0.0390164989 | Representation stopped adding corrections |
| 10 | 0.0366274154 | 0.0390164989 | Scalar/raw discrepancy about 0.00238908 |

The output discrepancy exceeds the unchanged error bound. Agreement of levels 7 and 8 did not predict the later contour displacement. The output remains refused; accepting the earlier pair as sufficient for this field would hide the unresolved discrepancy. The restored default provider accepts this same coordinate as Regular at level 8, distance `0.0390165011`, with estimated error only `2.71734e-5`. Its strict reported gates pass, but the deeper candidate contour evidence shows that this first-pair agreement is not persistent. This is additional evidence of numerical false convergence, not an exact-distance certificate for the finer contour.

The controlled alternatives did not repair this:

- Expanding from radius 6 to 12 gives the same inward completion counts.
- Resolving the representation twice as finely makes even the initial inward state Unresolved.
- Keeping refinement active using a differentiable coarse/fine distance-disagreement weight also refuses the initial inward state and fails four of twelve derivative probes. It was not adopted.
- Using a common 16,384-to-32,768 membership comparison for every tile gives exactly the same inward and near-Shore energy results as the original candidate. It does not explain this refusal. The iteration cap was unchanged.

A viable replacement still needs a coherent refinement rule with sufficient contour and membership evidence throughout its active support. Tile coefficient consistency, uncertainty cancellation and active-level provenance also need production tests before adoption. Local derivative agreement alone does not establish those properties. The production source was restored byte for byte after every prototype build.

Reproduce the archived candidate only in an isolated checkout whose provider source matches `00da849`:

```text
rtk git apply --check docs/validation/experiments/geometry-145-coherent-field.patch
rtk git apply docs/validation/experiments/geometry-145-coherent-field.patch
rtk cargo run --release --jobs 1 -p runtime_core --example geometry_derivative_report -- --check
rtk cargo run --release --jobs 1 -p runtime_core --example geometry_energy_report -- --case inward --check
```

The expected verdict is failure. The experiment is retained to make the next refinement investigation reproducible, not to bypass acceptance.

Fresh checks on the restored default provider:

| Check | Result |
|---|---|
| Original native crossing | Same CutLocus refusal at `(0.21474166530729011, 0.12774018997553147)`, 18.37 seconds |
| Complete original Python manifold-physics file | 26 passed, 1 failed; same Unresolved crossing endpoint `(0.35659381055433242, 0.06404656757776669)` at level 23 |
| Expanded derivative diagnostic | All 12 probes fail; unchanged tolerances |
| Near-Shore energy diagnostic | All three rollouts complete under unchanged 0.05 limits; recorded nonvanishing jumps reproduce exactly |
| New inward point sample | Repeat determinism and strict Regular gates pass |
| Diagnostic CLI | Selected case emits exactly two probes; missing case values, unknown cases and unknown arguments fail |
| Build and source checks | All three release examples build; Rust formatting, patch applicability and diff whitespace pass; production source, goldens and both crossing test sources have no diff |
| Independent review | No actionable Standards, Spec or Health Regression findings in this follow-up; variant measurements beyond the radius-6 patch are not independently archived implementations |


## October 23 investigation: new refusals and distance-field coherence

Production geometry remains `geometry-provider/6`, `orbit-controller/10` and `debug-snapshot/5`. No production math, golden vector, quality threshold or crossing assertion changes in this investigation. **PR #148 is not ready to merge.** The current query report has 60 rows, adding the two new refusal coordinates at four epsilon values. All 52 matching serialized jets are exactly unchanged from the previous report.

### Native refusal: near-tie crossing, then exhausted coverage

The [actual refinement trace](validation/geometry-145-new-refusal-traces.json) retains competing features through level 8 at `(0.21474166530729011,0.12774018997553147)`. Level 9 cannot contain the distance neighborhood within the bounded footprint, so the provider returns the last suspected CutLocus result. It does not stop merely because the estimated error passed.

An [analytic main-cardioid comparison](validation/geometry-145-native-cardioid-switch.json) distinguishes this case from the earlier false refusal. At the last accepted position, distance to the curved boundary is `0.130146833` and to the cusp is `0.130466164`. At the rejected endpoint, the cusp is nearer: `0.132516815` versus `0.132589169`. The two control-geometry minima tie about 81.264% along the step, near `(0.2150824864,0.1274343911)`, with distance directions differing by about 30.646 degrees. This is evidence supporting suspected nonsmoothness, not a certificate for the full Mandelbrot boundary. It does not justify forcing the endpoint to Regular or weakening the angle/tie rules.

### Python crossing refusal: missing consecutive membership certification

At `(0.35659381055433242,0.06404656757776669)`, membership certification fails at levels 17, 18, 19, 22 and 24. The instrumented tiles at levels 18 through 23 reach the existing 32,768-iteration limit with changing nodes. A changing node only invalidates the query when the existing neighborhood/stability rules require it, so some intervening levels still produce provisional fits.

Level 21 retains suspected competing features. Level 22 resets the convergence history because membership is unresolved. Level 23 has no competing normals, but it is the first certified sample after that reset; its infinite estimated error is therefore intentional. Level 24 again fails membership. Accepting level 23 would remove the required consecutive-level evidence. The [trace report](validation/geometry-145-new-refusal-traces.json) records the budgets, unstable-node counts and final jet. No budget, stability band or refusal rule was changed.

### Rejected derivative experiment: smooth curves still have distance cut loci

A temporary Rust prototype evaluated distance to fixed uniform cubic B-spline contour pieces, with analytic closest-point gradient and Hessian. Existing raw-field convergence and acceptance gates were retained. It passed all eight [local derivative probes](validation/geometry-145-fixed-curve-derivatives.json), unlike production. It was then **rejected** by the unchanged conservative-energy limits. The [controlled comparison](validation/geometry-145-fixed-curve-experiment.json) records all twelve trajectory/timestep pairs for production and the unshipped candidate.

| Near-Shore timestep | Production excursion / largest change | Rejected curve excursion / largest change |
|---|---:|---:|
| 0.002 | 0.0217245 / 0.0246512 | 0.246369 / 0.145100 |
| 0.001 | 0.0224304 / 0.0244727 | 0.246652 / 0.145128 |
| 0.0005 | 0.0228863 / 0.0243545 | 0.246792 / 0.145143 |

The candidate's peak stays at level 8 and the same tile ID on both sides. At timestep `0.0005`, the query moves only `1.1761e-5`, while the nearest foot inferred from `c - D*grad_D` changes by about `0.0019066` and the unit gradient turns about 5.614 degrees. Thus this candidate has a nearest-point branch change inside one accepted resolution region. Making the contour curve C2 does not make its distance field C2 throughout that region. This diagnosis is specific to the rejected model, not proof that every production energy defect has the same cause. The production peak still changes refinement levels. A safe representation must address both accepted-region distance-field coherence and transitions between levels; passing a few local derivative probes is insufficient.

The prototype and trace instrumentation were removed from production source byte for byte. Experimental binaries retain baseline version stamps for comparison but are not released geometry. Goldens and Python/WASM bindings retain their existing semantics.

### Reproducers and current verdicts

The new [Rust energy report](../runtime-core/examples/geometry_energy_report.rs) calls canonical `integrate_step`, `total_energy` and public geometry queries. It records before/after jets at each trajectory's largest energy change. The 0.05 limits are diagnostic copies of the existing energy acceptance limits; this report changes no CI test. Filtering a case shortens diagnosis without altering the original tests.

```text
rtk cargo run --release --jobs 1 -p runtime_core --example geometry_energy_report -- --check
rtk cargo run --release --jobs 1 -p runtime_core --example geometry_energy_report -- --case near_shore --check
rtk cargo run --release --jobs 1 -p runtime_core --example geometry_derivative_report -- --check
```

| Check | Fresh result |
|---|---|
| Native crossing, original test | Same CutLocus refusal, 19.44 seconds; no assertion changes |
| Python crossing, original test and final rebuilt provider | Same level-23 Unresolved refusal, 415.09 seconds |
| Production derivative check | Still fails all 8 probes |
| Production energy report | All 12 trajectory/timestep pairs complete under unchanged limits; targeted three-case near-Shore replay also passes, but its nonvanishing jumps remain visible |
| Rejected curve report | 8 derivative probes pass; 3 of 12 energy cases fail; not shipped |
| Query report | 60 rows pass repeat determinism and strict Regular gates; all 52 prior jets match exactly |
| Checks and review | Both diagnostic examples build and pass Rust formatting; diff whitespace passes; unknown energy-case names fail explicitly. Independent Standards, Spec and Health Regression reviews find no actionable source findings. |

CI at production code `1e692d2` has now completed: native/Python and WASM builds, artifacts, frontend/GPU, types and stubs pass. [Backend CI](https://github.com/rhythmatician/FractalSync/actions/runs/40044138353) reports **187 passed, 1 failed** in 1,303.13 seconds, with the identical crossing refusal. Rust CI reports 78 library passes and the same native failure, stopping before integration targets. The prior local full Rust result below remains 241 passed, one failed and one ignored. No new complete local Python suite is claimed for these diagnostics-only changes.

## October 23 follow-up: bounded footprint repair

The current candidate uses `geometry-provider/6` and `orbit-controller/10`; `debug-snapshot/5` is unchanged. A complete-tile expansion can now reach radius 5, at most 11 by 11 tiles, instead of radius 3, at most 7 by 7. Pruning, containment, membership/stability budgets, refinement limit, tie/angle thresholds and strict Regular quality gates are unchanged. Both crossing test sources are untouched. Regenerated golden vectors change only the controller stamp.

The new three-point public-query regression failed with the old footprint and passes after expansion. It covers the recorded native endpoint, seed-42 live training point and kinetic crossing refusal. These now resolve as Regular at levels 8, 16 and 18 respectively. Seed 42's complete training smoke also passes through the rebuilt extension, in 12.02 seconds. This establishes progress at the recorded points, not trajectory-wide or general training acceptance.

The native failure has a concrete coverage constraint. At level 8, the old square's maximum possible half-width is `0.095703125`, smaller than the roughly `0.1301` nearest-Shore distance even before adding the stencil margin. It cannot contain the query's distance neighborhood. An offline analytic main-cardioid comparison finds local minima at distances `0.130146833` and `0.130466164`, a gap of about `0.000319`. This is a diagnostic comparison with the cardioid boundary, not an independent exact-Mandelbrot acceptance rule. The expanded provider supplies the additional contour/refinement evidence under its existing gates.

The unchanged native rollout gets past that endpoint, then refuses the next candidate `(0.21474166530729011,0.12774018997553147)` as Singular/CutLocus. Its requested scale is `0.01324554337`, resolved scale `0.00048828125`, and estimated error `0.000125463459`. Native crossing remains a blocker.

Expanded refinement also changes the cache working set. The [provider-6 trace](validation/geometry-145-footprint-working-set.json) records 270 distinct tiles for the cut-locus stress query and 163, 161 and 210 for the repaired native, live-training and kinetic-crossing points. At 256 entries, replay misses all 270 tiles again on repeat; 384 and 512 both retain them. The final bound is 384, the smaller tested adequate capacity. This changes retention only. The existing eviction test still inserts more tiles than the actual capacity and passes with all 21 provider tests.

Sampled retained tile payload at 256 entries was about 94 to 482 KiB across those cases. This counts tile structs and their allocated segment/unstable-node vectors, excluding keys, hash-map buckets, combined contours and temporary orbit states. The expansion ceiling grows from 49 to 121 tiles per level, about 2.47 times as many. Current-version timing and whole-process memory evidence is recorded separately from the historical provider-5 measurements below.

The [final provider-6 cost report](validation/geometry-145-footprint-cost.json) uses the 384-entry executable, fresh processes and immediate repeats at epsilon `1e-4`. These samples were collected alongside acceptance work and are machine/load dependent.

| Case | Classification | Cold / repeat milliseconds |
|---|---|---:|
| Cut-locus stress query | Singular | 5776.15 / 15.53 |
| Recorded native endpoint | Regular | 1036.97 / 5.26 |
| Recorded seed-42 live point | Regular | 1702.37 / 3.78 |
| Recorded kinetic-crossing point | Regular | 987.99 / 3.94 |

All 52 serialized jets exactly match between cache capacities 256 and 384. The retention correction removes the measured repeat thrashing; it does not recover the extra cold work introduced by broader refinement. Separate fresh-worker Windows samples peak at 5.0 to 6.4 MiB working set and 1.18 to 2.80 MiB commit across these four cases. These whole-process samples are not a worst-case memory bound.

| Check | Footprint-repair result |
|---|---|
| Complete release Rust suite | 241 passed, 1 failed, 1 ignored with radius 5 and the initial 256-entry cache; all 33 snapshot integration tests pass, 660.44 seconds. The sole failure is the new native endpoint recorded above. |
| Final 384-entry provider suite | All 21 tests pass, including the three-point regression, true singular/refusal coverage and capacity-relative eviction. These focused tests overlap the full Rust suite. |
| Complete Python manifold-physics file | 26 passed, 1 failed with the rebuilt radius-5 extension at 256 entries. Energy tests pass; crossing reaches a new Unresolved point, detailed below. |
| Final rebuilt 384-entry training validation | 9 passed across reset regressions, the complete Controls-v2 smoke file and both existing end-to-end workflows. Separate seed-42 and seed-123 smoke runs pass, 10.64 and 12.10 seconds respectively. Seeded runs repeat an existing test, not additional unique cases. |
| Derivative diagnostic | Current-version report regenerated; explicit `--check` still fails all 8 inconsistent probes. |
| Canonical preflight | All mandatory checks pass on the final rebuilt extension; manifold mirror maximum error `1.490e-8`. Goldens change only the controller stamp. |
| Browser validation | 27 frontend files / 116 tests, final production build and final rebuilt real-WASM snapshot/startup contract pass. GPU checks pass 16 cases with maximum error `5.411e-8`, 262,144 vertices, 531,502 triangles and 1,000 updates. |
| Structural checks and review | Architecture guardrail and canonical-surfaces gate pass. Provider Rust formatting and diff whitespace checks pass. Standards, Spec and Health Regression find no actionable source findings. |

The Python crossing failure is now `(0.35659381055433242,0.06404656757776669)`, Unresolved at level 23, requested scale `1.00000152722e-5`, resolved scale `1.49011611938e-8`, estimated error infinity. The unchanged crossing test took 370.03 seconds and remains enabled. No complete provider-6 Python-suite result is claimed. The old provider-5 full run was stopped at 32% to rebuild this repair.

At code commit `1e692d2`, [CI run 40044138353](https://github.com/rhythmatician/FractalSync/actions/runs/40044138353) passes Python/native and WASM builds and frontend checks. Separate type-check and generated-stub workflows pass. Rust CI reproduces the new native refusal. Backend CI and the artifacts job remain in progress at this checkpoint. These results do not establish merge acceptance.

The generic [query report](validation/geometry-provider-145.json) now contains current provider-6 measurements. Earlier sections are historical checkpoints; the full provider-5 report is preserved in [commit f86cfaf](https://github.com/rhythmatician/FractalSync/blob/f86cfaf/docs/validation/geometry-provider-145.json).

## October 23 follow-up: orbit reuse and cache retention

This performance change keeps `geometry-provider/5`, `orbit-controller/9`, all quality thresholds and both crossing assertions. It resumes each canonical critical orbit as the existing tile-certification budget doubles, rather than replaying its prefix. The construction states are discarded after each tile is generated. A deterministic work regression first counted 3,584 updates for budgets 512/1,024/2,048, then passed with exactly 2,048. A second regression compares 214 points at 12 increasing/decreasing budgets against the original evaluator, including escape cutoffs and nonfinite inputs.

The bounded tile LRU now retains 256 entries instead of the initial plan's 64. A [trace replay](validation/geometry-145-cache-working-set.json) measured 103 distinct tiles for one cut-locus query and 134 for the live training refusal. At 64 entries, every repeated query missed all 103/134 tiles again. Even 128 entries thrashes on the latter case. This is evidence that the initial capacity is practically inadequate for these queries. The change increases retention, not membership budgets, footprint limits, refinement limits or acceptance tolerances. The eviction regression now generates more tiles than the configured capacity so it still exercises real eviction.

All 44 matching jets in the pre-change report are identical after orbit reuse alone and after the cache increase, including floating-point values, scales, errors, classification and provenance. The [provider-5 report at this checkpoint](https://github.com/rhythmatician/FractalSync/blob/f86cfaf/docs/validation/geometry-provider-145.json) adds the live training and kinetic crossing refusals: 13 cases at four epsilon values, 52 rows total. It checks strict Regular quality bounds and identical immediate-repeat jets.

The [controlled cost report](validation/geometry-145-orbit-resume-cost.json) compares the baseline `30c9b9b`, orbit reuse with the old cache, and the final implementation. Each sample uses a fresh process followed by an immediate repeat, epsilon `1e-4`; the second repetition reverses variant order. Median times in milliseconds over two repetitions are:

| Case | Baseline cold / repeat | Orbit reuse, 64 entries | Orbit reuse, 256 entries |
|---|---:|---:|---:|
| Regular Shore approach | 352.54 / 2.28 | 285.83 / 2.16 | 302.44 / 2.19 |
| Crossing corridor | 584.01 / 2.96 | 542.84 / 2.78 | 519.07 / 3.55 |
| Cut locus | 1951.48 / 2450.49 | 1654.06 / 1804.07 | 1608.76 / 10.49 |
| Native crossing refusal | 415.83 / 420.72 | 386.13 / 394.84 | 401.25 / 3.66 |

These are small machine/load-dependent samples. Cold cost improves modestly and remains far outside a live frame budget. In the separate final 52-row report, the live training refusal takes 1,684 ms cold and 3.52 ms on repeat. No successful training step or crossing follows from returning the same refusal faster.

Separate fresh-worker Windows memory samples peaked at 4.8–5.8 MiB of working set across the three representative cases and variants. The final workers' peak commit was 1.20–2.49 MiB. These are whole-process peaks, not isolated cache payload or a worst-case bound; the cache can retain four times as many tiles. Orbit construction states remain outside the LRU. The report records source hashes, raw timings and memory measurements.

The rebuilt Python extension still refuses seed 42 at the identical live coordinate and error recorded below. Standards, Spec and Health Regression reviews found no actionable code findings. Crossing, live training validity, differential consistency and cold query cost remain merge blockers.

| Check | Orbit-reuse follow-up result |
|---|---|
| Membership regressions | All 5 visual-metrics tests pass, including exact cutoff equivalence and deterministic work count |
| Complete release Rust suite | 240 passed, 1 failed, 1 ignored with `--jobs 1 --no-fail-fast -- --test-threads=1`; the sole failure is the unchanged native crossing refusal, and all 33 snapshot integration tests pass (586.19 seconds). Focused results overlap this total. |
| Query report | 52 rows pass repeat determinism and strict Regular gates; all 44 pre-existing serialized jets exactly match both optimized variants |
| Derivative diagnostic | Explicit `--check` remains nonzero on all 8 inconsistent probes |
| Canonical preflight | All mandatory checks pass; manifold mirror maximum error `1.490e-8` |
| Frontend / production build | 27 files / 116 tests pass; build passes with existing dependency/chunk-size warnings |
| Rebuilt real WASM / GPU | Snapshot/startup contract passes; 16 projection cases, maximum error `5.411e-8`, 262,144 vertices, 531,502 triangles and 1,000 updates pass |
| Structural checks | Architecture guardrail, canonical-surfaces gate, changed Rust formatting and diff whitespace checks pass |

At code commit `7417c0a`, [CI run 40041861426](https://github.com/rhythmatician/FractalSync/actions/runs/40041861426) passes native/Python and WASM builds, artifacts and frontend/real-WASM/GPU checks. Separate type-check and generated-stub workflows pass. Rust CI reproduces the identical native crossing refusal (77 library passes, one failure) and stops before integration targets. Backend CI remains in progress at this checkpoint. The local Python run was stopped at 32% to rebuild a subsequent footprint repair; it had one training-smoke failure and no complete-suite verdict. Its partial passes are not a full validation claim.

## October 23 follow-up: reset validity and derivative reproducers

At this historical checkpoint, the reset follow-up kept production geometry at `geometry-provider/5` and `orbit-controller/9`. Geometry source, golden vectors, quality thresholds and both crossing test sources were unchanged from `cdcc2cd`. **Crossing, live training trajectories, differential consistency and synchronous query cost still prevent merging.** The performance follow-up above subsequently changes orbit evaluation and cache retention while preserving the reported jets.

### Training resets

Controls-v2 training now conditions its existing randomized reset proposals on Rust's Regular geometry gate. Each proposal is rounded to the float32 position actually passed into the complex64 rollout before validation. The existing Rust `manifold_induced_metric` checks that position with the rollout epsilon. A Regular first proposal keeps its original position; rejected proposals are resampled from the same generator, up to 16 attempts. Exhaustion raises with the last Rust refusal chained as its cause. This changes the reset distribution by explicit rejection; it does not repair the geometry of excluded positions.

Only resets are retried. The live `controls_v2_sequence` remains outside that loop, so a later invalid step still fails the batch. The new reset tests first failed on a proposal that rounds onto the exact cusp and on unbounded acceptance of invalid proposals, then passed after the gate was added.

| Check | Follow-up result |
|---|---|
| New reset regressions | 3 passed: float32 cusp rejection, unchanged Regular first proposal, 16-attempt exhaustion with preserved Rust cause |
| Reset plus complete Controls-v2 smoke files | 7 passed; sampled training smoke took 9.92 seconds in the final run |
| End-to-end training plus reset tests | 5 passed, including both existing training workflows; overlaps the three reset tests above |
| Fixed-seed training smoke | Seed 123 passes; seed 42 still fails during a live rollout, after reset validation |
| Inward energy acceptance | Passed against the restored production provider, 10.93 seconds |
| Static checks | Mypy passes for helper and trainer; architecture guardrail and canonical-surfaces gate pass; new helper/test Black formatting and diff whitespace checks pass |
| Independent review | Standards, Spec and Health Regression found no actionable issues in the final code diff |

Seed 42 fails at `(0.30752043982706229,0.02876420131721053)`, Unresolved, requested scale `4.50042492e-5`, resolved scale `3.81469727e-6`, estimated error `0.00049565977`. The failure remains visible. To reproduce without changing the test assertions, set `RUNTIME_CORE_NO_BUILD=1` after rebuilding the extension, then run the project interpreter with this payload:

```python
import numpy as np
import pytest
import torch

np.random.seed(42)
torch.manual_seed(42)
raise SystemExit(pytest.main([
    "backend/tests/test_controls_v2_smoke.py::test_controls_v2_training_batch_smoke",
    "-q", "--tb=short",
]))
```

Reset validation adds one Rust metric query per accepted first proposal, or up to 16 queries before exhaustion. CUDA retries also repeat the existing scalar sampling synchronizations. Representative device throughput has not been measured; this is bounded correctness work, not a query-cost improvement.

### Differential consistency

The [Rust diagnostic example](../runtime-core/examples/geometry_derivative_report.rs) now regenerates the [eight-probe report](validation/geometry-145-derivatives.json) directly from public provider queries. It compares centered differences at two step sizes with the reported gradient/Hessian and records neighboring validity and resolution. Its `1e-4` comparison tolerance is diagnostic only and changes no provider or energy threshold.

```text
rtk cargo run --release --jobs 1 -p runtime_core --example geometry_derivative_report
rtk cargo run --release --jobs 1 -p runtime_core --example geometry_derivative_report -- --check
```

The first command emits JSON. The second deliberately returns nonzero on the existing defect: `8 derivative probes disagree`. The compiled diagnostic completes in about 1.11 seconds on this machine. It is not added to CI as a passing acceptance gate, and no existing test is skipped.

Three local prototypes were tested and rejected, then production geometry was restored:

| Prototype | Near-Shore excursion at dt=0.001 / 0.0005 | Largest step change | Disposition |
|---|---:|---:|---|
| Analytic derivatives of the moving scalar fit | 0.43809 / 0.43977 | about 0.04938 | Local derivative regression passed, but total energy drift worsened |
| Globally anchored cubic field with analytic derivatives | 0.08198 / 0.08154 | about 0.08105 | Derivatives matched and cell-boundary checks passed, but refinement switches exceeded the unchanged limit |
| Cubic field with local second-moment coefficient correction | 0.09743 / 0.09637 | about 0.09297 | Quadratic reproduction and all provider regressions passed, but energy acceptance worsened |

The cubic example's peak switched from a level-8 jet with gradient approximately `(0.76550,-0.64314)` to a level-7 jet with gradient `(0.73326,-0.68005)`. Thus making each fixed-level field C2 is insufficient: adaptive level selection must also preserve spatially coherent differential geometry. The next geometry repair needs to address that transition together with within-level consistency, and pass the unchanged conservative-energy and crossing tests. No prototype above is shipped as a fix.

Standards and Spec review found no actionable issues. Health Regression found no diff-attributable maintainability, verification, or reasoning regressions within its bounded checks; training throughput remains unproved. No complete new Rust or Python suite is claimed for this training-only policy change. The complete provider-5 Rust baseline below remains 238 passed, one native crossing failure and one ignored test.

## October 23 follow-up: inward field convergence

The follow-up from `580fbc6`, implemented in `f82d1395818a539c3356568afeb7f18528e0cbb9`, uses `geometry-provider/5`, `orbit-controller/9`, and the unchanged `debug-snapshot/5` wire shape. Golden numeric values are unchanged; only their controller stamp changes. Earlier sections below are historical provider-4 evidence. The linked JSON reports have been regenerated for provider 5.

The inward false convergence came from comparing only fitted jets. At the recorded destination `(0.3948230501337527,0.0499674412699672)`, the level-6 fitted error was `0.00114922`, small enough to pass, while the coarse and fine contour distance fields disagreed by `0.00603986` at matching stencil positions. Least-squares projection concealed that disagreement. Membership signs and the fitted center value were checked separately and did not explain this jump.

Convergence now includes the maximum signed-distance disagreement across the same 49 fine-stencil positions. One Rust evaluator owns both fitting and this comparison, including shared-vertex signs. The fine samples are retained from fitting; the previous contour segments are moved into explicit previous-level state. This avoids recomputing fine samples or cloning the segment vector. The requested-scale law, error threshold, tie band, angle threshold, refinement budget, 64-entry cache, and all crossing test assertions remain unchanged.

The new public-query regression failed before this correction and passes afterward. Across the original inward pair, D now changes by `0.00010373754` over a move of `0.00011770365`; both queries resolve at level 8 with errors about `2.77e-5`. The previous `0.00729421503` scalar jump is gone.

The rebuilt Python extension passes the original 100-step inward acceptance test. The [16-row energy report](validation/geometry-145-energy.json) preserves all three short trajectories and adds these completed inward results:

| dt | Maximum excursion | Maximum individual energy change |
|---|---:|---:|
| 0.005 | 0.0029482902 | 0.0000433818 |
| 0.0025 | 0.0014284124 | 0.0000105753 |
| 0.00125 | 0.0006686324 | 0.0000025055 |
| 0.000625 | 0.0002887823 | 0.0000005568 |

The original maximum individual jump was `0.4135659`. All 16 sampled rollouts now complete below the unchanged `0.05` excursion and step-change limits. The near-Shore short trajectory still approaches an excursion of `0.0252252` and a step change of `0.0243545` as dt decreases; spatial derivative/metric consistency remains an open concern.

A subsequent [offline derivative probe](validation/geometry-145-derivatives.json) narrows that concern. It queries the center and `c +/- delta` on each coordinate axis through the public Rust binding, then takes centered differences of returned D and gradients. Both delta=`1e-6` and `1e-7` give nearly identical results, and every neighboring query remains Regular at the same level as its center. Before the near-Shore peak, the returned mixed Hessian is `8.27792`, while differences of the queried gradient give `d(grad_y)/dx=2.51670` and `d(grad_x)/dy=6.34802`. After the peak, the corresponding values are `1.74260`, `1.15179` and `4.21919`. Thus spatially moving the fit already creates differential inconsistency within a single accepted level; changing levels is not required. At the repaired inward points, the analogous mixed-derivative discrepancies are much smaller, about `0.0063` or less. These are diagnostic finite differences, never a substitute Physics implementation or a new acceptance threshold. A spatially coherent value/gradient/Hessian representation is the next numerical frontier.

### Follow-up verification and remaining blockers

| Check | Provider-5 result |
|---|---|
| Provider regressions | 20 passed, including the new inward consistency regression |
| Complete release Rust suite | 238 passed, 1 failed, 1 ignored with `--jobs 1 --no-fail-fast -- --test-threads=1`. The sole failure is native crossing; all 33 snapshot integration tests pass (598.87 seconds). The first concurrent-build attempt exhausted the Windows paging file and is not a test verdict. Provider tests overlap this total. |
| Three previously failing Python nodes | 2 passed, 1 failed in 381.96 seconds: inward energy and the sampled training smoke pass; kinetic crossing still fails |
| Canonical preflight | All mandatory checks passed; manifold mirror maximum error `1.490e-8` |
| Frontend / production build | 27 files / 116 tests passed; build passed; existing unavailable-ONNX-runtime branches remain |
| Rebuilt real WASM | Snapshot/startup contract passed |
| GPU | 16 projection cases, maximum error `5.411e-8`, 262,144 vertices, 531,502 triangles, 1,000 camera updates passed |
| Structural checks | Architecture guardrail, canonical-surfaces gate, changed Rust formatting and diff whitespace checks passed; Python stub generation has no API diff |
| Independent review | Standards, Spec and Health Regression reviewed the final code diff with no actionable findings |

The Python crossing test now refuses `(0.35714730438616327,0.06380885577340460)` as Unresolved, requested scale `1.49003765e-5`, resolved scale `9.53674316e-7`, error `0.00023872088`. Native crossing still encounters the recorded Singular/CutLocus endpoint `(0.21656074663216787,0.12610803467485909)`. Neither crossing test was changed.

The isolated training smoke pass does not establish training validity: replaying the prior failed coordinate `(-0.01567560926387349,0.68496020433710436)` still returns Unresolved with error `0.0059855766`. No full Python-suite rerun is claimed for provider 5; the complete provider-4 result below remains the broader baseline.

At code commit `f82d139`, [CI run 40033947928](https://github.com/rhythmatician/FractalSync/actions/runs/40033947928) passes native/Python and WASM builds and frontend tests including real-WASM/GPU checks. The separate type-check and generated-stub workflows pass. Rust CI reproduces the identical native crossing refusal (75 library passes, 1 failure) and stops before integration targets. Backend CI remains pending at this checkpoint; the targeted rebuilt-extension verdict above is the current local follow-up result.

The [44-row query report](validation/geometry-provider-145.json) uses eleven cases at four epsilon values, each in a fresh process followed by an immediate repeat. It verifies repeated jets and Regular scale/error bounds. At epsilon `1e-4`, the inward pair costs about `665â€“681 ms` cold and `3.4 ms` on immediate repeat; the corridor costs `643 ms` cold and `3.2 ms` repeated. Across the report, cold queries reach `2369 ms` and repeated singular queries reach `3342 ms`. These timings were collected alongside acceptance work and are machine/load dependent. They do not establish a frame budget; synchronous cost remains a merge blocker.

**PR #148 remains unready to merge:** crossing/refusal semantics, training geometry validity, spatial consistency and performance acceptance remain outstanding. The inward false-convergence and excessive scalar energy jump now have a regression and timestep-refinement evidence.

## October 23: projection classification and bounded refinement repairs

Work resumed from PR head `9af409f62794d63dc43675cf7dc3caac19721330`, preserving the separate, diverged local `issue-145` branch. Versions at that checkpoint were `geometry-provider/4`, `orbit-controller/8`, and the unchanged `debug-snapshot/5` wire shape. Regenerated golden numeric values were unchanged; their controller stamp was `/8`.

### Implemented corrections

- Off-Shore competition uses displacement directions from actual nearest projection points. Incident edges sharing a projection cannot manufacture two distance gradients. At zero distance, incident boundary normals remain meaningful evidence of nonsmoothness.
- Refinement persistence matches projection locations within one coarse cell diagonal, as well as directions. Both matched coarse and fine pairs must compete; two coarse features cannot both match one fine feature and establish persistence. All near-tied candidates are retained so symmetric cases are independent of segment order.
- Both consecutive levels must satisfy the requested scale before accepting Regular or supporting Singular. Comparing against an oversized previous level had admitted two neighboring Regular queries with a distance jump of `0.00843` across a move of only `0.0000188`, beyond their combined reported errors. A regression now rejects that inconsistency. A qualifying Singular candidate remains provisional while bounded refinement is available. The observed native endpoint `(0.24978495883420873, 0.09661690704669372)` was prematurely refused at level 5 but resolves with further refinement. This deliberately extends the issue's one-refinement stopping policy; it does not change the `0.5*h` tie band, 30-degree angle, scale law, or `0.25*requested_scale` error bound. Regular also requires no remaining competition.
- The old nine-tile expansion shifted by one eight-cell core stride, expanding 56 cells to only 72. The replacement uses contiguous complete tiles and one bounded expansion of at most 7x7 tiles, stopping earlier when containment succeeds. It skips tiles only when their spatial bounds prove they cannot influence the distance stencil. The deterministic LRU remains 64 entries.
- Membership nodes still changing at the iteration limit are retained as uncertainty. Each stencil sample rejects uncertain adjacent cells that could contain a closer contour, and uncertain nodes within four cells of its nearest projection. Farther nodes inside an enclosing ball do not invalidate an unrelated local jet. An unstable level retries finer spatial resolution and clears convergence history, so nonconsecutive levels cannot certify convergence.
- The origin now correctly has a unique off-Shore cusp projection. Its distance, gradient and Hessian are checked against the local distance-to-cusp values. Singular-refusal fixtures use `(-1,0)`; unresolved snapshot fixtures use an unattainable scale at `(0.25,0.5)`. Explicit startup positions and genuine invalid-state refusal remain tested.
- The exactly known primary cardioid cusp `(0.25,0)` is explicitly Singular/NonUniqueNormal, with exact scalar `D=0` and unavailable derivatives/error. Finite orbit iteration numerically rounded this point into an apparently smooth contour even at level 23; that approximation cannot certify smoothness at a known boundary singularity. No neighborhood is special-cased. A wider boundary uncertainty-band experiment was discarded.
- Runtime refusal messages retain the exact failed coordinate, provider/tile, requested/resolved scales and estimated error, including failures reached through Python training.

The original rejected endpoint `(0.29835945330580743,0.0515177858286498)`, refinement corridor `(0.35,0.05)`, and former high-curvature refusal `(0.4005984941807052,0.054142245287925383)` now return Regular jets satisfying the unchanged scale/error limits. At the corridor, the new jet resolves at level 10 with `D=0.0070220696361` and error `7.3430772e-6`, against a permitted error of about `1.7557e-4`.

### Verification

| Check | October 23 result |
|---|---|
| Provider unit regressions | 19 passed, including endpoint/corner, true medial axis, projection persistence, neighboring distance consistency, local membership uncertainty, known cusp and repaired coordinates |
| Complete release Rust suite | 236 passed, 2 failed, 1 ignored. One stale boundary-contract assertion was corrected and its complete target rerun passes 3/3; native crossing remains the unresolved failure. Provider tests overlap this total. No complete green run is claimed. |
| Rust snapshot integration | 33/33 passed, including partial singular/unresolved diagnostics and JSON nulls |
| Python final-artifact suite | 182 passed, 3 failed, no skips, in 2456.5 seconds. Failures are Controls-v2 training, inward near-Shore energy and the kinetic-energy crossing test. The duplicate focused run was stopped and is not counted. |
| Canonical preflight | All mandatory checks passed; manifold mirror maximum error `1.490e-8` |
| Frontend | 27 files / 116 tests passed; existing model-dependent unavailable-ONNX-runtime branches remain |
| Production build | Passed |
| Real WASM | Both snapshot entry points, nulls, regular automatic and explicit startup, and singular/unresolved diagnostics passed |
| GPU | 16 projection cases, maximum error `5.411e-8`, 262,144 vertices, 531,502 triangles and 1,000 camera updates passed |
| Structural/static checks | Architecture guardrail and canonical-surfaces gate passed; diagnostic script Pyright has zero errors and one compiled-extension source warning; Python stub generation has no API diff |
| Independent review | Standards and Health Regression had no actionable findings. The Spec review's many-to-one persistence finding was fixed and re-reviewed; no remaining actionable Spec finding. |

The boundary-contract assertion previously assumed that every Unresolved jet had an oversized cell. Unresolved also legitimately includes a small cell with excessive consecutive-level error; the corrected assertion requires at least one acceptance gate to fail. Regular thresholds are unchanged. Rust snapshot integration took about 560 seconds on this run, adding test-loop cost evidence. New/updated provider, contract, example and generator Rust formatting checks pass; the diagnostic script passes Black. Whole-file formatting checks on the three existing Python test files and `manifold.rs` expose formatting already present at the prior PR head, which was not broadly rewritten.

The Controls-v2 smoke fixture `(0.8,0)` is itself Singular under the destination provider. Its smooth-motion comparison now uses two Regular charts and asserts different coordinate displacements, rather than accepting equality. That corrected scoped test passes. The native crossing assertions, energy limit and randomized trainer failure remain enabled.

The complete Python run passes all 21 snapshot tests, the high-curvature conservative rollout, preflight, and end-to-end training/export coverage outside the failing Controls-v2 smoke. Its two 33x33 terrain tests take about 523 and 773 seconds, the kinetic crossing case 401 seconds, and signed-distance continuity 123 seconds. The new footprint/refinement work needs explicit performance acceptance; these suite results do not establish an acceptable feedback loop or live terrain rate.

The remaining Python failures are:

- Controls-v2 training encounters Unresolved geometry at `(-0.01567560926387349,0.68496020433710436)`, with requested scale `0.0011339886`, resolved scale `0.0001220703` and estimated error `0.0059622441`. Initial seed acceptance and later rollout validity must be checked separately; this failure must not be hidden by skipping a step.
- `test_energy_drift_bounded_near_shore` starts at `(0.4,0.05)` with velocity `(-0.03,0)`, takes 100 steps at dt=0.005, and has a maximum individual energy change of `0.4135658728610736`, above the unchanged 0.05 limit. This differs from the shorter passing trajectories below.
- The kinetic-energy crossing test reaches `(0.35643710307087195,0.06400291023711235)` and is refused as Unresolved at level 21. Resolved cell size is `5.96046e-8`, requested scale `1.0000758e-5`, but estimated error is infinite: fine cells alone did not provide a certifying pair of stable levels. Successful crossing under timestep refinement remains unproven.

At code commit `5daca003e6df1f583753d5b9c38382956ce6d1d1`, [CI](https://github.com/rhythmatician/FractalSync/actions/runs/40024300688) passes native/Python and WASM builds, artifact builds, frontend/real-WASM/GPU checks, stub generation and Python/WASM type checks. Rust CI has 74 library passes and reproduces the identical native crossing refusal, then stops before integration targets. Backend CI had not completed at the time of this record; the complete Windows results above are the available Python verdict.

### Energy convergence and remaining merge blockers

The reproducible [energy report](validation/geometry-145-energy.json) runs three 0.3-second conservative trajectories and the failing 0.5-second inward trajectory, each at four timesteps through the rebuilt Rust bindings. Each row records its duration. The [driver](../scripts/validate_geometry_145.py) records endpoint state, maximum excursion, largest individual energy change and its coordinates, and any exact provider refusal.

| Geometry | Excursion at dt=0.004 | dt=0.002 | dt=0.001 | dt=0.0005 |
|---|---:|---:|---:|---:|
| Low curvature | 0.0004483 | 0.0002155 | 0.0000991 | 0.0000412 |
| High curvature | 0.0004634 | 0.0002340 | 0.0001192 | 0.0000619 |
| Near Shore | 0.0202737 | 0.0226497 | 0.0242506 | 0.0252252 |

All twelve short trajectories complete below the unchanged 0.05 limit. Low- and high-curvature drift decrease with timestep. Before requiring both comparison levels within the requested scale, high-curvature excursion was about 0.259 and did not improve with smaller timesteps; the excessive neighboring distance jump described above caused a single energy change of about 0.260. The repaired high-curvature maximum step change at dt=0.002 is `1.73e-6`. Near-Shore drift still increases toward about 0.025 under timestep refinement, indicating a remaining spatial consistency concern despite satisfying this short acceptance window. No tolerance was widened.

At the near-Shore dt=0.0005 peak, the move from `(0.30007240651412576,0.05348910097156354)` to `(0.3000728655316197,0.05350153585945869)` changes D by `1.1464e-5`, within the move length `1.2443e-5` and combined reported error `1.1819e-4`. However, the accepted level changes from 8 to 7, gradient changes from approximately `(0.74504,-0.66469)` to `(0.73542,-0.67585)`, and the largest step energy change is `0.02435`. This points the remaining investigation toward derivative/metric consistency across accepted local fits; it is not a recurrence of the excessive scalar distance jump fixed above.

The inward acceptance trajectory is different and **does expose another excessive scalar jump**. Its 0.5-second refinement results are:

| dt | Maximum excursion | Maximum individual energy change |
|---|---:|---:|
| 0.005 | 0.4153874 | 0.4135659 |
| 0.0025 | 0.4156670 | 0.4145049 |
| 0.00125 | 0.4153968 | 0.4146072 |
| 0.000625 | 0.4156798 | 0.4150337 |

At dt=0.005, the peak goes from `(0.39494073887023606,0.04996931486326729)` to `(0.3948230501337527,0.0499674412699672)`. The accepted jet changes from level 8 (`D=0.04230377736`, error `2.71172e-5`) to level 6 (`D=0.04959799239`, error `0.00114922197`). The D jump `0.00729421503` exceeds the move length `0.00011770365` plus combined reported error `0.00117633915`. Both comparison levels satisfy the requested-scale gate here, so the earlier guard is necessary but insufficient: two coarse levels can agree while missing the finer nearest-contour result. This is a concrete remaining convergence/representation defect, not ordinary timestep error. The existing inward energy assertion and recorded trajectory stay enabled.

Independently fitted jets also need a spatial consistency check: the query-centered 7x7 fit supplies one internally coherent quadratic, but moving the query or changing its accepted level changes that quadratic. After fixing the excessive scalar jump, compare centered differences of queried D with `grad_D`, then differences of queried gradients with `hessian_D`, around the failing pair. Comparing fixed-level and adaptive probes would separate moving-fit inconsistency from a level-selection discontinuity. That mechanism predates these repairs; the failing trajectory alone does not attribute a regression to this diff.

The native rollout now proceeds beyond its original refusal but still fails in its **undriven** phase. Its last accepted state is `c=(0.21830798283479041,0.1245443967216562)`, `v=(-0.08464911002539771,0.07579486122988947)`. The rejected endpoint is `(0.21656074663216787,0.12610803467485909)`, reported as Singular/CutLocus at level 7, requested scale `0.0130064219662`, resolved scale `0.0009765625`, error `0.0002209228102`. The isolated query report reproduces it at every sampled epsilon. This is retained evidence of persistent sampled competition at the last contained level, not proof that the exact Mandelbrot geometry has a genuine cut locus there. The driven crossing phase has not established acceptance.

An exploratory Rust probe sampled finer tiles centered on the competing level-7 feet, rather than attempting complete coverage around the query. The nearest tracked feature approaches `(0.31036,0.03589)` and the other approaches the cardioid cusp `(0.25,0)`. Their distance separation stays positive while the tie band shrinks:

| Level | Tracked distance gap | Tie band | Local competition |
|---|---:|---:|---|
| 7 | 0.00035829 | 0.00048828 | yes |
| 8 | 0.00032886 | 0.00024414 | no |
| 9 | 0.00033293 | 0.00012207 | no |
| 10 | 0.00032547 | 0.00006104 | no |
| 11 | 0.00032174 | 0.00003052 | no |

These local probes lack complete contour containment and stencil-membership certification, so they cannot promote the runtime result to Regular or prove exact uniqueness. They do strengthen the finite-band-aliasing diagnosis. The current terminal Singular classification retains the settled two-level rule's **suspected** nonsmoothness meaning; an Unresolved result on coverage exhaustion would communicate the provisional status more clearly, but would also change the public classification of genuine cut-locus fixtures whose refinement exhausts that footprint. Resolving this needs a coverage/refinement contract that can distinguish those outcomes, rather than a coordinate exception or another arbitrary persistence count.

Training still needs a policy for selecting valid initial geometry. Randomizing inside the radius-two domain does not guarantee a Regular jet. The new errors preserve failed coordinates instead of hiding them. Do not silently discard failed rollout steps, substitute the raster, or count successful reset coverage as navigation.

Runtime cost is also unaccepted. The [query report](validation/geometry-provider-145.json) measures nine cases at four epsilons, each in a fresh child process followed by an immediate repeat. This removes the old report's cross-row warm-cache ambiguity. The October 23 sample, taken while other validation jobs were active, includes roughly 496 ms cold / 1.44 ms repeat at the corridor, 1.80 seconds cold / 1.88 seconds repeat at the true cut-locus fixture, and 372 ms cold / 376 ms repeat at the remaining crossing refusal (epsilon=1e-4). Repeated refinement can churn the 64-entry cache. These are not live-frame-budget results; bounded correctness does not establish playable throughput.

Merge still requires repairing the inward false-convergence/energy failure, resolving/classifying the remaining rollout refusals, validating training resets, and meeting live query-cost requirements. Near-Shore spatial derivative consistency also remains an acceptance concern. #120 owns that acceptance; #82's visual crossing verdict is still outstanding. #145 and PR #148 remain open and unaccepted.

The next classification investigation should extend the local projection-track evidence above into a complete containment/error argument. The current public jet gives a reproducible refusal but cannot distinguish exact nonuniqueness from a persistent finite-width tie before containment ends. Training reset work should validate proposed seeds through the Rust provider with a bounded failure policy; it must not suppress invalid states reached during a rollout. Cost work should measure generated tiles and iteration work for moving queries as well as repeated identical queries; improving an exact-repeat cache alone would not establish navigation throughput.

### Reproduction and tooling

From the repository root, use the project interpreter and rebuilt extension:

```text
rtk cargo test -p runtime_core --release --no-fail-fast -- --test-threads=1
rtk proxy .venv/Scripts/python.exe -m pytest backend/tests -q --tb=short
rtk proxy .venv/Scripts/python.exe scripts/preflight_parity.py
rtk proxy .venv/Scripts/python.exe scripts/validate_geometry_145.py
rtk cargo run --release -p runtime_core --example geometry_provider_report
rtk proxy node frontend/scripts/test_debug_snapshot_contract.mjs
rtk npm --prefix frontend test -- --run
rtk npm --prefix frontend run build
rtk npm --prefix frontend run test:hyperbolic-gpu
```

The golden generator is now anchored to `CARGO_MANIFEST_DIR`, so root-level invocation writes the repository's `shared/golden_vectors.json`. Before that repair, an invocation from the root also overwrote the sibling generated file `C:/Users/JeffHall/git/shared/golden_vectors.json`; its previous contents were not captured. The repository artifact was subsequently regenerated in the correct location. No handwritten source outside this checkout was changed.

## October 2 follow-up: startup and partial diagnostics

The approved lifecycle change is implemented. An untouched controller entering manifold mode starts at the verified regular point `(-0.5, 0.1)`. Legacy initialization stays at the origin. Explicit positions, existing nonzero velocity, and established trajectories are preserved. There is no new collision or swept-path barrier at `(0, 0)`; the integrator still evaluates sampled states and endpoints and fails closed when a required derivative is unavailable.

`debug-snapshot/5` returns partial snapshots for Singular, Unresolved, and outside-provider geometry. Position, velocity, classification, provenance, and available finite scalar values remain visible. Unavailable metric, derivative, force, kinetic/total-energy, and camera geometry values serialize as `null`. Diagnostics report invalidity and an error instead of failing before the provider can be inspected. Malformed state/configuration inputs are rejected. The controller preserves the core invalidity flag. `orbit-controller/7` records the startup change; regenerated golden vectors changed only this version string.

The cockpit displays unavailable values explicitly and retains its last valid scene when the new snapshot cannot supply rendering derivatives. Manual startup no longer manufactures an explicit origin seed. Explicit saved seeds remain explicit.

Verification of this follow-up:

- Controller integration tests: 21 passed, including explicit origin refusal, preserved legacy defaults, and preservation of existing velocity at the origin.
- Snapshot-focused Rust tests: 16 passed, including singular/nonzero-velocity, unresolved, regular equivalence, JSON nulls, and malformed input.
- Full Rust suite: 225 passed, 4 failed, 1 ignored. One failure was an obsolete test expecting an error instead of a partial outside-provider snapshot; its corrected scoped test passes. The remaining three failures are the native undriven/driven rollout test and two pre-existing manifold assertions in the debug integration file. The full snapshot integration target had 31 passed, 2 failed.
- Canonical preflight parity passed. Python stub generation had no API diff.
- Final compiled Python smoke: `(0,0)` is invalid/Singular with no metric; `(-0.5,0.1)` is valid/Regular with a metric; `(0.35,0.05)` is invalid/Unresolved with no metric.
- Frontend production build passed; 27 test files / 116 tests passed. Model-dependent ONNX checks retain their existing unavailable-runtime skips.
- Three focused Python debug tests passed against `target/python-snapshot5`. The full debug file was interrupted during slow controller replay tests; no complete result is claimed.
- Rebuilt WASM passed `node frontend/scripts/test_debug_snapshot_contract.mjs`. The test caught and fixed the default serializer emitting `undefined` for Rust `None`: both snapshot bindings now serialize explicit `null`, matching Python and recorded JSON. The test covers the standalone and controller APIs, regular automatic startup, explicit origin preservation, and unresolved diagnostics.

**Remaining #145 blocker:** the native rollout test fails in its undriven phase, near `c=(0.29842574478946454, 0.051456454020995405)`, `v=(-0.003236304445455842, 0.0029941732581717675)`, with a CutLocus refusal. This is not evidence that the driven trajectory crossed the Shore and failed there. A narrow refinement probe also found that `(0.35,0.05)` remains Unresolved at level 8; level 9 cannot contain the nearest contour in the current expanded tile footprint. Its level-8 error is about 0.00657 for requested scale 0.001028. These results require further provider/classification validation; the startup and diagnostics repair does not establish that the refused rollout is a genuine geometric singularity, or that changing the footprint alone fixes it. Crossing tests remain enabled. PR #148 and issue #145 remain unaccepted.

## October 2 CI follow-up

All five initial CI failures on `a1db98b` were compiler failures at the deprecated `std::f64::INFINITY` constant with warnings denied. Replacing it with `f64::INFINITY` in `b6dfc3d` restored green CI stub generation, Python packaging, WASM compilation, artifact builds, and Python/WASM type checks.

The next run exposed a GPU harness fixture assuming every snapshot had rendering derivatives. The harness now asserts a Regular snapshot, samples a small canonical Rust terrain patch, and expands that fixture to the same 512x512 GPU mesh. All original projection tolerances, static-buffer assertions, triangle-count checks, and 1,000 camera updates remain. This explicitly tests GPU rendering rather than dense adaptive-provider throughput. The real local WebGL run passed 16 projection cases with maximum error 5.411e-8, 262,144 vertices, 531,502 triangles, and no browser/GL errors. CI now also runs the real-WASM snapshot/startup contract script.

Two Rust integration fixtures also requested smooth energy/motion at provider-classified singular points. Their ordinary-geometry assertions now use an explicitly verified Regular point; the energy fixture retains the same distance from the origin and thus the same wall contribution. The actual native rollout regression and singular-refusal tests remain enabled. **The complete Rust debug-snapshot integration target now passes 33/33.** The Python non-terrain snapshot/controller subset passes 16/16; its five terrain tests were not part of that bounded run.

One known Rust rollout regression remains. At the recorded pre-step state the jet is Regular; the candidate endpoint `(0.29835945330580743, 0.0515177858286498)` is classified Singular/CutLocus. A narrower probe has not established whether the competing normals represent genuine geometry or premature classification. Reducing epsilon does not refine this case because rho dominates the requested scale. No tie threshold, normal-angle threshold, physics rule, or crossing assertion was relaxed. CI and #145 are not claimed fully green or complete.

## September 26 repairs and evidence

The provider now measures nearest-segment distance, handles the sign at shared contour vertices, and resolves ambiguous marching-squares connectivity using membership. It requires stable contributing tiles, checks containment of the sampled distance neighborhood, and refines one dyadic level at a time. Cross-level gradient and Hessian error use the coarser level's spacing. Regular jets must satisfy both the requested resolution and error bound. Cut-locus classification excludes incidental corners but retains persistent competing nearest features. Missing contours return unresolved geometry instead of invented jets.

Loading or clearing the legacy raster does not alter destination queries. Provider failures also preserve destination provenance. Exact main-cardioid and period-two-bulb membership shortcuts reduce repeated interior iteration; regression tests compare them against the original orbit evaluator near their boundaries.

Manifold regularization, snapshots, and terrain diagnostics now query with the active epsilon. The cockpit displays the provider's resolution, estimated error, validity, singularity, and bridge provenance. Versions are `geometry-provider/3`, `debug-snapshot/4`, and `orbit-controller/6`. Golden vectors were regenerated. Their numeric values are unchanged; the controller version changed. Python stub generation produced no API diff.

## Verification

These are local Windows results using release Rust builds and a rebuilt Python extension isolated under `target/python`. The Python commands use the project's `.venv/Scripts/python.exe`, `PYTHONPATH=target/python;backend`, and `RUNTIME_CORE_NO_BUILD=1`. They are not a claim that GitHub CI is green.

| Check | Result |
|---|---|
| Rust provider unit tests | 10 passed |
| Rust public adaptive geometry contract | 3 passed |
| Rust Controls tests | 15 passed |
| Rust library suite | 65 passed, 1 failed, native Shore crossing |
| Full Rust suite with `--no-fail-fast -- --test-threads=1`, before the final snapshot epsilon repair | 207 passed, 16 failed, 1 ignored across 24 target summaries |
| Rust debug snapshot suite after that repair | 16 passed, 14 failed |
| Python manifold physics | 25 passed, 3 failed |
| Python debug snapshot subset covering versioned sections, time, control provenance, and initial action | 4 failed at the cut-locus origin |
| Python full debug snapshot file | Interrupted after nine failures and slow terrain diagnostics; no complete result |
| `scripts/preflight_parity.py` | All mandatory checks passed; manifold maximum error 1.494e-08 |
| Frontend tests and production build | 27 files, 113 tests passed; build passed |
| Final rebuilt WASM snapshot and focused cockpit tests | Live regular snapshot had current versions and estimatedError; 21 tests passed |
| Python stub generation | Completed, no API diff |
| `git diff --check` | Passed |

The provider, Controls, and public-contract results overlap the Rust library/full-suite counts. Do not add them together. A complete all-green suite, GPU rendering acceptance, and eyes-on Shore verdict are still outstanding.

Regular-geometry algebra/parity tests now assert regular fixtures explicitly. The randomized parity trajectory starts at `(-0.5, 0.1)` with low drive energy to remain in that chart. It proves binding parity there; it does not prove general navigation or successful Shore crossing. The crossing tests remain enabled and failing.

Useful reproduction commands from the repository root:

```text
cargo test -p runtime_core --release --no-fail-fast -- --test-threads=1
cargo test -p runtime_core --release --test test_adaptive_geometry_contract
cargo test -p runtime_core --release --test test_debug_snapshot -- --test-threads=1
python -m pytest backend/tests/test_manifold_physics.py -q
python scripts/preflight_parity.py
npm --prefix frontend test -- --run
npm --prefix frontend run build
cargo run --release -p runtime_core --example geometry_provider_report
```

Use the configured project interpreter for the Python commands and rebuild its extension before comparing results.

## Measured query sample

[The JSON report](validation/geometry-provider-145.json) contains five points at four epsilon values each, with jet quality and first/immediate-repeat timings. The example asserts identical repeated jets and the strict resolution/error bounds for every Regular result. Its first queries can reuse cache entries from earlier rows, so `first_us` is not an isolated cold-cache measurement.

At epsilon 1e-4, `(-0.5, 0.1)` and `(0.3, 0.05)` are Regular, `(0.35, 0.05)` is Unresolved, and the origin and cusp `(0.25, 0)` are Singular. The cusp becomes Unresolved at epsilon 1e-8. Immediate-repeat timings in this release sample were about 0.45 to 1.49 ms; first touches reached about 82 ms. These machine-specific measurements do not establish the live frame budget or deep-scale acceptance.

## September 26 blockers (updated October 2)

1. **Native Shore crossing.** The Rust crossing test still fails. Python high-curvature, near-Shore, and underpowered/above-barrier trajectories encounter Unresolved or Singular geometry. Recorded failing positions include `(0.4005984941807052, 0.054142245287925383)`, `(0.300068357754536, 0.05333168318631767)`, and `(0.38777728061779215, 0.05037298251515404)`. Determine which represent real geometric ambiguity and which are contour/jet-estimation defects, then establish the ordinary Controls behavior there. Do not relax quality thresholds merely to obtain a trajectory.
2. **Initialization and diagnostic behavior at invalid geometry â€” resolved October 2.** The follow-up above implements regular untouched startup and partial invalid snapshots. Explicit origin states remain covered by refusal and diagnostic tests.
3. **Broader numerical and perceptual acceptance.** Finish #120 validation of conditioning, energy drift, cut loci, perturbation sensitivity, deep scales, cost, and cross-language behavior. Then obtain the #82 crossing/non-crossing visual verdict. The passing sample and bindings checks above do not close these items.

The [migration inventory](geometry-provider-migration.md) records destination ownership and deletion triggers for this geometry seam. It is a partial #84 artifact, not completion of that broader issue.

## October 2 completed CI and classifier diagnosis

At PR head `2014f942fa42c66a4ef4ec496051826a2fc113d6`, [CI run 37077513809](https://github.com/rhythmatician/FractalSync/actions/runs/37077513809) completed with the frontend/GPU, native and WASM builds, stubs, and type checks green. Rust still fails the native crossing rollout. Python reports 180 passed and five failed: the two Controls-v2 rollout/training smoke tests, high-curvature energy, near-Shore energy, and Shore-crossing acceptance. Terrain tests pass in that CI run.

The rejected native candidate endpoint `(0.29835945330580743, 0.0515177858286498)` now has a concrete classifier diagnosis:

- At level 5, the nearest projection is the shared vertex `(0.313232421875, 0.0390625)`; incident segment normals differ by about 50.53 degrees.
- At level 6, the nearest projection is the shared vertex `(0.3133544921875, 0.0390625)`; incident segment normals differ by 45 degrees. The current persistence rule therefore returns CutLocus.
- At level 7, the nearest foot is approximately `(0.313052173, 0.038784095)`, at distance `0.01944281063`. The competing segment's foot is its endpoint `(0.312438965, 0.038085938)`, at distance `0.01945885903`. The edge normals differ by about 85.91 degrees, but the signed distance directions from the actual feet differ by only about 2.7 degrees.

Two incident edges sharing one nearest foot do not establish two distinct nearest points. Away from that vertex, distance to the vertex has a unique displacement direction. Likewise, an endpoint-clamped segment's edge normal need not equal the distance gradient at the query. The current classifier therefore has insufficient evidence for its CutLocus result; this does not prove that the underlying exact Mandelbrot distance is regular.

Issue #145's settled plan explicitly calls for persistent sharp incident tangents at a nearest vertex to be classified as non-unique-normal geometry. That rule must be reconciled with the query's distance-gradient semantics before changing the implementation. The proposed correction is to compare directions derived from actual distinct nearest feet away from the Shore, while handling zero-distance boundary normals explicitly. It needs synthetic corner and genuine medial-axis regressions plus the existing rollout tests; no production classifier change was made in this diagnosis.

Separate Unresolved failures remain. The Python high-curvature trajectory reaches `(0.4005984942, 0.0541422453)` and is refused as Unresolved. Changing cut-locus classification alone is not sufficient evidence that CI or #145 will pass. Six exploratory full-throttle, outward-gradient trajectories from Regular points also encountered Singular or Unresolved geometry before crossing; they establish no successful crossing. No crossing assertion, quality threshold, or refusal behavior was weakened. Work stops at this identified contract conflict pending resolution, as requested.

The two Python Controls-v2 failures also expose startup assumptions. The explicit rollout smoke fixture starts at `(0, 0)` (Singular/CutLocus); its second start `(0.8, 0)` is Unresolved. The trainer smoke failure occurs in a randomized initial state that the provider classifies Singular/CutLocus; the traceback does not retain its exact coordinate. The trainer needs an explicit policy for selecting valid initial geometry. Neither a successful fast passage through the origin nor the cockpit's default-start fix makes a stationary origin a valid training seed.
