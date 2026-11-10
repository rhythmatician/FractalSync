# Geometry #145: retain coefficient membership witnesses during preparation

October 30, 2026. Coefficient-witness seeding fixes a reproduced contour omission
and reduces frozen source-comparison failures from 1,099 to 720. On the original
neighboring grid, failures fall from 381 to 27. This is a bounded preparation
fix, not a distance certificate. PR #148 remains unready: the unchanged Python
physics file still has 25 passes and two failures.

## Reproduced defect and fix

Preparation previously refined a reference cell only when its coarse corners
disagreed. Finer coefficient locations were classified later, for signs alone.
Those required samples could observe a component that never entered the contour.
Its coefficient sign would describe one geometry while its magnitude used
distance to another, incomplete contour.

The independent regression uses two circles. One small circle fits between all
four coarse reference corners but is resolved by coefficient samples. Before
the fix, its expected distance of approximately `-0.02` becomes
`-0.4803946572`. The test was run red before implementation. With witness seeding,
the error is below `0.003`; no Mandelbrot membership proof is involved.

When `coefficient_witness_seeding=true`, preparation collects the already-required
coefficient membership values first, uses disagreement with coarse corners to
trigger cell refinement, and reuses those values for coefficient signs. Shared
edge witnesses can seed both incident cells. It adds no membership probes for
the seeding step itself. Extra refined cells still consume the existing query
and segment budgets. A plane control requires identical bytes and identical
membership-query counts with seeding on/off. Both modes refuse budget exhaustion.

The option defaults off so prior frozen input recipes retain their construction
semantics. The new recipes explicitly enable it. This does not change evaluation,
basis functions, derivative propagation, Physics or runtime query work. Features
missed by both coefficient and reference samples can still be absent. Refinement
of a seeded cell also remains finite; seeding is not a completeness certificate.

## Frozen assets and resource limits

The [refined recipe](geometry-145-operational-seeded-fine-experiment.json) was
selected for replay before acceptance validation. The [coarse recipe](geometry-145-operational-seeded-coarse-experiment.json)
exists for the same fixed source-sensitivity audit. Both preserve final contour
spacing `0.000125`, coefficient spacing `0.0005`, the 3,425 by 2,825 coefficient
grid, 8,192 membership iterations, eight bisections, 50-million membership-query
limit and 500,000-segment limit. No domain or coefficient replacement occurs
during a rollout.

| Preparation measurement | Coarse source | Refined source, selected |
| --- | --- | --- |
| Field identity | `d8da41ce56bdbcc4` | `ccc45e15857947e9` |
| Source coarse spacing | 0.002 | 0.001 |
| Additional homogeneous cells seeded | 1,666 | 1,996 |
| Total refined cells | 12,953 | 28,795 |
| Membership queries | 23,304,828 | 34,975,731 |
| Segments | 341,707 | 332,200 |
| Process preparation time | 117.78 seconds | 127.19 seconds |
| Sampled OS peak working set | 216,879,104 bytes | 216,866,816 bytes |

Both assets have 77,405,056 artifact bytes and 77,405,000 coefficient bytes.
Selected artifact SHA-256 is
`b28c4538de5e134448372d608be11509c1ef0ff6ef45268e5009c5c548ca58be`.
The [replay manifest](geometry-145-operational-seeded-replay.json) binds it to
the frozen reference policy. Timed Physics profile remains `54b525d2266d27eb`.

## Quality and mechanics remain separate

The unchanged 1,768-point source audit uses the same distance-unit scalar,
gradient and Hessian sensitivity formula described in the
[preceding report](geometry-145-operational-source-refinement.md). Its declared
coarse source spacing is not coefficient-resolution convergence or a resolved
scale. No source comparison is promoted to a reference-distance upper bound.

| Outcome | Previous unseeded pair | Seeded pair |
| --- | --- | --- |
| All comparison failures | 1,099 | 720 |
| Neighbor-grid failures | 381 / 1,000 | 27 / 1,000 |
| Near-Shore scalar-audit failures | 717 / 765 | 693 / 765 |
| Crossing-start comparison | Fails | Scalar, gradient and Hessian agree exactly |
| Quality-probe and ridge-start comparisons | Pass | Pass |

Among the remaining failures, 718 are dominated by scaled Hessian differences
and two by scaled gradient differences. Component counts overlap: scalar alone
exceeds budget at 255 points, gradient at 574, Hessian at 718. Maximum sensitivity
decreases from `0.0340304` to `0.00742741`. The fixed reference-realm audit retains
765 conclusive realms, zero sign mismatches and zero sign-based counterexamples.
These results do not establish scalar or reference derivative fidelity.

The selected asset passes all 35 frozen per-run expectations with complete
coverage, 21 operational crossings and 21 independently verified reference
crossing trajectories. Fourteen noncrossing controls remain separate. The
original native crossing body passes unchanged with a verified reference crossing.
Known reference-cut refusal and complete-state preservation still pass in WASM.
Reference pairs are for the piecewise-linear committed trajectory, not an exact
ODE event or a regional singularity certificate.

Neighboring self-consistency and patch joins pass at the unchanged `1e-4`
tolerance; normalized Hessian errors are `3.770e-6` and `4.765e-6`. The raw central
diagnostic still fails at `1.853e-4` and remains preserved. High-curvature and
near-Shore energy excursions are `2.309e-14` and `2.468e-11`, below the original
`0.05` limits. Native/WASM parity passes at the existing declared tolerances.

The original Python suite remains **25 passed, two failed**. Above-barrier
sampled crossing times are `3.864`, `3.864`, `3.8635` seconds. They fail the
unchanged strict refinement comparison because the coarse gap is zero and the
fine gap is `0.0005`. Unknown scalar-quality metadata remains the other failure;
the adapter does not invent a small error estimate from the passing quality probe.

The source fix moves the original crossing from about `2.044` to `3.864` seconds.
This material sensitivity is retained despite improvement at the initial state.
A separate five-timestep event diagnostic converges toward operational segment
zero time `3.86321002` seconds; whole-crossing maximum energy excursion decreases
from `2.412e-5` to `6.893e-10`. This does not replace the original failed assertion
or certify a precise Shore event.

## Cost and verification

Device: Intel Core i9-14900, Windows 10.0.26100, headless Chromium 153.0.8010.12.
Native cache-stressed query p99 is `3.0 / 2.6 / 4.0` microseconds in three fresh
processes; complete-step p99 is `63.2 / 55.2 / 177.0` microseconds. Browser p99 is
30 microseconds per query and 110 microseconds per complete step. All meet the
provisional experimental targets of 100 microseconds and 2 ms, respectively.

Browser first prepared query is 170 microseconds; cache-stressed query maximum
is 1.030 ms. First/maximum complete step is 1.010 ms. Native first prepared queries
are `4.9 / 4.0 / 4.9` microseconds. Unavailable-query browser p99 is 5 microseconds,
maximum 325 microseconds. Full first/p50/p99/max distributions remain separate.
Passing p99 is not a hard deadline guarantee.

Prepared steps still use 160 geometry evaluations and at most 5,760 coefficient
reads. Native loading takes `140.66 / 126.11 / 128.89` ms, with sampled process
peaks `248,090,624 / 242,737,152 / 250,728,448` bytes. Browser fetch takes
147.88 ms, decoding 153.70 ms, WASM initialization 27.68 ms, all outside query
timings. WASM capacity is 160,301,056 bytes, separate from the 77,405,056-byte
JavaScript input, 64 MiB stress scratch and other browser overhead.

All **29 experimental Rust controls pass**. Default native compilation,
formatting and diff checks pass. The Python wheel and WASM bundle reuse the
unchanged evaluator and mechanics from the previous checkpoint; their actual
binary hashes and the new builder/source hashes are recorded separately.
The entire Python test file and original native crossing body retain their
previous SHA-256 values, checked by the validation wrappers. No shipping math,
goldens, controller versions, test bodies or thresholds change.

The [machine report](geometry-145-operational-seeded.json) indexes recipe,
source, binary and artifact hashes. The [raw archive](experiments/geometry-145-operational/seeded-evidence.zip)
preserves both preparations, all source comparisons, original failures,
three native reports, browser parity, event diagnostics and the synthetic red
regression. Existing preparation and measurement commands accept the new recipes
and replay manifest. Earlier assets and failure reports remain intact.

This fixes a demonstrated omission without starting another proof engine.
Source stability near the Shore, supported scalar-quality evidence, the retained
crossing-time assertion and ADR 0005 adoption decisions still prevent merging.
