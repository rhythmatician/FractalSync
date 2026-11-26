# Boundary and readiness targets from saved SongFormer segments

The saved segments support a deterministic **hard-boundary diagnostic** for [#102](https://github.com/rhythmatician/FractalSync/issues/102). They show strong target imbalance, sensitivity to horizon, and many structural joins without a change in the saved section label. This gives a control for later soft supervision. It provides no Student or Player result.

## Assignment on the Rust sample clock

The [check](../../formal/checks/check_102_boundary_targets.py) reuses the [existing segment validator](../../formal/checks/check_teacher_timelines.py#L46) from [#188](188-teacher-timeline-alignment.md). The earlier check validates native segments and tests a synthetic teacher-window lookup. This one counts explicit boundary-now and future-readiness assignments on the canonical grid. It uses the installed Rust bindings for 48 kHz and the 1024-sample hop. The nominal audio end comes from the local MP3's decoded native frame count, converted to canonical samples by flooring the rate ratio. The check does not instantiate a Student input pipeline or claim that each nominal grid point is an emitted Rust AnalysisTick. A training consumer would join against actual tick identities and original source offsets. See [Rust sample time semantics](../../runtime-core/src/timebase.rs#L64).

Preserve native teacher timestamps, and round each one to the nearest canonical sample for this comparison. The rounding adds at most half a canonical sample to the exported time. It says nothing about teacher localization accuracy. A boundary event is the shared timestamp of adjacent contiguous exported segments. Identical neighboring labels still count, since the structure export can mark a join without changing the label. Exclude the start of the first segment, the end of the final segment, and unmatched sides of gaps. The observed support is the intersection of the exported segment spans with the nominal audio span. A positive-width gap remains unknown even when rounding maps its endpoints to the same sample.

| Target | Supported positive rule |
| --- | --- |
| Boundary now | A saved internal join falls in $(t-\mathrm{hop},t]$. This attaches the hard join to the first canonical tick at or after it. |
| 0.5, 1, 2 and 4 s readiness | A saved internal join falls in $(t,t+H]$. A join at the current tick belongs to the boundary-now target, while a join exactly at the horizon endpoint counts for readiness. |

A fully supported window with no join yields a hard zero. If either end falls in unknown coverage or the window crosses a gap, mask it. The right endpoint must precede the end of the supported span. This masks final horizons rather than training on end-of-song padding or treating the last exported endpoint as a transition. Readiness is future-aware evaluation or training supervision. The rule produces no runtime value.

~~~powershell
rtk proxy .\.venv\Scripts\python.exe -u formal/checks/check_102_boundary_targets.py
~~~

The ignored tmp/boundaries-102/results.json records native endpoints, matched audio and export hashes, canonical support/joins, and separate positive, negative and masked counts. Hard exports carry no boundary confidence. The top-k export retains its original probabilities and residual mass, with omitted classes left unassigned. Synthetic assertions cover both interval edges, repeated-label and silence joins, unknown heads, internal gaps and audio end. The corpus check also verifies nested horizons wherever a longer horizon has full support and identical tick counts for hard and top-k exports of the same song.

## Real timeline evidence

Six exports represented five local tracks and 139 distinct segments, giving 134 internal joins. Stinkfist's hard and top-k files had identical times and target counts and contribute one track to pooled summaries. Across 126,916 nominal ticks per target, the decoded and exported tails differed by −11.01 to +32.24 ms. Keep those offsets and mask uncovered portions. The correspondence by song name and local audio hash establishes reproducibility on these copies. The export files alone lack the original teacher-side audio hash and exact preprocessing offsets.

| Target | Positive | Negative | Masked | Positive among supported | Always-negative accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boundary now | 134 | 126,779 | 3 | 0.106% | 99.894% |
| 0.5 s | 3,132 | 123,664 | 120 | 2.470% | 97.530% |
| 1 s | 6,279 | 120,399 | 238 | 4.957% | 95.043% |
| 2 s | 12,546 | 113,899 | 471 | 9.922% | 90.078% |
| 4 s | 24,847 | 101,129 | 940 | 19.724% | 80.276% |

The 4 s prevalence ranged from 15.80% on Stinkfist to 23.94% on The Grudge. The shortest internal join spacing was 12.0005 s, so no supported tested horizon included two joins. Longer horizons predominantly add more neighboring positives around the same 134 joins. A random split over ticks would leak those neighborhoods across folds. Count unique joins and songs when reporting coverage. Use precision–recall and event-level timing comparisons with an always-negative control, rather than treating high tick accuracy as anticipation.

Sixty-one of the 134 joins had the same adjacent hard label. Three touched a segment labeled silence. A function-change target would omit those repeated-label joins, and easy silence-related cases warrant separate reporting. In Stinkfist the saved top choice agreed with the hard export at every join. The minimum top function probability was only 0.335 and the omitted mass ranged from 0.0027 to 0.1517 across segments. Timing agreement with a top-k export gives no framewise boundary probability.

## Dense evidence still needed

The six project exports are postprocessed. One additional local candidate was found in the adjacent SongFormer checkout, outside that set. TheGrudge_logits.npz had 4,305 boundary samples at a reported nominal 8.333 Hz with a final NaN. Its 128×4,305 function array had eight finite rows before the last frame, negative infinity for the other rows, and a last frame of all NaNs. This is an artifact inspection, with no inferred class names, alignment offsets, calibrated probabilities or teacher checkpoint identity. The diagnostic leaves that file untouched. The local fork at b07ea9e contains an optional save_logits branch in src/SongFormer/infer/infer.py at line 327. The [public inference code](https://github.com/ASLP-lab/SongFormer/blob/main/src/SongFormer/infer/infer.py) and [postprocessing](https://github.com/ASLP-lab/SongFormer/blob/main/src/SongFormer/postprocessing/functional.py) show where dense arrays become segment output.

Keep this horizon family as a hard-reference control. Before training, collect versioned native dense exports across whole-song development and holdout groups, with timestamps, active class masks, valid frame masks, original source hashes, offsets and model/checkpoint identity. A future soft maximum within a fixed horizon could be compared against the hard control after alignment. It would be a maximum of teacher evidence, not an inferred event probability from independent frames. Keep #102 open for the dense artifact contract, matched no-teacher and auxiliary-head training runs, causal input/export checks and downstream Player and listener evidence.
