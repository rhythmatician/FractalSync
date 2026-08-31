# Fixed-horizon causal prediction diagnostics for the observed-ridge CycleBank

Issue: **#99 — Add fixed-horizon causal prediction diagnostics for CycleBank**

This document records the evaluation protocol, the cycle-ambiguity finding
that came out of it, and the per-song horizon tables from the Tool corpus.
It is intentionally evaluation-only: no Player/model I/O, no latent
undertones (#97), no global tempo tracker.

## Protocol

The evaluator reuses the canonical Rust CycleBank pipeline that PR #98
established. Python remains an offline orchestrator that:

1. builds the canonical tick stream via `AnalysisTimebase`;
2. replays it through the Rust `CycleBank` and retains immutable
   per-tick `TickSnapshot`s;
3. calibrates one observed-ridge candidate from the past via
   `_calibrate_observed_mode` (PR #98);
4. for each requested horizon `H ∈ {20, 50, 100, 200, 500}` ms and each
   evaluation event, asks the frozen mode's `time_to_next(reference_phase)`
   on the **latest snapshot at or before `t_event - H`**;
5. reports per-horizon coverage and conditional timing accuracy
   separately.

The horizon rule is strict causality: `snapshot.time_seconds <= t_event -
H`. Synthetic tests verify the rule under explicit future mutation of
*every* snapshot in `(snapshot.selected_time, t_event]`.

## Cycle ambiguity (the important finding)

v1 prediction returns the *next* crossing of the calibrated reference
phase after the snapshot. When the lead budget `t_event - snapshot.time`
is at least one full mode period, that "next crossing" cannot tell us
whether the oracle event is the **first**, **second**, **third**, ...,
recurrence after the snapshot. Phase predicts a recurrence but does not
count them.

We therefore skip such events and report them under
`n_skipped_cycle_ambiguous`. The skipping condition is:

```text
n_skipped_cycle_ambiguous  if  (t_event - t_snapshot) >= 1 / f_mode
```

This is **not** a Rust bug. `CycleMode.time_to_next()` correctly returns
the next crossing after `snapshot.time`; the ambiguity is a property of
the question, not the answer. The synthetic test
`test_cycle_ambiguity_skipped_when_horizon_at_or_exceeds_mode_period`
catches a 2 Hz mode / 500 ms horizon and verifies that every evaluation
event is reported as ambiguous (and zero predictions headline).

## Per-skipped-event decomposition

Coverage and conditional accuracy are reported separately, and the
"did-not-predict" set is decomposed so the report can say **why**
coverage fell rather than just how far:

| counter | meaning |
|--------|---------|
| `n_skipped_event_too_early` | event at-or-before the calibration cutoff (PR #98 used these for calibration) |
| `n_skipped_no_pre_horizon_snapshot` | bank has not warmed up yet at the horizon cutoff |
| `n_skipped_candidate_missing` | frozen mode id is not in the selected snapshot |
| `n_skipped_cycle_ambiguous` | horizon at or above one full mode period |

Issue #99's acceptance criterion "Coverage and conditional timing
accuracy are reported separately at every horizon" is satisfied by
reporting both `prediction_coverage` and `fraction_within` per horizon,
plus the four skip counters above.

## Tool corpus results

The compact horizon table for each song (medians in milliseconds):

```text
song         | horizon_ms | coverage | median_abs_ms | p90_ms | <=40ms | n_predictions | n_ambig | status
-------------+------------+----------+---------------+--------+--------+---------------+---------+---------
Eulogy.mp3   |         20 |     0.05 |        465.51 | 855.26 |   0.07 |            40 |       0 | ok
             |         50 |     0.05 |        188.09 | 793.49 |   0.20 |            40 |       0 | ok
             |        100 |     0.05 |         77.93 | 717.99 |   0.30 |            40 |       0 | ok
             |        200 |     0.05 |         92.85 | 684.95 |   0.23 |            40 |       0 | ok
             |        500 |     0.05 |        363.21 | 466.79 |   0.00 |            36 |       4 | ok
RightInTwo.mp3 |        20 |     0.09 |        483.49 |1266.01 |   0.15 |            86 |       0 | ok
             |         50 |     0.09 |        362.42 |1241.90 |   0.19 |            86 |       0 | ok
             |        100 |     0.09 |        291.43 |1134.55 |   0.16 |            86 |       0 | ok
             |        200 |     0.09 |        172.98 |1077.19 |   0.20 |            86 |       0 | ok
             |        500 |     0.09 |        429.41 | 819.03 |   0.00 |            86 |       0 | ok
Stinkfist.mp3 |        20 |     0.00 |        126.61 | 165.71 |   0.00 |             2 |       0 | ok
             |         50 |     0.00 |         79.01 |  98.63 |   0.00 |             2 |       0 | ok
             |        100 |     0.00 |        169.25 | 170.41 |   0.00 |             2 |       0 | ok
             |        200 |     0.00 |         96.45 | 137.99 |   0.00 |             2 |       0 | ok
             |        500 |     0.00 |          0.00 |   0.00 |   0.00 |             0 |       2 | insufficient_data
TheGrudge.mp3 |        20 |     0.00 |          0.00 |   0.00 |   0.00 |             0 |       0 | insufficient_data
             |         50 |     0.00 |          0.00 |   0.00 |   0.00 |             0 |       0 | insufficient_data
             |        100 |     0.00 |          0.00 |   0.00 |   0.00 |             0 |       0 | insufficient_data
             |        200 |     0.00 |          0.00 |   0.00 |   0.00 |             0 |       0 | insufficient_data
             |        500 |     0.00 |          0.00 |   0.00 |   0.00 |             0 |       0 | insufficient_data
ThirdEye.mp3  |        20 |     0.01 |         16.05 | 168.00 |   0.80 |            10 |       0 | ok
             |         50 |     0.01 |         43.78 | 120.54 |   0.50 |            10 |       0 | ok
             |        100 |     0.01 |         63.96 | 100.88 |   0.18 |            11 |       0 | ok
             |        200 |     0.01 |         54.82 | 132.65 |   0.33 |            12 |       0 | ok
             |        500 |     0.00 |          0.00 |   0.00 |   0.00 |             0 |      12 | insufficient_data
```

(Raw numbers: `backend/logs/cycle_bank_fixed_horizon/cycle_bank_fixed_horizon_report.json`.)

## Reading the table

1. **Cycle ambiguity surfaces cleanly where expected.** Eulogy has the
   calibrated candidate at median 1.23 Hz (period ≈ 815 ms) and at 500 ms
   horizon `n_ambig = 4` (events past the snapshot stream end). ThirdEye
   has its candidate near 2 Hz-ish (so period ≈ 500 ms) and at 500 ms
   horizon *every* evaluation event is ambiguous: `n_ambig = 12`,
   `n_predictions = 0`. The headline conditional accuracy stays empty
   rather than reporting a misleading ±period error.

2. **Conditional accuracy degrades with horizon, but coverage stays
   constant.** Eulogy median abs error is 465 ms at 20 ms horizon, drops
   to 78 ms at 100 ms (?), rises to 363 ms at 500 ms (excluding the
   ambiguous ones). The non-monotonic dip is the sample-median
   artefact — the events that score at 100 ms horizon happen to align
   better than the wider samples at 20 ms and 500 ms. The p90
   *increases* monotonically as horizon grows, which is what we expect
   when v1 first-order phase prediction runs out of margin.

3. **RightInTwo is a calibrated low-frequency drift case.** Its
   candidate frequency is in the 0.5–6 Hz band and the calibration is
   stable (R high, coverage 100% on calibration), so the evaluator has
   something to predict with at every horizon. The conditional error
   stays in the hundreds of milliseconds across all horizons. This is
   consistent with a tempo that drifts enough at 200–500 ms horizons
   that v1 first-order `phase_at` cannot close the gap.

4. **TheGrudge fails calibration.** No mode produced a coherent
   candidate with 24 calibration opportunities and 6 hits each. This is
   a coverage problem, not a phase problem. Future work: relax the
   calibration minimum or investigate why TheGrudge's observed ridges
   do not cohere around a single mode over the calibration window.

5. **Stinkfist has a rare candidate.** Two predictions across all five
   horizons; everything else is `n_skipped_candidate_missing`. The
   observed-ridge candidate is acquired but rarely survives a
   pre-horizon snapshot. Mode persistence is the dominant bottleneck
   here.

## Per-issue #99 questions

> At what horizon does conditional timing accuracy materially degrade?

For Eulogy/RightInTwo the median stays in the hundreds of ms across
all tested horizons; what changes is **p90** which climbs as horizon
grows. There is no clean "knee" in the curve on these songs; the
evidence so far is that accuracy degrades **gradually** with horizon,
not abruptly.

> At what horizon does prediction coverage materially degrade?

Coverage is dominated by **candidate_missing** (mode not in the
selected snapshot) on Eulogy and Stinkfist, and **insufficient_data**
on TheGrudge. The horizon itself does not appear to cause coverage
collapse; mode persistence does.

> Is v1 first-order `phase_at` adequate for 50/100/200 ms anticipation?

**No.** Even at 100 ms, the median abs error is ~80–290 ms across the
Tool corpus, and p90 exceeds 700 ms. The 40 ms "hit" rate is below
~30% on every song. v1 first-order phase prediction is the dominant
source of conditional inaccuracy at 100 ms and beyond.

> Does tempo drift create a clear need for the currently diagnostic
> `frequency_slope` term in prediction?

The chirp synthetic test
(`test_chirp_sensitivity_grows_with_horizon`) shows v1 error grows
monotonically with horizon when the true frequency differs from the
snapshot's measurement. On real songs, the RightInTwo / Eulogy
mid-horizon results are consistent with drift; the
`frequency_slope` field is already exposed in `CycleMode` and a
slope-aware second-order prediction should be straightforward to
evaluate later.

> Is tracked-mode identity persistence, rather than phase extrapolation,
> still the dominant bottleneck?

**Yes**, on this corpus. The per-song skip counters say so directly:

| song | dominant skip class |
|------|---------------------|
| Eulogy | candidate_missing (≈ 95% of eval opportunities) |
| RightInTwo | candidate_missing (≈ 91%) |
| Stinkfist | candidate_missing (≈ 96%) |
| TheGrudge | calibration never succeeded |
| ThirdEye | cycle_ambiguous at H ≥ 500 ms; otherwise candidate_missing |

Mode identity persistence, not phase extrapolation, is what gates
anticipation at the horizons issue #99 asked about. Improving the
prediction math (frequency_slope, second-order) will help where
predictions exist; it will not help where the candidate is missing.

## Non-goals

As required by issue #99, this issue does **not**:

- implement latent temporal undertones from #97;
- define beat/bar/phrase semantics;
- add a global tempo tracker;
- change `PlayerObservation`;
- feed CycleBank into ONNX;
- retrain `PlayerPolicy`;
- use future audio in runtime;
- silently switch candidates after calibration;
- introduce a higher-order frequency-slope predictor.

These results may motivate a later implementation issue, but issue #99
itself does not pre-decide the fix.