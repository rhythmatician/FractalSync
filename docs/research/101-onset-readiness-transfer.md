# Does causal onset density help forecast saved section joins?

Adding onset density to the causal history baseline changed mean held-out average precision by less than 0.0021 at every tested readiness horizon. The largest gain was at two seconds, where average precision rose from 0.11706 to 0.11899. At four seconds it fell. The stronger gain in the earlier same-detector forecast did not carry over to this target. Keep the continuous summary as an offline test candidate. This result gives no reason to add it to AudioActivation or select it as an auxiliary target. [Onset-density issue](https://github.com/rhythmatician/FractalSync/issues/101), [earlier same-detector test](101-onset-density-ablation.md), [AudioActivation contract](https://github.com/rhythmatician/FractalSync/issues/109).

## Test

Use five complete local Tool recordings and the five matching hard SongFormer exports. Each fold holds out one entire song. The target is whether a saved structural join occurs in the ensuing 0.5, 1, 2, or 4 seconds, with the same interval endpoints and coverage masks as the boundary audit. SongFormer supplies labels for scoring and fitting only. The exported joins are model estimates of structure, with unknown source offsets and no independent human annotation. [Target-clock and mask audit](102-boundary-target-evidence.md).

Use the same fixed ridge fit and causal flux/RMS history, current spectral features and native CycleBank modes as the first onset study. Compare that baseline with the addition of the fixed continuous two-second density, and separately with the two indicators for the hysteretic variant. All configurations use the same warmup and supported rows. Standardization and regression use training songs only. The fixed detector, history widths, and regression penalty were carried over from the previous study. That study used excerpts from these songs, so the leave-one-song-out folds avoid fit leakage within this run while leaving corpus-level generalization untested.

Score held-out ranks by average precision. A constant score has average precision equal to the positive fraction. Cyclic shifts of the held-out scores provide time-alignment contrasts. Also score the predictions on a common mask after moving the target window by ±12 Rust ticks (±256 ms). The shifted windows are sensitivity checks for alignment uncertainty. Neither consecutive ticks nor shifted copies count as independent trials. The run includes 125,736 eligible clock rows before horizon masks and 134 saved structural joins across the source analyses.

## Results

Each value averages the five song-level scores. A win means any positive change in held-out average precision for continuous density against history.

| Future window | Positive fraction | History | History + density | History + regime | Density wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 s | 0.02389 | 0.03601 | 0.03615 | 0.03439 | 3 / 5 |
| 1 s | 0.04786 | 0.05642 | 0.05725 | 0.05561 | 4 / 5 |
| 2 s | 0.09597 | 0.11706 | 0.11899 | 0.11718 | 4 / 5 |
| 4 s | 0.19108 | 0.21113 | 0.20909 | 0.20909 | 2 / 5 |

At two seconds, the density gain ranged from −0.00237 to +0.00470 by song. The mean gain remained positive under both small target shifts, at +0.00197 and +0.00134. At 0.5 seconds it was −0.00028 and +0.00004 after shifting. For the two-second density fit, the median cyclic-shift average precision on Third Eye was 0.11258 against 0.11142 in the aligned evaluation. On Stinkfist the same comparison was 0.08859 versus 0.15659, showing that the usefulness of alignment varies by recording.

The held-out task has a different target generator, but both are derived from the same audio. The results measure a simple model's ranking of saved upcoming joins. They give no listener evidence or estimate of whether the Player would move at a better time. #101 remains open for the final AudioActivation tensor, a broader corpus, and a matched representation or behavioral outcome.

~~~powershell
rtk proxy .\.venv\Scripts\python.exe -u formal/checks/check_onset_teacher_readiness.py
~~~

The [check](../../formal/checks/check_onset_teacher_readiness.py) writes ignored `tmp/onset-readiness/results.json` with audio and teacher hashes, Rust pipeline versions, masks, and per-song scores. The input audio and saved teacher exports stay local.
