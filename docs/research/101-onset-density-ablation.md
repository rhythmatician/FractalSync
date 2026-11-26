# Causal onset density on five held-out music excerpts

A continuous density summary improved a small held-out predictor of future transient counts on these excerpts. The gain supports keeping the summary as an offline candidate. A discrete regime gave less improvement and spent most of its time in its highest state on three tracks. Keep [#101](https://github.com/rhythmatician/FractalSync/issues/101) open for independent target and production-style evidence and a comparison to the final [#109](https://github.com/rhythmatician/FractalSync/issues/109) tensor. The experiment changes no runtime channels.

## Candidate and timing

Read the canonical newest-frame onset, flux and RMS through the Rust CycleBank input adapter. Under features/2 the onset envelope copies spectral flux before the same fixed transform. The two named observations were equal on every analyzed tick. An instantaneous onset channel therefore duplicates this flux channel in the installed implementation. The time summary below can still supply useful history to a model given a shorter or compressed flux context. See the [Rust feature calculation](../../runtime-core/src/features.rs#L327) and [tick-to-CycleBank adapter](../../runtime-core/src/timebase.rs#L162).

At tick $k$ let $F_k$ be transformed flux and $R_k$ transformed RMS. Start a proposed event on a rising edge of the predicate $F_k-F_{k-1}\geq 0.15$ and $R_k\geq 0.02$. Space accepted events by at least four ticks. These are diagnostic transient events. A percussion hit, a musical note, and an event under this rule need not be the same. The thresholds are fixed across all evaluated files. An exploratory absolute flux-level gate saturated in most tracks and yielded almost no events. This is consistent with actual transformed flux often exceeding one. The one-hop rise formulation was chosen during this exploratory pass and needs prospective validation.

Count events in the trailing 94 ticks, about 2.005 s of tick bins. Divide by 24, the maximum count permitted by the spacing rule. This gives a continuous-valued scalar channel in $[0,1]$ with discrete sample increments of $1/24$. Count divided by 2.005 s gives diagnostic events per second. Between events the value persists until the oldest event leaves the window. For comparison only, use a hysteretic low/medium/high encoding with transitions upward at counts 4 and 8 and downward at counts 2 and 6. The event count remains recoverable from the same causal flux and RMS sequence.

Use the Rust sample clock rather than arrival time. At 48 kHz a 1024-sample hop is 21.333 ms and the 4096-sample FFT covers 85.333 ms. Wait for a complete FFT and then for 94 ticks before declaring the density usable. In the synthetic fixture the first valid density occurred at sample 99,328, about 2.069 s after start. A pulse introduced at 0.5 s first triggered at 0.512 s, and later silence let the rolling count return to valid zero. This 12 ms response is a measurement on that 48 kHz fixture. Audio hardware, buffering, causal resampling and threshold behavior affect production latency. The trailing window describes the last two seconds and typically takes that long to fully reflect a sustained density change.

On an epoch change, unexpected hop gap or nonfinite sample, clear accumulated events, previous flux, edge state and refractory state. Refill the support after the boundary. Distinguish a valid zero count in silence from pre-roll or invalid data. The generated tracks have one uninterrupted epoch each. Synthetic checks covered equivalent outputs under different ingest chunk lengths, equal outputs before a changed future, nonfinite input, explicit reset, and a native Rust stream gap that increments the epoch and restarts the sample clock. The predictor used no postroll or flushed resampler rows. See the [Rust sample clock and reset](../../runtime-core/src/timebase.rs#L464).

## Whole-song holdouts

~~~powershell
rtk proxy .\.venv\Scripts\python.exe -u formal/checks/check_101_onset_density.py
~~~

The [offline check](../../formal/checks/check_101_onset_density.py) reads the first 120 s of each of five ignored Tool MP3 files in backend/data/audio. The local 44.1 kHz encodings are decoded to mono with SoundFile, then resampled and analyzed in the installed Rust AnalysisTimebase. Audio stays local. The ignored report tmp/onset-101/results.json records source hashes, sample rates, settings, results and the features/2, analysis/1 and cycle-bank/2 versions. These are five songs from one band. The source filenames and timestamps support reproduction on the same local copies.

The candidate baseline consists of one current tick of Rust RMS, transformed flux, centroid, zero-crossing rate and rolloff, plus up to eight Rust CycleBank modes. The offline packing sorts modes by frequency and includes cosine/sine phase, log frequency, strength, confidence, missing-observation count and a slot mask. The history baseline adds trailing means and RMS of the already exposed flux and energy across 5, 24 and 94 ticks. It deliberately excludes any separately developed texture and harmonicity candidates, and the current #109 issue has no frozen tensor. The current spectral feature offsets are used under an explicit features/2 guard, while the CycleBank inputs come through the named Rust adapter.

Hold out all analyzed rows of one song, fit a fixed regularized linear model using the other four, and standardize inputs only on the training rows. Target the event rate in the following 47 ticks, about 1.003 s. The future target is evaluation-only. All 5,388 evaluated ticks per held-out song share the same warmup and future-window eligibility. Their overlap makes them unsuitable as independent replications. Treat each song as one fold and this run as exploratory.

| Held-out song | Detected events in 120 s | Density median | High regime fraction | Instant baseline | Instant + density | History baseline | History + density | History + regime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eulogy | 670 | 0.458 | 99.2% | 1.587 | 1.317 | 1.516 | 1.285 | 1.437 |
| Right In Two | 454 | 0.333 | 58.1% | 1.582 | 1.430 | 1.519 | 1.385 | 1.493 |
| Stinkfist | 422 | 0.292 | 49.9% | 1.401 | 1.275 | 1.452 | 1.308 | 1.363 |
| The Grudge | 577 | 0.417 | 87.5% | 1.633 | 1.428 | 1.572 | 1.379 | 1.504 |
| Third Eye | 573 | 0.417 | 84.5% | 1.698 | 1.557 | 1.670 | 1.544 | 1.611 |
| Mean RMSE across songs | | | | 1.580 | 1.401 | 1.546 | 1.380 | 1.482 |

Errors are RMSE in events/s. A density-only persistence forecast gave 1.241–1.628 events/s across the songs and beat the history-plus-density model on Eulogy. Keeping that simple comparison prevents the gain from being mistaken for a novel learned temporal representation. There were 3–52 regime transitions per track in the eligible region. The middle state was under 1% of Eulogy, which weakens the proposed three-bin interpretation.

Current density's single-feature Pearson correlation ranged from −0.28 to 0.46 with RMS, −0.22 to 0.41 with flux, and −0.76 to −0.05 with zero-crossing rate. A linear predictor using the full history baseline reconstructed current density with held-out correlations 0.15–0.52. These are properties of this corpus and model. They show that a coarse history and linear reconstruction leave information unused. Given the full flux/RMS sequence with the same initial condition, this density is a deterministic function.

## Disposition

Keep the continuous summary for offline ablations and set aside the tested regime encoding. Forecasting future counts from the same detector is a useful representation check with a shared source and no claim about Player behavior or human rhythm perception. A prospective test should freeze the detector settings, cover productions outside this band, compare against the eventual compact #109 baseline and its actual history capacity, and use an independent rhythm/activity target or a downstream Player ablation. Runtime channel, auxiliary-only and final drop decisions remain open. Any retained runtime version belongs in Rust under #109. The experiment adds no separate tempo tracker, section labels or visual rule.
