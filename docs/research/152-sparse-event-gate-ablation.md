# Sparse impulse gate ablation

On 192 synthetic evidence traces, a calibrated Schmitt gate with a short cooldown matched 191 of 216 target events with zero unmatched fires. The tested LIF and ALIF variants both trailed it. This is a result about the commitment mechanisms given the same synthetic evidence. The musical and visual comparison in [#152](https://github.com/rhythmatician/FractalSync/issues/152) still needs a matched causal Player trunk.

## Event and evidence

The candidate event is the existing bounded generalized impulse in Controls v2. Sustained confidence around one prospective accent can otherwise produce several taps on consecutive Player ticks. A gate emits zero or passes through the same ordinary analog impulse magnitude with the existing direction. Native impulse clamping, metric conversion, and Physics are outside this offline probe. The other Controls have no gate in the fixture. [Controls v2 implementation](../../runtime-core/src/controls.rs).

The local SongFormer outputs contain boundaries and a probability artifact. These can serve as training or scoring labels when the split and availability rules permit. The local `orbit_control` ONNX sidecar describes a ten-frame input window with shape `[1, 60]` and nine older outputs, including `s_target`, `alpha` and six band gates. This audit found no saved matched causal Player hidden-state and impulse-evidence traces with a frozen Controls-v2 payload and held-out event outcomes. The offline check therefore substitutes deterministic scalar evidence at the gate input and consults its synthetic target ticks only for scoring and offline parameter selection. [Causal Player trunk work](https://github.com/rhythmatician/FractalSync/issues/104), [teacher boundary](https://github.com/rhythmatician/FractalSync/issues/102).

## Tick and state semantics

The fixture uses the canonical $1024/48000$ s hop. Write $I_t\in[0,1]$ for the common evidence scalar. Plain fires whenever $I_t\ge\theta$. Schmitt starts armed, fires only when armed with zero cooldown, and disarms on fire. Evidence strictly below $\theta$ times the release ratio re-arms it, while cooldown blocks the next $C$ ticks after a fire. LIF updates $V\leftarrow\lambda V+I_t$, threshold-tests $V$, and resets $V$ to zero on fire. ALIF tests against $\theta+\beta A_t$ and updates $A_{t+1}=0.92A_t+s_t$ after deciding whether to fire. The adaptation coefficient is fixed in this probe. Adaptive thresholds have been studied in spiking networks, while here the evaluated recurrence sits after a supplied scalar evidence value. [Canonical timebase](../equations.md), [Bellec et al.](https://arxiv.org/abs/1803.09574).

Stream/fixture start sets voltage and adaptation to zero, arms Schmitt, and clears its cooldown. Plain needs no persistent state. Schmitt needs a flag and an integer counter, LIF one scalar, and ALIF two scalars. A chunk boundary carries forward the same state. Deterministic checks verify threshold equality, re-arming, cooldown countdown, voltage accumulation and reset, leak after silence, adaptation decay and override by stronger evidence, chunk-equivalence, and unchanged prefix fires when future evidence changes. They also verify that fired impulses retain the provided analog payload. Checkpoint serialization, browser startup, hot reload, and the choice of reset boundaries in a live Player would need a production contract and tests.

## Reproduce

```powershell
rtk proxy .venv/Scripts/python.exe formal/checks/check_sparse_event_gates.py
```

The probe generates 96 labeled calibration traces and 192 evaluation traces with seeds `15201` and `15202`. Each split uses the same eight scenario families with varied background, onset positions, and cue fluctuations. The evaluation is held out by seed within that generator. Families include isolated impulses, weak subthreshold pulses, slow ramps, noisy stable evidence, sustained strong evidence, repeat gaps of 2 to 18 ticks, stale evidence followed by a new cue, and alternating weak and strong cues. Any learned gate or head still needs song-level held-out evaluation. [Offline experiment](../../formal/checks/check_sparse_event_gates.py).

The experiment compares fixed hand parameters and a small gate-specific grid. Calibration minimizes missed targets plus unmatched fires, then duplicates, then total absolute matched lag. It reports the best active configuration separately if the grid selects silence. Parameter count differs because each recurrence has different controls. The scoring window spans two ticks early to six ticks late, with ordered one-to-one matching. A duplicate is an unmatched fire within eight ticks after a matched fire. The script separately reports bursts, minimum time to retrigger, rates, average lag and qualifying premature/late unmatched events. Those timing choices describe the probe, not a musical timing tolerance.

## Results

| Gate, calibrated | Matched / 216 | Unmatched | Duplicate | Premature / late | Events/min | Mean absolute matched lag | Minimum retrigger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain | 131 | 437 | 154 | 0 / 283 | 74.29 | 0.40 ticks | 1 tick |
| Schmitt + cooldown | 191 | 0 | 0 | 0 / 0 | 24.98 | 0.20 ticks | 4 ticks |
| LIF | 0 | 0 | 0 | 0 / 0 | 0 | — | — |
| LIF, best active | 55 | 172 | 58 | 0 / 114 | 29.69 | 2.42 ticks | 2 ticks |
| ALIF | 142 | 63 | 24 | 0 / 24 | 26.81 | 0.18 ticks | 4 ticks |

Schmitt matched 47 of 48 targets in the isolated-repeat family and 48 of 48 in the alternating family. Calibrated ALIF matched 24 of 48 in each. It fired at least one additional time on sustained evidence in every sustained fixture. Plain's excess fires came from sustained evidence. Calibrated Schmitt missed every weak-accumulation target and the second impulse in the two-tick repeat case. Its evaluation bursts had sizes one or two. The report records the burst distributions for all gates and the target counts by repeat gap.

Hand-set LIF matched all 216 targets but emitted 908 unmatched fires. Hand-set ALIF matched 213 and emitted 376 unmatched fires. These runs confirm that the accumulator can rescue weak evidence and that adaptation can reduce repeats. In these parameterizations the improvement came with much more chatter than Schmitt. The zero-fire LIF configuration also shows why a low duplicate rate alone would be a misleading selection metric.

## Disposition

Keep Schmitt plus cooldown as the reference for a future learned-state trial. Drop the tested LIF and ALIF configurations from the proposed default on the strength of this synthetic result. A learned evidence head could change which gate is best, so #152 remains open for its downstream comparison and final keep/drop decision. The outstanding evidence is matched causal hidden-state and payload traces, held-out event/retrigger and ordinary-motion behavior, independently judged visual coherence through #111, and runtime/export and training stability measurements. This probe has measured state width from the recurrences and used detached offline grid calibration. It has no surrogate-gradient, browser inference, ONNX export, or policy-training measurement. [Sparse event acceptance](https://github.com/rhythmatician/FractalSync/issues/152), [eyes-on evaluation](https://github.com/rhythmatician/FractalSync/issues/111).
