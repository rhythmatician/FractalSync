# 0002 — Musical timing as a bank of predictive cycle hypotheses

Status: Accepted (2026-08-30)

## Context

FractalSync needs to react not only to what the music is doing now, but to
where the music is going next.

The Domain Contract requires the Player to anticipate important musical
Transitions rather than merely react after they occur. Short-timescale musical
events such as beats, pulses, riffs, measures, phrase recurrences, and
polymetric cycles often contain predictable temporal structure. That structure
should be exposed causally to the Player without requiring future audio.

Previous approaches risked choosing the wrong abstraction:

- one global tempo estimate;
- a dedicated beat tracker;
- separate beat, bar, and phrase trackers;
- fixed octave-spaced frequency bins treated as musical truth;
- event detectors that reset phase whenever an onset occurs;
- centered/symmetric transforms that implicitly use future samples.

Those architectures encode semantic assumptions too early. Music may contain
several simultaneous periodicities, changing tempo, syncopation, missing
events, odd meters, polymeter, or recurring structure that does not correspond
cleanly to conventional beat/bar labels.

The architecture instead needs a generic representation of predictive temporal
structure.

ADR 0001 establishes `runtime-core` as the authority for deterministic
FractalSync domain behavior shared by training and runtime. PR #93 / issue #91
established an authoritative sample-clock audio pipeline:

```text
native-rate PCM
    ↓
runtime-core AnalysisTimebase
    ↓
canonical 48 kHz timeline
    ↓
exact 1024-sample analysis ticks
    ↓
AnalysisTick {
    features,
    sampleIndex,
    timeSeconds,
    dtSeconds,
    streamEpoch
}