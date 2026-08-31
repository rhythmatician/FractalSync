# CycleBank runtime performance (issue #92)

The CycleBank runs once per authoritative analysis hop: 1024 canonical samples
at 48 kHz, i.e. one observation every **21.33 ms** (~46.875 Hz). The v1 bank is
a causal constant-Q analytic filter bank (`ANALYTIC_STAGES = 2` one-pole
baseband sections per scale) plus ridge tracking — a cheap recursive streaming
update with no per-tick heap churn in the hot path.

## Measured cost (release build, this machine)

`runtime-core/examples/perf_benchmark.rs` (temporary harness, not shipped)
drove the bank with multi-channel sinusoidal evidence after a warm-up that
acquired live modes (so ridge tracking, merge, and relation diagnostics all
ran). Per-tick CPU cost vs the 21 333 µs real-time budget:

| scales/octave | evidence channels | transform scales | µs/tick | % of budget |
| ------------- | ----------------- | ---------------- | ------- | ----------- |
| 6             | 1                 | 43               | 12.4    | 0.058%      |
| 12            | 1                 | 85               | 19.9    | 0.093%      |
| **12**        | **3**             | **85**           | **40.6**| **0.190%**  |
| 24            | 3                 | 169              | 64.7    | 0.303%      |
| 48            | 3                 | 337              | 106.0   | 0.497%      |

The default configuration (**12 scales/octave, 3 channels**) costs about
**40 µs per hop — under 0.2% of the real-time budget**, leaving >99.8% of each
hop interval for feature extraction, inference, rendering, and the OS.

## Cost scaling

- Cost scales linearly in `transform scales × evidence channels ×
  ANALYTIC_STAGES` (the constant-Q grid) plus a small ridge-tracking term.
- The grid spans `f_min_hz..f_max_hz` (default 0.0625..8 Hz, 7 octaves), so
  scale count is `~7 × scales_per_octave + 1`.
- State is O(scales × channels) complex coefficients plus O(max_modes)
  trackers — a few KB; no per-tick allocation growth (histories are bounded
  ring buffers).

## Note

Even the densest measured grid (48 scales/octave) uses under 0.5% of the
budget, confirming that `scales_per_octave` is free to be chosen by numerical
convergence (see `docs/cycle_bank_scale_convergence.md`) rather than by a
performance constraint.
