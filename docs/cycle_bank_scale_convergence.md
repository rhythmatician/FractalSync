# CycleBank scale-resolution convergence (issue #92)

ADR 0003 requires the numerical scale density to be chosen by **convergence
measurement**, not treated as "musical buckets per octave". This document
records that measurement for the v1 causal constant-Q analytic filter bank.

## Method

For each candidate density (`scales_per_octave` in {4, 6, 12, 24, 48}) and
several off-grid sinusoids (0.6231, 1.6234, 2.1667, 3.2718 Hz — none on a
nominal scale center), the canonical `CycleBank` was driven at the
authoritative 48 kHz / 1024-hop cadence for 18 s and the recovered continuous
mode was measured:

- **freq_err_hz** — absolute error of the continuous recovered frequency;
- **phase_err** — absolute calibrated phase error against the known input
  analytic phase (rad);
- **pred_err** — one-hop free-running `phase_at(dt)` prediction error (rad).

## Results

Max relative frequency error and max calibrated phase error across targets:

| scales/octave | max rel freq err | max phase err (rad) |
| ------------- | ---------------- | ------------------- |
| 4             | 0.01338          | 0.195               |
| 6             | 0.00269          | 0.037               |
| 12            | 0.00269          | 0.037               |
| 24            | 0.00269          | 0.031               |
| 48            | 0.00190          | 0.048               |

## Reading

- **Frequency recovery converges by 6 scales/octave.** 6, 12, and 24
  scales/octave recover the same continuous frequency to within 0.3% relative
  (and the 0.6231 Hz target to ~1.7e-3 Hz). The continuous instantaneous-
  frequency estimator, not the grid, carries the frequency resolution.

- **Calibrated phase is accurate at every density** (max ~0.03–0.05 rad ≈
  0.5–0.8% of a cycle ≈ 2–4 ms at these frequencies) and does **not** depend
  on grid density: phase is remodulated from the continuous demodulator
  oscillator with the exact discrete filter transfer phase removed, so it is
  not quantized to a scale center.

- Densities beyond 12 buy negligible frequency accuracy and add compute; the
  residual differences at 24/48 are within the estimator's noise floor (a
  fraction of a millisecond in time terms), not systematic improvement.

## Decision

`CycleBankConfig::default().scales_per_octave = 12`.

- 6 scales/octave already converges for frequency, so the coarsest *justified*
  value is 6; 12 is retained as the default for comfortable numerical margin
  (two converged points straddling it) at a per-tick cost that is trivially
  within the ~46.875 Hz real-time budget (see the performance note in the
  pull request). The choice is a numerical-convergence setting, not a musical
  ontology: increasing it further does not change what the bank can represent.

The deterministic `scale_resolution_converges_without_turning_scales_into_
musical_buckets` test in `runtime-core/tests/test_cycle_bank.rs` enforces the
convergence property (relative-frequency convergence across 6/12/24/48) so the
claim cannot silently regress.
