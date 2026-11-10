# Bounded operational engine integration

This change extracts an explicitly selected, default-disabled prepared-field
engine from PR #148 onto the current base. It does not close #145 or change the
base provider. Each rollout owns one immutable field, Physics profile, and
frozen reference-refusal snapshot; attempts commit position, velocity, time,
work, and energy together only after all stages and the weighted endpoint pass.
A missing differential stencil refuses as `GeometryUnavailable`; there is no
fallback to the base provider during an operational rollout.

The field is a smooth operational scalar, not a certified Mandelbrot distance.
Reference-distance accuracy and reference-cut coverage outside the supplied
snapshot are unknown. The exact known cusp and cut at `(0.25, 0)` and `(-1, 0)`
remain refused, as do the snapshot's reference-singularity regions. The policy
does not claim complete cut-locus coverage. Shipping adoption, coverage-loss
behavior, reference accuracy and production device targets remain separate
decisions, not integration acceptance claims.

The crossing regression retains its physical setup, crossing requirement and
0.05-second absolute timing limit. It interpolates between bracketing committed
states and compares the fine trajectory against a separately run half-step
trajectory. Analytic linear controls include the sample phase that caused the
old strict timestamp-gap comparison to fail.

`scripts/replay_distant_feature_failure.py` preserves the historical adaptive
provider failure against a clean checkout of its exact recorded source. It
expects that historical test to fail at the recorded point; it is not a test
of this engine or a claim that the on-demand provider is fixed.

The following checks are required before this candidate is ready: exact-source
default and experimental binaries; original default crossing and energy tests;
operational crossing and energy replay on a frozen asset; analytical, join,
refusal and transaction tests; Python and WASM parity; and recorded source,
binary, asset and policy hashes. A measured field discrepancy is not a certified
reference error bound.

The frozen-asset replay uses the #148 starting positions and launch rules for
the driven/undriven Controls case and the underpowered/above-barrier Python case.
Its crossing verdict concerns the operational contour. Reference Shore crossing
is separately unverified for this extraction. The two retained high-curvature
and near-Shore energy replays compare absolute energy excursion and drift with
the original `0.05` limit. The crossing replay also checks absolute energy
excursion against that limit.

The fixed 1,000-point neighboring check and explicit joins retain the `1e-4`
tolerance. The neighboring verifier uses Richardson extrapolation at fixed
increments and checks agreement at two resolutions. The report also exposes the
original raw central-difference outcome separately; a failure there cannot be
described as a passing raw diagnostic. These tests check field self-consistency,
not fidelity to reference derivatives.
