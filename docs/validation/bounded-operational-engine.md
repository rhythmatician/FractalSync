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
