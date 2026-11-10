# Controls v2 training energy diagnostic, 2026-11-09

The training bridge now returns the complete energy record computed by the Rust
Controls v2 step. Previously it recomputed only kinetic and potential energy,
omitted the wall contribution, and silently returned zeros if that recomputation
failed. The existing straight-through estimator (STE) remains an approximation;
this change does not alter the Rust mechanics or the field.

## Frozen prototype replay

The replay used field `operational:9721bb7647a0b8e6`, asset SHA-256
`bc2604771c2cc48fb71cda6cb22c6e4338a776b6591fa923dd065977817ad9fb`,
and `Rk4FourSubsteps`. The staged wheel, extension, asset, and recorded Rust
source hashes match the preceding retained-field replay. The relevant Rust
sources did not change between that wheel's source checkpoint and this change.
The original manifold test file was unchanged (SHA-256
`2451ab5c7ffa18f9cc8259ef109a4bca27cd1af846f9ac6843a626ed93c84536`).

With `FRACTALSYNC_RETAINED_FIELD_STAGE_TEST=1`, the combined training energy,
Controls v2 smoke, and original manifold tests produced **35 passed, 2 failed**.
All six new training tests and all four existing Controls v2 smoke tests passed.
The two failures remain the original crossing-time refinement and scalar
convergence-quality gates; neither test nor threshold was changed. See
`geometry-145-training-acceptance.json` for individual outcomes and hashes.

The new tests compare all five diagnostic values with the actual Rust step,
including a nonzero wall potential, exercise a real intermediate RK-stage
coverage refusal, check that a native failure yields no partial sequence, and
verify that an optimizer can update parameters through the existing STE. A
regression with a controlled native record also demonstrates that diagnostic
values are no longer fabricated when redundant energy calls fail. On the
project-installed default extension, five new tests passed and the retained
field-specific stage test was skipped. That extension's SHA-256 is
`0e26f43b33b499a5da5346b412bea57605a64cb9dc8886976f585ea0ec99cac6`;
its source commit was not independently verified.

The CI-equivalent mypy check passed for 21 backend source files, the new test
passed its standalone mypy check, Black passed for the new test, and git diff
check passed. An additional stricter local mypy configuration reports two
pre-existing unused ignores in the touched module; the CI-equivalent check
passes. The previous completed shipping-profile CI on `3cff2d8` still had one
default Rust crossing refusal and one default Python crossing refusal; the
feature-gated Rust Controls tests passed (32 tests).

## Limits

The deterministic optimizer check uses CPU, fixed time step and fixed brake and
grip; the existing one-epoch smoke is randomized. An optimizer update does not
establish an exact physics gradient. The Python sequence is not an owned
`FrozenRollout` and does not pin a profile identity or maintain a persistent
mechanical-time/work ledger across frames. Arbitrary changing controls therefore
have not been established to meet the frozen-profile replay contract. The real
stage-refusal test checks propagation and unchanged caller inputs; owned-ledger
rollback is covered separately by the Rust rollout tests. CUDA was unavailable.

No new latency measurement was made for the training bridge. The earlier
prepared-field native and browser timings, including the observed browser query
maximum, are reported in `geometry-145-operational-retained.md`; they do not
measure a training rollout. This result does not establish reference-distance
fidelity, resolve either acceptance failure, or authorize shipping adoption.
