# Geometry #145: fresh coverage and reference-policy replay

Latest follow-up: [browser parity, resource measurements and scalar-fidelity audit](geometry-145-operational-followup.md). The larger field passes browser mechanical/parity checks but has 43 conclusive scalar-budget violations. The measurements below preserve the earlier coverage checkpoint.

October 30, 2026. This is a separate experiment after `55ebb8a`; the first
[frozen experiment](geometry-145-operational-prototype.md) and its failures remain
preserved. Shipping adoption and reference-cut motion policy remain pending.

The original native controlled-crossing test now passes. The full Python physics
file is **26 passed, 1 failed**, with only the unchanged scalar-quality assertion
remaining. These results establish operational mechanics, not reference Shore
crossing or true distance fidelity. PR #148 is still not ready to merge.

## Fixed inputs and separate replays

The [coverage recipe](geometry-145-operational-coverage-experiment.json) was fixed
before constructing the new asset. It retains spacing 0.0005 and the same C4
quintic evaluator, with origin (-1.002, -0.702) and shape 3425 by 2825. All tests
restart from their original initial states. No old trajectory was resumed and no
field was extended during a rollout. The new coordinate mapping has its own
field identity; cross-asset bitwise overlap equivalence is not assumed.

Field ID: `068e8e016b4ad6b7`.
Asset SHA-256: `d77495871dd3fd9d0b66876ffe9148bade43234227ca4eb918a56d234069925e`.

The diagnostic asset is untracked at `tmp/geometry145-coverage-v1.bin`. The
reproducible recipe, source hashes, preparation measurements and exact test
outcomes are in the [machine-readable record](geometry-145-operational-coverage.json).
The original 12.9 MB asset remains byte-identical under the preparation refactor.

| Replay | Native original crossing | Full Python physics file |
| --- | --- | --- |
| Wider field, no loaded reference policy | Pass | 25 pass, 2 fail |
| Same wider field, frozen reference policy | Pass | 26 pass, 1 fail |

The first replay removed coverage failures but exposed a missing raw-adapter
policy input. The adapter reported its smooth field as regular at (-1, 0), where
the current canonical provider has an authoritative cut-locus refusal. The
existing frozen-rollout path already honored supplied reference evidence; the
raw Python loader could not load it.

## Policy enforcement without an online proof

A separate [query plan](geometry-145-operational-reference-policy-experiment.json)
was fixed before the policy replay. The offline preparation command queries the
existing canonical provider at (-1, 0), requires its actual CutLocus/Singular
classification, and stores an exact-point refusal bound to the field ID.
No new classification algorithm or mathematical proof was added.

The raw adapter now accepts this validated immutable policy before replay.
Queries still return the operational jet, with a singular validity result at
the authoritative refusal; Physics rejects it. No inference extends the point
evidence to a region. Information elsewhere and reference distance fidelity
remain unknown. Bad field IDs, malformed boxes, excessive evidence and in-flight
replacement are rejected. There is no reference calculation in the query path.

The native replay also loads that policy before initialization. Its profile ID
changes from `0790a66e7df83330` to `b61c1e0bbba796e4`, while field coefficients and
the physical parameters stay fixed. Both runs start afresh. This makes the
evidence/policy change observable without a mid-rollout model switch.

The original native body is checked byte-for-byte against `9187815` and run
verbatim under the frozen context. Its original underpowered and driven
assertions both pass. The full Python file includes all retained energy,
crossing, connection, cut-locus, mirror and sequence tests. The remaining failure
is `test_jet_exposes_scale_and_convergence_quality`: unknown scalar error does
not satisfy the original `0.25 * requested_scale` gate. No assertion or threshold
was changed to make either replay pass.

Sixteen focused Rust contract controls pass, including two additions: validating
offline preparation before allocation and carrying an authoritative refusal
through the raw adapter, with rejection of evidence for another field. Default
native and experimental wasm32 compilation also pass. The verifier now records
the actual contract-test count while retaining its minimum-size and all-tests-
successful checks.

## Resource tradeoff and remaining work

The wider dense field consumes **77,405,000 coefficient bytes**, approximately
six times the first asset. The encoded asset is 77,405,056 bytes. Preparation
compute measured 0.461 s and total preparation/write 0.562 s, under the declared
80 MB coefficient ceiling. This is a prototype ceiling, not a shipping budget.

The earlier browser parity and latency measurements apply to the first asset.
Browser parity, loading/heap cost and timing for this larger field and policy
have not been measured. Reference Shore crossings, scalar/differential fidelity,
and complete reference-singularity coverage remain unverified. Current-policy
enforcement at a supplied known point does not authorize motion through other
reference cuts. Adoption, compatibility versions/goldens and deployment
validation remain open.

## Reproduce from fresh outputs

The builder refuses to overwrite an existing asset. Use a fresh output path, or
reuse this asset only after checking its SHA-256; never patch and resume a run.

```powershell
rtk cargo test --manifest-path runtime-core/Cargo.toml --release --features experimental-operational-field operational_field -- --test-threads=1
rtk cargo build --manifest-path runtime-core/Cargo.toml --release --features experimental-operational-field --example operational_coverage_prepare
rtk proxy target/release/examples/operational_coverage_prepare.exe docs/validation/geometry-145-operational-coverage-experiment.json tmp/geometry145-coverage-v1.bin
rtk proxy target/release/examples/operational_coverage_prepare.exe --policy tmp/geometry145-coverage-v1.bin docs/validation/geometry-145-operational-reference-policy-experiment.json > tmp/geometry145-coverage-v1-reference-policy.json
rtk proxy .venv/Scripts/python.exe -m maturin build --manifest-path runtime-core/Cargo.toml --release --features python,experimental-operational-field --out tmp/operational-coverage-wheels
rtk proxy .venv/Scripts/python.exe scripts/validate_operational_field.py tmp/operational-coverage-wheels/runtime_core-0.1.0-cp38-abi3-win_amd64.whl --asset tmp/geometry145-coverage-v1.bin --reference-policy tmp/geometry145-coverage-v1-reference-policy.json --report tmp/geometry145-coverage-v1-python-policy.json backend/tests/test_manifold_physics.py -q
rtk proxy .venv/Scripts/python.exe scripts/validate_operational_native_crossing.py tmp/geometry145-coverage-v1.bin tmp/geometry145-coverage-v1-native-policy.json tmp/geometry145-coverage-v1-reference-policy.json
```

The full Python command exits 1 for the preserved scalar-fidelity failure. The
native crossing command exits 0. Neither is a claim that PR #148 is merge-ready.
