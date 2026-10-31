# Geometry #145: globally refined reference-grid experiment

October 30, 2026. The frozen refined asset removes both previously observed
sign-based scalar counterexamples and verifies all 21 crossing trajectories
against conclusive reference realms. It does **not** clear merge acceptance.
The unchanged Python physics file has 25 passes and two failures, including a
crossing-time refinement regression. No production implementation changes in
this experiment. ADR 0005 still authorizes a bounded prototype only.

## Controlled preparation

The [frozen input recipe](geometry-145-operational-reference-grid-experiment.json)
halves the reference coarse spacing from 0.002 to 0.001. Subdivision changes from
16 to 8, retaining final contour spacing 0.000125. Membership iterations remain
8,192 and edge bisections remain eight. Coefficient spacing, domain, evaluator,
four-substep RK4 method, initial states, perturbations, test bodies and thresholds
remain unchanged. This isolates features omitted by homogeneous coarse cells;
it does not establish that every such feature is captured.

The node allowance increases from five million to twenty million, within the
existing hard limit. Preparation uses 17,648,401 reference nodes, 26,799 refined
cells, 313,421 segments and 34,513,294 membership queries, below the declared
50-million-query and 500,000-segment limits. It completes in 126.97 seconds
internally, 127.23 seconds process time, with sampled OS peak working set
216,313,856 bytes. Preparation and loading are excluded from query timings.

The complete asset is loaded before every fresh replay. It contains 9,675,625
coefficients, 77,405,000 coefficient bytes and 77,405,056 artifact bytes.

```text
field_id: 37ec6365e602ae31
asset_sha256: ca2b9a78c482946fe3760d8fb7919e48fc5bf4638a8c7eb47b499a8c6313d041
timed physics_profile_id: 54b525d2266d27eb
original native crossing physics_profile_id: d9f1b51a967aff60
```

Full profiles and preparation provenance are in the
[machine-readable report](geometry-145-operational-reference-grid.json).
The [replay manifest](geometry-145-operational-reference-grid-replay.json)
pins asset and reference-policy hashes. There is no extension, coefficient
replacement or resume after a coverage miss. The previous contour-v1 asset and
all its failures remain preserved in the earlier reports.

## Independent outcomes

| Gate | Result |
| --- | --- |
| Fixed 765-point scalar audit | 765 conclusive reference realms, zero sign mismatches and zero sign-based error-floor violations, previously two |
| Reference scalar-distance upper bound | Unknown; absence of counterexamples is not a bound |
| Reference derivative fidelity | Unknown |
| Neighboring differential self-consistency | Passes versioned Richardson check at unchanged `1e-4`; normalized Hessian error `2.706e-6` |
| Patch joins | Pass at unchanged `1e-4`; normalized Hessian error `5.163e-6` |
| Original finite-increment central diagnostic | Still fails, normalized Hessian discrepancy `1.0332e-4`; retained in raw evidence |
| Frozen per-run expectations | 35/35 pass with complete coverage; 21 operational crossings, 14 noncrossing controls |
| Reference crossing diagnostics | 21/21 crossing trajectories have conclusive ordered opposite-realm pairs, previously 15/21 |
| Original native crossing body | Passes unchanged; reference crossing separately verified |
| Original Python physics file | 25 pass, two fail, source unchanged |
| Energy diagnostics | High-curvature maximum excursion `1.799e-14`; near-Shore `4.042e-11`, both below unchanged `0.05` |
| Native/WASM parity | Passes declared scalar, gradient, Hessian, state and ledger tolerances |
| Identity and known reference-cut policy | Query-order/cache invariance passes; available operational jet does not bypass known cut refusal; state preserved |

Reference pairs concern the continuous piecewise-linear trajectory reconstructed
from committed endpoints, not an exact ODE solution or a precise Shore event.
Conclusive realm evidence is point scoped. No scalar sign is treated as a
reference-realm proof within an uncertified error band. This experiment adds no
proof expansion or change to motion policy.

The 27 existing Rust controls also pass on this unchanged runtime build,
including analytic derivative propagation and transactional refusal controls.
Every attempted complete step retains the existing intermediate-stage,
weighted-endpoint, energy and ledger availability checks before commit.

## Retained Python failures

`test_shore_ridge_reflects_below_barrier_and_crosses_above_it` reaches opposite
operational and reference realms at all three timesteps. Its sampled crossing
times round to 2.044 seconds for `dt=0.002`, `0.001` and `0.0005`. The test
requires `fine_gap < coarse_gap`; it observes `0.0 < 0.0` and fails. The owned
rollout report accumulates time and therefore records tiny rounding differences;
those must not be substituted for the original test's sample-index times.
Per-run crossing success does not satisfy this separate refinement assertion.

The previous asset's original crossing was about 2.649 seconds. The roughly
23% timing change from source refinement is material field sensitivity even
though both assets pass the energy and differential checks. No method or
coefficient adjustment was selected to move the crossing between sample bins.

`test_jet_exposes_scale_and_convergence_quality` still fails because the adapter
reports unknown scalar quality as the maximum finite value rather than claiming
an unsupported error estimate. The original `estimated_error` contract concerns
numerical convergence, which is distinct from a certified true-distance bound.
Neither regional convergence evidence nor a reference upper bound has been
established for this candidate. The zero-counterexample audit supplies neither.
Any future quality metadata must state which claim its evidence supports.

The entire Python file retains SHA-256
`2451ab5c7ffa18f9cc8259ef109a4bca27cd1af846f9ac6843a626ed93c84536`.
The original native crossing body retains SHA-256
`e1afb878f44a2644f1efbcf97a7b94a6e7b8f56c798dd59e41f0dfb6af340895`.

## Evaluation cost

Device: Intel Core i9-14900, Windows 10.0.26100; headless Chromium
153.0.8010.12 on the same device. These are experimental measurements, not
ratified device requirements or hard-deadline guarantees.

| Measurement | Native, three fresh processes | Browser |
| --- | --- | --- |
| Cache-stressed prepared query p99 | 4.8 / 3.9 / 4.6 microseconds | 40 microseconds |
| Complete owned-step p99 | 159.4 / 185.5 / 135.2 microseconds | 290 microseconds |
| First prepared query, final native process | 8.4 microseconds | 185 microseconds |
| Cache-stressed query maximum, final native process | 32.6 microseconds | 1.680 ms |
| First complete step, final native process | 108.2 microseconds | 1.920 ms |
| Complete-step maximum, final native process | 450.7 microseconds | 1.920 ms |
| Unavailable query p99, final native process | 0.1 microseconds | 5 microseconds |

All processes meet provisional query p99 <=100 microseconds and complete-step
p99 <=2 ms targets. Maxima remain separate; browser prepared-query maxima exceed
100 microseconds. Full first/p50/p99/max distributions for ordinary, unavailable
and cache-stressed queries and complete steps remain in the machine report.
Browser timer quantization can produce zero-microsecond individual samples.

Each timed step uses 160 geometry evaluations and at most 5,760 coefficient
reads. Cache stress touches 64 MiB between probes. Native loading in the final
process is 200.29 ms; browser fetch is 115.74 ms and decode is 169.27 ms, with
WASM initialization 24.78 ms. Native sampled process peaks are 247,390,208 /
247,001,088 / 240,709,632 bytes. Browser WASM linear-memory capacity is
160,301,056 bytes, separate from the 77,405,056-byte JavaScript asset input,
64 MiB stress scratch, and other browser overhead. Capacity is not resident use.

## Reproduction and decision

The approximately 507 KiB [raw evidence archive](experiments/geometry-145-operational/reference-grid-evidence.zip)
preserves preparation, fixed audit, policy, all three native processes, browser,
original native crossing and full Python failure reports. The machine report
indexes their SHA-256 values. The large asset is reproduced from the fixed input
recipe using `operational_contour_prepare`, then checked against the artifact
hash above. Existing measurement and validation scripts accept the replay
manifest, frozen policy and explicit `Rk4FourSubsteps` profile.

This experiment supports bounded query cost and coherent operational mechanics.
It does not yet show that one frozen field satisfies every retained test.
The remaining blockers are scalar quality evidence, sensitivity to reference
preparation, and the crossing-time refinement assertion. Shipping adoption and
reference-cut policy remain pending under ADR 0005. No threshold or crossing
test was relaxed, and a failed replay did not trigger a point-specific proof.
