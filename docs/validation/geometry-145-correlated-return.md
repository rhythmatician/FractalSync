# Geometry #145: correlated periodic-return enclosures

Production now retains parameter correlation while proving periodic returns of an enclosed critical-orbit prefix. This changes the empty-tile proof shortcut, not the distance-field definition. The coherent-field replacement remains unshipped.

The prefix enclosure has the form `z + d*delta_c + error_disk`. A return compares both the center and the parameter coefficient with their starting values. Acceptance requires the prefix disk to fit inside the starting remainder disk, every intermediate machine-orbit disk to stay within radius 2, and the sum of center defect, coefficient defect times parameter radius, and final remainder to be strictly smaller than the starting remainder radius. Propagation retains machine-update and coefficient-rounding bounds.

Interior proof attempts use bounded dyadic prefixes from 128 through 4096 updates. Exterior certificates still require escape before update 512. Ordinary tile membership budgets, convergence thresholds, cut-locus rules, cache policy, versions, and both crossing tests are unchanged. Standards, Spec, and Health reviews of `5816b5c` found no actionable issues; the Health review explicitly left full-query benefit unproved. `07053e1` removes an unused initializer without changing arithmetic.

## Validation

- All 39 selected production provider tests pass after cleanup. The wider Rust runs have 261 passes, one ignored test, and the original native crossing failure at `(0.21474166530729011, 0.12774018997553147)`, still `CutLocus`.
- The rebuilt Python extension has 26 physics passes and the original crossing failure at `(0.35659381055433242, 0.06404656757776669)`, still `Unresolved`.
- Mandatory parity, architecture, canonical-surface, and WASM compilation checks pass.
- All 64 production report jets exactly match the archived baseline through the same parent-report path. An initial comparison incorrectly mixed direct child output with parent output, whose additional Rust JSON parse changes some numbers by one ulp. Repeating the identical reporting protocol resolves all 35 apparent differences without a tolerance.
- The production-source standalone check passes five tests and catches all three mutations: dropping prefix containment, dropping coefficient drift, and dropping machine-rounding error. The rounding regression uses the known exact binary remainder `2^-54` in `(1 + 2^-27)^2`.

## Cost and remaining failures

In the 1296-box ablation, accepted interior boxes increase from 324 to 917, with all 324 existing certificates preserved. Uncertain boxes decrease from 972 to 379. Both sides of this ablation retain the 4096-prefix schedule; disabling correlation takes 0.373 seconds versus 0.633 seconds with correlation. A separate control restoring the original 512-prefix cap certifies zero boxes and takes 0.070 seconds. Thus the complete change buys more certificates with substantially more proof work. Avoided tile construction must justify that tradeoff; these timings do not establish a net query improvement.

The 64 production cases total 29.076 seconds cold versus the earlier record's 27.496 seconds under different concurrent loads. No general speedup is claimed.

The isolated coherent-field variant passes all 81 selected provider tests. However, both recorded difficult queries still time out at 1200 seconds. The native case is `(0.32049988639807619, 0.04733531447355364)` and the Python case is `(0.35659381055433242, 0.06404656757776669)`, each with epsilon `1e-4`. This proof change does not resolve coherent-field construction cost or establish crossing, training, or energy acceptance for adoption. PR #148 remains unready to merge.

## Reproduction

Run the production-source proof comparison with the project interpreter:

```powershell
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-correlated-return-bench.py --source-root . --output tmp/correlated-return.json
```

To reconstruct the coherent variant, apply [the correlated-return patch](experiments/geometry-145-correlated-return.patch) after the ten ranged-field patches and the scalar-sampler follow-up at `d8ec37a`. Forward and reverse patch checks pass. The patch includes the single-query diagnostic used for the terminal timeout runs.

[The machine-readable record](geometry-145-correlated-return.json) preserves source and executable hashes, mutations, test logs, exact jet comparisons, review scope, and both terminal timeouts.
