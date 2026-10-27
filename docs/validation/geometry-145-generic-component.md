# Generic component proposals for #145

PR #148 is still not ready to merge. This experiment supplies numerical scalar
distance proposals for the higher-period components at the two recorded coherent
query failures. It is archived, test-only Rust code, with no production adoption.

The proposal reuses the shared critical-orbit seed, reduces machine-cycle hints
with fresh strict local return-disk checks, and follows multiplier continuation
to propose stationary boundary points. Neither fixture periods nor boundary
angles are supplied to the solver. The Python case reduces a machine cycle of
308 to a model period of 154; the native case selects 85.

All 162 archived scalar-node reference intervals fit inside the original error
windows (native 5e-8; Python 5e-10). Reverse evaluation reproduces distance bits.
The existing quintic reconstruction passes finite-difference gradient and
distance-scaled Hessian checks at the unchanged 1e-5 limit, plus adjacent-cell
C2 joins. The largest measured gradient discrepancy is 1.15e-10. These checks
cover two reconstructed cells, not the entire field.

Validation: 87 provider tests pass. The standalone extraction of actual Rust
sources passes 23 tests and catches all five injected faults: bypassed root
proof, doubled model period, rejected long machine period, corrupted distance,
and corrupted gradient. The 162-node standalone pass takes about 3.36 seconds;
that excludes certification, refinement, cut coverage and Physics, so it is not
a replacement timing for the previously timed-out full queries.

The remaining work is to establish global nearest-feature coverage and scalar
uncertainty, preserve genuine cut-locus refusals, and integrate the scalar sampler
into the coherent field. Then rerun the original crossings, energy and inward
convergence checks, live training, and full query-cost validation. A local return
disk does not prove least mathematical period, component identity, or globally
nearest boundary distance. This proposal cannot issue a Regular geometry jet.

The patch applies after the ranged-field, scalar-sampler and correlated-return
patch chain recorded in the companion JSON. Reproduce the focused validation
after building that worktree with:

```powershell
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-generic-component-check.py --source-root tmp/geometry145-runtime-integration --target-dir target/geometry145-runtime-integration --output tmp/geometry-145-generic-component-check.json
```

The companion JSON retains source and fixture hashes, successful logs, all five
mutation failures, and the unsuccessful intermediate period-reduction trials.
Production thresholds, original crossing tests and production code are unchanged.
