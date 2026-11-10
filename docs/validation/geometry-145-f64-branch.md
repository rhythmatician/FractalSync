# Outward f64 local branch proofs for #145

PR #148 remains unready to merge. This test-only follow-up to the
[generic proposal experiment](geometry-145-generic-component.md) validates local
coupled-root families and their identity joins with outward-rounded f64 affine
arithmetic. Production code, quality thresholds and original crossing tests are
unchanged.

For each exact represented multiplier in a closed disk, the proof encloses a
root of `(f_c^p(z)-z, (f_c^p)'(z)-mu)`. The rounded inverse is treated as an exact
dyadic preconditioner. Three correlated variables retain multiplier displacement
and both root errors. Arithmetic bounds include center and coefficient rounding,
nonlinear remainder products and underflow. Acceptance requires strict image
inclusion in both requested root disks and a weighted contraction bound below
one. Automatic radius selection uses rejected bounds to propose larger domains;
it does not relax these gates.

All 162 discovered node branches certify multiplier disks of radius `1e-4`.
Their parameter-error bounds remain below the original scalar-node error windows.
The largest bound is approximately `2.026e-9`. Endpoint evaluation carries
subtraction, multiplication and addition rounding. Each node's proof also joins
a neighboring multiplier disk by containing the old endpoint root in the next
uniqueness domain, with cycle-phase advances checked explicitly. Closed-form
cardioid, distinct period-three components and a period-two phase change provide
independent positive and negative cases.

The existing 128-bit multiprecision branch proof independently checks all 162
domains using the same exact dyadic preconditioners. Its inclusion and contraction
bounds are enclosed by every corresponding f64 bound. Both independent tests
pass. This comparison checks the reported bounds, rather than merely accepting
nearby roots using a different preconditioner.

Focused validation passes 32 tests and catches 12 injected faults: the five
existing proposal/field faults, missing center rounding, missing nonlinear
remainder, skipped contraction, skipped inclusion, missing parameter containment,
omitted cycle-phase search, and refused singleton domains. The harness still passes the original 23-test,
five-mutation generic-only variant. Full provider results are retained in the
companion JSON. The all-node local-proof/join run takes about four seconds,
including numerical proposals; it does not measure a complete geometry query.

The certificate proves a local polynomial root family, not a globally nearest
Mandelbrot boundary point. The numerically generated unit-multiplier centers are
represented floating values, not certified exact points on the unit circle.
There is no closed attracting cover, exterior-witness upper bound, global
nearest-feature coverage, or true-distance remainder from this patch. These
remain required before integrating the coherent scalar sampler and rerunning
original crossings, inward convergence, energy, live training and query cost.

Apply `experiments/geometry-145-f64-branch.patch` after the generic-component patch
and its documented prerequisite chain. Build the reconstructed worktree, then:

```powershell
rtk cargo test --release --jobs 1 --manifest-path tmp/geometry145-runtime-integration/runtime-core/Cargo.toml --target-dir target/geometry145-runtime-integration geometry_provider::
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-generic-component-check.py --source-root tmp/geometry145-runtime-integration --target-dir target/geometry145-runtime-integration --output tmp/geometry-145-f64-branch-final.json
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-branch-hp-check.py --input tmp/geometry-145-f64-branch-final.json --target-dir target/geometry145-precision --output tmp/geometry-145-f64-branch-independent-final.json
```

The second command uses the already-built `astro-float` dependency of the existing
point-proof experiment. Both harnesses modify temporary copies only. Source and
input hashes, full terminal results, root domains and mutation assertion failures
are retained in `geometry-145-f64-branch.json`.

Independent standards, specification and maintenance reviews found two issues
that are fixed here: certificate authority is private with copy-returning
diagnostic getters, and exact represented multiplier centers can evaluate and
join singleton domains. A distinct adjacent floating multiplier is still refused.
Both fixes passed independent re-review.
